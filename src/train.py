from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from .preprocessing import load_all_with_text
from .feature_engineering import fit_vectorizer, cosine_batch
from .model_utils import save_joblib, save_metrics  # << importa só aqui, no topo
from .utils import normalize_text

BASE = Path(__file__).resolve().parents[1]

# mapeamento simples (ajuste se tiver outros valores nos seus dados)
STATUS_POS = {
    "aprovado", "encaminhado", "finalista", "contratado", "contratada", "contratação"
}
STATUS_NEG = {
    "reprovado", "reprovada", "desclassificado", "desclassificada",
    "sem_avanco", "sem_avanco_reprovado", "sem_avanco_reprovada",
    "desistente", "nao_selecionado", "não_selecionado"
}

def _status_label(s: str) -> int | None:
    if not isinstance(s, str):
        return None
    t = s.strip().lower().replace(" ", "_")
    if t in STATUS_POS: return 1
    if t in STATUS_NEG: return 0
    return None

def _build_pairs_from_prospects() -> pd.DataFrame:
    """Cria pares (vaga, candidato) + rótulo a partir de prospects.csv (processado)."""
    proc = BASE / "data" / "processed" / "prospects.csv"
    if not proc.exists():
        return pd.DataFrame(columns=["vaga_id", "applicant_id", "label"])

    df = pd.read_csv(proc)
    # tenta adivinhar as colunas
    cand_col = next((c for c in df.columns if "applicant" in c or "codigo" in c or "cand" in c), None)
    vaga_col = next((c for c in df.columns if "vaga" in c), None)
    stat_col = next((c for c in df.columns if "situacao" in c or "status" in c), None)
    if not all([cand_col, vaga_col, stat_col]):
        return pd.DataFrame(columns=["vaga_id", "applicant_id", "label"])

    out = df[[vaga_col, cand_col, stat_col]].rename(
        columns={vaga_col: "vaga_id", cand_col: "applicant_id", stat_col: "status"}
    )
    out["label"] = out["status"].map(_status_label)
    out = out.dropna(subset=["label"]).astype({"label": int})
    return out[["vaga_id", "applicant_id", "label"]]

def train():
    apps, vagas, _ = load_all_with_text()
    pairs = _build_pairs_from_prospects()
    if len(pairs) == 0:
        print("WARN: Não foi possível rotular exemplos a partir de prospects; treinando somente o vetorizador.")

    # --- Fit do TF-IDF em textos de candidatos e vagas
    all_texts = pd.concat([apps["app_text"], vagas["job_text"]]).fillna("").astype(str).tolist()
    vect = fit_vectorizer(all_texts, max_features=120000)
    save_joblib(vect, BASE / "models" / "tfidf_vectorizer.joblib")
    print("OK: tfidf_vectorizer.joblib salvo.")

    # --- Supervisionado opcional (se houver rótulos)
    if len(pairs) > 0:
        apps_j = apps[["applicant_id", "app_text"]].copy()
        vagas_j = vagas[["vaga_id", "job_text"]].copy()
        df = pairs.merge(apps_j, on="applicant_id", how="inner").merge(vagas_j, on="vaga_id", how="inner")
        if len(df) == 0:
            print("WARN: Sem interseção entre pairs e textos. Pulando treinamento supervisionado.")
            return

        # usa cosseno TF-IDF como feature
        scores = [
            cosine_batch(vect, [row["app_text"]], row["job_text"], batch=1)[0]
            for _, row in df.iterrows()
        ]
        X = np.array(scores).reshape(-1, 1)
        y = df["label"].astype(int).values

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=200, class_weight="balanced")
        clf.fit(Xtr, ytr)

        proba = clf.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, proba))
        ap = float(average_precision_score(yte, proba))
        prec, rec, thr = precision_recall_curve(yte, proba)
        f1max = float((2 * prec * rec / (prec + rec + 1e-12)).max())

        from joblib import dump
        dump(clf, BASE / "models" / "recommender.pkl")
        save_metrics({"AUC": auc, "AP": ap, "F1_max": f1max}, BASE / "models" / "metrics.json")
        print("OK: recommender.pkl salvo. Métricas:", {"AUC": auc, "AP": ap, "F1_max": f1max})

if __name__ == "__main__":
    train()
