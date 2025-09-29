
from __future__ import annotations
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

from .preprocessing import load_all_with_text
from .feature_engineering import fit_vectorizer, cosine_batch
from .model_utils import save_joblib, save_metrics
from .utils import normalize_text

BASE = Path(__file__).resolve().parents[1]

STATUS_POS = {
    "aprovado","encaminhado","finalista","contratado","contratada","contratação"
}
STATUS_NEG = {
    "reprovado","reprovada","desclassificado","desclassificada","sem_avanco","sem_avanco_reprovado",
    "sem_avanco_reprovada","desistente","nao_selecionado","não_selecionado"
}

def _status_label(s: str) -> int|None:
    if not isinstance(s, str): 
        return None
    t = s.strip().lower()
    t = t.replace(" ", "_")
    if t in STATUS_POS: return 1
    if t in STATUS_NEG: return 0
    return None

def _build_pairs_from_prospects() -> pd.DataFrame:
    """Cria pares (vaga, candidato) + rótulo a partir de prospects.csv/json."""
    # Preferir CSV processado
    proc = BASE/"data"/"processed"/"prospects.csv"
    if proc.exists():
        df = pd.read_csv(proc)
        # inferir nomes prováveis de colunas:
        cand_col = next((c for c in df.columns if "applicant" in c or "codigo" in c or "cand" in c), "applicant_id")
        vaga_col = next((c for c in df.columns if "vaga" in c), "vaga_id")
        stat_col = next((c for c in df.columns if "situacao" in c or "status" in c), None)
        if stat_col is None:
            return pd.DataFrame(columns=["vaga_id","applicant_id","label"])
        out = df[[vaga_col, cand_col, stat_col]].rename(columns={vaga_col:"vaga_id", cand_col:"applicant_id", stat_col:"status"})
        out["label"] = out["status"].map(_status_label)
        return out.dropna(subset=["label"]).astype({"label":int})
    # fallback: vazio
    return pd.DataFrame(columns=["vaga_id","applicant_id","label"])

def train():
    apps, vagas, _ = load_all_with_text()
    pairs = _build_pairs_from_prospects()
    if len(pairs)==0:
        print("WARN: Não foi possível rotular exemplos a partir de prospects; treinando somente o vetorizador.")
    # --- Fit vectorizer em todos textos (vaga + candidato)
    all_texts = pd.concat([apps["app_text"], vagas["job_text"]]).fillna("").astype(str).tolist()
    vect = fit_vectorizer(all_texts, max_features=120000)
    save_joblib(vect, BASE/"models"/"tfidf_vectorizer.joblib")
    print("OK: tfidf_vectorizer.joblib salvo.")
    # --- Se temos rótulos, treina um modelo simples sobre o cosseno
    if len(pairs) > 0:
        # junta textos
        apps_j = apps[["applicant_id","app_text"]].copy()
        vagas_j = vagas[["vaga_id","job_text"]].copy()
        df = pairs.merge(apps_j, on="applicant_id", how="inner").merge(vagas_j, on="vaga_id", how="inner")
        if len(df)==0:
            print("WARN: Sem interseção entre pairs e textos. Pulando treinamento supervisionado.")
            return
        # cosseno
        X_scores = []
        for _, row in df.iterrows():
            score = cosine_batch(vect, [row["app_text"]], row["job_text"], batch=1)[0]
            X_scores.append(score)
        X = np.array(X_scores).reshape(-1,1)
        y = df["label"].astype(int).values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=200, class_weight="balanced")
        clf.fit(Xtr, ytr)
        # métricas
        proba = clf.predict_proba(Xte)[:,1]
        auc = roc_auc_score(yte, proba)
        ap  = average_precision_score(yte, proba)
        # threshold ótimo por F1
        prec, rec, thr = precision_recall_curve(yte, proba)
        f1 = (2*prec*rec/(prec+rec+1e-12)).max()
        metrics = {"AUC": float(auc), "AP": float(ap), "F1_max": float(f1)}
        # salvar
        from .model_utils import save_joblib
        save_joblib(clf, BASE/"models"/"recommender.pkl")
        save_metrics(metrics, BASE/"models"/"metrics.json")
        print("OK: recommender.pkl salvo. Métricas:", metrics)

if __name__ == "__main__":
    train()
