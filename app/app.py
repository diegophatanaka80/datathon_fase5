# app/app.py  —  App de negócio para RH: Match Vaga × Candidato
from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# ============================================================
# Configuração da página
# ============================================================
st.set_page_config(page_title="Match Vaga × Candidato", layout="wide")

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"

# Estado global da shortlist
if "shortlist" not in st.session_state:
    st.session_state.shortlist = []


# ============================================================
# Utils
# ============================================================
def _clean(x):
    """Converte NaN/None/‘nan’ em string vazia e remove espaços extras."""
    return str(x).strip() if pd.notna(x) and str(x).strip().lower() != "nan" else ""


def _norm_id(x) -> str:
    """Normaliza IDs (apenas letras, números, - e _)."""
    return re.sub(r"[^\dA-Za-z_-]", "", str(x))


def normalize_text(text: str) -> str:
    """Normaliza texto: minúsculas, remove acentos e colapsa espaços."""
    try:
        # usa a do projeto se existir
        from src.utils import normalize_text as _nt
        return _nt(text)
    except Exception:
        import unicodedata, re as _re
        if text is None:
            return ""
        t = str(text)
        t = unicodedata.normalize("NFKD", t)
        t = "".join(c for c in t if not unicodedata.combining(c))
        t = t.lower()
        t = _re.sub(r"\s+", " ", t).strip()
        return t


def read_processed(base: str) -> pd.DataFrame:
    """Tenta ler _mini.csv.gz → .csv.gz → .csv (nessa ordem)."""
    for name in (f"{base}_mini.csv.gz", f"{base}.csv.gz", f"{base}.csv"):
        p = DATA / name
        if p.exists():
            return pd.read_csv(p, compression="infer")
    raise FileNotFoundError(f"Não achei {base} em {DATA} (tente *_mini.csv.gz, .csv.gz ou .csv).")


def build_job_text(vrow: pd.Series) -> str:
    """Monta o texto da vaga usando vários campos (para evitar vetor zerado)."""
    fields = [
        "requisitos_texto",
        "descricao",
        "descricao_detalhada",
        "responsabilidades",
        "atividades",
        "stack_desejada",
        "stack",
        "titulo_vaga",
        "titulo",
    ]
    parts = [_clean(vrow.get(f)) for f in fields if _clean(vrow.get(f))]
    return " ".join(parts)


def build_job_label(vrow: pd.Series) -> str:
    """Rótulo do select: 'vaga_id — título' (com fallbacks)."""
    title = _clean(vrow.get("titulo_vaga") or vrow.get("titulo") or vrow.get("descricao"))
    if not title:
        title = f"Vaga {vrow.get('vaga_id')}"
    return f"{vrow.get('vaga_id')} — {title}"


def make_prospects_map(prospects_df: pd.DataFrame) -> dict[str, set[str]]:
    """
    Mapeia vaga_id -> {applicant_ids} a partir do prospects.csv.
    Tenta achar colunas plausíveis (vaga/applicant/codigo).
    """
    if prospects_df is None or len(prospects_df) == 0:
        return {}
    cols = prospects_df.columns.str.lower().tolist()

    vaga_col = next((c for c in prospects_df.columns if "vaga" in c.lower()), None)
    cand_col = next(
        (c for c in prospects_df.columns if any(k in c.lower() for k in ["applicant", "codigo", "candidate"])),
        None,
    )
    if not vaga_col or not cand_col:
        return {}

    mp: dict[str, set[str]] = {}
    for _, r in prospects_df[[vaga_col, cand_col]].dropna().iterrows():
        v = str(r[vaga_col])
        a = _norm_id(r[cand_col])
        if not a:
            continue
        mp.setdefault(v, set()).add(a)
    return mp


def compute_sim_tfidf(vect, apps_df: pd.DataFrame, vaga_row: pd.Series, batch: int = 10000) -> np.ndarray:
    """Similaridade cosseno TF-IDF (com processamento em lotes e proteção a zero)."""
    job_text = normalize_text(build_job_text(vaga_row))
    B = vect.transform([job_text])  # (1, V)
    B_norm = float(np.sqrt(B.multiply(B).sum()))
    n = len(apps_df)
    out = np.zeros(n, dtype=float)
    if B_norm == 0.0 or n == 0:
        return out

    cv_vals = apps_df["cv_text_pt"].fillna("").astype(str).values
    stack_vals = apps_df["stack"].fillna("").astype(str).values

    for s in range(0, n, batch):
        e = min(n, s + batch)
        texts = [normalize_text(cv_vals[i] + " " + stack_vals[i]) for i in range(s, e)]
        A = vect.transform(texts)
        A_norm = np.sqrt(A.multiply(A).sum(axis=1)).A.ravel()
        dots = (A @ B.T).A.ravel()
        out[s:e] = dots / (A_norm * B_norm + 1e-12)
    return out


# ============================================================
# Carregamento (cache)
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    apps = read_processed("applicants")
    vagas = read_processed("vagas")
    try:
        prospects = read_processed("prospects")
    except Exception:
        prospects = pd.DataFrame()
    # garantias de tipos
    if "vaga_id" in vagas.columns:
        vagas["vaga_id"] = vagas["vaga_id"].astype(str)
    if "applicant_id" in apps.columns:
        apps["applicant_id"] = apps["applicant_id"].astype(str)
    return apps, vagas, prospects


@st.cache_resource(show_spinner=False)
def load_vectorizer():
    return load(MODELS / "tfidf_vectorizer.joblib")


apps, vagas, prospects = load_data()
vect = load_vectorizer()
prospects_map = make_prospects_map(prospects)

# ============================================================
# UI
# ============================================================
st.title("🔎 Match de Candidatos por Vaga")

left, right = st.columns([1.05, 2.0], gap="large")

# ---------------- Painel ESQUERDO ----------------
with left:
    st.subheader("1) Escolha a vaga")
    # selectbox com rótulo "vaga_id — título"
    idx = st.selectbox(
        "Código e nome da vaga",
        vagas.index,
        format_func=lambda i: build_job_label(vagas.loc[i]),
    )
    vaga_row = vagas.loc[idx]
    vaga_id = str(vaga_row.get("vaga_id"))

    st.subheader("2) Origem dos candidatos")
    fonte = st.radio(
        "Avaliar candidatos de:",
        ["Todos (Applicants)", "Apenas indicados a esta vaga (Prospects)"],
        index=0,
    )

    st.subheader("3) Parâmetros")
    topk = st.slider("Quantidade para revisar", 10, 500, 50, step=10)
    thr_p = st.slider("Corte mínimo de match (%)", 0.0, 100.0, 10.0, step=1.0)
    thr = thr_p / 100.0

    st.markdown("---")
    st.subheader("Selecionados para entrevista")
    if len(st.session_state.shortlist) == 0:
        st.caption("Nenhum candidato selecionado ainda.")
    else:
        df_sel = pd.DataFrame(st.session_state.shortlist)
        st.dataframe(
            df_sel[["ID Candidato", "Candidato", "Probabilidade de Match (%)"]],
            use_container_width=True,
            height=220,
        )
        st.download_button(
            "⬇️ Exportar selecionados (CSV)",
            df_sel.to_csv(index=False).encode("utf-8"),
            file_name="selecionados_para_entrevista.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ---------------- Painel DIREITO ----------------
with right:
    st.subheader("4) Ranking de candidatos")

    # Filtragem por fonte
    if fonte.startswith("Apenas"):
        ids = set(prospects_map.get(vaga_id, []))
        if ids:
            base = apps[apps["applicant_id"].astype(str).map(_norm_id).isin(ids)].copy()
        else:
            base = apps.iloc[0:0].copy()
        st.caption(f"Exibindo **{len(base)}** prospects indicados para a vaga.")
    else:
        base = apps.copy()
        st.caption(f"Exibindo **{len(base)}** candidatos do banco geral (applicants).")

    if len(base) == 0:
        st.warning("Nenhum candidato para avaliar.")
        st.stop()

    # Scoring por similaridade TF-IDF
    sims = compute_sim_tfidf(vect, base, vaga_row)
    base["match_score"] = sims
    base["Probabilidade de Match (%)"] = (base["match_score"] * 100.0).map(lambda v: f"{v:.1f}%")

    base = base[base["match_score"] >= thr].sort_values("match_score", ascending=False)

    # Info do contexto
    try:
        titulo = build_job_label(vaga_row).split(" — ", 1)[1]
    except Exception:
        titulo = vaga_id
    st.caption(
        f"Candidatos avaliados: **{len(sims)}** · "
        f"Com match ≥ **{int(thr_p)}%**: **{len(base)}** · "
        f"Vaga: **{titulo}**"
    )

    # ---- Cards (top 12 ou até topk) ----
    st.markdown("**Destaques**")
    cards = base.head(min(12, topk)).copy()
    cols = st.columns(3, gap="large")
    for i, (_, c) in enumerate(cards.iterrows()):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(
                    f"**{_clean(c.get('nome', '(sem nome)'))}**  \n"
                    f"Prob. de match: **{c['Probabilidade de Match (%)']}**"
                )
                if st.button("Selecionar para entrevista", key=f"sel_{c.get('applicant_id','NA')}_{i}"):
                    row = {
                        "ID Candidato": c.get("applicant_id", ""),
                        "Candidato": c.get("nome", ""),
                        "Probabilidade de Match (%)": c.get("Probabilidade de Match (%)", ""),
                        "vaga_id": vaga_id,
                    }
                    if row not in st.session_state.shortlist:
                        st.session_state.shortlist.append(row)

    # ---- Lista tabular compacta ----
    st.markdown("---")
    st.subheader("Lista compacta")
    table = base.loc[:, ["applicant_id", "nome", "Probabilidade de Match (%)"]].head(topk).copy()
    table = table.rename(columns={"applicant_id": "ID Candidato", "nome": "Candidato"})
    st.dataframe(table, use_container_width=True, height=420)
    st.download_button(
        "⬇️ Exportar lista (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="ranking_candidatos.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")
st.caption("Ajuste **Quantidade** e **Corte mínimo de match** para controlar sua lista.")
