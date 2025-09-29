# app/app.py — App simples para RH (Match Vaga × Candidato) — versão só com % de match
from pathlib import Path
import json, zipfile, re
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from src.utils import normalize_text

# ---------------- Configuração ----------------
st.set_page_config(page_title="Match Vaga × Candidato", layout="wide")

BASE = Path(__file__).resolve().parents[1]
DATA   = BASE / "data" / "processed"
RAW    = BASE / "data" / "raw"
RAWZIP = BASE / "data" / "raw_zips"
MODELS = BASE / "models"

if "shortlist" not in st.session_state:
    st.session_state.shortlist = []

# ---------------- Utilidades ----------------
def _clean(x):
    return str(x).strip() if pd.notna(x) and str(x).strip().lower() != "nan" else ""

def _norm_id(x) -> str:
    return re.sub(r"[^\dA-Za-z_-]", "", str(x))

def _load_json(folder: str, fname: str):
    p = RAW / folder / fname
    if p.exists():
        return json.load(open(p, "r", encoding="utf-8"))
    z = RAWZIP / f"{folder}.zip"
    if z.exists():
        with zipfile.ZipFile(z, "r") as zf:
            inside = f"{folder}/{fname}"
            if inside in zf.namelist():
                with zf.open(inside) as fh:
                    return json.load(fh)
            if fname in zf.namelist():
                with zf.open(fname) as fh:
                    return json.load(fh)
    return {}

def enrich_vagas_titulo(vagas_df: pd.DataFrame) -> pd.DataFrame:
    raw = _load_json("vagas", "vagas.json")
    out = vagas_df.copy()
    out["vaga_id"] = out["vaga_id"].astype(str)
    if isinstance(raw, dict) and raw:
        rows = []
        for vid, rec in raw.items():
            ib = rec.get("informacoes_basicas", {}) or {}
            rows.append({
                "vaga_id": str(vid),
                "titulo_raw": _clean(ib.get("titulo_vaga") or ib.get("titulo") or rec.get("titulo"))
            })
        raw_df = pd.DataFrame(rows)
        out = out.merge(raw_df, on="vaga_id", how="left")
    out["titulo_vaga"] = out.apply(
        lambda r: _clean(r.get("titulo_vaga") or r.get("titulo_raw") or r.get("titulo") or r.get("descricao") or f"Vaga {r.get('vaga_id')}"),
        axis=1
    )
    return out.drop(columns=[c for c in ["titulo_raw"] if c in out.columns])

def load_prospects_map() -> dict[str, list[str]]:
    raw = _load_json("prospects", "prospects.json")
    mp = {}
    if isinstance(raw, dict):
        for vaga_id, rec in raw.items():
            ids = []
            for p in (rec.get("prospects", []) or []):
                code = str(p.get("codigo") or "").strip()
                if code: ids.append(_norm_id(code))
            if ids:
                mp[str(vaga_id)] = ids
    return mp

def build_job_text(vrow: pd.Series) -> str:
    """Monta o texto da vaga com vários fallbacks para evitar vetor zerado."""
    fields = [
        "requisitos_texto", "descricao", "descricao_detalhada",
        "responsabilidades", "atividades",
        "stack_desejada", "stack",
        "titulo_vaga", "titulo"
    ]
    parts = [ _clean(vrow.get(f)) for f in fields if _clean(vrow.get(f)) ]
    return " ".join(parts)

def compute_sim_tfidf(vect, apps_df: pd.DataFrame, vaga_row: pd.Series, batch: int = 10000) -> np.ndarray:
    """Cosseno TF-IDF com proteção a divisor zero e processamento em lotes."""
    job_text = normalize_text(build_job_text(vaga_row))
    B = vect.transform([job_text])                   # (1, V)
    B_norm = float(np.sqrt(B.multiply(B).sum()))     # escalar

    n = len(apps_df)
    out = np.zeros(n, dtype=float)
    if B_norm == 0.0:
        # sem texto na vaga → todos 0%
        return out

    cv_vals    = apps_df["cv_text_pt"].fillna("").astype(str).values
    stack_vals = apps_df["stack"].fillna("").astype(str).values

    for s in range(0, n, batch):
        e = min(n, s + batch)
        texts = [ normalize_text(cv_vals[i] + " " + stack_vals[i]) for i in range(s, e) ]
        A = vect.transform(texts)                                        # (m, V)
        A_norm = np.sqrt(A.multiply(A).sum(axis=1)).A.ravel()            # (m,)
        dots = (A @ B.T).A.ravel()                                       # (m,)
        out[s:e] = dots / (A_norm * B_norm + 1e-12)
    return out

# ---------------- Carregamento ----------------
@st.cache_data
def load_data():
    apps = pd.read_csv(DATA / "applicants.csv")
    vagas = pd.read_csv(DATA / "vagas.csv")
    prospects = pd.read_csv(DATA / "prospects.csv")
    vagas = enrich_vagas_titulo(vagas)
    return apps, vagas, prospects

@st.cache_resource
def load_vectorizer():
    return load(MODELS / "tfidf_vectorizer.joblib")

apps, vagas, prospects = load_data()
vect = load_vectorizer()
prospects_map = load_prospects_map()

# ---------------- UI ----------------
st.title("🔎 Match de Candidatos por Vaga")

left, right = st.columns([1.05, 2.0], gap="large")

with left:
    st.subheader("1) Escolha a vaga")
    idx = st.selectbox(
        "Código e nome da vaga",
        vagas.index,
        format_func=lambda i: f"{vagas.loc[i,'vaga_id']} — {vagas.loc[i,'titulo_vaga'][:80]}",
    )
    vaga_row = vagas.loc[idx]
    vaga_id = str(vaga_row["vaga_id"])

    st.subheader("2) Origem dos candidatos")
    fonte = st.radio(
        "Avaliar candidatos de:",
        ["Todos (Applicants)", "Apenas indicados a esta vaga (Prospects)"],
        index=0
    )

    st.subheader("3) Parâmetros")
    topk  = st.slider("Quantidade para revisar", 10, 500, 50, step=10)
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
            use_container_width=True, height=220
        )
        st.download_button(
            "⬇️ Exportar selecionados (CSV)",
            df_sel.to_csv(index=False).encode("utf-8"),
            file_name="selecionados_para_entrevista.csv",
            mime="text/csv",
            use_container_width=True
        )

with right:
    st.subheader("4) Ranking de candidatos")

    # Base de avaliação
    if fonte.startswith("Apenas"):
        ids = set(prospects_map.get(vaga_id, []))
        base = apps[apps["applicant_id"].astype(str).map(_norm_id).isin(ids)].copy()
        st.caption(f"Exibindo **{len(base)}** prospects indicados para a vaga.")
    else:
        base = apps.copy()
        st.caption(f"Exibindo **{len(base)}** candidatos do banco geral (applicants).")

    if len(base) == 0:
        st.warning("Nenhum candidato para avaliar.")
        st.stop()

    # Scoring SOMENTE por similaridade TF-IDF (nada de modelo supervisionado)
    sims = compute_sim_tfidf(vect, base, vaga_row)
    base["match_score"] = sims
    base["Probabilidade de Match (%)"] = (base["match_score"] * 100.0).map(lambda v: f"{v:.1f}%")

    base = base[base["match_score"] >= thr].sort_values("match_score", ascending=False)
    st.caption(
        f"Candidatos avaliados: **{len(sims)}** · "
        f"Com match ≥ **{int(thr_p)}%**: **{len(base)}** · "
        f"Vaga: **{_clean(vaga_row.get('titulo_vaga')) or vaga_id}**"
    )

    # ---- Cards com Nome + % + Selecionar ----
    st.markdown("**Destaques**")
    cards = base.head(min(12, topk)).copy()
    cols = st.columns(3, gap="large")
    for i, (_, c) in enumerate(cards.iterrows()):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(
                    f"**{_clean(c.get('nome','(sem nome)'))}**  \n"
                    f"Prob. de match: **{c['Probabilidade de Match (%)']}**"
                )
                if st.button("Selecionar para entrevista", key=f"sel_{c.get('applicant_id','NA')}_{i}"):
                    row = {
                        "ID Candidato": c.get("applicant_id",""),
                        "Candidato": c.get("nome",""),
                        "Probabilidade de Match (%)": c.get("Probabilidade de Match (%)",""),
                        "vaga_id": vaga_id
                    }
                    if row not in st.session_state.shortlist:
                        st.session_state.shortlist.append(row)

    st.markdown("---")
    st.subheader("Lista compacta")
    table = base.loc[:, ["applicant_id", "nome", "Probabilidade de Match (%)"]].head(topk).copy()
    table = table.rename(columns={
        "applicant_id": "ID Candidato",
        "nome": "Candidato"
    })
    st.dataframe(table, use_container_width=True, height=420)
    st.download_button(
        "⬇️ Exportar lista (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="ranking_candidatos.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")
st.caption("Ajuste **Quantidade** e **Corte mínimo de match** para controlar sua lista.")
