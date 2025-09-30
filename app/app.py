import streamlit as st
import pandas as pd
import numpy as np
import json, joblib, sys
from pathlib import Path

# =========================================
# CONFIG (sempre relativo à raiz do repo)
# =========================================
BASE = Path.cwd()
DATA_PROCESSED = BASE / "data" / "processed"
DATA_RAW       = BASE / "data" / "raw"
MODELS_DIR     = BASE / "models"

if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# (Opcional) mesmas etapas do treino, se existirem
try:
    from src.preprocessing import basic_preprocessing
except Exception:
    basic_preprocessing = None
try:
    from src.feature_engineering import make_features
except Exception:
    make_features = None

st.set_page_config(page_title="Recomendador de Match — Decision", layout="wide")

# =========================================
# CARREGAMENTO DE MODELO E META (ROBUSTO)
# =========================================
MODEL_PATH_SMALL = MODELS_DIR / "recommender_small.joblib"
MODEL_PATH_FULL  = MODELS_DIR / "recommender.pkl"
META_PATH        = MODELS_DIR / "recommender_meta.json"

def _fail_with_hint(msg: str):
    hint = ""
    low = msg.lower()
    if "xgboost" in low or "xgb" in low:
        hint = (
            "Parece que o modelo foi treinado com **XGBoost** e o pacote não está instalado "
            "(ou a versão diverge). Adicione `xgboost==2.1.1` no **requirements.txt** e faça o redeploy."
        )
    elif "sklearn" in low or "scikit" in low:
        hint = (
            "Possível incompatibilidade de versão do **scikit-learn**. "
            "Tente fixar `scikit-learn==1.4.2` (junto de `joblib==1.3.2`, `numpy==1.26.4`, `pandas==2.2.2`) "
            "no **requirements.txt** e refaça o deploy."
        )
    else:
        hint = (
            "Verifique se **models/recommender_small.joblib** (ou **recommender.pkl**) "
            "foi treinado com bibliotecas que estão no seu **requirements.txt**."
        )

    st.error(f"Não foi possível carregar o modelo.\n\nDetalhe: {msg}\n\n{hint}")
    st.stop()

@st.cache_resource
def load_artifacts():
    # meta é obrigatório
    if not META_PATH.exists():
        st.error("Artefatos do modelo ausentes. Envie **models/recommender_meta.json** e o arquivo do modelo treinado.")
        st.stop()

    # tenta modelo compactado, depois o completo
    last_err = None
    for path in [MODEL_PATH_SMALL, MODEL_PATH_FULL]:
        if path.exists():
            try:
                pipe = joblib.load(path)
                with open(META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return pipe, meta
            except Exception as e:
                last_err = e

    if last_err is None:
        st.error("Arquivo de modelo não encontrado em **models/**. Envie `recommender_small.joblib` ou `recommender.pkl`.")
        st.stop()
    else:
        _fail_with_hint(str(last_err))

pipe, meta = load_artifacts()
BEST_THR = float(meta.get("best_threshold", 0.5))
NUM_COLS = list(meta.get("num_cols", []))
CAT_COLS = list(meta.get("cat_cols", []))

# =========================================
# HELPERS DE I/O E PRÉP
# =========================================
def _first_existing(cands):
    for p in cands:
        p = Path(p)
        if p.exists():
            return p
    return None

def _read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)

def _pick(df: pd.DataFrame, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def _prefix_except_keys(df: pd.DataFrame, prefix: str, keys: list[str]) -> pd.DataFrame:
    ren = {c: f"{prefix}{c}" for c in df.columns if c not in keys}
    out = df.rename(columns=ren).copy()
    for k in keys:
        if k in out.columns:
            out[k] = out[k].astype("string")
    return out

def _ensure_feature_columns(df: pd.DataFrame, num_cols, cat_cols):
    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
    for c in cat_cols:
        if c not in df.columns:
            df[c] = ""
    return df

# =========================================
# DADOS: dataset mínimo (preferido) ou merge dos 3 CSVs
# =========================================
MIN_CSV = DATA_PROCESSED / "app_inference.min.csv.gz"

@st.cache_data
def load_min_df():
    if not MIN_CSV.exists():
        return None
    df = pd.read_csv(MIN_CSV, low_memory=False)
    return df.loc[:, ~df.columns.duplicated()].copy()

@st.cache_data
def load_merged_df():
    vagas_path = _first_existing([
        DATA_PROCESSED / "vaga_vf.csv",
        DATA_PROCESSED / "vagas_vf.csv",
        DATA_RAW       / "vaga_vf.csv",
        DATA_RAW       / "vagas_vf.csv",
        BASE           / "vaga_vf.csv",
        BASE           / "vagas_vf.csv",
    ])
    apps_path = _first_existing([
        DATA_PROCESSED / "applicants_vf.csv",
        DATA_RAW       / "applicants_vf.csv",
        BASE           / "applicants_vf.csv",
    ])
    pros_path = _first_existing([
        DATA_PROCESSED / "prospects_vf.csv",
        DATA_RAW       / "prospects_vf.csv",
        BASE           / "prospects_vf.csv",
    ])
    if not all([vagas_path, apps_path, pros_path]):
        st.error("CSV(s) de **vagas/applicants/prospects** não encontrados no repositório.")
        st.stop()

    df_vagas = _read_csv_any(vagas_path)
    df_apps  = _read_csv_any(apps_path)
    df_pros  = _read_csv_any(pros_path)

    # Normaliza chaves -> job_id / applicant_id
    job_key_vagas = _pick(df_vagas, ["job_id","jobId","id_vaga","vaga_id","id"])
    job_key_pros  = _pick(df_pros,  ["job_id","jobId","id_vaga","vaga_id","id"])
    app_key_apps  = _pick(df_apps,  ["applicant_id","appId","id_candidato","candidato_id","id"])
    app_key_pros  = _pick(df_pros,  ["applicant_id","appId","id_candidato","candidato_id","id"])

    if job_key_vagas and job_key_vagas != "job_id": df_vagas = df_vagas.rename(columns={job_key_vagas:"job_id"})
    if job_key_pros  and job_key_pros  != "job_id": df_pros  = df_pros.rename(columns={job_key_pros:"job_id"})
    if app_key_apps  and app_key_apps  != "applicant_id": df_apps = df_apps.rename(columns={app_key_apps:"applicant_id"})
    if app_key_pros  and app_key_pros  != "applicant_id": df_pros = df_pros.rename(columns={app_key_pros:"applicant_id"})

    # Prefixa p/ evitar colisão
    vagas_pref = _prefix_except_keys(df_vagas, "job__", ["job_id"])
    apps_pref  = _prefix_except_keys(df_apps,  "app__", ["applicant_id"])
    pros_pref  = _prefix_except_keys(df_pros,  "prospect__", ["job_id","applicant_id"])

    # Merge
    df = pros_pref.merge(vagas_pref, on="job_id", how="left").merge(apps_pref, on="applicant_id", how="left")
    df["pair_id"] = df["job_id"].astype("string") + "::" + df["applicant_id"].astype("string")

    # Mesmo PRÉ + FEAT do treino (se disponíveis)
    if basic_preprocessing is not None:
        try:
            df = basic_preprocessing(df)
        except Exception:
            pass
    if make_features is not None:
        try:
            df = make_features(df)
        except Exception:
            pass

    return df.loc[:, ~df.columns.duplicated()].copy()

@st.cache_data
def load_data():
    df = load_min_df()
    if df is None:
        df = load_merged_df()
    df = _ensure_feature_columns(df, NUM_COLS, CAT_COLS)
    return df.loc[:, ~df.columns.duplicated()].copy()

df = load_data()

# =========================================
# DETECÇÃO: chave e título da vaga
# =========================================
JOB_KEY_CANDS   = ["job_id"]
JOB_TITLE_CANDS = ["job__título_vaga","job__titulo_vaga","job__titulo","job__nome","job__descricao","job__descricao_vaga"]

def _pick_first(cands, cols):
    return next((c for c in cands if c in cols), None)

key_col = _pick_first(JOB_KEY_CANDS, df.columns)
job_title_col = _pick_first(JOB_TITLE_CANDS, df.columns)

if key_col is None:
    st.error("Coluna de identificação da vaga (job_id) não encontrada.")
    st.stop()

# =========================================
# UI — simples e executiva
# =========================================
st.title("🔎 Recomendador de Match — Decision")
st.caption("Selecione a vaga e veja os candidatos com maior probabilidade de match.")

with st.sidebar:
    st.header("Opções")
    thr = st.slider("Threshold de decisão", 0.0, 1.0, value=BEST_THR, step=0.01,
                    help="Aumente para priorizar precisão; diminua para aumentar a abrangência.")
    top_n = st.number_input("Quantidade a exibir (Top-N)", min_value=1, max_value=1000, value=50, step=1)
    only_above = st.checkbox("Mostrar apenas acima do threshold", value=False)

# Dropdown — rótulo: [job_id] título_vaga
def _make_label(row):
    code = str(row.get(key_col, "")).strip()
    title = str(row.get(job_title_col, "")).strip() if job_title_col else ""
    if title and code:
        return f"[{code}] {title}"
    return title or code or "(vaga sem identificação)"

cols_for_jobs = [key_col] + ([job_title_col] if job_title_col else [])
jobs = df[cols_for_jobs].copy()
jobs["__key__"] = jobs[key_col].astype(str).str.strip()
jobs = jobs[jobs["__key__"].ne("") & jobs["__key__"].ne("-")]
jobs = jobs.drop_duplicates(subset="__key__")
jobs["__label__"] = jobs.apply(_make_label, axis=1)

label_to_key = dict(zip(jobs["__label__"], jobs["__key__"]))
opcoes = ["-"] + sorted(jobs["__label__"].tolist())
vaga_label = st.selectbox("Selecione a vaga:", options=opcoes, index=0)

# =========================================
# Filtro e inferência
# =========================================
df_view = df.copy()
if vaga_label != "-":
    vaga_key = label_to_key[vaga_label]
    df_view = df_view[df_view[key_col].astype(str).str.strip() == str(vaga_key)]

df_view = df_view.loc[:, ~df_view.columns.duplicated()].copy()
feature_cols = list(dict.fromkeys(NUM_COLS + CAT_COLS))
X_view = df_view[feature_cols].copy()

try:
    proba = pipe.predict_proba(X_view)[:, 1]
except Exception as e:
    _fail_with_hint(str(e))

df_view["proba_match"] = proba
df_view["pred@thr"] = (df_view["proba_match"] >= thr).astype(int)

# Indicadores executivos
total = len(df_view)
acima = int((df_view["pred@thr"] == 1).sum())
media = float(df_view["proba_match"].mean()) if total else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Candidatos na vaga", f"{total}")
col2.metric("Acima do threshold", f"{acima}")
col3.metric("Probabilidade média", f"{media:.2%}")

# Tabela de ranking
cand_id_cols = [c for c in ["app__id","app__nome","app__email"] if c in df_view.columns]
extra_cols = [c for c in [key_col, job_title_col] if c and (c in df_view.columns)]
display_cols = list(dict.fromkeys(cand_id_cols + extra_cols + ["proba_match","pred@thr"]))

df_rank = df_view.sort_values("proba_match", ascending=False)
if only_above:
    df_rank = df_rank[df_rank["pred@thr"] == 1]

df_rank = df_rank.loc[:, ~df_rank.columns.duplicated()].copy()

st.subheader("🧑‍💼 Ranking de candidatos")
st.dataframe(df_rank[display_cols].head(int(top_n)).reset_index(drop=True), use_container_width=True)

# Download
csv = df_rank[display_cols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Baixar ranking (CSV)", data=csv, file_name="ranking_match.csv", mime="text/csv")
