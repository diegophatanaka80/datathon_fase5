import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import sys

# ============================================================
# CONFIGURAÇÕES DO PROJETO  (ajuste BASE se sua pasta mudar)
# ============================================================
BASE = Path(r"C:\Users\dphat\OneDrive\Documentos\Cursos\FIAP\PosTech_DataAnalytics\fase5\Datathon Decision")
DATA_PROCESSED = BASE / "data" / "processed"
DATA_RAW = BASE / "data" / "raw"
MODELS_DIR = BASE / "models"

# Para permitir "from src import ..."
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

st.set_page_config(page_title="Recomendador de Match — Decision", layout="wide")

# ============================================================
# IMPORTA MESMAS ETAPAS DO TREINO
# ============================================================
try:
    from src.preprocessing import basic_preprocessing
except Exception as e:
    basic_preprocessing = None
    print("Aviso: src.preprocessing.basic_preprocessing não importada:", e)

try:
    from src.feature_engineering import make_features
except Exception as e:
    make_features = None
    print("Aviso: src.feature_engineering.make_features não importada:", e)

# ============================================================
# ARTEFATOS DO MODELO
# ============================================================
@st.cache_resource
def load_artifacts():
    model_path = MODELS_DIR / "recommender.pkl"
    meta_path = MODELS_DIR / "recommender_meta.json"
    if not model_path.exists() or not meta_path.exists():
        st.error("Artefatos em 'models/' não encontrados. Treine o modelo primeiro.")
        st.stop()
    pipe = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return pipe, meta

pipe, meta = load_artifacts()
best_thr = float(meta.get("best_threshold", 0.5))
num_cols = list(meta.get("num_cols", []))
cat_cols = list(meta.get("cat_cols", []))

# ============================================================
# LEITURA & JUNÇÃO DOS 3 CSVs (vagas/applicants/prospects)
# ============================================================
def _first_existing(cands):
    for p in cands:
        if p.exists():
            return p
    return None

def _read_csv_any(path: Path) -> pd.DataFrame:
    # tenta encodings comuns
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except Exception:
            continue
    # última tentativa sem encoding explícito
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

@st.cache_data
def load_merged_df() -> pd.DataFrame:
    # Candidatos de caminhos (processed tem prioridade, depois raw, depois a raiz do BASE)
    vagas_path = _first_existing([
        DATA_PROCESSED / "vaga_vf.csv",
        DATA_PROCESSED / "vagas_vf.csv",
        DATA_PROCESSED / "vagas.csv",
        DATA_RAW / "vaga_vf.csv",
        DATA_RAW / "vagas_vf.csv",
        DATA_RAW / "vagas.csv",
        BASE / "vaga_vf.csv",
        BASE / "vagas_vf.csv",
        BASE / "vagas.csv",
    ])
    apps_path = _first_existing([
        DATA_PROCESSED / "applicants_vf.csv",
        DATA_PROCESSED / "candidatos_vf.csv",
        DATA_PROCESSED / "applicants.csv",
        DATA_RAW / "applicants_vf.csv",
        DATA_RAW / "candidatos_vf.csv",
        DATA_RAW / "applicants.csv",
        BASE / "applicants_vf.csv",
        BASE / "candidatos_vf.csv",
        BASE / "applicants.csv",
    ])
    pros_path = _first_existing([
        DATA_PROCESSED / "prospects_vf.csv",
        DATA_PROCESSED / "prospects.csv",
        DATA_RAW / "prospects_vf.csv",
        DATA_RAW / "prospects.csv",
        BASE / "prospects_vf.csv",
        BASE / "prospects.csv",
    ])

    missing = [name for name, p in [("vaga_vf.csv", vagas_path), ("applicants_vf.csv", apps_path), ("prospects_vf.csv", pros_path)] if p is None]
    if missing:
        st.error(f"Arquivos CSV não encontrados: {', '.join(missing)}. Verifique se estão descompactados.")
        st.stop()

    df_vagas = _read_csv_any(vagas_path)
    df_apps  = _read_csv_any(apps_path)
    df_pros  = _read_csv_any(pros_path)

    # Normaliza nomes de chaves para job_id / applicant_id
    job_key_vagas = _pick(df_vagas, ["job_id", "jobId", "id_vaga", "vaga_id", "id"])
    job_key_pros  = _pick(df_pros,  ["job_id", "jobId", "id_vaga", "vaga_id", "id"])
    app_key_apps  = _pick(df_apps,  ["applicant_id", "appId", "id_candidato", "candidato_id", "id"])
    app_key_pros  = _pick(df_pros,  ["applicant_id", "appId", "id_candidato", "candidato_id", "id"])

    # renomeia para padrão
    if job_key_vagas and job_key_vagas != "job_id":
        df_vagas = df_vagas.rename(columns={job_key_vagas: "job_id"})
    if job_key_pros and job_key_pros != "job_id":
        df_pros = df_pros.rename(columns={job_key_pros: "job_id"})
    if app_key_apps and app_key_apps != "applicant_id":
        df_apps = df_apps.rename(columns={app_key_apps: "applicant_id"})
    if app_key_pros and app_key_pros != "applicant_id":
        df_pros = df_pros.rename(columns={app_key_pros: "applicant_id"})

    # Prefixa tudo exceto as chaves
    vagas_pref = _prefix_except_keys(df_vagas, "job__", ["job_id"])
    apps_pref  = _prefix_except_keys(df_apps,  "app__", ["applicant_id"])
    pros_pref  = _prefix_except_keys(df_pros,  "prospect__", ["job_id", "applicant_id"])

    # Merge prospects × vagas × applicants
    tmp = pros_pref.merge(vagas_pref, on="job_id", how="left")
    df = tmp.merge(apps_pref, on="applicant_id", how="left")

    # ID único vaga×candidato
    df["pair_id"] = df["job_id"].astype("string") + "::" + df["applicant_id"].astype("string")

    # Aplica o mesmo PRÉ + FEAT do treino
    if basic_preprocessing is not None:
        try:
            df = basic_preprocessing(df)
        except Exception as e:
            st.warning(f"basic_preprocessing falhou: {e}")
    if make_features is not None:
        try:
            df = make_features(df)
        except Exception as e:
            st.warning(f"make_features falhou: {e}")

    # Remove duplicatas de colunas
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

df = load_merged_df()

# Garante colunas esperadas pelo pipeline (evita erro do ColumnTransformer)
for c in num_cols:
    if c not in df.columns:
        df[c] = np.nan
for c in cat_cols:
    if c not in df.columns:
        df[c] = ""

# ============================================================
# DETECÇÃO DAS COLUNAS DE VAGA (CHAVE + TÍTULO)
#   — título vem de 'título_vaga' no CSV de vagas (após prefixo: job__título_vaga)
# ============================================================
JOB_KEY_CANDS   = ["job_id"]  # chave de filtro
JOB_TITLE_CANDS = ["job__título_vaga", "job__titulo_vaga", "job__titulo", "job__nome", "job__descricao", "job__descricao_vaga"]

def _pick_first(cands, cols):
    return next((c for c in cands if c in cols), None)

key_col = _pick_first(JOB_KEY_CANDS, df.columns)
job_title_col = _pick_first(JOB_TITLE_CANDS, df.columns)

if key_col is None:
    st.error("Coluna de chave da vaga não encontrada (esperado: job_id). Verifique os CSVs.")
    st.stop()

# ============================================================
# UI — CABEÇALHO & PARÂMETROS
# ============================================================
st.title("🔎 Recomendador de Match — Decision")
st.caption("Ranqueia candidatos por probabilidade de match para a vaga selecionada.")

with st.sidebar:
    st.header("Parâmetros")
    thr = st.slider("Threshold de decisão", 0.0, 1.0, value=float(best_thr), step=0.01)
    top_n = st.number_input("Top N para exibição", min_value=1, max_value=1000, value=50, step=1)
    only_above = st.checkbox("Mostrar apenas candidatos acima do threshold", value=False)
    st.caption(f"Threshold ótimo salvo (F1): {best_thr:.3f}")

# ============================================================
# DROPDOWN DE VAGAS — label = [job_id] título_vaga
# ============================================================
def _make_label(row):
    code = str(row.get(key_col, "")).strip()
    title = str(row.get(job_title_col, "")).strip() if job_title_col else ""
    if title and code:
        return f"[{code}] {title}"
    return title or code or "(vaga sem identificação)"

cols_for_jobs = [key_col] + ([job_title_col] if job_title_col else [])
jobs = df[cols_for_jobs].copy()
jobs["__key__"] = jobs[key_col].astype(str).str.strip()

# remove vazios/placeholder
jobs = jobs[jobs["__key__"].ne("") & jobs["__key__"].ne("-")]
jobs = jobs.drop_duplicates(subset="__key__")
jobs["__label__"] = jobs.apply(_make_label, axis=1)

label_to_key = dict(zip(jobs["__label__"], jobs["__key__"]))
opcoes = ["-"] + sorted(jobs["__label__"].tolist())
vaga_label = st.selectbox("Selecione a vaga:", options=opcoes, index=0)

# ============================================================
# FILTRO & FEATURES
# ============================================================
df_view = df.copy()
if vaga_label != "-":
    vaga_key = label_to_key[vaga_label]
    df_view = df_view[df_view[key_col].astype(str).str.strip() == str(vaga_key)]

df_view = df_view.loc[:, ~df_view.columns.duplicated()].copy()
feature_cols = list(dict.fromkeys(num_cols + cat_cols))
X_view = df_view[feature_cols].copy()

# ============================================================
# INFERÊNCIA
# ============================================================
try:
    proba = pipe.predict_proba(X_view)[:, 1]
except Exception as e:
    st.error(f"Falha na inferência (verifique pré-processamento/colunas esperadas): {e}")
    st.stop()

df_view["proba_match"] = proba
df_view["pred@thr"] = (df_view["proba_match"] >= thr).astype(int)

# ============================================================
# EXIBIÇÃO
# ============================================================
cand_id_cols = [c for c in ["app__id","app__nome","app__name","app__email"] if c in df_view.columns]
extra_cols = [c for c in [key_col, job_title_col] if c and (c in df_view.columns)]
display_cols = list(dict.fromkeys(cand_id_cols + extra_cols + ["proba_match","pred@thr"]))

df_rank = df_view.sort_values("proba_match", ascending=False)
if only_above:
    df_rank = df_rank[df_rank["pred@thr"] == 1]

# garante colunas únicas no DF final
df_rank = df_rank.loc[:, ~df_rank.columns.duplicated()].copy()

st.subheader("🧑‍💼 Ranking de candidatos por probabilidade de match")
st.dataframe(df_rank[display_cols].head(int(top_n)).reset_index(drop=True))

# Resumo
total = len(df_view)
acima = int((df_view["pred@thr"] == 1).sum())
st.caption(f"{acima}/{total} candidatos acima do threshold {thr:.2f} (após filtro da vaga).")

# ============================================================
# DOWNLOAD
# ============================================================
csv = df_rank[display_cols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Baixar ranking (CSV)", data=csv, file_name="ranking_match.csv", mime="text/csv")

# ============================================================
# DEBUG (opcional)
# ============================================================
with st.expander("Debug / Diagnóstico"):
    st.write("CSV usados: ")
    st.write(" - Preferência: data/processed → data/raw → raiz do projeto")
    st.write("Chave da vaga (key_col):", key_col, "| Título:", job_title_col)
    st.write("Features esperadas:", feature_cols)
    st.dataframe(df_view.head(5))
