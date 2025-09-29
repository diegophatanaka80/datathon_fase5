# bootstrap_datathon.py
# Cria a estrutura do projeto, processa JSONs (se existirem) e treina o TF-IDF.

from pathlib import Path
import os, json, zipfile, re, unicodedata
import pandas as pd

# ============ CONFIG ============
# Ajuste para a sua pasta local (use r"..." no Windows ou barras /)
BASE = Path(r"C:/Users/dphat/OneDrive/Documentos/Cursos/FIAP/PosTech_DataAnalytics/fase5/Datathon Decision")

# Flags de execução
DO_CREATE_STRUCTURE = True
DO_EXTRACT_JSONS    = True     # extrair zips de data/raw_zips para data/raw
DO_BUILD_CSVS       = True     # gerar CSVs normalizados em data/processed
DO_TRAIN_VECTORIZER = True     # treinar e salvar models/tfidf_vectorizer.joblib

# Limites p/ treino rápido
MAX_DOCS_EACH = 3000
MAX_FEATURES  = 20000
# ================================


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def ensure_structure(base: Path):
    """Cria as pastas principais do projeto."""
    for p in [
        base,
        base / "app",
        base / "src",
        base / "models",
        base / "tests",
        base / "notebooks",
        base / "data",
        base / "data" / "raw",
        base / "data" / "raw_zips",
        base / "data" / "processed",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def write_files(base: Path):
    """Escreve arquivos básicos (requirements, README, src/*, app/app.py)."""
    (base / "requirements.txt").write_text(
        "streamlit==1.37.0\n"
        "scikit-learn==1.5.0\n"
        "pandas==2.2.2\n"
        "joblib==1.4.2\n"
        "scipy==1.13.1\n",
        encoding="utf-8",
    )

    (base / "README.md").write_text(
        "# Datathon Fase 5 — MVP de Match Candidatos x Vagas\n\n"
        "## Como rodar\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "python -m src.train\n"
        "streamlit run app/app.py\n"
        "```\n",
        encoding="utf-8",
    )

    (base / "src" / "utils.py").write_text(
        "import re, unicodedata\n"
        "def normalize_text(s: str) -> str:\n"
        "    if not s: return \"\"\n"
        "    s = str(s)\n"
        "    s = unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode('ascii')\n"
        "    s = re.sub(r'\\s+', ' ', s).strip().lower()\n"
        "    return s\n",
        encoding="utf-8",
    )

    (base / "src" / "preprocessing.py").write_text(
        "import pandas as pd\nfrom pathlib import Path\n"
        "def load_processed(data_dir: Path):\n"
        "    apps = pd.read_csv(data_dir / 'applicants.csv')\n"
        "    vagas = pd.read_csv(data_dir / 'vagas.csv')\n"
        "    prospects = pd.read_csv(data_dir / 'prospects.csv')\n"
        "    return apps, vagas, prospects\n",
        encoding="utf-8",
    )

    (base / "src" / "feature_engineering.py").write_text(
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "def build_vectorizer():\n"
        "    return TfidfVectorizer(max_features=20000, ngram_range=(1,2), preprocessor=None, lowercase=True)\n",
        encoding="utf-8",
    )

    (base / "src" / "model_utils.py").write_text(
        "import joblib\nfrom pathlib import Path\n"
        "def save_model(obj, path: Path):\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    joblib.dump(obj, path)\n"
        "def load_model(path: Path):\n"
        "    return joblib.load(path)\n",
        encoding="utf-8",
    )

    (base / "src" / "train.py").write_text(
        "from pathlib import Path\nimport pandas as pd\n"
        "from .preprocessing import load_processed\n"
        "from .feature_engineering import build_vectorizer\n"
        "from .model_utils import save_model\n"
        f"MAX_DOCS_EACH = {MAX_DOCS_EACH}\n"
        f"MAX_FEATURES = {MAX_FEATURES}\n"
        "def train(data_dir: Path, models_dir: Path):\n"
        "    apps, vagas, _ = load_processed(data_dir)\n"
        "    vec = build_vectorizer()\n"
        "    app_text = (apps['cv_text_pt'].fillna('') + ' ' + apps['stack'].fillna('')).astype(str).tolist()[:MAX_DOCS_EACH]\n"
        "    job_text = (vagas['requisitos_texto'].fillna('') + ' ' + vagas['stack_desejada'].fillna('')).astype(str).tolist()[:MAX_DOCS_EACH]\n"
        "    corpus = app_text + job_text\n"
        "    vec.fit(corpus)\n"
        "    save_model(vec, models_dir / 'tfidf_vectorizer.joblib')\n"
        "    return {'vocab_size': len(vec.get_feature_names_out())}\n"
        "if __name__ == '__main__':\n"
        "    base = Path(__file__).resolve().parents[1]\n"
        "    print(train(base / 'data' / 'processed', base / 'models'))\n",
        encoding="utf-8",
    )

    (base / "app" / "app.py").write_text(
        "import streamlit as st\nimport pandas as pd\nfrom pathlib import Path\nfrom joblib import load\nimport numpy as np\n"
        "from src.utils import normalize_text\n"
        "BASE = Path(__file__).resolve().parents[1]\n"
        "DATA = BASE / 'data' / 'processed'\n"
        "MODELS = BASE / 'models'\n"
        "@st.cache_data\ndef load_data():\n"
        "    apps = pd.read_csv(DATA / 'applicants.csv')\n"
        "    vagas = pd.read_csv(DATA / 'vagas.csv')\n"
        "    prospects = pd.read_csv(DATA / 'prospects.csv')\n"
        "    return apps, vagas, prospects\n"
        "@st.cache_resource\ndef load_vectorizer():\n"
        "    return load(MODELS / 'tfidf_vectorizer.joblib')\n"
        "def compute_similarity(vectorizer, apps_df, vaga_row):\n"
        "    app_text = (apps_df['cv_text_pt'].fillna('') + ' ' + apps_df['stack'].fillna('')).map(normalize_text).tolist()\n"
        "    job_text = [normalize_text(str(vaga_row.get('requisitos_texto','')) + ' ' + str(vaga_row.get('stack_desejada','')))]\n"
        "    A = vectorizer.transform(app_text)\n"
        "    B = vectorizer.transform(job_text)\n"
        "    A_norm = np.sqrt((A.multiply(A)).sum(axis=1))\n"
        "    B_norm = np.sqrt((B.multiply(B)).sum(axis=1))\n"
        "    scores = (A @ B.T) / (A_norm * B_norm + 1e-12)\n"
        "    return np.asarray(scores).ravel()\n"
        "st.set_page_config(page_title='Match Candidatos x Vagas', layout='wide')\n"
        "st.title('Match de Candidatos x Vagas (MVP)')\n"
        "apps, vagas, prospects = load_data()\n"
        "vect = load_vectorizer()\n"
        "left, right = st.columns([1,2])\n"
        "with left:\n"
        "    st.subheader('Seleção da Vaga')\n"
        "    opt = vagas[['vaga_id','titulo']].astype(str)\n"
        "    idx = st.selectbox('Escolha a vaga', opt.index, format_func=lambda i: f\"{opt.loc[i,'vaga_id']} - {opt.loc[i,'titulo']}\")\n"
        "    vrow = vagas.loc[idx]\n"
        "    fonte = st.radio('Fonte de candidatos', ['applicants','prospects'])\n"
        "    top_k = st.slider('Top K', 10, 500, 50, step=10)\n"
        "    min_score = st.slider('Score mínimo (0-1)', 0.0, 1.0, 0.1, step=0.05)\n"
        "    filtro_local = st.text_input('Filtro de localidade (opcional)')\n"
        "with right:\n"
        "    st.subheader('Ranking de Candidatos')\n"
        "    cand_df = apps.copy()\n"
        "    scores = compute_similarity(vect, cand_df, vrow)\n"
        "    cand_df['match_score'] = scores\n"
        "    if filtro_local:\n"
        "        cand_df = cand_df[cand_df['localidade'].fillna('').str.contains(filtro_local, case=False)]\n"
        "    cand_df = cand_df.sort_values('match_score', ascending=False)\n"
        "    cand_df = cand_df[cand_df['match_score'] >= min_score].head(top_k)\n"
        "    st.dataframe(cand_df[['applicant_id','nome','localidade','senioridade','cargo_atual','stack','match_score']])\n"
        "    st.download_button('Baixar CSV', cand_df.to_csv(index=False).encode('utf-8'), file_name='ranking_candidatos.csv', mime='text/csv')\n"
        "st.markdown('---')\n"
        "st.caption('MVP baseado em TF-IDF entre requisitos da vaga e CV/stack do candidato.')\n",
        encoding="utf-8",
    )


def safe_unzip(zip_path: Path, outdir: Path):
    if zip_path.exists():
        outdir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(outdir)


def extract_raw_if_any(base: Path):
    """Extrai applicants.zip, prospects.zip, vagas.zip de data/raw_zips para data/raw."""
    raw_zips = base / "data" / "raw_zips"
    raw      = base / "data" / "raw"
    for name in ["applicants", "prospects", "vagas"]:
        zp = raw_zips / f"{name}.zip"
        if zp.exists() and not (raw / name / f"{name}.json").exists():
            safe_unzip(zp, raw / name)


def flatten_applicant(aid, rec):
    ib = rec.get("infos_basicas", {}) or rec.get("informacoes_basicas", {}) or {}
    prof = rec.get("informacoes_profissionais", {}) or {}
    edu  = rec.get("formacao_e_idiomas", {}) or {}
    stacks = []
    for k in ["linguagens","frameworks","habilidades","competencias","tecnologias"]:
        stacks += [str(x) for x in (prof.get(k) or []) if x]
    return {
        "applicant_id": aid,
        "nome": ib.get("nome"),
        "senioridade": ib.get("senioridade") or ib.get("nivel"),
        "area_atuacao": ib.get("area_de_atuacao") or ib.get("area"),
        "pretensao_salarial": ib.get("pretensao_salarial"),
        "localidade": ib.get("localidade") or ib.get("cidade"),
        "stack": ", ".join(sorted(set(stacks))),
        "idiomas": ", ".join([f"{i.get('idioma')}({i.get('nivel')})" for i in (edu.get("idiomas") or [])]),
        "cv_text_pt": rec.get("cv_pt") or "",
        "cv_text_en": rec.get("cv_en") or "",
        "cargo_atual": (rec.get("cargo_atual") or {}).get("cargo"),
    }


def flatten_vaga(vid, rec):
    ib = rec.get("informacoes_basicas", {}) or {}
    perfil = rec.get("perfil_vaga", {}) or {}
    stacks = []
    for k in ["linguagens","frameworks","tecnologias","habilidades"]:
        stacks += [str(x) for x in (perfil.get(k) or []) if x]
    return {
        "vaga_id": vid,
        "titulo": ib.get("titulo") or rec.get("titulo"),
        "descricao": ib.get("descricao") or ib.get("descricao_vaga"),
        "local": ib.get("localidade") or ib.get("cidade"),
        "modalidade": ib.get("modalidade") or rec.get("modalidade"),
        "nivel": ib.get("nivel") or ib.get("senioridade"),
        "salario": ib.get("salario") or ib.get("faixa_salarial"),
        "stack_desejada": ", ".join(sorted(set(stacks))),
        "requisitos_texto": " ".join([str(v) for v in perfil.values() if isinstance(v, (str,int,float))]),
        "num_prospects": len(rec.get("prospects", []) or []),
    }


def build_processed_csvs(base: Path):
    """Lê JSONs em data/raw e gera CSVs em data/processed."""
    raw  = base / "data" / "raw"
    proc = base / "data" / "processed"

    apps_json = raw / "applicants" / "applicants.json"
    vagas_json = raw / "vagas" / "vagas.json"
    pros_json  = raw / "prospects" / "prospects.json"

    if not (apps_json.exists() and vagas_json.exists() and pros_json.exists()):
        print(">> JSONs não encontrados em data/raw/. Pulei normalização (coloque os zips em data/raw_zips e rode novamente).")
        return

    with open(apps_json, "r", encoding="utf-8") as f:
        apps_raw = json.load(f)
    with open(vagas_json, "r", encoding="utf-8") as f:
        vagas_raw = json.load(f)
    with open(pros_json, "r", encoding="utf-8") as f:
        pros_raw = json.load(f)

    apps_df = pd.DataFrame([flatten_applicant(k,v) for k,v in apps_raw.items()])
    vagas_df = pd.DataFrame([flatten_vaga(k,v) for k,v in vagas_raw.items()])
    pros_df  = pd.DataFrame([{"prospect_id": k, **(v if isinstance(v, dict) else {"raw": v})} for k,v in pros_raw.items()])

    proc.mkdir(parents=True, exist_ok=True)
    apps_df.to_csv(proc / "applicants.csv", index=False)
    vagas_df.to_csv(proc / "vagas.csv", index=False)
    pros_df.to_csv(proc / "prospects.csv", index=False)

    print(">> CSVs gerados em data/processed/:", apps_df.shape, vagas_df.shape, pros_df.shape)


def train_vectorizer(base: Path, max_docs_each=3000, max_features=20000):
    """Treina TF-IDF a partir de CSVs e salva em models/tfidf_vectorizer.joblib."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from joblib import dump

    proc = base / "data" / "processed"
    apps_csv = proc / "applicants.csv"
    vagas_csv = proc / "vagas.csv"
    if not (apps_csv.exists() and vagas_csv.exists()):
        print(">> CSVs não encontrados. Pulei treino do vetorizador.")
        return

    apps = pd.read_csv(apps_csv)
    vagas = pd.read_csv(vagas_csv)
    app_text = (apps["cv_text_pt"].fillna("") + " " + apps["stack"].fillna("")).astype(str).tolist()[:max_docs_each]
    job_text = (vagas["requisitos_texto"].fillna("") + " " + vagas["stack_desejada"].fillna("")).astype(str).tolist()[:max_docs_each]
    corpus = [normalize_text(t) for t in (app_text + job_text)]

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), preprocessor=None, lowercase=False)
    vec.fit(corpus)

    out = base / "models" / "tfidf_vectorizer.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(vec, out)
    print(">> Vetorizador salvo em:", out)


def main():
    if DO_CREATE_STRUCTURE:
        ensure_structure(BASE)
        write_files(BASE)
        print(">> Estrutura criada/atualizada em:", BASE)

    if DO_EXTRACT_JSONS:
        extract_raw_if_any(BASE)

    if DO_BUILD_CSVS:
        build_processed_csvs(BASE)

    if DO_TRAIN_VECTORIZER:
        train_vectorizer(BASE, MAX_DOCS_EACH, MAX_FEATURES)


if __name__ == "__main__":
    main()
