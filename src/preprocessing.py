
from __future__ import annotations
from pathlib import Path
import json, zipfile, re
import pandas as pd
from .utils import normalize_text

BASE = Path(__file__).resolve().parents[1]

def _clean(x):
    return str(x).strip() if pd.notna(x) and str(x).strip().lower()!="nan" else ""

def _norm_id(x): 
    import re
    return re.sub(r"[^\dA-Za-z_-]", "", str(x))

def _load_json(folder: str, fname: str):
    raw = BASE/"data"/"raw"/folder/fname
    if raw.exists():
        return json.load(open(raw, "r", encoding="utf-8"))
    z = BASE/"data"/"raw_zips"/f"{folder}.zip"
    if z.exists():
        with zipfile.ZipFile(z, "r") as zf:
            inside = f"{folder}/{fname}"
            if inside in zf.namelist():
                with zf.open(inside) as fh: 
                    import io; return json.load(fh)
            if fname in zf.namelist():
                with zf.open(fname) as fh:
                    import io; return json.load(fh)
    return {}

def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = BASE/"data"/"processed"
    apps = pd.read_csv(data/"applicants.csv")
    vagas = pd.read_csv(data/"vagas.csv")
    prospects = pd.read_csv(data/"prospects.csv")
    return apps, vagas, prospects

def build_job_text(row: pd.Series) -> str:
    fields = ["requisitos_texto","descricao","descricao_detalhada",
              "responsabilidades","atividades","stack_desejada","stack",
              "titulo_vaga","titulo"]
    parts = [_clean(row.get(f)) for f in fields if _clean(row.get(f))]
    return " ".join(parts)

def add_job_text(vagas: pd.DataFrame) -> pd.DataFrame:
    out = vagas.copy()
    out["job_text"] = out.apply(build_job_text, axis=1)
    return out

def add_app_text(apps: pd.DataFrame) -> pd.DataFrame:
    out = apps.copy()
    out["app_text"] = (out["cv_text_pt"].fillna("").astype(str) + " " +
                       out["stack"].fillna("").astype(str))
    return out

def load_all_with_text() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    apps, vagas, prospects = load_processed()
    return add_app_text(apps), add_job_text(vagas), prospects
