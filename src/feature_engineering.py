
import pandas as pd
import numpy as np

def _map_sen(v):
    v = str(v).lower()
    if "jun" in v or "jr" in v: return 1
    if "pleno" in v or "mid" in v or "middle" in v: return 2
    if "sen" in v: return 3
    return np.nan

def _has_english(v): 
    s = str(v).lower()
    return ("ingl" in s) or ("english" in s)

def _overlap(a: str, b: str) -> float:
    if not a or not b: return 0.0
    A, B = set(str(a).lower().split()), set(str(b).lower().split())
    if not A or not B: return 0.0
    return len(A & B)/len(A | B)

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    # overlap de skills (tenta achar colunas prováveis)
    job_comp = next((c for c in X.columns if "competencia" in c or "competencias" in c or "competências" in c), None)
    app_skill = next((c for c in X.columns if "conhecimento" in c or "skill" in c), None)
    X["feat_skill_overlap"] = X.apply(
        lambda r: _overlap(r.get(job_comp, ""), r.get(app_skill, "")), axis=1
    )

    # senioridade e gap
    if "job__nivel_profissional" in X.columns: X["sen_job"] = X["job__nivel_profissional"].apply(_map_sen)
    if "nivel_profissional" in X.columns:      X["sen_app"] = X["nivel_profissional"].apply(_map_sen)
    X["feat_senioridade"] = X.get("sen_job", np.nan)
    X["feat_senioridade_gap"] = X.get("sen_app", np.nan) - X.get("sen_job", np.nan)

    # idioma
    if "job__nivel_ingles" in X.columns: X["feat_req_ingles"] = X["job__nivel_ingles"].apply(_has_english)
    if "nivel_ingles" in X.columns:      X["feat_app_ingles"] = X["nivel_ingles"].apply(_has_english)
    if "feat_req_ingles" in X.columns and "feat_app_ingles" in X.columns:
        X["feat_ingles_match"] = (X["feat_req_ingles"] & X["feat_app_ingles"]).astype(int)
    return X
