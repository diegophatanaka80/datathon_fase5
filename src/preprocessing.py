
import pandas as pd

TEXT_CANDS = [
    "prospect__prospect_status", "prospect_status",
    "job__nivel_profissional",
    "app__area"
]
COMMENT_CANDS = ["prospect__prospect_comment", "prospect_comment"]

def _norm(s):
    if s is None or (isinstance(s, float) and pd.isna(s)): 
        return ""
    return " ".join(str(s).strip().lower().split())

def _pick(df: pd.DataFrame, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def basic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    # normaliza textos se existirem
    for c in TEXT_CANDS:
        if c in X.columns:
            X[c] = X[c].apply(_norm)

    # comprimento do comentário (cria versões com e sem prefixo)
    cmt = _pick(X, COMMENT_CANDS)
    if cmt:
        if "prospect__prospect_comment_len" not in X.columns:
            X["prospect__prospect_comment_len"] = X[cmt].apply(lambda x: len(x) if isinstance(x, str) else 0)
        # alias sem prefixo (para retrocompatibilidade)
        X["prospect_comment_len"] = X["prospect__prospect_comment_len"]

    # status normalizado (cria versões com e sem prefixo)
    stcol = _pick(X, ["prospect__prospect_status", "prospect_status"])
    if stcol:
        X["prospect__prospect_status_norm"] = X[stcol].apply(_norm)
        X["prospect_status_norm"] = X["prospect__prospect_status_norm"]

    return X
