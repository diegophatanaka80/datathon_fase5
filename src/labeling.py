
import pandas as pd
import numpy as np
import re

def _pick(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

def label_match(df: pd.DataFrame) -> pd.Series:
    col = _pick(df, ["prospect__prospect_status_norm", "prospect_status_norm"])
    if col is None:
        return pd.Series([np.nan] * len(df))
    patt = re.compile(r"(contrat|admit|oferta aceita|aprovad)", re.IGNORECASE)
    return df[col].fillna("").apply(lambda s: int(bool(patt.search(str(s)))))

def label_engagement(df: pd.DataFrame) -> pd.Series:
    col = _pick(df, ["prospect__prospect_comment_len", "prospect_comment_len"])
    if col is None:
        return pd.Series([np.nan] * len(df))
    thr = df[col].median()
    return (df[col] >= thr).astype(int)
