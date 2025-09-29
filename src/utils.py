
from __future__ import annotations
import re, unicodedata
import pandas as pd

def normalize_text(text: str) -> str:
    """Normaliza texto: minúsculas, remove acentos, colapsa espaços."""
    if text is None:
        return ""
    t = str(text)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def is_nonempty(x) -> bool:
    if isinstance(x, str):
        t = x.strip().lower()
        return bool(t) and t not in {"nan", "none", "null"}
    return pd.notna(x)
