
from pathlib import Path
import pandas as pd

def load_consolidated(path: str = "data/processed/decision_consolidated.parquet") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Base consolidada não encontrada em {p.resolve()}.")
    return pd.read_parquet(p)

def save_consolidated(df: pd.DataFrame, path: str = "data/processed/decision_consolidated.parquet"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
