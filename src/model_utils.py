
from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parents[1]

def save_joblib(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_joblib(path: Path):
    return joblib.load(path)

def save_metrics(d: dict, path: Path|None=None):
    if path is None: 
        path = BASE/"models"/"metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
