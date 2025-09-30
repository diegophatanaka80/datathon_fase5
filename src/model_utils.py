
import json
import joblib
import pandas as pd
from pathlib import Path

def load_model(model_path="models/recommender.pkl", meta_path="models/recommender_meta.json"):
    pipe = joblib.load(model_path)
    best_thr = 0.5
    p = Path(meta_path)
    if p.exists():
        best_thr = json.load(open(p))["best_threshold"]
    return pipe, best_thr

def predict_with_threshold(pipe, X: pd.DataFrame, thr: float):
    proba = pipe.predict_proba(X)[:,1]
    pred = (proba >= thr).astype(int)
    return proba, pred
