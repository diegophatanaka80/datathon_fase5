
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from .preprocessing import load_all_with_text
from .feature_engineering import cosine_batch
from .model_utils import load_joblib

BASE = Path(__file__).resolve().parents[1]

def evaluate_holdout(sample_size: int = 2000):
    apps, vagas, _ = load_all_with_text()
    vect = load_joblib(BASE/"models"/"tfidf_vectorizer.joblib")
    # amostra aleatória: pega 1 vaga e N candidatos
    vaga = vagas.sample(1, random_state=42).iloc[0]
    cand = apps.sample(min(sample_size, len(apps)), random_state=123)
    scores = cosine_batch(vect, cand["app_text"].tolist(), vaga["job_text"])
    out = cand.assign(match_score=scores).sort_values("match_score", ascending=False).head(20)
    return out[["applicant_id","nome","match_score"]]

if __name__ == "__main__":
    print(evaluate_holdout(1000).head(10))
