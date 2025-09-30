
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import joblib

from src.data_io import load_consolidated
from src.preprocessing import basic_preprocessing
from src.feature_engineering import make_features
from src.labeling import label_match, label_engagement

def make_ohe_dense():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

def train_and_save(
    in_path="data/processed/decision_consolidated.parquet",
    model_path="models/recommender.pkl",
    meta_path="models/recommender_meta.json"
):
    df = load_consolidated(in_path)
    df = basic_preprocessing(df)
    df = make_features(df)

    df["target_match"] = label_match(df)
    df["target_engagement"] = label_engagement(df)

    num_cols = [c for c in ["prospect_comment_len","feat_skill_overlap","feat_senioridade","feat_senioridade_gap","feat_ingles_match"] if c in df.columns]
    cat_cols = [c for c in ["job__nivel_profissional","app__area"] if c in df.columns]

    data = df.dropna(subset=["target_match"]).copy()
    if "pair_id" in data.columns:
        data = data.drop_duplicates(subset=["pair_id"])
    y = data["target_match"].astype(int)
    X = data[num_cols + cat_cols].copy()

    ohe = make_ohe_dense()                                

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", ohe)
            ]), cat_cols),
        ],
        remainder="drop"
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"))
    ])

    if y.nunique() >= 2 and y.value_counts().min() >= 2:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe.fit(Xtr, ytr)

    proba = pipe.predict_proba(Xte)[:,1]
    pred05 = pipe.predict(Xte)

    precisions, recalls, thresholds = precision_recall_curve(yte, proba)
    f1_scores = 2*precisions*recalls/(precisions+recalls+1e-9)
    best_idx = int(np.argmax(f1_scores))
    best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    pred_best = (proba >= best_thr).astype(int)

    metrics = {
        "AUC": float(np.round(roc_auc_score(yte, proba), 4)),
        "F1@0.5": float(np.round(f1_score(yte, pred05), 4)),
        "BestThreshold": best_thr,
        "F1@Best": float(np.round(f1_score(yte, pred_best), 4)),
        "yte_pos_ratio": float(np.round(yte.mean(), 4)),
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    with open(meta_path, "w") as f:
        json.dump({"best_threshold": best_thr, "metrics": metrics}, f, indent=2)

    return metrics
