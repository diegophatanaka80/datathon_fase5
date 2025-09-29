
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import normalize_text

def fit_vectorizer(texts, max_features=100000):
    vect = TfidfVectorizer(max_features=max_features, min_df=2, ngram_range=(1,2))
    vect.fit([normalize_text(t) for t in texts])
    return vect

def transform_texts(vect, texts):
    return vect.transform([normalize_text(t) for t in texts])

def cosine_batch(vect, cand_texts, job_text, batch=10000):
    B = transform_texts(vect, [job_text])  # (1, V)
    n = len(cand_texts); out = np.zeros(n, dtype=float)
    B_norm = float(np.sqrt(B.multiply(B).sum()))
    if B_norm == 0: 
        return out
    for s in range(0, n, batch):
        e = min(n, s+batch)
        A = transform_texts(vect, cand_texts[s:e])
        A_norm = np.sqrt(A.multiply(A).sum(axis=1)).A.ravel()
        dots = (A @ B.T).A.ravel()
        out[s:e] = dots/(A_norm * B_norm + 1e-12)
    return out
