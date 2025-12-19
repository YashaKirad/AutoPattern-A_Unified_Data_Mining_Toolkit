# this file is for clustering task with two algorithms: K-Means and DBSCAN

from typing import Dict, Any
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def _make_model(model_name: str, random_state: int):
    if model_name == "kmeans":
        return KMeans(n_clusters=3, random_state=random_state, n_init="auto")
    if model_name == "dbscan":
        return DBSCAN(eps=0.6, min_samples=5)
    raise ValueError(f"Unknown model_name: {model_name}")

# this function runs unsupervised clustering on the input data using either K-Means or DBSCAN (with preprocessing and evaluation)
def run_clustering(
    X: pd.DataFrame,
    preprocess,
    model_name: str = "kmeans",
    random_state: int = 42,
) -> Dict[str, Any]:

    model = _make_model(model_name, random_state=random_state)
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    Z = preprocess.fit_transform(X)
    labels = model.fit_predict(Z)

    # for silhouette we need more than 2 clusters and no all-noise case
    sil = None
    try:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            sil = silhouette_score(Z, labels)
    except Exception:
        sil = None

    pca = PCA(n_components=2, random_state=random_state)
    p2 = pca.fit_transform(Z.toarray() if hasattr(Z, "toarray") else Z)

    return {
        "params": {
            "task": "clustering",
            "model_name": model_name,
            "random_state": random_state,
        },
        "metrics": {
            "silhouette": float(sil) if sil is not None else float("nan"),
            "n_clusters_found": float(len(set(labels)) - (1 if -1 in labels else 0)),
            "noise_points": float((labels == -1).sum()) if -1 in set(labels) else 0.0,
        },
        "labels": labels,
        "pca_2d": p2,
    }
