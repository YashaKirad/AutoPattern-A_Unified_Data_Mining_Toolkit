# this file is for anomaly detection task using Isolation Forest algorithm

from typing import Dict, Any
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

# this function runs anomaly detection using the Isolation Forest algorithm
# it fits an Isolation Forest model on the given dataset to identify anomalous data points. (unsupervised setting)
def run_anomaly(
    X: pd.DataFrame,
    preprocess,
    contamination: float = 0.05,   #Expected fraction of anomalies in the dataset
    random_state: int = 42,
) -> Dict[str, Any]:

    model = IsolationForest(
        n_estimators=400,
        contamination=contamination,
        random_state=random_state,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    Z = preprocess.fit_transform(X)
    model.fit(Z)

    # decision_function higher = more normal
    # We inverted it so higher = more anomalous
    normality = model.decision_function(Z)
    scores = (-normality)

    preds = model.predict(Z)  # -1 is anomaly, 1 is normal
    is_anomaly = (preds == -1)

    return {
        "params": {
            "task": "anomaly",
            "model_name": "isolation_forest",
            "contamination": contamination,
            "random_state": random_state,
        },
        "metrics": {
            "anomalies_found": float(is_anomaly.sum()),
            "anomaly_fraction": float(is_anomaly.mean()),
        },
        "scores": scores,
        "is_anomaly": is_anomaly,
    }
