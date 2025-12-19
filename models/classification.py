# thiis file is for classification tasks using 3 algorithms
# LogisticRegression
# RandomForestClassifier
# DecisionTreeClassifier

from typing import Dict, Any
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# instantiates a classification model
def _make_model(model_name: str, random_state: int):
    if model_name == "logreg":
        return LogisticRegression(max_iter=2000)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=random_state)
    if model_name == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state)
    raise ValueError(f"Unknown model_name: {model_name}")

# this function trains and evaluates a supervised classification model using the preprocessing + modeling pipeline
def run_classification(
    X: pd.DataFrame,
    y: pd.Series,
    preprocess,
    model_name: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:

    # we drop rows with missing label
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() <= 50 else None
    )

    model = _make_model(model_name, random_state=random_state)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    labels = sorted(list(pd.Series(y_test).unique()))
    cm = confusion_matrix(y_test, preds, labels=labels)

    # feature importance done only for tree/forest
    feature_importances = None
    try:
        if hasattr(model, "feature_importances_"):
            ohe = pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
            num_cols = pipe.named_steps["preprocess"].transformers_[0][2]
            cat_cols = pipe.named_steps["preprocess"].transformers_[1][2]
            cat_names = ohe.get_feature_names_out(cat_cols)
            feature_names = list(num_cols) + list(cat_names)
            importances = model.feature_importances_
            feature_importances = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
    except Exception:
        feature_importances = None

    return {
        "params": {
            "task": "classification",
            "model_name": model_name,
            "test_size": test_size,
            "random_state": random_state,
        },
        "metrics": {
            "accuracy": float(acc),
            "f1_weighted": float(f1),
            "n_classes": float(y.nunique()),
        },
        "confusion_matrix": cm,
        "class_labels": labels,
        "feature_importances": feature_importances,
    }
