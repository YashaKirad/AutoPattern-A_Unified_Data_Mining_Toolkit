import pandas as pd
from utils import is_probably_label

# this function basically analyzes a dataset and recommends machine learning tasks and algorithms based on the dataste
def recommend_task_and_algorithms(df: pd.DataFrame):
    n_rows, n_cols = df.shape
    numeric = df.select_dtypes(include="number").columns.tolist()
    cat = [c for c in df.columns if c not in numeric]

    # if label exists
    suggested_target = None
    for c in df.columns[::-1]: #we try last column first because most datasets have last column as target
        if is_probably_label(df[c]):
            suggested_target = c
            break

    suggested_tasks = []
    suggested_algos = []

    if suggested_target is not None:
        suggested_tasks.append("Classification")
        suggested_algos += ["random_forest", "logreg", "decision_tree"]

    # to check for clustering if there are muultiple features
    if n_cols >= 2:
        suggested_tasks.append("Clustering")
        suggested_algos += ["kmeans", "dbscan"]

    # checks for association rules if many categorical columns or few unique per column
    if len(cat) >= 2:
        suggested_tasks.append("Association Rules")
        suggested_algos += ["apriori"]

    # checks for anomaly detection for numeric features
    if len(numeric) >= 1:
        suggested_tasks.append("Anomaly Detection")
        suggested_algos += ["isolation_forest"]

    if not suggested_tasks:
        suggested_tasks = ["Clustering"]
        suggested_algos = ["kmeans"]

    summary = {
        "rows": n_rows,
        "cols": n_cols,
        "numeric_cols": len(numeric),
        "categorical_cols": len(cat),
        "suggested_target": suggested_target,
    }

    return {
        "summary": summary,
        "suggested_tasks": suggested_tasks,
        "suggested_algorithms": sorted(set(suggested_algos)),
        "suggested_target": suggested_target,
    }
