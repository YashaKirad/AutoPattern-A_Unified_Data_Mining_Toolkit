import streamlit as st
import pandas as pd

from preprocessing.pipeline import build_preprocess_pipeline, split_X_y_if_possible
from recommender.algorithm_selector import recommend_task_and_algorithms
from models.classification import run_classification
from models.clustering import run_clustering
from models.anomaly import run_anomaly
from models.association import run_association
from visualization.plots import (
    plot_numeric_distributions,
    plot_confusion_matrix,
    plot_clusters_2d,
    plot_anomaly_scores,
)
from mlflow_tracking import start_mlflow_run, log_dict_metrics_params

st.set_page_config(page_title="AutoPattern - Unified Data Mining Toolkit", layout="wide")

st.title("AutoPattern - Unified Data Mining Toolkit")
st.caption("Upload a CSV → Auto-preprocess → Run Classification / Clustering / Association / Anomaly → Visualize + Track with MLflow")

with st.sidebar:
    st.header("1) Upload Dataset")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    st.header("2) Choose Analysis")
    task = st.selectbox(
        "Analysis type",
        ["Auto (Recommend)", "Classification", "Clustering", "Association Rules", "Anomaly Detection"],
        index=0,
    )

    st.header("3) Settings")
    test_size = st.slider("Test size (classification)", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

    st.header("MLflow")
    enable_mlflow = st.checkbox("Enable MLflow tracking", value=True)
    mlflow_experiment = st.text_input("Experiment name", value="AutoPattern")

if not uploaded:
    st.info("Upload a CSV to begin.")
    st.stop()

# dataframe
df = pd.read_csv(uploaded)
st.subheader("Dataset Preview")
st.dataframe(df.head(30), use_container_width=True)

st.subheader("Quick Stats")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing cells", int(df.isna().sum().sum()))
col4.metric("Numeric cols", int(df.select_dtypes(include="number").shape[1]))

st.subheader("Numeric Distributions (quick look)")
fig = plot_numeric_distributions(df)
if fig is not None:
    st.pyplot(fig)
else:
    st.caption("No numeric columns to plot.")

rec = recommend_task_and_algorithms(df)
with st.expander("▶ View Auto Recommendations"):
    st.subheader("Dataset Summary")
    st.json(rec["summary"])

    st.subheader("Recommended Tasks")
    st.write(rec["suggested_tasks"])

    st.subheader("Recommended Algorithms")
    st.write(rec["suggested_algorithms"])

    if rec.get("suggested_target"):
        st.subheader("Suggested Target Column")
        st.code(rec["suggested_target"])


if task == "Auto (Recommend)":
    chosen_task = rec["suggested_tasks"][0] if rec["suggested_tasks"] else "Clustering"
else:
    chosen_task = task

st.divider()
st.subheader(f"Run: {chosen_task}")

run_ctx = None
if enable_mlflow:
    run_ctx = start_mlflow_run(mlflow_experiment, run_name=f"{chosen_task}")

try:
    if chosen_task == "Classification":
        st.caption("Classification needs a target (label) column.")
        default_target = rec.get("suggested_target")
        target_col = st.selectbox(
            "Select target column",
            options=list(df.columns),
            index=list(df.columns).index(default_target) if default_target in df.columns else (len(df.columns) - 1),
        )

        X, y = split_X_y_if_possible(df, target_col)
        preprocess = build_preprocess_pipeline(X)

        algo = st.selectbox("Model", ["logreg", "random_forest", "decision_tree"], index=1)
        res = run_classification(
            X, y, preprocess=preprocess, model_name=algo, test_size=float(test_size), random_state=int(random_state)
        )

        st.write("### Metrics")
        st.json(res["metrics"])

        st.write("### Confusion Matrix")
        cm_fig = plot_confusion_matrix(res["confusion_matrix"], res["class_labels"])
        st.pyplot(cm_fig)

        st.write("### Top Features (if available)")
        if res.get("feature_importances") is not None:
            st.dataframe(res["feature_importances"].head(20), use_container_width=True)
        else:
            st.caption("Feature importance not available for this model.")

        if run_ctx:
            log_dict_metrics_params(run_ctx, params=res["params"], metrics=res["metrics"])

    elif chosen_task == "Clustering":
        X = df.copy()
        preprocess = build_preprocess_pipeline(X)

        algo = st.selectbox("Clustering algorithm", ["kmeans", "dbscan"], index=0)
        res = run_clustering(X, preprocess=preprocess, model_name=algo, random_state=int(random_state))

        st.write("### Metrics")
        st.json(res["metrics"])

        st.write("### Cluster Assignments (first 30 rows)")
        preview = df.head(30).copy()
        preview["cluster"] = res["labels"][:30]
        st.dataframe(preview, use_container_width=True)

        st.write("### 2D Visualization (PCA)")
        fig2 = plot_clusters_2d(res["pca_2d"], res["labels"])
        st.pyplot(fig2)

        if run_ctx:
            log_dict_metrics_params(run_ctx, params=res["params"], metrics=res["metrics"])

    elif chosen_task == "Anomaly Detection":
        X = df.copy()
        preprocess = build_preprocess_pipeline(X)

        contamination = st.slider("Expected anomaly fraction (contamination)", 0.01, 0.2, 0.05, 0.01)
        res = run_anomaly(X, preprocess=preprocess, contamination=float(contamination), random_state=int(random_state))

        st.write("### Metrics")
        st.json(res["metrics"])

        st.write("### Top anomalies")
        out = df.copy()
        out["anomaly_score"] = res["scores"]
        out["is_anomaly"] = res["is_anomaly"]
        st.dataframe(out.sort_values("anomaly_score", ascending=False).head(30), use_container_width=True)

        st.write("### Score Plot")
        figA = plot_anomaly_scores(res["scores"], res["is_anomaly"])
        st.pyplot(figA)

        if run_ctx:
            log_dict_metrics_params(run_ctx, params=res["params"], metrics=res["metrics"])

    elif chosen_task == "Association Rules":
        st.caption("Association mining typically expects transactional/basket-style data.")
        min_support = st.slider("min_support", 0.01, 0.5, 0.05, 0.01)
        min_conf = st.slider("min_confidence", 0.01, 0.95, 0.3, 0.05)

        res = run_association(df, min_support=float(min_support), min_confidence=float(min_conf))

        st.write("### Frequent Itemsets (top)")
        st.dataframe(res["itemsets"].head(50), use_container_width=True)

        

        if run_ctx:
            log_dict_metrics_params(run_ctx, params=res["params"], metrics=res["metrics"])

    else:
        st.warning("Unknown task selection.")

finally:
    if run_ctx:
        run_ctx.end()
        st.success("MLflow run finished (logged).")

