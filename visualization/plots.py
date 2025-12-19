import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_numeric_distributions(df: pd.DataFrame):
    num = df.select_dtypes(include="number")
    if num.shape[1] == 0:
        return None

    cols = num.columns[:6]  
    fig = plt.figure(figsize=(10, 6))
    for i, c in enumerate(cols, start=1):
        ax = fig.add_subplot(2, 3, i)
        ax.hist(num[c].dropna().values, bins=30)
        ax.set_title(c)
    fig.tight_layout()
    return fig

def plot_confusion_matrix(cm, labels):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    return fig

def plot_clusters_2d(pca_2d, labels):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=labels)
    ax.set_title("Clusters (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    return fig

def plot_anomaly_scores(scores, is_anomaly):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    x = np.arange(len(scores))
    ax.scatter(x, scores, c=is_anomaly.astype(int))
    ax.set_title("Anomaly Scores (higher = more anomalous)")
    ax.set_xlabel("Row index")
    ax.set_ylabel("Score")
    fig.tight_layout()
    return fig
