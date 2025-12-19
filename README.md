
# AutoPattern: A Unified Data Mining Toolkit

AutoPattern is a unified and interactive data mining toolkit designed to streamline end-to-end data analysis workflows. The system supports multiple data mining paradigms, including classification, clustering, anomaly detection, and association rule mining, within a single platform. AutoPattern combines automated preprocessing, explainable recommendation logic, and interactive visualization to reduce manual effort while preserving transparency and user control.

---

## Project Overview

Data mining workflows often require extensive manual configuration, repeated preprocessing, and trial-and-error algorithm selection. AutoPattern addresses these challenges by providing a single framework that automates routine steps, guides users toward appropriate analytical choices, and enables rapid experimentation.

The system is designed for educational use, exploratory data analysis, and applied data mining scenarios where interpretability and reproducibility are essential.

---

## Key Features

* Upload and analyze structured CSV datasets
* Automated preprocessing pipeline

  * Missing value handling
  * Feature scaling
  * Categorical encoding
* Explainable auto-recommendation system for task, target, and algorithm selection
* Support for supervised and unsupervised data mining tasks
* Interactive visualizations for result interpretation
* Experiment tracking and reproducibility using MLflow
* User-controlled parameter tuning for deeper analysis

---

## Supported Data Mining Tasks

### Classification

AutoPattern supports multiple classification algorithms:

* Logistic Regression
* Decision Tree
* Random Forest

These algorithms provide different trade-offs between interpretability and predictive performance. Logistic Regression offers a simple and interpretable baseline, Decision Trees capture non-linear feature interactions, and Random Forest provides robustness and improved generalization through ensemble learning. Users are encouraged to compare these models to understand their behavior on different datasets.

---

### Clustering

For unsupervised clustering, AutoPattern provides:

* K-Means
* DBSCAN

K-Means is effective for datasets with well-separated, spherical clusters and requires the number of clusters to be specified in advance. DBSCAN is a density-based algorithm that can identify arbitrarily shaped clusters and detect noise without requiring the number of clusters. Providing both options allows users to explore different clustering assumptions.

---

### Anomaly Detection

AutoPattern implements anomaly detection using Isolation Forest. A key user-tunable parameter is the contamination rate, which represents the expected proportion of anomalous observations in the dataset. Adjusting this parameter allows users to control the sensitivity of anomaly detection based on domain requirements.

---

### Association Rule Mining

Association rule mining is performed using the Apriori algorithm. Users can tune:

* Minimum support
* Minimum confidence

These parameters control the frequency, reliability, and strength of discovered rules. Adjusting them enables users to explore how dataset sparsity and co-occurrence patterns affect rule generation.

---

## System Architecture

AutoPattern follows a modular and layered architecture:

1. Data Ingestion Layer:
   Loads structured datasets and extracts schema information.

2. Preprocessing Layer:
   Automatically handles missing values, feature scaling, and categorical encoding using a unified scikit-learn pipeline.

3. Auto-Recommendation Layer:
   Analyzes dataset meta-features and suggests suitable tasks, target variables, and algorithms using rule-based heuristics.

4. Modeling Layer:
   Executes selected data mining algorithms for classification, clustering, anomaly detection, or association rule mining.

5. Visualization and Experiment Tracking Layer:
   Presents results using interactive plots and logs experiments using MLflow for reproducibility.

---

## Example Datasets

The following datasets were used during development and evaluation:

| Task                    | Dataset                                           |
| ----------------------- | ------------------------------------------------- |
| Classification          | Titanic Dataset                                   |
| Clustering              | Iris Dataset, Mall Customers Dataset              |
| Anomaly Detection       | Thyroid Disease Dataset, Credit Card Transactions |
| Association Rule Mining | Grocery Basket Dataset                            |

---

## Installation and Usage

### Clone the repository

```bash
git clone https://github.com/your-username/AutoPattern.git
cd AutoPattern
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run the application

```
python -m streamlit run app.py
```

---

## Project Structure

```
AutoPattern/
├── app.py
├── requirements.txt
├── preprocessing/
│   └── pipeline.py
├── recommender/
│   └── algorithm_selector.py
├── models/
│   ├── classification.py
│   ├── clustering.py
│   ├── anomaly.py
│   └── association.py
├── visualization/
│   └── plots.py
└── mlflow_tracking.py
```

---

## Auto-Recommendation Logic

AutoPattern employs rule-based meta-learning to generate recommendations. Dataset characteristics such as feature types, cardinality, and semantic cues in column names are analyzed to identify suitable tasks and potential target variables. Algorithms are recommended based on data composition and robustness considerations. All recommendations are advisory and can be overridden by the user.

---

## Reproducibility and Evaluation

AutoPattern integrates MLflow for experiment tracking. Model parameters, evaluation metrics, and run configurations are logged automatically, allowing users to compare results across different analyses and ensure reproducibility.

---

## Limitations

* Recommendation logic is heuristic-based and may not generalize to all datasets.
* Hyperparameter optimization is not currently supported.
* Association rule mining may be computationally expensive for large datasets.
* The system is limited to structured tabular data.

---

## Future Work

Potential extensions include learned meta-models for recommendation, automated hyperparameter optimization, support for additional algorithms, and extensions to time-series or unstructured data.

---

## Authors
* Yashshree Kirad
* Shaivi Bansal
* Kushal Hawaldar


