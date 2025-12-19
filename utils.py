import pandas as pd
import numpy as np

def infer_column_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, cat_cols

def safe_copy_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)

def is_probably_label(col: pd.Series) -> bool:
    n = len(col.dropna())
    if n == 0:
        return False
    u = col.dropna().nunique()
    return (u <= 20) or (u / max(n, 1) < 0.05)
