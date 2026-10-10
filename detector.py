"""Reusable anomaly detection helpers for numeric tabular data."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

METHODS = ("Isolation Forest", "Local Outlier Factor", "One-Class SVM")

def prepare_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Provide a non-empty pandas DataFrame.")
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] == 0:
        raise ValueError("No numeric columns found. Include at least one numeric feature.")
    # Ignore columns that have no usable values; imputation cannot rescue them.
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        raise ValueError("Numeric columns contain no usable values.")
    # Infinite values are treated as missing.
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        raise ValueError("Numeric columns contain no finite values.")
    return numeric, numeric.columns.tolist()

def detect(df: pd.DataFrame, method: str = "Isolation Forest", contamination: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    """Return a copy sorted by anomaly score (largest first).

    Scores are model-specific ranking scores, not calibrated probabilities.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Choose one of: {', '.join(METHODS)}")
    if not 0.001 <= float(contamination) <= 0.5:
        raise ValueError("contamination must be between 0.001 and 0.5.")
    numeric, features = prepare_numeric(df)
    if len(numeric) < 5:
        raise ValueError("At least 5 rows are required for anomaly detection.")
    if len(features) == 0:
        raise ValueError("At least one usable numeric feature is required.")
    X = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]).fit_transform(numeric)
    if method == "Isolation Forest":
        model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
        labels = model.fit_predict(X)
        scores = -model.score_samples(X)
    elif method == "Local Outlier Factor":
        neighbors = min(20, max(2, len(X) - 1))
        model = LocalOutlierFactor(n_neighbors=neighbors, contamination=contamination)
        labels = model.fit_predict(X)
        scores = -model.negative_outlier_factor_
    else:
        model = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
        labels = model.fit_predict(X)
        scores = -model.score_samples(X)
    result = df.copy()
    result["anomaly_label"] = np.where(labels == -1, "Anomaly", "Normal")
    result["anomaly_score"] = np.asarray(scores, dtype=float)
    result["anomaly_rank"] = pd.Series(scores, index=result.index).rank(method="first", ascending=False).astype(int)
    result.attrs["features_used"] = features
    result.attrs["method"] = method
    result.attrs["score_note"] = "Scores are model-specific ranking values, not probabilities."
    return result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

def profile_dataset(df: pd.DataFrame) -> dict:
    """Create concise data-quality statistics for the UI."""
    numeric, features = prepare_numeric(df)
    return {
        "rows": int(len(df)), "columns": int(len(df)), "numeric_features": int(len(features)),
        "missing_cells": int(df.isna().sum().sum()), "duplicate_rows": int(df.duplicated().sum()),
        "constant_numeric_features": [c for c in features if numeric[c].nunique(dropna=True) <= 1],
        "features_used": features,
    }
