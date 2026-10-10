import numpy as np
import pandas as pd
import pytest
from detector import detect, profile_dataset

def sample():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, size=(80, 2))
    x[-3:] = [[8, 8], [-8, 7], [9, -9]]
    return pd.DataFrame(x, columns=["temperature", "pressure"])

def test_detect_returns_scores_and_labels():
    out = detect(sample(), contamination=0.05)
    assert len(out) == 80
    assert {"anomaly_label", "anomaly_score", "anomaly_rank"}.issubset(out.columns)
    assert out.anomaly_score.is_monotonic_decreasing
    assert set(out.anomaly_label) <= {"Normal", "Anomaly"}

def test_all_methods_work():
    for method in ("Isolation Forest", "Local Outlier Factor", "One-Class SVM"):
        assert len(detect(sample(), method=method)) == 80

def test_rejects_non_numeric_data():
    with pytest.raises(ValueError, match="No numeric"):
        detect(pd.DataFrame({"name": ["a", "b", "c", "d", "e"]}))

def test_rejects_too_few_rows():
    with pytest.raises(ValueError, match="At least 5"):
        detect(pd.DataFrame({"x": [1, 2, 3, 4]}))

def test_profile_counts_missing_and_duplicates():
    df = pd.DataFrame({"x": [1, 1, np.nan], "y": [2, 2, 3]})
    p = profile_dataset(df)
    assert p["rows"] == 3 and p["missing_cells"] == 1 and p["duplicate_rows"] == 1
