from __future__ import annotations
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from detector import METHODS, detect, profile_dataset

st.set_page_config(page_title="AnomalyLens", page_icon="🔎", layout="wide")
st.title("🔎 AnomalyLens")
st.caption("Explore unusual records in tabular data — locally, with explainable controls and downloadable results.")
with st.sidebar:
    st.header("Detection settings")
    method = st.selectbox("Algorithm", METHODS, help="Each method scores unusual records differently.")
    contamination = st.slider("Expected anomaly fraction", min_value=0.01, max_value=0.25, value=0.05, step=0.01,
                              help="Controls the approximate fraction flagged. It is not a known ground-truth rate.")
    st.caption("Your data is processed by this local app; no API key is required.")
source = st.radio("Dataset source", ["Built-in demo", "Upload CSV"], horizontal=True)
if source == "Built-in demo":
    X, _ = make_blobs(n_samples=700, centers=[(-1, 0), (3, 3)], cluster_std=[0.75, 1.0], random_state=42)
    rng = np.random.default_rng(42)
    outliers = rng.uniform(-7, 9, size=(35, 2))
    df = pd.DataFrame(np.vstack([X, outliers]), columns=["feature_1", "feature_2"])
    st.info("Demo data contains two clusters plus injected unusual points.")
else:
    upload = st.file_uploader("Upload a CSV file", type=["csv"])
    if upload is None:
        st.stop()
    try:
        df = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
if df.empty:
    st.error("The dataset has no rows."); st.stop()
try:
    profile = profile_dataset(df)
except ValueError as exc:
    st.error(str(exc)); st.stop()
cols = st.columns(4)
cols[0].metric("Rows", f"{profile['rows']:,}")
cols[1].metric("Columns", f"{profile['columns']:,}")
cols[2].metric("Missing cells", f"{profile['missing_cells']:,}")
cols[3].metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
with st.expander("Preview and data quality"):
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Numeric features used:", ", ".join(profile["features_used"]))
    if profile["constant_numeric_features"]:
        st.warning("Constant numeric features: " + ", ".join(profile["constant_numeric_features"]))
if st.button("Run anomaly detection", type="primary"):
    try:
        st.session_state["anomaly_results"] = detect(df, method, contamination)
        st.session_state["anomaly_method"] = method
    except Exception as exc:
        st.error(str(exc))
if "anomaly_results" in st.session_state:
    result = st.session_state["anomaly_results"]
    n_anom = int((result["anomaly_label"] == "Anomaly").sum())
    a,b,c = st.columns(3)
    a.metric("Flagged anomalies", f"{n_anom:,}")
    b.metric("Flag rate", f"{n_anom/len(result):.1%}")
    c.metric("Algorithm", st.session_state.get("anomaly_method", method))
    st.subheader("Highest-scoring records")
    st.caption("Scores rank records by unusualness for the selected algorithm; they are not probabilities or proof of an error.")
    show = st.checkbox("Show only anomalies", value=False)
    display = result[result.anomaly_label == "Anomaly"] if show else result
    st.dataframe(display.head(100), use_container_width=True)
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if df[c].nunique(dropna=True) > 1]
    if len(numeric_cols) >= 2:
        st.subheader("Feature-space view")
        xcol = st.selectbox("X axis", numeric_cols, index=0)
        ycol = st.selectbox("Y axis", numeric_cols, index=1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        normal = result[result.anomaly_label == "Normal"]
        anomalies = result[result.anomaly_label == "Anomaly"]
        ax.scatter(normal[xcol], normal[ycol], s=22, alpha=.65, label="Normal")
        ax.scatter(anomalies[xcol], anomalies[ycol], s=34, alpha=.9, marker="x", label="Anomaly")
        ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.legend(); ax.grid(alpha=.2)
        st.pyplot(fig); plt.close(fig)
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("Download scored CSV", csv, file_name="anomalylens_results.csv", mime="text/csv")
    st.download_button("Download run summary (JSON)", pd.Series({"method": st.session_state.get("anomaly_method", method), "rows": len(result), "anomalies": n_anom, "features": profile["features_used"]}).to_json(indent=2).encode(), file_name="anomalylens_summary.json", mime="application/json")
