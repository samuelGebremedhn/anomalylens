# AnomalyLens — Local Anomaly Detection Studio

AnomalyLens is a lightweight, local-first machine-learning app for finding unusual records in tabular datasets. Upload a CSV, inspect basic data quality, compare unsupervised algorithms, explore a feature-space plot, and export scored results.

## Features
- **Three algorithms:** Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM.
- **Dataset overview:** row/column counts, missing cells, duplicates, and numeric feature checks.
- **Practical preprocessing:** median imputation, scaling, and safe handling of infinite/missing values.
- **Exploration:** anomaly ranking, filter-to-anomalies view, and selectable 2D feature plot.
- **Exports:** scored CSV and run-summary JSON.
- **Local-first:** no API key or hosted service required.
- **Tests:** reusable detection logic is covered by automated tests.

> Anomaly scores are algorithm-specific ranking values, not probabilities. A flagged row is a prompt for investigation, not proof that the data is wrong. The algorithms use numeric columns; text/category columns are retained in the exported rows but not used as model features.

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal. Choose the built-in demo or upload a CSV, set the algorithm and expected anomaly fraction, then click **Run anomaly detection**.

## Run tests

```bash
pytest -q
```

## Repository layout

```text
anomalylens/
├── app.py             # Streamlit UI
├── detector.py        # reusable detection + profiling functions
├── tests/             # automated tests
├── data/              # optional local sample data
├── requirements.txt
└── README.md
```

## Limitations
- Numeric features only; categorical encoding is not automatic.
- Results depend on algorithm settings and dataset shape.
- The contamination setting influences how many points are flagged.
- Large datasets may require additional memory and runtime.

## Suggested GitHub metadata
- **Repository:** `anomalylens`
- **Description:** `A local-first anomaly detection studio for CSV data, with Isolation Forest, LOF, One-Class SVM, visual exploration, and downloadable results.`
- **Topics:** `machine-learning`, `anomaly-detection`, `scikit-learn`, `streamlit`, `data-science`, `unsupervised-learning`, `python`

## License
MIT. See [LICENSE](LICENSE).
