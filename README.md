# Predictive Patent Application Dashboard

This project analyzes USPTO patent office action data to estimate
(1) patent allowance likelihood and
(2) review outcomes prior to submission.

The focus is not just prediction accuracy, but **evaluation across legal, citation,
and examiner-related feature groups**, with an emphasis on identifying
systematic error patterns and edge cases.

The project includes a Streamlit dashboard for exploration and a
reproducible model-training notebook.

---

## Data

The full dataset is derived from the **USPTO Office Action Research Dataset**
(millions of records) and is not included in this repository.

This repo contains:
- Precomputed EDA outputs (`data/`) derived from the full dataset
- A public training notebook demonstrating preprocessing, feature engineering,
  and model evaluation on a large sampled subset

**Source:**  
https://www.uspto.gov/ip-policy/economic-research/research-datasets/office-action-research-dataset-patents

---

## Repository Structure
```
├── app.py
├── README.md
├── requirements.txt
├── setup.sh
│
├── pages/
│   ├── about_proj.py
│   ├── eda.py
│   └── glossary.py
│
├── utils/
│   ├── header.py
│   └── utils.py
│
├── data/
│   ├── eda_avg_rejections.csv
│   ├── eda_cite_counts.csv
│   ├── eda_prior_trend.csv
│   ├── eda_rejection_counts.csv
│   └── eda_tc_allowance.csv
│
└── notebooks/
    └── patent_notebook.ipynb
```
---

## Modeling Overview

The training pipeline (see `notebooks/patent_notebook.ipynb`) includes:
- Cleaning and deduplication of office action records
- Feature engineering on legal rejections, citation patterns, and examiner groupings
- Handling severe class imbalance using SMOTE
- Training and evaluating an XGBoost classifier
- Slice-based evaluation across feature subsets (e.g., tech center, prior art severity)

Evaluation emphasizes:
- Precision / recall tradeoffs
- Confusion matrices
- Performance variation across examiner and legal feature groups

---

## Running the Dashboard Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Note on scope and data availability:
The dashboard uses precomputed summary files located in the data directory.
The full raw dataset and production model artifacts are intentionally omitted.
This repository is intended to demonstrate:
- Structuring and evaluating large-scale legal datasets
- Feature engineering and model evaluation under real-world ambiguity
- Clear separation between data ingestion, modeling, and presentation layers
