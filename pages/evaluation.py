import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.header import render_header

REPORTS = Path(__file__).resolve().parents[1] / "reports"

st.set_page_config(layout="wide")
render_header()
st.title("Model Evaluation")

if not (REPORTS / "model_comparison.csv").exists():
    st.warning("No evaluation results yet. Run `python scripts/train_evaluate.py` to generate the files in `reports/`.")
    st.stop()

split = json.loads((REPORTS / "split.json").read_text())
st.markdown(
    f"""
**Task:** at the first office action, predict whether the application is eventually **granted** (1) or **abandoned** (0).
Applications still pending in the 2022 data are excluded.

**Time-based split** (by year the first action was mailed; no shuffling, no resampling):
- Train {split['train']['years'][0]}–{split['train']['years'][1]}: {split['train']['n']:,} applications ({split['train']['granted_rate']:.1%} granted)
- Validation {split['validation']['years'][0]}: {split['validation']['n']:,} ({split['validation']['granted_rate']:.1%} granted), used for hyperparameters and the threshold
- Test {split['test']['years'][0]}–{split['test']['years'][1]}: {split['test']['n']:,} ({split['test']['granted_rate']:.1%} granted), evaluated once
"""
)

st.markdown("### Baselines vs. model (test set)")
comp = pd.read_csv(REPORTS / "model_comparison.csv")
show = comp[["model", "roc_auc", "avg_precision_abandoned", "log_loss", "brier", "threshold",
             "precision_abandoned", "recall_abandoned", "precision_granted", "recall_granted", "n", "positive_rate"]]
st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c not in ("model", "n")} | {"n": "{:,}"}),
             use_container_width=True, hide_index=True)
st.caption("Average precision and precision/recall for **abandoned** treat abandonment as the event to flag. "
           "The constant baseline predicts the training grant rate for every application. "
           "Ablation rows retrain XGBoost with one feature group removed.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Precision–recall (flagging abandonment)")
    pr = pd.read_csv(REPORTS / "pr_curve_abandoned.csv")
    base = 1 - split["test"]["granted_rate"]
    fig = px.line(pr, x="recall_abandoned", y="precision_abandoned")
    fig.add_hline(y=base, line_dash="dash", annotation_text=f"abandonment rate {base:.1%}")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.markdown("### Calibration")
    cal = pd.read_csv(REPORTS / "calibration.csv")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cal["mean_predicted"], y=cal["observed_grant_rate"], mode="lines+markers", name="XGBoost"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="perfect"))
    fig.update_layout(xaxis_title="mean predicted P(grant), by decile", yaxis_title="observed grant rate")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Confusion matrix at the selected threshold")
cm = json.loads((REPORTS / "confusion_matrix.json").read_text())
st.dataframe(pd.DataFrame(
    {"Predicted granted": [cm["true_granted_pred_granted"], cm["true_abandoned_pred_granted"]],
     "Predicted abandoned": [cm["true_granted_pred_abandoned"], cm["true_abandoned_pred_abandoned"]]},
    index=["Actually granted", "Actually abandoned"]).style.format("{:,}"), use_container_width=True)
st.caption(f"Threshold {cm['threshold']:.2f}, chosen on validation data to maximize balanced accuracy.")

st.markdown("### Performance by slice (test set)")
t1, t2 = st.tabs(["Technology center", "Year of first action"])
with t1:
    st.dataframe(pd.read_csv(REPORTS / "slices_tech_center.csv").round(3), use_container_width=True, hide_index=True)
with t2:
    st.dataframe(pd.read_csv(REPORTS / "slices_year.csv").round(3), use_container_width=True, hide_index=True)
st.caption("Compare `mean_predicted` with `actual_grant_rate`: a gap means the model is over- or under-confident for that slice.")

st.markdown("### Largest errors")
st.dataframe(pd.read_csv(REPORTS / "error_examples.csv", dtype={"app_id": str}), use_container_width=True, hide_index=True)
notes = REPORTS / "error_analysis.md"
if notes.exists():
    st.markdown(notes.read_text())
