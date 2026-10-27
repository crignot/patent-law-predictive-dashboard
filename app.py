import pandas as pd
import plotly.express as px
import streamlit as st

#to run: streamlit run app.py   (model files come from scripts/train_evaluate.py; see README)

from utils.features import TECH_CENTERS, MODEL_COLUMNS
from utils.header import render_header
from utils.labels import feature_label
from utils.model import load_model, predict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#page config
st.set_page_config(page_title="Patent Allowance Predictor", layout="wide")
render_header()

with st.sidebar:
    st.title("Patent Predictor")
    st.caption("USPTO office actions (2008–2017) joined to final outcomes (PatEx 2022)")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_examples():
    path = ROOT / "data" / "example_office_actions.csv"
    return pd.read_csv(path, dtype={"app_id": str, "art_unit": str}) if path.exists() else None


booster, meta = get_model()
examples = get_examples()

st.markdown("## Allowance Prediction")
st.write(
    "Estimates the probability that an application is **eventually granted**, using only what is known on the day "
    "its **first office action** is mailed: the rejections and citations in that action, the art unit, timing, "
    "and the applicant's entity size."
)

if booster is None:
    st.warning(
        "Model files not found (`models/xgb_model.json`). The EDA and Evaluation pages still work. "
        "To enable predictions, run `python scripts/build_dataset.py` then `python scripts/train_evaluate.py` "
        "(see README)."
    )
    st.stop()

FLAG_LABELS = {
    "rejection_101": "§101 rejection (eligibility)",
    "rejection_102": "§102 rejection (novelty)",
    "rejection_103": "§103 rejection (obviousness)",
    "rejection_112": "§112 rejection (clarity / support)",
    "rejection_dp": "Double-patenting rejection",
    "objection": "Objection",
    "allowed_claims": "Examiner indicated allowable claim(s)",
    "cite102_gt1": "More than 1 reference in a §102 rejection",
    "cite103_gt3": "More than 3 references in a §103 rejection",
    "cite103_eq1": "Exactly 1 reference in a §103 rejection",
}
TC_OPTIONS = {f"{name} ({code})": code for code, name in TECH_CENTERS.items()}
SIGNATURE_OPTIONS = {
    "Examiner only": 0, "Primary examiner": 1, "Examiner + primary examiner": 2,
    "Examiner + supervisor": 3, "Examiner + primary/supervisor + director": 4,
}

#choose input source
mode = st.radio("Input", ["Load an example office action", "Enter office action fields"], horizontal=True)

if mode == "Load an example office action" and examples is not None:
    label = examples.apply(
        lambda r: f"App {r.app_id} · TC {str(r.art_unit)[:2]}00 · {int(r.first_action_year)}", axis=1)
    choice = st.selectbox("Held-out test example (first actions mailed 2015–2017)", label.tolist())
    row = examples.loc[label == choice].iloc[[0]].copy()
    actual = int(row["granted"].iloc[0])
    raw = row.drop(columns=["granted", "app_id"])
else:
    actual = None
    with st.form("office_action_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Application**")
            tc_label = st.selectbox("Technology center", list(TC_OPTIONS), index=2,
                                    help="The model uses the technology center (first two digits of the art unit).")
            art_unit = str(TC_OPTIONS[tc_label])
            entity = st.selectbox("Applicant entity size", ["UNDISCOUNTED", "SMALL", "MICRO"])
            year = st.number_input("Year first action was mailed", 2008, 2017, 2016,
                                   help="The model was trained on 2008–2013 actions; years outside 2008–2017 are not supported.")
            months = st.number_input("Months from filing to first action", 0.0, 120.0, 18.0, 0.5)
            signature = st.selectbox("Signed by", list(SIGNATURE_OPTIONS))
        with c2:
            st.markdown("**Rejections / objections in the first action**")
            flags = {k: int(st.checkbox(FLAG_LABELS[k], value=(k == "rejection_103")))
                     for k in ["rejection_101", "rejection_102", "rejection_103", "rejection_112", "rejection_dp",
                               "objection", "allowed_claims"]}
        with c3:
            st.markdown("**Prior-art citations**")
            cites = {k: int(st.checkbox(FLAG_LABELS[k])) for k in ["cite102_gt1", "cite103_gt3", "cite103_eq1"]}
            cite_max = st.number_input("Max references cited in a §103 rejection", 0, 20, 2)
        st.form_submit_button("Predict")
    raw = pd.DataFrame([{**flags, **cites, "cite103_max": cite_max, "art_unit": art_unit, "entity_size": entity,
                         "signature_type": SIGNATURE_OPTIONS[signature], "first_action_year": year,
                         "months_to_first_action": months}])

prob, contribs, X = predict(booster, raw)
prob = float(prob[0])
threshold = meta["threshold"]

#results
st.markdown("---")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Estimated probability of eventual grant", f"{prob*100:.1f}%")
    st.caption(f"Historical grant rate in training data: {meta['train_grant_rate']*100:.1f}%. "
               f"Decision threshold (chosen on 2014 validation data): {threshold:.2f}.")
    bar_color = "green" if prob >= 0.75 else ("orange" if prob >= threshold else "red")
    st.markdown(
        f"""<div style="height: 20px; background-color: lightgray; border-radius: 5px;">
              <div style="width: {prob*100}%; height: 100%; background-color: {bar_color}; border-radius: 5px;"></div>
            </div>""", unsafe_allow_html=True)
    if actual is not None:
        st.info(f"Actual outcome: **{'Granted' if actual == 1 else 'Abandoned'}**")

with col2:
    st.markdown("### Input summary")
    present = [FLAG_LABELS[k] for k in FLAG_LABELS if int(raw[k].iloc[0]) == 1]
    st.write(f"- Technology center {str(raw['art_unit'].iloc[0])[:2]}00, first action in {int(raw['first_action_year'].iloc[0])}, "
             f"{float(raw['months_to_first_action'].iloc[0]):.1f} months after filing, {str(raw['entity_size'].iloc[0]).lower()} entity.")
    st.write("- In the first action: " + (", ".join(present) if present else "no rejections or objections flagged") + ".")
    st.write(f"- Max references in a §103 rejection: {int(raw['cite103_max'].iloc[0])}.")

st.markdown("### What pushed this prediction up or down")
st.caption("Per-feature contributions for this one prediction (XGBoost TreeSHAP, in log-odds). "
           "These explain the model's output, not the causes of real examination outcomes.")
c = contribs.iloc[0].rename("contribution").reset_index().rename(columns={"index": "feature"})
c = c[c["contribution"].abs() > 1e-4].reindex(c["contribution"].abs().sort_values(ascending=False).index).head(10)
c["direction"] = c["contribution"].map(lambda v: "toward grant" if v > 0 else "toward abandonment")
c["feature"] = c["feature"].map(lambda f: f"{feature_label(f)} = {X.iloc[0][f]:g}")
fig = px.bar(c.sort_values("contribution"), x="contribution", y="feature", orientation="h", color="direction",
             color_discrete_map={"toward grant": "#2ca02c", "toward abandonment": "#d62728"},
             labels={"contribution": "Contribution (log-odds)", "feature": "", "direction": ""})
fig.update_layout(legend=dict(orientation="h", y=1.08, x=0))
st.plotly_chart(fig, use_container_width=True)

#what-if
st.markdown("## What-if: change in model prediction")
with st.expander("Toggle first-action fields and compare the model's output", expanded=False):
    st.caption("This shows how the model's prediction changes when an input changes. It does not show that "
               "changing the real application would change its outcome.")
    w = raw.copy()
    cols = st.columns(4)
    for i, k in enumerate(["rejection_101", "rejection_102", "rejection_103", "rejection_112",
                           "objection", "allowed_claims", "cite103_gt3", "cite103_eq1"]):
        with cols[i % 4]:
            w[k] = int(st.checkbox(FLAG_LABELS[k], value=bool(int(raw[k].iloc[0])), key=f"w_{k}"))
    w_prob = float(predict(booster, w)[0][0])
    diff = (w_prob - prob) * 100
    st.markdown(f"**What-if probability:** {w_prob*100:.1f}%  ·  **Change in model prediction:** "
                f"{'+' if diff > 0 else ''}{diff:.1f} percentage points")

#global importance
st.markdown("---")
st.markdown("### Global feature importance")
st.caption("Average gain across all trees: which inputs the model relies on overall (not specific to this prediction).")
imp_path = ROOT / "reports" / "feature_importance_gain.csv"
if imp_path.exists():
    imp = pd.read_csv(imp_path, index_col=0)["gain"].head(12).reset_index()
    imp.columns = ["feature", "gain"]
    imp["feature"] = imp["feature"].map(feature_label)
    st.plotly_chart(px.bar(imp.sort_values("gain"), x="gain", y="feature", orientation="h",
                           labels={"gain": "Average gain", "feature": ""}), use_container_width=True)
