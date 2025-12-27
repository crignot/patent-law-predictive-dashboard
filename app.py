import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


#if on mac use setup.sh to ensure app runs smoothly
#to run: streamlit run app.py

from utils import (
    load_model,
    preprocess_inputs,
    MODEL_COLUMNS,
    TECHCENTER_MAP
)

from header import render_header

#page config
st.set_page_config(
    page_title="Patent Allowance Predictor",
    layout="wide"
)
render_header()

#sidebar layout
with st.sidebar:
    st.title("Patent Predictor")
    st.caption("JCP Project — USPTO Office Action Model")


#loading trained xgb model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

#input form
st.markdown("## Allowance Prediction Tool")
st.write("Enter patent details below to estimate the likelihood of allowance.")
with st.form("patent_input_form"):
    st.subheader("Patent Inputs")

    tech_center = st.selectbox("Tech Center", list(TECHCENTER_MAP.keys()), index=2)

    novelty_similar = st.radio(
        "Is there a very similar existing invention?",
        ["No", "Yes"], index=0
    )

    obviousness_combo = st.radio(
        "Does the invention combine existing known methods?",
        ["No", "Yes"], index=0
    )

    clarity = st.radio(
        "Is the specification clear and well-supported?",
        ["Yes", "No"], index=0
    )

    num_similar_prior = st.slider(
        "Estimated severity of prior art",
        0, 10, 0,
        help="0–1 = very low, 2–3 = moderate, 4–5 = strong, 6+ = heavy prior art"
    )

    submitted = st.form_submit_button("Predict")

#initialize session state
if "baseline_prob" not in st.session_state:
    st.session_state.baseline_prob = None
    st.session_state.baseline_inputs = None

#prediction + output
if submitted:

    #using trained model
    inputs = preprocess_inputs(
        tech_center,
        novelty_similar,
        obviousness_combo,
        clarity,
        num_similar_prior
    )

    prob = model.predict_proba(inputs)[0][1]

    # Save to session_state so the simulator persists
    st.session_state.baseline_prob = prob
    st.session_state.baseline_inputs = {
        "tech_center": tech_center,
        "novelty": novelty_similar,
        "obvious": obviousness_combo,
        "clarity": clarity,
        "num_prior": num_similar_prior,
    }

    st.success("Prediction updated! Scroll down to view results.")


#show prediction + what if AFTER
if st.session_state.baseline_prob is not None:

    prob = st.session_state.baseline_prob
    base_inputs = st.session_state.baseline_inputs

    st.markdown("---")
    st.subheader("Prediction Result")

    col1, col2 = st.columns([1, 2])

    #left: probability + risk bar
    with col1:
        st.metric("Estimated Allowance Probability", f"{prob*100:.1f}%")

        if prob >= 0.7:
            st.success("High likelihood of allowance.")
        elif prob >= 0.4:
            st.warning("Uncertain outcome. Some risk factors present.")
        else:
            st.error("Significant risk of rejection.")

        # risk bar (colored)
        if prob >= 0.7:
            bar_color = "green"
        elif prob >= 0.4:
            bar_color = "yellow"
        else:
            bar_color = "red"

        st.markdown(
            f"""
            <div style="height: 20px; background-color: lightgray; border-radius: 5px;">
                <div style="
                    width: {prob*100}%;
                    height: 100%;
                    background-color: {bar_color};
                    border-radius: 5px;">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    #Right: suggestions
    with col2:
        st.markdown("### Suggested Next Steps")

        suggestions = []

        if base_inputs["novelty"] == "Yes":
            suggestions.append("- Strengthen novelty by adjusting claims or adding new inventive distinctions.")
        if base_inputs["obvious"] == "Yes":
            suggestions.append("- Highlight technical advantages or unexpected results to counter obviousness (§103).")
        if base_inputs["clarity"] == "No":
            suggestions.append("- Improve clarity and specification support to avoid §112 issues.")
        if base_inputs["num_prior"] >= 4:
            suggestions.append("- Expect multiple prior-art references; prepare strong non-obviousness arguments.")

        if not suggestions:
            if prob >= 0.7:
                suggestions.append("- Application appears strong overall based on the provided inputs.")
            elif prob >= 0.4:
                suggestions.append("- Mixed signals: consider double-checking novelty, non-obviousness, and clarity even though no obvious red flags were detected.")
            else:
                suggestions.append(
                    "- Model predicts a low allowance probability based on factors like tech center and art unit, "
                    "even though your answers did not flag specific issues. It may be worth reviewing novelty, "
                    "non-obviousness, and clarity with a patent practitioner."
                )

        for s in suggestions:
            st.write(s)

    # Reasoning summary
    st.markdown("### Model Reasoning Summary")
    explanations = []

    if base_inputs["novelty"] == "Yes":
        explanations.append("- Similar inventions may challenge novelty under §102.")
    if base_inputs["obvious"] == "Yes":
        explanations.append("- Combination of existing methods can trigger obviousness (§103).")
    if base_inputs["clarity"] == "No":
        explanations.append("- Specification clarity or support concerns may lead to §112 rejections.")
    if base_inputs["num_prior"] >= 4:
        explanations.append("- High number of prior art references suggests competition and examiner scrutiny.")


    if not explanations:
        if prob >= 0.7:
            explanations.append("- No major weaknesses detected in the selected factors; model outlook is driven mostly by favorable historical patterns.")
        elif prob >= 0.4:
            explanations.append("- No specific weaknesses detected in the selected factors, but historical outcomes for similar cases are mixed.")
        else:
            explanations.append(
                "- No specific weaknesses were flagged in the selected factors, but the model predicts a low allowance probability "
                "based on historical outcomes for similar tech centers and art units."
            )
    for e in explanations:
        st.write(e)

    # what if simulator
    st.markdown("## What-If Scenario Simulator")
    with st.expander("Open What-If Simulator", expanded=False):
        
        # controls
        w_tc = st.selectbox(
            "Tech Center (What-If)",
            list(TECHCENTER_MAP.keys()),
            index=list(TECHCENTER_MAP.keys()).index(base_inputs["tech_center"])
        )

        w_novel = st.radio(
            "Novelty Issue? (What-If)",
            ["No", "Yes"],
            index=0 if base_inputs["novelty"] == "No" else 1
        )

        w_obvious = st.radio(
            "Obviousness Issue? (What-If)",
            ["No", "Yes"],
            index=0 if base_inputs["obvious"] == "No" else 1
        )

        w_clarity = st.radio(
            "Clarity Issue? (What-If)",
            ["Yes", "No"],
            index=0 if base_inputs["clarity"] == "Yes" else 1
        )

        w_prior = st.slider(
            "Number of Prior Art References (What-If)",
            min_value=0, max_value=10,
            value=base_inputs["num_prior"]
        )


        # comput new prediction
        whatif_df = preprocess_inputs(w_tc, w_novel, w_obvious, w_clarity, w_prior)
        whatif_prob = model.predict_proba(whatif_df)[0][1]

        # compare
        diff = (whatif_prob - prob) * 100
        sign = "+" if diff > 0 else ""
        color = "green" if diff > 0 else "red"

        st.markdown(f"""
            ### What-If Predicted Allowance: **{whatif_prob*100:.1f}%**
            **Impact vs Original:** <span style="color:{color}; font-weight:bold;">{sign}{diff:.1f}%</span>
            """, unsafe_allow_html=True)

        # small chart comparison
        st.bar_chart(
            pd.DataFrame({
                "Baseline": [prob],
                "What-If": [whatif_prob]
            }).T
        )

    # Feature importance
    st.markdown("---")
    st.markdown("### Top Model Features")

    importances = model.feature_importances_
    feature_scores = pd.Series(importances, index=MODEL_COLUMNS).sort_values(ascending=False).head(10)

    feature_df = feature_scores.reset_index()
    feature_df.columns = ["Feature", "Importance"]

    fig_feat = px.bar(
        feature_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Signals Driving the Model",
        color="Importance",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_feat, use_container_width=True)