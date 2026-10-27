from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.header import render_header

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

st.set_page_config(layout="wide")
render_header()
st.title("USPTO Exploratory Data Insights")
st.caption("All rates are the share of applications **eventually granted**, grouped by what was in their first "
           "office action (2008–2017 first actions, outcomes as of the 2022 PatEx release). These are descriptive "
           "associations, not causal effects.")

eda_tc = pd.read_csv(DATA_DIR / "eda_tc_grant_rate.csv")
eda_rej = pd.read_csv(DATA_DIR / "eda_grant_rate_by_rejection.csv")
eda_year = pd.read_csv(DATA_DIR / "eda_grant_rate_by_year.csv")
eda_cite = pd.read_csv(DATA_DIR / "eda_grant_rate_by_cite103.csv")
eda_allow = pd.read_csv(DATA_DIR / "eda_grant_rate_by_allowable_claims.csv")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "By Tech Center", "By Rejection Type", "By §103 Citations", "Allowable Claims", "By Year"])

with tab1:
    d = eda_tc[eda_tc["applications"] >= 100]
    st.plotly_chart(px.bar(d, x="TechCenter", y="grant_rate", hover_data=["applications"],
                           title="Eventual grant rate by technology center"), use_container_width=True)

with tab2:
    d = eda_rej.assign(present=eda_rej["value"].map({0: "Not in first action", 1: "In first action"}))
    st.plotly_chart(px.bar(d, x="flag", y="grant_rate", color="present", barmode="group", hover_data=["applications"],
                           title="Grant rate when each rejection type does / does not appear in the first action"),
                    use_container_width=True)

with tab3:
    st.plotly_chart(px.line(eda_cite, x="cite103_max_capped", y="grant_rate", markers=True, hover_data=["applications"],
                            title="Grant rate by max references cited in a §103 rejection (6 = 6 or more)"),
                    use_container_width=True)

with tab4:
    d = eda_allow.assign(first_action=eda_allow["allowed_claims"].map({0: "No allowable claims", 1: "Some claims allowable"}))
    st.plotly_chart(px.bar(d, x="first_action", y="grant_rate", hover_data=["applications"],
                           title="Grant rate when the first action already indicates allowable claims"),
                    use_container_width=True)

with tab5:
    st.plotly_chart(px.line(eda_year, x="first_action_year", y="grant_rate", markers=True, hover_data=["applications"],
                            title="Grant rate by year of first action"), use_container_width=True)
