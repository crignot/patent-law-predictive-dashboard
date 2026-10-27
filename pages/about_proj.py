import streamlit as st

from utils.header import render_header

st.set_page_config(page_title="About", layout="wide")
render_header()

st.title("About This Project")

with st.container(border=True):
    st.markdown("""
This dashboard was built as part of a JCP research initiative studying how what happens in a patent
application's **first observed office action** relates to whether the application is eventually granted.

The data joins the USPTO Office Action Research Dataset (about 2.2 million utility applications whose first
office action was mailed between 2008 and 2017) with final outcomes from the USPTO Patent Examination
Research Dataset (2022 release).

**The dashboard combines:**

- Exploratory analysis of grant rates by rejection type, citations, and technology center
- An XGBoost model, evaluated against baselines on a time-based split
- A what-if view showing how the model's prediction responds to changes in its inputs
""")

st.warning("This tool **does not replace legal advice**. It describes historical examination patterns; it does not "
           "show that changing an application would change its outcome.")
