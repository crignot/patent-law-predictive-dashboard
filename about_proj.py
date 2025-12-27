import streamlit as st

from header import render_header

st.set_page_config(layout="wide")
render_header()

#card style from main page
st.markdown("""
<style>
.card {
    background-color: #fafbfc;
    padding: 20px 26px;
    border: 1px solid #e5e5e5;
    border-radius: 10px;
    margin-top: 15px;
    font-family: "Inter", sans-serif;
}
</style>
""", unsafe_allow_html=True)

# --- PAGE TITLE ---
st.title("About This Project")

# --- PAGE CONTENT ---
st.markdown("""
<div class="card">
This dashboard was built as part of a JCP research initiative studying how USPTO
Office Actions influence the likelihood that a patent application receives allowance.  
Our dataset contains thousands of applications across USPTO Technology Centers with detailed
information on rejection types, citation severity, prior-art characteristics, and outcomes.

The goal is to make these insights usable early in the drafting process by combining:

- Exploratory data analysis across USPTO Tech Centers  
- A trained XGBoost model predicting allowance probability  
- Scenario testing to evaluate how specific factors influence outcomes  

This tool **does not replace legal advice**. It provides a data-driven perspective on historical
examination patterns to support strategic decision-making.
</div>
""", unsafe_allow_html=True)
