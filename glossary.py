import streamlit as st

from header import render_header

st.set_page_config(layout="wide")
render_header()
st.title("Model Feature Glossary")

col1, col2 = st.columns(2)
    
with col1:
    st.markdown("""
    **Rejection102** — Novelty rejection (§102).  
    **Rejection103** — Obviousness rejection (§103).  
    **Rejection112** — Clarity / written-description issues (§112).  
    **Objection** — Examiner objection (non-rejection clarity or form issue).  
    **PriorArtScore** — Aggregated severity score (0–10) representing prior-art strength.
    """)

with col2:
    st.markdown("""
    **Cite102_GT1** — More than one §102 prior-art reference.  
    **Cite103_GT3** — 3+ obviousness citations.  
    **Cite103_EQ1** — Exactly one §103 citation.  
    **Cite103_Max** — Maximum-weighted obviousness citation.  
    - **TechCenter** — Machine-learning encoding of USPTO Tech Centers
    """)