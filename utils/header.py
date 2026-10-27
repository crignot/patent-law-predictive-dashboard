import streamlit as st


def render_header():
    #style for header
    st.markdown("""
    <style>
    .top-banner {
        background: linear-gradient(90deg, #003366, #006699);
        padding: 25px 40px;
        border-radius: 8px;
        color: white;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    #banner
    st.markdown("""
    <div class="top-banner">
        <h2 style="margin-bottom:0;">USPTO Patent Allowance Prediction Dashboard</h2>
        <p style="margin-top:6px; opacity:0.85;">
            Estimating, at the first observed office action, whether a patent application will eventually be granted
        </p>
    </div>
    """, unsafe_allow_html=True)

    #navigation bar
    cols = st.columns(5)
    with cols[0]:
        st.page_link(page="app.py", label="Predict")
    with cols[1]:
        st.page_link(page="pages/evaluation.py", label="Model Evaluation")
    with cols[2]:
        st.page_link(page="pages/eda.py", label="EDA Insights")
    with cols[3]:
        st.page_link(page="pages/glossary.py", label="Feature Glossary")
    with cols[4]:
        st.page_link(page="pages/about_proj.py", label="About")
