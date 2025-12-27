import streamlit as st
import pandas as pd
import plotly.express as px

from header import render_header

st.set_page_config(layout="wide")
render_header()
st.title("USPTO Exploratory Data Insights")

#EDA datasets
eda_tc = pd.read_csv("eda_tc_allowance.csv", index_col=0)
eda_avg_rej = pd.read_csv("eda_avg_rejections.csv", index_col=0)
eda_rej_counts = pd.read_csv("eda_rejection_counts.csv", index_col=0)
eda_cite_counts = pd.read_csv("eda_cite_counts.csv", index_col=0)
eda_prior_trend = pd.read_csv("eda_prior_trend.csv")

#plotly colors
TC_COLORS = {
    "Biotech": "#1f77b4",
    "Chemical": "#d62728",
    "Software": "#9467bd",
    "Networking": "#17becf",
    "Crypto": "#bcbd22",
    "Semiconductors": "#ff7f0e",
    "Business": "#8c564b",
    "Mechanical": "#2ca02c",
    "Design": "#e377c2",
}

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Tech Center Allowance Rates",
    "Prior Art Severity",
    "Citation Frequency",
    "Avg Rejections",
    "Rejection Types"
])

with tab1:
    st.markdown("### Tech Center Allowance Rates")

    fig_tc = px.bar(
        eda_tc.reset_index().rename(columns={"index": "TechCenterShort"}),
        x="TechCenterShort",
        y="AllowanceRate",
        color="TechCenterShort",
        color_discrete_map=TC_COLORS,
        title="Allowance Rate by Tech Center"
    )
    st.plotly_chart(fig_tc, use_container_width=True)

    st.caption(
        "Allowance rates vary widely. Semiconductor and Business-related tech centers show stronger approval trends, while Biotech and Chemical areas face stricter examination."
    )

with tab2:
    st.markdown("### Approval Rate vs Prior Art Score (by Tech Center)")

    fig_trend = px.line(
        eda_prior_trend,
        x="PriorArtScore",
        y="approval_rate",
        color="TechCenterShort",
        color_discrete_map=TC_COLORS,
        markers=True,
        title="Approval Rate vs Prior Art Score (by Tech Center)"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.caption(
        "Allowance rates decrease sharply as prior-art severity increases. "
        "Moderate scores retain reasonable approval chances, but heavy prior art "
        "dramatically lowers the likelihood of allowance across all tech centers."
    )

with tab3:
    st.markdown("### Citation Types")

    fig_cite = px.bar(
        eda_cite_counts.reset_index(),
        x="index",
        y="Count",
        title="Citation Severity Frequency",
        color_discrete_sequence=["#2ca02c"]
    )
    st.plotly_chart(fig_cite, use_container_width=True)

    st.caption(
        "High-severity prior art citations—especially 103-Max—occur frequently "
        "and are typically associated with significantly lower allowance odds."
    )

with tab4:
    st.markdown("### Avg. Rejections per Application")

    fig_avg_rej = px.bar(
        eda_avg_rej.reset_index(),
        x="index",
        y="Average",
        title="Average Number of Rejections Per Application",
        color_discrete_sequence=["#d62728"]
    )
    st.plotly_chart(fig_avg_rej, use_container_width=True)

    st.caption(
        "Obviousness (§103) and novelty (§102) rejections appear most frequently, "
        "representing the primary barriers during examination."
    )

with tab5:
    st.markdown("### Rejection Types")

    fig_rej_counts = px.bar(
        eda_rej_counts.reset_index(),
        x="index",
        y="Count",
        title="Rejection Type Frequency",
        color_discrete_sequence=["#8b0000"]
    )
    st.plotly_chart(fig_rej_counts, use_container_width=True)

    st.caption(
        "§103 obviousness rejections dominate the dataset, followed by novelty (§102). "
        "These trends highlight how heavily examiners rely on prior art in decision-making."
    )
