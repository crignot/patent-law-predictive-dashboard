import streamlit as st

from utils.header import render_header

st.set_page_config(layout="wide")
render_header()
st.title("Feature Glossary")
st.caption("Every feature describes the application's FIRST office action or facts known on that day. "
           "Definitions follow the USPTO Office Action Research Dataset variable tables (2017 release).")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **rejection_101** — §101 rejection raised (patent-eligible subject matter).
    **rejection_102** — §102 rejection raised (novelty).
    **rejection_103** — §103 rejection raised (obviousness).
    **rejection_112** — §112 rejection raised (clarity, written description, enablement).
    **rejection_dp** — non-statutory double-patenting rejection raised.
    **objection** — objection raised (form or informality, not a rejection).
    **allowed_claims** — the examiner indicated at least one allowable claim in this action.
    """)

with col2:
    st.markdown("""
    **cite102_gt1** — more than one prior-art reference cited in a §102 rejection.
    **cite103_gt3** — more than three references cited in a §103 rejection.
    **cite103_eq1** — exactly one reference cited in a §103 rejection.
    **cite103_max** — the most references cited in any single §103 rejection (a count).
    **tc_XXXX** — technology center, from the first two digits of the art unit.
    **entity_small / entity_micro** — applicant's claimed fee status (large entity is the baseline).
    **signature_X** — who signed the action (1 primary examiner, 2 examiner + primary, 3 examiner + supervisor, 4 with director).
    **first_action_year / months_to_first_action** — when the action was mailed, and how long after filing.
    """)

st.markdown("**Target:** `granted` = 1 if the application issued as a patent, 0 if it was abandoned "
            "(PatEx application status, 2022 release). Still-pending applications are excluded.")
