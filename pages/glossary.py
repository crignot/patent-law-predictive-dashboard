import streamlit as st

from utils.features import TECH_CENTERS
from utils.header import render_header

st.set_page_config(page_title="Feature Glossary", layout="wide")
render_header()
st.title("Feature Glossary")
st.caption("Every feature describes the application's first office action or facts known on that day. "
           "Definitions follow the USPTO Office Action Research Dataset variable tables (2017 release).")

GLOSSARY = {
    "Rejections and objections in the first action": [
        ("rejection_101", "§101 rejection", "Claims are not patent-eligible subject matter (e.g. abstract ideas)."),
        ("rejection_102", "§102 rejection", "Claims are not novel: a single prior-art reference discloses the invention."),
        ("rejection_103", "§103 rejection", "Claims are obvious in light of one or more prior-art references."),
        ("rejection_112", "§112 rejection", "Claims are unclear, or the specification does not describe or enable them."),
        ("rejection_dp", "Double-patenting rejection", "Claims are not distinct from the applicant's own other patent or application."),
        ("objection", "Objection", "A problem of form or informality (e.g. drawings, wording), not a rejection on the merits."),
        ("allowed_claims", "Allowable claims", "The examiner indicated at least one claim would be allowable."),
    ],
    "Prior-art citations": [
        ("cite102_gt1", "§102: more than 1 reference", "More than one reference cited in a §102 rejection."),
        ("cite103_gt3", "§103: more than 3 references", "More than three references combined in a §103 rejection."),
        ("cite103_eq1", "§103: exactly 1 reference", "A §103 rejection resting on a single reference."),
        ("cite103_max", "Max references in a §103 rejection", "The most references cited in any one §103 rejection (a count)."),
    ],
    "Application and examiner context": [
        ("tc_XXXX", "Technology center", "From the first two digits of the art unit. "
                                         + ", ".join(f"{c} {n}" for c, n in TECH_CENTERS.items()) + "."),
        ("entity_small / entity_micro", "Entity size", "Applicant's claimed fee status. Large (undiscounted) entity is the baseline."),
        ("signature_1 … signature_4", "Who signed", "1 primary examiner, 2 examiner + primary, 3 examiner + supervisor, "
                                                    "4 with director. Examiner alone is the baseline."),
        ("first_action_year", "Year of first action", "Calendar year the first action was mailed."),
        ("months_to_first_action", "Months to first action", "Time from filing to the first action."),
    ],
}

for section, rows in GLOSSARY.items():
    st.markdown(f"#### {section}")
    table = "| Feature | Name | Meaning |\n|---|---|---|\n"
    table += "\n".join(f"| `{code}` | **{name}** | {meaning} |" for code, name, meaning in rows)
    st.markdown(table)

st.info("**Target:** `granted` = 1 if the application issued as a patent, 0 if it was abandoned "
        "(PatEx application status, 2022 release). Still-pending applications are excluded.")
