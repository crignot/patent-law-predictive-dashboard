"""Shared feature construction for training and for the dashboard.

Both scripts/train_evaluate.py and the Streamlit app call build_features(), so a
prediction in the app is computed from exactly the same columns the model was
trained on.

Prediction moment: the day the FIRST office action on an application is mailed.
Every input below is known at that moment.
"""

import pandas as pd

# Raw fields taken from the first office action (USPTO Office Action Research
# Dataset, office_actions table). All are 0/1 flags except cite103_max.
ACTION_FLAGS = [
    "rejection_101",   # subject-matter eligibility rejection
    "rejection_102",   # novelty rejection
    "rejection_103",   # obviousness rejection
    "rejection_112",   # clarity / written-description rejection
    "rejection_dp",    # double-patenting rejection
    "objection",       # objection (form / informality)
    "allowed_claims",  # examiner indicated at least one allowable claim
]
CITATION_FIELDS = [
    "cite102_gt1",     # more than one reference cited in a 102 rejection
    "cite103_gt3",     # more than three references cited in a 103 rejection
    "cite103_eq1",     # exactly one reference cited in a 103 rejection
    "cite103_max",     # max references cited in any 103 rejection (count)
]
TIMING_FIELDS = [
    "first_action_year",          # calendar year the first action was mailed
    "months_to_first_action",     # filing date -> first action, in months
]

TECH_CENTERS = {
    1600: "Biotech & Organic Chemistry",
    1700: "Chemical & Materials Engineering",
    2100: "Computer Architecture & Software",
    2400: "Networking & Communications",
    2600: "Communications & Imaging",
    2800: "Semiconductors & Electrical Systems",
    2900: "Designs",
    3600: "Business Methods & Transportation",
    3700: "Mechanical Engineering & Medical Devices",
}
TECH_CENTER_COLUMNS = [f"tc_{code}" for code in TECH_CENTERS]

# Applicant entity size from PatEx (fee status claimed by the applicant; large entity is the baseline)
ENTITY_SIZES = {"SMALL": "entity_small", "MICRO": "entity_micro"}
# Who signed the first action (office_actions.signature_type): 0 examiner only, 1 primary examiner,
# 2 examiner + primary, 3 examiner + supervisor, 4 examiner + primary/supervisor + director
SIGNATURE_COLUMNS = [f"signature_{k}" for k in (1, 2, 3, 4)]
APPLICANT_EXAMINER_COLUMNS = list(ENTITY_SIZES.values()) + SIGNATURE_COLUMNS

MODEL_COLUMNS = ACTION_FLAGS + CITATION_FIELDS + TIMING_FIELDS + APPLICANT_EXAMINER_COLUMNS + TECH_CENTER_COLUMNS

# Column groups used for ablations
FEATURE_GROUPS = {
    "citations": CITATION_FIELDS,
    "tech_center": TECH_CENTER_COLUMNS,
}


def tech_center_from_art_unit(art_unit) -> int | None:
    """USPTO art units are 4 digits; the first two give the technology center (e.g. 2134 -> 2100)."""
    try:
        code = int(str(art_unit).strip()[:4])
    except (TypeError, ValueError):
        return None
    tc = (code // 100) * 100
    return tc if tc in TECH_CENTERS else None


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw first-office-action fields into the model matrix (one row per application).

    `raw` needs: the ACTION_FLAGS and CITATION_FIELDS columns, `art_unit`,
    `first_action_year`, `months_to_first_action`, `entity_size`, and `signature_type`.
    """
    X = pd.DataFrame(index=raw.index)
    for col in ACTION_FLAGS + CITATION_FIELDS:
        X[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).astype(int)
    X["first_action_year"] = pd.to_numeric(raw["first_action_year"], errors="coerce")
    X["months_to_first_action"] = pd.to_numeric(raw["months_to_first_action"], errors="coerce")

    entity = raw["entity_size"].astype(str).str.upper()
    for size, col in ENTITY_SIZES.items():
        X[col] = (entity == size).astype(int)
    sig = pd.to_numeric(raw["signature_type"], errors="coerce")
    for k in (1, 2, 3, 4):
        X[f"signature_{k}"] = (sig == k).astype(int)

    tc = raw["art_unit"].map(tech_center_from_art_unit)
    for code in TECH_CENTERS:
        X[f"tc_{code}"] = (tc == code).astype(int)

    return X[MODEL_COLUMNS]


def drop_group(columns: list[str], group: str) -> list[str]:
    """Feature list with one ablation group removed."""
    removed = set(FEATURE_GROUPS[group])
    return [c for c in columns if c not in removed]
