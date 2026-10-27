"""Readable names for model columns and report fields, shared by every page."""

from utils.features import TECH_CENTERS

FEATURE_LABELS = {
    "rejection_101": "§101 rejection",
    "rejection_102": "§102 rejection",
    "rejection_103": "§103 rejection",
    "rejection_112": "§112 rejection",
    "rejection_dp": "Double-patenting rejection",
    "objection": "Objection",
    "allowed_claims": "Allowable claim(s) indicated",
    "cite102_gt1": "§102: more than 1 reference",
    "cite103_gt3": "§103: more than 3 references",
    "cite103_eq1": "§103: exactly 1 reference",
    "cite103_max": "Max references in a §103 rejection",
    "first_action_year": "Year of first observed action",
    "months_to_first_action": "Months from filing to first observed action",
    "signature_1": "Signed: primary examiner",
    "signature_2": "Signed: examiner + primary",
    "signature_3": "Signed: examiner + supervisor",
    "signature_4": "Signed: with director",
    **{f"tc_{code}": f"TC {code}: {name}" for code, name in TECH_CENTERS.items()},
}

# report / table columns
COLUMN_LABELS = {
    "model": "Model",
    "roc_auc": "ROC-AUC",
    "avg_precision_abandoned": "Avg. precision (abandoned)",
    "log_loss": "Log loss",
    "brier": "Brier",
    "threshold": "Threshold",
    "precision_abandoned": "Precision (abandoned)",
    "recall_abandoned": "Recall (abandoned)",
    "precision_granted": "Precision (granted)",
    "recall_granted": "Recall (granted)",
    "n": "Applications",
    "positive_rate": "Grant rate",
    "tech_center": "TC",
    "tech_center_name": "Technology center",
    "first_action_year": "First observed action year",
    "applications": "Applications",
    "actual_grant_rate": "Actual grant rate",
    "mean_predicted": "Mean predicted",
    "accuracy_at_threshold": "Accuracy at threshold",
    "app_id": "Application",
    "entity_size": "Entity (2022 status)",
    "appl_status_desc": "Final status",
    "granted": "Granted",
    "p_granted": "P(grant)",
    "allowed_claims": "Allowable claims",
    "rejection_101": "§101",
    "rejection_102": "§102",
    "rejection_103": "§103",
    "rejection_112": "§112",
    "cite103_max": "Max §103 refs",
    "months_to_first_action": "Months to 1st action",
    "error": "Error type",
}


def feature_label(col: str) -> str:
    return FEATURE_LABELS.get(col, col)
