"""Load the trained model and score first office actions for the dashboard."""

import json
from pathlib import Path

import pandas as pd

from utils.features import MODEL_COLUMNS, build_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xgb_model.json"
META_PATH = ROOT / "models" / "model_meta.json"


def model_available() -> bool:
    return MODEL_PATH.exists() and META_PATH.exists()


def load_model():
    """Returns (booster, meta) or (None, None) if the model files have not been generated."""
    if not model_available():
        return None, None
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    meta = json.loads(META_PATH.read_text())
    return booster, meta


def predict(booster, raw: pd.DataFrame):
    """Probability of eventual grant plus per-feature contributions (log-odds units, from XGBoost's TreeSHAP)."""
    import xgboost as xgb

    X = build_features(raw)
    dm = xgb.DMatrix(X, feature_names=MODEL_COLUMNS)
    prob = booster.predict(dm)
    contribs = booster.predict(dm, pred_contribs=True)  # last column = bias
    contrib_df = pd.DataFrame(contribs[:, :-1], columns=MODEL_COLUMNS, index=raw.index)
    return prob, contrib_df, X
