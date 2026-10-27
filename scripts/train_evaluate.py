"""Train and evaluate allowance models on a time-based split.

Task: at the moment an application's first OBSERVED office action is mailed, predict whether the
application will eventually be granted (1) or abandoned (0).

Split by first-action year (older -> newer, no shuffling):
  train       2008-2013   fit models
  validation  2014        choose hyperparameters and the decision threshold
  test        2015-2017   evaluated once, after everything above is fixed

No resampling (SMOTE) is used: models are fit on the real class balance, so predicted
probabilities stay interpretable. Run scripts/build_dataset.py first.

Run:  python scripts/train_evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, log_loss,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.features import MODEL_COLUMNS, TECH_CENTERS, build_features, drop_group  # noqa: E402

RAW = ROOT / "data_raw"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
SPLITS = {"train": (2008, 2013), "validation": (2014, 2014), "test": (2015, 2017)}
SEED = 42


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "avg_precision_granted": average_precision_score(y, p),
        "avg_precision_abandoned": average_precision_score(1 - y, 1 - p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "brier": brier_score_loss(y, p),
        "threshold": threshold,
        "precision_granted": precision_score(y, pred, zero_division=0),
        "recall_granted": recall_score(y, pred, zero_division=0),
        "precision_abandoned": precision_score(1 - y, 1 - pred, zero_division=0),
        "recall_abandoned": recall_score(1 - y, 1 - pred, zero_division=0),
        "n": len(y),
        "positive_rate": float(np.mean(y)),
    }


def pick_threshold(y, p):
    """Threshold maximizing balanced accuracy (mean recall of granted and abandoned) on validation."""
    grid = np.linspace(0.05, 0.95, 181)
    scores = [(recall_score(y, p >= t) + recall_score(1 - y, p < t)) / 2 for t in grid]
    return float(grid[int(np.argmax(scores))])


def xgb(**overrides):
    params = dict(n_estimators=2000, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, tree_method="hist", eval_metric="logloss", early_stopping_rounds=50,
                  random_state=SEED, n_jobs=-1)
    params.update(overrides)
    return XGBClassifier(**params)


def main():
    df = pd.read_parquet(RAW / "first_actions.parquet").dropna(subset=["granted"]).reset_index(drop=True)
    df["granted"] = df["granted"].astype(int)
    X_all = build_features(df)
    y_all = df["granted"].to_numpy()
    year = df["first_action_year"]
    idx = {k: np.where(year.between(*v))[0] for k, v in SPLITS.items()}
    X = {k: X_all.iloc[i] for k, i in idx.items()}
    y = {k: y_all[i] for k, i in idx.items()}
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    split_info = {k: {"years": list(SPLITS[k]), "n": int(len(idx[k])), "granted_rate": float(y[k].mean())} for k in SPLITS}
    (REPORTS / "split.json").write_text(json.dumps(split_info, indent=2))
    print(json.dumps(split_info, indent=2))

    results, val_probs, test_probs = [], {}, {}

    # 1) constant baseline: training-set grant rate for everyone
    base = float(y["train"].mean())
    val_probs["Constant (train grant rate)"] = np.full(len(y["validation"]), base)
    test_probs["Constant (train grant rate)"] = np.full(len(y["test"]), base)

    # 2) logistic regression
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(X["train"], y["train"])
    val_probs["Logistic regression"] = lr.predict_proba(X["validation"])[:, 1]
    test_probs["Logistic regression"] = lr.predict_proba(X["test"])[:, 1]

    # 3) XGBoost: pick max_depth on validation log loss (early stopping also uses validation)
    best = None
    for depth in (4, 6, 8):
        m = xgb(max_depth=depth)
        m.fit(X["train"], y["train"], eval_set=[(X["validation"], y["validation"])], verbose=False)
        pv = m.predict_proba(X["validation"])[:, 1]
        ll = log_loss(y["validation"], pv)
        print(f"xgb depth={depth} trees={m.best_iteration + 1} val_logloss={ll:.4f} val_auc={roc_auc_score(y['validation'], pv):.4f}")
        if best is None or ll < best[0]:
            best = (ll, depth, m, pv)
    _, best_depth, model, pv = best
    val_probs["XGBoost"] = pv
    test_probs["XGBoost"] = model.predict_proba(X["test"])[:, 1]

    # 4) ablations: same settings, one feature group removed
    for group, label in (("citations", "XGBoost without citation features"),
                         ("tech_center", "XGBoost without tech center / art unit")):
        cols = drop_group(MODEL_COLUMNS, group)
        m = xgb(max_depth=best_depth)
        m.fit(X["train"][cols], y["train"], eval_set=[(X["validation"][cols], y["validation"])], verbose=False)
        val_probs[label] = m.predict_proba(X["validation"][cols])[:, 1]
        test_probs[label] = m.predict_proba(X["test"][cols])[:, 1]

    # thresholds are chosen on validation only, then applied unchanged to test
    for name in test_probs:
        t = pick_threshold(y["validation"], val_probs[name]) if name.startswith(("Logistic", "XGBoost")) else base
        row = {"model": name, **metrics(y["test"], test_probs[name], t)}
        if name.startswith("Constant"):
            row["roc_auc"] = 0.5  # a constant score has no ranking ability
        results.append(row)
    comp = pd.DataFrame(results)
    comp.to_csv(REPORTS / "model_comparison.csv", index=False)
    print(comp.round(4).to_string(index=False))

    # ---- artifacts for the final model (XGBoost) ----
    p_test = test_probs["XGBoost"]
    thr = float(comp.loc[comp.model == "XGBoost", "threshold"].iloc[0])
    model.save_model(MODELS / "xgb_model.json")
    meta = {"columns": MODEL_COLUMNS, "threshold": thr, "max_depth": best_depth,
            "n_trees": int(model.best_iteration + 1), "splits": split_info,
            "task": "P(application eventually granted | first observed office action)",
            "train_grant_rate": base}
    (MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2))

    prec, rec, _ = precision_recall_curve(1 - y["test"], 1 - p_test)  # abandonment-risk view
    step = max(1, len(prec) // 400)
    pd.DataFrame({"recall_abandoned": rec[::step], "precision_abandoned": prec[::step]}).to_csv(REPORTS / "pr_curve_abandoned.csv", index=False)

    frac_pos, mean_pred = calibration_curve(y["test"], p_test, n_bins=10, strategy="quantile")
    pd.DataFrame({"mean_predicted": mean_pred, "observed_grant_rate": frac_pos}).to_csv(REPORTS / "calibration.csv", index=False)

    pred = (p_test >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y["test"], pred).ravel()
    (REPORTS / "confusion_matrix.json").write_text(json.dumps(
        {"threshold": thr, "true_granted_pred_granted": int(tp), "true_granted_pred_abandoned": int(fn),
         "true_abandoned_pred_granted": int(fp), "true_abandoned_pred_abandoned": int(tn)}, indent=2))

    test_df = df.iloc[idx["test"]].copy()
    test_df["p_granted"] = p_test
    test_df["pred"] = pred

    def slice_table(key):
        rows = []
        for k, g in test_df.groupby(key):
            rows.append({key: k, "applications": len(g), "actual_grant_rate": g["granted"].mean(),
                         "mean_predicted": g["p_granted"].mean(),
                         "accuracy_at_threshold": (g["pred"] == g["granted"]).mean(),
                         "roc_auc": roc_auc_score(g["granted"], g["p_granted"]) if g["granted"].nunique() > 1 else np.nan})
        return pd.DataFrame(rows)

    tc = slice_table("tech_center")
    tc.insert(1, "tech_center_name", tc["tech_center"].map(TECH_CENTERS))
    tc.to_csv(REPORTS / "slices_tech_center.csv", index=False)
    slice_table("first_action_year").to_csv(REPORTS / "slices_year.csv", index=False)

    show = ["app_id", "first_action_year", "tech_center", "entity_size", "appl_status_desc", "granted", "p_granted",
            "allowed_claims", "rejection_101", "rejection_102", "rejection_103", "rejection_112", "cite103_max",
            "months_to_first_action"]
    fp_ex = test_df[(test_df.granted == 0)].nlargest(8, "p_granted")[show].assign(error="Confident, but abandoned")
    fn_ex = test_df[(test_df.granted == 1)].nsmallest(8, "p_granted")[show].assign(error="Low score, but granted")
    pd.concat([fp_ex, fn_ex]).round(3).to_csv(REPORTS / "error_examples.csv", index=False)

    imp = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(MODEL_COLUMNS).fillna(0)
    imp.sort_values(ascending=False).rename("gain").to_csv(REPORTS / "feature_importance_gain.csv", header=True)

    # held-out examples for the dashboard ("Load an example office action")
    raw_cols = ["app_id", "first_action_year", "months_to_first_action", "art_unit", "signature_type",
                "rejection_101", "rejection_102", "rejection_103", "rejection_112", "rejection_dp", "objection",
                "allowed_claims", "cite102_gt1", "cite103_gt3", "cite103_eq1", "cite103_max", "granted"]
    ex = pd.concat(test_df[test_df["granted"] == v].sample(60, random_state=SEED) for v in (0, 1))[raw_cols]
    ex["months_to_first_action"] = ex["months_to_first_action"].round(1)
    ex.sample(frac=1, random_state=SEED).to_csv(ROOT / "data" / "example_office_actions.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
