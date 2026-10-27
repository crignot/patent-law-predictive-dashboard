"""Build the modeling dataset: one row per application, features from its FIRST office action,
label = whether the application was eventually granted.

Inputs (download from USPTO, not committed; see README "Reproducing the results"):
  data_raw/office_actions.csv     USPTO Office Action Research Dataset (2017 release), office_actions table
  data_raw/application_data.csv   USPTO Patent Examination Research Dataset (PatEx, 2022 release), application_data table

Outputs:
  data_raw/first_actions.parquet  modeling table (not committed; too large)
  data/example_office_actions.csv small sample of held-out first actions used by the dashboard
  data/eda_*.csv                  summary tables for the EDA page
  reports/dataset_summary.json    row counts at every step

Run:  python scripts/build_dataset.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.features import ACTION_FLAGS, CITATION_FIELDS, TECH_CENTERS, tech_center_from_art_unit  # noqa: E402

GRANTED_STATUSES = {
    "Patented Case",
    "Patent Expired Due to NonPayment of Maintenance Fees Under 37 CFR 1.362",
}


def label_outcome(app: pd.DataFrame) -> pd.Series:
    """1 = granted, 0 = abandoned, NaN = still pending / unclear (kept out of training and evaluation)."""
    status = app["appl_status_desc"].fillna("")
    granted = app["patent_number"].notna() | status.isin(GRANTED_STATUSES)
    abandoned = status.str.contains("abandon", case=False) & ~granted
    out = pd.Series(np.nan, index=app.index)
    out[granted] = 1
    out[abandoned] = 0
    return out


def main(raw_dir: Path) -> None:
    counts = {}

    oa_cols = ["app_id", "ifw_number", "document_cd", "mail_dt", "art_unit", "signature_type"] + ACTION_FLAGS + CITATION_FIELDS
    oa = pd.read_csv(raw_dir / "office_actions.csv", usecols=oa_cols, dtype={"app_id": str, "art_unit": str, "ifw_number": str})
    counts["office_action_rows"] = len(oa)
    counts["office_action_applications"] = oa["app_id"].nunique()

    # Parse dates BEFORE sorting so "first" really means earliest.
    oa["mail_dt"] = pd.to_datetime(oa["mail_dt"], errors="coerce")
    oa = oa.dropna(subset=["mail_dt"])
    oa["app_id"] = oa["app_id"].str.zfill(8)
    # On the same day, a non-final action (CTNF) comes before a final one (CTFR).
    oa["is_final"] = (oa["document_cd"] == "CTFR").astype(int)
    oa = oa.sort_values(["app_id", "mail_dt", "is_final", "ifw_number"])
    first = oa.drop_duplicates("app_id", keep="first").copy()
    counts["applications_with_dated_first_action"] = len(first)

    app_cols = ["application_number", "filing_date", "application_invention_type", "appl_status_desc",
                "patent_number", "patent_issue_date", "small_entity_indicator"]
    chunks = pd.read_csv(raw_dir / "application_data.csv", usecols=app_cols, dtype=str, chunksize=2_000_000)
    ids = set(first["app_id"])
    app = pd.concat(c[c["application_number"].isin(ids)] for c in chunks)
    app = app.drop_duplicates("application_number")
    counts["applications_matched_to_patex"] = len(app)

    df = first.merge(app, left_on="app_id", right_on="application_number", how="inner")
    df = df[df["application_invention_type"] == "Utility"].copy()
    counts["utility_applications"] = len(df)

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["months_to_first_action"] = (df["mail_dt"] - df["filing_date"]).dt.days / 30.44
    df["first_action_year"] = df["mail_dt"].dt.year
    df["entity_size"] = df["small_entity_indicator"].fillna("UNKNOWN")
    df["tech_center"] = df["art_unit"].map(tech_center_from_art_unit)
    df = df[df["months_to_first_action"].between(0, 240) & df["tech_center"].notna()].copy()
    counts["after_date_and_art_unit_checks"] = len(df)

    df["granted"] = label_outcome(df)
    counts["label_granted"] = int((df["granted"] == 1).sum())
    counts["label_abandoned"] = int((df["granted"] == 0).sum())
    counts["label_unknown_excluded"] = int(df["granted"].isna().sum())

    keep = ["app_id", "mail_dt", "first_action_year", "months_to_first_action", "art_unit", "tech_center",
            "signature_type", "entity_size", "document_cd", "granted", "appl_status_desc"] + ACTION_FLAGS + CITATION_FIELDS
    df = df[keep]
    df.to_parquet(raw_dir / "first_actions.parquet", index=False)

    known = df.dropna(subset=["granted"])
    counts["modeling_rows"] = len(known)
    counts["first_action_date_range"] = [str(known["mail_dt"].min().date()), str(known["mail_dt"].max().date())]

    # ---- small committed artifacts for the dashboard ----
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    tc_name = known["tech_center"].map(TECH_CENTERS)

    (known.assign(TechCenter=tc_name).groupby("TechCenter")
          .agg(grant_rate=("granted", "mean"), applications=("granted", "size"))
          .sort_values("grant_rate").to_csv(data_dir / "eda_tc_grant_rate.csv"))

    rej = ["rejection_101", "rejection_102", "rejection_103", "rejection_112", "rejection_dp", "objection"]
    rows = []
    for r in rej:
        for v in (0, 1):
            s = known.loc[known[r] == v, "granted"]
            rows.append({"flag": r, "value": v, "grant_rate": s.mean(), "applications": len(s)})
    pd.DataFrame(rows).to_csv(data_dir / "eda_grant_rate_by_rejection.csv", index=False)

    (known.groupby("first_action_year")
          .agg(grant_rate=("granted", "mean"), applications=("granted", "size"))
          .to_csv(data_dir / "eda_grant_rate_by_year.csv"))

    known.assign(cite103_max_capped=known["cite103_max"].clip(upper=6)).groupby("cite103_max_capped") \
         .agg(grant_rate=("granted", "mean"), applications=("granted", "size")).to_csv(data_dir / "eda_grant_rate_by_cite103.csv")

    (known.groupby("allowed_claims")
          .agg(grant_rate=("granted", "mean"), applications=("granted", "size"))
          .to_csv(data_dir / "eda_grant_rate_by_allowable_claims.csv"))

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "dataset_summary.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data_raw")
    main(p.parse_args().raw_dir)
