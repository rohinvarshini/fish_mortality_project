# data/label_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Scientifically validated composite risk labeling engine.
#
# Labels each water quality row as:
#   0 = Low Risk
#   1 = Moderate Risk
#   2 = High Risk
#
# Based on weighted multi-parameter scoring:
#   - DO (weight 3×)        — Primary mortality driver (FAO, NIH)
#   - Turbidity (weight 2×) — Algal bloom proxy / leading indicator
#   - pH (weight 1×)        — Ammonia toxicity amplifier
#   - Temperature (weight 1×)— O₂ solubility modulator
#   - Ammonia (weight 2×)   — Direct toxin (if column present)
#
# DO OVERRIDE: if DO < 3.0 mg/L → always High Risk (FAO validated)
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DO_HIGH_THRESHOLD, DO_MOD_THRESHOLD,
    TURB_HIGH_NTU, TURB_MOD_NTU,
    PH_HIGH_UPPER, PH_MOD_UPPER, PH_HIGH_LOWER, PH_MOD_LOWER,
    TEMP_HIGH, TEMP_MOD,
    AMMONIA_HIGH, AMMONIA_MOD,
)


def _score_do(val: float) -> int:
    """DO is the primary driver — weight applied at call site."""
    if pd.isna(val):
        return 1  # treat missing as moderate (conservative)
    if val < DO_HIGH_THRESHOLD:
        return 2
    if val < DO_MOD_THRESHOLD:
        return 1
    return 0


def _score_turbidity(val: float) -> int:
    if pd.isna(val):
        return 0
    if val > TURB_HIGH_NTU:
        return 2
    if val > TURB_MOD_NTU:
        return 1
    return 0


def _score_ph(val: float) -> int:
    if pd.isna(val):
        return 0
    if val > PH_HIGH_UPPER or val < PH_HIGH_LOWER:
        return 2
    if val > PH_MOD_UPPER or val < PH_MOD_LOWER:
        return 1
    return 0


def _score_temperature(val: float) -> int:
    if pd.isna(val):
        return 0
    if val > TEMP_HIGH:
        return 2
    if val > TEMP_MOD:
        return 1
    return 0


def _score_ammonia(val: float) -> int:
    if pd.isna(val):
        return 0
    if val > AMMONIA_HIGH:
        return 2
    if val > AMMONIA_MOD:
        return 1
    return 0


def compute_risk_label(row: pd.Series) -> int:
    """
    Computes composite risk label for a single water quality row.

    Returns
    -------
    int : 0 = Low, 1 = Moderate, 2 = High
    """
    do_val = row.get("DO", row.get("dissolved_oxygen", np.nan))

    # ── DO OVERRIDE (FAO / Global Seafood Alliance validated) ────────────
    if not pd.isna(do_val) and do_val < DO_HIGH_THRESHOLD:
        return 2  # Always High Risk regardless of other parameters

    # ── Weighted composite scoring ────────────────────────────────────────
    score = 0
    score += 3 * _score_do(do_val)

    turb_val = row.get("turbidity", row.get("turbidity_NTU", np.nan))
    score += 2 * _score_turbidity(turb_val)

    ph_val = row.get("pH", row.get("ph", np.nan))
    score += 1 * _score_ph(ph_val)

    temp_val = row.get("temperature", row.get("temp", np.nan))
    score += 1 * _score_temperature(temp_val)

    amm_val = row.get("ammonia", row.get("ammonia_mg_L", np.nan))
    score += 2 * _score_ammonia(amm_val)

    # ── Convert score to label ────────────────────────────────────────────
    # Max possible score = 3×2 + 2×2 + 1×2 + 1×2 + 2×2 = 6+4+2+2+4 = 18
    # Thresholds: 0–3 = Low, 4–7 = Moderate, ≥8 = High
    if score <= 3:
        return 0
    if score <= 7:
        return 1
    return 2


def label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies compute_risk_label to every row in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at minimum a 'DO' column.

    Returns
    -------
    pd.DataFrame
        Original df with added 'risk_label' (int) and 'risk_name' (str) columns.
    """
    df = df.copy()
    df["risk_label"] = df.apply(compute_risk_label, axis=1)
    label_map = {0: "Low", 1: "Moderate", 2: "High"}
    df["risk_name"] = df["risk_label"].map(label_map)
    return df


def print_label_distribution(df: pd.DataFrame):
    """Prints class distribution with percentages."""
    counts = df["risk_label"].value_counts().sort_index()
    total  = len(df)
    print("\n=== Risk Label Distribution ===")
    for label_id, count in counts.items():
        name = {0: "Low", 1: "Moderate", 2: "High"}[label_id]
        pct  = count / total * 100
        bar  = "#" * int(pct / 2)
        print(f"  {label_id} ({name:>8}): {count:>6} rows  {pct:5.1f}%  {bar}")
    print("=" * 52)


if __name__ == "__main__":
    # Quick self-test with synthetic data
    test_data = pd.DataFrame({
        "DO":          [6.5, 4.2, 2.0, 7.1, 1.5,  5.5],
        "pH":          [7.5, 8.6, 9.1, 7.2, 9.3,  7.8],
        "temperature": [28,  31,  33,  27,  35,   29],
        "turbidity":   [40,  120, 210, 35,  220,  65],
        "ammonia":     [0.2, 0.9, 2.1, 0.1, 2.5,  0.4],
    })

    labeled = label_dataframe(test_data)
    print(labeled[["DO", "turbidity", "pH", "risk_label", "risk_name"]])
    print_label_distribution(labeled)
