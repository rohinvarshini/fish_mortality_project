# data/preprocess_tabular.py
# ─────────────────────────────────────────────────────────────────────────────
# Preprocesses tabular water quality datasets into sliding windows
# ready for BiLSTM training.
#
# Steps:
#   1. Load + clean T1 (Tilapia IoT) and T2 (Aquaponics 12-pond)
#   2. Harmonise column names across datasets
#   3. Clean physical/biological sensor outliers (clip inf / invalid numbers)
#   4. Apply composite risk labeling (see label_engine.py)
#   5. Create sliding windows: X=(window_size, n_features), y=DO scalar
#   6. Stratified train/val/test split
#   7. Standardise features (fit on train, apply to val/test)
#   8. Save processed arrays to data/processed/
#
# Also saves the V1 holdout (Zenodo) as a separate test set.
#
# Usage:
#   python data/preprocess_tabular.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TILAPIA_CSV, AQUAPONICS_CSV, ZENODO_CSV,
    DATA_PROCESSED, TABULAR_FEATURES, TARGET_FEATURE,
    WINDOW_SIZE, FORECAST_HORIZON,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_SEED,
)
from data.label_engine import label_dataframe, print_label_distribution


# ── Column Standardisation ───────────────────────────────────────────────────
def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises water quality column names to standard internal names:
    'DO', 'pH', 'temperature', 'turbidity', 'ammonia'.
    Handles case differences, unit suffixes, and spacing variations.
    Uses target tracking to prevent mapping multiple columns to the same name.
    """
    rename_map = {}
    for col in df.columns:
        c_clean = col.strip().lower()
        
        # 1. Dissolved Oxygen -> DO
        if "dissolved oxygen" in c_clean or "disolved oxygen" in c_clean or c_clean == "do":
            if "average" not in c_clean and "high" not in c_clean and "low" not in c_clean:
                rename_map[col] = "DO"
                
        # 2. pH
        elif c_clean == "ph":
            rename_map[col] = "pH"
            
        # 3. Temperature -> temperature
        elif "temperature" in c_clean or "temp" in c_clean:
            if "average" not in c_clean and "high" not in c_clean and "low" not in c_clean:
                rename_map[col] = "temperature"
                
        # 4. Turbidity -> turbidity
        elif "turbidity" in c_clean or "turbidez" in c_clean:
            if "average" not in c_clean and "high" not in c_clean and "low" not in c_clean:
                rename_map[col] = "turbidity"
                
        # 5. Ammonia -> ammonia
        elif "ammonia" in c_clean or "amonio" in c_clean:
            rename_map[col] = "ammonia"

    # Prevent duplicate target mappings by keeping only the first match
    seen_targets = set()
    final_rename = {}
    for src, dst in rename_map.items():
        if dst not in seen_targets:
            final_rename[src] = dst
            seen_targets.add(dst)

    df.rename(columns=final_rename, inplace=True)
    return df


def load_and_clean_tilapia(path: str) -> pd.DataFrame:
    """Load T1: IoT Tilapia dataset."""
    print(f"\n[1/3] Loading T1 (Tilapia IoT): {path}")
    df = pd.read_csv(path)
    df = standardise_columns(df)

    # Parse timestamp
    ts_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    if ts_col:
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
        df.sort_values("timestamp", inplace=True)

    df = df.reset_index(drop=True)
    print(f"   Rows: {len(df):,}   Columns: {list(df.columns)}")
    return df


def load_and_clean_aquaponics(path: str) -> pd.DataFrame:
    """
    Load T2: Aquaponics 12-Pond dataset.
    Handles a folder of multiple CSVs (one per pond).
    """
    print(f"\n[2/3] Loading T2 (Aquaponics folder): {path}")

    if os.path.isdir(path):
        csv_files = sorted([f for f in os.listdir(path) if f.lower().endswith(".csv")])
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {path}")

        print(f"   Found {len(csv_files)} pond CSV files:")
        frames = []
        for fname in csv_files:
            fpath  = os.path.join(path, fname)
            pond   = os.path.splitext(fname)[0]   # e.g. "IoTPond1"
            df_tmp = pd.read_csv(fpath)
            df_tmp = standardise_columns(df_tmp)   # Standardise per file first!
            df_tmp["pond_id"] = pond               # Tag every row with pond name
            frames.append(df_tmp)
            print(f"     {pond:<15}  {len(df_tmp):>6} rows  standardised_cols={list(df_tmp.columns[:5])}")

        df = pd.concat(frames, ignore_index=True)
        print(f"   Total merged rows: {len(df):,}")
    else:
        df = pd.read_csv(path)
        df = standardise_columns(df)

    ts_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    if ts_col:
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
        df.sort_values("timestamp", inplace=True)

    df = df.reset_index(drop=True)
    print(f"   Rows: {len(df):,}   Columns: {list(df.columns)}")
    return df


def load_and_clean_zenodo(path: str) -> pd.DataFrame:
    """Load V1: Zenodo holdout dataset."""
    print(f"\n[3/3] Loading V1 (Zenodo holdout): {path}")
    df = pd.read_csv(path)
    df = standardise_columns(df)

    df = df.reset_index(drop=True)
    print(f"   Rows: {len(df):,}   Columns: {list(df.columns)}")
    return df


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures all TABULAR_FEATURES columns exist; fills missing with NaN."""
    for feat in TABULAR_FEATURES:
        if feat not in df.columns:
            df[feat] = np.nan
    return df


def clean_biological_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans physically and biologically impossible sensor readings:
      - Replaces inf/invalid values with NaN
      - Replaces out-of-bounds outliers with NaN
      - These NaNs are subsequently filled by ffill/bfill/median.
    """
    df = df.copy()
    
    # 1. DO: clip to [0, 25]
    df.loc[(df["DO"] < 0.0) | (df["DO"] > 25.0), "DO"] = np.nan
    
    # 2. pH: clip to [0, 14]
    df.loc[(df["pH"] < 0.0) | (df["pH"] > 14.0), "pH"] = np.nan
    
    # 3. Temperature: clip to [0, 45] (DS18B20 unplugged error is typically -127)
    df.loc[(df["temperature"] < 0.0) | (df["temperature"] > 45.0), "temperature"] = np.nan
    
    # 4. Turbidity: clip to [0, 500] (negative turbidity is impossible)
    df.loc[(df["turbidity"] < 0.0) | (df["turbidity"] > 500.0), "turbidity"] = np.nan
    
    # 5. Ammonia: replace inf and clip to [0, 20]
    df["ammonia"] = pd.to_numeric(df["ammonia"], errors="coerce")
    df.loc[np.isinf(df["ammonia"]), "ammonia"] = np.nan
    df.loc[(df["ammonia"] < 0.0) | (df["ammonia"] > 20.0), "ammonia"] = np.nan
    
    return df


def forward_fill_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans missing values: forward fill, then backward fill, then median."""
    df[TABULAR_FEATURES] = (
        df[TABULAR_FEATURES]
        .ffill()
        .bfill()
        .fillna(df[TABULAR_FEATURES].median())
    )
    return df


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    horizon: int = FORECAST_HORIZON,
) -> tuple:
    """
    Converts a time-series DataFrame into sliding window arrays.

    If a 'pond_id' column exists, windows are created PER POND so
    they never span across two different ponds (no data leakage).

    Returns
    -------
    X      : np.ndarray  shape (N, window_size, n_features)
    y_do   : np.ndarray  shape (N,)
    y_risk : np.ndarray  shape (N,)
    """
    target_idx = TABULAR_FEATURES.index(TARGET_FEATURE)
    X, y_do, y_risk = [], [], []

    # If pond_id exists, window per pond to avoid cross-pond contamination
    groups = df.groupby("pond_id") if "pond_id" in df.columns else [("all", df)]

    for pond_name, group in groups:
        group = group.reset_index(drop=True)
        vals  = group[TABULAR_FEATURES].values   # (T, F)
        risks = group["risk_label"].values        # (T,)
        n     = len(group)

        if n < window_size + horizon + 1:
            print(f"     [skip] pond '{pond_name}': only {n} rows "
                  f"(need >{window_size + horizon})")
            continue

        for i in range(n - window_size - horizon):
            X.append(vals[i : i + window_size])
            fi = i + window_size + horizon
            y_do.append(vals[fi, target_idx])
            y_risk.append(risks[fi])

    total = len(X)
    print(f"   Sliding windows created: {total:,}")
    return (
        np.array(X,      dtype=np.float32),
        np.array(y_do,   dtype=np.float32),
        np.array(y_risk, dtype=np.int64),
    )


def main():
    os.makedirs(DATA_PROCESSED, exist_ok=True)

    # ── Load datasets ─────────────────────────────────────────────────────
    frames = []

    if os.path.exists(TILAPIA_CSV):
        t1 = load_and_clean_tilapia(TILAPIA_CSV)
        t1 = ensure_features(t1)
        frames.append(("T1", t1))
    else:
        print(f"[!] T1 not found at {TILAPIA_CSV}. Run download_datasets.py first.")

    if os.path.exists(AQUAPONICS_CSV):
        t2 = load_and_clean_aquaponics(AQUAPONICS_CSV)
        t2 = ensure_features(t2)
        frames.append(("T2", t2))
    else:
        print(f"[!] T2 not found at {AQUAPONICS_CSV}.")

    if not frames:
        print("[✗] No datasets found. Cannot proceed.")
        sys.exit(1)

    # == Merge and label ===================================================
    print("\n=== Merging datasets ===")
    combined = pd.concat([f[1] for f in frames], ignore_index=True)
    print(f"   Combined rows: {len(combined):,}")

    # Clean sensor malfunctions/outliers
    combined = clean_biological_outliers(combined)

    combined = forward_fill_and_interpolate(combined)
    combined = label_dataframe(combined)
    print_label_distribution(combined)

    # == Create sliding windows ============================================
    print(f"\n=== Creating sliding windows (size={WINDOW_SIZE}, horizon={FORECAST_HORIZON}) ===")
    X, y_do, y_risk = create_sliding_windows(combined)
    print(f"   X shape:      {X.shape}")
    print(f"   y_do shape:   {y_do.shape}")
    print(f"   y_risk shape: {y_risk.shape}")

    # == Train / Val / Test split =========================================
    print("\n=== Train / Val / Test split ===")
    indices = np.arange(len(X))
    idx_train, idx_temp, _, y_temp = train_test_split(
        indices, y_risk, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=y_risk, random_state=RANDOM_SEED
    )
    val_frac = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=(1 - val_frac),
        stratify=y_temp, random_state=RANDOM_SEED
    )

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    ydo_train, ydo_val, ydo_test = y_do[idx_train], y_do[idx_val], y_do[idx_test]
    yr_train, yr_val, yr_test    = y_risk[idx_train], y_risk[idx_val], y_risk[idx_test]

    print(f"   Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # == Standardise features (fit on train only) ==========================
    print("\n=== Standardising features ===")
    scaler = StandardScaler()
    # Reshape to (N*T, F), fit, reshape back
    n_train, T, F = X_train.shape
    X_train_2d = X_train.reshape(-1, F)
    scaler.fit(X_train_2d)

    X_train = scaler.transform(X_train.reshape(-1, F)).reshape(n_train, T, F)
    X_val   = scaler.transform(X_val.reshape(-1, F)).reshape(-1, T, F)
    X_test  = scaler.transform(X_test.reshape(-1, F)).reshape(-1, T, F)

    # Also standardise DO targets using only the DO column's scale
    do_mean = scaler.mean_[TABULAR_FEATURES.index(TARGET_FEATURE)]
    do_std  = scaler.scale_[TABULAR_FEATURES.index(TARGET_FEATURE)]
    ydo_train = (ydo_train - do_mean) / do_std
    ydo_val   = (ydo_val   - do_mean) / do_std
    ydo_test  = (ydo_test  - do_mean) / do_std

    # == Save ==============================================================
    print("\n=== Saving processed data ===")
    np.save(os.path.join(DATA_PROCESSED, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_PROCESSED, "X_val.npy"),   X_val)
    np.save(os.path.join(DATA_PROCESSED, "X_test.npy"),  X_test)
    np.save(os.path.join(DATA_PROCESSED, "ydo_train.npy"), ydo_train)
    np.save(os.path.join(DATA_PROCESSED, "ydo_val.npy"),   ydo_val)
    np.save(os.path.join(DATA_PROCESSED, "ydo_test.npy"),  ydo_test)
    np.save(os.path.join(DATA_PROCESSED, "yr_train.npy"),  yr_train)
    np.save(os.path.join(DATA_PROCESSED, "yr_val.npy"),    yr_val)
    np.save(os.path.join(DATA_PROCESSED, "yr_test.npy"),   yr_test)

    with open(os.path.join(DATA_PROCESSED, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"   Saved arrays + scaler to: {DATA_PROCESSED}")

    # == Process V1 holdout (Zenodo) =======================================
    if os.path.exists(ZENODO_CSV):
        print("\n=== Processing V1 holdout (Zenodo) ===")
        v1 = load_and_clean_zenodo(ZENODO_CSV)
        v1 = ensure_features(v1)
        v1 = clean_biological_outliers(v1)
        v1 = forward_fill_and_interpolate(v1)
        v1 = label_dataframe(v1)
        X_hold, ydo_hold, yr_hold = create_sliding_windows(v1)

        n_h, T_h, F_h = X_hold.shape
        X_hold = scaler.transform(X_hold.reshape(-1, F_h)).reshape(n_h, T_h, F_h)
        ydo_hold = (ydo_hold - do_mean) / do_std

        np.save(os.path.join(DATA_PROCESSED, "X_holdout.npy"),   X_hold)
        np.save(os.path.join(DATA_PROCESSED, "ydo_holdout.npy"), ydo_hold)
        np.save(os.path.join(DATA_PROCESSED, "yr_holdout.npy"),  yr_hold)
        print(f"   Holdout saved: {X_hold.shape}")

    print("\n[v] Tabular preprocessing complete.")


if __name__ == "__main__":
    main()
