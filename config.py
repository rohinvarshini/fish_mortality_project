# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration file for the Fish Mortality Prediction project.
# Edit values here; all modules import from this file.
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_RAW       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINTS    = os.path.join(BASE_DIR, "checkpoints")
LOGS           = os.path.join(BASE_DIR, "logs")

# ── Dataset filenames (after download) ───────────────────────────────────────
TILAPIA_CSV    = os.path.join(DATA_RAW, "tilapia_iot.csv")   # Dataset T1 — single CSV
AQUAPONICS_DIR = os.path.join(DATA_RAW, "aquaponics")        # Dataset T2 — folder of 11 CSVs
AQUAPONICS_CSV = AQUAPONICS_DIR                              # alias used by loader
ZENODO_CSV     = os.path.join(DATA_RAW, "zenodo_fishpond.csv")   # Dataset V1 (holdout)

# ── Features ──────────────────────────────────────────────────────────────────
# Tabular features used by BiLSTM (order matters — must match CSV columns)
TABULAR_FEATURES = ["DO", "pH", "temperature", "turbidity", "ammonia"]
TARGET_FEATURE   = "DO"          # What BiLSTM forecasts
N_FEATURES       = len(TABULAR_FEATURES)

# ── Labeling thresholds (scientifically validated — see labeling_strategy.md) ─
DO_HIGH_THRESHOLD  = 3.0    # mg/L — below this = always High Risk (DO override)
DO_MOD_THRESHOLD   = 5.0    # mg/L — below this = at least Moderate Risk
TURB_HIGH_NTU      = 150    # NTU
TURB_MOD_NTU       = 50     # NTU
PH_HIGH_UPPER      = 9.0
PH_MOD_UPPER       = 8.5
PH_HIGH_LOWER      = 6.0
PH_MOD_LOWER       = 6.5
TEMP_HIGH          = 35.0   # °C
TEMP_MOD           = 32.0   # °C
AMMONIA_HIGH       = 2.0    # mg/L TAN
AMMONIA_MOD        = 0.5    # mg/L TAN

RISK_LABELS        = {0: "Low", 1: "Moderate", 2: "High"}

# ── Time-Series Windowing ─────────────────────────────────────────────────────
WINDOW_SIZE   = 24   # hours of history fed to BiLSTM
FORECAST_HORIZON = 12  # hours ahead to predict DO

# ── BiLSTM ────────────────────────────────────────────────────────────────────
BILSTM_HIDDEN  = 128
BILSTM_LAYERS  = 2
BILSTM_DROPOUT = 0.3

# ── Classifier ────────────────────────────────────────────────────────────────
CLASSIFIER_HIDDEN  = 64
CLASSIFIER_DROPOUT = 0.4
NUM_RISK_CLASSES = 3  # Low / Moderate / High

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
MAX_EPOCHS     = 60
PATIENCE       = 10          # early stopping patience

# ── Train/Val/Test split ──────────────────────────────────────────────────────
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Device ───────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
