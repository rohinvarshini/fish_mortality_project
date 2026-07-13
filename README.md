# Fish Mortality Prediction 🐟

A sensor-aware, image + time-series fusion deep learning system that predicts fish mortality risk from algal blooms in aquaculture ponds — with full SHAP explainability.

---

## Problem
Algal blooms silently drain oxygen from ponds overnight, killing entire fish stocks before farmers notice. This system predicts **High / Moderate / Low mortality risk 12 hours in advance** by combining pond photos with water quality sensor readings.

---

## Pipeline

```
Pond Photo ──► CNN (MobileNetV2) ──► Algae Severity Embedding
                                              │
24-hr WQ Readings ──► BiLSTM ──► Predicted DO (12hr ahead)
                                              │
                          ┌───────────────────┘
                          ▼
                   Fusion Classifier ──► 🔴 High / 🟡 Moderate / 🟢 Low
                          │
                          ▼
                   SHAP Explainer ──► "Why this alert?"
```

---

## Datasets

| # | Dataset | Source | Role |
|---|---------|--------|------|
| T1 | IoT Monitoring — Tilapia | [Kaggle](https://www.kaggle.com/datasets/anibalpolanco/iot-monitoring-of-water-quality-and-tilapia) | Primary tabular (has real survival %) |
| T2 | Aquaponics 12-Pond | [Kaggle](https://www.kaggle.com/datasets/blessingogbuokiri/sensor-based-aquaponics-fish-pond-datasets) | Supplementary tabular |
| I1 | NASA Tick Tick Bloom | [GitHub](https://github.com/IoannisNasios/HarmfulAlgalBloomDetection) | Bloom severity images |
| V1 | Zenodo Fishpond WQ | [Zenodo](https://zenodo.org/record/19210095) | Geographic holdout |

---

## Project Structure

```
fish_mortality_prediction/
├── config.py                   ← All hyperparameters and paths
├── run_pipeline.py             ← Single entry point
├── requirements.txt
│
├── data/
│   ├── download_datasets.py    ← Downloads all datasets
│   ├── preprocess_tabular.py   ← Sliding windows, labeling, splitting
│   ├── preprocess_images.py    ← Image resize, augment, split by class
│   └── label_engine.py         ← Composite risk labeling (scientifically validated)
│
├── models/
│   ├── cnn_branch.py           ← MobileNetV2 + embedding head
│   ├── bilstm_branch.py        ← Bidirectional LSTM DO forecaster
│   └── fusion_model.py         ← Late-fusion risk classifier
│
├── training/
│   ├── train_cnn.py            ← Phase 1: train CNN
│   ├── train_bilstm.py         ← Phase 2: train BiLSTM
│   ├── train_fusion.py         ← Phase 3: train fusion head (frozen branches)
│   └── utils.py                ← EarlyStopping, checkpointing, class weights
│
├── evaluation/
│   └── evaluate.py             ← Metrics + confusion matrix
│
├── explainability/
│   └── shap_explainer.py       ← SHAP KernelExplainer + importance plots
│
├── checkpoints/                ← Saved model weights (.pt)
└── logs/                       ← Training history (.json)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Kaggle API
Place your `kaggle.json` token at `~/.kaggle/kaggle.json`  
(Download from: https://www.kaggle.com/settings → API → Create New Token)

### 3. Run everything
```bash
python run_pipeline.py --all
```

### Or step by step
```bash
python run_pipeline.py --download      # Download datasets
python run_pipeline.py --preprocess    # Clean + window + label data
python run_pipeline.py --train         # Train CNN → BiLSTM → Fusion
python run_pipeline.py --evaluate      # Metrics + SHAP plots
```

---

## Labeling Strategy (Scientific Basis)

Risk labels use a **composite weighted scoring system** — not a single threshold.

| Parameter | Weight | 🟢 Low | 🟡 Moderate | 🔴 High |
|-----------|--------|--------|------------|--------|
| DO (mg/L) | 3× | > 5.0 | 3.0 – 5.0 | < 3.0 |
| Turbidity (NTU) | 2× | < 50 | 50 – 150 | > 150 |
| pH | 1× | 6.5 – 8.5 | 8.5 – 9.0 | > 9.0 |
| Temperature (°C) | 1× | 26 – 32 | 32 – 35 | > 35 |
| Ammonia (mg/L) | 2× | < 0.5 | 0.5 – 2.0 | > 2.0 |

**DO Override**: if DO < 3.0 mg/L → always High Risk (FAO validated).

---

## Tech Stack

| Role | Tool |
|------|------|
| Deep Learning | PyTorch 2.0+ |
| CNN Backbone | MobileNetV2 (torchvision, pretrained) |
| Time-Series | BiLSTM (custom PyTorch) |
| Explainability | SHAP 0.44+ |
| Data | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |

---

## Expected Results (Target)

| Model | Metric | Target |
|-------|--------|--------|
| BiLSTM | Val RMSE (std. DO) | < 0.5 |
| CNN | Val Macro F1 | > 0.75 |
| Fusion | Test Macro F1 | > 0.72 |
