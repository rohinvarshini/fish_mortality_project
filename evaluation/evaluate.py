# evaluation/evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# Evaluates the BiLSTM forecaster and tabular RiskClassifier.
# Generates performance reports for both the local test split and the
# blind out-of-distribution (OOD) geographic holdout (Zenodo).
#
# Usage:
#   python evaluation/evaluate.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, mean_squared_error, mean_absolute_error,
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PROCESSED, CHECKPOINTS, DEVICE
from models.bilstm_branch import BiLSTMForecaster
from models.risk_classifier import RiskClassifier
from training.utils import load_checkpoint

LABEL_NAMES = ["Low", "Moderate", "High"]


def load_dataset(split="test"):
    """Loads a processed tabular split."""
    X    = np.load(os.path.join(DATA_PROCESSED, f"X_{split}.npy"))
    ydo  = np.load(os.path.join(DATA_PROCESSED, f"ydo_{split}.npy"))
    yr   = np.load(os.path.join(DATA_PROCESSED, f"yr_{split}.npy"))
    return X, ydo, yr


def evaluate_bilstm(bilstm, X, ydo, name="Test Set"):
    """Evaluates BiLSTM DO forecasting."""
    print(f"\n=== BiLSTM DO Forecaster ({name}) ===")
    bilstm.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        # Handle large arrays in chunks to prevent GPU OOM
        preds = []
        batch_size = 4096
        for i in range(0, len(X), batch_size):
            batch_x = X_tensor[i : i + batch_size]
            batch_pred = bilstm(batch_x).squeeze(1).cpu().numpy()
            preds.extend(batch_pred)
        preds = np.array(preds)

    rmse = np.sqrt(mean_squared_error(ydo, preds))
    mae  = mean_absolute_error(ydo, preds)

    print(f"  RMSE (standardised DO): {rmse:.4f}")
    print(f"  MAE  (standardised DO): {mae:.4f}")
    return preds


def evaluate_classifier(bilstm, classifier, X, yr, name="Test Set", save_suffix="test"):
    """Evaluates tabular risk classifier."""
    print(f"\n=== Risk Classifier ({name}) ===")
    bilstm.eval()
    classifier.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    all_preds, all_probs = [], []
    batch_size = 4096

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            wq_seq = X_tensor[i : i + batch_size]
            pred_do = bilstm(wq_seq)
            current_wq = wq_seq[:, -1, :]
            
            logits = classifier(current_wq, pred_do)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = (all_preds == yr).mean()
    f1  = f1_score(yr, all_preds, average="macro", zero_division=0)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")

    print("\n  Classification Report:")
    print(classification_report(yr, all_preds, target_names=LABEL_NAMES, zero_division=0))

    # Confusion matrix plot
    os.makedirs("evaluation", exist_ok=True)
    cm = confusion_matrix(yr, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
    )
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    out_path = os.path.join("evaluation", f"confusion_matrix_{save_suffix}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {out_path}")

    return all_preds, all_probs


def main():
    print("=" * 60)
    print("  Model Evaluation Report")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ── Load models ───────────────────────────────────────────────────────
    bilstm     = BiLSTMForecaster().to(DEVICE)
    classifier = RiskClassifier().to(DEVICE)

    try:
        load_checkpoint(bilstm, "bilstm_best", device=DEVICE)
    except FileNotFoundError:
        print("[!] Pretrained 'bilstm_best' checkpoint not found.")
        sys.exit(1)

    try:
        load_checkpoint(classifier, "classifier_best", device=DEVICE)
    except FileNotFoundError:
        print("[!] Pretrained 'classifier_best' checkpoint not found. Using random weights.")

    # ── 1. Evaluate on Local Test Split (Colombia + Nigeria) ──────────────
    try:
        X_test, ydo_test, yr_test = load_dataset("test")
        print(f"\nLocal Test set (Colombia + Nigeria): {len(X_test):,} samples")
        evaluate_bilstm(bilstm, X_test, ydo_test, name="Local Test Split")
        evaluate_classifier(bilstm, classifier, X_test, yr_test, name="Local Test Split", save_suffix="local_test")
    except FileNotFoundError:
        print("[!] Local test dataset not found.")

    # ── 2. Evaluate on Holdout Geographic Split (Zenodo) ──────────────────
    try:
        X_hold, ydo_hold, yr_hold = load_dataset("holdout")
        print(f"\nGeographic Holdout set (Zenodo): {len(X_hold):,} samples")
        evaluate_bilstm(bilstm, X_hold, ydo_hold, name="Zenodo OOD Holdout")
        evaluate_classifier(bilstm, classifier, X_hold, yr_hold, name="Zenodo OOD Holdout", save_suffix="zenodo_holdout")
    except FileNotFoundError:
        print("[!] Zenodo holdout dataset not found.")

    print("\n[v] Evaluation complete.")


if __name__ == "__main__":
    main()
