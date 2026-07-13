# explainability/shap_explainer.py
# ─────────────────────────────────────────────────────────────────────────────
# SHAP explainability for the tabular RiskClassifier.
# Explains how each sensor reading (DO, pH, Temp, Turbidity, Ammonia) and the
# BiLSTM predicted DO contributes to the final High / Moderate / Low risk alert.
#
# Usage:
#   python explainability/shap_explainer.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PROCESSED, DEVICE, TABULAR_FEATURES, RANDOM_SEED
from models.bilstm_branch import BiLSTMForecaster
from models.risk_classifier import RiskClassifier
from training.utils import load_checkpoint

LABEL_NAMES  = ["Low Risk", "Moderate Risk", "High Risk"]
FEATURE_NAMES = TABULAR_FEATURES + ["predicted_DO_12h"]  # 6 features in total


def build_classifier_predictor(classifier, device):
    """
    Creates a predict function compatible with shap.KernelExplainer.
    Takes a 2D numpy array of shape (n_samples, 6) and returns
    softmax probabilities of shape (n_samples, 3).
    """
    def predict_fn(inputs: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
        current_wq  = tensor[:, :5]
        predicted_do = tensor[:, 5:]

        with torch.no_grad():
            logits = classifier(current_wq, predicted_do)
            probs  = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    return predict_fn


def get_classifier_inputs(bilstm, X_wq, device, n_samples=200):
    """
    Passes WQ sequences through BiLSTM to get predicted DO,
    and returns concatenated (current_wq, predicted_do) array of shape (n_samples, 6).
    """
    bilstm.eval()
    X_tensor = torch.tensor(X_wq[:n_samples], dtype=torch.float32).to(device)

    with torch.no_grad():
        predicted_do = bilstm(X_tensor)              # (n, 1)
        current_wq   = X_tensor[:, -1, :]             # (n, 5)

    inputs = torch.cat([current_wq, predicted_do], dim=1) # (n, 6)
    return inputs.cpu().numpy()


def run_shap_explanation(n_background=100, n_explain=10):
    """
    Computes SHAP values and outputs explainability plots.
    """
    print("=" * 60)
    print("  SHAP Explainability Analysis")
    print("=" * 60)

    # ── Load data and models ──────────────────────────────────────────────
    try:
        X_test = np.load(os.path.join(DATA_PROCESSED, "X_test.npy"))
        yr_test = np.load(os.path.join(DATA_PROCESSED, "yr_test.npy"))
    except FileNotFoundError:
        print("[!] Test data not found. Run preprocessing first.")
        sys.exit(1)

    bilstm     = BiLSTMForecaster().to(DEVICE)
    classifier = RiskClassifier().to(DEVICE)

    try:
        load_checkpoint(bilstm, "bilstm_best", device=DEVICE)
        load_checkpoint(classifier, "classifier_best", device=DEVICE)
    except FileNotFoundError as e:
        print(f"[!] Checkpoints not found: {e}. Run training scripts first.")
        sys.exit(1)

    # ── Prepare inputs ────────────────────────────────────────────────────
    print(f"\n[1/3] Generating inputs for SHAP explainer...")
    total_needed = n_background + n_explain
    inputs = get_classifier_inputs(bilstm, X_test, DEVICE, total_needed)

    background  = inputs[:n_background]    # (100, 6)
    explain_set = inputs[n_background:]    # (10, 6)
    explain_labels = yr_test[n_background:n_background + n_explain]

    # ── Build predictor and explainer ─────────────────────────────────────
    print("[2/3] Building SHAP KernelExplainer...")
    predict_fn = build_classifier_predictor(classifier, DEVICE)
    explainer  = shap.KernelExplainer(predict_fn, background)

    # ── Compute SHAP values ───────────────────────────────────────────────
    # shap_values is a list of 3 arrays (one per class), each (n_explain, 6)
    print("[3/3] Calculating SHAP values (this may take ~30 seconds)...")
    shap_values = explainer.shap_values(explain_set, nsamples=200)
    print("[ok] SHAP values computed successfully.")

    # ── Per-sample explanations ───────────────────────────────────────────
    print("\n=== Per-Sample Explanations (High Risk prediction) ===")
    for i in range(n_explain):
        true_label = LABEL_NAMES[explain_labels[i]]
        pred_class = np.argmax(predict_fn(explain_set[i:i+1])[0])
        pred_label = LABEL_NAMES[pred_class]

        print(f"\n  Sample {i+1} | True: {true_label} | Pred: {pred_label}")
        print(f"  {'Feature':<22} | {'Sensor Value':<12} | {'SHAP (High Risk Contribution)':<30}")
        print(f"  {'-'*70}")

        # Extract features and SHAP values for High Risk (class 2)
        feats = explain_set[i]
        sv    = shap_values[2][i]

        for idx, name in enumerate(FEATURE_NAMES):
            arrow = "▲" if sv[idx] > 0 else "▼"
            print(f"  {name:<22} | {feats[idx]:>12.4f} | {arrow} {sv[idx]:+10.4f}")

    # ── Global feature importance plot ────────────────────────────────────
    os.makedirs("explainability", exist_ok=True)

    # Mean absolute SHAP values for High Risk class across explain_set
    mean_abs_shap = np.abs(shap_values[2]).mean(axis=0)

    # Plot
    plt.figure(figsize=(8, 5))
    y_pos = np.arange(len(FEATURE_NAMES))
    
    # Sort features by importance
    sorted_idx = np.argsort(mean_abs_shap)
    plt.barh(y_pos, mean_abs_shap[sorted_idx], color="#f44336", height=0.6)
    plt.yticks(y_pos, [FEATURE_NAMES[i] for i in sorted_idx])
    plt.xlabel("Mean |SHAP Value| (Impact on High Risk prediction)")
    plt.title("Global Feature Importance (SHAP)\nFish Mortality Risk Classifier")
    plt.tight_layout()

    out_plot = os.path.join("explainability", "shap_global.png")
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"\n  Global importance chart saved -> {out_plot}")
    print("\n[v] SHAP explainability complete.")


if __name__ == "__main__":
    run_shap_explanation()
