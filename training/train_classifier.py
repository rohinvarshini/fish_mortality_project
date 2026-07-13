# training/train_classifier.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Train the Tabular Risk Classifier.
#
# Strategy:
#   - Load pretrained BiLSTM from checkpoints/bilstm_best.pt → FROZEN
#   - Train the Classifier MLP head on current WQ features + forecasted DO
#   - Loss: weighted CrossEntropy (handles class imbalance)
#   - Metrics: accuracy, macro-F1
#
# Usage:
#   python training/train_classifier.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_PROCESSED, LOGS, CHECKPOINTS,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MAX_EPOCHS, PATIENCE, DEVICE, RANDOM_SEED,
    NUM_RISK_CLASSES,
)
from models.bilstm_branch import BiLSTMForecaster
from models.risk_classifier import RiskClassifier
from training.utils import (
    EarlyStopping, save_checkpoint, load_checkpoint,
    get_class_weights, MetricsLogger, print_epoch,
)

torch.manual_seed(RANDOM_SEED)


# ── Tabular Classifier Dataset ────────────────────────────────────────────────
class TabularClassifierDataset(Dataset):
    """
    Yields (wq_sequence, risk_label) per sample.
    """

    def __init__(self, X_wq: np.ndarray, y_risk: np.ndarray):
        self.X_wq   = torch.tensor(X_wq,   dtype=torch.float32)
        self.y_risk = torch.tensor(y_risk,  dtype=torch.long)

    def __len__(self):
        return len(self.X_wq)

    def __getitem__(self, idx):
        return self.X_wq[idx], self.y_risk[idx]


def build_dataloaders() -> tuple:
    """Builds train/val/test DataLoaders for classifier training."""
    def _load(split):
        X  = np.load(os.path.join(DATA_PROCESSED, f"X_{split}.npy"))
        yr = np.load(os.path.join(DATA_PROCESSED, f"yr_{split}.npy"))
        return X, yr

    try:
        X_train, yr_train = _load("train")
        X_val,   yr_val   = _load("val")
        X_test,  yr_test  = _load("test")
    except FileNotFoundError as e:
        print(f"[!] Processed data not found: {e}")
        print("    Run: python data/preprocess_tabular.py")
        sys.exit(1)

    train_ds = TabularClassifierDataset(X_train, yr_train)
    val_ds   = TabularClassifierDataset(X_val, yr_val)
    test_ds  = TabularClassifierDataset(X_test, yr_test)

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, yr_train


def load_pretrained_bilstm(device: str) -> BiLSTMForecaster:
    """Loads BiLSTM from best checkpoint, then freezes it."""
    bilstm = BiLSTMForecaster().to(device)

    try:
        load_checkpoint(bilstm, "bilstm_best", device=device)
    except FileNotFoundError as e:
        print(f"[!] Pretrained BiLSTM model not found: {e}")
        print("    Run: python training/train_bilstm.py first.")
        sys.exit(1)

    for p in bilstm.parameters():
        p.requires_grad = False

    bilstm.eval()
    print("[ok] Pretrained BiLSTM loaded and frozen.")
    return bilstm


def run_epoch(bilstm, classifier, loader, criterion, optimizer, device, phase="train"):
    """One training or evaluation epoch."""
    is_train = phase == "train"
    classifier.train() if is_train else classifier.eval()
    bilstm.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels      = [], []

    with torch.set_grad_enabled(is_train):
        for wq_seq, labels in loader:
            wq_seq  = wq_seq.to(device)
            labels  = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            # Pass through pre-trained BiLSTM to get forecasted DO -> (B, 1)
            with torch.no_grad():
                pred_do = bilstm(wq_seq)

            # Latest timestep features -> (B, 5)
            current_wq = wq_seq[:, -1, :]

            # Forward pass through classifier
            logits = classifier(current_wq, pred_do)
            loss   = criterion(logits, labels)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * wq_seq.size(0)
            correct    += (preds == labels).sum().item()
            total      += wq_seq.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1, all_preds, all_labels


def main():
    print("=" * 60)
    print("  Phase 3 - Training Tabular Risk Classifier")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    train_loader, val_loader, test_loader, yr_train = build_dataloaders()

    # ── Load frozen pretrained BiLSTM ─────────────────────────────────────
    bilstm = load_pretrained_bilstm(DEVICE)

    # ── Classifier ────────────────────────────────────────────────────────
    classifier = RiskClassifier().to(DEVICE)

    # ── Loss with class weighting ─────────────────────────────────────────
    class_weights = get_class_weights(yr_train, NUM_RISK_CLASSES, DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=LEARNING_RATE * 2, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=PATIENCE, mode="max")
    logger     = MetricsLogger("classifier_training", LOGS)
    best_f1    = 0.0

    print("\n" + "=" * 60)
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(
            bilstm, classifier, train_loader, criterion, optimizer, DEVICE, "train"
        )
        va_loss, va_acc, va_f1, _, _ = run_epoch(
            bilstm, classifier, val_loader, criterion, optimizer, DEVICE, "val"
        )
        scheduler.step()

        print_epoch(
            epoch, MAX_EPOCHS,
            {"loss": tr_loss, "acc": tr_acc, "f1": tr_f1},
            {"loss": va_loss, "acc": va_acc, "f1": va_f1},
        )
        logger.log("train", epoch, {"loss": tr_loss, "acc": tr_acc, "f1": tr_f1})
        logger.log("val",   epoch, {"loss": va_loss, "acc": va_acc, "f1": va_f1})

        if va_f1 > best_f1:
            best_f1 = va_f1
            path = save_checkpoint(classifier, optimizer, epoch, va_f1, "classifier_best")
            print(f"  [ok] New best saved  val_f1={va_f1:.4f}  -> {path}")

        if early_stop(va_f1):
            print(f"\n[EarlyStopping] Triggered at epoch {epoch}. Best val F1={best_f1:.4f}")
            break

    # ── Final test evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    load_checkpoint(classifier, "classifier_best", device=DEVICE)

    _, te_acc, te_f1, te_preds, te_labels = run_epoch(
        bilstm, classifier, test_loader, criterion, None, DEVICE, "val"
    )
    print(f"\n  TEST RESULTS:")
    print(f"    Accuracy  : {te_acc:.4f}")
    print(f"    Macro F1  : {te_f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        te_labels, te_preds,
        target_names=["Low", "Moderate", "High"],
        zero_division=0,
    ))
    print(f"\n[v] Classifier training complete. Checkpoint: checkpoints/classifier_best.pt")


if __name__ == "__main__":
    main()
