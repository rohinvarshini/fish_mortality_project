# training/train_bilstm.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Train the BiLSTM DO forecaster on tabular water quality data.
#
# Flow:
#   1. Load preprocessed numpy arrays from data/processed/
#   2. Build PyTorch Dataset + DataLoader
#   3. Train with MSE loss + gradient clipping + ReduceLROnPlateau
#   4. Early stopping on val RMSE
#   5. Save best checkpoint to checkpoints/bilstm_best.pt
#
# Usage:
#   python training/train_bilstm.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_PROCESSED, LOGS,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MAX_EPOCHS, PATIENCE, DEVICE, RANDOM_SEED,
)
from models.bilstm_branch import BiLSTMForecaster
from training.utils import (
    EarlyStopping, save_checkpoint, load_checkpoint,
    MetricsLogger, print_epoch,
)

torch.manual_seed(RANDOM_SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────
class WQSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping numpy (X, y_do) arrays.

    X   : (N, window_size, n_features)  — input WQ sequences
    y   : (N,)                           — target DO values (standardised)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataloaders() -> tuple:
    """Loads processed numpy arrays and wraps them in DataLoaders."""
    def _load(split):
        X   = np.load(os.path.join(DATA_PROCESSED, f"X_{split}.npy"))
        ydo = np.load(os.path.join(DATA_PROCESSED, f"ydo_{split}.npy"))
        return X, ydo

    try:
        X_train, ydo_train = _load("train")
        X_val,   ydo_val   = _load("val")
        X_test,  ydo_test  = _load("test")
    except FileNotFoundError as e:
        print(f"[!] Processed data not found: {e}")
        print("    Run: python data/preprocess_tabular.py")
        sys.exit(1)

    # CPU Subsampling to keep local baseline training fast (~1 min)
    if DEVICE == "cpu":
        print("  [Note] CPU detected. Subsampling dataset for fast baseline training...")
        np.random.seed(RANDOM_SEED)
        
        # Train split: 20k
        if len(X_train) > 20000:
            idx = np.random.choice(len(X_train), 20000, replace=False)
            X_train, ydo_train = X_train[idx], ydo_train[idx]
            
        # Val split: 5k
        if len(X_val) > 5000:
            idx = np.random.choice(len(X_val), 5000, replace=False)
            X_val, ydo_val = X_val[idx], ydo_val[idx]
            
        # Test split: 5k
        if len(X_test) > 5000:
            idx = np.random.choice(len(X_test), 5000, replace=False)
            X_test, ydo_test = X_test[idx], ydo_test[idx]

    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"  X shape: {X_train.shape}  (samples, window, features)")

    train_loader = DataLoader(
        WQSequenceDataset(X_train, ydo_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        WQSequenceDataset(X_val, ydo_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        WQSequenceDataset(X_test, ydo_test),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, phase="train"):
    """
    One training or evaluation epoch.
    Returns (avg_loss, RMSE, MAE).
    """
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss, total_samples = 0.0, 0
    all_preds, all_targets    = [], []

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if is_train:
                optimizer.zero_grad()

            pred = model(X_batch).squeeze(1)   # (batch,)
            loss = criterion(pred, y_batch)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss    += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)
            all_preds.extend(pred.detach().cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total_samples
    preds    = np.array(all_preds)
    targets  = np.array(all_targets)
    rmse     = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae      = float(np.mean(np.abs(preds - targets)))
    return avg_loss, rmse, mae


def main():
    print("=" * 60)
    print("  Phase 2 - Training BiLSTM DO Forecaster")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    train_loader, val_loader, test_loader = build_dataloaders()

    # ── Model ────────────────────────────────────────────────────────────
    model     = BiLSTMForecaster().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=PATIENCE, mode="min")   # monitor val RMSE
    logger     = MetricsLogger("bilstm_training", LOGS)
    best_rmse  = float("inf")

    print("\n" + "=" * 60)
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_rmse, tr_mae = run_epoch(
            model, train_loader, criterion, optimizer, DEVICE, "train"
        )
        va_loss, va_rmse, va_mae = run_epoch(
            model, val_loader, criterion, optimizer, DEVICE, "val"
        )
        scheduler.step(va_rmse)

        print_epoch(
            epoch, MAX_EPOCHS,
            {"loss": tr_loss, "rmse": tr_rmse, "mae": tr_mae},
            {"loss": va_loss, "rmse": va_rmse, "mae": va_mae},
        )
        logger.log("train", epoch, {"loss": tr_loss, "rmse": tr_rmse, "mae": tr_mae})
        logger.log("val",   epoch, {"loss": va_loss, "rmse": va_rmse, "mae": va_mae})

        # ── Save best checkpoint ──────────────────────────────────────────
        if va_rmse < best_rmse:
            best_rmse = va_rmse
            path = save_checkpoint(model, optimizer, epoch, va_rmse, "bilstm_best")
            print(f"  [ok] New best saved  val_rmse={va_rmse:.4f}  -> {path}")

        if early_stop(va_rmse):
            print(f"\n[EarlyStopping] Triggered at epoch {epoch}. "
                  f"Best val RMSE={best_rmse:.4f}")
            break

    # ── Test evaluation ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Loading best checkpoint for test evaluation...")
    load_checkpoint(model, "bilstm_best", device=DEVICE)

    te_loss, te_rmse, te_mae = run_epoch(
        model, test_loader, criterion, None, DEVICE, "val"
    )
    print(f"\n  TEST RESULTS (standardised DO units):")
    print(f"    Loss : {te_loss:.4f}")
    print(f"    RMSE : {te_rmse:.4f}")
    print(f"    MAE  : {te_mae:.4f}")
    print(f"\n[v] BiLSTM training complete. Checkpoint: checkpoints/bilstm_best.pt")


if __name__ == "__main__":
    main()
