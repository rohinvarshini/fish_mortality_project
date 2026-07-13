# training/utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared training utilities: EarlyStopping, checkpoint save/load,
# class weight computation, metrics logging.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINTS


class EarlyStopping:
    """
    Stops training when the monitored metric has not improved for
    `patience` epochs.

    Parameters
    ----------
    patience  : int   — epochs to wait before stopping
    mode      : str   — 'min' (loss) or 'max' (accuracy/F1)
    delta     : float — minimum change to qualify as improvement
    """

    def __init__(self, patience: int = 10, mode: str = "min", delta: float = 1e-4):
        self.patience  = patience
        self.mode      = mode
        self.delta     = delta
        self.counter   = 0
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.triggered = False

    def __call__(self, metric: float) -> bool:
        """Returns True if training should stop."""
        improved = (
            metric < self.best - self.delta
            if self.mode == "min"
            else metric > self.best + self.delta
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered

    def reset(self):
        self.counter   = 0
        self.best      = float("inf") if self.mode == "min" else float("-inf")
        self.triggered = False


def save_checkpoint(
    model:      torch.nn.Module,
    optimizer:  torch.optim.Optimizer,
    epoch:      int,
    metric:     float,
    name:       str,
    extra:      dict = None,
):
    """Saves model + optimizer state to checkpoints/<name>.pt"""
    os.makedirs(CHECKPOINTS, exist_ok=True)
    path = os.path.join(CHECKPOINTS, f"{name}.pt")
    payload = {
        "epoch":      epoch,
        "metric":     metric,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(
    model:     torch.nn.Module,
    name:      str,
    optimizer: torch.optim.Optimizer = None,
    device:    str = "cpu",
) -> dict:
    """Loads checkpoint from checkpoints/<name>.pt into model (in-place)."""
    path = os.path.join(CHECKPOINTS, f"{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[v] Loaded checkpoint '{name}' (epoch {ckpt.get('epoch','?')}, "
          f"metric={ckpt.get('metric', '?'):.4f})")
    return ckpt


def get_class_weights(labels: np.ndarray, num_classes: int, device: str) -> torch.Tensor:
    """
    Computes inverse-frequency class weights to handle class imbalance.
    These are passed as `weight=` to CrossEntropyLoss.
    """
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    print(f"[ClassWeights] {dict(zip(['Low','Moderate','High'], weights.round(3)))}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


class MetricsLogger:
    """Logs training/val metrics per epoch and saves to logs/<name>.json."""

    def __init__(self, name: str, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.path    = os.path.join(log_dir, f"{name}.json")
        self.history = {"train": [], "val": []}

    def log(self, phase: str, epoch: int, metrics: dict):
        entry = {"epoch": epoch, **metrics}
        self.history[phase].append(entry)
        with open(self.path, "w") as f:
            json.dump(self.history, f, indent=2)

    def get(self, phase: str, key: str) -> list:
        return [e[key] for e in self.history[phase] if key in e]


def print_epoch(epoch: int, max_epoch: int, train_metrics: dict, val_metrics: dict):
    """Compact one-line epoch log."""
    train_str = "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
    val_str   = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
    print(f"Epoch [{epoch:>3}/{max_epoch}]  Train: {train_str}  |  Val: {val_str}")
