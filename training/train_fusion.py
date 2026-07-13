# training/train_fusion.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Train the Fusion Risk Classifier.
#
# Strategy:
#   - Load pretrained CNN  from checkpoints/cnn_best.pt   → FROZEN
#   - Load pretrained BiLSTM from checkpoints/bilstm_best.pt → FROZEN
#   - Only the fusion MLP head is trained
#   - Loss: weighted CrossEntropy (handles class imbalance)
#   - Metrics: accuracy, macro-F1, per-class F1
#
# Usage:
#   python training/train_fusion.py
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
    NUM_RISK_CLASSES, IMAGE_SIZE, WINDOW_SIZE, N_FEATURES,
)
from models.cnn_branch import AlgaeCNN
from models.bilstm_branch import BiLSTMForecaster
from models.fusion_model import FusionRiskClassifier
from training.utils import (
    EarlyStopping, save_checkpoint, load_checkpoint,
    get_class_weights, MetricsLogger, print_epoch,
)

torch.manual_seed(RANDOM_SEED)


# ── Fusion Dataset ────────────────────────────────────────────────────────────
class FusionDataset(Dataset):
    """
    Yields (image_tensor, wq_sequence, risk_label) per sample.

    Since our datasets provide images and tabular data from different sources,
    we use a paired synthetic pairing strategy:
      - For each WQ window, we find the image whose bloom label best matches
        the turbidity-derived severity at that window's endpoint.
      - If no image data is available, we generate a placeholder embedding.

    In practice, replace this with your actual paired loader once you have
    co-located pond image + sensor data.
    """

    def __init__(
        self,
        X_wq:       np.ndarray,   # (N, window, features)
        y_risk:     np.ndarray,   # (N,) risk labels
        image_dir:  str  = None,  # path to processed images/split/
        transform        = None,
        use_real_images: bool = False,
    ):
        self.X_wq      = torch.tensor(X_wq,    dtype=torch.float32)
        self.y_risk    = torch.tensor(y_risk,   dtype=torch.long)
        self.image_dir = image_dir
        self.transform = transform
        self.use_real  = use_real_images and image_dir is not None

        if self.use_real:
            self._load_image_paths()
        else:
            # Generate fixed random image tensors as placeholders
            # (replace with real pairing when images are available)
            torch.manual_seed(RANDOM_SEED)
            self.image_tensors = torch.randn(
                len(X_wq), 3, IMAGE_SIZE, IMAGE_SIZE
            )

    def _load_image_paths(self):
        """Load image paths from processed image folder by class label."""
        from PIL import Image
        class_names = ["Low", "Moderate", "High"]
        self.images_by_class = {i: [] for i in range(3)}

        for class_id, cls in enumerate(class_names):
            folder = os.path.join(self.image_dir, cls)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.images_by_class[class_id].append(
                        os.path.join(folder, fname)
                    )

        total = sum(len(v) for v in self.images_by_class.values())
        print(f"  [FusionDataset] Real images loaded: {total}")

    def _get_image(self, idx: int) -> torch.Tensor:
        if not self.use_real:
            return self.image_tensors[idx]

        from PIL import Image
        label = self.y_risk[idx].item()
        paths = self.images_by_class.get(label, [])
        if not paths:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

        # Cycle through images for that class
        path = paths[idx % len(paths)]
        img  = Image.open(path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

    def __len__(self):
        return len(self.X_wq)

    def __getitem__(self, idx):
        return self._get_image(idx), self.X_wq[idx], self.y_risk[idx]


def build_dataloaders(use_real_images: bool = False) -> tuple:
    """Builds train/val/test DataLoaders for fusion training."""
    from data.preprocess_images import get_transforms

    def _load(split):
        X   = np.load(os.path.join(DATA_PROCESSED, f"X_{split}.npy"))
        yr  = np.load(os.path.join(DATA_PROCESSED, f"yr_{split}.npy"))
        return X, yr

    try:
        X_train, yr_train = _load("train")
        X_val,   yr_val   = _load("val")
        X_test,  yr_test  = _load("test")
    except FileNotFoundError as e:
        print(f"[!] Processed data not found: {e}")
        print("    Run: python data/preprocess_tabular.py")
        sys.exit(1)

    img_train_dir = os.path.join(DATA_PROCESSED, "images", "train")
    img_val_dir   = os.path.join(DATA_PROCESSED, "images", "val")
    img_test_dir  = os.path.join(DATA_PROCESSED, "images", "test")

    train_ds = FusionDataset(
        X_train, yr_train,
        image_dir=img_train_dir if use_real_images else None,
        transform=get_transforms("train"),
        use_real_images=use_real_images,
    )
    val_ds = FusionDataset(
        X_val, yr_val,
        image_dir=img_val_dir if use_real_images else None,
        transform=get_transforms("val"),
        use_real_images=use_real_images,
    )
    test_ds = FusionDataset(
        X_test, yr_test,
        image_dir=img_test_dir if use_real_images else None,
        transform=get_transforms("test"),
        use_real_images=use_real_images,
    )

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, yr_train


def load_pretrained_models(device: str) -> tuple:
    """Loads CNN and BiLSTM from best checkpoints, then freezes both."""
    cnn    = AlgaeCNN().to(device)
    bilstm = BiLSTMForecaster().to(device)

    try:
        load_checkpoint(cnn,    "cnn_best",    device=device)
        load_checkpoint(bilstm, "bilstm_best", device=device)
    except FileNotFoundError as e:
        print(f"[!] Pretrained model not found: {e}")
        print("    Run: python training/train_cnn.py && python training/train_bilstm.py")
        print("    Continuing with random weights for demonstration...")

    cnn.freeze_all()
    for p in bilstm.parameters():
        p.requires_grad = False

    cnn.eval()
    bilstm.eval()
    print("[✓] CNN + BiLSTM loaded and frozen.")
    return cnn, bilstm


def run_epoch(cnn, bilstm, fusion, loader, criterion, optimizer, device, phase="train"):
    """One fusion training or evaluation epoch."""
    is_train = phase == "train"
    fusion.train() if is_train else fusion.eval()
    cnn.eval()
    bilstm.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels      = [], []

    with torch.set_grad_enabled(is_train):
        for images, wq_seq, labels in loader:
            images  = images.to(device)
            wq_seq  = wq_seq.to(device)
            labels  = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            with torch.no_grad():
                algae_emb = cnn(images, return_embedding=True)  # (B, 128)
                pred_do   = bilstm(wq_seq)                       # (B, 1)

            logits = fusion(algae_emb, pred_do)                  # (B, 3)
            loss   = criterion(logits, labels)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(fusion.parameters(), max_norm=1.0)
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1, all_preds, all_labels


def main():
    print("=" * 60)
    print("  Phase 3 — Training Fusion Risk Classifier")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ── Check for real images ─────────────────────────────────────────────
    img_dir = os.path.join(DATA_PROCESSED, "images", "train")
    use_real = os.path.exists(img_dir)
    if use_real:
        print("[✓] Real image data found — using paired image+tabular training")
    else:
        print("[!] No image data found — using placeholder embeddings for now")
        print("    Run python data/preprocess_images.py to enable real image fusion")

    train_loader, val_loader, test_loader, yr_train = build_dataloaders(use_real)

    # ── Load frozen pretrained branches ──────────────────────────────────
    cnn, bilstm = load_pretrained_models(DEVICE)

    # ── Fusion model ──────────────────────────────────────────────────────
    fusion = FusionRiskClassifier().to(DEVICE)

    # ── Loss with class weighting ─────────────────────────────────────────
    class_weights = get_class_weights(yr_train, NUM_RISK_CLASSES, DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiser: only fusion params ────────────────────────────────────
    optimizer = torch.optim.Adam(
        fusion.parameters(), lr=LEARNING_RATE * 2, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=PATIENCE, mode="max")
    logger     = MetricsLogger("fusion_training", LOGS)
    best_f1    = 0.0

    print(f"\n{'─'*60}")
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(
            cnn, bilstm, fusion, train_loader, criterion, optimizer, DEVICE, "train"
        )
        va_loss, va_acc, va_f1, _, _ = run_epoch(
            cnn, bilstm, fusion, val_loader, criterion, optimizer, DEVICE, "val"
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
            path = save_checkpoint(fusion, optimizer, epoch, va_f1, "fusion_best")
            print(f"  ✓ New best saved  val_f1={va_f1:.4f}  → {path}")

        if early_stop(va_f1):
            print(f"\n[EarlyStopping] Triggered at epoch {epoch}. Best val F1={best_f1:.4f}")
            break

    # ── Final test evaluation ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    load_checkpoint(fusion, "fusion_best", device=DEVICE)

    _, te_acc, te_f1, te_preds, te_labels = run_epoch(
        cnn, bilstm, fusion, test_loader, criterion, None, DEVICE, "val"
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
    print(f"\n[✓] Fusion training complete. Checkpoint: checkpoints/fusion_best.pt")


if __name__ == "__main__":
    main()
