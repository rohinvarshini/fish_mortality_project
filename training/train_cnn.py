# training/train_cnn.py
# ─────────────────────────────────────────────────────────────────────────────
# Trains the CNN vision branch (models/cnn_branch.py) on labeled algal bloom
# photos to classify Low / Moderate / High mortality risk from a pond/water
# surface image.
#
# Strategy:
#   - MobileNetV2 backbone pretrained on ImageNet, FROZEN
#   - Train only the embedding + classification heads
#   - Loss: weighted CrossEntropy (handles class imbalance)
#   - Data augmentation: random flip, rotation, color jitter (small dataset)
#
# Usage:
#   python data/preprocess_images.py   (once, to build manifests)
#   python training/train_cnn.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_PROCESSED, LOGS, IMAGE_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, DEVICE, RANDOM_SEED,
    NUM_RISK_CLASSES,
)
from models.cnn_branch import CNNBranch
from training.utils import (
    EarlyStopping, save_checkpoint, load_checkpoint,
    get_class_weights, MetricsLogger, print_epoch,
)

torch.manual_seed(RANDOM_SEED)

MANIFEST_DIR = os.path.join(DATA_PROCESSED, "images")
BATCH_SIZE = 16  # small dataset — smaller batches give more gradient updates per epoch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class BloomImageDataset(Dataset):
    """Yields (image_tensor, risk_label) per sample, read from a manifest CSV."""

    def __init__(self, manifest_path: str, transform):
        self.rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append((row["image_path"], int(row["label"])))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image_path, label = self.rows[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label


def build_dataloaders() -> tuple:
    train_manifest = os.path.join(MANIFEST_DIR, "train_manifest.csv")
    val_manifest = os.path.join(MANIFEST_DIR, "val_manifest.csv")

    if not (os.path.exists(train_manifest) and os.path.exists(val_manifest)):
        print(f"[!] Manifests not found in {MANIFEST_DIR}")
        print("    Run: python data/preprocess_images.py")
        sys.exit(1)

    train_ds = BloomImageDataset(train_manifest, TRAIN_TRANSFORM)
    val_ds = BloomImageDataset(val_manifest, EVAL_TRANSFORM)

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    y_train = np.array([label for _, label in train_ds.rows])
    return train_loader, val_loader, y_train


def run_epoch(model, loader, criterion, optimizer, device, phase="train"):
    is_train = phase == "train"
    model.train() if is_train else model.eval()
    # backbone stays frozen/eval regardless of phase (no BatchNorm/dropout updates in it)
    model.features.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1, all_preds, all_labels


def main():
    print("=" * 60)
    print("  Training CNN Bloom-Severity Branch")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    train_loader, val_loader, y_train = build_dataloaders()

    model = CNNBranch(pretrained=True, freeze_backbone=True).to(DEVICE)

    class_weights = get_class_weights(y_train, NUM_RISK_CLASSES, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # only the unfrozen heads have requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE * 2, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    early_stop = EarlyStopping(patience=PATIENCE, mode="max")
    logger = MetricsLogger("cnn_training", LOGS)
    best_f1 = 0.0

    print("\n" + "=" * 60)
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer, DEVICE, "train")
        va_loss, va_acc, va_f1, _, _ = run_epoch(model, val_loader, criterion, optimizer, DEVICE, "val")
        scheduler.step()

        print_epoch(
            epoch, MAX_EPOCHS,
            {"loss": tr_loss, "acc": tr_acc, "f1": tr_f1},
            {"loss": va_loss, "acc": va_acc, "f1": va_f1},
        )
        logger.log("train", epoch, {"loss": tr_loss, "acc": tr_acc, "f1": tr_f1})
        logger.log("val", epoch, {"loss": va_loss, "acc": va_acc, "f1": va_f1})

        if va_f1 > best_f1:
            best_f1 = va_f1
            path = save_checkpoint(model, optimizer, epoch, va_f1, "cnn_best")
            print(f"  [ok] New best saved  val_f1={va_f1:.4f}  -> {path}")

        if early_stop(va_f1):
            print(f"\n[EarlyStopping] Triggered at epoch {epoch}. Best val F1={best_f1:.4f}")
            break

    print("\n" + "=" * 60)
    load_checkpoint(model, "cnn_best", device=DEVICE)
    _, va_acc, va_f1, va_preds, va_labels = run_epoch(model, val_loader, criterion, None, DEVICE, "val")
    print(f"\n  FINAL VAL RESULTS:")
    print(f"    Accuracy  : {va_acc:.4f}")
    print(f"    Macro F1  : {va_f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(va_labels, va_preds, target_names=["Low", "Moderate", "High"], zero_division=0))
    print(f"\n[v] CNN training complete. Checkpoint: checkpoints/cnn_best.pt")


if __name__ == "__main__":
    main()
