# data/preprocess_images.py
# -----------------------------------------------------------------------------
# Builds train/val manifests for the CNN bloom-severity branch from three
# sources, all under algal_blooms/:
#   1. bloom_labels.json       - the original 2023 archive, human-graded
#                                 Low/Moderate/Severe (via the Triage tool)
#   2. new_bloom_labels.json   - a second batch of bloom photos, also
#                                 human-graded via the same tool (optional -
#                                 skipped with a note if not present yet)
#   3. nobloom/                - photos with no visible bloom at all; these
#                                 are unambiguous and auto-labeled "low"
#                                 without needing a human grading pass
#
# "excluded" labels (unusable photos: screenshots, duplicates, etc.) are
# dropped. Remaining photos are stratified by severity into train/val sets.
# Output: data/processed/images/{train,val}_manifest.csv
#   columns: image_path,label,severity
# -----------------------------------------------------------------------------

import os
import sys
import json
import csv
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, DATA_PROCESSED, RANDOM_SEED

ALGAL_BLOOMS_DIR   = os.path.join(BASE_DIR, "algal_blooms")
LABELS_JSON        = os.path.join(ALGAL_BLOOMS_DIR, "bloom_labels.json")
NEW_LABELS_JSON     = os.path.join(ALGAL_BLOOMS_DIR, "new_bloom_labels.json")
NOBLOOM_DIR         = os.path.join(ALGAL_BLOOMS_DIR, "nobloom")
OUT_DIR             = os.path.join(DATA_PROCESSED, "images")

# maps the triage tool's severity keys -> the project's RISK_LABELS index
SEVERITY_TO_CLASS = {"low": 0, "moderate": 1, "severe": 2}
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
VAL_FRACTION = 0.15


def _load_labeled_json(path: str) -> list:
    """Reads a Triage-tool export; returns (image_path, label, severity) tuples."""
    if not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    items, missing = [], []
    for row in rows:
        severity = row["severity"]
        if severity not in SEVERITY_TO_CLASS:
            continue  # "excluded" or unrecognized
        image_path = os.path.join(ALGAL_BLOOMS_DIR, row["source_rel_path"])
        if not os.path.exists(image_path):
            missing.append(row["source_rel_path"])
            continue
        items.append((image_path, SEVERITY_TO_CLASS[severity], severity))
    return items, missing


def _load_nobloom_dir(path: str) -> list:
    """Every photo here has no visible bloom -> unambiguously "low" risk."""
    if not os.path.isdir(path):
        return []
    items = []
    for fname in sorted(os.listdir(path)):
        if fname.lower().endswith(IMAGE_EXTS):
            items.append((os.path.join(path, fname), 0, "low"))
    return items


def build_manifests(out_dir: str = OUT_DIR, val_fraction: float = VAL_FRACTION):
    by_class = {0: [], 1: [], 2: []}

    archive_items, missing = _load_labeled_json(LABELS_JSON)
    for image_path, label, severity in archive_items:
        by_class[label].append((image_path, label, severity))
    print(f"[v] {len(archive_items)} labeled photo(s) from bloom_labels.json")

    new_items, new_missing = _load_labeled_json(NEW_LABELS_JSON)
    missing += new_missing
    if new_items:
        for image_path, label, severity in new_items:
            by_class[label].append((image_path, label, severity))
        print(f"[v] {len(new_items)} labeled photo(s) from new_bloom_labels.json")
    else:
        print("[!] new_bloom_labels.json not found yet - skipping the new bloom/ batch.")
        print("    Grade it with algal_blooms/new_bloom_triage_local.html, export, then re-run.")

    nobloom_items = _load_nobloom_dir(NOBLOOM_DIR)
    for image_path, label, severity in nobloom_items:
        by_class[label].append((image_path, label, severity))
    print(f"[v] {len(nobloom_items)} photo(s) auto-labeled 'low' from nobloom/")

    if missing:
        print(f"\n[!] {len(missing)} labeled photo(s) not found on disk, skipped:")
        for m in missing[:10]:
            print(f"      {m}")

    random.seed(RANDOM_SEED)
    train_rows, val_rows = [], []
    for label, items in by_class.items():
        random.shuffle(items)
        n_val = max(1, round(len(items) * val_fraction)) if items else 0
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])

    os.makedirs(out_dir, exist_ok=True)
    for name, split_rows in [("train", train_rows), ("val", val_rows)]:
        path = os.path.join(out_dir, f"{name}_manifest.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label", "severity"])
            writer.writerows(split_rows)
        print(f"[v] Wrote {path} ({len(split_rows)} rows)")

    print("\n[Class distribution]")
    label_names = {0: "Low", 1: "Moderate", 2: "High"}
    for label in (0, 1, 2):
        n_train = sum(1 for r in train_rows if r[1] == label)
        n_val = sum(1 for r in val_rows if r[1] == label)
        print(f"  {label_names[label]:>8}: train={n_train:>3}  val={n_val:>3}")

    return train_rows, val_rows


if __name__ == "__main__":
    build_manifests()
