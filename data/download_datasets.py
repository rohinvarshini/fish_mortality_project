# data/download_datasets.py
# ─────────────────────────────────────────────────────────────────────────────
# Downloads all datasets required for the project:
#   T1 — IoT Tilapia (Kaggle)
#   T2 — Aquaponics 12-Pond (Kaggle)
#   I1 — NASA Tick Tick Bloom images (GitHub release)
#   V1 — Zenodo Fishpond CSV (Zenodo)
#
# Usage:
#   python data/download_datasets.py
#
# Prerequisites:
#   - Kaggle API token at ~/.kaggle/kaggle.json  OR  set env vars:
#       KAGGLE_USERNAME and KAGGLE_KEY
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import zipfile
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_RAW, IMAGE_DIR


def ensure_dirs():
    os.makedirs(DATA_RAW, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    print(f"[✓] Data directories ready: {DATA_RAW}")


def download_kaggle_dataset(dataset_slug: str, output_dir: str, filename: str = None):
    """
    Downloads a Kaggle dataset using the kaggle CLI.
    dataset_slug: e.g. 'anibalpolanco/iot-monitoring-of-water-quality-and-tilapia'
    """
    try:
        import kaggle  # noqa — checks API token is present
    except ImportError:
        print("[!] kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    zip_path = os.path.join(output_dir, dataset_slug.split("/")[-1] + ".zip")
    print(f"\n[↓] Downloading Kaggle dataset: {dataset_slug}")

    os.system(
        f'kaggle datasets download -d "{dataset_slug}" -p "{output_dir}" --unzip'
    )

    if filename:
        # Rename the primary CSV to a known name if provided
        candidates = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
        if candidates:
            src = os.path.join(output_dir, candidates[0])
            dst = os.path.join(output_dir, filename)
            if src != dst:
                shutil.move(src, dst)
            print(f"[✓] Saved as: {dst}")


def download_zenodo_dataset(record_id: str, filename_hint: str, out_filename: str):
    """
    Downloads a file from Zenodo using its record ID.
    Uses the Zenodo REST API to find the actual download URL.
    """
    import urllib.request
    import json

    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"\n[↓] Fetching Zenodo record: {record_id}")

    with urllib.request.urlopen(api_url) as resp:
        record = json.loads(resp.read())

    files = record.get("files", [])
    target = None
    for f in files:
        if filename_hint.lower() in f["key"].lower():
            target = f
            break

    if target is None and files:
        target = files[0]  # fallback: take first file

    if target is None:
        print(f"[!] No files found in Zenodo record {record_id}")
        return

    url      = target["links"]["self"]
    out_path = os.path.join(DATA_RAW, out_filename)
    print(f"[↓] Downloading: {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"[✓] Saved to: {out_path}")


def download_bloom_images():
    """
    Downloads the Tick Tick Bloom image dataset from the GitHub release.
    Falls back to instructions if the file is too large.
    """
    import urllib.request

    # Public release zip from the DrivenData competition mirror
    url = (
        "https://github.com/IoannisNasios/HarmfulAlgalBloomDetection"
        "/archive/refs/heads/main.zip"
    )
    zip_path = os.path.join(DATA_RAW, "bloom_images.zip")
    print(f"\n[↓] Downloading bloom image repository: {url}")

    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"[✓] Downloaded to {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_RAW)
        os.remove(zip_path)
        print(f"[✓] Extracted to {DATA_RAW}")

    except Exception as e:
        print(f"[!] Automatic image download failed: {e}")
        print(
            "\n[Manual Step Required]\n"
            "Please download the Tick Tick Bloom dataset manually from:\n"
            "  https://www.drivendata.org/competitions/143/tick-tick-bloom/\n"
            f"  and place the images folder at: {IMAGE_DIR}\n"
            "  Labels CSV should be at: data/raw/bloom_labels.csv"
        )


def main():
    print("=" * 60)
    print("  Fish Mortality Prediction — Dataset Downloader")
    print("=" * 60)

    ensure_dirs()

    # ── T1: IoT Tilapia dataset (primary tabular) ─────────────────────────
    download_kaggle_dataset(
        dataset_slug="anibalpolanco/iot-monitoring-of-water-quality-and-tilapia",
        output_dir=DATA_RAW,
        filename="tilapia_iot.csv",
    )

    # ── T2: Aquaponics 12-pond dataset (supplementary tabular) ────────────
    download_kaggle_dataset(
        dataset_slug="blessingogbuokiri/sensor-based-aquaponics-fish-pond-datasets",
        output_dir=DATA_RAW,
        filename="aquaponics_ponds.csv",
    )

    # ── V1: Zenodo Fishpond (geographic holdout) ──────────────────────────
    download_zenodo_dataset(
        record_id="19210095",
        filename_hint=".csv",
        out_filename="zenodo_fishpond.csv",
    )

    # ── I1: NASA Tick Tick Bloom images ───────────────────────────────────
    download_bloom_images()

    print("\n" + "=" * 60)
    print("  Download complete. Run next:")
    print("    python data/preprocess_tabular.py")
    print("    python data/preprocess_images.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
