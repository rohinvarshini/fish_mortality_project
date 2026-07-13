# run_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# Single entry point to run the full tabular training pipeline in order.
#
# Usage:
#   python run_pipeline.py --all          # run everything
#   python run_pipeline.py --preprocess   # only data prep
#   python run_pipeline.py --train        # only training
#   python run_pipeline.py --evaluate     # only evaluation + SHAP
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import subprocess
import sys


STEPS = {
    "download":    ("data/download_datasets.py",      "Downloading datasets"),
    "preprocess":  ("data/preprocess_tabular.py",     "Preprocessing tabular data"),
    "train_bilstm":("training/train_bilstm.py",       "Training BiLSTM (Phase 1)"),
    "train_class": ("training/train_classifier.py",   "Training Classifier (Phase 2)"),
    "evaluate":    ("evaluation/evaluate.py",         "Evaluating models"),
    "shap":        ("explainability/shap_explainer.py","Running SHAP analysis"),
}


def run_step(name: str):
    script, label = STEPS[name]
    print(f"\n{'='*60}")
    print(f"  >>> {label}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n[x] Step '{name}' failed. Fix the error above and re-run.")
        sys.exit(result.returncode)
    print(f"[v] {label} done.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fish Mortality Prediction — Pipeline Runner"
    )
    parser.add_argument("--all",        action="store_true", help="Run full pipeline")
    parser.add_argument("--download",   action="store_true", help="Download datasets only")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess data only")
    parser.add_argument("--train",      action="store_true", help="Train all models")
    parser.add_argument("--evaluate",   action="store_true", help="Evaluate + SHAP")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        print("\nExample: python run_pipeline.py --all")
        sys.exit(0)

    if args.all or args.download:
        run_step("download")

    if args.all or args.preprocess:
        run_step("preprocess")

    if args.all or args.train:
        run_step("train_bilstm")
        run_step("train_class")

    if args.all or args.evaluate:
        run_step("evaluate")
        run_step("shap")

    print("\n" + "="*60)
    print("  Pipeline complete!")
    print("  Checkpoints : checkpoints/")
    print("  Logs        : logs/")
    print("  Results     : evaluation/")
    print("  SHAP plot   : explainability/shap_global.png")
    print("="*60)


if __name__ == "__main__":
    main()
