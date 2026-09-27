#!/usr/bin/env python3
"""Run preprocessing, pair generation, and conditional-VAE training."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def run_stage(name, command):
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}", flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_path",
        help="Path to the extracted Groove dataset directory containing info.csv.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not (dataset_path / "info.csv").is_file():
        parser.error(f"Dataset path does not contain info.csv: {dataset_path}")

    run_stage(
        "Stage 1 of 3: preprocessing the Groove MIDI Dataset",
        [sys.executable, "midi_data_preprocessing.py", str(dataset_path)],
    )
    run_stage(
        "Stage 2 of 3: generating training pairs",
        [sys.executable, "generate_training_pairs.py"],
    )
    run_stage(
        "Stage 3 of 3: training the conditional VAE",
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=-1",
            "--output",
            "training_conditional_vae_executed.ipynb",
            "training_conditional_vae.ipynb",
        ],
    )

    print("\nPipeline completed successfully.", flush=True)
    print(f"Model checkpoint: {PROJECT_DIR / 'text2groove_model.pt'}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(
            f"\nPipeline stopped: a stage exited with code {error.returncode}.",
            file=sys.stderr,
        )
        raise SystemExit(error.returncode) from error
