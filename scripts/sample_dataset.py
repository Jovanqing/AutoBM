"""
Sample 10% of each dataset partition to create example data for open-source release.

Usage:
    python scripts/sample_dataset.py

This script reads the full datasets from data_example/data_AutoBM/ and creates
10% samples in data_example/data_AutoBM_sample/, preserving the directory structure.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data_example", "data_AutoBM")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_example", "data_AutoBM_sample")

SAMPLE_RATIO = 0.10
RANDOM_SEED = 42

PARTITIONS = [
    ("Data_SFT", "train.parquet"),
    ("Data_SFT", "val.parquet"),
    ("Data_RL", "train.parquet"),
    ("Data_RL", "test.parquet"),
]


def sample_parquet(input_path: str, output_path: str, ratio: float, seed: int):
    df = pd.read_parquet(input_path)
    n_total = len(df)
    n_sample = max(1, int(n_total * ratio))  # at least 1 sample

    df_sample = df.sample(n=n_sample, random_state=seed)
    df_sample = df_sample.reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sample.to_parquet(output_path, index=False)

    print(f"  {os.path.basename(input_path)}: {n_total} -> {n_sample} samples ({ratio*100:.0f}%)")


def main():
    print(f"Sampling {SAMPLE_RATIO*100:.0f}% of each dataset partition")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    for subdir, filename in PARTITIONS:
        input_path = os.path.join(INPUT_DIR, subdir, filename)
        output_path = os.path.join(OUTPUT_DIR, subdir, filename)

        if not os.path.exists(input_path):
            print(f"  [SKIP] {subdir}/{filename} not found")
            continue

        print(f"[{subdir}]")
        sample_parquet(input_path, output_path, SAMPLE_RATIO, RANDOM_SEED)

    print()
    print("Done! Sampled data saved to data_example/data_AutoBM_sample/")


if __name__ == "__main__":
    main()
