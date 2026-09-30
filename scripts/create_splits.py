"""Explicitly generate the team's manifests; never download data."""

import argparse
from pathlib import Path
from src.config import DATASET_DIR, SPLIT_DIR
from src.data_utils import build_dataframe, stratified_split


def main() -> None:
    """Validate the chosen class-parent directory and save reproducible splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=SPLIT_DIR)
    args = parser.parse_args()
    parts = stratified_split(build_dataframe(args.dataset_dir), args.output_dir)
    print(dict(zip(("train", "val", "test"), map(len, parts))))


if __name__ == "__main__":
    main()
