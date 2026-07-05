"""
SQL-LoRA — Dataset Preparation

Downloads the b-mc2/sql-create-context dataset from Hugging Face,
cleans malformed entries, splits into train/validation/test sets,
and saves to disk for use in training and evaluation.

Usage:
    python data/prepare_dataset.py [--cache_dir PATH]
"""


def download_and_split(cache_dir: str | None = None) -> dict:
    """Download sql-create-context, clean, and split into train/val/test.

    Returns a dict with keys 'train', 'validation', 'test', each being a
    Hugging Face Dataset object.
    """
    ...


def clean_example(example: dict) -> dict | None:
    """Validate a single dataset example. Returns None if malformed."""
    ...


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    download_and_split(cache_dir=args.cache_dir)
    print("Dataset prepared.")
