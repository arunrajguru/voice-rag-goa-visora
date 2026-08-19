"""
Dataset Download Helper for ai4bharat/MSMARCO-XI
"""

import sys
import argparse
from datasets import load_dataset

DATASET_NAME = "ai4bharat/MSMARCO-XI"

def main():
    parser = argparse.ArgumentParser(description="Download ai4bharat/MSMARCO-XI dataset")
    parser.add_argument("--split", type=str, default="train", help="Split to download (default: train)")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of records to download")
    args = parser.parse_args()

    print(f"Downloading {args.limit} records from {DATASET_NAME} (split={args.split})...")
    try:
        ds = load_dataset(DATASET_NAME, split=f"{args.split}[:{args.limit}]")
        print(f"Successfully downloaded {len(ds)} records.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
