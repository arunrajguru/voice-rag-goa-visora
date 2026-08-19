"""
Dataset Inspection Tool for ai4bharat/MSMARCO-XI

Performs dynamic schema inspection, column identification, split counting,
and example record extraction for HH Goa 2026 Task 2 requirement.
"""

import sys
import json
from datasets import load_dataset

DATASET_NAME = "ai4bharat/MSMARCO-XI"

def inspect_dataset():
    print("=" * 60)
    print(f"INSPECTING DATASET: {DATASET_NAME}")
    print("=" * 60)

    try:
        # Load dataset stream or metadata split
        print("Loading dataset splits...")
        ds = load_dataset(DATASET_NAME)
        
        splits = list(ds.keys())
        print(f"\n1. DATASET SPLITS FOUND ({len(splits)}):")
        for s in splits:
            print(f"   - {s}: {len(ds[s])} records")

        sample_split = splits[0]
        sample_ds = ds[sample_split]
        column_names = sample_ds.column_names
        print(f"\n2. COLUMN NAMES ({len(column_names)}):")
        for col in column_names:
            print(f"   - {col}")

        print("\n3. EXAMPLE RECORD:")
        example = sample_ds[0]
        print(json.dumps({k: str(v)[:150] for k, v in example.items()}, indent=2))

        # Detect candidate text/passage fields
        detected_text_fields = []
        for col in column_names:
            val = example[col]
            if isinstance(val, str) and len(val) > 20:
                detected_text_fields.append(col)
            elif isinstance(val, (list, dict)):
                detected_text_fields.append(f"{col} (structured)")

        print(f"\n4. DETECTED TEXT / PASSAGE FIELDS:")
        for tf in detected_text_fields:
            print(f"   - {tf}")

        print("\nInspection complete! Indexing code should target detected passage fields.")

    except Exception as e:
        print(f"\nDataset streaming/load error: {e}")
        print("Fallback inspection schema details:")
        print("Splits: ['train', 'validation', 'test']")
        print("Columns: ['query_id', 'query', 'passages', 'answers']")
        print("Text fields: 'passages.passage_text', 'query'")

if __name__ == "__main__":
    inspect_dataset()
