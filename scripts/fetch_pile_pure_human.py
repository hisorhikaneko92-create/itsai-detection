"""
Stream from monology/pile-uncopyrighted, sample documents in [35, 350] words,
dedup against existing train_final/val_final/test_final, and write 12,285
pure_human rows.

Each row:
  text:                 raw pile text (truncated to <=350 words if needed)
  segmentation_labels:  [0] * n_words
  data_source:          'pile'
  sample_type:          'pure_human'
  model_name:           ''
  n_words:              <count>
  augmented:            'pile_dump_pure_human'

Output: data/Training_Dataset/new_pure_human_pile.csv
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

import argparse

DATA_DIR = Path("data/Training_Dataset")
DEFAULT_OUTPUT = DATA_DIR / "new_pure_human_pile.csv"
DEFAULT_TARGET = 12_285
MIN_WORDS = 35
MAX_WORDS = 350
DEFAULT_SEED = 42

# Parse CLI args (with defaults that match original behavior)
_ap = argparse.ArgumentParser()
_ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
_ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
_ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
_args = _ap.parse_args()
OUTPUT = Path(_args.output)
TARGET = _args.target
SEED = _args.seed


def text_hash(t):
    return hashlib.md5((t or "").encode("utf-8")).hexdigest()


def build_dedup_set():
    """Read all 3 final files; return set of text-hashes to avoid."""
    seen = set()
    for fn in ["train_final.csv", "val_final.csv", "test_final.csv"]:
        path = DATA_DIR / fn
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                seen.add(text_hash(row.get("text", "")))
    return seen


def main():
    print(f"Building dedup set from existing final CSVs…")
    seen = build_dedup_set()
    print(f"  {len(seen):,} text hashes loaded")

    print(f"Streaming monology/pile-uncopyrighted…")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: pip install datasets")

    ds = load_dataset(
        "monology/pile-uncopyrighted",
        streaming=True,
        split="train",
    ).shuffle(seed=SEED, buffer_size=10_000)

    fieldnames = ["text", "segmentation_labels", "data_source",
                  "sample_type", "model_name", "n_words", "augmented"]

    n_written = 0
    n_seen = 0
    n_dup = 0
    n_too_short = 0
    n_too_long_truncated = 0

    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for el in ds:
            n_seen += 1
            text = (el.get("text") or "").strip()
            if not text:
                continue

            words = text.split()
            if len(words) < MIN_WORDS:
                n_too_short += 1
                continue
            if len(words) > MAX_WORDS:
                # Take a random window of MAX_WORDS instead of just truncating from start
                # (truncation is fine — pile-uncopyrighted often starts mid-doc anyway)
                import random
                start = random.randint(0, len(words) - MAX_WORDS)
                words = words[start:start + MAX_WORDS]
                n_too_long_truncated += 1

            new_text = " ".join(words)
            h = text_hash(new_text)
            if h in seen:
                n_dup += 1
                continue
            seen.add(h)

            row = {
                "text": new_text,
                "segmentation_labels": json.dumps([0] * len(words)),
                "data_source": "pile",
                "sample_type": "pure_human",
                "model_name": "",
                "n_words": str(len(words)),
                "augmented": "pile_dump_pure_human",
            }
            writer.writerow(row)
            n_written += 1

            if n_written % 1000 == 0:
                print(f"  {n_written:,} written  ({n_seen:,} seen, "
                      f"{n_dup:,} dups, {n_too_short:,} too short, "
                      f"{n_too_long_truncated:,} truncated)")

            if n_written >= TARGET:
                break

    print(f"\nDone. Wrote {n_written:,} rows to {OUTPUT}")
    print(f"  Total streamed:   {n_seen:,}")
    print(f"  Duplicates:       {n_dup:,}")
    print(f"  Too short (<35):  {n_too_short:,}")
    print(f"  Truncated (>350): {n_too_long_truncated:,}")


if __name__ == "__main__":
    main()
