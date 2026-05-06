"""
Fix #5 Option B — duplicate-row oversampling on train_rebalanced.csv.

Multipliers (compounded when a row matches multiple categories):

  HARD ADVERSARIAL MODELS  ×2
    google/gemma-2-27b-it
    mistralai/mistral-small-24b-instruct-2501
    nousresearch/hermes-3-llama-3.1-70b
    cohere/command-r-plus-08-2024
    microsoft/phi-4

  LATE-SEAM (single-seam, position >= 0.75)  ×1.5
    Pushes seam-localization gradient toward the previously-blind region.

A row matching both gets 2 * 1.5 = 3 (rounded with random tiebreaker).

Reads:  data/Training_Dataset/train_rebalanced.csv
Writes: data/Training_Dataset/train_rebalanced_oversampled.csv

Originals untouched. Reports the multiplier distribution and final size.
"""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

DATA_DIR = Path("data/Training_Dataset")
INPUT = DATA_DIR / "train_rebalanced.csv"
OUTPUT = DATA_DIR / "train_rebalanced_oversampled.csv"
SEED = 42

HARD_MODELS = {
    "google/gemma-2-27b-it",
    "mistralai/mistral-small-24b-instruct-2501",
    "nousresearch/hermes-3-llama-3.1-70b",
    "cohere/command-r-plus-08-2024",
    "microsoft/phi-4",
}

HARD_MODEL_MULT = 2.0
LATE_SEAM_MULT = 1.5    # applies when seam position >= 0.75
LATE_SEAM_THRESHOLD = 0.75


def num_seams(labels):
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])


def first_seam_pos(labels):
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            return i / len(labels)
    return None


def compute_multiplier(row, rng):
    """Return integer multiplier (>=1) for this row."""
    mult = 1.0

    model = (row.get("model_name") or "").strip()
    if model in HARD_MODELS:
        mult *= HARD_MODEL_MULT

    try:
        labels = json.loads(row.get("segmentation_labels") or "[]")
        if num_seams(labels) == 1:
            p = first_seam_pos(labels)
            if p is not None and p >= LATE_SEAM_THRESHOLD:
                mult *= LATE_SEAM_MULT
    except Exception:
        pass

    # Round non-integer multipliers stochastically to avoid bias
    base = int(mult)
    frac = mult - base
    if rng.random() < frac:
        base += 1
    return max(1, base)


def main():
    rng = random.Random(SEED)
    if not INPUT.exists():
        sys.exit(f"Not found: {INPUT}")

    print(f"Reading {INPUT}…")
    n_in = 0
    n_out = 0
    multiplier_counts = Counter()
    by_model_in = Counter()
    by_model_out = Counter()
    by_seam_pos_in = Counter()
    by_seam_pos_out = Counter()

    with open(INPUT, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        fields = list(reader.fieldnames or [])

        with open(OUTPUT, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fields)
            writer.writeheader()

            for row in reader:
                n_in += 1
                m = compute_multiplier(row, rng)
                multiplier_counts[m] += 1
                model = (row.get("model_name") or "(none)").strip() or "(none)"
                by_model_in[model] += 1
                by_model_out[model] += m

                # Seam position bucket (for stats only)
                try:
                    labels = json.loads(row.get("segmentation_labels") or "[]")
                    if num_seams(labels) == 1:
                        p = first_seam_pos(labels)
                        bucket = "0-25" if p < 0.25 else ("25-50" if p < 0.50 else
                                  ("50-75" if p < 0.75 else "75+"))
                        by_seam_pos_in[bucket] += 1
                        by_seam_pos_out[bucket] += m
                except Exception:
                    pass

                # Write the row m times
                for _ in range(m):
                    writer.writerow({k: row.get(k, "") for k in fields})
                    n_out += 1

    print(f"\nInput rows:  {n_in:,}")
    print(f"Output rows: {n_out:,}  (×{n_out/max(1,n_in):.2f})")
    print(f"\nMultiplier distribution:")
    for m in sorted(multiplier_counts.keys()):
        n = multiplier_counts[m]
        print(f"  ×{m}: {n:>8,} rows  ({100*n/n_in:.1f}%)")

    print(f"\nHard-model row counts (input -> output):")
    for m in HARD_MODELS:
        i = by_model_in.get(m, 0)
        o = by_model_out.get(m, 0)
        if i > 0:
            print(f"  {m[:55]:<55s} {i:>7,} -> {o:>7,}  (+{o-i:,})")

    print(f"\nSeam position (single-seam, input -> output):")
    for b in ["0-25", "25-50", "50-75", "75+"]:
        i = by_seam_pos_in.get(b, 0)
        o = by_seam_pos_out.get(b, 0)
        print(f"  {b:<7s} {i:>8,} -> {o:>8,}  (×{o/max(1,i):.2f})")

    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
