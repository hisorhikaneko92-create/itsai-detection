"""
Merge multiple training-dataset CSVs (from build_training_dataset.py) into a
single shuffled file, optionally balancing per-sample-type counts.

Usage:
    python scripts/merge_csv_datasets.py \
        --input data/train_50k_humanthenai.csv \
                data/train_50K_part1.csv \
                data/train_50K_part2.csv \
        --output data/train_merged.csv \
        --balance
"""
import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", nargs="+", required=True,
                    help="One or more input CSV files (build_training_dataset.py output).")
    ap.add_argument("--output", required=True,
                    help="Output CSV path. Existing file will be overwritten.")
    ap.add_argument("--balance", action="store_true",
                    help="Subsample each sample_type to the count of the rarest "
                         "type so all four types end up equal-sized. Without this "
                         "flag the merged file keeps every row and the order is "
                         "shuffled but counts stay as-merged.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Shuffle / subsampling seed (default 42).")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Cap final row count after balance + shuffle.")
    args = ap.parse_args()

    # Load every input row, remembering column order from the first file.
    rows = []
    fieldnames = None
    for path in args.input:
        p = Path(path)
        if not p.exists():
            sys.exit(f"Input file not found: {p}")
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            n_before = len(rows)
            for r in reader:
                rows.append(r)
            print(f"  loaded {n_before:>7} -> {len(rows):>7}  ({p})")
    print(f"\nTotal merged rows: {len(rows)}")

    if not rows:
        sys.exit("No rows to write.")

    # Per-type / per-data-source breakdown BEFORE balancing.
    type_counts = Counter(r.get("sample_type", "?") for r in rows)
    src_counts = Counter(r.get("data_source", "?") for r in rows)
    print("\nBefore balance/shuffle:")
    print(f"  sample_type  : {dict(type_counts)}")
    print(f"  data_source  : {dict(src_counts)}")

    if args.balance:
        groups = {}
        for r in rows:
            groups.setdefault(r.get("sample_type", "?"), []).append(r)
        if not groups:
            sys.exit("Nothing to balance.")
        min_n = min(len(g) for g in groups.values())
        rng = random.Random(args.seed)
        balanced = []
        for t, g in groups.items():
            if len(g) > min_n:
                g = rng.sample(g, min_n)
            balanced.extend(g)
        rows = balanced
        print(f"\nBalanced each sample_type to {min_n} rows -> {len(rows)} total.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    if args.max_rows is not None and len(rows) > args.max_rows:
        rows = rows[: args.max_rows]
        print(f"Capped to --max-rows {args.max_rows}.")

    type_counts = Counter(r.get("sample_type", "?") for r in rows)
    src_counts = Counter(r.get("data_source", "?") for r in rows)
    print("\nAfter shuffle/balance:")
    print(f"  sample_type  : {dict(type_counts)}")
    print(f"  data_source  : {dict(src_counts)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # Re-emit only the known columns; skip any extra fields a row may
            # have, and fill missing ones with the empty string.
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
