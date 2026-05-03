"""
Stratified 80/10/10 train/val/test split for the merged HSSD training CSV.

The split honors two stratification axes simultaneously:

  1. sample_type        (pure_human / pure_ai / human_then_ai / ai_in_middle)
  2. seam-position bucket per the AI_Detection split-strategy doc:
       early   -- first 0->1 / 1->0 transition lies in the first 20% of words
       middle  -- transition lies in the 20-80% range
       late    -- transition lies in the last 20% of words
       pure_*  -- no transition at all (entire sequence is one class)

Each bucket is shuffled independently and partitioned 80/10/10, then the
resulting splits are re-shuffled. This keeps each split's internal
distribution identical to the source CSV while guaranteeing no row appears
in more than one split.

Why this matters:
  * Random split alone over-represents middle-seam samples in val/test
    because most of our data is middle-seam. A model trained that way
    looks great on validation but flunks early/late seams in production.
  * Same row can never leak across splits, so the test set really is held
    out from training.

Optional `--rebalance-seam-buckets` subsamples the middle bucket so the
final split matches the doc's 25 / 50 / 25 target distribution
(early / middle / late). Use this if you have enough data to spare.

Usage:
    python scripts/split_dataset.py \
        --input data/train_merged.csv \
        --output-train data/train.csv \
        --output-val   data/val.csv \
        --output-test  data/test.csv

    # With seam-bucket rebalancing:
    python scripts/split_dataset.py \
        --input data/train_merged.csv \
        --output-train data/train.csv \
        --output-val   data/val.csv \
        --output-test  data/test.csv \
        --rebalance-seam-buckets
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _seam_bucket(labels: List[int]) -> str:
    """Return one of: pure_human, pure_ai, early, middle, late, unknown.

    Matches AI_Detection_Document.md's 25/50/25 early/middle/late rule by
    measuring the first label transition's position relative to the total
    sequence length."""
    if not labels:
        return "unknown"
    s = sum(labels)
    n = len(labels)
    if s == 0:
        return "pure_human"
    if s == n:
        return "pure_ai"
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            pos = i / n
            if pos < 0.2:
                return "early"
            if pos > 0.8:
                return "late"
            return "middle"
    return "unknown"


def _load_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, fieldnames


def _bucketize(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    """Group rows by (sample_type, seam_bucket). Rows whose
    segmentation_labels are malformed are dropped with a warning."""
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    skipped = 0
    for r in rows:
        try:
            labels = json.loads(r.get("segmentation_labels", "[]"))
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(labels, list):
            skipped += 1
            continue
        sample_type = r.get("sample_type", "?")
        bucket = _seam_bucket(labels)
        groups[(sample_type, bucket)].append(r)
    if skipped:
        print(f"WARNING: dropped {skipped} rows with malformed labels")
    return groups


def _stratified_split(
    groups: Dict[Tuple[str, str], List[Dict[str, str]]],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """80/10/10 split applied PER STRATIFICATION GROUP, so every split
    keeps the same per-(type, bucket) proportions as the input."""
    rng = random.Random(seed)
    train, val, test = [], [], []
    for key, items in groups.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        # Guard small groups: at least 1 in val/test if possible.
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = max(1, min(n_val, n - n_train - 1))
        train.extend(items[:n_train])
        val.extend(items[n_train: n_train + n_val])
        test.extend(items[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _rebalance_seam_buckets(
    rows: List[Dict[str, str]],
    target: Dict[str, float],
    seed: int,
) -> List[Dict[str, str]]:
    """Subsample middle/late/early to match a target ratio (only the
    early/middle/late buckets are rebalanced; pure_* rows are preserved
    in full so single-class samples remain at their original count)."""
    rng = random.Random(seed)
    bucketed: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        try:
            labels = json.loads(r.get("segmentation_labels", "[]"))
        except json.JSONDecodeError:
            continue
        bucketed[_seam_bucket(labels)].append(r)

    pure_rows = bucketed.get("pure_human", []) + bucketed.get("pure_ai", [])
    transition_rows = {b: bucketed.get(b, []) for b in ("early", "middle", "late")}

    counts = {b: len(rs) for b, rs in transition_rows.items()}
    if not any(counts.values()):
        return rows  # nothing to rebalance

    # Pick the largest feasible total such that each bucket's target share
    # can be met without exceeding the available rows.
    feasible = min(
        counts[b] / target[b] for b in target if target[b] > 0 and b in counts
    )
    total = int(feasible)
    chosen: List[Dict[str, str]] = []
    for b, share in target.items():
        n_take = int(round(total * share))
        rs = transition_rows.get(b, [])
        if n_take >= len(rs):
            chosen.extend(rs)
        else:
            chosen.extend(rng.sample(rs, n_take))
    chosen.extend(pure_rows)
    rng.shuffle(chosen)
    return chosen


def _print_report(label: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        print(f"  {label:<7s}  (empty)")
        return
    type_counts = Counter(r.get("sample_type", "?") for r in rows)
    bucket_counts: Counter = Counter()
    for r in rows:
        try:
            labels = json.loads(r.get("segmentation_labels", "[]"))
            bucket_counts[_seam_bucket(labels)] += 1
        except json.JSONDecodeError:
            bucket_counts["unknown"] += 1
    print(f"  {label:<7s}  total={len(rows):,}")
    print(f"           sample_type: {dict(type_counts)}")
    print(f"           seam_bucket: {dict(bucket_counts)}")


def _write(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True,
                    help="Merged training CSV from merge_csv_datasets.py.")
    ap.add_argument("--output-train", required=True)
    ap.add_argument("--output-val", required=True)
    ap.add_argument("--output-test", required=True)
    ap.add_argument("--ratios", default="0.8,0.1,0.1",
                    help="Comma-separated train/val/test ratios. Default 0.8,0.1,0.1.")
    ap.add_argument("--rebalance-seam-buckets", action="store_true",
                    help="Subsample the middle bucket so transition samples "
                         "land in a 25/50/25 (early/middle/late) ratio per "
                         "the AI_Detection_Document.md recommendation. Costs "
                         "rows but yields a positionally balanced split.")
    ap.add_argument("--target-early", type=float, default=0.25)
    ap.add_argument("--target-middle", type=float, default=0.50)
    ap.add_argument("--target-late", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    parts = [float(x) for x in args.ratios.split(",")]
    if len(parts) != 3 or abs(sum(parts) - 1.0) > 1e-6 or any(p < 0 for p in parts):
        sys.exit(f"--ratios must be three positive numbers summing to 1.0, got {args.ratios}")
    ratios = (parts[0], parts[1], parts[2])

    print(f"Loading {in_path}")
    rows, fieldnames = _load_rows(in_path)
    print(f"  {len(rows):,} rows loaded")
    print(f"  columns: {fieldnames}")

    if args.rebalance_seam_buckets:
        target = {
            "early":  args.target_early,
            "middle": args.target_middle,
            "late":   args.target_late,
        }
        if abs(sum(target.values()) - 1.0) > 1e-6:
            sys.exit("--target-early + --target-middle + --target-late must sum to 1.0")
        n_before = len(rows)
        rows = _rebalance_seam_buckets(rows, target, seed=args.seed)
        print(f"  rebalanced from {n_before:,} -> {len(rows):,} rows "
              f"({target['early']:.0%}/{target['middle']:.0%}/{target['late']:.0%} "
              f"early/middle/late)")

    print("\nStratifying by (sample_type, seam_bucket)...")
    groups = _bucketize(rows)
    print(f"  {len(groups)} stratification groups")

    train, val, test = _stratified_split(groups, ratios, args.seed)

    print("\nFinal splits:")
    _print_report("train", train)
    _print_report("val",   val)
    _print_report("test",  test)

    _write(Path(args.output_train), train, fieldnames)
    _write(Path(args.output_val),   val,   fieldnames)
    _write(Path(args.output_test),  test,  fieldnames)
    print(f"\nWrote:")
    print(f"  {args.output_train}  ({len(train):,} rows)")
    print(f"  {args.output_val}    ({len(val):,} rows)")
    print(f"  {args.output_test}   ({len(test):,} rows)")


if __name__ == "__main__":
    main()
