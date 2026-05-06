"""
Final merger that produces train_rebalanced.csv (and optionally
val_rebalanced.csv / test_rebalanced.csv) at the validator-target distribution
T=300K, h_then_ai:ai_then_human = 2:1, pile/CC = 60/40.

Operates by:
  1. Reading train_final.csv (READ-ONLY — never modified)
  2. Grouping rows into target cells (data_source, sample_type, seam_pos_bucket)
  3. For each cell with surplus, deleting rows in priority order:
        a. C4 first (rows from original train.csv — augmented in {false, original})
        b. Then derived rows (long_row_split, two_seam_split, late_seam_synth*)
        c. Last, original CC-net adv_* rows (the 'good' data we want to preserve)
     After sorting, the first N (= target count) are KEPT.
     Within each priority tier, we shuffle to keep model_name balance even.
  4. Dropping all 2-seam and multi-seam rows (off validator distribution)
  5. Appending all 4 cut files + 4 NEW generation files
  6. Text-hash dedup against itself (no duplicate rows)
  7. Writing train_rebalanced.csv

Same deletion logic for val/test but no additions (those files are size-target
~30K and just need surplus deletion).
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

DATA_DIR = Path("data/Training_Dataset")
SEED = 42

# Target distribution (matches the rebalance plan you confirmed)
TGT_ST = {
    "pure_human":     0.25,
    "pure_ai":        0.25,
    "human_then_ai":  1 / 3,    # ≈ 0.3333
    "ai_then_human":  1 / 6,    # ≈ 0.1667
}
TGT_POS = [0.10, 0.15, 0.25, 0.25, 0.15, 0.10]   # widths 0-10/10-25/25-50/50-75/75-90/90+
TGT_SRC = {"pile": 0.60, "common_crawl": 0.40}

POS_LABELS = ["0-10", "10-25", "25-50", "50-75", "75-90", "90+"]


def num_seams(labels):
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])


def first_seam_idx(labels):
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            return i
    return None


def pos_bucket(p):
    if p < 0.10: return 0
    if p < 0.25: return 1
    if p < 0.50: return 2
    if p < 0.75: return 3
    if p < 0.90: return 4
    return 5


def cell_of(row):
    """Return (data_source, sample_type, pos_bucket | None) or None if row
    doesn't belong in any target cell (e.g., 2-seam, multi-seam, unknown source)."""
    src = row.get("data_source")
    st = row.get("sample_type")
    if src not in TGT_SRC or st not in TGT_ST:
        return None
    try:
        lbl = json.loads(row.get("segmentation_labels", "[]"))
    except Exception:
        return None
    if st in ("pure_human", "pure_ai"):
        return (src, st, None) if num_seams(lbl) == 0 else None
    if num_seams(lbl) != 1:
        return None
    return (src, st, pos_bucket(first_seam_idx(lbl) / len(lbl)))


def origin(row):
    """Classify row origin for deletion priority.
    Returns 'CC-net' (preserve), 'derived' (medium), 'C4' (delete first)."""
    a = (row.get("augmented") or "").strip().lower()
    if a in ("false", "original", "", "(none)"):
        return "C4"
    if a in ("true", "augmented_unspecified"):
        return "CC-net"
    return "derived"


def text_hash(t):
    return hashlib.md5((t or "").encode("utf-8")).hexdigest()


def build_targets(T):
    """Construct dict: cell_key -> target row count."""
    out = {}
    for src, p_src in TGT_SRC.items():
        for st, p_st in TGT_ST.items():
            if st in ("pure_human", "pure_ai"):
                out[(src, st, None)] = int(round(T * p_src * p_st))
            else:
                for i, p_pos in enumerate(TGT_POS):
                    out[(src, st, i)] = int(round(T * p_src * p_st * p_pos))
    return out


def load_csv(path):
    rows = []
    fields = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)
    return rows, fields


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def cell_label(cell):
    src, st, pos = cell
    src_short = "p" if src == "pile" else "c"
    if pos is None:
        return f"{src_short}_{st}"
    return f"{src_short}_{st}_{POS_LABELS[pos]}"


def rebalance_core(input_path, T, additional_files, rng, output_path):
    """Deletion + optional additions for any of the three files."""
    print(f"\n{'='*72}")
    print(f"  {input_path.name}  →  {output_path.name}  (target T = {T:,})")
    print(f"{'='*72}")

    targets = build_targets(T)
    rows_in, fields = load_csv(input_path)
    print(f"\nLoaded {len(rows_in):,} rows from {input_path.name}")

    # Group by cell
    by_cell = defaultdict(list)
    n_no_cell = 0
    for row in rows_in:
        c = cell_of(row)
        if c is None:
            n_no_cell += 1
            continue
        by_cell[c].append(row)
    print(f"  Off-target rows (2-seam/multi-seam/etc.) — DROPPED: {n_no_cell:,}")
    in_target = sum(len(v) for v in by_cell.values())
    print(f"  In-target rows: {in_target:,}")

    # Per-cell deletion
    keep_priority = {"CC-net": 0, "derived": 1, "C4": 2}
    final_rows = []
    deleted_summary = Counter()

    for cell in sorted(by_cell.keys()):
        rows = by_cell[cell]
        target = targets.get(cell, 0)
        cur = len(rows)
        if cur <= target:
            final_rows.extend(rows)
            continue
        # Sort by priority (low number = keep), shuffle within tier
        rng.shuffle(rows)
        rows_sorted = sorted(rows, key=lambda r: keep_priority.get(origin(r), 3))
        kept = rows_sorted[:target]
        # Track what got deleted
        for r in rows_sorted[target:]:
            deleted_summary[(cell, origin(r))] += 1
        final_rows.extend(kept)

    print(f"\n  Kept after per-cell deletion: {len(final_rows):,}")
    print(f"  Deleted by (cell, origin):")
    for (cell, org), n in sorted(deleted_summary.items(), key=lambda x: -x[1])[:15]:
        print(f"    {cell_label(cell):<22s} {org:<8s} {n:>7,}")

    # Build hash set for dedup against additions
    seen = set(text_hash(r.get("text", "")) for r in final_rows)

    # Append additional files (cuts + new generations)
    if additional_files:
        print(f"\n  Appending {len(additional_files)} additional file(s):")
        for path in additional_files:
            if not path.exists():
                print(f"    SKIP  {path.name}  (not found)")
                continue
            add_rows, _ = load_csv(path)
            kept_add = 0
            dropped_dup = 0
            for r in add_rows:
                h = text_hash(r.get("text", ""))
                if h in seen:
                    dropped_dup += 1
                    continue
                seen.add(h)
                final_rows.append(r)
                kept_add += 1
            note = f" ({dropped_dup:,} dups skipped)" if dropped_dup else ""
            print(f"    {path.name:<40s} +{kept_add:>7,}{note}")

    # Shuffle so additions are interleaved with kept rows
    rng.shuffle(final_rows)

    # Write output
    write_csv(output_path, final_rows, fields)
    print(f"\n  WROTE  {output_path.name}  ({len(final_rows):,} rows)")
    return final_rows, fields


def report_distribution(rows, label):
    """Report final distribution."""
    print(f"\n{'='*72}")
    print(f"  Final distribution: {label}  ({len(rows):,} rows)")
    print(f"{'='*72}")
    by_st = Counter()
    by_src = Counter()
    by_seams = Counter()
    by_cell = defaultdict(int)
    for row in rows:
        by_st[row.get("sample_type", "?")] += 1
        by_src[row.get("data_source", "?")] += 1
        try:
            lbl = json.loads(row.get("segmentation_labels", "[]"))
        except Exception:
            continue
        n = num_seams(lbl)
        by_seams[n] += 1

    print(f"\n  Sample types:")
    for st, n in sorted(by_st.items()):
        print(f"    {st:<22s} {n:>8,}  ({100*n/len(rows):>5.1f}%)")
    print(f"\n  Data sources:")
    for src, n in sorted(by_src.items()):
        print(f"    {src:<22s} {n:>8,}  ({100*n/len(rows):>5.1f}%)")
    print(f"\n  Seam counts:")
    for sc in sorted(by_seams.keys()):
        n = by_seams[sc]
        print(f"    seams={sc:<6d} {n:>8,}  ({100*n/len(rows):>5.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true",
                    help="Rebalance train_final.csv -> train_rebalanced.csv")
    ap.add_argument("--val", action="store_true",
                    help="Rebalance val_final.csv -> val_rebalanced.csv  (deletion only)")
    ap.add_argument("--test", action="store_true",
                    help="Rebalance test_final.csv -> test_rebalanced.csv (deletion only)")
    ap.add_argument("--all", action="store_true",
                    help="Run all three.")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.all:
        args.train = args.val = args.test = True
    if not (args.train or args.val or args.test):
        sys.exit("Specify at least one of --train / --val / --test / --all")

    rng = random.Random(args.seed)

    # ---- TRAIN ----
    if args.train:
        # Train gets: 4 cut files + 4 new generation files
        additional = [
            DATA_DIR / "cuts_cc_a0_10.csv",
            DATA_DIR / "cuts_pile_h50_75.csv",
            DATA_DIR / "cuts_pile_a_early.csv",
            DATA_DIR / "cuts_pile_pure_ai_from_a.csv",
            DATA_DIR / "new_pure_human_pile.csv",
            DATA_DIR / "new_pure_ai_pile.csv",
            DATA_DIR / "new_h50_75_pile.csv",
            DATA_DIR / "new_h90plus_pile.csv",
        ]
        rows, _ = rebalance_core(
            input_path=DATA_DIR / "train_final.csv",
            T=300_000,
            additional_files=additional,
            rng=rng,
            output_path=DATA_DIR / "train_rebalanced.csv",
        )
        report_distribution(rows, "train_rebalanced.csv")

    # ---- VAL ----
    if args.val:
        # Look for any val_new_*.csv files in DATA_DIR (added top-up generations)
        val_additional = sorted(DATA_DIR.glob("val_new_*.csv"))
        if val_additional:
            print(f"\nFound val top-up files: {[p.name for p in val_additional]}")
        rows, _ = rebalance_core(
            input_path=DATA_DIR / "val_final.csv",
            T=30_000,
            additional_files=val_additional or None,
            rng=rng,
            output_path=DATA_DIR / "val_rebalanced.csv",
        )
        report_distribution(rows, "val_rebalanced.csv")

    # ---- TEST ----
    if args.test:
        rows, _ = rebalance_core(
            input_path=DATA_DIR / "test_final.csv",
            T=30_000,
            additional_files=None,
            rng=rng,
            output_path=DATA_DIR / "test_rebalanced.csv",
        )
        report_distribution(rows, "test_rebalanced.csv")


if __name__ == "__main__":
    main()
