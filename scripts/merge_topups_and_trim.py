"""Merge all topup_*.csv files into the two master training CSVs and trim
each (source × class) bucket down to the validator-aligned target counts.

Default mode is --dry-run: prints projected before/after distributions
without touching any files. Pass --apply to actually write.

Snapshots both masters as
    train_<source>_with_adv.bak.before_internlm_merge.<UTC-timestamp>.csv
before any write. Topup rows are preserved preferentially during trims —
the random drop is applied only to pre-existing master rows.

Usage:
    # default: dry run, shows what would happen
    python scripts/merge_topups_and_trim.py

    # commit
    python scripts/merge_topups_and_trim.py --apply

    # different seed
    python scripts/merge_topups_and_trim.py --seed 7
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path("/root/llm-detection/data/MainData")
BY_SRC = ROOT / "by_source"
PILE_MASTER = BY_SRC / "train_pile_with_adv.csv"
CC_MASTER   = BY_SRC / "train_common_crawl_with_adv.csv"

# Validator-aligned target distributions
PILE_TARGET_TOTAL = 140_000
CC_TARGET_TOTAL   = 70_000
TARGET_RATIOS = {"pure_human": 0.25, "pure_ai": 0.25, "human_then_ai": 0.40, "ai_then_human": 0.10}

PILE_TARGETS = {k: int(round(v * PILE_TARGET_TOTAL)) for k, v in TARGET_RATIOS.items()}
CC_TARGETS   = {k: int(round(v * CC_TARGET_TOTAL))   for k, v in TARGET_RATIOS.items()}

# Topup files per source. Order matters only for reproducibility, not behavior.
PILE_TOPUPS = [
    ROOT / "from_a100_1" / "topup_pure_human_pile.csv",   # 4,500 pure_h Pile (md5-identical on VPS — included once)
    ROOT / "from_a100_1" / "topup_cohere_pile.csv",
    ROOT / "from_a100_1" / "topup_fast_pile.csv",
    ROOT / "from_a100_1" / "topup_close_pile.csv",
    ROOT / "from_a100_1" / "topup_other_pile.csv",
    ROOT / "from_a100_1" / "topup_internlm_pile.csv",     # InternLM 3-model run (~18K rows)
    ROOT / "from_a100_1" / "topup_internlm_20bchat_pile.csv",  # InternLM 39GB-only follow-up (~4K rows)
]
CC_TOPUPS = [
    ROOT / "from_a100_1" / "topup_cc_cohere.csv",
    ROOT / "from_a100_1" / "topup_cc_other.csv",
    ROOT / "from_a100_2" / "topup_cc_internlm3.csv",
    ROOT / "from_a100_2" / "topup_cc_internlm_20bchat.csv",
]


def _existing(paths):
    return [p for p in paths if p.exists()]


def _read_with_origin(path, origin_label):
    df = pd.read_csv(path)
    df["__origin__"] = origin_label  # "master" or topup-file basename
    return df


def _summarize(df, label):
    counts = df["sample_type"].value_counts().to_dict()
    print(f"  {label:<32} total={len(df):>6}  "
          f"pure_h={counts.get('pure_human',0):>5}  "
          f"pure_ai={counts.get('pure_ai',0):>5}  "
          f"h_then_ai={counts.get('human_then_ai',0):>5}  "
          f"ai_then_h={counts.get('ai_then_human',0):>5}")


def _trim_bucket(df, sample_type, target_count, seed):
    """Trim rows of a given sample_type to target_count, preferentially
    keeping topup rows. Returns trimmed dataframe of just this bucket."""
    bucket = df[df["sample_type"] == sample_type]
    if len(bucket) <= target_count:
        return bucket  # under target — keep all
    topup_rows  = bucket[bucket["__origin__"] != "master"]
    master_rows = bucket[bucket["__origin__"] == "master"]
    n_to_keep_master = max(0, target_count - len(topup_rows))
    if n_to_keep_master >= len(master_rows):
        # topups + all masters still under target — won't actually trim anything from master
        return pd.concat([master_rows, topup_rows], ignore_index=True)
    if n_to_keep_master == 0:
        # too many topup rows alone; need to drop topups too. Sample randomly from topups.
        kept_topups = topup_rows.sample(n=target_count, random_state=seed)
        return kept_topups
    kept_master = master_rows.sample(n=n_to_keep_master, random_state=seed)
    return pd.concat([kept_master, topup_rows], ignore_index=True)


def process_source(source_label, master_path, topup_paths, target_total, target_per_class, seed, apply, snapshot_tag):
    print(f"\n=== {source_label.upper()} ({master_path.name}) ===")

    # Read master + topups (only those that exist)
    master_df = _read_with_origin(master_path, "master")
    _summarize(master_df, "master (before)")

    topup_dfs = []
    for p in _existing(topup_paths):
        d = _read_with_origin(p, p.name)
        _summarize(d, f"+ {p.name}")
        topup_dfs.append(d)

    missing = [p.name for p in topup_paths if not p.exists()]
    if missing:
        print(f"  MISSING (skipped): {missing}")

    # Drop columns that don't exist consistently across files (defensive)
    common_cols = set(master_df.columns)
    for d in topup_dfs:
        common_cols &= set(d.columns)
    common_cols = sorted(common_cols)
    master_df = master_df[common_cols]
    topup_dfs = [d[common_cols] for d in topup_dfs]

    combined = pd.concat([master_df] + topup_dfs, ignore_index=True)
    # Drop the rare "other"/NaN sample_type rows (1 each in some topups)
    valid = combined["sample_type"].isin(TARGET_RATIOS.keys())
    n_dropped_other = int((~valid).sum())
    combined = combined[valid].copy()
    if n_dropped_other:
        print(f"  dropped {n_dropped_other} rows with non-target sample_type")
    print()
    _summarize(combined, "combined (master+topups)")

    # Trim per class
    pieces = []
    for st, tgt in target_per_class.items():
        t = _trim_bucket(combined, st, tgt, seed)
        pieces.append(t)
    final = pd.concat(pieces, ignore_index=True)
    # Shuffle so the file isn't class-sorted
    final = final.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    final = final.drop(columns=["__origin__"])

    print()
    _summarize(final, f"FINAL {source_label}")
    print(f"  target totals: total={target_total}  per_class={target_per_class}")

    if apply:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        bak = master_path.with_name(master_path.stem + f".bak.{snapshot_tag}.{ts}.csv")
        print(f"\n  WRITING snapshot:    {bak.name}")
        master_path.rename(bak)  # atomic rename
        print(f"  WRITING new master:  {master_path.name}  ({len(final)} rows)")
        final.to_csv(master_path, index=False)
    else:
        print(f"\n  [dry-run] would snapshot then write {len(final)} rows to {master_path.name}")
    return len(final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually write files. Without this, only print what would happen.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snapshot-tag", default="before_internlm_merge",
                    help="Filename tag inserted into the .bak. snapshot.")
    args = ap.parse_args()

    if not PILE_MASTER.exists() or not CC_MASTER.exists():
        sys.exit(f"missing master file(s): {PILE_MASTER}, {CC_MASTER}")

    print(f"mode = {'APPLY' if args.apply else 'DRY-RUN'}    seed = {args.seed}")
    print(f"Pile target {PILE_TARGET_TOTAL}: {PILE_TARGETS}")
    print(f"CC   target {CC_TARGET_TOTAL}: {CC_TARGETS}")

    n_pile = process_source("pile", PILE_MASTER, PILE_TOPUPS,
                            PILE_TARGET_TOTAL, PILE_TARGETS, args.seed, args.apply, args.snapshot_tag)
    n_cc   = process_source("cc",   CC_MASTER,   CC_TOPUPS,
                            CC_TARGET_TOTAL,   CC_TARGETS,   args.seed, args.apply, args.snapshot_tag)

    print(f"\n=== TOTALS === Pile {n_pile} + CC {n_cc} = {n_pile + n_cc}")
    if not args.apply:
        print("(dry-run — no files modified. Pass --apply to commit.)")


if __name__ == "__main__":
    main()
