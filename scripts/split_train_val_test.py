"""Stratified 80/10/10 split of the merged 210K master CSVs into train,
val, and test files. Stratification key is (data_source, sample_type,
model_name) — every (source, class, generator-model) bucket is split
independently at the same ratio so train/val/test all carry the validator
distribution.

Outputs three files under data/MainData/by_source/splits/:
    train.csv  (~168K)
    val.csv    (~21K)
    test.csv   (~21K)

Reproducible via --seed (default 42). Pass --apply to actually write
files; default is dry-run.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/llm-detection/data/MainData")
BY_SRC = ROOT / "by_source"
SPLITS_DIR = BY_SRC / "splits"
PILE_MASTER = BY_SRC / "train_pile_with_adv.csv"
CC_MASTER   = BY_SRC / "train_common_crawl_with_adv.csv"

DEFAULT_RATIOS = (0.80, 0.10, 0.10)  # train, val, test
STRATIFY_KEY = ["data_source", "sample_type", "model_name"]


def stratified_split(df, ratios, seed):
    """Per-bucket split. Each (source, class, model) bucket is shuffled
    and partitioned at the requested ratios. Floor + remainder allocation
    so very small buckets don't lose a row to rounding."""
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []
    counts = []
    df = df.copy()
    df["model_name"] = df["model_name"].fillna("__none__")

    for keys, bucket in df.groupby(STRATIFY_KEY, sort=False):
        n = len(bucket)
        idx = bucket.index.to_numpy()
        rng.shuffle(idx)
        # floor counts then distribute remainders
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        n_test  = n - n_train - n_val
        # tiny buckets: ensure val & test get at least 1 row each if n>=3
        if n >= 3 and n_val == 0:
            n_val = 1
            if n_train + n_val + n_test > n:
                n_train -= 1
        if n >= 3 and n_test == 0:
            n_test = 1
            if n_train + n_val + n_test > n:
                n_train -= 1
        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train + n_val]
        test_idx  = idx[n_train + n_val:]
        train_parts.append(df.loc[train_idx])
        val_parts.append(df.loc[val_idx])
        test_parts.append(df.loc[test_idx])
        counts.append((keys, n, len(train_idx), len(val_idx), len(test_idx)))

    train = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val   = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test  = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, val, test, counts


def _summarize(df, label):
    counts = df["sample_type"].value_counts().to_dict()
    pct = (df["sample_type"].value_counts(normalize=True) * 100).round(2).to_dict()
    print(f"  {label:<10} total={len(df):>6}  "
          f"pure_h={counts.get('pure_human',0):>5} ({pct.get('pure_human',0):>5.2f}%)  "
          f"pure_ai={counts.get('pure_ai',0):>5} ({pct.get('pure_ai',0):>5.2f}%)  "
          f"h_then_ai={counts.get('human_then_ai',0):>5} ({pct.get('human_then_ai',0):>5.2f}%)  "
          f"ai_then_h={counts.get('ai_then_human',0):>5} ({pct.get('ai_then_human',0):>5.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually write split CSVs. Without this, only print stats.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    ap.add_argument("--val-ratio",   type=float, default=DEFAULT_RATIOS[1])
    ap.add_argument("--test-ratio",  type=float, default=DEFAULT_RATIOS[2])
    args = ap.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {sum(ratios):.4f}")

    print(f"mode = {'APPLY' if args.apply else 'DRY-RUN'}    seed = {args.seed}")
    print(f"ratios train/val/test = {ratios}")
    print(f"stratify key = {STRATIFY_KEY}")

    df_pile = pd.read_csv(PILE_MASTER)
    df_cc   = pd.read_csv(CC_MASTER)
    df = pd.concat([df_pile, df_cc], ignore_index=True)
    print(f"\nloaded: pile={len(df_pile)}  cc={len(df_cc)}  total={len(df)}")

    train, val, test, counts = stratified_split(df, ratios, args.seed)
    print(f"\nsplit produced: train={len(train)}  val={len(val)}  test={len(test)}  "
          f"sum={len(train)+len(val)+len(test)}  (input {len(df)})")

    print("\n=== distribution by sample_type ===")
    _summarize(train, "train")
    _summarize(val,   "val")
    _summarize(test,  "test")

    print("\n=== distribution by data_source ===")
    for label, d in [("train", train), ("val", val), ("test", test)]:
        c = d["data_source"].value_counts().to_dict()
        print(f"  {label:<10} pile={c.get('pile',0):>6} ({c.get('pile',0)/len(d)*100:>5.2f}%)  "
              f"cc={c.get('common_crawl',0):>6} ({c.get('common_crawl',0)/len(d)*100:>5.2f}%)")

    # report tightest 5 buckets so user can see edge-cases
    sm = sorted(counts, key=lambda x: x[1])[:5]
    print("\n=== smallest 5 buckets (sanity check) ===")
    for keys, n, nt, nv, nte in sm:
        print(f"  {keys}  n={n}  ->  train={nt}  val={nv}  test={nte}")

    if args.apply:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        for name, d in [("train", train), ("val", val), ("test", test)]:
            out = SPLITS_DIR / f"{name}.csv"
            d.to_csv(out, index=False)
            print(f"\n  WROTE {out}  ({len(d)} rows)")
    else:
        print(f"\n[dry-run — pass --apply to write to {SPLITS_DIR}/]")


if __name__ == "__main__":
    main()
