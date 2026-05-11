"""Extract per-word features for the LightGBM ensemble meta-learner.

Runs L1 + HSSD on every row of a CSV, then computes 13 features per word
(plus the ground-truth label and metadata). Output is a compressed .npz
file ready for the training script.

Features per word:
   1.  l1_pred                : L1 raw output
   2.  hssd_pred              : HSSD raw output
   3.  l1_run_length          : length of current L1 matched run
   4.  l1_neighbor_count_3    : L1 matches in ±3 word window
   5.  l1_neighbor_count_5    : L1 matches in ±5 word window
   6.  l1_match_fraction      : doc-level L1 match fraction
   7.  hssd_neighbor_avg_3    : average HSSD in ±3
   8.  hssd_neighbor_avg_5    : average HSSD in ±5
   9.  hssd_neighbor_min_3    : min HSSD in ±3
  10.  hssd_neighbor_max_3    : max HSSD in ±3
  11.  word_position_rel      : (word_idx + 0.5) / n_words
  12.  n_words                : total words in doc
  13.  l1_hssd_disagree       : |l1 - hssd|

Usage:
    python scripts/extract_ensemble_features.py \\
        --csv data/MainData/by_source/splits_subsampled/val.csv \\
        --sample 15000 \\
        --out /tmp/ensemble_features_val.npz
"""
from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor_l1 import L1Predictor
from predict_document import HSSDPredictor


def compute_features(
    l1_pred: list[float],
    hssd_pred: list[float],
    n_words: int,
) -> np.ndarray:
    """Compute the 13-feature matrix for one document. Returns shape [n_words, 13]."""
    l1 = np.asarray(l1_pred, dtype=np.float32)
    hs = np.asarray(hssd_pred, dtype=np.float32)
    n = len(l1)
    assert n == n_words == len(hs)

    # Boolean: was this word L1-matched (< 0.5)?
    l1_match = (l1 < 0.5).astype(np.int32)

    # run length: for each position, the length of the matched run it belongs to
    run_lengths = np.zeros(n, dtype=np.int32)
    i = 0
    while i < n:
        if l1_match[i]:
            j = i
            while j < n and l1_match[j]:
                j += 1
            run_lengths[i:j] = j - i
            i = j
        else:
            i += 1

    # Neighbor windows for L1 and HSSD (sliding sums via cumulative sums)
    cum_match = np.concatenate(([0], np.cumsum(l1_match)))
    cum_hssd  = np.concatenate(([0], np.cumsum(hs, dtype=np.float64)))

    def window_count(idx: int, half: int) -> int:
        lo, hi = max(0, idx - half), min(n, idx + half + 1)
        return int(cum_match[hi] - cum_match[lo])

    def window_avg(idx: int, half: int) -> float:
        lo, hi = max(0, idx - half), min(n, idx + half + 1)
        return float((cum_hssd[hi] - cum_hssd[lo]) / max(hi - lo, 1))

    # Min/max via Python loops (n is small per doc; total ops still fast)
    def window_min(arr: np.ndarray, idx: int, half: int) -> float:
        lo, hi = max(0, idx - half), min(n, idx + half + 1)
        return float(arr[lo:hi].min()) if hi > lo else 0.0

    def window_max(arr: np.ndarray, idx: int, half: int) -> float:
        lo, hi = max(0, idx - half), min(n, idx + half + 1)
        return float(arr[lo:hi].max()) if hi > lo else 0.0

    l1_match_fraction = float(l1_match.mean()) if n else 0.0

    feats = np.zeros((n, 13), dtype=np.float32)
    for idx in range(n):
        feats[idx, 0]  = l1[idx]
        feats[idx, 1]  = hs[idx]
        feats[idx, 2]  = run_lengths[idx]
        feats[idx, 3]  = window_count(idx, 3)
        feats[idx, 4]  = window_count(idx, 5)
        feats[idx, 5]  = l1_match_fraction
        feats[idx, 6]  = window_avg(idx, 3)
        feats[idx, 7]  = window_avg(idx, 5)
        feats[idx, 8]  = window_min(hs, idx, 3)
        feats[idx, 9]  = window_max(hs, idx, 3)
        feats[idx, 10] = (idx + 0.5) / max(n, 1)
        feats[idx, 11] = n
        feats[idx, 12] = abs(l1[idx] - hs[idx])
    return feats


FEATURE_NAMES = [
    "l1_pred", "hssd_pred", "l1_run_length",
    "l1_neighbor_count_3", "l1_neighbor_count_5", "l1_match_fraction",
    "hssd_neighbor_avg_3", "hssd_neighbor_avg_5",
    "hssd_neighbor_min_3", "hssd_neighbor_max_3",
    "word_position_rel", "n_words", "l1_hssd_disagree",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--bloom", action="append", default=None, type=Path)
    p.add_argument("--hssd-model-dir", type=Path, default=Path("models/best"))
    p.add_argument("--hssd-base-model", default="microsoft/deberta-v3-large")
    p.add_argument("--min-run", type=int, default=3)
    p.add_argument("--sample", type=int, default=None,
                   help="Subsample N rows for speed (default: all)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="HSSD inner batch size")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.bloom is None:
        args.bloom = [Path(f"indexes/pile_l1_shard{i}.bloom") for i in range(4)]
    if not args.csv.exists():
        print(f"ERROR: missing {args.csv}", file=sys.stderr); return 1

    print(f"Loading {args.csv}...", file=sys.stderr)
    df = pd.read_csv(args.csv)
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
    print(f"  {len(df)} rows", file=sys.stderr)

    print(f"Loading L1 ({len(args.bloom)} shards)...", file=sys.stderr)
    t0 = time.time()
    l1 = L1Predictor(args.bloom, min_run=args.min_run)
    print(f"  L1 loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    print(f"Loading HSSD ({args.hssd_model_dir})...", file=sys.stderr)
    t0 = time.time()
    hssd = HSSDPredictor(model_dir=str(args.hssd_model_dir),
                          base_model=args.hssd_base_model)
    print(f"  HSSD loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    # Accumulators
    X_chunks = []
    y_chunks = []
    doc_id_chunks = []
    source_chunks = []
    sample_type_chunks = []

    t0 = time.time()
    BATCH = args.batch_size
    for start in range(0, len(df), BATCH):
        batch = df.iloc[start:start + BATCH]
        texts = batch["text"].tolist()
        truths = []
        valid_idx = []
        for j, ls in enumerate(batch["segmentation_labels"].tolist()):
            try:
                y_true = ast.literal_eval(ls)
                if isinstance(y_true, list) and texts[j]:
                    truths.append(y_true); valid_idx.append(j)
                else:
                    truths.append(None)
            except Exception:
                truths.append(None)

        if not valid_idx:
            continue

        # HSSD batched call (efficient)
        valid_texts = [texts[j] for j in valid_idx]
        hssd_preds_batch = hssd.predict_batch_with_probs(
            valid_texts, max_batch_size=BATCH,
        )

        # L1 per-text (microseconds, no benefit to batching)
        for bi, real_j in enumerate(valid_idx):
            text = texts[real_j]
            y_true = truths[real_j]
            hssd_pred = hssd_preds_batch[bi]
            l1_pred = l1.predict(text)
            if not (len(l1_pred) == len(hssd_pred) == len(y_true)):
                continue

            n_words = len(l1_pred)
            feats = compute_features(l1_pred, hssd_pred, n_words)
            X_chunks.append(feats)
            y_chunks.append(np.asarray(y_true, dtype=np.int32))
            doc_id_chunks.append(
                np.full(n_words, start + real_j, dtype=np.int32)
            )
            source_chunks.append(
                np.full(n_words, str(batch.iloc[real_j]["data_source"]),
                        dtype="<U16")
            )
            sample_type_chunks.append(
                np.full(n_words, str(batch.iloc[real_j]["sample_type"]),
                        dtype="<U16")
            )

        if (start + BATCH) % (BATCH * 20) == 0:
            elapsed = time.time() - t0
            done = start + len(batch)
            rate = done / max(elapsed, 1e-9)
            eta = (len(df) - done) / max(rate, 1e-9)
            print(f"  {done}/{len(df)}  {rate:.1f} rows/s  ETA {eta/60:.1f}min",
                  file=sys.stderr)

    # Concatenate
    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    doc_id = np.concatenate(doc_id_chunks, axis=0)
    source = np.concatenate(source_chunks, axis=0)
    sample_type = np.concatenate(sample_type_chunks, axis=0)

    print(f"\nFeatures shape: {X.shape}", file=sys.stderr)
    print(f"Labels shape:   {y.shape}", file=sys.stderr)
    print(f"Pos/neg ratio:  {y.mean():.3f}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X, y=y, doc_id=doc_id, source=source, sample_type=sample_type,
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"\nSaved {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
