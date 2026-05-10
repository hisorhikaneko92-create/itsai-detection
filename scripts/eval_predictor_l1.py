"""Evaluate the L1 retrieval predictor against a labeled test set.

Mirrors the validator's reward.py scoring exactly so the output predicts what
gate-3 (OOD F1 EMA) and the in-domain reward will look like at deploy time.

For each row in the test CSV:
  1. Run L1Predictor → y_pred (per-word AI probabilities)
  2. Parse segmentation_labels → y_true (per-word ground truth 0/1)
  3. Compute fp_score / f1_score / ap_score / reward per row

Then aggregate by:
  - data_source ('pile' = in-domain reward driver, 'common_crawl' = OOD gate)
  - sample_type (pure_human / pure_ai / human_then_ai / ai_then_human)
  - overall

This is the closest possible offline preview of validator-side reward.

Usage:
    python scripts/eval_predictor_l1.py --csv data/adv_test.csv
    python scripts/eval_predictor_l1.py --min-run 2 --sample 1000
"""
from __future__ import annotations

import argparse
import ast
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor_l1 import L1Predictor


def per_row_metrics(y_pred: list[float], y_true: list[int]) -> dict:
    """Mirror detection/validator/reward.py:reward() exactly."""
    if len(y_pred) == 0 or len(y_pred) != len(y_true):
        return {"fp_score": 0.0, "f1_score": 0.0, "ap_score": 0.0,
                "reward": 0.0, "valid": False}
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=int)
    preds_rounded = np.round(y_pred_arr).astype(int)

    # f1 + fp need at least one positive and negative label
    try:
        cm = confusion_matrix(y_true_arr, preds_rounded, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fp_score = 1 - fp / max(len(y_pred_arr), 1)
        f1 = f1_score(y_true_arr, preds_rounded, zero_division=0)
    except Exception:
        fp_score = 0.0
        f1 = 0.0

    # ap_score needs both classes in truth
    try:
        if len(np.unique(y_true_arr)) >= 2:
            ap = average_precision_score(y_true_arr, y_pred_arr)
        else:
            ap = 1.0 if y_true_arr[0] == np.round(y_pred_arr.mean()).astype(int) else 0.0
    except Exception:
        ap = 0.0

    return {
        "fp_score": float(fp_score),
        "f1_score": float(f1),
        "ap_score": float(ap),
        "reward":   float((fp_score + f1 + ap) / 3.0),
        "valid":    True,
    }


def aggregate_metrics(y_preds_flat: list[float], y_trues_flat: list[int]) -> dict:
    """Same metrics, computed over flat-concatenated predictions/labels.
    This is exactly how the validator scores in reward.py:114-127 — predictions
    are concatenated across all texts in the batch."""
    if not y_preds_flat:
        return {"fp_score": 0.0, "f1_score": 0.0, "ap_score": 0.0,
                "reward": 0.0, "n_words": 0}
    return {**per_row_metrics(y_preds_flat, y_trues_flat),
            "n_words": len(y_preds_flat)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("data/adv_test.csv"))
    p.add_argument("--bloom", action="append", default=None, type=Path,
                   help="L1 Bloom shard path (repeat for each). "
                        "Default: indexes/pile_l1_shard{0,1,2,3}.bloom")
    p.add_argument("--min-run", type=int, default=3)
    p.add_argument("--sample", type=int, default=None,
                   help="Subsample N rows for faster runs (default: all)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.bloom is None:
        args.bloom = [Path(f"indexes/pile_l1_shard{i}.bloom") for i in range(4)]
    for b in args.bloom:
        if not b.exists():
            print(f"ERROR: missing {b}", file=sys.stderr); return 1
    if not args.csv.exists():
        print(f"ERROR: missing {args.csv}", file=sys.stderr); return 1

    print(f"Loading dataset: {args.csv}", file=sys.stderr)
    df = pd.read_csv(args.csv)
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)
    print(f"  {len(df)} rows", file=sys.stderr)

    print(f"Loading L1 predictor ({len(args.bloom)} shards, min_run={args.min_run})...",
          file=sys.stderr)
    t0 = time.time()
    pred = L1Predictor(args.bloom, min_run=args.min_run)
    print(f"  loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    # Per-row evaluation; bucket by data_source and sample_type
    by_source: dict[str, list[dict]] = defaultdict(list)
    by_sample: dict[str, list[dict]] = defaultdict(list)
    overall_pred: list[float] = []
    overall_true: list[int] = []
    flat_by_source: dict[str, tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
    flat_by_sample: dict[str, tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
    skipped = 0
    bad_align = 0

    t0 = time.time()
    for i, row in df.iterrows():
        text = row.get("text") or ""
        labels_str = row.get("segmentation_labels") or "[]"
        try:
            y_true = ast.literal_eval(labels_str)
        except Exception:
            skipped += 1; continue
        if not isinstance(y_true, list) or not text:
            skipped += 1; continue

        y_pred = pred.predict(text)
        if len(y_pred) != len(y_true):
            # Word-count mismatch (predictor uses text.split(), labels were from
            # build-time tokenization). Skip with note.
            bad_align += 1; continue

        m = per_row_metrics(y_pred, y_true)
        if not m["valid"]:
            skipped += 1; continue

        src = str(row.get("data_source") or "unknown")
        smp = str(row.get("sample_type") or "unknown")
        by_source[src].append(m)
        by_sample[smp].append(m)
        overall_pred.extend(y_pred); overall_true.extend(y_true)
        fp_s, ft_s = flat_by_source[src]; fp_s.extend(y_pred); ft_s.extend(y_true)
        fp_t, ft_t = flat_by_sample[smp]; fp_t.extend(y_pred); ft_t.extend(y_true)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(df)} rows  ({(i+1)/elapsed:.0f} rows/sec)",
                  file=sys.stderr)

    elapsed = time.time() - t0
    n_eval = len(overall_pred) and sum(1 for _ in by_source.values() for _ in _)
    n_eval = sum(len(v) for v in by_source.values())
    print(f"\nDone in {elapsed:.1f}s "
          f"({n_eval} rows evaluated, {skipped} skipped, "
          f"{bad_align} word-count mismatch)",
          file=sys.stderr)

    # Reporting helpers ---------------------------------------------------
    def fmt_block(title: str, rows: list[dict], flat: tuple[list[float], list[int]]):
        if not rows:
            return
        print(f"\n=== {title} (n_rows={len(rows)}) ===")
        # Per-row mean metrics
        f1 = np.mean([r["f1_score"] for r in rows])
        fp = np.mean([r["fp_score"] for r in rows])
        ap = np.mean([r["ap_score"] for r in rows])
        rw = np.mean([r["reward"]   for r in rows])
        print(f"  Per-row avg:   f1={f1:.4f}  fp_score={fp:.4f}  ap={ap:.4f}  reward={rw:.4f}")
        # Validator-style flat-concat metrics
        flat_m = aggregate_metrics(flat[0], flat[1])
        print(f"  Validator-eq:  f1={flat_m['f1_score']:.4f}  "
              f"fp_score={flat_m['fp_score']:.4f}  "
              f"ap={flat_m['ap_score']:.4f}  reward={flat_m['reward']:.4f}")

    # --- per data_source (this is the GATE-3 lens) ---
    print()
    print("=" * 72)
    print("BY data_source  (Pile = in-domain reward driver, CC = gate-3 EMA)")
    print("=" * 72)
    for src in sorted(by_source.keys()):
        fmt_block(f"data_source = {src}", by_source[src], flat_by_source[src])

    # --- per sample_type ---
    print()
    print("=" * 72)
    print("BY sample_type")
    print("=" * 72)
    for smp in sorted(by_sample.keys()):
        fmt_block(f"sample_type = {smp}", by_sample[smp], flat_by_sample[smp])

    # --- combined Pile × sample_type cross-tab (matters for in-domain reward) ---
    print()
    print("=" * 72)
    print("Pile × sample_type cross-tab  (the slice that drives miner_reward)")
    print("=" * 72)
    for smp in ["pure_human", "human_then_ai", "ai_then_human", "pure_ai"]:
        pred_chunk: list[float] = []; true_chunk: list[int] = []
        for src in ("pile",):
            n_rows = 0
            for i, row in df.iterrows():
                if str(row.get("data_source")) != src or str(row.get("sample_type")) != smp:
                    continue
                try:
                    y_true = ast.literal_eval(row["segmentation_labels"])
                except Exception:
                    continue
                if not isinstance(y_true, list):
                    continue
                y_pred = pred.predict(row["text"])
                if len(y_pred) != len(y_true):
                    continue
                pred_chunk.extend(y_pred); true_chunk.extend(y_true); n_rows += 1
        if not pred_chunk:
            continue
        flat_m = aggregate_metrics(pred_chunk, true_chunk)
        print(f"  pile/{smp:>15}:  f1={flat_m['f1_score']:.4f}  "
              f"fp={flat_m['fp_score']:.4f}  ap={flat_m['ap_score']:.4f}  "
              f"reward={flat_m['reward']:.4f}  (n_words={flat_m['n_words']})")

    # --- overall ---
    print()
    print("=" * 72)
    print("OVERALL")
    print("=" * 72)
    flat_m = aggregate_metrics(overall_pred, overall_true)
    print(f"  Validator-eq flat: f1={flat_m['f1_score']:.4f}  "
          f"fp_score={flat_m['fp_score']:.4f}  "
          f"ap={flat_m['ap_score']:.4f}  reward={flat_m['reward']:.4f}  "
          f"n_words={flat_m['n_words']}")
    print()
    print(">>> The CC f1_score above is the closest predictor of gate-3 OOD F1.")
    print(">>> The pile reward above is the closest predictor of validator-side reward.")
    print(">>> Both assume penalty=1 (gate-2 passes deterministically with retrieval).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
