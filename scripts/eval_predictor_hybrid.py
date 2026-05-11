"""Evaluate the hybrid L1+HSSD predictor against the labeled test set.

Same metrics as eval_predictor_l1.py but uses HybridPredictor. The numbers
here directly predict what the validator will score after deployment:

  - per-data_source f1/fp/ap/reward  — Pile = miner_reward driver,
                                       CC   = gate-3 OOD EMA driver
  - per-sample_type breakdown        — which validator text categories are
                                       strong/weak
  - Pile × sample_type cross-tab     — the slice that directly drives reward

Usage:
    python scripts/eval_predictor_hybrid.py
    python scripts/eval_predictor_hybrid.py --sample 1500
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
from predictor_hybrid import HybridPredictor


def per_row_metrics(y_pred, y_true):
    if len(y_pred) == 0 or len(y_pred) != len(y_true):
        return {"fp_score": 0.0, "f1_score": 0.0, "ap_score": 0.0,
                "reward": 0.0, "valid": False}
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=int)
    preds_rounded = np.round(y_pred_arr).astype(int)
    try:
        cm = confusion_matrix(y_true_arr, preds_rounded, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fp_score = 1 - fp / max(len(y_pred_arr), 1)
        f1 = f1_score(y_true_arr, preds_rounded, zero_division=0)
    except Exception:
        fp_score = 0.0; f1 = 0.0
    try:
        if len(np.unique(y_true_arr)) >= 2:
            ap = average_precision_score(y_true_arr, y_pred_arr)
        else:
            ap = 1.0 if y_true_arr[0] == np.round(y_pred_arr.mean()).astype(int) else 0.0
    except Exception:
        ap = 0.0
    return {"fp_score": float(fp_score), "f1_score": float(f1),
            "ap_score": float(ap),
            "reward":   float((fp_score + f1 + ap) / 3.0),
            "valid":    True}


def aggregate_metrics(y_preds_flat, y_trues_flat):
    if not y_preds_flat:
        return {"fp_score": 0.0, "f1_score": 0.0, "ap_score": 0.0,
                "reward": 0.0, "n_words": 0}
    return {**per_row_metrics(y_preds_flat, y_trues_flat),
            "n_words": len(y_preds_flat)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("data/adv_test.csv"))
    p.add_argument("--bloom", action="append", default=None, type=Path,
                   help="L1 Bloom shard. Repeat for each.")
    p.add_argument("--hssd-model-dir", type=Path,
                   default=Path("models/best"),
                   help="HSSD model dir containing lora_adapter/")
    p.add_argument("--hssd-base-model", default="microsoft/deberta-v3-large")
    p.add_argument("--hssd-device", default=None)
    p.add_argument("--min-run", type=int, default=3)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32,
                   help="HSSD inner batch size")
    return p.parse_args()


def main():
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

    print(f"Initializing HybridPredictor "
          f"(L1: {len(args.bloom)} shards, min_run={args.min_run}; "
          f"HSSD: {args.hssd_model_dir})...", file=sys.stderr)
    t0 = time.time()
    pred = HybridPredictor(
        l1_blooms=args.bloom,
        hssd_model_dir=args.hssd_model_dir,
        hssd_base_model=args.hssd_base_model,
        hssd_device=args.hssd_device,
        l1_min_run=args.min_run,
    )
    print(f"  loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    # Process in batches so HSSD's GPU batching works
    by_source: dict = defaultdict(list)
    by_sample: dict = defaultdict(list)
    overall_pred: list[float] = []
    overall_true: list[int] = []
    flat_by_source: dict = defaultdict(lambda: ([], []))
    flat_by_sample: dict = defaultdict(lambda: ([], []))
    skipped = 0
    bad_align = 0

    t0 = time.time()
    BATCH = args.batch_size
    for start in range(0, len(df), BATCH):
        batch = df.iloc[start:start + BATCH]
        texts = batch["text"].tolist()
        labels_strs = batch["segmentation_labels"].tolist()
        sources = batch["data_source"].astype(str).tolist()
        samples = batch["sample_type"].astype(str).tolist()

        # Parse labels first; skip bad rows
        truths = []
        valid_idx = []
        for i, ls in enumerate(labels_strs):
            try:
                y_true = ast.literal_eval(ls) if isinstance(ls, str) else None
                if isinstance(y_true, list) and texts[i]:
                    truths.append(y_true); valid_idx.append(i)
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        if not valid_idx:
            continue

        texts_valid = [texts[i] for i in valid_idx]
        preds_batch = pred.predict_batch(texts_valid)

        for bi, real_i in enumerate(valid_idx):
            y_pred = preds_batch[bi]
            y_true = truths[bi]
            if len(y_pred) != len(y_true):
                bad_align += 1; continue
            m = per_row_metrics(y_pred, y_true)
            if not m["valid"]:
                skipped += 1; continue
            src = sources[real_i]; smp = samples[real_i]
            by_source[src].append(m); by_sample[smp].append(m)
            overall_pred.extend(y_pred); overall_true.extend(y_true)
            fp_s, ft_s = flat_by_source[src]; fp_s.extend(y_pred); ft_s.extend(y_true)
            fp_t, ft_t = flat_by_sample[smp]; fp_t.extend(y_pred); ft_t.extend(y_true)

        if (start + BATCH) % 500 == 0 or start == 0:
            elapsed = time.time() - t0
            rate = (start + len(batch)) / max(elapsed, 1e-9)
            print(f"  {start + len(batch)}/{len(df)} rows  ({rate:.1f} rows/sec)",
                  file=sys.stderr)

    elapsed = time.time() - t0
    n_eval = sum(len(v) for v in by_source.values())
    print(f"\nDone in {elapsed:.1f}s "
          f"({n_eval} rows evaluated, {skipped} skipped, "
          f"{bad_align} word-count mismatch)",
          file=sys.stderr)

    def fmt_block(title, rows, flat):
        if not rows: return
        print(f"\n=== {title} (n_rows={len(rows)}) ===")
        f1 = np.mean([r["f1_score"] for r in rows])
        fp = np.mean([r["fp_score"] for r in rows])
        ap = np.mean([r["ap_score"] for r in rows])
        rw = np.mean([r["reward"]   for r in rows])
        print(f"  Per-row avg:   f1={f1:.4f}  fp_score={fp:.4f}  ap={ap:.4f}  reward={rw:.4f}")
        flat_m = aggregate_metrics(flat[0], flat[1])
        print(f"  Validator-eq:  f1={flat_m['f1_score']:.4f}  "
              f"fp_score={flat_m['fp_score']:.4f}  "
              f"ap={flat_m['ap_score']:.4f}  reward={flat_m['reward']:.4f}")

    print()
    print("=" * 72)
    print("BY data_source")
    print("=" * 72)
    for src in sorted(by_source.keys()):
        fmt_block(f"data_source = {src}", by_source[src], flat_by_source[src])

    print()
    print("=" * 72)
    print("BY sample_type")
    print("=" * 72)
    for smp in sorted(by_sample.keys()):
        fmt_block(f"sample_type = {smp}", by_sample[smp], flat_by_sample[smp])

    print()
    print("=" * 72)
    print("Pile × sample_type cross-tab")
    print("=" * 72)
    for smp in ["pure_human", "human_then_ai", "ai_then_human", "pure_ai"]:
        pred_chunk: list[float] = []; true_chunk: list[int] = []
        for i, row in df.iterrows():
            if str(row.get("data_source")) != "pile" or str(row.get("sample_type")) != smp:
                continue
            try:
                y_true = ast.literal_eval(row["segmentation_labels"])
            except Exception:
                continue
            if not isinstance(y_true, list): continue
            y_pred = pred.predict_one(row["text"])
            if len(y_pred) != len(y_true): continue
            pred_chunk.extend(y_pred); true_chunk.extend(y_true)
        if not pred_chunk: continue
        flat_m = aggregate_metrics(pred_chunk, true_chunk)
        print(f"  pile/{smp:>15}:  f1={flat_m['f1_score']:.4f}  "
              f"fp={flat_m['fp_score']:.4f}  ap={flat_m['ap_score']:.4f}  "
              f"reward={flat_m['reward']:.4f}  (n_words={flat_m['n_words']})")

    print()
    print("=" * 72)
    print("OVERALL")
    print("=" * 72)
    flat_m = aggregate_metrics(overall_pred, overall_true)
    print(f"  Validator-eq flat: f1={flat_m['f1_score']:.4f}  "
          f"fp_score={flat_m['fp_score']:.4f}  "
          f"ap={flat_m['ap_score']:.4f}  reward={flat_m['reward']:.4f}  "
          f"n_words={flat_m['n_words']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
