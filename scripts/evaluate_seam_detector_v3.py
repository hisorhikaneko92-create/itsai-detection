"""
Evaluate a trained HSSD v3 (SeamDetector) checkpoint on held-out CSVs.

Uses sliding-window inference by default (window_size=512, stride=256)
via HSSDPredictor, so the metrics measured here are exactly what the
miner will see at production inference time.

Reports:
  * Overall: mean_seam_offset, f1@5 words, accuracy@3 / @10, token-F1
  * Stratified by sample_type, seam_bucket, data_source, model_name
  * Long-input tracker: how many test rows triggered the sliding window

Usage:
    python scripts/evaluate_seam_detector_v3.py \
        --checkpoint-dir models/seam_detector_v3/best \
        --test-csv data/Training_Dataset/test.csv \
        --output-dir models/seam_detector_v3/eval

    # Single-window mode (faster, but truncates long inputs):
    python scripts/evaluate_seam_detector_v3.py \
        --checkpoint-dir models/seam_detector_v3/best \
        --test-csv data/Training_Dataset/test.csv \
        --no-sliding-window
"""
# Pre-import for Windows stack-overflow workaround.
import pandas  # noqa: F401
import sklearn  # noqa: F401

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_document import HSSDPredictor   # noqa: E402


# ---------------------------------------------------------------------------
# Metrics (re-exported / copied from train_seam_detector.py to keep this
# file self-contained -- evaluation should not depend on the training
# script's internals beyond the model / predictor).
# ---------------------------------------------------------------------------
def _first_transition(arr: List[int]) -> Optional[int]:
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            return i
    return None


def _seam_bucket(labels: List[int]) -> str:
    """Same logic as split_dataset.py — categorize the seam by its
    fractional position so we can stratify metrics."""
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


def compute_seam_metrics(gt_arrays: List[List[int]],
                         pred_arrays: List[List[int]]) -> Dict[str, float]:
    offsets: List[int] = []
    correct_at_5 = 0
    total = 0
    for gt, pred in zip(gt_arrays, pred_arrays):
        total += 1
        gt_seam = _first_transition(gt)
        pred_seam = _first_transition(pred)
        if gt_seam is None and pred_seam is None:
            correct_at_5 += 1
            continue
        if gt_seam is None or pred_seam is None:
            continue
        dist = abs(gt_seam - pred_seam)
        offsets.append(dist)
        if dist <= 5:
            correct_at_5 += 1
    return {
        "mean_seam_offset": float(np.mean(offsets)) if offsets else float("nan"),
        "f1_at_5":          correct_at_5 / max(1, total),
        "n_with_seam":      len(offsets),
        "n_total":          total,
    }


def compute_token_f1(gt_arrays: List[List[int]],
                     pred_arrays: List[List[int]]) -> float:
    from sklearn.metrics import f1_score
    flat_gt: List[int] = []
    flat_pr: List[int] = []
    for gt, pr in zip(gt_arrays, pred_arrays):
        n = min(len(gt), len(pr))
        flat_gt.extend(gt[:n])
        flat_pr.extend(pr[:n])
    if not flat_gt:
        return 0.0
    return float(f1_score(flat_gt, flat_pr, zero_division=0))


def _accuracy_at_k(gt_arrays: List[List[int]],
                   pred_arrays: List[List[int]], k: int) -> float:
    correct = 0
    total = 0
    for gt, pred in zip(gt_arrays, pred_arrays):
        total += 1
        gt_seam = _first_transition(gt)
        pr_seam = _first_transition(pred)
        if gt_seam is None and pr_seam is None:
            correct += 1
        elif gt_seam is not None and pr_seam is not None:
            if abs(gt_seam - pr_seam) <= k:
                correct += 1
    return correct / max(1, total)


def _metrics_for(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {"n": 0}
    gts = [r["word_labels"] for r in rows]
    prs = [r["pred_word_labels"] for r in rows]
    seam = compute_seam_metrics(gts, prs)
    return {
        "n":                len(rows),
        "n_with_seam":      seam["n_with_seam"],
        "mean_seam_offset": seam["mean_seam_offset"],
        "f1_at_5":          seam["f1_at_5"],
        "accuracy_at_3":    _accuracy_at_k(gts, prs, 3),
        "accuracy_at_10":   _accuracy_at_k(gts, prs, 10),
        "token_f1":         compute_token_f1(gts, prs),
    }


def _print_metrics_row(label: str, m: Dict[str, float], width: int = 28) -> None:
    if m.get("n", 0) == 0:
        print(f"  {label:<{width}s}  (empty)")
        return
    seam_off = m["mean_seam_offset"]
    seam_str = f"{seam_off:6.2f}" if not math.isnan(seam_off) else "  n/a "
    print(f"  {label:<{width}s}  n={m['n']:>5d}  "
          f"seam={seam_str}  "
          f"f1@5={m['f1_at_5']:.3f}  "
          f"acc@3={m['accuracy_at_3']:.3f}  "
          f"acc@10={m['accuracy_at_10']:.3f}  "
          f"tok_f1={m['token_f1']:.3f}")


def _stratified(rows: List[Dict], key: str) -> Dict[str, Dict]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key, "?"))].append(r)
    return {k: _metrics_for(rs)
            for k, rs in sorted(groups.items(), key=lambda kv: -len(kv[1]))}


# ---------------------------------------------------------------------------
# Loading + inference loop
# ---------------------------------------------------------------------------
def evaluate_csv(predictor: HSSDPredictor,
                 csv_path: Path,
                 max_rows: Optional[int] = None,
                 progress_every: int = 200) -> List[Dict]:
    """Predict each row of `csv_path` and return per-row results with both
    ground-truth and predicted word labels + the metadata needed for
    stratified reporting."""
    results: List[Dict] = []
    n_total = 0
    n_long = 0
    n_skipped = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if max_rows and n_total >= max_rows:
                break
            text = row.get("text", "") or ""
            try:
                word_labels = json.loads(row.get("segmentation_labels", "[]"))
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            if not text or not isinstance(word_labels, list):
                n_skipped += 1
                continue
            words = text.split()
            if len(words) != len(word_labels):
                n_skipped += 1
                continue

            # Quick token count check just to track sliding-window usage.
            n_tokens = len(predictor.tokenizer(
                words,
                is_split_into_words=True,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"])
            if n_tokens > predictor.window_size:
                n_long += 1

            pred = predictor.predict(text)
            if len(pred) != len(word_labels):
                # Defensive: should never happen, but if it does, pad/clip.
                if len(pred) < len(word_labels):
                    pred = pred + [0] * (len(word_labels) - len(pred))
                else:
                    pred = pred[:len(word_labels)]

            results.append({
                "word_labels":      list(map(int, word_labels)),
                "pred_word_labels": pred,
                "sample_type":      row.get("sample_type", "?"),
                "data_source":      row.get("data_source", "?"),
                "model_name":       row.get("model_name", "?"),
                "seam_bucket":      _seam_bucket(word_labels),
                "n_words":          len(word_labels),
                "n_tokens":         n_tokens,
                "long_input":       n_tokens > predictor.window_size,
                "text_preview":     text[:160],
            })
            n_total += 1
            if n_total % progress_every == 0:
                print(f"  predicted {n_total:>6}  "
                      f"({n_long} long-input via sliding window)")

    if n_skipped:
        print(f"  skipped {n_skipped} malformed rows")
    print(f"  total processed: {n_total}  "
          f"({n_long} = {100*n_long/max(n_total,1):.1f}% needed sliding window)")
    return results


# ---------------------------------------------------------------------------
# Worst-failure dump (for manual inspection)
# ---------------------------------------------------------------------------
def dump_worst_failures(results: List[Dict], n: int, out_path: Path) -> None:
    def offset(r: Dict) -> int:
        gt_seam = _first_transition(r["word_labels"])
        pr_seam = _first_transition(r["pred_word_labels"])
        if gt_seam is None and pr_seam is None:
            return 0
        if gt_seam is None or pr_seam is None:
            return r["n_words"]   # max penalty
        return abs(gt_seam - pr_seam)

    ranked = sorted(results, key=offset, reverse=True)[:n]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "offset", "sample_type", "seam_bucket", "data_source",
            "model_name", "n_words", "n_tokens", "long_input",
            "gt_first_seam", "pred_first_seam",
            "gt_labels", "pred_labels", "text_preview",
        ])
        for r in ranked:
            writer.writerow([
                offset(r),
                r["sample_type"], r["seam_bucket"], r["data_source"],
                r["model_name"], r["n_words"], r["n_tokens"], r["long_input"],
                _first_transition(r["word_labels"]),
                _first_transition(r["pred_word_labels"]),
                json.dumps(r["word_labels"]),
                json.dumps(r["pred_word_labels"]),
                r["text_preview"],
            ])
    print(f"\nWrote {len(ranked)} worst failures to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint-dir", required=True,
                    help="HSSD v3 checkpoint dir (e.g. "
                         "models/seam_detector_v3/best). Must contain "
                         "lora_adapter/ or full_model.pth.")
    ap.add_argument("--test-csv", nargs="+", required=True,
                    help="One or more test CSVs in build_training_dataset.py format.")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write metrics.json (and worst_failures.csv "
                         "if requested). Default: <checkpoint-dir>/eval_v3.")
    ap.add_argument("--base-model", default="microsoft/deberta-v3-large",
                    help="Base model for tokenizer + DeBERTa weights. "
                         "Must match what the checkpoint was trained on.")
    ap.add_argument("--window-size", type=int, default=512,
                    help="Token window size. Default 512 (DeBERTa's "
                         "architectural ceiling).")
    ap.add_argument("--stride", type=int, default=256,
                    help="Token stride between sliding windows. Default 256 "
                         "(50%% overlap; doc default).")
    ap.add_argument("--no-sliding-window", action="store_true",
                    help="Disable sliding window. Long inputs will be "
                         "truncated to window_size by the tokenizer. Faster "
                         "but the metrics no longer match production "
                         "inference -- use only for debugging.")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dump-worst-failures", type=int, default=0,
                    help="If > 0, write the N worst-seam-offset examples "
                         "to <output-dir>/worst_failures.csv.")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint_dir)
    if not ckpt.exists():
        sys.exit(f"Checkpoint dir not found: {ckpt}")

    out_dir = Path(args.output_dir) if args.output_dir else (ckpt / "eval_v3")
    out_dir.mkdir(parents=True, exist_ok=True)

    # If --no-sliding-window is set, set stride >= window_size so each
    # window stands alone and the "long input" branch effectively
    # truncates (we still call predict_document but it'll only produce
    # the first window's labels).
    effective_stride = args.window_size if args.no_sliding_window else args.stride

    print(f"Loading checkpoint from {ckpt}...")
    predictor = HSSDPredictor(
        model_dir=str(ckpt),
        base_model=args.base_model,
        device=args.device,
        window_size=args.window_size,
        stride=effective_stride,
    )
    print(f"  window_size={args.window_size}  stride={effective_stride}  "
          f"sliding_window={'OFF (debug)' if args.no_sliding_window else 'ON'}")

    # Run inference on every test CSV
    all_results: List[Dict] = []
    for path_str in args.test_csv:
        path = Path(path_str)
        print(f"\nEvaluating {path} ...")
        all_results.extend(evaluate_csv(
            predictor, path, max_rows=args.max_rows,
        ))

    # Overall + stratified reports
    print("\n" + "=" * 80)
    print(f" Overall metrics on {len(all_results):,} examples")
    print("=" * 80)
    overall = _metrics_for(all_results)
    _print_metrics_row("ALL", overall)

    print("\n--- by sample_type ---")
    for k, m in _stratified(all_results, "sample_type").items():
        _print_metrics_row(k, m)

    print("\n--- by seam_bucket ---")
    for k, m in _stratified(all_results, "seam_bucket").items():
        _print_metrics_row(k, m)

    print("\n--- by data_source ---")
    for k, m in _stratified(all_results, "data_source").items():
        _print_metrics_row(k, m)

    print("\n--- by model_name (top 12 by count) ---")
    for k, m in list(_stratified(all_results, "model_name").items())[:12]:
        _print_metrics_row(k, m)

    long_only = [r for r in all_results if r.get("long_input")]
    if long_only:
        print(f"\n--- LONG INPUTS only (>{args.window_size} tokens, "
              f"sliding window fired) ---")
        _print_metrics_row("long_inputs", _metrics_for(long_only))

    # Dump metrics + worst failures
    metrics_payload = {
        "checkpoint":   str(ckpt),
        "test_csvs":    args.test_csv,
        "n_examples":   len(all_results),
        "n_long":       len(long_only),
        "window_size":  args.window_size,
        "stride":       effective_stride,
        "sliding_window": not args.no_sliding_window,
        "overall":      overall,
        "by_sample_type": _stratified(all_results, "sample_type"),
        "by_seam_bucket": _stratified(all_results, "seam_bucket"),
        "by_data_source": _stratified(all_results, "data_source"),
        "by_model_name":  _stratified(all_results, "model_name"),
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, default=str)
    print(f"\nWrote {metrics_path}")

    if args.dump_worst_failures:
        dump_worst_failures(
            all_results, args.dump_worst_failures,
            out_dir / "worst_failures.csv",
        )


if __name__ == "__main__":
    main()
