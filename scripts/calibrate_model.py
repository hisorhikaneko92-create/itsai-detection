"""
Post-training calibration of the HSSD v4 emission γ-shift.

What this does
--------------
After training, the model's raw emissions can be slightly mis-calibrated
along the AI/human axis — e.g. the model may be confident enough that
P(AI) ≈ 0.55 wins on borderline tokens that actually should be human.
Adding a constant γ to emissions[..., 1] (the AI-class logit) before
CRF decode shifts that decision boundary cleanly without retraining.

We sweep γ ∈ [-3, +3] in 0.1 increments, run the full validation set
through the predict path at each γ, and select the γ that maximizes
f1_at_5. The result is written to:

    <model_dir>/calibration.json    {"gamma": <best>, "metrics": {...}}

predict_document.py reads this file at load time.

Usage
-----
    python scripts/calibrate_model.py `
        --model-dir   models/seam_detector_v4/best `
        --val-csv     data/Training_Dataset/val_rebalanced.csv `
        --gamma-min   -3.0 `
        --gamma-max   +3.0 `
        --gamma-step  0.1

Pass --output-csv to dump the full sweep curve as well, useful for
visualizing the F1-vs-γ landscape.

Cost: one full inference pass per γ, so ~60 evaluations for the default
range. On a 5090 with stride=128 that's ~10-15 minutes for a 30K-row
val set. If you want a faster coarse pass first, use --gamma-step 0.5
then narrow with a finer second pass around the winner.
"""
import pandas  # noqa: F401  (Windows stack-overflow workaround)
import sklearn  # noqa: F401

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

# Reuse the predictor and its per-window emission code path so we
# calibrate the EXACT decode the miner will run in production.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_document import HSSDPredictor    # noqa: E402


def first_transition(arr):
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            return i
    return None


def compute_metrics(rows_gt, rows_pred):
    """Returns {f1_at_5, mean_seam_offset, n_with_seam, n_total}."""
    offsets = []
    correct_at_5 = 0
    total = 0
    n_with_seam = 0
    for gt, pr in zip(rows_gt, rows_pred):
        total += 1
        gs = first_transition(gt)
        ps = first_transition(pr)
        if gs is not None:
            n_with_seam += 1
        if gs is None and ps is None:
            correct_at_5 += 1
            continue
        if gs is None or ps is None:
            continue
        d = abs(gs - ps)
        offsets.append(d)
        if d <= 5:
            correct_at_5 += 1
    return {
        "f1_at_5":          correct_at_5 / max(1, total),
        "mean_seam_offset": (sum(offsets) / max(1, len(offsets))) if offsets else float("nan"),
        "n_with_seam":      n_with_seam,
        "n_total":          total,
    }


def evaluate_at_gamma(predictor: HSSDPredictor,
                      texts: list,
                      gts: list,
                      gamma: float):
    """Run the predictor at a fixed γ and score against gts."""
    predictor.calibration_gamma = float(gamma)
    # The per-text cache would short-circuit re-prediction at different
    # γ values, defeating the sweep. Clear it for each γ.
    with predictor._pred_cache_lock:
        predictor._pred_cache.clear()
    preds = []
    for t in texts:
        # predict() returns per-word labels of length len(t.split()).
        preds.append(predictor.predict(t))
    return compute_metrics(gts, preds)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True,
                    help="Path to a trained model directory (contains "
                         "lora_adapter/ or full_model.pth).")
    ap.add_argument("--val-csv", required=True,
                    help="CSV with text + segmentation_labels columns.")
    ap.add_argument("--gamma-min", type=float, default=-3.0)
    ap.add_argument("--gamma-max", type=float, default=+3.0)
    ap.add_argument("--gamma-step", type=float, default=0.1)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Cap how many val rows are used. Useful for a "
                         "fast coarse sweep.")
    ap.add_argument("--base-model", default="microsoft/deberta-v3-large")
    ap.add_argument("--device", default=None)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--output-csv", default=None,
                    help="Optional path to write the (gamma, f1, offset) "
                         "sweep curve as CSV.")
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

    # ----- Load val data -----
    print(f"Loading val rows from {args.val_csv}…")
    texts, gts = [], []
    with open(args.val_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text") or ""
            label_field = row.get("segmentation_labels") or "[]"
            try:
                labels = json.loads(label_field)
            except json.JSONDecodeError:
                continue
            words = text.split()
            if len(words) != len(labels):
                continue
            texts.append(text)
            gts.append([int(x) for x in labels])
            if args.max_rows and len(texts) >= args.max_rows:
                break
    print(f"  {len(texts):,} val rows loaded.")

    if not texts:
        sys.exit("No usable val rows; check --val-csv contents.")

    # ----- Load predictor (γ=0 to start) -----
    print(f"Loading model from {args.model_dir}…")
    predictor = HSSDPredictor(
        model_dir=args.model_dir,
        base_model=args.base_model,
        device=args.device,
        stride=args.stride,
        calibration_gamma=0.0,    # we'll override per sweep step
    )

    # ----- Sweep -----
    n_steps = int(round((args.gamma_max - args.gamma_min) / args.gamma_step)) + 1
    gammas = [args.gamma_min + i * args.gamma_step for i in range(n_steps)]
    print(f"Sweeping γ ∈ [{args.gamma_min:+.2f}, {args.gamma_max:+.2f}] "
          f"in {args.gamma_step:.2f} steps  ({n_steps} evaluations)")

    sweep_rows = []
    best = {"gamma": 0.0, "f1_at_5": -1.0, "mean_seam_offset": float("inf")}
    for g in gammas:
        m = evaluate_at_gamma(predictor, texts, gts, g)
        f1 = m["f1_at_5"]
        off = m["mean_seam_offset"]
        marker = ""
        if f1 > best["f1_at_5"] or (
            math.isclose(f1, best["f1_at_5"]) and off < best["mean_seam_offset"]
        ):
            best = {"gamma": g, **m}
            marker = "  *NEW BEST*"
        print(f"  γ={g:+.2f}   f1@5={f1:.4f}   offset={off:.2f}   "
              f"({m['n_with_seam']}/{m['n_total']} had seam){marker}")
        sweep_rows.append({"gamma": g, **m})

    # ----- Write outputs -----
    cal_path = Path(args.model_dir) / "calibration.json"
    cal_path.write_text(json.dumps({
        "gamma":   best["gamma"],
        "metrics": {k: v for k, v in best.items() if k != "gamma"},
        "val_csv": str(args.val_csv),
        "n_val":   len(texts),
        "stride":  args.stride,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {cal_path}")
    print(f"Best γ = {best['gamma']:+.2f}   "
          f"f1@5={best['f1_at_5']:.4f}   "
          f"offset={best['mean_seam_offset']:.2f}")

    if args.output_csv:
        with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            w.writeheader()
            w.writerows(sweep_rows)
        print(f"Sweep curve written to {args.output_csv}")


if __name__ == "__main__":
    main()
