"""Train the LightGBM ensemble meta-learner.

Reads feature files produced by extract_ensemble_features.py, trains a
LightGBM binary classifier with early stopping, and saves the model + a
tiny eval report.

Validator-equivalent metrics are computed by FLATTENING predictions across
all words (like the validator does) and splitting by data_source so we can
predict gate-3 (CC F1) and Pile-domain reward separately.

Usage:
    python scripts/train_ensemble_lgbm.py \\
        --train-features /tmp/ensemble_features_val.npz \\
        --val-features   /tmp/ensemble_features_test_small.npz \\
        --eval-features  /tmp/ensemble_features_test.npz \\
        --out /workspace/llm-detection/models/ensemble_lgbm.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score, average_precision_score


def per_slice_metrics(y_pred, y_true):
    """Validator-equivalent flat-concat metrics on an arbitrary slice."""
    if len(y_pred) == 0:
        return {"f1": 0.0, "fp": 0.0, "ap": 0.0, "reward": 0.0, "n": 0}
    rounded = np.round(y_pred).astype(int)
    fp = int(((rounded == 1) & (y_true == 0)).sum())
    fp_score = 1 - fp / len(y_pred)
    try:
        f1 = f1_score(y_true, rounded, zero_division=0)
    except Exception:
        f1 = 0.0
    try:
        if len(np.unique(y_true)) >= 2:
            ap = average_precision_score(y_true, y_pred)
        else:
            ap = 1.0 if rounded[0] == y_true[0] else 0.0
    except Exception:
        ap = 0.0
    return {"f1": float(f1), "fp": float(fp_score),
            "ap": float(ap), "reward": float((fp_score + f1 + ap) / 3.0),
            "n": int(len(y_pred))}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-features", type=Path, required=True)
    p.add_argument("--val-features", type=Path, required=True,
                   help="Used for early stopping. Must be disjoint from eval.")
    p.add_argument("--eval-features", type=Path, required=True,
                   help="Final eval. Held out completely from training.")
    p.add_argument("--out", type=Path, required=True,
                   help="Where to save trained LightGBM model (.txt)")
    p.add_argument("--report", type=Path, default=None,
                   help="Where to save metrics report (default: <out>.report.json)")
    p.add_argument("--num-boost-round", type=int, default=1000)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=0.05)
    return p.parse_args()


def load_features(path: Path):
    d = np.load(path)
    return {
        "X": d["X"], "y": d["y"],
        "doc_id": d["doc_id"], "source": d["source"],
        "sample_type": d["sample_type"],
        "feature_names": list(d["feature_names"]),
    }


def main():
    args = parse_args()
    print(f"Loading features...", file=sys.stderr)
    train = load_features(args.train_features)
    val   = load_features(args.val_features)
    eval_ = load_features(args.eval_features)
    print(f"  train: {train['X'].shape}", file=sys.stderr)
    print(f"  val:   {val['X'].shape}",   file=sys.stderr)
    print(f"  eval:  {eval_['X'].shape}", file=sys.stderr)
    print(f"  features ({len(train['feature_names'])}): {train['feature_names']}",
          file=sys.stderr)

    # LightGBM dataset
    dtrain = lgb.Dataset(train["X"], label=train["y"],
                          feature_name=train["feature_names"])
    dval = lgb.Dataset(val["X"], label=val["y"],
                       reference=dtrain,
                       feature_name=train["feature_names"])

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": args.learning_rate,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 100,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbose": -1,
    }
    print(f"\nTraining LightGBM...", file=sys.stderr)
    t0 = time.time()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=20),
        ],
    )
    print(f"  trained in {time.time() - t0:.1f}s "
          f"(best iter: {booster.best_iteration})", file=sys.stderr)

    # Save model — truncate to best_iteration so predict at inference time uses
    # exactly the same trees the eval below uses. Otherwise the model file
    # contains all `num_boost_round` trees and `best_iteration` is lost on load.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.out), num_iteration=booster.best_iteration)
    print(f"\nSaved model -> {args.out}  "
          f"(truncated to best_iteration={booster.best_iteration})",
          file=sys.stderr)
    # Also dump feature importance
    importance = booster.feature_importance(importance_type="gain")
    print(f"\nFeature importance (gain):", file=sys.stderr)
    for name, score in sorted(zip(train["feature_names"], importance),
                              key=lambda x: -x[1]):
        print(f"  {name:>26}: {score:.0f}", file=sys.stderr)

    # Evaluate on eval set
    print(f"\nEvaluating on held-out eval set...", file=sys.stderr)
    y_pred = booster.predict(eval_["X"], num_iteration=booster.best_iteration)
    y_true = eval_["y"]
    source = eval_["source"]
    sample_type = eval_["sample_type"]

    report = {
        "model_path": str(args.out),
        "best_iteration": int(booster.best_iteration),
        "train_size": int(len(train["y"])),
        "val_size": int(len(val["y"])),
        "eval_size": int(len(y_true)),
        "feature_importance": dict(zip(train["feature_names"],
                                       importance.astype(int).tolist())),
    }

    # Baselines for comparison
    hssd_raw = eval_["X"][:, 1]  # column 1 = hssd_pred
    l1_raw = eval_["X"][:, 0]    # column 0 = l1_pred
    # Current production hybrid rule (L1 < 0.5 -> 0; else hssd)
    cur_hybrid = np.where(l1_raw < 0.5, 0.0, hssd_raw)

    print("\n=== OVERALL (flat-concat across all eval words) ===", file=sys.stderr)
    for name, preds in (("L1 alone", l1_raw),
                        ("HSSD alone", hssd_raw),
                        ("Current hybrid (rule)", cur_hybrid),
                        ("LightGBM ENSEMBLE", y_pred)):
        m = per_slice_metrics(preds, y_true)
        print(f"  {name:>22}: f1={m['f1']:.4f}  fp={m['fp']:.4f}  "
              f"ap={m['ap']:.4f}  reward={m['reward']:.4f}", file=sys.stderr)
        report.setdefault("overall", {})[name] = m

    print("\n=== BY data_source (validator's gate-3 / Pile-reward lens) ===",
          file=sys.stderr)
    for src in sorted(np.unique(source)):
        mask = source == src
        print(f"\n  data_source = {src} (n_words={int(mask.sum())})", file=sys.stderr)
        for name, preds in (("L1 alone", l1_raw),
                            ("HSSD alone", hssd_raw),
                            ("Current hybrid (rule)", cur_hybrid),
                            ("LightGBM ENSEMBLE", y_pred)):
            m = per_slice_metrics(preds[mask], y_true[mask])
            print(f"    {name:>22}: f1={m['f1']:.4f}  fp={m['fp']:.4f}  "
                  f"ap={m['ap']:.4f}  reward={m['reward']:.4f}", file=sys.stderr)
            report.setdefault(f"by_source_{src}", {})[name] = m

    report_path = args.report or args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report -> {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
