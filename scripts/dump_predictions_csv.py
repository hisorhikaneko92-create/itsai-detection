"""Dump per-word predictions (L1, HSSD, Hybrid) + ground truth to CSV.

Two output files:

  word_predictions.csv  — one row PER WORD across all sampled docs
      doc_id, source, sample_type, n_words, word_idx, word, true_label,
      l1_pred, hssd_pred, hybrid_pred,
      l1_round, hssd_round, hybrid_round,
      l1_correct, hssd_correct, hybrid_correct

  doc_metrics.csv       — one row PER DOC with aggregate metrics
      doc_id, source, sample_type, n_words, augmented,
      l1_f1, l1_fp, l1_ap, l1_reward,
      hssd_f1, hssd_fp, hssd_ap, hssd_reward,
      hybrid_f1, hybrid_fp, hybrid_ap, hybrid_reward,
      l1_match_count, l1_match_fraction

Usage:
    python scripts/dump_predictions_csv.py
    python scripts/dump_predictions_csv.py --per-type 25 \
        --out-words /tmp/word_predictions.csv \
        --out-docs  /tmp/doc_metrics.csv
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor_l1 import L1Predictor
from predict_document import HSSDPredictor


def per_text_metrics(y_pred: list[float], y_true: list[int]) -> dict:
    if not y_pred or len(y_pred) != len(y_true):
        return {"f1": 0.0, "fp": 0.0, "ap": 0.0, "reward": 0.0}
    arr_pred = np.asarray(y_pred, dtype=float)
    arr_true = np.asarray(y_true, dtype=int)
    rounded = np.round(arr_pred).astype(int)
    fp = int(((rounded == 1) & (arr_true == 0)).sum())
    fp_score = 1 - fp / max(len(arr_pred), 1)
    try:
        f1 = f1_score(arr_true, rounded, zero_division=0)
    except Exception:
        f1 = 0.0
    try:
        if len(np.unique(arr_true)) >= 2:
            ap = average_precision_score(arr_true, arr_pred)
        else:
            ap = 1.0 if rounded[0] == arr_true[0] else 0.0
    except Exception:
        ap = 0.0
    return {"f1": float(f1), "fp": float(fp_score),
            "ap": float(ap), "reward": float((fp_score + f1 + ap) / 3.0)}


def merge_hybrid(l1: list[float], hssd: list[float]) -> list[float]:
    return [0.0 if l1[i] < 0.5 else hssd[i] for i in range(len(l1))]


def stratified_sample(df: pd.DataFrame, per_type: int, seed: int = 42) -> pd.DataFrame:
    parts = []
    for st in sorted(df["sample_type"].unique()):
        sub = df[df["sample_type"] == st]
        sub_pile = sub[sub["data_source"] == "pile"]
        sub_cc = sub[sub["data_source"] == "common_crawl"]
        n_pile = per_type // 2
        n_cc = per_type - n_pile
        pieces = []
        if len(sub_pile) > 0:
            pieces.append(sub_pile.sample(n=min(n_pile, len(sub_pile)), random_state=seed))
        if len(sub_cc) > 0:
            pieces.append(sub_cc.sample(n=min(n_cc, len(sub_cc)), random_state=seed + 1))
        if pieces:
            parts.append(pd.concat(pieces, ignore_index=True))
    return pd.concat(parts, ignore_index=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("data/adv_test.csv"))
    p.add_argument("--bloom", action="append", default=None, type=Path)
    p.add_argument("--hssd-model-dir", type=Path, default=Path("models/best"))
    p.add_argument("--hssd-base-model", default="microsoft/deberta-v3-large")
    p.add_argument("--min-run", type=int, default=3)
    p.add_argument("--per-type", type=int, default=25,
                   help="Docs per sample_type (default 25 → 100 total)")
    p.add_argument("--out-words", type=Path, default=Path("/tmp/word_predictions.csv"))
    p.add_argument("--out-docs", type=Path, default=Path("/tmp/doc_metrics.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.bloom is None:
        args.bloom = [Path(f"indexes/pile_l1_shard{i}.bloom") for i in range(4)]

    print("Loading dataset...", file=sys.stderr)
    df = pd.read_csv(args.csv)
    sample = stratified_sample(df, args.per_type)
    print(f"Sampled {len(sample)} docs:", file=sys.stderr)
    for (src, st), group in sample.groupby(["data_source", "sample_type"]):
        print(f"  {src:>13} / {st:<15}: {len(group)} docs", file=sys.stderr)

    print(f"Loading L1 ({len(args.bloom)} shards, min_run={args.min_run})...", file=sys.stderr)
    t0 = time.time()
    l1 = L1Predictor(args.bloom, min_run=args.min_run)
    print(f"  L1 loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    print(f"Loading HSSD ({args.hssd_model_dir})...", file=sys.stderr)
    t0 = time.time()
    hssd = HSSDPredictor(model_dir=str(args.hssd_model_dir),
                          base_model=args.hssd_base_model)
    print(f"  HSSD loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    args.out_words.parent.mkdir(parents=True, exist_ok=True)
    args.out_docs.parent.mkdir(parents=True, exist_ok=True)

    f_words = open(args.out_words, "w", newline="", encoding="utf-8")
    f_docs  = open(args.out_docs,  "w", newline="", encoding="utf-8")

    w_words = csv.writer(f_words)
    w_words.writerow([
        "doc_id", "source", "sample_type", "n_words",
        "word_idx", "word", "true_label",
        "l1_pred", "hssd_pred", "hybrid_pred",
        "l1_round", "hssd_round", "hybrid_round",
        "l1_correct", "hssd_correct", "hybrid_correct",
    ])
    w_docs = csv.writer(f_docs)
    w_docs.writerow([
        "doc_id", "source", "sample_type", "n_words", "augmented",
        "l1_f1", "l1_fp", "l1_ap", "l1_reward",
        "hssd_f1", "hssd_fp", "hssd_ap", "hssd_reward",
        "hybrid_f1", "hybrid_fp", "hybrid_ap", "hybrid_reward",
        "l1_match_count", "l1_match_fraction",
    ])

    written_docs = 0
    written_words = 0
    skipped = 0

    t0 = time.time()
    for doc_id, row in sample.iterrows():
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        words = text.split()
        try:
            y_true = ast.literal_eval(row["segmentation_labels"])
        except Exception:
            skipped += 1
            continue
        if not isinstance(y_true, list):
            skipped += 1
            continue

        l1_pred = l1.predict(text)
        hssd_pred = hssd.predict_with_probs(text)
        hybrid_pred = merge_hybrid(l1_pred, hssd_pred)

        if not (len(l1_pred) == len(hssd_pred) == len(hybrid_pred) == len(y_true) == len(words)):
            skipped += 1
            continue

        # word rows
        l1_match_count = 0
        for j in range(len(words)):
            true_lbl = int(y_true[j])
            l1_p = float(l1_pred[j])
            hssd_p = float(hssd_pred[j])
            hybrid_p = float(hybrid_pred[j])
            l1_r = 1 if l1_p >= 0.5 else 0
            hssd_r = 1 if hssd_p >= 0.5 else 0
            hybrid_r = 1 if hybrid_p >= 0.5 else 0
            if l1_p < 0.5:
                l1_match_count += 1
            w_words.writerow([
                doc_id, row["data_source"], row["sample_type"], len(words),
                j, words[j], true_lbl,
                f"{l1_p:.4f}", f"{hssd_p:.4f}", f"{hybrid_p:.4f}",
                l1_r, hssd_r, hybrid_r,
                int(l1_r == true_lbl), int(hssd_r == true_lbl), int(hybrid_r == true_lbl),
            ])
            written_words += 1

        # doc row
        m_l1 = per_text_metrics(l1_pred, y_true)
        m_hssd = per_text_metrics(hssd_pred, y_true)
        m_hy = per_text_metrics(hybrid_pred, y_true)
        w_docs.writerow([
            doc_id, row["data_source"], row["sample_type"], len(words),
            row.get("augmented", ""),
            f"{m_l1['f1']:.4f}", f"{m_l1['fp']:.4f}", f"{m_l1['ap']:.4f}", f"{m_l1['reward']:.4f}",
            f"{m_hssd['f1']:.4f}", f"{m_hssd['fp']:.4f}", f"{m_hssd['ap']:.4f}", f"{m_hssd['reward']:.4f}",
            f"{m_hy['f1']:.4f}", f"{m_hy['fp']:.4f}", f"{m_hy['ap']:.4f}", f"{m_hy['reward']:.4f}",
            l1_match_count, f"{l1_match_count / max(len(words), 1):.4f}",
        ])
        written_docs += 1

        if written_docs % 25 == 0:
            elapsed = time.time() - t0
            print(f"  {written_docs}/{len(sample)} docs  "
                  f"({written_words:,} words written, {elapsed:.0f}s)",
                  file=sys.stderr)

    f_words.close(); f_docs.close()

    print(f"\nWrote {args.out_words} ({written_words:,} word rows)", file=sys.stderr)
    print(f"Wrote {args.out_docs}  ({written_docs} doc rows)", file=sys.stderr)
    print(f"Skipped {skipped} bad rows.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
