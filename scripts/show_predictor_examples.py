"""Walk-through demonstration of L1 vs HSSD vs Hybrid predictions on real docs.

Samples a small number of validator-style docs (balanced across sample_types
and data_sources) and writes a human-readable file showing:

  - Each doc's text, ground-truth labels, and metadata
  - PER-WORD: word | y_true | L1 pred | HSSD pred | Hybrid pred | flag
    (flags ✗ where L1 and HSSD disagree, * where Hybrid changes the rounded class)
  - Per-doc f1 / fp / ap / reward for L1-alone, HSSD-alone, Hybrid
  - End-of-file aggregate summary across the sample

Usage:
    python scripts/show_predictor_examples.py
    python scripts/show_predictor_examples.py --per-type 5 --max-words 40 \
        --out /tmp/predictor_examples.txt
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
from sklearn.metrics import f1_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor_l1 import L1Predictor
from predict_document import HSSDPredictor


def per_text_metrics(y_pred: list[float], y_true: list[int]) -> dict:
    """Mirror reward.py for a single text."""
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
            # All-one or all-zero labels: AP convention
            ap = 1.0 if rounded[0] == arr_true[0] else 0.0
    except Exception:
        ap = 0.0
    return {"f1": float(f1), "fp": float(fp_score),
            "ap": float(ap), "reward": float((fp_score + f1 + ap) / 3.0)}


def merge_hybrid(l1: list[float], hssd: list[float]) -> list[float]:
    """Current production merge rule: L1 wins when matched, HSSD fills rest."""
    return [0.0 if l1[i] < 0.5 else hssd[i] for i in range(len(l1))]


def stratified_sample(df: pd.DataFrame, per_type: int, seed: int = 42) -> pd.DataFrame:
    """Pick `per_type` docs from each sample_type. Mix data sources within each."""
    rng = np.random.default_rng(seed)
    parts = []
    for st in sorted(df["sample_type"].unique()):
        sub = df[df["sample_type"] == st]
        # Mix Pile and CC roughly evenly within the slice
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


def truncate_word(w: str, width: int = 22) -> str:
    if len(w) > width:
        return w[: width - 2] + ".."
    return w


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("data/adv_test.csv"))
    p.add_argument("--bloom", action="append", default=None, type=Path)
    p.add_argument("--hssd-model-dir", type=Path, default=Path("models/best"))
    p.add_argument("--hssd-base-model", default="microsoft/deberta-v3-large")
    p.add_argument("--min-run", type=int, default=3)
    p.add_argument("--per-type", type=int, default=5,
                   help="Docs per sample_type (default 5 → 20 total)")
    p.add_argument("--max-words", type=int, default=40,
                   help="Max words to print per doc (rest summarized)")
    p.add_argument("--out", type=Path, default=Path("/tmp/predictor_examples.txt"))
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

    print(f"Loading L1 ({len(args.bloom)} shards)...", file=sys.stderr)
    t0 = time.time()
    l1 = L1Predictor(args.bloom, min_run=args.min_run)
    print(f"  L1 loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    print(f"Loading HSSD ({args.hssd_model_dir})...", file=sys.stderr)
    t0 = time.time()
    hssd = HSSDPredictor(
        model_dir=str(args.hssd_model_dir),
        base_model=args.hssd_base_model,
    )
    print(f"  HSSD loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    # Aggregate counters for final summary
    agg_l1, agg_hssd, agg_hybrid = [], [], []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.out, "w", encoding="utf-8")
    fh.write("L1 / HSSD / Hybrid predictor walkthrough\n")
    fh.write(f"Generated against {args.csv.name}\n")
    fh.write(f"L1 min_run={args.min_run}; current hybrid merge rule:\n")
    fh.write("    output[i] = 0.0 if L1[i] < 0.5 else HSSD[i]\n\n")

    for i, row in sample.iterrows():
        text = row["text"]
        words = text.split()
        y_true = ast.literal_eval(row["segmentation_labels"])

        l1_pred = l1.predict(text)
        hssd_pred = hssd.predict_with_probs(text)
        hybrid_pred = merge_hybrid(l1_pred, hssd_pred)

        # Skip if word-count alignment fails (shouldn't happen with adv_test.csv)
        if not (len(l1_pred) == len(hssd_pred) == len(hybrid_pred) == len(y_true) == len(words)):
            fh.write(f"\n[SKIPPED row {i}: word-count mismatch L1={len(l1_pred)} "
                     f"HSSD={len(hssd_pred)} truth={len(y_true)} words={len(words)}]\n")
            continue

        fh.write("\n")
        fh.write("=" * 92 + "\n")
        fh.write(f"DOC {i+1:>2}/{len(sample)}  source={row['data_source']:<13}  "
                 f"sample_type={row['sample_type']:<15}  "
                 f"augmented={row['augmented']}  n_words={len(words)}\n")
        fh.write("=" * 92 + "\n")
        fh.write(f"text preview: {text[:200]!r}\n")
        if len(text) > 200:
            fh.write(f"           ...({len(text) - 200} more chars)\n")
        fh.write("\n")
        fh.write(f"{'#':>3}  {'word':<22}  {'true':>4}  "
                 f"{'L1':>6}  {'HSSD':>6}  {'HYBRID':>6}   flags\n")
        fh.write("-" * 82 + "\n")

        # Print up to max_words rows; if longer, also print a sample of the rest
        n_show = min(args.max_words, len(words))
        for j in range(n_show):
            w = truncate_word(words[j])
            l1_p = l1_pred[j]
            hssd_p = hssd_pred[j]
            hy_p = hybrid_pred[j]
            t = y_true[j]
            # Flags:
            flags = []
            # Round disagreement
            l1_round = 1 if l1_p >= 0.5 else 0
            hssd_round = 1 if hssd_p >= 0.5 else 0
            hy_round = 1 if hy_p >= 0.5 else 0
            if l1_round != hssd_round:
                flags.append("L1/HSSD-DIFF")
            if hy_round != t:
                flags.append("WRONG")
            elif (l1_round != t and hy_round == t) or (hssd_round != t and hy_round == t):
                flags.append("FIXED")
            fh.write(f"{j+1:>3}  {w:<22}  {t:>4}  "
                     f"{l1_p:>6.3f}  {hssd_p:>6.3f}  {hy_p:>6.3f}   "
                     f"{','.join(flags)}\n")

        if len(words) > n_show:
            fh.write(f"     ... {len(words) - n_show} more words "
                     f"(showing first {n_show} only)\n")

        # Per-doc metrics
        m_l1 = per_text_metrics(l1_pred, y_true)
        m_hssd = per_text_metrics(hssd_pred, y_true)
        m_hy = per_text_metrics(hybrid_pred, y_true)
        agg_l1.append(m_l1); agg_hssd.append(m_hssd); agg_hybrid.append(m_hy)
        fh.write("\n")
        fh.write(f"Per-doc metrics:\n")
        fh.write(f"  L1     alone : f1={m_l1['f1']:.3f}  fp={m_l1['fp']:.3f}  "
                 f"ap={m_l1['ap']:.3f}  reward={m_l1['reward']:.3f}\n")
        fh.write(f"  HSSD   alone : f1={m_hssd['f1']:.3f}  fp={m_hssd['fp']:.3f}  "
                 f"ap={m_hssd['ap']:.3f}  reward={m_hssd['reward']:.3f}\n")
        fh.write(f"  HYBRID       : f1={m_hy['f1']:.3f}  fp={m_hy['fp']:.3f}  "
                 f"ap={m_hy['ap']:.3f}  reward={m_hy['reward']:.3f}\n")

        # Print progress to stderr
        print(f"  doc {i+1}/{len(sample)} done "
              f"(L1 r={m_l1['reward']:.2f}, "
              f"HSSD r={m_hssd['reward']:.2f}, "
              f"HY r={m_hy['reward']:.2f})", file=sys.stderr)

    # Aggregate summary
    fh.write("\n")
    fh.write("=" * 92 + "\n")
    fh.write(f"SUMMARY ACROSS {len(agg_hybrid)} DOCS\n")
    fh.write("=" * 92 + "\n")
    def avg(ms, k): return float(np.mean([m[k] for m in ms])) if ms else 0.0
    for name, agg in (("L1     alone", agg_l1), ("HSSD   alone", agg_hssd), ("HYBRID      ", agg_hybrid)):
        fh.write(f"  {name}: f1={avg(agg, 'f1'):.3f}  fp={avg(agg, 'fp'):.3f}  "
                 f"ap={avg(agg, 'ap'):.3f}  reward={avg(agg, 'reward'):.3f}\n")

    fh.write("\nFlags legend:\n")
    fh.write("  L1/HSSD-DIFF  L1 and HSSD round to different labels\n")
    fh.write("  WRONG         Hybrid's rounded prediction != truth\n")
    fh.write("  FIXED         L1 or HSSD alone was wrong; Hybrid fixed it\n")

    fh.close()
    print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
