"""
Extract a stratified subset from train_archive.csv with multi-axis
constraints. Output is a single CSV in the same training-format the
rest of the pipeline expects (text / segmentation_labels /
data_source / sample_type / model_name / n_words / augmented).

Constraints applied simultaneously:

  * sample_type counts             default: 18000 / 20000 / 2000
                                   (pure_human / pure_ai / human_then_ai)
  * word-count distribution        default: 5% / 35% / 60%
                                   (35-100 / 101-200 / 201-350)
                                   <35 and >350 are excluded.
  * source-domain split            50/50 between pile-like and CC-like
                                   pile-like keywords: wikihow, wikipedia,
                                                       arxiv, PeerRead
                                   cc-like keywords:   eli5, reddit
  * generation-type split          50/50 between standard and adversarial
                                   (AI samples only; pure_human gets
                                   model='none' so the gen split is N/A).
                                   adversarial models: gpt-3.5-turbo,
                                       chatgpt, cohere*, command-xlarge-*,
                                       bigscience/bloomz
                                   standard models:    dolly-v2*, davinci,
                                       text-davinci-003, flan-t5

The script samples WITHOUT replacement per (sample_type x word_bucket x
source x gen) cell. If a cell is undersupplied vs its target, the
script takes the entire pool and reports the shortfall (no duplicates).

The output `data_source` field is REWRITTEN to either 'pile' or
'common_crawl' based on the heuristic, so the downstream
split_dataset.py and training scripts see a clean two-source taxonomy
instead of 'archive:wikihow' / 'archive:eli5' etc.

Usage:
    python scripts/extract_archive_subset.py \
        --input  data/train_archive.csv \
        --output data/train_archive_balanced_40k.csv

    # Custom counts:
    python scripts/extract_archive_subset.py \
        --target-pure-human 18000 --target-pure-ai 20000 \
        --target-human-then-ai 2000 \
        --output data/train_archive_balanced_40k.csv

    # Different word-count shares:
    python scripts/extract_archive_subset.py \
        --word-share 0.10,0.40,0.50 \
        --output data/train_archive_balanced_40k.csv
"""
import argparse
import csv
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PILE_LIKE_KEYS = ("wikihow", "wikipedia", "arxiv", "PeerRead")
CC_LIKE_KEYS   = ("eli5", "reddit")

# Adversarial: chat / RLHF-tuned models that produce more human-like
# output. These tend to make the detection problem harder, which is
# what "adversarial" means in this dataset's MAGE / M4 lineage.
ADVERSARIAL_MODELS = {
    "gpt-3.5-turbo", "chatgpt",
    "cohere", "cohere-xlarge-nightly", "command-xlarge-nightly",
    "bigscience/bloomz",
}
# Standard: vanilla / older / less RLHF-tuned models. AI artifacts are
# more obvious so detection is easier.
STANDARD_MODELS = {
    "dolly-v2-12b", "dolly-v2",
    "davinci", "text-davinci-003",
    "flan-t5",
}

WORD_BUCKETS = ("35-100", "101-200", "201-350")


def categorize_source(ds: str) -> str:
    for k in PILE_LIKE_KEYS:
        if k in ds:
            return "pile"
    for k in CC_LIKE_KEYS:
        if k in ds:
            return "common_crawl"
    return "other"


def categorize_generation(model: str) -> str:
    if model in ADVERSARIAL_MODELS:
        return "adversarial"
    if model in STANDARD_MODELS:
        return "standard"
    return "other"


def word_bucket(n: int) -> str:
    if n < 35:    return "<35"
    if n <= 100:  return "35-100"
    if n <= 200:  return "101-200"
    if n <= 350:  return "201-350"
    return ">350"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default="data/train_archive.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-pure-human",     type=int, default=18000)
    ap.add_argument("--target-pure-ai",        type=int, default=20000)
    ap.add_argument("--target-human-then-ai",  type=int, default=2000)
    ap.add_argument("--word-share", default="0.05,0.35,0.60",
                    help="Comma-separated shares for the 35-100, 101-200, "
                         "201-350 word buckets. Must sum to 1.0.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    word_shares = [float(x) for x in args.word_share.split(",")]
    if len(word_shares) != 3 or abs(sum(word_shares) - 1.0) > 1e-6:
        sys.exit(f"--word-share must be 3 values summing to 1.0, got {args.word_share}")

    targets = {
        "pure_human":    args.target_pure_human,
        "pure_ai":       args.target_pure_ai,
        "human_then_ai": args.target_human_then_ai,
    }
    target_total = sum(targets.values())
    print(f"Target total: {target_total:,}  "
          f"(pure_human={args.target_pure_human:,}, "
          f"pure_ai={args.target_pure_ai:,}, "
          f"human_then_ai={args.target_human_then_ai:,})")

    # ------------------------------------------------------------------
    # Read input, bin into pool cells.
    # ------------------------------------------------------------------
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    print(f"\nReading {in_path}...")
    pools: Dict[Tuple[str, str, str, str], List[Dict]] = defaultdict(list)
    skipped: Counter = Counter()
    fieldnames: Optional[List[str]] = None
    n_in = 0

    with open(in_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for r in reader:
            n_in += 1
            st = r["sample_type"]
            if st not in targets:
                skipped["sample_type_other"] += 1
                continue

            try:
                w = int(r["n_words"])
            except (KeyError, ValueError):
                skipped["bad_n_words"] += 1
                continue
            wb = word_bucket(w)
            if wb not in WORD_BUCKETS:
                skipped[f"word_bucket_{wb}"] += 1
                continue

            src = categorize_source(r["data_source"])
            if src == "other":
                skipped["source_other"] += 1
                continue

            if st == "pure_human":
                gen = "na"
            else:
                gen = categorize_generation(r["model_name"])
                if gen == "other":
                    skipped[f"unclass_model:{r['model_name']}"] += 1
                    continue

            pools[(st, wb, src, gen)].append(r)

    eligible = sum(len(v) for v in pools.values())
    print(f"Read {n_in:,} rows; {eligible:,} eligible after filtering.")
    if skipped:
        print("Skipped breakdown:")
        for reason, n in skipped.most_common():
            print(f"  {reason:<35s} {n:>9,}")

    # ------------------------------------------------------------------
    # Build per-cell targets.
    #
    # Layer 1: sample_type x word_bucket gets `total * share` rows.
    # Layer 2: split that count 50/50 across source (pile vs CC), and
    #          for AI types ALSO 50/50 across generation (std vs adv) ->
    #          4 cells per (st, wb). For pure_human, just 2 cells.
    # ------------------------------------------------------------------
    cell_targets: Dict[Tuple[str, str, str, str], int] = {}
    for st, total in targets.items():
        for wb, share in zip(WORD_BUCKETS, word_shares):
            cell_total = int(round(total * share))
            if st == "pure_human":
                half = cell_total // 2
                cell_targets[(st, wb, "pile",         "na")] = half
                cell_targets[(st, wb, "common_crawl", "na")] = cell_total - half
            else:
                quarter = cell_total // 4
                cell_targets[(st, wb, "pile",         "standard")]    = quarter
                cell_targets[(st, wb, "pile",         "adversarial")] = quarter
                cell_targets[(st, wb, "common_crawl", "standard")]    = quarter
                cell_targets[(st, wb, "common_crawl", "adversarial")] = cell_total - 3 * quarter

    # ------------------------------------------------------------------
    # Sample. If a cell is short, take the whole pool and record.
    # ------------------------------------------------------------------
    print(f"\nSampling per cell:")
    print(f"  {'cell':<70s} {'supply':>8s} {'target':>8s} {'taken':>8s}")
    print(f"  {'-'*70} {'-'*8} {'-'*8} {'-'*8}")
    chosen: List[Dict] = []
    shortfalls: List[Tuple] = []
    for cell in sorted(cell_targets.keys()):
        target = cell_targets[cell]
        if target == 0:
            continue
        pool = pools.get(cell, [])
        supply = len(pool)
        if supply >= target:
            taken = rng.sample(pool, target)
        else:
            taken = list(pool)
            shortfalls.append((cell, target, supply))
        chosen.extend(taken)
        cell_str = " | ".join(cell)
        print(f"  {cell_str:<70s} {supply:>8d} {target:>8d} {len(taken):>8d}")

    rng.shuffle(chosen)

    # ------------------------------------------------------------------
    # Rewrite data_source to the canonical 'pile' / 'common_crawl' tag
    # so downstream scripts see a clean taxonomy. The original
    # 'archive:wikihow' style string is dropped here.
    # ------------------------------------------------------------------
    for r in chosen:
        r["data_source"] = categorize_source(r["data_source"])

    # ------------------------------------------------------------------
    # Write output.
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in chosen:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nWrote {len(chosen):,} rows to {out_path}")

    # ------------------------------------------------------------------
    # Final distribution sanity print.
    # ------------------------------------------------------------------
    by_type: Counter = Counter()
    by_word: Counter = Counter()
    by_src:  Counter = Counter()
    by_gen:  Counter = Counter()
    by_st_gen: Counter = Counter()
    for r in chosen:
        by_type[r["sample_type"]] += 1
        by_word[word_bucket(int(r["n_words"]))] += 1
        by_src[r["data_source"]] += 1
        if r["sample_type"] != "pure_human":
            g = categorize_generation(r["model_name"])
            by_gen[g] += 1
            by_st_gen[(r["sample_type"], g)] += 1

    print("\n--- Final distribution ---")
    print("sample_type:")
    for k, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {k:<14s} {n:>7,}  ({100*n/len(chosen):.1f}%)")
    print("word_bucket:")
    for k in WORD_BUCKETS:
        if by_word.get(k):
            print(f"  {k:<14s} {by_word[k]:>7,}  ({100*by_word[k]/len(chosen):.1f}%)")
    print("data_source (rewritten to canonical tag):")
    for k, n in sorted(by_src.items(), key=lambda x: -x[1]):
        print(f"  {k:<14s} {n:>7,}  ({100*n/len(chosen):.1f}%)")
    print("generation_type (AI samples only):")
    ai_total = sum(by_gen.values())
    if ai_total:
        for k, n in sorted(by_gen.items(), key=lambda x: -x[1]):
            print(f"  {k:<14s} {n:>7,}  ({100*n/ai_total:.1f}% of AI)")
    print("sample_type x generation_type (AI samples):")
    for (st, g), n in sorted(by_st_gen.items()):
        print(f"  {st:<14s} {g:<12s} {n:>7,}")

    if shortfalls:
        print("\nWARNING: cells were undersupplied:")
        for cell, target, supply in shortfalls:
            cell_str = " | ".join(cell)
            print(f"  {cell_str}: needed {target}, had {supply}")
    else:
        print("\nAll cells fully satisfied -- no shortfalls.")


if __name__ == "__main__":
    main()
