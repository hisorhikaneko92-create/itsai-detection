"""Pre-process the train/val/test split CSVs so every row's text is in
[35, 350] words — matching the validator's runtime subsample_words crop.

Why: validator's segmentation_processer crops every served sample to
35-350 words before sending it to miners. Our masters were generated
with --no-subsample so they accumulated full Pile/CC docs (up to ~300k
words). Training on that length distribution doesn't transfer to
production. This script fixes the on-disk splits once, deterministically.

What's preserved (so the dataset distribution still matches validator):
  - sample_type column (rows that resolve to a different type after the
    multi-seam collapse are tracked & re-labelled to the derived type;
    in our cleaned masters this should be 0 rows)
  - data_source column (pile/common_crawl)
  - model_name column (which generator produced the AI text)
  - the (sample_type x data_source x model_name) bucket counts to
    within rounding (no row is dropped, only cropped)
  - seam-position distribution: when there's a single 0->1 or 1->0
    transition, the crop window is sampled so the seam can land at
    any uniform position 0..cnt-1 within the new window — same as the
    validator does at query time

Run:
    python scripts/preprocess_split_subsample.py --apply
By default writes to data/MainData/by_source/splits_subsampled/.
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

csv.field_size_limit(sys.maxsize)


# ---------------------------------------------------------------------------
# Subsample logic — copied verbatim from train_seam_detector.py so this
# script has zero non-stdlib deps beyond numpy.
# ---------------------------------------------------------------------------
def subsample_words(words: List[str], labels: List[int],
                    min_cnt: int = 35, max_cnt: int = 350,
                    rng=None,
                    preserve_sample_type: bool = True
                    ) -> Tuple[List[str], List[int]]:
    """Crop (words, labels) to a random length in [min_cnt, max_cnt],
    matching validator's segmentation_processer.subsample_words.

    When preserve_sample_type=True (default), the random crop window is
    forced to contain at least one word from EACH side of any seam, so a
    `human_then_ai` row never silently collapses into `pure_human` or
    `pure_ai`. This is a small deviation from validator's exact algorithm
    (validator allows the window to land on either side of the seam) but
    it keeps the (sample_type x data_source x model_name) distribution
    of the training set strictly invariant under subsampling — which is
    what we want: we tuned the dataset to the validator distribution
    once already, and that distribution should survive the crop.

    When preserve_sample_type=False, validator's exact window-range
    formula is used (some single-seam rows collapse to pure_*).
    """
    rng = rng or random
    if len(words) <= min_cnt:
        return words, labels

    has_01 = any(labels[i] == 0 and labels[i + 1] == 1
                 for i in range(len(labels) - 1))
    has_10 = any(labels[i] == 1 and labels[i + 1] == 0
                 for i in range(len(labels) - 1))

    if has_01 and has_10:
        # multi-seam: strip everything before the first 0->1 transition,
        # then recurse. Result has only one (1->0) transition.
        ind = None
        for i in range(len(labels) - 1):
            if labels[i] == 0 and labels[i + 1] == 1:
                ind = i + 1
                break
        return subsample_words(
            words[ind:], labels[ind:],
            min_cnt=min_cnt, max_cnt=max_cnt, rng=rng,
            preserve_sample_type=preserve_sample_type,
        )

    cnt = rng.randint(min_cnt, min(max_cnt, len(words)))

    split_index = None
    for i in range(len(labels) - 1):
        if labels[i] != labels[i + 1]:
            split_index = i
            break

    if split_index is not None:
        if preserve_sample_type:
            # Strict mode: require >=1 word from each side of the seam in
            # the window. The seam transitions between word `split_index`
            # (last of first segment) and word `split_index + 1` (first
            # of second segment). For both to be in the window:
            #   window start `ind` <= split_index
            #   window end `ind + cnt - 1` >= split_index + 1   ==>
            #   ind >= split_index + 2 - cnt
            lo = max(split_index + 2 - cnt, 0)
            hi = min(len(words) - cnt, split_index)
        else:
            # Validator's exact range — allows the window to fall entirely
            # on one side of the seam (sample_type collapse possible).
            lo = max(split_index - cnt, 0)
            hi = min(len(words) - cnt, split_index)
        if lo > hi:
            # Edge case: cnt is too tight to span both segments at this
            # row's seam position. Fall back to forcing the seam to the
            # middle of the window — guarantees both segments are in.
            lo = hi = max(0, min(len(words) - cnt,
                                 split_index - cnt // 2 + 1))
        ind = rng.randint(lo, hi)
    else:
        ind = rng.randint(0, len(words) - cnt)

    return words[ind:ind + cnt], labels[ind:ind + cnt]


def detect_sample_type(labels: List[int]) -> Optional[str]:
    """Map a label sequence back to a sample_type tag.
    Returns None for empty / malformed sequences."""
    if not labels:
        return None
    has_01 = any(labels[i] == 0 and labels[i + 1] == 1
                 for i in range(len(labels) - 1))
    has_10 = any(labels[i] == 1 and labels[i + 1] == 0
                 for i in range(len(labels) - 1))
    if has_01 and has_10:
        return "multi_seam"  # should be rare/zero in our cleaned masters
    if has_01:
        return "human_then_ai"
    if has_10:
        return "ai_then_human"
    if all(l == 0 for l in labels):
        return "pure_human"
    if all(l == 1 for l in labels):
        return "pure_ai"
    return None


def first_transition(labels: List[int]) -> Optional[int]:
    for i in range(len(labels) - 1):
        if labels[i] != labels[i + 1]:
            return i + 1
    return None


def _summarize_lengths(arr, label):
    if not arr:
        print(f"  {label:<25}  (no rows)")
        return
    a = np.asarray(arr)
    print(f"  {label:<25}  n={len(a):>7,}  "
          f"min={a.min():>4}  p25={int(np.percentile(a,25)):>4}  "
          f"p50={int(np.percentile(a,50)):>4}  "
          f"p75={int(np.percentile(a,75)):>4}  "
          f"p95={int(np.percentile(a,95)):>4}  "
          f"max={a.max():>6}  mean={a.mean():>6.1f}")


def process_one_split(in_path: Path, out_path: Path,
                      min_cnt: int, max_cnt: int,
                      seed: int, apply: bool):
    print(f"\n========== {in_path.name} ==========")

    rng = random.Random(seed)

    rows_in = []
    with open(in_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows_in.append(r)
    print(f"  read {len(rows_in):,} rows")

    rows_out = []
    skipped_malformed = 0
    type_changed = 0
    type_changed_breakdown = Counter()

    n_words_before = []
    n_words_after = []
    seam_pos_before = []   # absolute index of seam in row (if any)
    seam_relpos_after = [] # seam position / new_len  (0..1)

    by_type_before = Counter()
    by_type_after = Counter()
    by_bucket_before = Counter()  # (sample_type, data_source, model_name)
    by_bucket_after = Counter()

    for r in rows_in:
        text = r.get("text", "") or ""
        labels_str = r.get("segmentation_labels", "") or ""
        original_st = (r.get("sample_type") or "").strip()
        ds = (r.get("data_source") or "").strip()
        mn = (r.get("model_name") or "").strip() or "none"

        if not text or not labels_str:
            skipped_malformed += 1
            continue
        try:
            labels = json.loads(labels_str)
        except Exception:
            skipped_malformed += 1
            continue
        if not isinstance(labels, list):
            skipped_malformed += 1
            continue

        words = text.split()
        if len(words) != len(labels):
            skipped_malformed += 1
            continue

        # Pre-stats
        n_words_before.append(len(words))
        by_type_before[original_st] += 1
        by_bucket_before[(original_st, ds, mn)] += 1
        st_pre = first_transition(labels)
        if st_pre is not None:
            seam_pos_before.append(st_pre)

        # Apply subsample
        words_sub, labels_sub = subsample_words(
            words, labels, min_cnt=min_cnt, max_cnt=max_cnt, rng=rng,
        )

        # Derive sample_type from new labels (handles multi-seam collapse)
        derived_st = detect_sample_type(labels_sub) or original_st
        if derived_st != original_st:
            type_changed += 1
            type_changed_breakdown[(original_st, derived_st)] += 1

        # Post-stats
        n_words_after.append(len(words_sub))
        by_type_after[derived_st] += 1
        by_bucket_after[(derived_st, ds, mn)] += 1
        st_post = first_transition(labels_sub)
        if st_post is not None:
            seam_relpos_after.append(st_post / max(1, len(labels_sub)))

        r_out = dict(r)
        r_out["text"] = " ".join(words_sub)
        r_out["segmentation_labels"] = json.dumps(labels_sub)
        r_out["n_words"] = str(len(words_sub))
        r_out["sample_type"] = derived_st
        rows_out.append(r_out)

    # ------- Report -------
    print(f"  skipped (malformed): {skipped_malformed}")
    print(f"  sample_type changed by collapse: {type_changed}")
    if type_changed_breakdown:
        for (a, b), c in type_changed_breakdown.most_common():
            print(f"      {a} -> {b}: {c}")

    print()
    print("  Sample type counts (preservation check):")
    print(f"    {'type':<16} {'before':>8} {'after':>8}  {'change':>6}")
    for st in ["pure_human", "pure_ai", "human_then_ai", "ai_then_human", "multi_seam"]:
        b = by_type_before.get(st, 0)
        a = by_type_after.get(st, 0)
        if b or a:
            print(f"    {st:<16} {b:>8} {a:>8}  {a-b:>+6}")

    print()
    print("  Word count distribution:")
    _summarize_lengths(n_words_before, "before")
    _summarize_lengths(n_words_after,  "after")

    print()
    if seam_relpos_after:
        sp = np.asarray(seam_relpos_after)
        print(f"  Seam position (relative, 0..1) in new window: "
              f"mean={sp.mean():.3f}  std={sp.std():.3f}  "
              f"buckets: <0.1={int((sp<0.1).sum())}  "
              f"0.1-0.4={int(((sp>=0.1)&(sp<0.4)).sum())}  "
              f"0.4-0.6={int(((sp>=0.4)&(sp<0.6)).sum())}  "
              f"0.6-0.9={int(((sp>=0.6)&(sp<0.9)).sum())}  "
              f">0.9={int((sp>=0.9).sum())}")

    print()
    print(f"  bucket count (sample_type x data_source x model_name):")
    print(f"    distinct buckets before: {len(by_bucket_before)}")
    print(f"    distinct buckets after:  {len(by_bucket_after)}")
    new_buckets = set(by_bucket_after) - set(by_bucket_before)
    lost_buckets = set(by_bucket_before) - set(by_bucket_after)
    print(f"    new buckets after subsample: {len(new_buckets)}  "
          f"(rows that changed sample_type)")
    print(f"    lost buckets:               {len(lost_buckets)}")

    if apply:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows_out:
                writer.writerow(r)
        print(f"\n  WROTE {out_path}  ({len(rows_out):,} rows)")
    else:
        print(f"\n  [dry-run] would write {len(rows_out):,} rows to {out_path}")

    return len(rows_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir",
                    default="/root/llm-detection/data/MainData/by_source/splits",
                    help="Directory containing train.csv, val.csv, test.csv")
    ap.add_argument("--output-dir",
                    default="/root/llm-detection/data/MainData/by_source/splits_subsampled",
                    help="Directory to write the cropped CSVs")
    ap.add_argument("--min-words", type=int, default=35)
    ap.add_argument("--max-words", type=int, default=350)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply", action="store_true",
                    help="Actually write output files (otherwise dry-run only)")
    args = ap.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    print(f"mode = {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"input  = {in_dir}")
    print(f"output = {out_dir}")
    print(f"crop range = [{args.min_words}, {args.max_words}] words")
    print(f"seed = {args.seed}")

    total = 0
    for name in ("train.csv", "val.csv", "test.csv"):
        in_path = in_dir / name
        if not in_path.exists():
            print(f"\n[skip] {in_path} does not exist")
            continue
        out_path = out_dir / name
        n = process_one_split(in_path, out_path,
                              args.min_words, args.max_words,
                              args.seed, args.apply)
        total += n
    print(f"\n=== TOTAL rows written: {total:,} ===")
    if not args.apply:
        print("(dry-run — pass --apply to commit)")


if __name__ == "__main__":
    main()
