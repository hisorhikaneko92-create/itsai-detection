"""
Augment 2-seam rows by splitting each into two single-seam variants targeting
~200 words on average, then keep only 6,000 of the original 2-seam rows and
drop the rest.

A 2-seam row has actual label shape 0->1->0 (ai_in_middle) or 1->0->1
(human_in_middle). For each one we emit:

  Doc A: covers the FIRST seam only -> single-seam slice
  Doc B: covers the SECOND seam only -> single-seam slice

Each slice is ~target words long, with the seam placed at a random fractional
position in [0.30, 0.75] of the slice so the model doesn't learn a fixed
seam-position prior. Slices that can't be made >= MIN_LEN words while
containing exactly one transition are dropped.

  0->1->0 originals -> Doc A: 0->1 (human_then_ai), Doc B: 1->0 (ai_then_human)
  1->0->1 originals -> Doc A: 1->0 (ai_then_human), Doc B: 0->1 (human_then_ai)

Multi-seam (3+ seams) rows are left untouched. Single-seam and no-seam rows
are passed through unchanged.

Operates in place on data/Training_Dataset/train_final.csv via tmp + atomic
rename. Originals can be regenerated with merge_datasets.py + relabel_sample_types.py.
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

DATA_DIR = Path("data/Training_Dataset")
DEFAULT_TARGET = "train_final.csv"
TARGET_WORDS = 200
SEAM_POS_MIN = 0.30
SEAM_POS_MAX = 0.75
MIN_LEN = 50
KEEP_TWO_SEAM = 6_000
SEED = 42


def num_seams(labels):
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])


def shape_to_name(labels):
    """Same convention as relabel_sample_types.py."""
    if not labels:
        return "empty"
    n = num_seams(labels)
    if n == 0:
        return "pure_human" if labels[0] == 0 else "pure_ai"
    if n == 1:
        return "human_then_ai" if labels[0] == 0 else "ai_then_human"
    if n == 2:
        return "ai_in_middle" if labels[0] == 0 else "human_in_middle"
    return "multi_seam"


def find_two_seams(labels):
    """Return (S1, S2) — indices of the first two transitions, or (None, None)."""
    seams = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            seams.append(i)
            if len(seams) == 2:
                break
    return (seams[0], seams[1]) if len(seams) == 2 else (None, None)


def split_two_seam(words, labels, target, rng):
    """Split a 2-seam doc into two single-seam slices (Doc A, Doc B).

    Returns ((wA, lA), (wB, lB)) on success; (None, None) if either slice would
    be shorter than MIN_LEN or wouldn't contain exactly one transition.
    """
    S1, S2 = find_two_seams(labels)
    if S1 is None:
        return None, None

    # ---- Doc A: window covering S1, ending strictly before S2 -----
    p_A = rng.uniform(SEAM_POS_MIN, SEAM_POS_MAX)
    start_A = int(S1 - p_A * target)
    end_A = start_A + target
    start_A = max(0, start_A)
    end_A = min(S2, end_A)
    if end_A - start_A < MIN_LEN or start_A >= S1 or end_A <= S1:
        return None, None
    wA, lA = words[start_A:end_A], labels[start_A:end_A]
    if num_seams(lA) != 1:
        return None, None

    # ---- Doc B: window starting at-or-after S1, covering S2 --------
    p_B = rng.uniform(SEAM_POS_MIN, SEAM_POS_MAX)
    start_B = int(S2 - p_B * target)
    end_B = start_B + target
    start_B = max(S1, start_B)
    end_B = min(len(words), end_B)
    if end_B - start_B < MIN_LEN or start_B >= S2 or end_B <= S2:
        return None, None
    wB, lB = words[start_B:end_B], labels[start_B:end_B]
    if num_seams(lB) != 1:
        return None, None

    return (wA, lA), (wB, lB)


def main():
    global MIN_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA_DIR / DEFAULT_TARGET))
    ap.add_argument("--target-words", type=int, default=TARGET_WORDS)
    ap.add_argument("--min-len", type=int, default=MIN_LEN)
    ap.add_argument("--keep-two-seam", type=int, default=KEEP_TWO_SEAM)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    MIN_LEN = args.min_len

    rng = random.Random(args.seed)
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Not found: {in_path}")

    # ---------- Load ------------------------------------------------
    rows = []
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows):,} rows from {in_path}")

    # ---------- Bucket by actual seam count -------------------------
    bucket = defaultdict(list)   # actual shape name -> rows
    bad = 0
    for row in rows:
        try:
            labels = json.loads(row.get("segmentation_labels", "[]"))
        except Exception:
            bad += 1
            continue
        bucket[shape_to_name(labels)].append((row, labels))
    if bad:
        print(f"  ({bad:,} rows had unparseable labels and were dropped)")

    print("\nInitial actual-shape distribution:")
    for name in sorted(bucket.keys()):
        print(f"  {name:<20s} {len(bucket[name]):>8,}")

    two_seam_pool = bucket["ai_in_middle"] + bucket["human_in_middle"]
    print(f"\nTwo-seam pool (ai_in_middle + human_in_middle): {len(two_seam_pool):,}")

    # ---------- Augment every 2-seam row ----------------------------
    aug_rows = []
    aug_lens = []
    aug_counts = Counter()
    failed = 0
    for row, labels in two_seam_pool:
        words = (row.get("text") or "").split()
        if len(words) != len(labels):
            failed += 1
            continue
        result = split_two_seam(words, labels, target=args.target_words, rng=rng)
        if result == (None, None):
            failed += 1
            continue
        (wA, lA), (wB, lB) = result
        for w, l in [(wA, lA), (wB, lB)]:
            new_row = dict(row)
            new_row["text"] = " ".join(w)
            new_row["segmentation_labels"] = json.dumps(l)
            new_row["n_words"] = str(len(w))
            new_row["sample_type"] = shape_to_name(l)
            if "augmented" in new_row:
                new_row["augmented"] = "two_seam_split"
            aug_rows.append(new_row)
            aug_lens.append(len(w))
            aug_counts[new_row["sample_type"]] += 1

    print(f"\nAugmented {len(aug_rows):,} rows from {len(two_seam_pool):,} two-seam originals")
    print(f"  failed/skipped: {failed:,}")
    if aug_lens:
        avg = sum(aug_lens) / len(aug_lens)
        mn, mx = min(aug_lens), max(aug_lens)
        print(f"  word counts:    min={mn}  avg={avg:.1f}  max={mx}")
    print(f"  by new sample_type:")
    for st, c in aug_counts.most_common():
        print(f"    {st:<20s} {c:>8,}")

    # ---------- Sample 6K originals to keep -------------------------
    rng.shuffle(two_seam_pool)
    kept = two_seam_pool[:args.keep_two_seam]
    dropped = len(two_seam_pool) - len(kept)
    print(f"\nKeeping {len(kept):,} of {len(two_seam_pool):,} original two-seam rows  "
          f"(dropping {dropped:,})")
    kept_rows = [row for row, _ in kept]

    # ---------- Assemble final dataset ------------------------------
    final_rows = []
    for name, rows_in in bucket.items():
        if name in ("ai_in_middle", "human_in_middle"):
            continue
        for row, _ in rows_in:
            final_rows.append(row)
    final_rows.extend(kept_rows)
    final_rows.extend(aug_rows)
    rng.shuffle(final_rows)

    final_dist = Counter(r.get("sample_type", "?") for r in final_rows)
    print(f"\nFinal sample_type distribution ({len(final_rows):,} rows):")
    for st in sorted(final_dist.keys()):
        pct = 100 * final_dist[st] / len(final_rows)
        print(f"  {st:<20s} {final_dist[st]:>8,}  ({pct:>5.1f}%)")

    # Distribution by seam-count group
    groups = Counter()
    for r in final_rows:
        try:
            n = num_seams(json.loads(r["segmentation_labels"]))
        except Exception:
            continue
        if n == 0:   groups["no seam"] += 1
        elif n == 1: groups["single seam"] += 1
        elif n == 2: groups["two seam"] += 1
        else:        groups["multi seam"] += 1
    print(f"\nBy seam-count group:")
    for g, c in groups.most_common():
        pct = 100 * c / len(final_rows)
        print(f"  {g:<14s} {c:>8,}  ({pct:>5.1f}%)")

    # ---------- Write back ------------------------------------------
    tmp = in_path.with_suffix(in_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in final_rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    os.replace(tmp, in_path)
    print(f"\nWrote {len(final_rows):,} rows to {in_path}")


if __name__ == "__main__":
    main()
