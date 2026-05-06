"""
Score how well train_final.csv matches the validator's actual query distribution
along the dimensions the validator's pipeline controls:

  1. Seam count        — validator only sends 0 or 1 seam (after subsample_words)
  2. Word count        — validator's subsample_words bounds at [min_cnt, max_cnt],
                         observed range from production logs is roughly 35..350
                         with avg around 130
  3. Sample type mix   — pure_human, pure_ai, human_then_ai, ai_then_human
  4. Data source       — validator uses common_crawl (cc_net stream); other sources
                         are off-distribution
  5. Seam position     — uniform across the doc after subsample_words; lopsided
                         positions hurt f1@5
  6. Direction balance — single-seam: 0->1 (human_then_ai) vs 1->0 (ai_then_human)
                         should both appear; the validator's truncation produces
                         both roughly equally

Outputs a per-dimension match score and a final aggregate score in [0, 1].
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

PATH = Path("data/Training_Dataset/train_final.csv")

# Validator's empirical word-count bounds (from segmentation_processer.subsample_words
# defaults: min_cnt=35, max_cnt=350) and observed production log range.
VALIDATOR_MIN_WORDS = 35
VALIDATOR_MAX_WORDS = 350
VALIDATOR_OBS_AVG_WORDS = 130   # from May-3 production logs


def num_seams(labels):
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])


def first_seam_pos(labels):
    """Fractional position of the first 0/1 transition, or None."""
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            return i / len(labels)
    return None


def percentile(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(p * len(s))))
    return s[k]


def main():
    rows = []
    with open(PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["_labels"] = json.loads(r.get("segmentation_labels") or "[]")
            except Exception:
                continue
            r["_n_words"] = len(r["_labels"])
            r["_seams"] = num_seams(r["_labels"])
            rows.append(r)
    N = len(rows)
    print(f"Loaded {N:,} rows from {PATH}\n")

    # ============================================================
    # 1. Seam count
    # ============================================================
    print("=" * 70)
    print("1. SEAM-COUNT DISTRIBUTION  (validator: 100% should be <=1 seam)")
    print("=" * 70)
    seam_counts = Counter(r["_seams"] for r in rows)
    no_or_one = seam_counts[0] + seam_counts[1]
    seam_score = no_or_one / N
    for s in sorted(seam_counts.keys()):
        c = seam_counts[s]
        flag = "✓" if s <= 1 else "✗"
        print(f"  {flag}  seams={s}: {c:>8,}  ({100*c/N:>5.1f}%)")
    print(f"\n  Match score (rows with <= 1 seam): {seam_score:.4f}")

    # ============================================================
    # 2. Word count
    # ============================================================
    print("\n" + "=" * 70)
    print(f"2. WORD-COUNT DISTRIBUTION  (validator: {VALIDATOR_MIN_WORDS}–{VALIDATOR_MAX_WORDS}, "
          f"obs. avg ~{VALIDATOR_OBS_AVG_WORDS})")
    print("=" * 70)
    wc = [r["_n_words"] for r in rows]
    print(f"  min={min(wc):>4}  p10={percentile(wc,0.10):>4}  "
          f"p50={percentile(wc,0.50):>4}  avg={sum(wc)/N:>5.1f}  "
          f"p90={percentile(wc,0.90):>4}  max={max(wc):>4}")
    in_band = sum(1 for w in wc if VALIDATOR_MIN_WORDS <= w <= VALIDATOR_MAX_WORDS)
    word_score = in_band / N
    too_short = sum(1 for w in wc if w < VALIDATOR_MIN_WORDS)
    too_long  = sum(1 for w in wc if w > VALIDATOR_MAX_WORDS)
    print(f"  In  [{VALIDATOR_MIN_WORDS}, {VALIDATOR_MAX_WORDS}]: "
          f"{in_band:>8,}  ({100*in_band/N:>5.1f}%)")
    print(f"  Too short (<{VALIDATOR_MIN_WORDS}): {too_short:>5,}  "
          f"({100*too_short/N:>5.1f}%)")
    print(f"  Too long  (>{VALIDATOR_MAX_WORDS}): {too_long:>5,}  "
          f"({100*too_long/N:>5.1f}%)")
    avg_dev = abs(sum(wc)/N - VALIDATOR_OBS_AVG_WORDS) / VALIDATOR_OBS_AVG_WORDS
    print(f"  Avg vs validator-obs deviation: {avg_dev*100:.1f}%")
    print(f"\n  Match score (in-band fraction): {word_score:.4f}")

    # ============================================================
    # 3. Sample type
    # ============================================================
    print("\n" + "=" * 70)
    print("3. SAMPLE-TYPE DISTRIBUTION")
    print("=" * 70)
    valid_st = {"pure_human", "pure_ai", "human_then_ai", "ai_then_human"}
    st_counts = Counter(r.get("sample_type", "?") for r in rows)
    in_dist = sum(c for st, c in st_counts.items() if st in valid_st)
    st_score = in_dist / N
    for st in sorted(st_counts.keys()):
        c = st_counts[st]
        flag = "✓" if st in valid_st else "✗"
        print(f"  {flag}  {st:<22s} {c:>8,}  ({100*c/N:>5.1f}%)")
    print(f"\n  Match score (valid sample_types): {st_score:.4f}")

    # ============================================================
    # 4. Data source
    # ============================================================
    print("\n" + "=" * 70)
    print("4. DATA-SOURCE DISTRIBUTION  (validator: 100% common_crawl)")
    print("=" * 70)
    ds_counts = Counter(r.get("data_source", "?") for r in rows)
    cc = ds_counts.get("common_crawl", 0)
    ds_score = cc / N
    for ds, c in ds_counts.most_common():
        flag = "✓" if ds == "common_crawl" else "✗"
        print(f"  {flag}  {ds:<22s} {c:>8,}  ({100*c/N:>5.1f}%)")
    print(f"\n  Match score (CC fraction): {ds_score:.4f}")

    # ============================================================
    # 5. Seam position (single-seam rows only)
    # ============================================================
    print("\n" + "=" * 70)
    print("5. SEAM POSITION  (single-seam rows; ideal: uniform [0, 1])")
    print("=" * 70)
    pos_buckets = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    pos_counts = [0] * (len(pos_buckets) - 1)
    n_singleseam = 0
    for r in rows:
        if r["_seams"] != 1:
            continue
        n_singleseam += 1
        p = first_seam_pos(r["_labels"])
        for i in range(len(pos_buckets) - 1):
            if pos_buckets[i] <= p < pos_buckets[i+1]:
                pos_counts[i] += 1
                break
        else:
            pos_counts[-1] += 1
    print(f"  Single-seam rows: {n_singleseam:,}")
    for i in range(len(pos_buckets) - 1):
        a, b = pos_buckets[i], pos_buckets[i+1]
        c = pos_counts[i]
        bar = "#" * int(40 * c / max(1, n_singleseam))
        print(f"  {a:.2f}–{b:.2f}: {c:>7,}  ({100*c/max(1,n_singleseam):>5.1f}%)  {bar}")
    # Heuristic score: how uniform is the position distribution?
    expected = n_singleseam / (len(pos_buckets) - 1)
    pos_chi = sum((c - expected)**2 / expected for c in pos_counts) / max(1, len(pos_counts))
    pos_score = max(0.0, 1.0 - pos_chi / max(1, expected) * 5.0)
    print(f"\n  Position uniformity score (1.0 = uniform): {pos_score:.4f}")

    # ============================================================
    # 6. Direction balance for single-seam
    # ============================================================
    print("\n" + "=" * 70)
    print("6. SINGLE-SEAM DIRECTION  (ideal: ~50/50 between 0->1 and 1->0)")
    print("=" * 70)
    dir_counts = Counter()
    for r in rows:
        if r["_seams"] != 1:
            continue
        l = r["_labels"]
        if l[0] == 0:
            dir_counts["0->1 (human_then_ai)"] += 1
        else:
            dir_counts["1->0 (ai_then_human)"] += 1
    total_dir = sum(dir_counts.values())
    for d, c in dir_counts.most_common():
        print(f"  {d:<28s} {c:>8,}  ({100*c/max(1,total_dir):>5.1f}%)")
    if dir_counts:
        ratio = min(dir_counts.values()) / max(dir_counts.values())
    else:
        ratio = 0.0
    dir_score = ratio  # 1.0 = perfectly balanced
    print(f"\n  Balance score (min/max ratio): {dir_score:.4f}")

    # ============================================================
    # Aggregate
    # ============================================================
    print("\n" + "=" * 70)
    print("AGGREGATE MATCH SCORE")
    print("=" * 70)
    scores = {
        "seam_count":   seam_score,
        "word_count":   word_score,
        "sample_type":  st_score,
        "data_source":  ds_score,
        "seam_pos":     pos_score,
        "direction":    dir_score,
    }
    weights = {
        "seam_count":   0.25,   # most important
        "word_count":   0.15,
        "sample_type":  0.15,
        "data_source":  0.15,
        "seam_pos":     0.15,
        "direction":    0.15,
    }
    aggregate = sum(scores[k] * weights[k] for k in scores)
    for k, s in scores.items():
        print(f"  {k:<14s} score={s:.4f}  weight={weights[k]:.2f}")
    print(f"\n  >>> WEIGHTED AGGREGATE MATCH SCORE: {aggregate:.4f}  ({aggregate*100:.1f}%)")


if __name__ == "__main__":
    main()
