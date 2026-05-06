"""
Generate a late-seam synthesis CSV with explicit per-data_source balance and
text-level deduplication.

Difference from scripts/synthesize_late_seam.py:
  - --target-count is split EQUALLY across data_source values supplied via
    --balance-sources (default: pile,common_crawl). Each source contributes
    target_count // n_sources rows. Rows from a source that runs out of
    feasible candidates are NOT made up by oversampling another source — the
    final count may be slightly under target if a source can't yield enough.
  - Skips any synth row whose final text is already present in the input
    (exact-text dedup), so appending the output to the input CSV won't
    duplicate any document.
  - Skips duplicate synth rows within the output too (e.g., two source rows
    that happen to trim to the same text).

Why we split 50/50 even though the validator runs 67% pile / 33% CC:
  - CC is the validator's *out-of-domain gate* (forward.py:183). Failing the
    CC F1 threshold zeroes the entire reward via count_penalty. So strong
    late-seam coverage on CC samples is gate-critical.
  - Pile is the bulk of normal queries — needs late-seam coverage too.
  - 50/50 ensures both sides get ~10K new late-seam rows of supervision.

Usage:
  python scripts/synthesize_late_seam_balanced.py \\
      --input data/Training_Dataset/train_final.csv \\
      --output data/Training_Dataset/late_seam_synth.csv \\
      --target-count 20000

Then append the output to train_final.csv with whatever tool you prefer
(simple "type X >> Y" on Windows, "cat X >> Y" on Linux, or a Python merger).
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

# Sentence-ending punctuation we'll snap to when trimming the AI portion.
SENTENCE_END = re.compile(r'[.!?]["\')\]]?(?:\s|$)')


def first_seam_index(labels: List[int]) -> Optional[int]:
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            return i
    return None


def find_sentence_boundary_word_idx(words, start, end):
    for i in range(end - 1, start - 1, -1):
        stripped = words[i].rstrip('"\')]')
        if stripped and stripped[-1] in '.!?':
            return i + 1
    return None


def synthesize_late_seam(words, labels, target_pos_min, target_pos_max,
                         min_ai_words, min_total_words, rng):
    """Trim a 0->1 (or 0->1->...) doc so the FIRST seam lands in
    [target_pos_min, target_pos_max] of the resulting doc."""
    if labels[0] != 0:
        return None
    seam = first_seam_index(labels)
    if seam is None:
        return None

    n_human = seam
    n_ai_total = len(words) - seam
    if n_ai_total < min_ai_words:
        return None

    new_total_min = max(int(seam / target_pos_max),
                        n_human + min_ai_words,
                        min_total_words)
    new_total_max = min(int(seam / target_pos_min), len(words))
    if new_total_min > new_total_max:
        return None

    boundary = find_sentence_boundary_word_idx(words, n_human + min_ai_words, new_total_max)
    if boundary is not None and boundary >= new_total_min:
        new_total = boundary
    else:
        new_total = rng.randint(new_total_min, new_total_max)

    if new_total > len(words):
        return None

    new_words = words[:new_total]
    new_labels = labels[:new_total]
    # Final sanity: must still end with a 1 segment (single seam still present)
    if new_labels[-1] != 1:
        return None
    return new_words, new_labels


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-count", type=int, default=20_000)
    ap.add_argument("--target-pos-min", type=float, default=0.75)
    ap.add_argument("--target-pos-max", type=float, default=0.95)
    ap.add_argument("--min-ai-words", type=int, default=10)
    ap.add_argument("--min-total-words", type=int, default=50)
    ap.add_argument("--balance-sources", default="pile,common_crawl",
                    help="Comma-separated data_source values to balance equally.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sources = [s.strip() for s in args.balance_sources.split(",") if s.strip()]
    per_source_target = args.target_count // len(sources)
    print(f"Balancing across {len(sources)} sources: {sources}")
    print(f"Target rows per source: {per_source_target}  "
          f"(total target {per_source_target * len(sources):,})")

    rng = random.Random(args.seed)

    # ---- Load input, bucket by source ---------------------------------
    candidates_by_source = defaultdict(list)
    fieldnames = None
    skipped = Counter()
    seen_hashes = set()

    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            text = row.get("text") or ""
            seen_hashes.add(text_hash(text))   # for dedup against source

            try:
                labels = [int(x) for x in json.loads(row.get("segmentation_labels") or "[]")]
            except (json.JSONDecodeError, ValueError, TypeError):
                skipped["bad_labels"] += 1
                continue
            words = text.split()
            if len(words) != len(labels) or len(words) < args.min_total_words:
                skipped["wrong_length"] += 1
                continue
            if labels[0] != 0:
                skipped["doesnt_start_with_human"] += 1
                continue
            src = row.get("data_source") or "?"
            if src not in sources:
                skipped[f"source_{src}_not_balanced"] += 1
                continue
            candidates_by_source[src].append((row, words, labels))

    print(f"\nLoaded candidates by source:")
    for src in sources:
        print(f"  {src:<22s} {len(candidates_by_source[src]):>8,}")
    if skipped:
        print(f"  Skipped (not eligible):")
        for r, n in skipped.most_common():
            print(f"    {r:<32s} {n:>7,}")

    # ---- Synthesize per-source ----------------------------------------
    out_rows = []
    out_stats_by_source = {s: Counter() for s in sources}
    seam_pos_dist = Counter()
    output_hashes = set()   # also dedup output against itself

    for src in sources:
        candidates = list(candidates_by_source[src])
        rng.shuffle(candidates)
        produced = 0
        stats = out_stats_by_source[src]

        for row, words, labels in candidates:
            if produced >= per_source_target:
                break
            result = synthesize_late_seam(
                words, labels,
                target_pos_min=args.target_pos_min,
                target_pos_max=args.target_pos_max,
                min_ai_words=args.min_ai_words,
                min_total_words=args.min_total_words,
                rng=rng,
            )
            if result is None:
                stats["skipped_infeasible"] += 1
                continue
            new_words, new_labels = result
            new_seam = first_seam_index(new_labels)
            new_pos = new_seam / len(new_labels) if new_seam else 0.0

            if not (args.target_pos_min <= new_pos <= args.target_pos_max):
                stats["skipped_out_of_range"] += 1
                continue

            new_text = " ".join(new_words)
            h = text_hash(new_text)
            if h in seen_hashes:
                stats["skipped_dup_with_input"] += 1
                continue
            if h in output_hashes:
                stats["skipped_dup_within_output"] += 1
                continue
            output_hashes.add(h)

            new_row = dict(row)
            new_row["text"] = new_text
            new_row["segmentation_labels"] = json.dumps(new_labels)
            new_row["n_words"] = str(len(new_words))
            new_row["sample_type"] = "human_then_ai"
            if "augmented" in fieldnames:
                new_row["augmented"] = "late_seam_synth"

            out_rows.append(new_row)
            stats["generated"] += 1
            produced += 1

            # bucket for histogram
            if new_pos < 0.80:    seam_pos_dist["0.75-0.80"] += 1
            elif new_pos < 0.85:  seam_pos_dist["0.80-0.85"] += 1
            elif new_pos < 0.90:  seam_pos_dist["0.85-0.90"] += 1
            else:                  seam_pos_dist["0.90-0.95"] += 1

    # ---- Report --------------------------------------------------------
    print(f"\nSynthesis complete. {len(out_rows):,} rows generated.\n")
    for src in sources:
        s = out_stats_by_source[src]
        print(f"  [{src}]")
        for r, n in s.most_common():
            print(f"      {r:<28s} {n:>7,}")

    print(f"\nSeam position distribution in output:")
    for b in ("0.75-0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95"):
        print(f"  {b:<10s} {seam_pos_dist.get(b, 0):>7,}")

    word_counts = [int(r["n_words"]) for r in out_rows]
    if word_counts:
        print(f"\nWord-count of synthesized rows: "
              f"min={min(word_counts)}  avg={sum(word_counts)/len(word_counts):.1f}  "
              f"max={max(word_counts)}")

    src_counts = Counter(r.get("data_source") for r in out_rows)
    print(f"\nFinal data_source distribution in output:")
    for s, n in src_counts.most_common():
        pct = 100*n/len(out_rows) if out_rows else 0
        print(f"  {s:<22s} {n:>7,}  ({pct:>5.1f}%)")

    # ---- Write output --------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nWrote {len(out_rows):,} rows to {out_path}")


if __name__ == "__main__":
    main()
