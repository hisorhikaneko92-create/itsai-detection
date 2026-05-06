"""
Generate ai_then_human (1->0) late-seam rows — the mirror of
synthesize_late_seam_balanced.py.

For an `ai_then_human` row, the original structure is [AI...][human...] with
a 1->0 seam at position S. To put the seam at p (75-95%) of the resulting
doc, we trim the right tail (the HUMAN tail) so the new total length is
S / p. The algorithm is identical to the 0->1 case — only the direction
check at the start and end differs.

Output is balanced 50/50 across data_source (pile / common_crawl) and
de-duplicated against the input CSV.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)


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


def synthesize_late_seam_ath(words, labels, target_pos_min, target_pos_max,
                             min_post_seam_words, min_total_words, rng):
    """For a 1->0 doc, trim the human tail so the seam lands at
    [target_pos_min, target_pos_max] of the resulting doc.
    """
    if labels[0] != 1:                       # must start with AI
        return None
    seam = first_seam_index(labels)
    if seam is None:
        return None

    n_pre_seam = seam                        # AI portion length
    n_post_seam = len(words) - seam          # human portion length
    if n_post_seam < min_post_seam_words:
        return None

    new_total_min = max(int(seam / target_pos_max),
                        n_pre_seam + min_post_seam_words,
                        min_total_words)
    new_total_max = min(int(seam / target_pos_min), len(words))
    if new_total_min > new_total_max:
        return None

    boundary = find_sentence_boundary_word_idx(words, n_pre_seam + min_post_seam_words, new_total_max)
    if boundary is not None and boundary >= new_total_min:
        new_total = boundary
    else:
        new_total = rng.randint(new_total_min, new_total_max)

    if new_total > len(words):
        return None

    new_words = words[:new_total]
    new_labels = labels[:new_total]
    # Sanity: seam must still be in the slice (start and end labels must differ)
    if new_labels[0] == new_labels[-1]:
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
    ap.add_argument("--min-post-seam-words", type=int, default=10)
    ap.add_argument("--min-total-words", type=int, default=50)
    ap.add_argument("--balance-sources", default="pile,common_crawl")
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    sources = [s.strip() for s in args.balance_sources.split(",") if s.strip()]
    per_source_target = args.target_count // len(sources)
    print(f"Balancing across {len(sources)} sources: {sources}")
    print(f"Target rows per source: {per_source_target}")

    rng = random.Random(args.seed)

    candidates_by_source = defaultdict(list)
    fieldnames = None
    skipped = Counter()
    seen_hashes = set()

    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            text = row.get("text") or ""
            seen_hashes.add(text_hash(text))
            try:
                labels = [int(x) for x in json.loads(row.get("segmentation_labels") or "[]")]
            except (json.JSONDecodeError, ValueError, TypeError):
                skipped["bad_labels"] += 1
                continue
            words = text.split()
            if len(words) != len(labels) or len(words) < args.min_total_words:
                skipped["wrong_length"] += 1
                continue
            if labels[0] != 1:
                skipped["doesnt_start_with_ai"] += 1
                continue
            src = row.get("data_source") or "?"
            if src not in sources:
                skipped[f"source_{src}"] += 1
                continue
            candidates_by_source[src].append((row, words, labels))

    print(f"\nLoaded 1->0 candidates by source:")
    for src in sources:
        print(f"  {src:<22s} {len(candidates_by_source[src]):>8,}")
    if skipped:
        print(f"  Skipped (not eligible):")
        for r, n in skipped.most_common():
            print(f"    {r:<32s} {n:>7,}")

    out_rows = []
    out_stats_by_source = {s: Counter() for s in sources}
    seam_pos_dist = Counter()
    output_hashes = set()

    for src in sources:
        candidates = list(candidates_by_source[src])
        rng.shuffle(candidates)
        produced = 0
        stats = out_stats_by_source[src]
        for row, words, labels in candidates:
            if produced >= per_source_target:
                break
            result = synthesize_late_seam_ath(
                words, labels,
                target_pos_min=args.target_pos_min,
                target_pos_max=args.target_pos_max,
                min_post_seam_words=args.min_post_seam_words,
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
            new_row["sample_type"] = "ai_then_human"
            if "augmented" in fieldnames:
                new_row["augmented"] = "late_seam_synth_ath"
            out_rows.append(new_row)
            stats["generated"] += 1
            produced += 1
            if new_pos < 0.80:    seam_pos_dist["0.75-0.80"] += 1
            elif new_pos < 0.85:  seam_pos_dist["0.80-0.85"] += 1
            elif new_pos < 0.90:  seam_pos_dist["0.85-0.90"] += 1
            else:                  seam_pos_dist["0.90-0.95"] += 1

    print(f"\nGenerated {len(out_rows):,} rows.\n")
    for src in sources:
        s = out_stats_by_source[src]
        print(f"  [{src}]")
        for r, n in s.most_common():
            print(f"      {r:<28s} {n:>7,}")

    print(f"\nSeam position distribution:")
    for b in ("0.75-0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95"):
        print(f"  {b:<10s} {seam_pos_dist.get(b, 0):>7,}")
    word_counts = [int(r["n_words"]) for r in out_rows]
    if word_counts:
        print(f"\nWord-count: min={min(word_counts)}  "
              f"avg={sum(word_counts)/len(word_counts):.1f}  max={max(word_counts)}")
    src_counts = Counter(r.get("data_source") for r in out_rows)
    print(f"\nFinal data_source distribution:")
    for s, n in src_counts.most_common():
        pct = 100*n/len(out_rows) if out_rows else 0
        print(f"  {s:<22s} {n:>7,}  ({pct:>5.1f}%)")

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
