"""
Synthesize late-seam training samples from existing human_then_ai and
ai_in_middle samples by trimming the AI portion's tail until the first
0->1 seam lands in the 80-95% region of the document.

Why: late-seam detection is the worst-performing bucket on test
(f1@5 ~0.63 vs ~0.82 overall). The dataset is over-represented in
middle seams because the default generation produces roughly
equal-length human + AI portions. This script creates late-seam
variants from those existing samples by sentence-boundary truncation
of the AI tail -- no API calls, ~30 seconds runtime.

Output rows are valid training samples in the same schema as
build_training_dataset.py:
    text, segmentation_labels, data_source, sample_type, model_name,
    n_words, augmented (optional)

Trimming respects sentence boundaries when possible -- we look for the
last sentence-ending punctuation in the candidate AI portion and cut
there, so the AI continuation still reads coherently. This avoids
teaching the model "incomplete sentence => AI" as a shortcut.

Usage:
    # Default: synthesize 8000 late-seam samples
    python scripts/synthesize_late_seam.py \\
        --input data/Training_Dataset/train.csv \\
        --output data/late_seam_synth.csv

    # Larger / smaller batches
    python scripts/synthesize_late_seam.py \\
        --input data/Training_Dataset/train.csv \\
        --output data/late_seam_synth.csv \\
        --target-count 12000

    # Different seam window (e.g. 75-90% instead of 80-95%)
    python scripts/synthesize_late_seam.py \\
        --input data/Training_Dataset/train.csv \\
        --output data/late_seam_synth.csv \\
        --target-pos-min 0.75 --target-pos-max 0.90
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

# Bump CSV field-size limit; some training rows have very long text fields.
csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

# Sentence-ending punctuation we'll snap to when trimming the AI portion.
# Match punctuation followed by whitespace OR end-of-string.
SENTENCE_END = re.compile(r'[.!?]["\')\]]?(?:\s|$)')


def first_seam_index(labels: List[int]) -> Optional[int]:
    """Return position of the first 0/1 (or 1/0) transition, or None."""
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            return i
    return None


def find_sentence_boundary_word_idx(words: List[str],
                                    start: int,
                                    end: int) -> Optional[int]:
    """Within words[start:end], find the index of the LAST word whose
    rendered character ends with sentence-ending punctuation. Returns
    the word index just AFTER that word (so it's a valid trim point),
    or None if no sentence boundary exists in the range.
    """
    for i in range(end - 1, start - 1, -1):
        # Strip trailing quotes/brackets to expose the punctuation
        stripped = words[i].rstrip('"\')]')
        if stripped and stripped[-1] in '.!?':
            return i + 1   # cut AFTER this word
    return None


def synthesize_late_seam(
    words: List[str],
    labels: List[int],
    target_pos_min: float,
    target_pos_max: float,
    min_ai_words: int,
    min_total_words: int,
    rng: random.Random,
) -> Optional[Tuple[List[str], List[int]]]:
    """Trim words+labels so the first seam lands in
    [target_pos_min, target_pos_max] of the doc, preferring to cut at a
    sentence boundary inside the AI portion. Returns the trimmed
    (words, labels) or None if no valid trim exists.
    """
    if labels[0] != 0:
        return None  # Need to start with human for "human_then_ai" semantics

    seam = first_seam_index(labels)
    if seam is None:
        return None

    n_human = seam            # words 0..seam-1 are human
    n_ai_total = len(words) - seam

    if n_ai_total < min_ai_words:
        return None  # AI portion already too short

    # We want: seam / new_total ∈ [target_pos_min, target_pos_max]
    # → new_total ∈ [seam / target_pos_max, seam / target_pos_min]
    new_total_min = max(
        int(seam / target_pos_max),
        n_human + min_ai_words,
        min_total_words,
    )
    new_total_max = min(
        int(seam / target_pos_min),
        len(words),
    )

    if new_total_min > new_total_max:
        return None

    # Try to find a sentence boundary inside the candidate AI window.
    # Search range in word indices: [n_human + min_ai_words, new_total_max]
    boundary = find_sentence_boundary_word_idx(
        words, n_human + min_ai_words, new_total_max,
    )

    if boundary is not None and boundary >= new_total_min:
        new_total = boundary
    else:
        # No sentence boundary -- pick a random length in [min, max].
        new_total = rng.randint(new_total_min, new_total_max)

    if new_total > len(words):
        return None

    return words[:new_total], labels[:new_total]


def seam_position_bucket(labels: List[int]) -> str:
    """Match the test-eval bucket scheme (early / middle / late)."""
    seam = first_seam_index(labels)
    if seam is None:
        return "no_seam"
    pos = seam / max(1, len(labels))
    if pos < 0.20:
        return "early"
    if pos > 0.80:
        return "late"
    return "middle"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True,
                    help="Source training CSV with text + segmentation_labels.")
    ap.add_argument("--output", required=True,
                    help="Destination CSV. Same schema as input.")
    ap.add_argument("--target-count", type=int, default=8000,
                    help="Target number of late-seam samples to emit. "
                         "Default 8000.")
    ap.add_argument("--target-pos-min", type=float, default=0.80,
                    help="Minimum seam fractional position. Default 0.80.")
    ap.add_argument("--target-pos-max", type=float, default=0.95,
                    help="Maximum seam fractional position. Default 0.95.")
    ap.add_argument("--min-ai-words", type=int, default=10,
                    help="Minimum AI portion length in words. Below this "
                         "the model has no signal to learn from. Default 10.")
    ap.add_argument("--min-total-words", type=int, default=50,
                    help="Minimum total document length in words. Default 50.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---- Read all candidate rows --------------------------------------
    candidates: List[Tuple[dict, List[str], List[int]]] = []
    fieldnames: Optional[List[str]] = None
    skipped: Counter = Counter()

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if row.get("sample_type") not in ("human_then_ai", "ai_in_middle"):
                skipped["wrong_sample_type"] += 1
                continue
            text = row.get("text") or ""
            labels_str = row.get("segmentation_labels") or ""
            try:
                labels = [int(x) for x in json.loads(labels_str)]
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
            candidates.append((row, words, labels))

    print(f"Loaded {len(candidates):,} candidate samples")
    if skipped:
        print(f"  Skipped:")
        for reason, n in skipped.most_common():
            print(f"    {reason:<28s} {n:>7,}")

    # Shuffle so we draw from the full distribution, not just the start
    rng.shuffle(candidates)

    # ---- Synthesize ---------------------------------------------------
    output_rows: List[dict] = []
    out_stats: Counter = Counter()
    seam_pos_dist: Counter = Counter()

    for row, words, labels in candidates:
        if len(output_rows) >= args.target_count:
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
            out_stats["skipped_infeasible"] += 1
            continue

        new_words, new_labels = result
        new_seam = first_seam_index(new_labels)
        new_pos = new_seam / len(new_labels) if new_seam else 0.0

        # Final validity check
        if not (args.target_pos_min <= new_pos <= args.target_pos_max):
            out_stats["skipped_out_of_range"] += 1
            continue

        new_row = dict(row)
        new_row["text"] = " ".join(new_words)
        new_row["segmentation_labels"] = json.dumps(new_labels)
        new_row["n_words"] = str(len(new_words))
        # Original row may have been ai_in_middle (2 seams); after trim
        # we cut off the second seam, so it's now human_then_ai.
        new_row["sample_type"] = "human_then_ai"
        if "augmented" in (new_row.keys() if isinstance(new_row, dict) else fieldnames):
            new_row["augmented"] = "late_seam_synth"

        output_rows.append(new_row)
        out_stats["generated"] += 1

        # Position bucket histogram for the report
        bucket = "0.80-0.85" if new_pos < 0.85 else (
            "0.85-0.90" if new_pos < 0.90 else "0.90-0.95"
        )
        seam_pos_dist[bucket] += 1

    # ---- Write output -------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # ---- Report -------------------------------------------------------
    print(f"\nWrote {len(output_rows):,} late-seam samples to {out_path}")
    print(f"Synthesis stats:")
    for reason, n in out_stats.most_common():
        print(f"  {reason:<28s} {n:>7,}")
    print(f"\nSeam position distribution:")
    for bucket in ("0.80-0.85", "0.85-0.90", "0.90-0.95"):
        print(f"  {bucket:<10s} {seam_pos_dist.get(bucket, 0):>7,}")

    # Source-of-input model breakdown so we can confirm we're not all
    # one model
    model_counts: Counter = Counter()
    for r in output_rows:
        model_counts[r.get("model_name", "?")] += 1
    print(f"\nModel name distribution in output:")
    for m, n in model_counts.most_common():
        print(f"  {m:<50s} {n:>5,}")


if __name__ == "__main__":
    main()
