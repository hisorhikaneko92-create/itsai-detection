"""
Chunk training-CSV rows so every row falls within [min_words, max_words].
Default range is the validator's window: 35-350 words.

Built specifically for data/train_archive.csv (where rows can be 1500+
words because they came from raw Wikipedia / ArXiv / WikiHow JSONL),
but works on any CSV with the same columns:
    text, segmentation_labels, data_source, sample_type, model_name,
    n_words, augmented

Behavior per row:
  * 35 <= n_words <= 350  -> kept verbatim
  * n_words < 35          -> dropped (rare; informational warning)
  * n_words > 350         -> split into k = ceil(n_words / max_words)
    roughly-equal NON-OVERLAPPING chunks. The split is at word
    boundaries; segmentation_labels is sliced in lockstep so each
    chunk's labels still align 1:1 with text.split().

Reclassification of split chunks:
  * If a chunk is all 0s -> sample_type = pure_human, model_name = "none"
  * If a chunk is all 1s -> sample_type = pure_ai (model_name kept)
  * If a chunk has both  -> sample_type kept as the parent's
                            (human_then_ai stays human_then_ai if the
                             0->1 boundary survives in this chunk;
                             ai_in_middle stays ai_in_middle if both
                             0->1 and 1->0 are inside this chunk; etc.)
This means a 1000-word human_then_ai with the seam at word 500 split
into 3 chunks typically yields pure_human + human_then_ai + pure_ai --
which is *more* training signal, not less.

Why non-overlapping (no stride): overlapping chunks would create near-
duplicate rows that could leak across train / val / test after
split_dataset.py runs, biasing the held-out metrics.

Usage:
    # Chunk train_archive.csv -> new file
    python scripts/chunk_long_rows.py \\
        --input data/train_archive.csv \\
        --output data/train_archive_chunked.csv

    # Overwrite in place (the input is read first, then truncated)
    python scripts/chunk_long_rows.py \\
        --input data/train_archive.csv \\
        --in-place

    # Custom word window (e.g. allow up to 512 words for DeBERTa max_length)
    python scripts/chunk_long_rows.py \\
        --input data/train_archive.csv \\
        --output data/train_archive_512.csv \\
        --max-words 512
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def split_into_chunks(words: List[str], labels: List[int],
                      max_words: int) -> List[Tuple[List[str], List[int]]]:
    """Roughly-equal non-overlapping word-level partition. The number
    of chunks is ceil(n / max_words); chunk sizes differ by at most 1
    so the partition is as balanced as possible. Concatenating the
    output reconstructs the input exactly."""
    n = len(words)
    if n <= max_words:
        return [(words, labels)]
    n_chunks = math.ceil(n / max_words)
    base = n // n_chunks
    rem = n % n_chunks
    out: List[Tuple[List[str], List[int]]] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        end = start + size
        out.append((words[start:end], labels[start:end]))
        start = end
    return out


def reclassify(parent_type: str, parent_model: str,
               labels: List[int]) -> Tuple[str, str]:
    """Given the parent row's sample_type/model_name and a chunk's
    labels, return (sample_type, model_name) for the chunk. Pure
    chunks get demoted; mixed chunks inherit the parent's type."""
    has_h = any(l == 0 for l in labels)
    has_a = any(l == 1 for l in labels)
    if has_h and not has_a:
        return "pure_human", "none"
    if has_a and not has_h:
        return "pure_ai", parent_model
    # Mixed chunk: keep parent type. Note that if parent was
    # ai_in_middle but this chunk only contains the first 0->1 (no
    # 1->0), it's structurally a human_then_ai now. We leave it as
    # ai_in_middle because the seam-bucket classifier in
    # split_dataset.py inspects the actual labels, not the type
    # string -- so the strat split still gets the right bucket.
    return parent_type, parent_model


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True,
                    help="Input CSV. Must have the standard training "
                         "columns (text, segmentation_labels, "
                         "data_source, sample_type, model_name, "
                         "n_words, augmented).")
    out_grp = ap.add_mutually_exclusive_group(required=True)
    out_grp.add_argument("--output",
                         help="Output CSV path. Cannot equal --input "
                              "unless you pass --in-place.")
    out_grp.add_argument("--in-place", action="store_true",
                         help="Read --input first, then OVERWRITE it. "
                              "Safe -- the input is fully read into "
                              "memory before the file is opened for "
                              "writing.")
    ap.add_argument("--max-words", type=int, default=350,
                    help="Maximum words per output chunk. Default 350 "
                         "matches the validator's window upper bound.")
    ap.add_argument("--min-words", type=int, default=35,
                    help="Minimum words per output chunk. Chunks "
                         "smaller than this are dropped. Default 35 "
                         "matches the validator's window lower bound.")
    args = ap.parse_args()

    if args.max_words < 50:
        sys.exit(f"--max-words {args.max_words} is unreasonably small")
    if args.min_words >= args.max_words:
        sys.exit(f"--min-words ({args.min_words}) must be < "
                 f"--max-words ({args.max_words})")

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")
    out_path = in_path if args.in_place else Path(args.output)

    # Read everything first (so --in-place is safe).
    print(f"Reading {in_path}")
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            sys.exit(f"{in_path} has no header")
        required = {"text", "segmentation_labels", "sample_type",
                    "model_name", "n_words"}
        missing = required - set(fieldnames)
        if missing:
            sys.exit(f"Input is missing required columns: {sorted(missing)}")
        rows = list(reader)
    print(f"  {len(rows):,} input rows")

    # Stats
    n_in = len(rows)
    n_out = 0
    n_dropped_short = 0
    n_kept_short_input = 0
    n_split = 0
    n_chunks_emitted = 0
    type_in: Counter = Counter()
    type_out: Counter = Counter()
    parent_to_chunks: Counter = Counter()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            parent_type = row.get("sample_type", "?")
            parent_model = row.get("model_name", "unknown") or "unknown"
            type_in[parent_type] += 1

            text = row.get("text", "") or ""
            words = text.split()
            try:
                labels = json.loads(row.get("segmentation_labels", "[]"))
            except json.JSONDecodeError:
                n_dropped_short += 1
                continue
            if not isinstance(labels, list) or len(labels) != len(words):
                # Length mismatch -> can't safely chunk. Skip.
                n_dropped_short += 1
                continue

            if len(words) < args.min_words:
                # Below validator's floor; drop.
                n_dropped_short += 1
                continue

            if len(words) <= args.max_words:
                # Already within range -- pass through unchanged.
                writer.writerow(row)
                n_out += 1
                type_out[parent_type] += 1
                parent_to_chunks[1] += 1
                continue

            # Long row: split.
            chunks = split_into_chunks(words, labels, args.max_words)
            n_split += 1
            kept_this_row = 0
            for chunk_words, chunk_labels in chunks:
                if len(chunk_words) < args.min_words:
                    # Tail too short; drop. With balanced splitting +
                    # max_words >= 350 this is rare but possible if a
                    # row is exactly 351-369 words long (chunk1 = 350,
                    # chunk2 = 1-19). The min_size enforcement above
                    # would have caught it but we'd rather drop the
                    # tail than emit an invalid row.
                    n_dropped_short += 1
                    continue
                new_type, new_model = reclassify(
                    parent_type, parent_model, chunk_labels,
                )
                new_text = " ".join(chunk_words)
                new_row = {k: row.get(k, "") for k in fieldnames}
                new_row["text"] = new_text
                new_row["segmentation_labels"] = json.dumps(chunk_labels)
                new_row["sample_type"] = new_type
                new_row["model_name"] = new_model
                new_row["n_words"] = len(chunk_words)
                writer.writerow(new_row)
                n_out += 1
                kept_this_row += 1
                type_out[new_type] += 1
            n_chunks_emitted += kept_this_row
            parent_to_chunks[kept_this_row] += 1

    print()
    print(f"Input  rows: {n_in:,}")
    print(f"Output rows: {n_out:,}  (split {n_split:,} long rows -> "
          f"{n_chunks_emitted:,} chunks)")
    print(f"Dropped (too short or malformed): {n_dropped_short:,}")
    print(f"Sample-type before -> after:")
    for t in sorted(set(type_in) | set(type_out)):
        print(f"  {t:<14s}  {type_in.get(t, 0):>9,}  ->  "
              f"{type_out.get(t, 0):>9,}")
    print(f"Chunks-per-long-row distribution:")
    for k in sorted(parent_to_chunks):
        if k <= 1:
            continue
        print(f"  {k} chunks: {parent_to_chunks[k]:>7,} rows")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
