"""
Recover long rows (>350 words) from the source CSVs by splitting them into
consecutive validator-shaped chunks (35..350 words each, single-seam or
no-seam structure preferred), then append the chunks to train_final.csv.

Why source CSVs?  The long rows were already filtered out of train_final.csv
by filter_word_count.py. They still exist in the originals (train.csv,
adv_train.csv, adv_ai_in_middle.csv, adv_human_then_ai.csv,
adv_human_then_ai_0.csv).

Splitting strategy:
  - Walk the row in chunks of TARGET_CHUNK +/- jitter words
  - Don't leave a final tail < MIN_CHUNK words (merge into previous chunk if so)
  - Each emitted chunk has its sample_type re-derived from its actual labels
  - Multi-seam chunks (3+ seams) are dropped (off validator distribution)

Dedup against current train_final.csv by exact text hash.
"""
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

DATA_DIR = Path("data/Training_Dataset")
TRAIN_FINAL = DATA_DIR / "train_final.csv"
SOURCES = [
    "train.csv",
    "adv_train.csv",
    "adv_ai_in_middle.csv",
    "adv_human_then_ai.csv",
    "adv_human_then_ai_0.csv",
]
MIN_CHUNK = 35
MAX_CHUNK = 350
TARGET_CHUNK = 200
JITTER = 50          # chunks land in [TARGET-JITTER, TARGET+JITTER]
VALIDATOR_LONG_THRESHOLD = 350
SEED = 42


def num_seams(labels):
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])


def shape_to_name(labels):
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


def text_hash(text):
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def split_long_row(words, labels, rng):
    """Split into consecutive chunks of size in [TARGET-JITTER, TARGET+JITTER]
    words, ensuring no chunk falls below MIN_CHUNK."""
    N = len(words)
    chunks = []
    pos = 0
    while pos < N:
        remaining = N - pos
        if remaining <= MAX_CHUNK:
            if remaining >= MIN_CHUNK:
                chunks.append((words[pos:], labels[pos:]))
            elif chunks:
                # Merge tail into previous chunk if it fits
                pw, pl = chunks[-1]
                if len(pw) + remaining <= MAX_CHUNK:
                    chunks[-1] = (pw + words[pos:], pl + labels[pos:])
            break
        size = rng.randint(TARGET_CHUNK - JITTER, TARGET_CHUNK + JITTER)
        # Don't leave a sub-MIN tail
        if N - (pos + size) < MIN_CHUNK and N - pos > size:
            size = max(MIN_CHUNK, N - pos - MIN_CHUNK)
            size = min(size, MAX_CHUNK)
        chunks.append((words[pos:pos+size], labels[pos:pos+size]))
        pos += size
    return chunks


def main():
    rng = random.Random(SEED)

    # ---- Build hash set from current train_final.csv ------------------
    seen_hashes = set()
    fieldnames = []
    if not TRAIN_FINAL.exists():
        sys.exit(f"Not found: {TRAIN_FINAL}")
    with open(TRAIN_FINAL, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            seen_hashes.add(text_hash(row.get("text", "")))
    print(f"Loaded {len(seen_hashes):,} text-hashes from train_final.csv")

    # ---- Collect unique long rows from source CSVs --------------------
    long_rows = []
    src_seen = set()
    print(f"\nScanning sources for rows with > {VALIDATOR_LONG_THRESHOLD} words:")
    for name in SOURCES:
        path = DATA_DIR / name
        if not path.exists():
            print(f"  SKIP {name} (not found)")
            continue
        n_long = 0
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text") or ""
                h = text_hash(text)
                if h in src_seen:
                    continue
                src_seen.add(h)
                words = text.split()
                if len(words) > VALIDATOR_LONG_THRESHOLD:
                    long_rows.append((row, words))
                    n_long += 1
        print(f"  {name:<35s} {n_long:>5,} long rows")
    print(f"Total unique long rows: {len(long_rows):,}")

    # ---- Split + dedup against train_final.csv ------------------------
    out_rows = []
    chunks_per_row = []
    skipped_dup = 0
    skipped_multi_seam = 0
    skipped_invalid = 0
    chunk_dist = Counter()
    chunk_lens = []

    for row, words in long_rows:
        try:
            labels = json.loads(row.get("segmentation_labels") or "[]")
        except Exception:
            skipped_invalid += 1
            continue
        if len(words) != len(labels):
            skipped_invalid += 1
            continue
        chunks = split_long_row(words, labels, rng)
        chunks_per_row.append(len(chunks))
        for cw, cl in chunks:
            if len(cw) < MIN_CHUNK or len(cw) > MAX_CHUNK:
                skipped_invalid += 1
                continue
            if num_seams(cl) >= 3:
                skipped_multi_seam += 1
                continue
            new_text = " ".join(cw)
            h = text_hash(new_text)
            if h in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(h)
            new_row = dict(row)
            new_row["text"] = new_text
            new_row["segmentation_labels"] = json.dumps(cl)
            new_row["n_words"] = str(len(cw))
            new_row["sample_type"] = shape_to_name(cl)
            if "augmented" in fieldnames:
                new_row["augmented"] = "long_row_split"
            out_rows.append(new_row)
            chunk_dist[new_row["sample_type"]] += 1
            chunk_lens.append(len(cw))

    # ---- Append to train_final.csv ------------------------------------
    with open(TRAIN_FINAL, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # ---- Report -------------------------------------------------------
    print(f"\nSplit {len(long_rows):,} long rows into {len(out_rows):,} chunks")
    if chunks_per_row:
        print(f"  avg chunks per long row: {sum(chunks_per_row)/len(chunks_per_row):.1f}")
    print(f"  skipped (duplicate vs train_final): {skipped_dup:,}")
    print(f"  skipped (multi-seam after split):   {skipped_multi_seam:,}")
    print(f"  skipped (invalid):                  {skipped_invalid:,}")
    if chunk_lens:
        print(f"\nChunk word counts: "
              f"min={min(chunk_lens)}  avg={sum(chunk_lens)/len(chunk_lens):.1f}  "
              f"max={max(chunk_lens)}")
    print(f"\nChunks by sample_type:")
    for st, c in chunk_dist.most_common():
        print(f"  {st:<22s} {c:>8,}")

    # Final wc
    n_final = sum(1 for _ in open(TRAIN_FINAL, "r", encoding="utf-8")) - 1
    print(f"\ntrain_final.csv now has {n_final:,} rows")


if __name__ == "__main__":
    main()
