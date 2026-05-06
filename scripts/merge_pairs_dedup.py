"""
Merge each pair (X.csv, Y.csv) by appending Y's rows to X, deduplicating
by text-hash so rows already present in X aren't added twice.

Pairs:
  train_final.csv + adv_train.csv  -> train_final.csv
  val_final.csv   + adv_val.csv    -> val_final.csv
  test_final.csv  + adv_test.csv   -> test_final.csv

Operates in place on the *_final.csv targets via tmp + atomic rename.
Reports per-pair: rows considered, appended, skipped as duplicates.
"""
import csv
import hashlib
import os
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

DATA_DIR = Path("data/Training_Dataset")

PAIRS = [
    ("train_final.csv", "adv_train.csv"),
    ("val_final.csv",   "adv_val.csv"),
    ("test_final.csv",  "adv_test.csv"),
]


def text_hash(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def merge_pair(target: Path, source: Path):
    if not target.exists():
        print(f"SKIP — target not found: {target}")
        return
    if not source.exists():
        print(f"SKIP — source not found: {source}")
        return

    # Pass 1: read target, build hash set + capture fieldnames + initial count
    seen = set()
    fieldnames = []
    initial_count = 0
    with open(target, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            seen.add(text_hash(row.get("text", "")))
            initial_count += 1

    # Pass 2: copy target -> tmp, then stream source rows, appending
    # only those whose text-hash isn't already in `seen`.
    tmp = target.with_suffix(target.suffix + ".tmp")
    appended = 0
    skipped = 0

    with open(tmp, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        # Copy existing target rows
        with open(target, "r", encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        # Append non-duplicate source rows
        with open(source, "r", encoding="utf-8", newline="") as fsrc:
            src_reader = csv.DictReader(fsrc)
            for row in src_reader:
                h = text_hash(row.get("text", ""))
                if h in seen:
                    skipped += 1
                    continue
                seen.add(h)
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                appended += 1

    os.replace(tmp, target)

    final_count = initial_count + appended
    print(f"\n{target.name} <- {source.name}")
    print(f"  initial rows in target:   {initial_count:>8,}")
    print(f"  source rows considered:   {appended + skipped:>8,}")
    print(f"  appended (new):           {appended:>8,}")
    print(f"  skipped (duplicates):     {skipped:>8,}")
    print(f"  final rows in target:     {final_count:>8,}")


def main():
    print(f"Merging pairs in {DATA_DIR}\n" + "=" * 64)
    for tgt, src in PAIRS:
        merge_pair(DATA_DIR / tgt, DATA_DIR / src)
    print("\n" + "=" * 64)
    print("Done.")


if __name__ == "__main__":
    main()
