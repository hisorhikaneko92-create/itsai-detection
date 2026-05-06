"""
Rewrite the `sample_type` column in train_final.csv / test_final.csv / val_final.csv
to reflect the *actual* segmentation_labels content, using the standard convention:

    0            -> pure_human
    1            -> pure_ai
    0->1         -> human_then_ai
    1->0         -> ai_then_human
    0->1->0      -> ai_in_middle
    1->0->1      -> human_in_middle
    0->1->0->1+  -> multi_seam

The CSV's old `sample_type` was set at generation time and got stale after the
validator's subsample_words truncation chopped the leading or trailing portion
off many rows. After this script, sample_type matches what the labels actually
show, so any downstream code that filters/oversamples by sample_type gets
accurate slices.

Streams row-by-row through a tmp file then atomic-renames, so files are never
left half-written if the script is interrupted. Originals can be regenerated
via scripts/merge_datasets.py if you ever want to revert.
"""
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

DATA_DIR = Path("data/Training_Dataset")
TARGETS = ["train_final.csv", "test_final.csv", "val_final.csv"]


def transitions(labels):
    """Compress runs of equal labels into a tuple, e.g. [0,0,1,1,1,0] -> (0,1,0)."""
    if not labels:
        return ()
    runs = [labels[0]]
    for v in labels[1:]:
        if v != runs[-1]:
            runs.append(v)
    return tuple(runs)


def shape_to_name(pat):
    if pat == (0,):                 return "pure_human"
    if pat == (1,):                 return "pure_ai"
    if pat == (0, 1):               return "human_then_ai"
    if pat == (1, 0):               return "ai_then_human"
    if pat == (0, 1, 0):            return "ai_in_middle"
    if pat == (1, 0, 1):            return "human_in_middle"
    if len(pat) >= 4:               return "multi_seam"
    if not pat:                     return "empty"
    return "unknown"


def relabel(path: Path):
    if not path.exists():
        print(f"SKIP {path} (not found)")
        return None, 0

    transition_counts = Counter()  # (old, new) -> count
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0

    with open(path, "r", encoding="utf-8", newline="") as fin, \
         open(tmp, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        fields = list(reader.fieldnames or [])
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            n += 1
            old = row.get("sample_type", "?")
            try:
                labels = json.loads(row.get("segmentation_labels", "[]"))
                new = shape_to_name(transitions(labels))
            except Exception:
                new = "BAD_LABELS"
            transition_counts[(old, new)] += 1
            row["sample_type"] = new
            writer.writerow({k: row.get(k, "") for k in fields})

    os.replace(tmp, path)
    return transition_counts, n


def report(name, tc, n):
    if tc is None:
        return 0
    print(f"\n{'='*68}\n{name}    ({n:,} rows)\n{'='*68}")

    # Group by old sample_type, show what each became
    by_old = Counter()
    by_new = Counter()
    for (old, new), c in tc.items():
        by_old[old] += c
        by_new[new] += c

    changed = sum(c for (old, new), c in tc.items() if old != new)

    print(f"\nOld sample_type -> new sample_type:")
    for old in sorted(by_old.keys()):
        print(f"\n  {old}  (was {by_old[old]:,})")
        inner = [(new, c) for (o, new), c in tc.items() if o == old]
        inner.sort(key=lambda x: -x[1])
        for new, c in inner:
            pct = 100 * c / by_old[old]
            mark = "  " if new == old else "->"
            print(f"    {mark} {new:<20s} {c:>8,}  ({pct:>5.1f}%)")

    print(f"\nFinal sample_type distribution after relabel:")
    for new in sorted(by_new.keys()):
        pct = 100 * by_new[new] / n
        print(f"    {new:<20s} {by_new[new]:>8,}  ({pct:>5.1f}%)")

    print(f"\nRows whose sample_type changed: {changed:,}  ({100*changed/n:.1f}% of file)")
    return changed


def main():
    total = 0
    for name in TARGETS:
        path = DATA_DIR / name
        tc, n = relabel(path)
        total += report(path.name, tc, n)

    print(f"\n{'='*68}")
    print(f"GRAND TOTAL rows relabeled across all 3 files: {total:,}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
