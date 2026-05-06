"""Count seam-transition patterns within each sample_type.

Shows how many label-sequence shapes (0->1, 1->0, 0->1->0, 1->0->1, none, etc.)
exist within each declared sample_type, so we can see whether 'ai_in_middle'
really means 0->1->0 in the data or whether it's been truncated to single
seams by the validator's subsample_words logic.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)


def transitions(labels):
    """Return a tuple describing the label-pattern structure, e.g.
    (0,)               -> all zeros (pure human)
    (1,)               -> all ones (pure AI)
    (0, 1)             -> 0->1 single seam (human_then_ai)
    (1, 0)             -> 1->0 single seam (ai_then_human)
    (0, 1, 0)          -> 0->1->0 (AI in middle of human)
    (1, 0, 1)          -> 1->0->1 (human in middle of AI)
    (0, 1, 0, 1, ...)  -> multi-seam
    """
    if not labels:
        return ()
    runs = [labels[0]]
    for v in labels[1:]:
        if v != runs[-1]:
            runs.append(v)
    return tuple(runs)


def main():
    path = Path("data/Training_Dataset/train_final.csv")
    if not path.exists():
        sys.exit(f"Not found: {path}")

    by_sample_type = defaultdict(Counter)
    by_sample_type_seams = defaultdict(Counter)

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                labels = json.loads(row["segmentation_labels"])
            except Exception:
                continue
            st = row.get("sample_type", "?")
            pat = transitions(labels)
            by_sample_type[st][pat] += 1
            n_seams = max(0, len(pat) - 1)
            by_sample_type_seams[st][n_seams] += 1

    print(f"{'sample_type':<18s} {'#seams':>7s}  count")
    print("-" * 46)
    for st in sorted(by_sample_type_seams.keys()):
        for n_seams, count in sorted(by_sample_type_seams[st].items()):
            print(f"{st:<18s} {n_seams:>7d}  {count:>8,}")
        print()

    print("=" * 64)
    print("Top label-shapes per sample_type")
    print("=" * 64)
    for st, patcount in by_sample_type.items():
        total = sum(patcount.values())
        print(f"\n{st}  (total {total:,})")
        for pat, count in patcount.most_common(8):
            pct = 100.0 * count / total
            label = "->".join(str(x) for x in pat) or "(empty)"
            print(f"    {label:<28s} {count:>8,}  ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
