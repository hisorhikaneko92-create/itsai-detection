"""
All in-dataset cut transformations for the rebalance plan.

Reads train_final.csv (untouched) and writes 4 new files in data/Training_Dataset/:

  1. cuts_cc_a0_10.csv          — CC a10-25 → CC a0-10 (target 323)
                                  Method: cut front AI words, seam moves earlier in shorter doc
  2. cuts_pile_h50_75.csv       — Pile h75-90 → Pile h50-75 (target 3,144)
                                  Method: cut front human words
  3. cuts_pile_a_early.csv      — Pile a25-50/a50-75 → Pile a0-10/a10-25 (target 1,769 + 1,720)
                                  Method: cut front AI words
  4. cuts_pile_pure_ai_from_a.csv — Pile a75-90/a90+ → Pile pure_ai (~10,269)
                                    Method: cut human tail; result is all-AI labels=1

Math reminder for cuts (front-cut a words, original seam at S, total N):
  new_total = N - a
  new_seam  = S - a
  new_pos   = (S - a) / (N - a)
  To land in [target_min, target_max):
      a > (S - target_max*N) / (1 - target_max)
      a <= (S - target_min*N) / (1 - target_min)
"""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

DATA_DIR = Path("data/Training_Dataset")
INPUT = DATA_DIR / "train_final.csv"
SEED = 42
MIN_LEN = 35
MAX_LEN = 350

TARGET_CC_A0_10 = 323
TARGET_PILE_H50_75 = 3144
TARGET_PILE_A0_10 = 1769
TARGET_PILE_A10_25 = 1720
# Cap cuts to pure_ai per source bucket (= surplus only) so a75-90 and a90+
# buckets retain their targets after transformation.
CAP_PILE_A75_90_TO_PURE_AI = 8491   # surplus of pile a75-90
CAP_PILE_A90_PLUS_TO_PURE_AI = 1778  # surplus of pile a90+


def num_seams(labels):
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])


def first_seam_idx(labels):
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            return i
    return None


def cut_front_to_target_pos(words, labels, S, N,
                            target_pos_min, target_pos_max, rng):
    """Generic front-cut to land seam in [target_pos_min, target_pos_max)."""
    if target_pos_max >= 1.0:
        a_min_pos = 1
    else:
        a_min_pos = (S - target_pos_max * N) / (1 - target_pos_max)
    if target_pos_min <= 0.0:
        a_max_pos = S - 1
    else:
        a_max_pos = (S - target_pos_min * N) / (1 - target_pos_min)

    a_min = max(1, int(a_min_pos) + 1)
    a_max = min(int(a_max_pos), S - 1, N - MIN_LEN)
    if a_min > a_max:
        return None

    # Sample within [a_min, a_max], retry up to 5 times to land in target band
    for _ in range(5):
        a = rng.randint(a_min, a_max)
        new_words = words[a:]
        new_labels = labels[a:]
        if len(new_words) < MIN_LEN or len(new_words) > MAX_LEN:
            continue
        new_seam = S - a
        new_pos = new_seam / len(new_labels)
        if target_pos_min <= new_pos < target_pos_max:
            return new_words, new_labels
    return None


def cut_human_tail(words, labels, S, N):
    """For ai_then_human (1->0), cut everything after the seam: words[:S].
    Result is all-AI labels=1 → pure_ai sample.
    """
    new_words = words[:S]
    new_labels = labels[:S]
    if len(new_words) < MIN_LEN:
        return None
    if len(new_words) > MAX_LEN:
        new_words = new_words[:MAX_LEN]
        new_labels = new_labels[:MAX_LEN]
    return new_words, new_labels


def write_output(path: Path, rows: list, fieldnames: list):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    rng = random.Random(SEED)

    cc_a10_25 = []
    pile_h75_90 = []
    pile_a_late = []     # a25-50, a50-75 (used for cuts to early a)
    pile_a_pure = []     # a75-90, a90+ (used for pure_ai conversion)

    fieldnames = None
    print(f"Reading {INPUT}…")
    with open(INPUT, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            try:
                labels = json.loads(row.get("segmentation_labels", "[]"))
            except Exception:
                continue
            if num_seams(labels) != 1:
                continue
            words = (row.get("text") or "").split()
            if len(words) != len(labels):
                continue
            S = first_seam_idx(labels)
            N = len(words)
            pos = S / N
            src = row.get("data_source")
            st = row.get("sample_type")

            if src == "common_crawl" and st == "ai_then_human" and 0.10 <= pos < 0.25:
                cc_a10_25.append((row, words, labels, S, N))
            elif src == "pile":
                if st == "human_then_ai" and 0.75 <= pos < 0.90:
                    pile_h75_90.append((row, words, labels, S, N))
                elif st == "ai_then_human":
                    if 0.25 <= pos < 0.75:
                        pile_a_late.append((row, words, labels, S, N))
                    elif 0.75 <= pos < 0.90:
                        pile_a_pure.append((row, words, labels, S, N, "a75_90"))
                    elif pos >= 0.90:
                        pile_a_pure.append((row, words, labels, S, N, "a90_plus"))

    print(f"Pool sizes:")
    print(f"  CC a10-25:               {len(cc_a10_25):,}")
    print(f"  Pile h75-90:             {len(pile_h75_90):,}")
    print(f"  Pile a25-50 + a50-75:    {len(pile_a_late):,}")
    print(f"  Pile a75-90 + a90+:      {len(pile_a_pure):,}")

    # ---- 1. CC a10-25 → CC a0-10 ----
    print(f"\n[1] CC a10-25 → CC a0-10 (target {TARGET_CC_A0_10})")
    rng.shuffle(cc_a10_25)
    out1 = []
    for row, words, labels, S, N in cc_a10_25:
        if len(out1) >= TARGET_CC_A0_10:
            break
        result = cut_front_to_target_pos(words, labels, S, N, 0.0, 0.10, rng)
        if result is None:
            continue
        new_words, new_labels = result
        new_row = dict(row)
        new_row["text"] = " ".join(new_words)
        new_row["segmentation_labels"] = json.dumps(new_labels)
        new_row["n_words"] = str(len(new_words))
        new_row["augmented"] = "cut_to_a0_10"
        out1.append(new_row)
    write_output(DATA_DIR / "cuts_cc_a0_10.csv", out1, fieldnames)
    print(f"    Wrote {len(out1):,} rows to cuts_cc_a0_10.csv")

    # ---- 2. Pile h75-90 → Pile h50-75 ----
    print(f"\n[2] Pile h75-90 → Pile h50-75 (target {TARGET_PILE_H50_75})")
    rng.shuffle(pile_h75_90)
    out2 = []
    for row, words, labels, S, N in pile_h75_90:
        if len(out2) >= TARGET_PILE_H50_75:
            break
        result = cut_front_to_target_pos(words, labels, S, N, 0.50, 0.75, rng)
        if result is None:
            continue
        new_words, new_labels = result
        new_row = dict(row)
        new_row["text"] = " ".join(new_words)
        new_row["segmentation_labels"] = json.dumps(new_labels)
        new_row["n_words"] = str(len(new_words))
        new_row["augmented"] = "cut_to_h50_75"
        out2.append(new_row)
    write_output(DATA_DIR / "cuts_pile_h50_75.csv", out2, fieldnames)
    print(f"    Wrote {len(out2):,} rows to cuts_pile_h50_75.csv")

    # ---- 3. Pile a25-50/a50-75 → Pile a0-10 + a10-25 ----
    print(f"\n[3] Pile a25-50/a50-75 → Pile a0-10 + a10-25  "
          f"(targets {TARGET_PILE_A0_10} + {TARGET_PILE_A10_25})")
    rng.shuffle(pile_a_late)
    out_a0_10 = []
    out_a10_25 = []
    for row, words, labels, S, N in pile_a_late:
        if len(out_a0_10) >= TARGET_PILE_A0_10 and len(out_a10_25) >= TARGET_PILE_A10_25:
            break
        if len(out_a0_10) < TARGET_PILE_A0_10:
            result = cut_front_to_target_pos(words, labels, S, N, 0.0, 0.10, rng)
            if result is not None:
                new_words, new_labels = result
                new_row = dict(row)
                new_row["text"] = " ".join(new_words)
                new_row["segmentation_labels"] = json.dumps(new_labels)
                new_row["n_words"] = str(len(new_words))
                new_row["augmented"] = "cut_to_a0_10"
                out_a0_10.append(new_row)
                continue
        if len(out_a10_25) < TARGET_PILE_A10_25:
            result = cut_front_to_target_pos(words, labels, S, N, 0.10, 0.25, rng)
            if result is not None:
                new_words, new_labels = result
                new_row = dict(row)
                new_row["text"] = " ".join(new_words)
                new_row["segmentation_labels"] = json.dumps(new_labels)
                new_row["n_words"] = str(len(new_words))
                new_row["augmented"] = "cut_to_a10_25"
                out_a10_25.append(new_row)
    out3 = out_a0_10 + out_a10_25
    write_output(DATA_DIR / "cuts_pile_a_early.csv", out3, fieldnames)
    print(f"    Wrote {len(out_a0_10):,} a0-10 + {len(out_a10_25):,} a10-25 = {len(out3):,} rows")

    # ---- 4. Pile a75-90/a90+ → Pile pure_ai (cut human tail), per-pool cap ----
    print(f"\n[4] Pile a75-90/a90+ → Pile pure_ai  "
          f"(cap a75-90={CAP_PILE_A75_90_TO_PURE_AI}, a90+={CAP_PILE_A90_PLUS_TO_PURE_AI})")
    rng.shuffle(pile_a_pure)
    out4 = []
    used_a75_90 = 0
    used_a90_plus = 0
    for tup in pile_a_pure:
        row, words, labels, S, N, pool = tup
        if pool == "a75_90" and used_a75_90 >= CAP_PILE_A75_90_TO_PURE_AI:
            continue
        if pool == "a90_plus" and used_a90_plus >= CAP_PILE_A90_PLUS_TO_PURE_AI:
            continue
        result = cut_human_tail(words, labels, S, N)
        if result is None:
            continue
        new_words, new_labels = result
        new_row = dict(row)
        new_row["text"] = " ".join(new_words)
        new_row["segmentation_labels"] = json.dumps(new_labels)
        new_row["n_words"] = str(len(new_words))
        new_row["sample_type"] = "pure_ai"
        new_row["augmented"] = "cut_human_tail"
        out4.append(new_row)
        if pool == "a75_90":
            used_a75_90 += 1
        else:
            used_a90_plus += 1
    write_output(DATA_DIR / "cuts_pile_pure_ai_from_a.csv", out4, fieldnames)
    print(f"    Wrote {len(out4):,} rows to cuts_pile_pure_ai_from_a.csv  "
          f"(from a75-90: {used_a75_90:,}, from a90+: {used_a90_plus:,})")

    print(f"\n{'='*64}")
    print(f"GRAND TOTAL: {len(out1) + len(out2) + len(out3) + len(out4):,} new rows from cuts")
    print(f"  cuts_cc_a0_10.csv               {len(out1):>7,}")
    print(f"  cuts_pile_h50_75.csv            {len(out2):>7,}")
    print(f"  cuts_pile_a_early.csv           {len(out3):>7,}")
    print(f"  cuts_pile_pure_ai_from_a.csv    {len(out4):>7,}")


if __name__ == "__main__":
    main()
