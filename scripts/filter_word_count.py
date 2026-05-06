"""
Drop rows from train_final.csv whose word count falls outside [35, 350]
(the validator's subsample_words bounds). In place via tmp + atomic rename.
"""
import csv
import os
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

TARGET = Path("data/Training_Dataset/train_final.csv")
MIN_WORDS = 35
MAX_WORDS = 350


def main():
    if not TARGET.exists():
        sys.exit(f"Not found: {TARGET}")
    tmp = TARGET.with_suffix(TARGET.suffix + ".tmp")
    n_kept = n_too_short = n_too_long = 0
    with open(TARGET, "r", encoding="utf-8", newline="") as fin, \
         open(tmp, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        fields = list(reader.fieldnames or [])
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            n = len((row.get("text") or "").split())
            if n < MIN_WORDS:
                n_too_short += 1
                continue
            if n > MAX_WORDS:
                n_too_long += 1
                continue
            writer.writerow({k: row.get(k, "") for k in fields})
            n_kept += 1
    os.replace(tmp, TARGET)
    print(f"kept     {n_kept:>8,}")
    print(f"too short (<{MIN_WORDS}):  {n_too_short:>5,}")
    print(f"too long (>{MAX_WORDS}):  {n_too_long:>5,}")
    print(f"total dropped:    {n_too_short + n_too_long:>5,}")


if __name__ == "__main__":
    main()
