"""
Merge dataset files for resume training from step_12000.

Steps:
1. train.csv + adv_train.csv -> train_final.csv  (initial merge)
   test.csv  + adv_test.csv  -> test_final.csv
   val.csv   + adv_val.csv   -> val_final.csv

2. Drop sample_type == 'multi_seam' from test_final.csv and val_final.csv
   (validator never sends multi-seam docs in production)

3. Append to train_final.csv:
     adv_ai_in_middle.csv + adv_human_then_ai.csv + adv_human_then_ai_0.csv

Originals are NOT modified. Outputs go to *_final.csv next to the inputs.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

# Some text fields exceed the default 128KB csv limit
csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

DATA_DIR = Path("data/Training_Dataset")


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv(path: Path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def report(label: str, rows):
    counts = Counter(r.get("sample_type", "?") for r in rows)
    print(f"\n{label}: {len(rows):,} rows")
    for st, n in counts.most_common():
        print(f"    {st:<20s} {n:>8,}")


def main():
    if not DATA_DIR.exists():
        sys.exit(f"Data dir not found: {DATA_DIR.resolve()}")

    # ---------- STEP 1: pairwise merges -----------------------------
    print("=" * 64)
    print("STEP 1  pairwise merges (X.csv + adv_X.csv)")
    print("=" * 64)

    train_rows, fields = read_csv(DATA_DIR / "train.csv")
    adv_train_rows, _  = read_csv(DATA_DIR / "adv_train.csv")
    report("train.csv",     train_rows)
    report("adv_train.csv", adv_train_rows)
    train_merged = train_rows + adv_train_rows
    report("=> train (after pairwise merge)", train_merged)

    test_rows, _     = read_csv(DATA_DIR / "test.csv")
    adv_test_rows, _ = read_csv(DATA_DIR / "adv_test.csv")
    report("test.csv",     test_rows)
    report("adv_test.csv", adv_test_rows)
    test_merged = test_rows + adv_test_rows
    report("=> test (after pairwise merge)", test_merged)

    val_rows, _     = read_csv(DATA_DIR / "val.csv")
    adv_val_rows, _ = read_csv(DATA_DIR / "adv_val.csv")
    report("val.csv",     val_rows)
    report("adv_val.csv", adv_val_rows)
    val_merged = val_rows + adv_val_rows
    report("=> val (after pairwise merge)", val_merged)

    # ---------- STEP 2: drop multi_seam from test/val ---------------
    print("\n" + "=" * 64)
    print("STEP 2  drop sample_type=='multi_seam' from test + val")
    print("=" * 64)

    test_clean = [r for r in test_merged if r.get("sample_type") != "multi_seam"]
    val_clean  = [r for r in val_merged  if r.get("sample_type") != "multi_seam"]

    print(f"\n  test:  {len(test_merged):,} -> {len(test_clean):,}  "
          f"(removed {len(test_merged) - len(test_clean):,} multi_seam rows)")
    print(f"  val:   {len(val_merged):,} -> {len(val_clean):,}  "
          f"(removed {len(val_merged)  - len(val_clean):,} multi_seam rows)")

    report("=> test_final (multi_seam dropped)", test_clean)
    report("=> val_final  (multi_seam dropped)", val_clean)

    # ---------- STEP 3: append more training files -----------------
    print("\n" + "=" * 64)
    print("STEP 3  append adv_ai_in_middle + adv_human_then_ai + adv_human_then_ai_0 to train")
    print("=" * 64)

    extras = []
    for name in ["adv_ai_in_middle.csv",
                 "adv_human_then_ai.csv",
                 "adv_human_then_ai_0.csv"]:
        rows, _ = read_csv(DATA_DIR / name)
        report(name, rows)
        extras.extend(rows)

    train_final = train_merged + extras
    report("=> train_final (after appending extras)", train_final)

    # ---------- Write outputs ---------------------------------------
    print("\n" + "=" * 64)
    print("Writing outputs")
    print("=" * 64)

    out_train = DATA_DIR / "train_final.csv"
    out_test  = DATA_DIR / "test_final.csv"
    out_val   = DATA_DIR / "val_final.csv"

    write_csv(out_train, train_final, fields)
    write_csv(out_test,  test_clean,  fields)
    write_csv(out_val,   val_clean,   fields)

    print(f"  {str(out_train):<55s} {len(train_final):>8,} rows")
    print(f"  {str(out_test):<55s} {len(test_clean):>8,} rows")
    print(f"  {str(out_val):<55s} {len(val_clean):>8,} rows")

    print("\nDone. Originals untouched. Use the *_final.csv files for the resume run.")


if __name__ == "__main__":
    main()
