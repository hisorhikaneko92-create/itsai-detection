"""
Audit duplication WITHIN each CSV and ACROSS multiple CSVs (cross-split
contamination check).

Usage:
    # Single file
    python scripts/audit_contamination.py --csv data/train.csv

    # Multiple files (the most useful mode — detects train->val/test leaks)
    python scripts/audit_contamination.py \\
        --csv data/Training_Dataset/train.csv \\
        --csv data/Training_Dataset/val.csv \\
        --csv data/Training_Dataset/test.csv
"""
import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path


def text_fingerprint(text: str, n: int = 200) -> str:
    """Stable identifier for the head of a text — a proxy for the
    root document the row was generated from."""
    return hashlib.md5(text[:n].encode("utf-8", errors="replace")).hexdigest()


def scan_csv(path: Path, prefix_chars: int):
    """Returns (rows_meta, distribution) for one CSV.

    rows_meta:    list of dicts {fp_head, fp_full, sample_type, model_name,
                                 data_source, n_words}
    distribution: dict with counts by sample_type, data_source, model_name
    """
    rows = []
    dist = {
        "sample_type": Counter(),
        "data_source": Counter(),
        "model_name":  Counter(),
        "n_words_buckets": Counter(),
        "exact_dupes_in_file": 0,
    }
    seen_full = set()

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            fp_head = text_fingerprint(text, prefix_chars)
            fp_full = text_fingerprint(text, max(len(text), prefix_chars))

            if fp_full in seen_full:
                dist["exact_dupes_in_file"] += 1
            else:
                seen_full.add(fp_full)

            sample_type  = row.get("sample_type", "?") or "?"
            data_source  = row.get("data_source", "?") or "?"
            model_name   = row.get("model_name",  "?") or "?"
            try:
                n_words = int(row.get("n_words", 0))
            except (TypeError, ValueError):
                n_words = len(text.split())

            if n_words < 35:        bucket = "<35"
            elif n_words <= 100:    bucket = "35-100"
            elif n_words <= 200:    bucket = "101-200"
            elif n_words <= 350:    bucket = "201-350"
            else:                   bucket = ">350"

            dist["sample_type"][sample_type] += 1
            dist["data_source"][data_source] += 1
            dist["model_name"][model_name]   += 1
            dist["n_words_buckets"][bucket]  += 1

            rows.append({
                "fp_head":     fp_head,
                "fp_full":     fp_full,
                "sample_type": sample_type,
                "model_name":  model_name,
                "data_source": data_source,
                "n_words":     n_words,
            })

    return rows, dist


def print_distribution(name: str, dist: dict, total: int) -> None:
    print(f"\n--- {name}  ({total:,} rows, {dist['exact_dupes_in_file']:,} exact dupes within file) ---")

    print(" sample_type:")
    for k, c in dist["sample_type"].most_common():
        print(f"   {k:<20s} {c:>7,}  ({100*c/total:.1f}%)")

    print(" data_source:")
    for k, c in dist["data_source"].most_common():
        print(f"   {k:<20s} {c:>7,}  ({100*c/total:.1f}%)")

    print(" word-count buckets:")
    bucket_order = ["<35", "35-100", "101-200", "201-350", ">350"]
    for k in bucket_order:
        c = dist["n_words_buckets"].get(k, 0)
        if c:
            print(f"   {k:<10s} {c:>10,}  ({100*c/total:.1f}%)")

    print(" model_name (top 12 by count):")
    for k, c in dist["model_name"].most_common(12):
        print(f"   {k:<45s} {c:>7,}  ({100*c/total:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", action="append", required=True,
                    help="CSV path. Pass multiple --csv flags for cross-split audit.")
    ap.add_argument("--prefix-chars", type=int, default=200,
                    help="Leading chars defining 'same root doc'. Default 200.")
    args = ap.parse_args()

    paths = [Path(p) for p in args.csv]
    for p in paths:
        if not p.exists():
            sys.exit(f"Not found: {p}")

    # Per-file scan
    per_file_rows = {}
    per_file_dist = {}
    for p in paths:
        print(f"Scanning {p} ...")
        rows, dist = scan_csv(p, args.prefix_chars)
        per_file_rows[p.name] = rows
        per_file_dist[p.name] = dist

    # ------------------------------------------------------------------
    # Per-file distributions
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" Per-file distributions")
    print("=" * 70)
    for p in paths:
        rows  = per_file_rows[p.name]
        dist  = per_file_dist[p.name]
        print_distribution(p.name, dist, len(rows))

    # ------------------------------------------------------------------
    # Cross-file leakage (the contamination check the user asked about)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" Cross-file leakage (same root document appearing in multiple files)")
    print("=" * 70)

    # fp_head -> set of file names it appears in
    fp_to_files = defaultdict(set)
    fp_to_full_in_files = defaultdict(lambda: defaultdict(int))
    for fname, rows in per_file_rows.items():
        for r in rows:
            fp_to_files[r["fp_head"]].add(fname)
            fp_to_full_in_files[r["fp_full"]][fname] += 1

    # Pairwise leak counts
    file_pairs = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            file_pairs.append((paths[i].name, paths[j].name))

    # Head-prefix leaks (same root doc)
    head_leak_pairs = Counter()
    for fp, files in fp_to_files.items():
        if len(files) > 1:
            files_sorted = sorted(files)
            for i in range(len(files_sorted)):
                for j in range(i + 1, len(files_sorted)):
                    head_leak_pairs[(files_sorted[i], files_sorted[j])] += 1

    # Full-text leaks (exact duplicates across files)
    full_leak_pairs = Counter()
    for fp, files_count in fp_to_full_in_files.items():
        files_present = list(files_count.keys())
        if len(files_present) > 1:
            files_sorted = sorted(files_present)
            for i in range(len(files_sorted)):
                for j in range(i + 1, len(files_sorted)):
                    full_leak_pairs[(files_sorted[i], files_sorted[j])] += 1

    print(f"\nLeak counts by file pair  (root-doc-level prefix match, {args.prefix_chars} chars):")
    for pair in file_pairs:
        a, b = pair
        c = head_leak_pairs.get(pair, 0) + head_leak_pairs.get((b, a), 0)
        # Compute as % of the SMALLER file (the test/val pair)
        size_a = len(per_file_rows[a])
        size_b = len(per_file_rows[b])
        denom = min(size_a, size_b)
        pct = 100 * c / max(1, denom)
        print(f"  {a:<35s} <-> {b:<35s}  {c:>6,}  ({pct:.2f}% of smaller)")

    print(f"\nLeak counts by file pair  (EXACT full-text match):")
    for pair in file_pairs:
        a, b = pair
        c = full_leak_pairs.get(pair, 0) + full_leak_pairs.get((b, a), 0)
        size_a = len(per_file_rows[a])
        size_b = len(per_file_rows[b])
        denom = min(size_a, size_b)
        pct = 100 * c / max(1, denom)
        print(f"  {a:<35s} <-> {b:<35s}  {c:>6,}  ({pct:.2f}% of smaller)")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" Verdict")
    print("=" * 70)
    train_name = next((p.name for p in paths if "train" in p.name.lower()), None)
    test_name  = next((p.name for p in paths if "test"  in p.name.lower()), None)
    val_name   = next((p.name for p in paths if "val"   in p.name.lower()), None)

    def pair_pct(a, b, leak_counter):
        c = leak_counter.get((a, b), 0) + leak_counter.get((b, a), 0)
        denom = min(len(per_file_rows[a]), len(per_file_rows[b]))
        return c, 100 * c / max(1, denom)

    if train_name and test_name:
        c_head, p_head = pair_pct(train_name, test_name, head_leak_pairs)
        c_full, p_full = pair_pct(train_name, test_name, full_leak_pairs)
        print(f"\n train <-> test:")
        print(f"   root-doc leaks: {c_head:>6,}  ({p_head:.2f}% of test)")
        print(f"   exact   leaks:  {c_full:>6,}  ({p_full:.2f}% of test)")
        if p_full > 5.0:
            print("   STATUS: ❌ HIGH leak rate. Test metrics are partly memorization.")
        elif p_head > 5.0:
            print("   STATUS: ⚠️ Moderate root-doc overlap. Test metrics slightly inflated.")
        else:
            print("   STATUS: ✅ Acceptable. Test metrics are trustworthy.")

    if train_name and val_name:
        c_head, p_head = pair_pct(train_name, val_name, head_leak_pairs)
        c_full, p_full = pair_pct(train_name, val_name, full_leak_pairs)
        print(f"\n train <-> val:")
        print(f"   root-doc leaks: {c_head:>6,}  ({p_head:.2f}% of val)")
        print(f"   exact   leaks:  {c_full:>6,}  ({p_full:.2f}% of val)")
        if p_full > 5.0:
            print("   STATUS: ❌ HIGH leak rate. Val metrics are partly memorization.")
        elif p_head > 5.0:
            print("   STATUS: ⚠️ Moderate root-doc overlap. Val metrics slightly inflated.")
        else:
            print("   STATUS: ✅ Acceptable. Val metrics are trustworthy.")


if __name__ == "__main__":
    main()
