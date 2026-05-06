"""
Multi-dimensional analysis of the final dataset (train_final, val_final, test_final).

Covers ten facets:
  1. Row counts + seam-count breakdown
  2. Word-count distribution (overall + by sample_type)
  3. Sample-type distribution
  4. Data-source mix (overall + cross-tab vs sample_type)
  5. Model-name distribution (which AI generators produced the AI content)
  6. Augmentation source (original / late_seam_synth / two_seam_split / long_row_split)
  7. Token-level class balance (fraction of words labeled 0 vs 1)
  8. Seam-position distribution per direction (0->1 and 1->0 separately)
  9. AI-portion length distribution (how much AI content per seam-containing row)
 10. Cross-file leakage check (rows shared between train/val/test by text hash)

Reads train_final.csv, val_final.csv, test_final.csv.
"""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

DATA_DIR = Path("data/Training_Dataset")
FILES = ["train_final.csv", "val_final.csv", "test_final.csv"]


def num_seams(labels):
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])


def first_seam_idx(labels):
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            return i
    return None


def percentile(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return s[max(0, min(len(s)-1, int(p*len(s))))]


def text_hash(t):
    return hashlib.md5((t or "").encode("utf-8")).hexdigest()


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["_labels"] = json.loads(r.get("segmentation_labels") or "[]")
            except Exception:
                continue
            r["_words"] = (r.get("text") or "").split()
            r["_n_words"] = len(r["_labels"])
            r["_seams"] = num_seams(r["_labels"])
            r["_first_seam"] = first_seam_idx(r["_labels"])
            r["_text_hash"] = text_hash(r.get("text", ""))
            rows.append(r)
    return rows


def header(s):
    print("\n" + "=" * 76)
    print(s)
    print("=" * 76)


def report_per_file(name, rows):
    header(f"FILE: {name}   total rows = {len(rows):,}")

    # ---- 1. Seam-count breakdown ----
    sc = Counter(r["_seams"] for r in rows)
    print("\n[1] Seam-count distribution")
    for s in sorted(sc.keys()):
        print(f"    seams={s}: {sc[s]:>8,}  ({100*sc[s]/len(rows):>5.1f}%)")

    # ---- 2. Word count overall + by sample_type ----
    wc = [r["_n_words"] for r in rows]
    print("\n[2] Word counts overall")
    print(f"    min={min(wc)}  p10={percentile(wc,0.1)}  p50={percentile(wc,0.5)}  "
          f"avg={sum(wc)/len(wc):.1f}  p90={percentile(wc,0.9)}  max={max(wc)}")
    print("\n    By sample_type:")
    by_st = defaultdict(list)
    for r in rows:
        by_st[r.get("sample_type", "?")].append(r["_n_words"])
    for st in sorted(by_st.keys()):
        v = by_st[st]
        print(f"      {st:<22s} n={len(v):>7,}  "
              f"avg={sum(v)/len(v):>5.1f}  p50={percentile(v,0.5):>4}  p90={percentile(v,0.9):>4}")

    # ---- 3. Sample-type distribution ----
    st_counts = Counter(r.get("sample_type", "?") for r in rows)
    print("\n[3] Sample-type distribution")
    for st in sorted(st_counts.keys()):
        c = st_counts[st]
        print(f"    {st:<22s} {c:>8,}  ({100*c/len(rows):>5.1f}%)")

    # ---- 4. Data-source mix overall + cross-tab vs sample_type ----
    ds_counts = Counter(r.get("data_source", "?") for r in rows)
    print("\n[4] Data-source mix")
    for ds in sorted(ds_counts.keys()):
        c = ds_counts[ds]
        print(f"    {ds:<22s} {c:>8,}  ({100*c/len(rows):>5.1f}%)")
    print("\n    Cross-tab (sample_type x data_source):")
    print(f"      {'sample_type':<22s} {'pile':>10s} {'cc':>10s}")
    cross = defaultdict(lambda: Counter())
    for r in rows:
        cross[r.get("sample_type", "?")][r.get("data_source", "?")] += 1
    for st in sorted(cross.keys()):
        c = cross[st]
        p = c.get("pile", 0)
        cc = c.get("common_crawl", 0)
        print(f"      {st:<22s} {p:>10,} {cc:>10,}")

    # ---- 5. Model-name distribution (AI rows only) ----
    print("\n[5] Model-name distribution (top 15)")
    mn = Counter()
    for r in rows:
        m = r.get("model_name") or "(none)"
        if m.lower() in ("none", "(none)", "") or r.get("sample_type") == "pure_human":
            continue
        mn[m] += 1
    for m, c in mn.most_common(15):
        print(f"    {m[:55]:<55s} {c:>8,}")
    if not mn:
        print("    (no model names recorded)")

    # ---- 6. Augmentation source ----
    aug = Counter()
    for r in rows:
        a = r.get("augmented") or "(none)"
        if a.lower() == "false":
            a = "original"
        elif a.lower() == "true":
            a = "augmented_unspecified"
        aug[a] += 1
    print("\n[6] Augmentation source")
    for a in sorted(aug.keys()):
        c = aug[a]
        print(f"    {a:<28s} {c:>8,}  ({100*c/len(rows):>5.1f}%)")

    # ---- 7. Token-level class balance ----
    n_zero = n_one = 0
    for r in rows:
        for x in r["_labels"]:
            if x == 0: n_zero += 1
            else: n_one += 1
    total_tokens = n_zero + n_one
    print("\n[7] Token-level class balance")
    if total_tokens:
        print(f"    label=0 (human): {n_zero:>10,}  ({100*n_zero/total_tokens:>5.1f}%)")
        print(f"    label=1 (AI):    {n_one:>10,}  ({100*n_one/total_tokens:>5.1f}%)")

    # ---- 8. Seam-position distribution per direction ----
    print("\n[8] Seam-position distribution (single-seam rows only)")
    pos_buckets = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    for direction_label, label0 in [("0->1 (human_then_ai)", 0),
                                     ("1->0 (ai_then_human)", 1)]:
        counts = [0]*(len(pos_buckets)-1)
        n_total = 0
        for r in rows:
            if r["_seams"] != 1: continue
            if r["_labels"][0] != label0: continue
            n_total += 1
            p = r["_first_seam"] / r["_n_words"]
            for i in range(len(pos_buckets)-1):
                if pos_buckets[i] <= p < pos_buckets[i+1]:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1
        print(f"    {direction_label}  (n={n_total:,})")
        for i in range(len(pos_buckets)-1):
            a, b = pos_buckets[i], pos_buckets[i+1]
            c = counts[i]
            pct = 100*c/max(1, n_total)
            bar = "#" * int(30 * c / max(1, n_total))
            print(f"      {a:.2f}-{b:.2f}: {c:>7,}  ({pct:>5.1f}%)  {bar}")

    # ---- 9. AI-portion length distribution (seam-containing rows) ----
    print("\n[9] AI-portion length within seam-containing rows")
    ai_lens = []
    for r in rows:
        if r["_seams"] not in (1, 2):
            continue
        ai_count = sum(1 for x in r["_labels"] if x == 1)
        if ai_count > 0:
            ai_lens.append(ai_count)
    if ai_lens:
        print(f"    n={len(ai_lens):,}  min={min(ai_lens)}  "
              f"p10={percentile(ai_lens,0.1)}  p50={percentile(ai_lens,0.5)}  "
              f"avg={sum(ai_lens)/len(ai_lens):.1f}  "
              f"p90={percentile(ai_lens,0.9)}  max={max(ai_lens)}")
    else:
        print("    (no seam-containing rows)")


def report_cross_file_leakage(rows_per_file):
    header("[10] CROSS-FILE LEAKAGE CHECK  (rows shared by exact-text match)")
    hashes = {name: set(r["_text_hash"] for r in rows) for name, rows in rows_per_file.items()}
    pairs = [
        ("train_final.csv", "val_final.csv"),
        ("train_final.csv", "test_final.csv"),
        ("val_final.csv",   "test_final.csv"),
    ]
    for a, b in pairs:
        if a not in hashes or b not in hashes:
            continue
        overlap = hashes[a] & hashes[b]
        print(f"  {a}  ∩  {b}:  {len(overlap):,} shared rows  "
              f"({100*len(overlap)/len(hashes[b]):.2f}% of {b})")


def main():
    rows_per_file = {}
    for name in FILES:
        path = DATA_DIR / name
        if not path.exists():
            print(f"SKIP {path} (not found)")
            continue
        rows = load(path)
        rows_per_file[name] = rows
        report_per_file(name, rows)
    report_cross_file_leakage(rows_per_file)


if __name__ == "__main__":
    main()
