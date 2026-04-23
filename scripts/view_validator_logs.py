"""
Convenience viewer for captured validator requests.

Usage:
  python scripts/view_validator_logs.py             # overview stats
  python scripts/view_validator_logs.py --tail 20   # show last 20 summary lines
  python scripts/view_validator_logs.py --show 5    # full detail for last 5 requests
  python scripts/view_validator_logs.py --hotkey 5F # filter by validator hotkey prefix
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neurons", "validator_logs")
RAW_DIR = os.path.join(LOG_DIR, "raw")
SUMMARY = os.path.join(LOG_DIR, "summary.log")


def load_all():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    out = []
    for fp in files:
        try:
            with open(fp) as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out


def overview(records):
    if not records:
        print("No validator requests captured yet.")
        print(f"Watching directory: {RAW_DIR}")
        return

    n = len(records)
    by_hotkey = Counter(r["validator_hotkey"][:12] for r in records)
    by_version = Counter(r["validator_version"] for r in records)
    version_probes = sum(1 for r in records if not r["version_in_range"])
    dup_batches = sum(1 for r in records if r["duplicates"])
    total_texts = sum(r["num_texts"] for r in records)
    word_counts = [t["n_words"] for r in records for t in r["texts"]]
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms")]
    avg_preds = [t["avg_prediction"] for r in records for t in r["texts"]
                 if t.get("avg_prediction") is not None]

    print(f"=== Validator Request Stats ===")
    print(f"Total requests:           {n}")
    print(f"Total texts seen:         {total_texts}")
    print(f"Unique validators:        {len(by_hotkey)}")
    print(f"Version probes received:  {version_probes}  ({version_probes*100//max(1,n)}%)")
    print(f"Batches with duplicates:  {dup_batches}     ({dup_batches*100//max(1,n)}%)")
    print()
    print(f"Words per text  -- min: {min(word_counts) if word_counts else 0}, "
          f"avg: {sum(word_counts)//max(1,len(word_counts))}, "
          f"max: {max(word_counts) if word_counts else 0}")
    print(f"Latency (ms)    -- min: {min(latencies) if latencies else 0}, "
          f"avg: {sum(latencies)//max(1,len(latencies))}, "
          f"max: {max(latencies) if latencies else 0}")
    if avg_preds:
        print(f"Your avg pred   -- min: {min(avg_preds):.3f}, "
              f"avg: {sum(avg_preds)/len(avg_preds):.3f}, "
              f"max: {max(avg_preds):.3f}")
    print()
    print("Top validators by request count:")
    for hk, c in by_hotkey.most_common(10):
        print(f"  {hk}...  {c}")
    print()
    print("Versions sent:")
    for v, c in by_version.most_common():
        print(f"  {v}  ({c})")


def show_detail(records, n):
    for r in records[-n:]:
        print("=" * 80)
        print(f"Time:      {r['timestamp_utc']}")
        print(f"Validator: {r['validator_hotkey']}")
        print(f"Version:   {r['validator_version']}  (in_range={r['version_in_range']})")
        print(f"Texts:     {r['num_texts']}    Latency: {r['latency_ms']}ms")
        if r["duplicates"]:
            print(f"Duplicates (idx pairs): {r['duplicates']}")
        for t in r["texts"]:
            tag = "AI " if (t.get("avg_prediction") or 0) > 0.5 else "HUM"
            pred = t.get("avg_prediction")
            pred_s = f"{pred:.3f}" if pred is not None else " - "
            print(f"  [{t['idx']:>2}] {tag} pred={pred_s} words={t['n_words']:>3} hash={t['hash']}")
            print(f"       HEAD: {t['preview_head'][:120]}")
            print(f"       TAIL: ...{t['preview_tail'][-120:]}")


def tail_summary(n):
    if not os.path.exists(SUMMARY):
        print("No summary.log yet.")
        return
    with open(SUMMARY) as f:
        lines = f.readlines()
    for ln in lines[-n:]:
        print(ln.rstrip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, help="Show last N summary lines")
    ap.add_argument("--show", type=int, help="Show full detail of last N requests")
    ap.add_argument("--hotkey", type=str, help="Filter by validator hotkey prefix")
    args = ap.parse_args()

    if args.tail:
        tail_summary(args.tail)
        return

    records = load_all()
    if args.hotkey:
        records = [r for r in records if r["validator_hotkey"].startswith(args.hotkey)]

    if args.show:
        show_detail(records, args.show)
    else:
        overview(records)


if __name__ == "__main__":
    main()
