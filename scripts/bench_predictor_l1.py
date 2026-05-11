"""Benchmark + sanity-check the L1 predictor on captured validator texts.

Loads the 4 L1 shard Bloom filters, runs the predictor against each captured
text, and reports:

  * Per-text Pile fraction (= fraction of words labeled 0.0 vs 0.99)
  * Per-text mean prediction (mirror of the validator's avg_prediction)
  * Distribution by bucket — high Pile-density / mixed / low
  * Predictor latency for a realistic 120-text batch

We have NO ground-truth labels miner-side (validator never reveals them), but
we can compare to the OLD HSSD predictions in the captured logs to see where
the new L1 predictor diverges.

Usage:
    python scripts/bench_predictor_l1.py
    python scripts/bench_predictor_l1.py --min-run 2 --limit 200
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from pathlib import Path

# Allow importing predictor_l1 from same dir
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor_l1 import L1Predictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bloom", action="append", default=None, type=Path,
                   help="L1 Bloom shard path. Default: indexes/pile_l1_shard{0,1,2,3}.bloom")
    p.add_argument("--logs-glob", default="neurons/validator_logs/raw/*.json")
    p.add_argument("--limit", type=int, default=200,
                   help="Cap the number of texts (after dedup) to analyze")
    p.add_argument("--min-run", type=int, default=3,
                   help="Run-length smoother — min consecutive matched windows")
    p.add_argument("--bench-batch", type=int, default=120,
                   help="Batch size for the timing benchmark")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.bloom is None:
        args.bloom = [Path(f"indexes/pile_l1_shard{i}.bloom") for i in range(4)]
    for b in args.bloom:
        if not b.exists():
            print(f"ERROR: missing {b}", file=sys.stderr)
            return 1

    print(f"Loading {len(args.bloom)} L1 filter(s) — this can take a moment...",
          file=sys.stderr)
    t0 = time.time()
    pred = L1Predictor(args.bloom, min_run=args.min_run)
    print(f"  loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    # Collect unique texts from captured logs (dedup by hash)
    seen: set[str] = set()
    rows: list[dict] = []
    for fp in sorted(glob.glob(args.logs_glob)):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        for t in d.get("texts", []) or []:
            h = t.get("hash") or ""
            text = t.get("full_text") or ""
            if not text or h in seen:
                continue
            seen.add(h)
            rows.append({
                "ts":      d.get("timestamp_utc", "")[:19],
                "hk":      d.get("validator_hotkey", "")[:10],
                "hash":    h,
                "text":    text,
                "old_avg": t.get("avg_prediction"),
            })
            if len(rows) >= args.limit:
                break
        if len(rows) >= args.limit:
            break

    if not rows:
        print("No texts found.", file=sys.stderr)
        return 1
    print(f"  collected {len(rows)} unique texts", file=sys.stderr)

    # Warm up (first call pages filters into RAM via mmap)
    pred.predict(rows[0]["text"])

    # Run predictor on every text
    print(f"\nRunning L1 predictor on {len(rows)} texts ...", file=sys.stderr)
    t0 = time.time()
    for r in rows:
        preds = pred.predict(r["text"])
        n = len(preds)
        n_pile = sum(1 for p in preds if p < 0.5)
        r["n_words"] = n
        r["pile_frac"] = n_pile / max(n, 1)
        r["new_avg"]  = sum(preds) / max(n, 1)
    total_time = time.time() - t0
    print(f"  done in {total_time:.2f}s "
          f"({len(rows) / total_time:.0f} texts/sec)",
          file=sys.stderr)

    # Realistic 120-text batch latency
    bench_n = min(args.bench_batch, len(rows))
    print(f"\n--- benchmark: {bench_n}-text batch (validator-style) ---")
    t0 = time.time()
    for r in rows[:bench_n]:
        _ = pred.predict(r["text"])
    bench_time = time.time() - t0
    print(f"  {bench_n} texts in {bench_time*1000:.0f} ms")
    print(f"  {bench_time*1000/bench_n:.1f} ms/text")
    print(f"  Validator timeout: ~18s → margin: {18 / bench_time:.0f}× under")

    # Sorted distribution
    rows.sort(key=lambda r: r["pile_frac"], reverse=True)

    # Sample at quartiles
    print(f"\n--- sample at quartiles (sorted desc by pile_frac) ---")
    print(f"{'pile_frac':>9} {'new_avg':>7} {'old_avg':>7} {'n_words':>7}  hash")
    n = len(rows)
    for i in (0, n//4, n//2, 3*n//4, n-1):
        r = rows[i]
        old = f"{r['old_avg']:.3f}" if r['old_avg'] is not None else "  N/A"
        print(f"{r['pile_frac']:>9.3f} {r['new_avg']:>7.3f} {old:>7} "
              f"{r['n_words']:>7d}  {r['hash']}")

    # Aggregate buckets
    pf = [r["pile_frac"] for r in rows]
    print(f"\n--- pile_frac summary ({len(pf)} texts) ---")
    print(f"  min:    {min(pf):.3f}")
    print(f"  median: {statistics.median(pf):.3f}")
    print(f"  mean:   {statistics.mean(pf):.3f}")
    print(f"  max:    {max(pf):.3f}")
    print(f"  >= 0.85:   {sum(1 for x in pf if x >= 0.85):>3d}  Pile-confirmed")
    print(f"  0.10-0.85: {sum(1 for x in pf if 0.10 <= x < 0.85):>3d}  mixed")
    print(f"  <  0.10:   {sum(1 for x in pf if x < 0.10):>3d}  non-Pile")

    # If old_avg available, show correlation
    paired = [(r["new_avg"], r["old_avg"]) for r in rows if r["old_avg"] is not None]
    if paired:
        # Simple Pearson corr
        n = len(paired)
        mx = sum(p[0] for p in paired) / n
        my = sum(p[1] for p in paired) / n
        num = sum((p[0]-mx)*(p[1]-my) for p in paired)
        denom = (sum((p[0]-mx)**2 for p in paired) * sum((p[1]-my)**2 for p in paired)) ** 0.5
        r_pearson = num / denom if denom > 0 else 0.0
        print(f"\n--- correlation with old HSSD predictions ---")
        print(f"  paired:  {n}")
        print(f"  Pearson r:  {r_pearson:+.3f}")
        print(f"  (positive = both predictors agree on which texts are AI vs human)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
