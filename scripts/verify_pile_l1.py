"""Verify the L1 Pile 5-gram Bloom filter against captured validator texts.

Reads the JSONs produced by neurons/miner.py, hashes 5-grams of each captured
text, and reports how many match the index. Expected pattern after a successful
build:

    Pile-sourced human texts:   hit_ratio >= 0.85   (most 5-grams in index)
    CC-sourced human texts:     hit_ratio <= 0.10
    Pure-AI texts:              hit_ratio <= 0.05

If everything looks like 0.05 or everything looks like 0.95, the index is
under-built or the FPR is wrong.

Multi-filter mode (parallel build): pass multiple --bloom args. A 5-gram is
considered matched if it's in ANY of the filters (OR semantics). This lets
you verify the parallel-built shards directly without first merging them.

Usage:
    # Single filter
    python scripts/verify_pile_l1.py --bloom indexes/pile_l1_5gram.bloom --limit 50

    # Parallel-build verification (4 shards)
    python scripts/verify_pile_l1.py \\
        --bloom indexes/pile_l1_shard0.bloom \\
        --bloom indexes/pile_l1_shard1.bloom \\
        --bloom indexes/pile_l1_shard2.bloom \\
        --bloom indexes/pile_l1_shard3.bloom \\
        --limit 50
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

import xxhash
from rbloom import Bloom


_RE_NONALPHANUM = re.compile(r"[^a-z0-9 ]")
_SIGN_MASK = 1 << 127
_TWO_128   = 1 << 128


def hash_func(obj) -> int:
    """Must match scripts/build_pile_l1.py.hash_func bit-for-bit."""
    b = obj.encode("utf-8") if isinstance(obj, str) else obj
    h = xxhash.xxh3_128_intdigest(b)
    return h - _TWO_128 if h & _SIGN_MASK else h


def normalize(text: str) -> list[str]:
    return _RE_NONALPHANUM.sub(" ", text.lower()).split()


def hit_ratio(text: str, blooms: list[Bloom], n: int = 5) -> float:
    """Fraction of 5-grams that hit ANY of the given filters (OR semantics).

    Lets you query parallel-build shards without a merge step.
    """
    w = normalize(text)
    if len(w) < n:
        return 0.0
    total = len(w) - n + 1
    matched = 0
    for i in range(total):
        gram = " ".join(w[i:i + n])
        for bf in blooms:
            if gram in bf:
                matched += 1
                break
    return matched / total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bloom", type=Path, action="append", default=None,
                   help="Bloom filter path. Repeat for multi-filter (parallel) "
                        "verification. Default: indexes/pile_l1_5gram.bloom")
    p.add_argument("--logs-glob", default="neurons/validator_logs/raw/*.json")
    p.add_argument("--limit", type=int, default=50,
                   help="Look at the most recent N capture files")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.bloom is None:
        args.bloom = [Path("indexes/pile_l1_5gram.bloom")]

    blooms = []
    for bp in args.bloom:
        if not bp.exists():
            print(f"ERROR: Bloom filter not found at {bp}", file=sys.stderr)
            return 1
        print(f"Loading Bloom filter from {bp} ...", file=sys.stderr)
        blooms.append(Bloom.load(str(bp), hash_func))
    print(f"Loaded {len(blooms)} filter(s); querying with OR semantics.",
          file=sys.stderr)

    files = sorted(glob.glob(args.logs_glob))[-args.limit:]
    if not files:
        print(f"ERROR: no captured logs at {args.logs_glob}", file=sys.stderr)
        return 1

    rows = []
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        for t in d.get("texts", []) or []:
            text = t.get("full_text") or ""
            if not text:
                continue
            rows.append({
                "ts":     d.get("timestamp_utc", "")[:19],
                "hk":     d.get("validator_hotkey", "")[:10],
                "hash":   t.get("hash"),
                "n_words": len(text.split()),
                "hit_ratio": hit_ratio(text, blooms),
            })

    if not rows:
        print("No texts found in captured logs.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["hit_ratio"], reverse=True)
    print(f"{'ts':<19}  {'hk':<10}  {'hash':<8}  {'words':>5}  {'hit_ratio':>9}")
    print("-" * 65)
    for r in rows:
        print(f"{r['ts']:<19}  {r['hk']:<10}  {r['hash']:<8}  "
              f"{r['n_words']:>5d}  {r['hit_ratio']:>9.3f}")

    ratios = [r["hit_ratio"] for r in rows]
    print()
    print("=== summary ===")
    print(f"texts:        {len(ratios)}")
    print(f"min:          {min(ratios):.3f}")
    print(f"median:       {statistics.median(ratios):.3f}")
    print(f"mean:         {statistics.mean(ratios):.3f}")
    print(f"max:          {max(ratios):.3f}")
    print(f">= 0.85:      {sum(1 for r in ratios if r >= 0.85):>3d}  (likely Pile)")
    print(f"0.10 - 0.85:  {sum(1 for r in ratios if 0.10 < r < 0.85):>3d}  (mixed/ambiguous)")
    print(f"<= 0.10:      {sum(1 for r in ratios if r <= 0.10):>3d}  (CC or pure-AI)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
