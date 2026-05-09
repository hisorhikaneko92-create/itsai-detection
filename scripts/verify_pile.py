"""Tiered Pile retrieval verifier — checks captured validator texts against
L1 (exact 5-gram) and L2 (stem 4-gram) Bloom filters together.

The match rule for any text:

    word i is "Pile" if any 5-gram or stem-4-gram window starting at or
    spanning i hits its respective filter

This is the same logic the inference predictor will use. By running it on
your captured validator logs you preview exactly what reward Phase 1+2 will
deliver before deploying.

Expected pattern after a successful build:

    Pile-sourced human texts:   hit_ratio >= 0.85
    CC-sourced human texts:     hit_ratio <= 0.10
    Pure-AI texts:              hit_ratio <= 0.05
    Mixed Pile texts:           bimodal — half ~0.95, half ~0.05

L2 should narrow the gap on augmented Pile texts compared to L1 alone — if
you see the same numbers with and without --l2 args, augmentation isn't
hurting recall and L2 is optional.

Usage:
    # L1 only (4 shards)
    python scripts/verify_pile.py \\
        --l1 indexes/pile_l1_shard0.bloom \\
        --l1 indexes/pile_l1_shard1.bloom \\
        --l1 indexes/pile_l1_shard2.bloom \\
        --l1 indexes/pile_l1_shard3.bloom

    # L1 + L2 combined (full Phase-1+2 verify)
    python scripts/verify_pile.py \\
        --l1 indexes/pile_l1_shard{0,1,2,3}.bloom \\
        --l2 indexes/pile_l2_shard{0,1,2,3}.bloom
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

try:
    import snowballstemmer
    _SNOWBALL = snowballstemmer.stemmer("english")
    def _stem_many(words):
        return _SNOWBALL.stemWords(words)
except ImportError:
    from nltk.stem import PorterStemmer
    _PS = PorterStemmer()
    def _stem_many(words):
        return [_PS.stem(w) for w in words]


_RE_NONALPHANUM = re.compile(r"[^a-z0-9 ]")
_SIGN_MASK = 1 << 127
_TWO_128   = 1 << 128


def hash_func(obj) -> int:
    b = obj.encode("utf-8") if isinstance(obj, str) else obj
    h = xxhash.xxh3_128_intdigest(b)
    return h - _TWO_128 if h & _SIGN_MASK else h


def normalize(text: str) -> list[str]:
    return _RE_NONALPHANUM.sub(" ", text.lower()).split()


def normalize_stem(text: str) -> list[str]:
    return _stem_many(normalize(text))


def hit_ratio_l1(text: str, blooms: list[Bloom], n: int = 5) -> float:
    """Fraction of 5-gram windows that hit ANY of the given filters."""
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


def hit_ratio_l2(text: str, blooms: list[Bloom], n: int = 4) -> float:
    """Fraction of stem-4-gram windows that hit ANY of the given filters."""
    w = normalize_stem(text)
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
    p.add_argument("--l1", action="append", default=None, type=Path,
                   help="L1 (exact 5-gram) Bloom path. Repeat for multiple shards.")
    p.add_argument("--l2", action="append", default=None, type=Path,
                   help="L2 (stem 4-gram) Bloom path. Repeat for multiple shards. "
                        "Optional — omit to verify L1 only.")
    p.add_argument("--logs-glob", default="neurons/validator_logs/raw/*.json")
    p.add_argument("--limit", type=int, default=50)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.l1:
        print("ERROR: at least one --l1 filter is required", file=sys.stderr)
        return 1

    l1_blooms = []
    for bp in args.l1:
        if not bp.exists():
            print(f"ERROR: L1 filter not found at {bp}", file=sys.stderr)
            return 1
        print(f"Loading L1 from {bp} ...", file=sys.stderr)
        l1_blooms.append(Bloom.load(str(bp), hash_func))

    l2_blooms = []
    if args.l2:
        for bp in args.l2:
            if not bp.exists():
                print(f"ERROR: L2 filter not found at {bp}", file=sys.stderr)
                return 1
            print(f"Loading L2 from {bp} ...", file=sys.stderr)
            l2_blooms.append(Bloom.load(str(bp), hash_func))

    print(f"Loaded {len(l1_blooms)} L1 filter(s)"
          + (f" + {len(l2_blooms)} L2 filter(s)" if l2_blooms else "")
          + ".", file=sys.stderr)

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
            hr_l1 = hit_ratio_l1(text, l1_blooms)
            hr_l2 = hit_ratio_l2(text, l2_blooms) if l2_blooms else 0.0
            hr_combined = max(hr_l1, hr_l2)
            rows.append({
                "ts":     d.get("timestamp_utc", "")[:19],
                "hk":     d.get("validator_hotkey", "")[:10],
                "hash":   t.get("hash"),
                "n_words": len(text.split()),
                "hr_l1":  hr_l1,
                "hr_l2":  hr_l2,
                "hr":     hr_combined,
            })

    if not rows:
        print("No texts found in captured logs.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["hr"], reverse=True)
    if l2_blooms:
        print(f"{'ts':<19}  {'hk':<10}  {'hash':<8}  {'words':>5}  "
              f"{'L1':>6}  {'L2':>6}  {'best':>6}")
        print("-" * 78)
        for r in rows:
            print(f"{r['ts']:<19}  {r['hk']:<10}  {r['hash']:<8}  {r['n_words']:>5d}  "
                  f"{r['hr_l1']:>6.3f}  {r['hr_l2']:>6.3f}  {r['hr']:>6.3f}")
    else:
        print(f"{'ts':<19}  {'hk':<10}  {'hash':<8}  {'words':>5}  {'hit_ratio':>9}")
        print("-" * 65)
        for r in rows:
            print(f"{r['ts']:<19}  {r['hk']:<10}  {r['hash']:<8}  "
                  f"{r['n_words']:>5d}  {r['hr_l1']:>9.3f}")

    ratios = [r["hr"] for r in rows]
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

    if l2_blooms:
        l2_lift = sum(1 for r in rows if r["hr_l2"] > r["hr_l1"] + 0.05)
        print(f"L2 added 5+pp recall on {l2_lift}/{len(rows)} texts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
