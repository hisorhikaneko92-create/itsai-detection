"""Build the L2 Pile stem-4gram Bloom filter index.

Pairs with L1 (exact 5-gram) to add recall on augmented Pile content. While L1
breaks on any single character perturbation in a 5-word window, L2 normalizes
each word to its English stem and matches 4-grams of stems — so a 5-word
window with one corrupted word may still produce a matching stem-4-gram.

Run AFTER scripts/build_pile_l1.py for the same Pile shards. Architecturally
identical (same hashing, same Bloom layout, same parallel sharding); the only
differences are the normalizer (Porter stemming on top of L1's normalization)
and gram size (4 instead of 5).

Resumable, signal-aware, supports local-shards mode for the fast path.

Usage (production, 4 parallel workers, after L1 finishes):
    nohup python scripts/build_pile_l2.py \\
        --local-shards data/pile-uncopyrighted/train/00.jsonl.zst \\
        --local-shards data/pile-uncopyrighted/train/04.jsonl.zst \\
        ... (matching L1's worker 0 shard list) ... \\
        --expected 6_000_000_000 --fpr 1e-6 \\
        --out indexes/pile_l2_shard0.bloom \\
        > logs/pile_l2_shard0.log 2>&1 &

Test mode (single shard, 50k docs, ~3 min):
    python scripts/build_pile_l2.py \\
        --local-shards data/pile-uncopyrighted/train/00.jsonl.zst \\
        --max-docs 50_000 \\
        --expected 100_000_000 \\
        --out indexes/pile_l2_test.bloom
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Iterator

import xxhash
import zstandard as zstd
from datasets import load_dataset
from rbloom import Bloom
from tqdm import tqdm

# Faster than nltk's pure-Python PorterStemmer (C extension under the hood).
# Falls back to nltk if snowballstemmer isn't installed.
try:
    import snowballstemmer
    _SNOWBALL = snowballstemmer.stemmer("english")
    def _stem_many(words: list[str]) -> list[str]:
        return _SNOWBALL.stemWords(words)
except ImportError:
    from nltk.stem import PorterStemmer
    _PS = PorterStemmer()
    def _stem_many(words: list[str]) -> list[str]:
        return [_PS.stem(w) for w in words]


# ----- Normalization & hashing ------------------------------------------------

_RE_NONALPHANUM = re.compile(r"[^a-z0-9 ]")
_SIGN_MASK = 1 << 127
_TWO_128   = 1 << 128


def hash_func(obj) -> int:
    """rbloom-compatible hash: stable, deterministic, signed 128-bit.

    Identical to L1's hash_func — both indexes share the hash scheme so the
    verify and predictor code can reuse the same function."""
    b = obj.encode("utf-8") if isinstance(obj, str) else obj
    h = xxhash.xxh3_128_intdigest(b)
    return h - _TWO_128 if h & _SIGN_MASK else h


def normalize_stem(text: str) -> list[str]:
    """L1-style alphanumeric normalization, then Porter stem each word."""
    words = _RE_NONALPHANUM.sub(" ", text.lower()).split()
    return _stem_many(words)


def grams4(words: list[str]):
    """Yield 4-stem grams as strings."""
    for i in range(len(words) - 3):
        yield " ".join(words[i:i + 4])


# ----- IO helpers (atomic save) -----------------------------------------------

def atomic_save_bloom(bf: Bloom, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    bf.save(str(tmp))
    os.replace(tmp, path)


def atomic_save_state(state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)


# ----- Local zstd shard reader ------------------------------------------------

def local_shard_stream(shard_paths: list[str]) -> Iterator[dict]:
    decompressor = zstd.ZstdDecompressor()
    for shard_path in shard_paths:
        with open(shard_path, "rb") as fh, decompressor.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ----- CLI --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("indexes/pile_l2_4gram.bloom"))
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--expected", type=int, default=20_000_000_000,
                   help="Expected unique stem-4grams. ~30%% smaller than the "
                        "5-gram count because stems collapse word variants.")
    p.add_argument("--fpr", type=float, default=1e-6)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=100_000)
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--shard-stride", type=int, default=1)
    p.add_argument("--shard-offset", type=int, default=0)
    p.add_argument("--local-shards", action="append", default=None)
    return p.parse_args()


# ----- Main loop --------------------------------------------------------------

def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.state or args.out.with_suffix(".state.json")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stderr)
    log = logging.getLogger("pile_l2")

    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "docs": 0,
        "grams": 0,
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if args.out.exists():
        log.info("Loading existing Bloom filter from %s", args.out)
        bf = Bloom.load(str(args.out), hash_func)
        log.info("Resumed at doc=%d grams=%d", state["docs"], state["grams"])
    else:
        log.info("Creating new Bloom filter expected=%d fpr=%.0e at %s",
                 args.expected, args.fpr, args.out)
        bf = Bloom(args.expected, args.fpr, hash_func)

    saved = {"v": False}
    def save_and_exit(signum, _frame):
        if saved["v"]:
            return
        saved["v"] = True
        log.info("Caught signal %d — saving filter + state", signum)
        atomic_save_bloom(bf, args.out)
        atomic_save_state(state, state_path)
        log.info("Saved at doc=%d grams=%d. Exiting.", state["docs"], state["grams"])
        sys.exit(0)
    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    # Pick source: local zstd files (fast) vs HF stream (slow).
    if args.local_shards:
        for sp in args.local_shards:
            if not Path(sp).exists():
                log.error("Local shard not found: %s", sp)
                return 2
        log.info("Reading from %d local shard(s): %s",
                 len(args.local_shards), args.local_shards)
        full_iter = local_shard_stream(args.local_shards)
        if state["docs"] > 0:
            log.info("Resuming: skipping first %d docs already processed", state["docs"])
            for _ in range(state["docs"]):
                try:
                    next(full_iter)
                except StopIteration:
                    log.warning("Skip past end of stream — already complete")
                    break
        ds = full_iter
    elif args.shard_stride > 1:
        if not (0 <= args.shard_offset < args.shard_stride):
            log.error("--shard-offset (%d) must be in [0, --shard-stride=%d)",
                      args.shard_offset, args.shard_stride)
            return 2
        log.info("Streaming monology/pile-uncopyrighted (shard %d/%d)...",
                 args.shard_offset, args.shard_stride)
        ds = (
            load_dataset("monology/pile-uncopyrighted", streaming=True)["train"]
            .shard(num_shards=args.shard_stride, index=args.shard_offset)
            .skip(state["docs"])
        )
    else:
        log.info("Streaming monology/pile-uncopyrighted train split...")
        ds = (
            load_dataset("monology/pile-uncopyrighted", streaming=True)["train"]
            .skip(state["docs"])
        )

    last_log_doc = state["docs"]
    last_log_t = time.time()
    iterator = iter(ds)
    pbar = tqdm(iterator, initial=state["docs"], unit="doc",
                smoothing=0.05, dynamic_ncols=True)

    for doc in pbar:
        stems = normalize_stem(doc["text"])
        gram_count = max(len(stems) - 3, 0)
        for g in grams4(stems):
            bf.add(g)
        state["docs"] += 1
        state["grams"] += gram_count

        if state["docs"] - last_log_doc >= args.log_every:
            now = time.time()
            rate = (state["docs"] - last_log_doc) / max(now - last_log_t, 1e-9)
            log.info("doc=%d grams=%d rate=%.1f docs/s",
                     state["docs"], state["grams"], rate)
            last_log_doc = state["docs"]
            last_log_t = now

        if state["docs"] % args.checkpoint_every == 0:
            atomic_save_bloom(bf, args.out)
            atomic_save_state(state, state_path)

        if args.max_docs is not None and state["docs"] >= args.max_docs:
            log.info("Reached --max-docs=%d, stopping", args.max_docs)
            break

    atomic_save_bloom(bf, args.out)
    atomic_save_state(state, state_path)
    log.info("DONE. doc=%d grams=%d filter=%s state=%s",
             state["docs"], state["grams"], args.out, state_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
