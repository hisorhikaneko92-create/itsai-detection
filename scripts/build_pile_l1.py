"""Build the L1 Pile 5-gram Bloom filter index.

Streams `monology/pile-uncopyrighted`, normalizes each document, hashes every
5-word n-gram with xxhash, and inserts into a disk-backed Bloom filter.

The output is the cornerstone of the SN32 retrieval miner: a per-text lookup
that classifies words as Pile-sourced human (label 0) vs not (label 1).

Resumable: ctrl-c or SIGTERM saves and exits cleanly. Re-run to continue.

Usage (production, ~24h on a beefy box):
    nohup python scripts/build_pile_l1.py > logs/pile_l1_build.log 2>&1 &

Test mode (verify pipeline on 5k docs, ~5 min):
    python scripts/build_pile_l1.py --max-docs 5000 \
        --out indexes/pile_l1_test.bloom --expected 100_000_000

CLI args allow shrinking the filter for test runs so you don't pre-allocate
75 GB on a tiny test.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path

import xxhash
from datasets import load_dataset
from rbloom import Bloom
from tqdm import tqdm


# ----- Normalization & hashing ------------------------------------------------

_RE_NONALPHANUM = re.compile(r"[^a-z0-9 ]")
_SIGN_MASK = 1 << 127
_TWO_128   = 1 << 128


def hash_func(obj) -> int:
    """rbloom-compatible hash: stable, deterministic, signed 128-bit.

    rbloom refuses to save filters built with Python's built-in hash because
    it's seeded per-process. We supply our own xxhash3-128 wrapper which is
    deterministic across runs/machines (so verify_pile_l1.py reproduces the
    exact same bits)."""
    b = obj.encode("utf-8") if isinstance(obj, str) else obj
    h = xxhash.xxh3_128_intdigest(b)
    return h - _TWO_128 if h & _SIGN_MASK else h


def normalize(text: str) -> list[str]:
    """Lowercase + strip non-alphanumerics. Tolerates ZWS, deletion,
    and most spelling-attack augmentations the validator uses."""
    return _RE_NONALPHANUM.sub(" ", text.lower()).split()


def grams5(words: list[str]):
    """Yield 5-word grams as strings. rbloom's hash_func handles hashing."""
    for i in range(len(words) - 4):
        yield " ".join(words[i:i + 5])


# ----- IO helpers -------------------------------------------------------------

def atomic_save_bloom(bf: Bloom, path: Path) -> None:
    """Write Bloom to <path>.tmp then rename → never leave a torn file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    bf.save(str(tmp))
    os.replace(tmp, path)


def atomic_save_state(state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)


# ----- CLI --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("indexes/pile_l1_5gram.bloom"),
                   help="Output Bloom filter path")
    p.add_argument("--state", type=Path, default=None,
                   help="State file (default: <out>.state.json next to --out)")
    p.add_argument("--expected", type=int, default=30_000_000_000,
                   help="Expected unique 5-grams (sizes the filter)")
    p.add_argument("--fpr", type=float, default=1e-6,
                   help="Target false-positive rate")
    p.add_argument("--max-docs", type=int, default=None,
                   help="Stop after N documents (for test runs)")
    p.add_argument("--checkpoint-every", type=int, default=100_000,
                   help="Save filter+state every N documents")
    p.add_argument("--log-every", type=int, default=10_000,
                   help="Print progress to stderr every N documents")
    p.add_argument("--shard-stride", type=int, default=1,
                   help="Total number of parallel workers. Use the same value "
                        "for every worker. With stride=4 each worker processes "
                        "1/4 of the dataset. The shards are file-based, so each "
                        "worker reads ~7-8 of the 30 Pile train files.")
    p.add_argument("--shard-offset", type=int, default=0,
                   help="This worker's index within --shard-stride. Worker 0 "
                        "uses offset=0, worker 1 offset=1, etc. Must be < stride.")
    return p.parse_args()


# ----- Main loop --------------------------------------------------------------

def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.state or args.out.with_suffix(".state.json")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("pile_l1")

    # Resume if state file exists
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

    # Save handler — fires on Ctrl-C and SIGTERM
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

    # Stream — apply shard partition first (file-level for streaming datasets),
    # then resume from last checkpoint within this worker's slice.
    if args.shard_stride > 1:
        if not (0 <= args.shard_offset < args.shard_stride):
            log.error("--shard-offset (%d) must be in [0, --shard-stride=%d)",
                      args.shard_offset, args.shard_stride)
            return 2
        log.info("Streaming monology/pile-uncopyrighted train split "
                 "(shard %d/%d)...", args.shard_offset, args.shard_stride)
        ds = (
            load_dataset("monology/pile-uncopyrighted", streaming=True)["train"]
            .shard(num_shards=args.shard_stride, index=args.shard_offset)
            .skip(state["docs"])
        )
    else:
        log.info("Streaming monology/pile-uncopyrighted train split (no shard)...")
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
        words = normalize(doc["text"])
        gram_count = max(len(words) - 4, 0)
        for h in grams5(words):
            bf.add(h)
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
