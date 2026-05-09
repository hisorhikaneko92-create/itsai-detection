"""Pre-download `monology/pile-uncopyrighted` train shards to a local directory.

Once downloaded, build_pile_l1.py runs 5–10× faster against local files than
streaming from HF (no HTTP round-trips, no per-IP throttling, decompression
overlaps with subsequent Python work via stream_reader).

Disk: 30 shards × ~11 GB = ~335 GB total. Resumable — if a shard already
exists locally with the right size, hf_hub_download skips it.

Usage:
    python scripts/download_pile_shards.py
    python scripts/download_pile_shards.py --dest data/pile-uncopyrighted --shards 0-7
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm


def parse_shard_spec(spec: str) -> list[int]:
    """Parse '0-7' or '0,3,5' or 'all' into a list of shard indices [0..29]."""
    if spec == "all":
        return list(range(30))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    bad = [i for i in out if not (0 <= i <= 29)]
    if bad:
        raise ValueError(f"Shard indices out of [0,29]: {bad}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dest", type=Path,
                   default=Path("data/pile-uncopyrighted"),
                   help="Local directory to download to")
    p.add_argument("--shards", default="all",
                   help="Which shards (e.g. 'all', '0-7', '0,3,5'). Default: all 30")
    p.add_argument("--repo-id", default="monology/pile-uncopyrighted")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stderr)
    log = logging.getLogger("download_pile")

    shards = parse_shard_spec(args.shards)
    args.dest.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %d shard(s) of %s to %s",
             len(shards), args.repo_id, args.dest)

    start = time.time()
    for idx in tqdm(shards, desc="shards", unit="file"):
        filename = f"train/{idx:02d}.jsonl.zst"
        # hf_hub_download is incremental + resumable + integrity-checked
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(args.dest),
        )
        log.info("ok  %s", local_path)

    elapsed = time.time() - start
    log.info("DONE in %.1f min — %d shards under %s",
             elapsed / 60.0, len(shards), args.dest)
    log.info("Use --local-shards in build_pile_l1.py for the fast path:")
    log.info("  --local-shards %s", " --local-shards ".join(
        str(args.dest / f"train/{i:02d}.jsonl.zst") for i in shards[:3]))
    log.info("  ... etc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
