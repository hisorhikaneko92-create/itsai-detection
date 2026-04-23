"""
Tail-style follower for validator requests.
Prints each new request as a formatted block the moment it arrives.

Usage:
  python scripts/follow_validator_requests.py
"""
import glob
import json
import os
import time
from datetime import datetime, timezone

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neurons", "validator_logs", "raw")


def print_block(rec):
    print("=" * 78)
    print(f"Time       : {rec['timestamp_utc']}")
    print(f"Validator  : {rec['validator_hotkey']}")
    print(f"Version    : {rec['validator_version']}  (in_range={rec['version_in_range']})")
    print(f"Texts      : {rec['num_texts']}    Latency: {rec['latency_ms']} ms")
    if rec.get("duplicates"):
        print(f"Duplicates : {rec['duplicates']}")
    for t in rec["texts"]:
        pred = t.get("avg_prediction")
        tag = "AI " if (pred or 0) > 0.5 else "HUM"
        pred_s = f"{pred:.3f}" if pred is not None else "  -  "
        print(f"  [{t['idx']:>3}] {tag} pred={pred_s} words={t['n_words']:>3} hash={t['hash']}")
        print(f"         HEAD: {t['preview_head'][:100]}")
    print()


def main():
    seen = set(os.listdir(RAW_DIR)) if os.path.isdir(RAW_DIR) else set()
    print(f"Watching {RAW_DIR}")
    print(f"Ignoring {len(seen)} existing files. New requests will appear below.\n")

    while True:
        try:
            current = set(os.listdir(RAW_DIR)) if os.path.isdir(RAW_DIR) else set()
        except FileNotFoundError:
            current = set()

        new_files = sorted(current - seen)
        for fname in new_files:
            path = os.path.join(RAW_DIR, fname)
            for _ in range(5):
                try:
                    with open(path) as f:
                        rec = json.load(f)
                    break
                except (json.JSONDecodeError, FileNotFoundError):
                    time.sleep(0.1)
            else:
                continue
            print_block(rec)

        seen = current
        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
