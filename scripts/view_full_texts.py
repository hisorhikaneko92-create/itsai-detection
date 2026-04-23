"""
Print the FULL text of each text in recent validator requests.

Usage:
  python scripts/view_full_texts.py                # last request, all texts
  python scripts/view_full_texts.py --requests 3   # last 3 requests
  python scripts/view_full_texts.py --idx 5        # only text index 5 per request
  python scripts/view_full_texts.py --hotkey 5DW   # filter by validator hotkey prefix
"""
import argparse
import glob
import json
import os

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neurons", "validator_logs", "raw")


def get_text(t):
    if "full_text" in t:
        return t["full_text"], False
    # Fallback: stitch head + tail (middle may be missing for long texts)
    head = t.get("preview_head", "")
    tail = t.get("preview_tail", "")
    n_chars = t.get("n_chars", 0)
    if n_chars <= len(head):
        return head, False
    if n_chars <= len(head) + len(tail):
        # Overlap — just join uniquely
        overlap = len(head) + len(tail) - n_chars
        return head + tail[overlap:], False
    return head + "\n...[MIDDLE TRUNCATED - enable full_text in miner.py]...\n" + tail, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=1, help="How many recent requests to show.")
    ap.add_argument("--idx", type=int, default=None, help="Only show this text index.")
    ap.add_argument("--hotkey", type=str, default=None, help="Filter by validator hotkey prefix.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))[::-1]
    shown = 0
    for path in files:
        if shown >= args.requests:
            break
        with open(path) as f:
            r = json.load(f)
        if args.hotkey and not r["validator_hotkey"].startswith(args.hotkey):
            continue
        shown += 1
        print("#" * 78)
        print(f"# Time:      {r['timestamp_utc']}")
        print(f"# Validator: {r['validator_hotkey']}")
        print(f"# Texts:     {r['num_texts']}   Latency: {r['latency_ms']} ms")
        print("#" * 78)
        for t in r["texts"]:
            if args.idx is not None and t["idx"] != args.idx:
                continue
            text, truncated = get_text(t)
            pred = t.get("avg_prediction")
            pred_s = f"{pred:.3f}" if pred is not None else "-"
            print()
            print(f"--- [{t['idx']}] words={t['n_words']} chars={t['n_chars']} pred={pred_s}"
                  f"{' (TRUNCATED)' if truncated else ''} ---")
            print(text)
        print()


if __name__ == "__main__":
    main()
