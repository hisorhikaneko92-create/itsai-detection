"""
Export validator-request texts to plain-text files you can open in an editor.

One .txt file per request is written to neurons/validator_logs/texts/.
Each file contains all texts from that request separated by dividers.

Usage:
  python scripts/export_validator_texts.py                   # export everything
  python scripts/export_validator_texts.py --last 5          # only last 5 requests
  python scripts/export_validator_texts.py --hotkey 5DW      # filter validator
  python scripts/export_validator_texts.py --one-file-per-text   # separate file per text
"""
import argparse
import glob
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, "neurons", "validator_logs", "raw")
OUT_DIR = os.path.join(REPO_ROOT, "neurons", "validator_logs", "texts")


def get_text(t):
    if "full_text" in t:
        return t["full_text"]
    head = t.get("preview_head", "")
    tail = t.get("preview_tail", "")
    n_chars = t.get("n_chars", 0)
    if n_chars <= len(head):
        return head
    if n_chars <= len(head) + len(tail):
        overlap = len(head) + len(tail) - n_chars
        return head + tail[overlap:]
    return head + "\n\n...[MIDDLE TRUNCATED — enable full_text in miner.py]...\n\n" + tail


def write_bundled(r, ts_slug, hotkey_slug, out_dir):
    path = os.path.join(out_dir, f"{ts_slug}_{hotkey_slug}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Timestamp (UTC): {r['timestamp_utc']}\n")
        f.write(f"Validator:       {r['validator_hotkey']}\n")
        f.write(f"Version:         {r['validator_version']}  (in_range={r['version_in_range']})\n")
        f.write(f"Texts:           {r['num_texts']}\n")
        f.write(f"Latency:         {r['latency_ms']} ms\n")
        if r.get("duplicates"):
            f.write(f"Duplicates:      {r['duplicates']}\n")
        f.write("\n")

        for t in r["texts"]:
            text = get_text(t)
            pred = t.get("avg_prediction")
            pred_s = f"{pred:.3f}" if pred is not None else "-"
            f.write("=" * 78 + "\n")
            f.write(f"[{t['idx']}]  words={t['n_words']}  chars={t['n_chars']}  pred={pred_s}  hash={t['hash']}\n")
            f.write("=" * 78 + "\n")
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            f.write("\n")
    return path


def write_per_text(r, ts_slug, hotkey_slug, out_dir):
    paths = []
    for t in r["texts"]:
        text = get_text(t)
        pred = t.get("avg_prediction")
        pred_s = f"{pred:.3f}" if pred is not None else "-"
        fname = f"{ts_slug}_{hotkey_slug}_idx{t['idx']:03d}_pred{pred_s}.txt"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# ts={r['timestamp_utc']}  validator={r['validator_hotkey']}  "
                    f"idx={t['idx']}  words={t['n_words']}  pred={pred_s}\n\n")
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=None, help="Export only the last N requests.")
    ap.add_argument("--hotkey", type=str, default=None, help="Filter by validator hotkey prefix.")
    ap.add_argument("--one-file-per-text", action="store_true",
                    help="Write one file per text instead of bundling by request.")
    ap.add_argument("--out", type=str, default=OUT_DIR, help="Output directory.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if args.last:
        files = files[-args.last:]

    written = 0
    for path in files:
        with open(path) as f:
            r = json.load(f)
        if args.hotkey and not r["validator_hotkey"].startswith(args.hotkey):
            continue
        ts_slug = r["timestamp_utc"].replace(":", "-").replace("+00:00", "Z").split(".")[0]
        hotkey_slug = r["validator_hotkey"][:12]

        if args.one_file_per_text:
            out_paths = write_per_text(r, ts_slug, hotkey_slug, args.out)
            written += len(out_paths)
        else:
            out_path = write_bundled(r, ts_slug, hotkey_slug, args.out)
            written += 1

    print(f"Wrote {written} file(s) to {args.out}")


if __name__ == "__main__":
    main()
