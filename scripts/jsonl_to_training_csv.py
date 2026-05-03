"""
Convert M4/MAGE-style JSONL files into the training CSV format used by
scripts/train_seam_detector.py + scripts/split_dataset.py.

Input format (one JSON object per line):
    {
        "prompt":      "Write a Wikipedia article ...",
        "human_text":  "<the human-written response>",
        "machine_text":"<the LLM-generated response for the same prompt>",
        "model":       "dolly-v2-12b",
        "source":      "wikipedia-20220301.en",
        "source_id":   12345
    }

Output CSV columns (matching build_training_dataset.py):
    text, segmentation_labels, data_source, sample_type, model_name,
    n_words, augmented

Each input row produces up to THREE output rows depending on --types:
  * pure_human    -- human_text alone, labels = [0]*N
  * pure_ai       -- machine_text alone, labels = [1]*N
  * human_then_ai -- random prefix(human) + random suffix(machine);
                    labels = [0]*p + [1]*s

ai_in_middle is intentionally NOT supported here because the
machine_text was generated INDEPENDENTLY of the human prefix in this
dataset (both answer the same prompt but neither sees the other's
output). Sandwiching machine_text inside human_text would produce a
stylistic seam but the content would be redundant or contradictory --
poor training signal compared to ai_in_middle samples from
build_training_dataset.py where the LLM is given begin+end+summary
context and asked to fill the middle. Use this script alongside
build_training_dataset.py rather than as a replacement for it.

Usage examples:
    # Convert every .jsonl in data/archive/ -> single CSV
    python scripts/jsonl_to_training_csv.py \\
        --input data/archive \\
        --output data/train_archive.csv

    # Pure_human + pure_ai only (no synthesized seam samples)
    python scripts/jsonl_to_training_csv.py \\
        --input data/archive \\
        --output data/train_archive_pure.csv \\
        --types pure_human,pure_ai

    # Cap each input file to 1000 rows for a quick smoke run
    python scripts/jsonl_to_training_csv.py \\
        --input data/archive \\
        --output data/train_archive_smoke.csv \\
        --max-per-input 1000

    # Then merge with the existing validator-shaped CSVs and split:
    python scripts/merge_csv_datasets.py \\
        --input data/train_50K_part1.csv data/train_50K_part2.csv \\
                data/train_archive.csv \\
        --output data/train_merged_with_archive.csv

    python scripts/split_dataset.py \\
        --input data/train_merged_with_archive.csv \\
        --output-train data/train.csv \\
        --output-val   data/val.csv \\
        --output-test  data/test.csv
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional


# Minimum word count for a row to be useful for seam training. Mirrors
# subsample_words(min_cnt=35) in build_training_dataset.py so emitted
# rows aren't smaller than what the validator's window would produce.
MIN_WORDS_PER_ROW = 35
# Minimum prefix/suffix size for synthesized human_then_ai. Prevents
# degenerate "1 human word + 50 AI words" shapes that confuse the
# stratified split's early/middle/late seam-bucket classifier.
MIN_HALF_WORDS = 10

VALID_TYPES = ("pure_human", "pure_ai", "human_then_ai")


def _coerce_to_str(value) -> str:
    """Some files in the M4 dataset (peerread_*, in particular) store
    human_text / machine_text as a LIST of paragraph strings rather
    than a single string. Coerce to str by newline-joining; pass
    strings through; turn anything else into the empty string so the
    caller's MIN_WORDS_PER_ROW check filters it out."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Join only the string entries; ignore any nested junk.
        return "\n".join(str(x) for x in value if isinstance(x, (str, int, float)))
    return ""


def emit_pure_human(text: str, source_tag: str) -> Optional[Dict[str, object]]:
    words = text.split()
    if len(words) < MIN_WORDS_PER_ROW:
        return None
    return {
        "text":                " ".join(words),
        "segmentation_labels": json.dumps([0] * len(words)),
        "data_source":         source_tag,
        "sample_type":         "pure_human",
        "model_name":          "none",
        "n_words":             len(words),
        "augmented":           False,
    }


def emit_pure_ai(text: str, source_tag: str, model: str) -> Optional[Dict[str, object]]:
    words = text.split()
    if len(words) < MIN_WORDS_PER_ROW:
        return None
    return {
        "text":                " ".join(words),
        "segmentation_labels": json.dumps([1] * len(words)),
        "data_source":         source_tag,
        "sample_type":         "pure_ai",
        "model_name":          model,
        "n_words":             len(words),
        "augmented":           False,
    }


def emit_human_then_ai(human: str, machine: str,
                       source_tag: str, model: str,
                       rng: random.Random,
                       cut_low: float = 0.25,
                       cut_high: float = 0.75) -> Optional[Dict[str, object]]:
    """Synthesize a one-seam sample by concatenating
    `human_words[:p]` + `machine_words[m:]`. The cut points are drawn
    independently in [cut_low, cut_high] of each text's length so the
    seam can land anywhere in the early / middle / late part of the
    final document (the stratified split classifies based on its
    fractional position)."""
    h_words = human.split()
    m_words = machine.split()
    if len(h_words) < MIN_HALF_WORDS or len(m_words) < MIN_HALF_WORDS:
        return None

    def _pick_cut(n: int) -> int:
        lo = max(MIN_HALF_WORDS, int(n * cut_low))
        hi = max(lo, min(n - MIN_HALF_WORDS, int(n * cut_high)))
        if hi <= lo:
            return lo
        return rng.randint(lo, hi)

    h_cut = _pick_cut(len(h_words))
    m_cut = _pick_cut(len(m_words))

    prefix = h_words[:h_cut]
    suffix = m_words[m_cut:]
    if len(prefix) < MIN_HALF_WORDS or len(suffix) < MIN_HALF_WORDS:
        return None

    all_words = prefix + suffix
    if len(all_words) < MIN_WORDS_PER_ROW:
        return None
    labels = [0] * len(prefix) + [1] * len(suffix)
    return {
        "text":                " ".join(all_words),
        "segmentation_labels": json.dumps(labels),
        "data_source":         source_tag,
        "sample_type":         "human_then_ai",
        "model_name":          model,
        "n_words":             len(all_words),
        "augmented":           False,
    }


def _resolve_inputs(inputs: List[str]) -> List[Path]:
    """Expand directories to their .jsonl contents; preserve files as-is."""
    out: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            found = sorted(p.glob("*.jsonl"))
            if not found:
                print(f"WARN: {p} contains no .jsonl files")
            out.extend(found)
        elif p.is_file():
            out.append(p)
        else:
            print(f"WARN: skipping {p} (not found)")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", nargs="+", required=True,
                    help="JSONL files or directories containing .jsonl files. "
                         "Directories are recursively scanned for *.jsonl.")
    ap.add_argument("--output", required=True,
                    help="Output CSV path. Existing file will be overwritten.")
    ap.add_argument("--types", default=",".join(VALID_TYPES),
                    help=f"Comma-separated subset of {VALID_TYPES}. "
                         f"Default emits all three.")
    ap.add_argument("--max-per-input", type=int, default=None,
                    help="Cap each input JSONL to N rows. Useful for "
                         "smoke runs across many large files. "
                         "Default: read every line.")
    ap.add_argument("--data-source-prefix", default="archive",
                    help="String prepended to each row's data_source tag. "
                         "Default 'archive' produces tags like "
                         "'archive:wikipedia-20220301.en'. The stratified "
                         "split honors (sample_type, seam_bucket) so this "
                         "tag is informational only.")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle output rows before writing. Without this "
                         "flag, output order follows input file order then "
                         "per-file line order.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Used for human_then_ai cut-point sampling and "
                         "(if --shuffle) the output shuffle.")
    args = ap.parse_args()

    requested = [t.strip() for t in args.types.split(",") if t.strip()]
    unknown = [t for t in requested if t not in VALID_TYPES]
    if unknown:
        sys.exit(f"--types contains unknown values: {unknown}\n"
                 f"Valid options: {VALID_TYPES}")
    requested_set = set(requested)

    paths = _resolve_inputs(args.input)
    if not paths:
        sys.exit("No JSONL files found")

    print(f"Found {len(paths)} JSONL file(s) to convert.")
    print(f"Emitting types: {sorted(requested_set)}")

    rng = random.Random(args.seed)
    fieldnames = ["text", "segmentation_labels", "data_source",
                  "sample_type", "model_name", "n_words", "augmented"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_input_rows = 0
    n_skipped_json = 0
    n_skipped_short = 0
    n_out_by_type: Dict[str, int] = {t: 0 for t in VALID_TYPES}

    rows_buffer: List[Dict[str, object]] = []
    write_streaming = not args.shuffle

    csvfile = open(out_path, "w", encoding="utf-8", newline="") if write_streaming else None
    writer = None
    if write_streaming:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    try:
        for path in paths:
            n_this_file = 0
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if args.max_per_input and n_this_file >= args.max_per_input:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        n_skipped_json += 1
                        continue

                    human   = _coerce_to_str(rec.get("human_text", ""))
                    machine = _coerce_to_str(rec.get("machine_text", ""))
                    model   = rec.get("model", "unknown") or "unknown"
                    src     = rec.get("source", path.stem) or path.stem
                    source_tag = f"{args.data_source_prefix}:{src}"

                    n_this_file += 1
                    n_input_rows += 1

                    new_rows: List[Dict[str, object]] = []
                    if "pure_human" in requested_set and human:
                        r = emit_pure_human(human, source_tag)
                        if r: new_rows.append(r)
                        else: n_skipped_short += 1
                    if "pure_ai" in requested_set and machine:
                        r = emit_pure_ai(machine, source_tag, model)
                        if r: new_rows.append(r)
                        else: n_skipped_short += 1
                    if ("human_then_ai" in requested_set
                            and human and machine):
                        r = emit_human_then_ai(human, machine, source_tag,
                                               model, rng)
                        if r: new_rows.append(r)
                        else: n_skipped_short += 1

                    for r in new_rows:
                        n_out_by_type[r["sample_type"]] = (
                            n_out_by_type.get(r["sample_type"], 0) + 1
                        )
                    if write_streaming:
                        for r in new_rows:
                            writer.writerow(r)
                    else:
                        rows_buffer.extend(new_rows)
            print(f"  {path.name:<40s}  {n_this_file:>7,} input rows")

        if not write_streaming:
            rng.shuffle(rows_buffer)
            with open(out_path, "w", encoding="utf-8", newline="") as fout:
                writer2 = csv.DictWriter(fout, fieldnames=fieldnames)
                writer2.writeheader()
                for r in rows_buffer:
                    writer2.writerow(r)
    finally:
        if csvfile is not None:
            csvfile.close()

    total_out = sum(n_out_by_type.values())
    print()
    print(f"Total input rows read:       {n_input_rows:,}")
    print(f"Skipped (JSON decode error): {n_skipped_json:,}")
    print(f"Skipped (text < {MIN_WORDS_PER_ROW} words):  {n_skipped_short:,}")
    print(f"Output rows by type:")
    for t in VALID_TYPES:
        if n_out_by_type.get(t, 0):
            print(f"  {t:<14s} {n_out_by_type[t]:>10,}")
    print(f"Total output rows:           {total_out:,}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
