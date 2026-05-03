"""
Evaluate a trained Hybrid Semantic Seam Detector (HSSD) on a held-out CSV.

Loads the LoRA adapter + LSTM/classifier/CRF head saved by
train_seam_detector.py, runs inference on the test set, and reports:

  Overall:
      mean_seam_offset, f1_at_5, token_f1, accuracy_at_3
  Per stratification (sample_type, seam_bucket, data_source, model_name):
      same metrics broken down so you can see where the model is weak

Outputs:
  - Console table with the above
  - <output-dir>/metrics.json with the full numeric report
  - Optional: <output-dir>/worst_failures.csv with the N largest seam-offset
    examples for inspection (--dump-worst-failures N)

Usage:
    python scripts/evaluate_seam_detector.py \
        --checkpoint-dir models/seam_detector_v1/best \
        --test-csv data/test.csv \
        --output-dir reports/seam_detector_v1_test

The checkpoint dir must be the one written by save_checkpoint() in
train_seam_detector.py: it contains training_args.json, head.pth, and
either lora_adapter/ (if LoRA) or backbone_state_dict.pth.
"""

# Pre-load pandas + sklearn before transformers (Windows stack-overflow workaround).
import pandas  # noqa: F401
import sklearn  # noqa: F401

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

# Add repo root so we can import the model class from the training script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# These imports must come AFTER the sys.path tweak. The training script's
# top-of-file pandas/sklearn pre-import also fires here -- harmless dup.
from scripts.train_seam_detector import (  # noqa: E402
    PrincipalDetector,
    compute_seam_metrics,
    compute_token_f1,
)

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # only needed if checkpoint used LoRA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seam_bucket(labels: List[int]) -> str:
    """Mirror of split_dataset.py:_seam_bucket so reports stay consistent."""
    n = len(labels)
    if not n:
        return "unknown"
    s = sum(labels)
    if s == 0:
        return "pure_human"
    if s == n:
        return "pure_ai"
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            pos = i / n
            if pos < 0.2:
                return "early"
            if pos > 0.8:
                return "late"
            return "middle"
    return "unknown"


def _load_examples(csv_path: Path, tokenizer, max_length: int):
    """Read CSV, tokenize each row once, attach metadata + word_ids for
    sub-token-to-word aggregation later.

    Returns: list of dicts. Each dict has tokenized tensors AND the original
    word_labels + sample_type / data_source / etc. needed for reporting."""
    examples: List[Dict] = []
    skipped = 0
    truncated = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = row.get("text", "")
            try:
                word_labels = json.loads(row.get("segmentation_labels", "[]"))
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(word_labels, list):
                skipped += 1
                continue
            words = text.split()
            if len(words) != len(word_labels):
                skipped += 1
                continue
            if len(words) > max_length:
                # Rare for validator-faithful subsamples (35-350 words).
                # Truncate rather than skip so the chunk still contributes.
                words = words[:max_length]
                word_labels = word_labels[:max_length]
                truncated += 1

            encoding = tokenizer(
                words,
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            examples.append({
                "input_ids":      encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "word_ids":       encoding.word_ids(0),
                "word_labels":    list(map(int, word_labels)),
                "sample_type":    row.get("sample_type", "?"),
                "data_source":    row.get("data_source", "?"),
                "model_name":     row.get("model_name", "?"),
                "augmented":      row.get("augmented", "?"),
                "seam_bucket":    _seam_bucket(word_labels),
                "n_words":        len(word_labels),
                "text_preview":   text[:160],
            })
    if skipped:
        print(f"  skipped {skipped} malformed rows")
    if truncated:
        print(f"  truncated {truncated} rows that exceeded --max-length")
    return examples


def _word_preds_from_path(pred_path: List[int],
                          word_ids: List[Optional[int]],
                          n_words: int) -> List[int]:
    """Map sub-token predictions to word-level via the first-subword rule.

    `pred_path` is the CRF Viterbi output for positions where the mask was
    True (we keep the attention_mask plus the forced position-0 True).
    `word_ids` is parallel to the FULL token sequence (length = max_length);
    None for special tokens, integer-word-index for content tokens."""
    word_preds: List[Optional[int]] = [None] * n_words
    pi = 0
    for word_idx in word_ids:
        if pi >= len(pred_path):
            break
        if word_idx is None:
            pi += 1
            continue
        if word_preds[word_idx] is None:
            word_preds[word_idx] = int(pred_path[pi])
        pi += 1
    return [0 if p is None else p for p in word_preds]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _load_checkpoint(ckpt_dir: Path, device: torch.device) -> Tuple[PrincipalDetector, Dict, AutoTokenizer]:
    args_path = ckpt_dir / "training_args.json"
    if not args_path.exists():
        sys.exit(f"Missing {args_path}. Pass the checkpoint dir saved by "
                 f"train_seam_detector.py (contains training_args.json, "
                 f"head.pth, and lora_adapter/ or backbone_state_dict.pth).")
    train_args = json.loads(args_path.read_text(encoding="utf-8"))

    print(f"Loaded training_args.json:")
    for k in ("model_name", "max_length", "lstm_hidden", "lstm_layers",
              "dropout", "use_lora", "lora_r", "lora_alpha", "bf16"):
        if k in train_args:
            print(f"  {k}: {train_args[k]}")

    tokenizer = AutoTokenizer.from_pretrained(train_args["model_name"], use_fast=True)
    model = PrincipalDetector(
        model_name=train_args["model_name"],
        lstm_hidden=int(train_args.get("lstm_hidden", 256)),
        lstm_layers=int(train_args.get("lstm_layers", 2)),
        num_labels=2,
        dropout=float(train_args.get("dropout", 0.1)),
    )

    # Backbone weights
    use_lora = bool(train_args.get("use_lora", True))
    if use_lora:
        if PeftModel is None:
            sys.exit("Checkpoint uses LoRA but `peft` isn't installed. "
                     "pip install peft")
        adapter_dir = ckpt_dir / "lora_adapter"
        if not adapter_dir.exists():
            sys.exit(f"LoRA checkpoint missing adapter at {adapter_dir}")
        model.backbone = PeftModel.from_pretrained(
            model.backbone, str(adapter_dir), is_trainable=False,
        )
        print(f"  loaded LoRA adapter from {adapter_dir}")
    else:
        bb_path = ckpt_dir / "backbone_state_dict.pth"
        state = torch.load(bb_path, map_location="cpu")
        model.backbone.load_state_dict(state)
        print(f"  loaded backbone state dict from {bb_path}")

    # Head (LSTM + classifier + CRF)
    head_path = ckpt_dir / "head.pth"
    head = torch.load(head_path, map_location="cpu")
    model.lstm.load_state_dict(head["lstm"])
    model.classifier.load_state_dict(head["classifier"])
    model.crf.load_state_dict(head["crf"])
    print(f"  loaded LSTM/classifier/CRF from {head_path}")

    model.to(device)
    if train_args.get("bf16", True) and device.type == "cuda":
        model = model.to(torch.bfloat16)

    model.eval()
    return model, train_args, tokenizer


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
def _run_inference(model: PrincipalDetector, examples: List[Dict],
                   device: torch.device, batch_size: int, bf16: bool
                   ) -> List[Dict]:
    """Returns the input examples with `pred_word_labels` added per example."""
    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if bf16 and device.type == "cuda" else nullcontext())
    out: List[Dict] = []
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            input_ids = torch.stack([ex["input_ids"] for ex in batch]).to(device)
            attention_mask = torch.stack([ex["attention_mask"] for ex in batch]).to(device)
            with autocast_ctx:
                paths = model(input_ids, attention_mask)
            for ex, path in zip(batch, paths):
                pred_words = _word_preds_from_path(
                    path, ex["word_ids"], ex["n_words"],
                )
                ex_out = dict(ex)
                ex_out["pred_word_labels"] = pred_words
                # Strip tensors so we can json-serialize the metadata later
                ex_out.pop("input_ids", None)
                ex_out.pop("attention_mask", None)
                ex_out.pop("word_ids", None)
                out.append(ex_out)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _accuracy_at_k(gt_arrays: List[List[int]],
                   pred_arrays: List[List[int]], k: int) -> float:
    """Fraction of examples whose first transition is within k words of the
    ground-truth transition (or where both have no transition)."""
    correct = 0
    total = 0
    for gt, pred in zip(gt_arrays, pred_arrays):
        total += 1
        gt_seam = _first_transition_idx(gt)
        pr_seam = _first_transition_idx(pred)
        if gt_seam is None and pr_seam is None:
            correct += 1
        elif gt_seam is not None and pr_seam is not None:
            if abs(gt_seam - pr_seam) <= k:
                correct += 1
    return correct / max(1, total)


def _first_transition_idx(arr: List[int]) -> Optional[int]:
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            return i
    return None


def _metrics_for(results: List[Dict]) -> Dict[str, float]:
    if not results:
        return {"n": 0}
    gts = [r["word_labels"] for r in results]
    prs = [r["pred_word_labels"] for r in results]
    seam = compute_seam_metrics(gts, prs)
    return {
        "n":                 len(results),
        "n_with_seam":       seam["n_with_seam"],
        "mean_seam_offset":  seam["mean_seam_offset"],
        "f1_at_5":           seam["f1_at_5"],
        "accuracy_at_3":     _accuracy_at_k(gts, prs, 3),
        "accuracy_at_10":    _accuracy_at_k(gts, prs, 10),
        "token_f1":          compute_token_f1(gts, prs),
    }


def _print_metrics_row(label: str, m: Dict[str, float], width: int = 28) -> None:
    if m.get("n", 0) == 0:
        print(f"  {label:<{width}s}  (empty)")
        return
    n = m["n"]
    seam_off = m["mean_seam_offset"]
    seam_str = f"{seam_off:6.2f}" if not math.isnan(seam_off) else "  n/a "
    print(f"  {label:<{width}s}  n={n:>5d}  "
          f"seam_offset={seam_str}  "
          f"f1@5={m['f1_at_5']:.3f}  "
          f"acc@3={m['accuracy_at_3']:.3f}  "
          f"acc@10={m['accuracy_at_10']:.3f}  "
          f"token_f1={m['token_f1']:.3f}")


def _stratified_report(results: List[Dict], key: str) -> Dict[str, Dict]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        groups[str(r.get(key, "?"))].append(r)
    return {k: _metrics_for(rs)
            for k, rs in sorted(groups.items(), key=lambda kv: -len(kv[1]))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint-dir", required=True,
                    help="Directory containing training_args.json + head.pth "
                         "+ (lora_adapter/ or backbone_state_dict.pth). Usually "
                         "models/<run>/best or models/<run>/epoch_N.")
    ap.add_argument("--test-csv", required=True, nargs="+",
                    help="One or more CSVs to evaluate on. Same schema as "
                         "build_training_dataset.py output.")
    ap.add_argument("--output-dir", default=None,
                    help="Directory to write metrics.json (and worst_failures.csv "
                         "if --dump-worst-failures is set). Defaults to "
                         "<checkpoint-dir>/eval_<test_basename>.")
    ap.add_argument("--max-length", type=int, default=None,
                    help="Tokenizer max length. Defaults to the value from "
                         "training_args.json.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Inference batch size. Default 8.")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Cap rows for a fast smoke evaluation (debug).")
    ap.add_argument("--dump-worst-failures", type=int, default=0,
                    help="If > 0, write the N worst-seam-offset examples to "
                         "<output-dir>/worst_failures.csv for inspection.")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_dir():
        sys.exit(f"--checkpoint-dir not found: {ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, train_args, tokenizer = _load_checkpoint(ckpt_dir, device)
    max_length = args.max_length or int(train_args.get("max_length", 512))

    # Default output dir
    if args.output_dir is None:
        first_test = Path(args.test_csv[0])
        out_dir = ckpt_dir / f"eval_{first_test.stem}"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Load + tokenize examples
    examples: List[Dict] = []
    for path in args.test_csv:
        p = Path(path)
        if not p.exists():
            sys.exit(f"Test CSV not found: {p}")
        print(f"\nLoading {p}")
        examples.extend(_load_examples(p, tokenizer, max_length=max_length))
    if args.max_rows is not None:
        examples = examples[: args.max_rows]
    print(f"Total examples: {len(examples)}")

    if not examples:
        sys.exit("No usable examples to evaluate.")

    # Inference
    print(f"\nRunning inference (batch_size={args.batch_size})...")
    bf16 = bool(train_args.get("bf16", True))
    results = _run_inference(model, examples, device, args.batch_size, bf16)

    # Overall + stratified metrics
    overall = _metrics_for(results)
    print("\n" + "=" * 100)
    print("OVERALL")
    print("=" * 100)
    _print_metrics_row("overall", overall)

    print("\n" + "=" * 100)
    print("BY sample_type")
    print("=" * 100)
    by_sample_type = _stratified_report(results, "sample_type")
    for k, m in by_sample_type.items():
        _print_metrics_row(k, m)

    print("\n" + "=" * 100)
    print("BY seam_bucket")
    print("=" * 100)
    by_seam_bucket = _stratified_report(results, "seam_bucket")
    for k, m in by_seam_bucket.items():
        _print_metrics_row(k, m)

    print("\n" + "=" * 100)
    print("BY data_source")
    print("=" * 100)
    by_data_source = _stratified_report(results, "data_source")
    for k, m in by_data_source.items():
        _print_metrics_row(k, m)

    # Per-model report — only top 15 by row count to keep the table readable
    print("\n" + "=" * 100)
    print("BY model_name (top 15 by frequency)")
    print("=" * 100)
    by_model = _stratified_report(results, "model_name")
    for i, (k, m) in enumerate(by_model.items()):
        if i >= 15:
            break
        _print_metrics_row(k, m, width=36)

    # Save metrics.json
    report = {
        "checkpoint_dir":  str(ckpt_dir),
        "test_csv":        [str(p) for p in args.test_csv],
        "n_examples":      len(results),
        "overall":         overall,
        "by_sample_type":  by_sample_type,
        "by_seam_bucket":  by_seam_bucket,
        "by_data_source":  by_data_source,
        "by_model_name":   by_model,
        "training_args":   train_args,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {metrics_path}")

    # Worst-failures dump
    if args.dump_worst_failures > 0:
        scored: List[Tuple[int, Dict]] = []
        for r in results:
            gt_seam = _first_transition_idx(r["word_labels"])
            pr_seam = _first_transition_idx(r["pred_word_labels"])
            if gt_seam is not None and pr_seam is not None:
                scored.append((abs(gt_seam - pr_seam), r))
            elif gt_seam is None and pr_seam is None:
                pass  # perfect single-class case
            else:
                # Massive penalty: missed or hallucinated
                scored.append((r["n_words"], r))
        scored.sort(key=lambda kv: -kv[0])
        keep = scored[: args.dump_worst_failures]
        worst_path = out_dir / "worst_failures.csv"
        with open(worst_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "seam_offset", "sample_type", "seam_bucket", "data_source",
                "model_name", "n_words", "gt_seam_idx", "pred_seam_idx",
                "text_preview",
            ])
            writer.writeheader()
            for offset, r in keep:
                writer.writerow({
                    "seam_offset":    offset,
                    "sample_type":    r["sample_type"],
                    "seam_bucket":    r["seam_bucket"],
                    "data_source":    r["data_source"],
                    "model_name":     r["model_name"],
                    "n_words":        r["n_words"],
                    "gt_seam_idx":    _first_transition_idx(r["word_labels"]),
                    "pred_seam_idx":  _first_transition_idx(r["pred_word_labels"]),
                    "text_preview":   r["text_preview"],
                })
        print(f"Wrote {worst_path} ({len(keep)} worst examples)")

    print("\nDone.")


if __name__ == "__main__":
    main()
