"""
Visual side-by-side comparison of model predictions vs ground truth on
the first N rows of a test CSV. Intended for quick-look manual review
of a trained checkpoint, not formal benchmarking (use
evaluate_seam_detector_v3.py for that).

Per row, prints:
  - Sample type / seam bucket / word count / token count
  - Ground-truth seam position vs predicted seam position
  - Word-level offset and within-5-words verdict
  - Aligned label strings (GT vs pred), with seam markers
  - A snippet of words around the seam for human-readable context

Usage:
    python scripts/inspect_test_predictions.py \\
        --model-dir models/best \\
        --test-csv data/Training_Dataset/test.csv \\
        --n 20
"""
import pandas  # noqa: F401  (Windows stack-overflow workaround)
import sklearn  # noqa: F401

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_document import HSSDPredictor   # noqa: E402


def first_transition(arr: List[int]) -> Optional[int]:
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            return i
    return None


def labels_to_string(labels: List[int], gt_seam: Optional[int],
                     pr_seam: Optional[int]) -> str:
    """Render labels as a string of digits, with carets under the GT
    and predicted seam positions on a second line."""
    digits = "".join(str(x) for x in labels)
    marker = [" "] * len(labels)
    if gt_seam is not None and 0 <= gt_seam < len(labels):
        marker[gt_seam] = "^"   # ground-truth seam
    if pr_seam is not None and 0 <= pr_seam < len(labels):
        # Use a different char if it doesn't collide with GT
        if marker[pr_seam] == "^":
            marker[pr_seam] = "X"   # both align (perfect prediction)
        else:
            marker[pr_seam] = "v"   # predicted seam
    return digits, "".join(marker)


def chunk_print(s: str, width: int = 80, indent: str = "    ") -> None:
    """Print a long string in fixed-width chunks for readability."""
    for i in range(0, len(s), width):
        print(f"{indent}{s[i:i+width]}")


def truncate_for_display(labels: List[int], gt_seam: Optional[int],
                         pr_seam: Optional[int], max_len: int) -> tuple:
    """If the doc is very long, center the displayed window around the
    seam(s) so we see the relevant region instead of just the start."""
    if len(labels) <= max_len:
        return labels, gt_seam, pr_seam, 0

    # Pick a center: prefer GT seam, then predicted seam, then middle
    if gt_seam is not None:
        center = gt_seam
    elif pr_seam is not None:
        center = pr_seam
    else:
        center = len(labels) // 2

    half = max_len // 2
    start = max(0, center - half)
    end = min(len(labels), start + max_len)
    start = max(0, end - max_len)

    new_gt = (gt_seam - start) if gt_seam is not None and start <= gt_seam < end else None
    new_pr = (pr_seam - start) if pr_seam is not None and start <= pr_seam < end else None
    return labels[start:end], new_gt, new_pr, start


def context_snippet(words: List[str], seam: Optional[int], window: int = 4) -> str:
    """Return ' ... ' separating the words just before and just after
    the seam, so the reader can see the linguistic transition."""
    if seam is None or seam <= 0 or seam >= len(words):
        return "(no seam)"
    before = " ".join(words[max(0, seam - window):seam])
    after = " ".join(words[seam:min(len(words), seam + window)])
    return f"...{before}  ||  {after}..."


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model-dir", required=True,
                    help="Checkpoint dir containing lora_adapter/.")
    ap.add_argument("--test-csv", required=True,
                    help="CSV with text + segmentation_labels columns.")
    ap.add_argument("--n", type=int, default=20,
                    help="Number of rows to inspect. Default 20.")
    ap.add_argument("--base-model", default="microsoft/deberta-v3-large")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu. Auto-detect if omitted.")
    ap.add_argument("--max-display-words", type=int, default=120,
                    help="Truncate label rendering to this many words "
                         "(centered on the seam) for readability. Default 120.")
    args = ap.parse_args()

    print(f"Loading checkpoint from {args.model_dir} ...")
    predictor = HSSDPredictor(
        model_dir=args.model_dir,
        base_model=args.base_model,
        device=args.device,
    )
    print("Loaded.\n")

    n_total = 0
    n_correct_at_5 = 0
    n_with_seam_pair = 0
    sum_offset = 0

    with open(args.test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if n_total >= args.n:
                break

            text = (row.get("text") or "").strip()
            try:
                gt_labels = json.loads(row.get("segmentation_labels", "[]"))
            except json.JSONDecodeError:
                continue
            if not text or not isinstance(gt_labels, list):
                continue
            words = text.split()
            if len(words) != len(gt_labels):
                continue

            n_total += 1

            # Predict
            pred = predictor.predict(text)
            if len(pred) != len(words):
                if len(pred) < len(words):
                    pred = pred + [0] * (len(words) - len(pred))
                else:
                    pred = pred[:len(words)]

            gt_seam = first_transition(gt_labels)
            pr_seam = first_transition(pred)

            # Score
            if gt_seam is None and pr_seam is None:
                verdict = "✓ (both no-seam)"
                offset_str = "  -"
                n_correct_at_5 += 1
            elif gt_seam is None or pr_seam is None:
                verdict = "✗ (one missed seam)"
                offset_str = "  -"
            else:
                offset = abs(gt_seam - pr_seam)
                sum_offset += offset
                n_with_seam_pair += 1
                if offset <= 5:
                    verdict = f"✓ within 5 (offset={offset})"
                    n_correct_at_5 += 1
                else:
                    verdict = f"✗ off by {offset}"
                offset_str = f"{offset:>3}"

            # Pretty-print this row
            print("=" * 80)
            print(f"Doc {n_total:>2}  |  sample={row.get('sample_type','?'):<14s} "
                  f"source={row.get('data_source','?'):<14s} "
                  f"model={row.get('model_name','?'):<24s}")
            print(f"          n_words={len(words):>3}  "
                  f"gt_seam={str(gt_seam):>5}  pr_seam={str(pr_seam):>5}  "
                  f"offset={offset_str}  →  {verdict}")
            print(f"  Text: {text[:140]}{'...' if len(text) > 140 else ''}")

            # Truncate label display if doc is very long
            disp_gt, dgt_seam, _, offset_start = truncate_for_display(
                gt_labels, gt_seam, pr_seam, args.max_display_words,
            )
            disp_pr, _, dpr_seam, _ = truncate_for_display(
                pred, gt_seam, pr_seam, args.max_display_words,
            )
            offset_note = f" (showing words {offset_start}-{offset_start + len(disp_gt) - 1})" \
                if offset_start > 0 else ""

            gt_str, gt_marker = labels_to_string(disp_gt, dgt_seam, None)
            pr_str, pr_marker = labels_to_string(disp_pr, None, dpr_seam)

            print(f"  Labels{offset_note}:")
            print(f"    GT  : {gt_str}")
            print(f"          {gt_marker}   (^ = GT seam)")
            print(f"    Pred: {pr_str}")
            print(f"          {pr_marker}   (v = pred seam)")

            # Linguistic context around seams
            if gt_seam is not None:
                print(f"  GT seam context:   {context_snippet(words, gt_seam)}")
            if pr_seam is not None and pr_seam != gt_seam:
                print(f"  Pred seam context: {context_snippet(words, pr_seam)}")
            print()

    # Summary
    print("=" * 80)
    print(f"Summary over {n_total} test docs:")
    print(f"  f1@5 (within-5-words, including no-seam pairs): "
          f"{n_correct_at_5}/{n_total} = {n_correct_at_5/max(1,n_total):.3f}")
    if n_with_seam_pair > 0:
        print(f"  mean seam offset (over {n_with_seam_pair} both-have-seam docs): "
              f"{sum_offset/n_with_seam_pair:.2f} words")
    else:
        print(f"  mean seam offset: n/a (no docs where both GT and pred had a seam)")


if __name__ == "__main__":
    main()
