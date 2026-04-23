"""
Evaluate prediction modes on a validator-style labeled CSV.

The CSV is expected to have the shape produced by
``detection/validator/data_generator.py`` — one row per sample with at least:

    text_auged                    : str     (the exact text validators send)
    auged_segmentation_labels     : list[bool] stored as a Python-literal string
                                              (one label per word via .split())
    data_source                   : str     (optional; rows with 'common_crawl'
                                              count as out-of-domain for the gate)

For each requested mode we call ``predict_word_probabilities_batch`` and compute
the same metrics the validator uses in ``detection/validator/reward.py``:

    fp_score  = 1 - fp / n_words
    f1_score  = sklearn.metrics.f1_score(y_true, round(y_pred))
    ap_score  = sklearn.metrics.average_precision_score(y_true, y_pred)
    reward    = mean(fp_score, f1_score, ap_score)

and additionally report F1 on the out-of-domain subset against the 0.9 gate.

Typical usage (run on your GPU PC where the model weights live)::

    python scripts/eval_prediction_modes.py \
        --csv data/generated_data_v3.5_0.csv \
        --modes document sentence_nltk hybrid_nltk \
        --limit 200
"""
import argparse
import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_label_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [int(bool(x)) for x in value]
    if not isinstance(value, str):
        return None
    try:
        return [int(bool(x)) for x in ast.literal_eval(value)]
    except (ValueError, SyntaxError):
        return None


def load_labeled_csv(path, text_col, labels_col, domain_col, limit):
    df = pd.read_csv(path)
    missing = [c for c in (text_col, labels_col) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"CSV is missing required column(s) {missing}. Found: {list(df.columns)}"
        )

    has_domain = domain_col in df.columns

    texts, labels, domains = [], [], []
    skipped_bad_labels = 0
    skipped_mismatch = 0
    for _, row in df.iterrows():
        text = row[text_col]
        if not isinstance(text, str) or not text.strip():
            continue

        lbl = parse_label_list(row[labels_col])
        if lbl is None:
            skipped_bad_labels += 1
            continue

        n_words = len(text.split())
        if n_words != len(lbl):
            # Validator-side guarantee can still be off on rare augmentation
            # edge cases — skip those rows so the comparison stays honest.
            skipped_mismatch += 1
            continue

        texts.append(text)
        labels.append(lbl)
        domains.append(str(row[domain_col]) if has_domain else "")

        if limit and len(texts) >= limit:
            break

    print(
        f"Loaded {len(texts)} rows from {path}"
        + (f" (skipped {skipped_mismatch} mismatch, {skipped_bad_labels} unparseable)"
           if skipped_mismatch or skipped_bad_labels else "")
    )
    return texts, labels, domains


def align_to_word_count(pred, n_words):
    if isinstance(pred, (int, float)):
        return [float(pred)] * n_words
    pred = [float(x) for x in pred]
    if len(pred) == n_words:
        return pred
    if len(pred) < n_words:
        fill = pred[-1] if pred else 0.0
        return pred + [fill] * (n_words - len(pred))
    return pred[:n_words]


def score(preds_per_text, labels_per_text):
    y_pred = np.concatenate(
        [np.asarray(p, dtype=float) for p in preds_per_text]
    )
    y_true = np.concatenate(
        [np.asarray(l, dtype=bool) for l in labels_per_text]
    ).astype(int)
    if len(y_pred) == 0:
        return {"n_words": 0}

    y_hat = np.round(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

    f1 = f1_score(y_true, y_hat, zero_division=0.0)
    if len(set(y_true)) > 1:
        ap = float(average_precision_score(y_true, y_pred))
    else:
        ap = float("nan")
    fp_score = 1 - fp / len(y_pred)
    reward = (fp_score + f1 + ap) / 3 if not np.isnan(ap) else float("nan")

    return {
        "n_words": int(len(y_pred)),
        "fp_score": float(fp_score),
        "f1_score": float(f1),
        "ap_score": ap,
        "reward": reward,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def load_local_model(device, foundation_path, model_path):
    from neurons.miners.deberta_classifier import DebertaClassifier

    return DebertaClassifier(
        foundation_model_path=foundation_path,
        model_path=model_path,
        device=device,
    )


def predict_all(model, texts, mode):
    preds = model.predict_word_probabilities_batch(texts, mode=mode)
    return [align_to_word_count(p, len(t.split())) for p, t in zip(preds, texts)]


def print_table(rows):
    headers = ["mode", "f1", "ood_f1", "reward", "fp", "ap", "gate_OOD>=0.9", "secs"]
    widths = [16, 8, 8, 8, 8, 8, 14, 8]
    line = " ".join(h.rjust(w) if i else h.ljust(w) for i, (h, w) in enumerate(zip(headers, widths)))
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        passes = "PASS" if r.get("ood_f1", 0) >= 0.9 else "FAIL"
        cells = [
            r["mode"].ljust(widths[0]),
            f"{r['f1_score']:.4f}".rjust(widths[1]),
            (f"{r['ood_f1']:.4f}" if r.get("ood_f1") is not None else "-").rjust(widths[2]),
            f"{r['reward']:.4f}".rjust(widths[3]),
            f"{r['fp_score']:.4f}".rjust(widths[4]),
            f"{r['ap_score']:.4f}".rjust(widths[5]),
            passes.rjust(widths[6]),
            f"{r['secs']:.2f}".rjust(widths[7]),
        ]
        print(" ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Labeled CSV produced by data_generator.py")
    ap.add_argument("--text-col", default="text_auged")
    ap.add_argument("--labels-col", default="auged_segmentation_labels")
    ap.add_argument("--domain-col", default="data_source",
                    help="Rows with this column == 'common_crawl' score as OOD for the gate.")
    ap.add_argument("--modes", nargs="+",
                    default=["document", "sentence_nltk", "hybrid_nltk"],
                    help="Prediction modes to compare.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only use the first N rows (useful for quick runs).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--foundation", default="models/deberta-v3-large-hf-weights")
    ap.add_argument("--model-path", default="models/deberta-large-ls03-ctx1024.pth")
    args = ap.parse_args()

    texts, labels, domains = load_labeled_csv(
        args.csv, args.text_col, args.labels_col, args.domain_col, args.limit,
    )
    if not texts:
        raise SystemExit("No rows with aligned per-word labels — nothing to evaluate.")

    ood_mask = np.array([d == "common_crawl" for d in domains])
    print(f"Out-of-domain rows (common_crawl): {int(ood_mask.sum())} / {len(texts)}")

    print(f"Loading DeBERTa on {args.device} ...")
    model = load_local_model(args.device, args.foundation, args.model_path)

    rows = []
    for mode in args.modes:
        print(f"  running mode = {mode} ...")
        t0 = time.time()
        preds = predict_all(model, texts, mode)
        dt = time.time() - t0

        overall = score(preds, labels)
        row = {"mode": mode, "secs": dt, **overall}

        if ood_mask.any():
            ood_preds = [preds[i] for i in np.where(ood_mask)[0]]
            ood_labels = [labels[i] for i in np.where(ood_mask)[0]]
            ood = score(ood_preds, ood_labels)
            row["ood_f1"] = ood.get("f1_score")
        else:
            row["ood_f1"] = None

        rows.append(row)

    print_table(rows)

    # Friendly diff: pick best vs document
    if any(r["mode"] == "document" for r in rows):
        base = next(r for r in rows if r["mode"] == "document")
        print("\nF1 delta vs document mode:")
        for r in rows:
            if r["mode"] == "document":
                continue
            df = r["f1_score"] - base["f1_score"]
            od = (r["ood_f1"] - base["ood_f1"]) if (r.get("ood_f1") is not None and base.get("ood_f1") is not None) else None
            od_s = f"{od:+.4f}" if od is not None else "-"
            print(f"  {r['mode']:<16}  f1 {df:+.4f}   ood_f1 {od_s}")


if __name__ == "__main__":
    main()
