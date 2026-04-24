import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurons.miners.deberta_classifier import DebertaClassifier


LOGGER = logging.getLogger("remote_inference")

# File-append lock so concurrent requests don't interleave partial summary
# lines. The raw-JSON path per request is unique (ts_ms + hash) so no lock
# is needed for those.
_SUMMARY_LOCK = threading.Lock()


def _text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def _per_text_stats(pred):
    """Collapse a prediction (list-of-floats or scalar) into a small dict of
    the stats we actually care about."""
    if isinstance(pred, list):
        if not pred:
            return {}
        n = len(pred)
        total = 0.0
        hi = lo = mid = 0
        for p in pred:
            p = float(p)
            total += p
            if p > 0.8:
                hi += 1
            elif p < 0.2:
                lo += 1
            elif 0.3 <= p <= 0.7:
                mid += 1
        return {
            "avg_pred": round(total / n, 4),
            "confident_ai_pct": round(hi / n, 3),
            "confident_human_pct": round(lo / n, 3),
            "uncertain_pct": round(mid / n, 3),
            # Fraction of words that are NOT in the muddy middle — higher is
            # better, means the model is taking clear positions.
            "bimodality": round((n - mid) / n, 3),
        }
    if isinstance(pred, (int, float)):
        return {"avg_pred": round(float(pred), 4)}
    return {}


def log_prediction_request(texts, predictions, mode, latency_ms, raw_dir, summary_path):
    """Write a raw JSON (full detail) and append a one-line summary for each
    processed /predict request. Emits the summary line via the logger too so
    it shows up in the server's console/pm2 log."""
    ts = datetime.now(timezone.utc)
    ts_iso = ts.isoformat()
    ts_ms = int(ts.timestamp() * 1000)

    hashes = [_text_hash(t) for t in texts]

    # Duplicate detection — same pair the miner's summary.log flags.
    seen = {}
    duplicates = []
    for i, h in enumerate(hashes):
        if h in seen:
            duplicates.append((seen[h], i))
        else:
            seen[h] = i

    per_text = []
    for i, (text, pred) in enumerate(zip(texts, predictions)):
        n_words = len(text.split())
        entry = {
            "idx": i,
            "hash": hashes[i],
            "n_words": n_words,
            "n_chars": len(text),
            "preview_head": text[:120],
        }
        entry.update(_per_text_stats(pred))
        per_text.append(entry)

    # Batch-level aggregates
    valid = [e for e in per_text if "avg_pred" in e]
    word_counts = [e["n_words"] for e in per_text] or [0]
    total_words = sum(word_counts)
    mean_avg = (
        round(sum(e["avg_pred"] for e in valid) / len(valid), 4)
        if valid else None
    )
    mean_bimodality = (
        round(sum(e.get("bimodality", 0) for e in valid) / len(valid), 3)
        if any("bimodality" in e for e in valid) else None
    )

    raw = {
        "timestamp_utc": ts_iso,
        "mode": mode,
        "num_texts": len(texts),
        "total_words": total_words,
        "duplicates": duplicates,
        "latency_ms": latency_ms,
        "mean_avg_pred": mean_avg,
        "mean_bimodality": mean_bimodality,
        "texts": per_text,
    }

    try:
        raw_path = raw_dir / f"{ts_ms}.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
    except Exception as e:
        LOGGER.warning("failed to write raw request log: %s", e)

    avg_word = sum(word_counts) // max(1, len(word_counts))
    summary_line = (
        f"[{ts_iso}] mode={mode} n_texts={len(texts)} "
        f"words(min/avg/max)={min(word_counts)}/{avg_word}/{max(word_counts)} "
        f"total_words={total_words} dups={len(duplicates)} latency_ms={latency_ms} "
        f"avg_pred(mean)={mean_avg if mean_avg is not None else 'NA'} "
        f"bimodality(mean)={mean_bimodality if mean_bimodality is not None else 'NA'}"
    )
    LOGGER.info(summary_line)
    with _SUMMARY_LOCK:
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(summary_line + "\n")
        except Exception as e:
            LOGGER.warning("failed to append summary log: %s", e)


def build_model(args):
    if args.model_type == "ppl":
        from neurons.miners.ppl_model import PPLModel
        model = PPLModel(device=args.device)
        model.load_pretrained(args.ppl_model_path)
        return model

    return DebertaClassifier(
        foundation_model_path=args.deberta_foundation_model_path,
        model_path=args.deberta_model_path,
        device=args.device,
    )


class InferenceHandler(BaseHTTPRequestHandler):
    model = None
    model_lock = threading.Lock()
    token = ""
    model_type = ""
    device = ""
    prediction_mode = "document"
    sentence_smoothing = 0.15
    window_size = 96
    window_stride = 48
    hybrid_window_weight = 0.65
    min_sentence_words = 4
    log_raw_dir = None       # Path — directory for per-request JSONs
    log_summary_path = None  # Path — append-only one-line summary log

    def _write_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        if not self.token:
            return True

        auth_header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.token}"
        return auth_header == expected

    def do_GET(self):
        if self.path != "/health":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        self._write_json(
            HTTPStatus.OK,
            {
                "status": "ok",
                "model_type": self.model_type,
                "device": self.device,
                "prediction_mode": self.prediction_mode,
            },
        )

    def do_POST(self):
        if self.path != "/predict":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        if not self._authorized():
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
        except Exception:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return

        texts = payload.get("texts")
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "texts_must_be_a_list_of_strings"})
            return

        start_time = time.time()
        try:
            with self.model_lock:
                if (
                    self.model_type == "deberta"
                    and hasattr(self.model, "predict_word_probabilities_batch")
                ):
                    predictions = self.model.predict_word_probabilities_batch(
                        texts,
                        mode=self.prediction_mode,
                        sentence_smoothing=self.sentence_smoothing,
                        window_size=self.window_size,
                        window_stride=self.window_stride,
                        hybrid_window_weight=self.hybrid_window_weight,
                        min_sentence_words=self.min_sentence_words,
                    )
                else:
                    predictions = self.model.predict_batch(texts)
                    predictions = [float(prediction) for prediction in predictions]
        except Exception as exc:
            LOGGER.exception("Prediction failed")
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "prediction_failed", "detail": str(exc)},
            )
            return

        latency_ms = int((time.time() - start_time) * 1000)
        self._write_json(
            HTTPStatus.OK,
            {"predictions": predictions, "latency_ms": latency_ms},
        )

        if self.log_raw_dir is not None and self.log_summary_path is not None:
            try:
                log_prediction_request(
                    texts=texts,
                    predictions=predictions,
                    mode=self.prediction_mode,
                    latency_ms=latency_ms,
                    raw_dir=self.log_raw_dir,
                    summary_path=self.log_summary_path,
                )
            except Exception as e:
                LOGGER.warning("request logging failed: %s", e)

    def log_message(self, format, *args):
        LOGGER.info("%s - %s", self.address_string(), format % args)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a local GPU-backed inference service for the SN32 miner.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=18091, help="Port to bind to.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for model inference.")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["deberta", "ppl"],
        default="deberta",
        help="Which detector model to load.",
    )
    parser.add_argument(
        "--prediction-mode",
        type=str,
        choices=["document", "sentence", "sentence_nltk", "window", "hybrid", "hybrid_nltk"],
        default="sentence_nltk",
        help=(
            "How to convert model scores into outputs for miner responses. "
            "'sentence_nltk' uses the same NLTK punkt tokenizer the validator "
            "uses when building mixed AI/human labels, so each sentence is "
            "classified independently and its score is attached to every word "
            "in that sentence."
        ),
    )
    parser.add_argument(
        "--sentence-smoothing",
        type=float,
        default=0.15,
        help="Neighbor smoothing factor for sentence-level scoring.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=96,
        help="Word window size for window-based scoring.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=48,
        help="Word stride for window-based scoring.",
    )
    parser.add_argument(
        "--hybrid-window-weight",
        type=float,
        default=0.65,
        help="Blend weight for window scores in hybrid mode.",
    )
    parser.add_argument(
        "--min-sentence-words",
        type=int,
        default=4,
        help=(
            "Minimum words per sentence group for sentence_nltk/hybrid_nltk. "
            "Sentences below this length are merged with their neighbors so "
            "the classifier sees enough context; the merged score is then "
            "assigned back to each original sentence's words."
        ),
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("REMOTE_INFERENCE_TOKEN", ""),
        help="Optional bearer token. Defaults to REMOTE_INFERENCE_TOKEN if set.",
    )
    parser.add_argument(
        "--deberta-foundation-model-path",
        type=str,
        default="models/deberta-v3-large-hf-weights",
        help="Path to the DeBERTa foundation model.",
    )
    parser.add_argument(
        "--deberta-model-path",
        type=str,
        default="models/deberta-large-ls03-ctx1024.pth",
        help="Path to the finetuned DeBERTa checkpoint.",
    )
    parser.add_argument(
        "--ppl-model-path",
        type=str,
        default="models/ppl_model.pk",
        help="Path to the PPL logistic regression checkpoint.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="inference_server_logs",
        help=(
            "Directory where per-request logs are written. A raw JSON per "
            "request is dropped into <log-dir>/raw/, and a one-line summary "
            "per request is appended to <log-dir>/summary.log. Use the same "
            "format as the VPS miner's neurons/validator_logs/ directory so "
            "you can tail it the same way."
        ),
    )
    parser.add_argument(
        "--disable-request-logging",
        action="store_true",
        help="Turn off per-request file logging (summary line is still emitted to stdout).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Use UTC for every "asctime" the logger renders so the Local PC console
    # lines up with the VPS miner/validator timestamps (which are always UTC).
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
    )

    LOGGER.info("Loading %s model on %s", args.model_type, args.device)
    model = build_model(args)
    LOGGER.info("Model loaded, starting server on %s:%s", args.host, args.port)

    InferenceHandler.model = model
    InferenceHandler.token = args.token
    InferenceHandler.model_type = args.model_type
    InferenceHandler.device = args.device
    InferenceHandler.prediction_mode = args.prediction_mode
    InferenceHandler.sentence_smoothing = args.sentence_smoothing
    InferenceHandler.window_size = args.window_size
    InferenceHandler.window_stride = args.window_stride
    InferenceHandler.hybrid_window_weight = args.hybrid_window_weight
    InferenceHandler.min_sentence_words = args.min_sentence_words

    if not args.disable_request_logging:
        log_dir = Path(args.log_dir).expanduser().resolve()
        raw_dir = log_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        summary_path = log_dir / "summary.log"
        # touch so tails don't complain about missing file before first request
        summary_path.touch(exist_ok=True)
        InferenceHandler.log_raw_dir = raw_dir
        InferenceHandler.log_summary_path = summary_path
        LOGGER.info(
            "Per-request logs: %s  (raw JSONs in %s, summary at %s)",
            log_dir, raw_dir, summary_path,
        )
    else:
        LOGGER.info("Per-request file logging disabled (--disable-request-logging).")

    server = ThreadingHTTPServer((args.host, args.port), InferenceHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down inference server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
