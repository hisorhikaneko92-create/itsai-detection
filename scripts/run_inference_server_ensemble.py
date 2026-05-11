"""SN32 ensemble inference server.

Drop-in replacement for run_inference_server_hssd.py that uses the
EnsemblePredictor (L1 + HSSD + LightGBM) instead of HSSD alone.

Same wire protocol:
  POST /predict  body: {"texts": ["...", ...]}
                 returns: {"predictions": [[float, ...], ...], "latency_ms": int}
  GET  /health   returns: {"status": "ok", "model_type": "ensemble", ...}

Usage:
    python scripts/run_inference_server_ensemble.py \\
        --hssd-model-dir   models/best \\
        --lgbm-model       models/ensemble_lgbm.txt \\
        --bloom            indexes/pile_l1_shard0.bloom \\
        --bloom            indexes/pile_l1_shard1.bloom \\
        --bloom            indexes/pile_l1_shard2.bloom \\
        --bloom            indexes/pile_l1_shard3.bloom \\
        --port             18091
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Same-dir imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor_ensemble import EnsemblePredictor


LOGGER = logging.getLogger("ensemble_server")


# ----- Lightweight request logger (mirrors HSSD server's behavior) -----------

def _truncate(s: str, n: int = 80) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 2] + ".."


def log_prediction_request(texts, predictions, mode, latency_ms, raw_dir, summary_path):
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ts_ms = int(time.time() * 1000)
    # avg per-text prediction (for trend monitoring)
    avg_preds = []
    for p in (predictions or []):
        if isinstance(p, list) and p:
            avg_preds.append(round(float(sum(p) / len(p)), 4))
        else:
            avg_preds.append(None)
    avg_str = (
        "NA" if not avg_preds or all(a is None for a in avg_preds)
        else f"{sum(a for a in avg_preds if a is not None) / max(1, sum(1 for a in avg_preds if a is not None)):.3f}"
    )
    word_counts = [len(t.split()) for t in (texts or [])] or [0]
    summary_line = (
        f"[{ts_iso}] mode={mode} n_texts={len(texts)} "
        f"words(min/avg/max)={min(word_counts)}/{sum(word_counts)//max(1, len(word_counts))}/{max(word_counts)} "
        f"latency_ms={latency_ms} avg_pred={avg_str}\n"
    )
    try:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(summary_line)
    except Exception as e:
        LOGGER.warning("failed to append summary log: %s", e)

    # Full raw dump
    raw = {
        "timestamp_utc": ts_iso, "mode": mode, "n_texts": len(texts),
        "latency_ms": latency_ms,
        "texts": [
            {"idx": i, "n_words": len(t.split()),
             "preview_head": _truncate(t[:200]),
             "preview_tail": _truncate(t[-200:]),
             "avg_prediction": avg_preds[i] if i < len(avg_preds) else None,
             "predictions": (
                 [round(float(v), 4) for v in predictions[i]]
                 if predictions and i < len(predictions) and isinstance(predictions[i], list)
                 else None
             )}
            for i, t in enumerate(texts)
        ],
    }
    raw_path = Path(raw_dir) / f"{ts_ms}_ensemble.json"
    try:
        with open(raw_path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh)
    except Exception as e:
        LOGGER.warning("failed to write raw dump: %s", e)


# ----- HTTP handler ----------------------------------------------------------

class EnsembleInferenceHandler(BaseHTTPRequestHandler):
    predictor = None
    model_lock = threading.Lock()
    token = ""
    max_batch_size = 32
    log_raw_dir = None
    log_summary_path = None

    def _write_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
            LOGGER.info("client %s disconnected before response (%s)",
                        self.address_string(), exc.__class__.__name__)

    def _authorized(self):
        if not self.token:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {self.token}"

    def do_GET(self):
        if self.path != "/health":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._write_json(HTTPStatus.OK, {
            "status":     "ok",
            "model_type": "ensemble_lgbm",
            "device":     str(self.predictor.hssd.device),
        })

    def do_POST(self):
        if self.path != "/predict":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if not self._authorized():
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._write_json(HTTPStatus.BAD_REQUEST,
                             {"error": "invalid_content_length"})
            return
        try:
            raw_body = self.rfile.read(content_length)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
            LOGGER.info("client %s hung up mid-request (%s); skipping",
                        self.address_string(), exc.__class__.__name__)
            return
        try:
            payload = json.loads(raw_body)
        except Exception:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return

        texts = payload.get("texts")
        if (not isinstance(texts, list)
                or not all(isinstance(t, str) for t in texts)):
            self._write_json(HTTPStatus.BAD_REQUEST,
                             {"error": "texts_must_be_a_list_of_strings"})
            return

        start_time = time.time()
        try:
            with self.model_lock:
                predictions = self.predictor.predict_batch(
                    texts, max_batch_size=self.max_batch_size,
                )
        except Exception as exc:
            LOGGER.exception("Prediction failed")
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                             {"error": "prediction_failed", "detail": str(exc)})
            return

        latency_ms = int((time.time() - start_time) * 1000)
        self._write_json(HTTPStatus.OK,
                         {"predictions": predictions, "latency_ms": latency_ms})

        if self.log_raw_dir is not None and self.log_summary_path is not None:
            try:
                log_prediction_request(
                    texts=texts, predictions=predictions, mode="ensemble_lgbm",
                    latency_ms=latency_ms,
                    raw_dir=self.log_raw_dir,
                    summary_path=self.log_summary_path,
                )
            except Exception as e:
                LOGGER.warning("request logging failed: %s", e)

    def log_message(self, format, *args):
        LOGGER.info("%s - %s", self.address_string(), format % args)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18091)
    p.add_argument("--device", default=None)
    p.add_argument("--bloom", action="append", required=False, default=None,
                   type=Path, help="L1 Bloom shard. Repeat for each shard. "
                                   "Default: indexes/pile_l1_shard{0,1,2,3}.bloom")
    p.add_argument("--hssd-model-dir", type=Path, default=Path("models/best"))
    p.add_argument("--hssd-base-model", default="microsoft/deberta-v3-large")
    p.add_argument("--lgbm-model", type=Path,
                   default=Path("models/ensemble_lgbm.txt"))
    p.add_argument("--l1-min-run", type=int, default=3)
    p.add_argument("--max-batch-size", type=int, default=32)
    p.add_argument("--token", default=os.environ.get("REMOTE_INFERENCE_TOKEN", ""))
    p.add_argument("--log-dir", default="inference_server_logs")
    p.add_argument("--disable-request-logging", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
    )

    if args.bloom is None:
        args.bloom = [Path(f"indexes/pile_l1_shard{i}.bloom") for i in range(4)]

    # Validate file paths
    for p_ in args.bloom:
        if not p_.exists():
            LOGGER.error("Missing Bloom shard: %s", p_); return 1
    if not args.lgbm_model.exists():
        LOGGER.error("Missing LightGBM model: %s", args.lgbm_model); return 1
    if not args.hssd_model_dir.exists():
        LOGGER.error("Missing HSSD model dir: %s", args.hssd_model_dir); return 1

    LOGGER.info("Loading EnsemblePredictor "
                "(bloom=%d shards, hssd=%s, lgbm=%s, device=%s)",
                len(args.bloom), args.hssd_model_dir, args.lgbm_model,
                args.device or "auto")
    predictor = EnsemblePredictor(
        l1_blooms=args.bloom,
        hssd_model_dir=args.hssd_model_dir,
        lgbm_model_path=args.lgbm_model,
        hssd_base_model=args.hssd_base_model,
        hssd_device=args.device,
        l1_min_run=args.l1_min_run,
    )
    LOGGER.info("Ensemble ready. Serving on %s:%s", args.host, args.port)

    EnsembleInferenceHandler.predictor = predictor
    EnsembleInferenceHandler.token = args.token
    EnsembleInferenceHandler.max_batch_size = args.max_batch_size

    if not args.disable_request_logging:
        log_dir = Path(args.log_dir).expanduser().resolve()
        raw_dir = log_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        summary_path = log_dir / "summary.log"
        summary_path.touch(exist_ok=True)
        EnsembleInferenceHandler.log_raw_dir = raw_dir
        EnsembleInferenceHandler.log_summary_path = summary_path
        LOGGER.info("Per-request logs: %s (raw=%s, summary=%s)",
                    log_dir, raw_dir, summary_path)
    else:
        LOGGER.info("Per-request file logging disabled.")

    server = ThreadingHTTPServer((args.host, args.port), EnsembleInferenceHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down inference server")
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
