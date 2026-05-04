"""
HSSD v3 inference server. Drop-in replacement for run_inference_server.py
that uses HSSDPredictor (LoRA + CLAF + conv + BPM + CRF + sliding-window
emission averaging) instead of the legacy DebertaClassifier.

Same HTTP API as the original server:
  POST /predict  body: {"texts": ["...", ...]}
                 returns: {"predictions": [[float, ...], ...], "latency_ms": int}
  GET  /health   returns: {"status": "ok", "model_type": "hssd_v3", ...}

So your existing VPS<->LocalPC routing (RemoteInferenceClient on the VPS
hitting your LocalPC's tunnel) works unchanged — just launch this server
instead of the old one.

Usage on your Local PC:

    python scripts/run_inference_server_hssd.py \\
        --model-dir models/best \\
        --base-model models/deberta-v3-large-hf-weights \\
        --port 18091
"""
# === CUDA determinism (must run BEFORE torch CUDA init) ===
# Required for SN32's batch-consistency / determinism gate. The validator
# probes the same document both alone and inside a batched call, then
# fails the miner (penalty=0, reward=0) if the predictions differ. Three
# sources of cross-batch variance get pinned down here:
#
#   1. CUBLAS_WORKSPACE_CONFIG=":4096:8" -- reserves a fixed workspace so
#      cuBLAS can satisfy torch.use_deterministic_algorithms(True). Without
#      this env var, cuBLAS gemm calls under deterministic mode raise.
#   2. torch.use_deterministic_algorithms(True, warn_only=True) -- forces
#      every op that has a deterministic implementation to use it. warn_only
#      keeps benign non-deterministic ops (e.g. some scatter variants) from
#      crashing the server -- they print a UserWarning instead.
#   3. cudnn.deterministic + cudnn.benchmark=False -- disables cuDNN's
#      kernel-picking algorithm, which would otherwise select different
#      conv kernels for different (batch, seq_len) shapes and produce
#      shape-dependent rounding differences.
#
# This block MUST run before any torch CUDA call -- before model load,
# before tensor creation -- otherwise cuBLAS/cuDNN initialise their
# handles with the non-deterministic defaults and the settings only take
# partial effect. Putting it at the top of the file (above the rest of
# the imports) guarantees that ordering.
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# === end determinism block ===

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

# Import the existing HSSDPredictor — we don't reimplement it.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from predict_document import HSSDPredictor   # noqa: E402

LOGGER = logging.getLogger("hssd_inference")
_SUMMARY_LOCK = threading.Lock()


def _text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def _per_text_stats(pred):
    """Collapse a per-word prediction list into batch-level stats."""
    if not isinstance(pred, list) or not pred:
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
        "avg_pred":            round(total / n, 4),
        "confident_ai_pct":    round(hi / n, 3),
        "confident_human_pct": round(lo / n, 3),
        "uncertain_pct":       round(mid / n, 3),
        "bimodality":          round((n - mid) / n, 3),
    }


def log_prediction_request(texts, predictions, mode, latency_ms,
                           raw_dir, summary_path):
    ts = datetime.now(timezone.utc)
    ts_iso = ts.isoformat()
    ts_ms = int(ts.timestamp() * 1000)
    hashes = [_text_hash(t) for t in texts]

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

    valid = [e for e in per_text if "avg_pred" in e]
    word_counts = [e["n_words"] for e in per_text] or [0]
    total_words = sum(word_counts)
    mean_avg = (round(sum(e["avg_pred"] for e in valid) / len(valid), 4)
                if valid else None)

    raw = {
        "timestamp_utc": ts_iso,
        "mode": mode,
        "num_texts": len(texts),
        "total_words": total_words,
        "duplicates": duplicates,
        "latency_ms": latency_ms,
        "mean_avg_pred": mean_avg,
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
        f"total_words={total_words} dups={len(duplicates)} "
        f"latency_ms={latency_ms} "
        f"avg_pred(mean)={mean_avg if mean_avg is not None else 'NA'}"
    )
    LOGGER.info(summary_line)
    with _SUMMARY_LOCK:
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(summary_line + "\n")
        except Exception as e:
            LOGGER.warning("failed to append summary log: %s", e)


class HSSDInferenceHandler(BaseHTTPRequestHandler):
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
            "model_type": "hssd_v3",
            "device":     str(self.predictor.device),
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
                # Batched inference — single GPU forward over groups of
                # texts. Drops per-text overhead from ~150ms to ~5ms on
                # a typical GPU. Long inputs (>window_size tokens) still
                # use sliding-window per-text inside the batch method.
                predictions = self.predictor.predict_batch_with_probs(
                    texts,
                    max_batch_size=self.max_batch_size,
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
                    texts=texts, predictions=predictions, mode="hssd_v3",
                    latency_ms=latency_ms,
                    raw_dir=self.log_raw_dir,
                    summary_path=self.log_summary_path,
                )
            except Exception as e:
                LOGGER.warning("request logging failed: %s", e)

    def log_message(self, format, *args):
        LOGGER.info("%s - %s", self.address_string(), format % args)


def parse_args():
    p = argparse.ArgumentParser(
        description="HSSD v3 inference server (LoRA + CLAF + CRF, "
                    "sliding-window inference)."
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address. Default 127.0.0.1 (loopback only — "
                        "use 0.0.0.0 to accept LAN connections, but only "
                        "if your tunnel/auth setup is correct).")
    p.add_argument("--port", type=int, default=18091)
    p.add_argument("--device", default=None,
                   help="cuda/cpu/cuda:0. Auto-detect if omitted.")
    p.add_argument("--model-dir", default="models/best",
                   help="HSSD v3 checkpoint dir containing lora_adapter/.")
    p.add_argument("--base-model", default="models/deberta-v3-large-hf-weights",
                   help="DeBERTa weights + tokenizer dir or HF repo id.")
    p.add_argument("--window-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--max-batch-size", type=int, default=32,
                   help="Max texts per GPU forward pass for short-input "
                        "batching. 32 is comfortable on 24-GB A6000 with "
                        "bf16. Reduce to 16 if you OOM. Larger values "
                        "give more speedup but bigger transient peaks.")
    p.add_argument("--token", default=os.environ.get("REMOTE_INFERENCE_TOKEN", ""),
                   help="Optional bearer token. Defaults to env var.")
    p.add_argument("--log-dir", default="inference_server_logs",
                   help="Per-request raw JSON + summary log dir.")
    p.add_argument("--disable-request-logging", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
    )

    LOGGER.info("Loading HSSD v3 from %s (base=%s, device=%s)",
                args.model_dir, args.base_model, args.device or "auto")
    predictor = HSSDPredictor(
        model_dir=args.model_dir,
        base_model=args.base_model,
        device=args.device,
        window_size=args.window_size,
        stride=args.stride,
    )
    LOGGER.info("HSSD ready on %s. Serving on %s:%s",
                predictor.device, args.host, args.port)

    HSSDInferenceHandler.predictor = predictor
    HSSDInferenceHandler.token = args.token
    HSSDInferenceHandler.max_batch_size = args.max_batch_size

    if not args.disable_request_logging:
        log_dir = Path(args.log_dir).expanduser().resolve()
        raw_dir = log_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        summary_path = log_dir / "summary.log"
        summary_path.touch(exist_ok=True)
        HSSDInferenceHandler.log_raw_dir = raw_dir
        HSSDInferenceHandler.log_summary_path = summary_path
        LOGGER.info("Per-request logs: %s (raw=%s, summary=%s)",
                    log_dir, raw_dir, summary_path)
    else:
        LOGGER.info("Per-request file logging disabled.")

    server = ThreadingHTTPServer((args.host, args.port), HSSDInferenceHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down inference server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
