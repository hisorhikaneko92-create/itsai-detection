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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurons.miners.deberta_classifier import DebertaClassifier


LOGGER = logging.getLogger("remote_inference")


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

        self._write_json(
            HTTPStatus.OK,
            {
                "predictions": predictions,
                "latency_ms": int((time.time() - start_time) * 1000),
            },
        )

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
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
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

    server = ThreadingHTTPServer((args.host, args.port), InferenceHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down inference server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
