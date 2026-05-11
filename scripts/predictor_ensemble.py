"""Production ensemble predictor: L1 + HSSD + LightGBM meta-learner.

For each text:
   1. Run L1     → per-word L1 prediction (0.0 or 0.99)
   2. Run HSSD   → per-word HSSD probability (0–1)
   3. Compute 13 features per word (l1, hssd, neighborhoods, position, etc.)
   4. LightGBM predicts the final per-word AI probability
   5. Output continuous probabilities matching the validator's
      `predictions: List[List[float]]` protocol.

Gate-2 (count_penalty consistency) is automatically satisfied:
  - L1 lookups are deterministic
  - HSSD has its per-text cache (predict_document.py)
  - LightGBM inference is deterministic
  - Therefore: same text -> same output bits across calls.

To be extra defensive, an additional per-text dict cache wraps everything
in HybridPredictor.predict_batch — same as the existing HSSD layer.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numpy as np

from predictor_l1 import L1Predictor
from predict_document import HSSDPredictor
from extract_ensemble_features import compute_features, FEATURE_NAMES


_LOG = logging.getLogger("ensemble")


class EnsemblePredictor:
    """End-to-end ensemble predictor for the SN32 inference server.

    Args:
        l1_blooms: paths to 4 L1 Bloom shard files
        hssd_model_dir: directory holding HSSD's best/ (with lora_adapter/)
        lgbm_model_path: path to trained LightGBM model (.txt file from
                         train_ensemble_lgbm.py)
        hssd_base_model: HuggingFace name of base DeBERTa (default v3-large)
        hssd_device: 'cuda', 'cpu', or None for auto
        l1_min_run: smoothing run length for L1 (default 3)
        cache_ttl_sec: per-text output cache TTL (gate-2 belt+suspenders)
        cache_max: cache LRU capacity
    """

    def __init__(
        self,
        l1_blooms: Iterable[Path | str],
        hssd_model_dir: Path | str,
        lgbm_model_path: Path | str,
        hssd_base_model: str = "microsoft/deberta-v3-large",
        hssd_device: str | None = None,
        l1_min_run: int = 3,
        cache_ttl_sec: int = 600,
        cache_max: int = 2000,
    ):
        _LOG.info("Loading L1 (4 shards) ...")
        self.l1 = L1Predictor(l1_blooms, min_run=l1_min_run)

        _LOG.info("Loading HSSD from %s ...", hssd_model_dir)
        self.hssd = HSSDPredictor(
            model_dir=str(hssd_model_dir),
            base_model=hssd_base_model,
            device=hssd_device,
        )

        _LOG.info("Loading LightGBM ensemble from %s ...", lgbm_model_path)
        self.lgbm = lgb.Booster(model_file=str(lgbm_model_path))
        _LOG.info("  best_iteration=%d  num_features=%d",
                  self.lgbm.best_iteration, self.lgbm.num_feature())

        # Per-text deterministic cache (gate-2 defense)
        self._cache: "OrderedDict[str, tuple[float, list[float]]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_ttl = cache_ttl_sec
        self._cache_max = cache_max

    # ----- Cache helpers ----------------------------------------------------

    def _cache_get(self, text: str) -> list[float] | None:
        now = time.time()
        with self._cache_lock:
            entry = self._cache.get(text)
            if entry is None:
                return None
            ts, preds = entry
            if now - ts > self._cache_ttl:
                del self._cache[text]
                return None
            self._cache.move_to_end(text)
            return list(preds)  # defensive copy

    def _cache_set(self, text: str, preds: list[float]) -> None:
        with self._cache_lock:
            self._cache[text] = (time.time(), list(preds))
            self._cache.move_to_end(text)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)

    # ----- Core prediction --------------------------------------------------

    def _ensemble_for_pair(self, l1_pred: list[float], hssd_pred: list[float]) -> list[float]:
        """Compute the 13-feature matrix and run LightGBM inference. Returns
        a list of per-word AI probabilities matching len(l1_pred)."""
        n = len(l1_pred)
        if n == 0:
            return []
        if len(hssd_pred) != n:
            # Word-count mismatch shouldn't happen — both come from text.split()
            _LOG.warning("L1/HSSD word-count mismatch (%d vs %d); using L1 fallback",
                         n, len(hssd_pred))
            return l1_pred
        feats = compute_features(l1_pred, hssd_pred, n)
        probs = self.lgbm.predict(feats, num_iteration=self.lgbm.best_iteration)
        # Clip to valid [0, 1] range (LightGBM is fine but be defensive)
        return [float(min(1.0, max(0.0, p))) for p in probs]

    def predict_one(self, text: str) -> list[float]:
        """Single-text prediction. Output length == len(text.split())."""
        if not text:
            return []
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        l1_pred = self.l1.predict(text)
        if not l1_pred:
            self._cache_set(text, [])
            return []
        hssd_pred = self.hssd.predict_with_probs(text)
        out = self._ensemble_for_pair(l1_pred, hssd_pred)
        self._cache_set(text, out)
        return out

    def predict_batch(self, texts: list[str], max_batch_size: int = 32) -> list[list[float]]:
        """Batched entry point. Preserves HSSD's GPU batching efficiency."""
        if not texts:
            return []

        n = len(texts)
        results: list[list[float] | None] = [None] * n

        # 1. Cache lookup pass — gate-2 belt+suspenders
        miss_idx: list[int] = []
        for i, text in enumerate(texts):
            if not text:
                results[i] = []
                continue
            cached = self._cache_get(text)
            if cached is not None:
                results[i] = cached
            else:
                miss_idx.append(i)

        if not miss_idx:
            return [r if r is not None else [] for r in results]

        # 2. L1 (cheap, ~1ms per text)
        miss_texts = [texts[i] for i in miss_idx]
        l1_preds = [self.l1.predict(t) for t in miss_texts]

        # 3. HSSD batched (the slow part)
        hssd_preds = self.hssd.predict_batch_with_probs(
            miss_texts, max_batch_size=max_batch_size,
        )

        # 4. Features + LightGBM per text, then cache
        for k, real_i in enumerate(miss_idx):
            out = self._ensemble_for_pair(l1_preds[k], hssd_preds[k])
            results[real_i] = out
            self._cache_set(texts[real_i], out)

        return [r if r is not None else [] for r in results]
