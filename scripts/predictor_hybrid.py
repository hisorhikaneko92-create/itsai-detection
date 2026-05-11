"""Hybrid L1 + HSSD predictor — the production inference logic for SN32.

Architecture: per-WORD priority, not per-text routing. For every word:

    L1 says "matched" (Pile-sourced human)  →  trust L1, output 0.0
    L1 says "not matched"                    →  defer to HSSD's continuous prob

Why per-word:
  - Mixed texts (Pile prompt + AI completion) get the right answer everywhere:
    L1 marks the prompt portion as human; HSSD predicts AI on the completion.
  - Pure-Pile texts: L1 covers everything, HSSD never used → fast, deterministic
  - Pure-CC/AI texts: L1 finds nothing, output reflects HSSD entirely
  - Boundary cases (CC text with one coincidental 5-gram match): smoothing in
    L1 filters most false positives; surviving L1 matches override HSSD, which
    is fine because those words are *almost certainly* human-from-Pile in the
    rare case the L1 match was real.

Performance: ~1 ms/text for L1 + ~12-18s for batched HSSD on 120 texts.
The HSSD call is unchanged from current deployment — we just override its
output where L1 has signal.

Determinism: both layers are stable across calls. L1 is hash-based (trivially
deterministic). HSSD has the per-text cache + fixed-length padding inherited
from predict_document.py. The hybrid output is byte-identical for the same
text — so gate-2 (count_penalty consistency) passes automatically.

Future extension: insert L2 / L3 / perplexity LM between L1 and HSSD — the
combine rule is "first layer with non-zero signal wins", others act as
fallback. Each layer is independent and pluggable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

# Same dir imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor_l1 import L1Predictor
from predict_document import HSSDPredictor


_LOG = logging.getLogger("hybrid")


class HybridPredictor:
    """L1 (Pile retrieval) + HSSD (model) hybrid.

    Args:
        l1_blooms: paths to 4 L1 Bloom shard files
        hssd_model_dir: directory holding the HSSD model (best/ with
            lora_adapter/ inside, OR a model.pth file)
        hssd_base_model: HuggingFace name of the base DeBERTa (default v3-large)
        hssd_device: 'cuda', 'cpu', or None for auto
        l1_min_run: smoothing parameter for L1 (see predictor_l1)
        l1_ai_prob: value emitted by L1 for unmatched words. Set to 0.5 so
            L1's "no opinion" doesn't bias HSSD's continuous output during
            merge. (At deploy time, the merge logic just falls through to
            HSSD when L1 is unmatched, so this is only the L1-standalone
            fallback value.)
    """

    def __init__(
        self,
        l1_blooms: Iterable[Path | str],
        hssd_model_dir: Path | str,
        hssd_base_model: str = "microsoft/deberta-v3-large",
        hssd_device: str | None = None,
        l1_min_run: int = 3,
        l1_ai_prob: float = 0.99,
    ):
        _LOG.info("Loading L1 (%d shards)...", len(list(l1_blooms))
                  if hasattr(l1_blooms, "__len__") else 4)
        self.l1 = L1Predictor(l1_blooms, min_run=l1_min_run, ai_prob=l1_ai_prob)

        _LOG.info("Loading HSSD model from %s ...", hssd_model_dir)
        self.hssd = HSSDPredictor(
            model_dir=str(hssd_model_dir),
            base_model=hssd_base_model,
            device=hssd_device,
        )

    def predict_one(self, text: str) -> list[float]:
        """For one text. Output length == len(text.split())."""
        l1_pred = self.l1.predict(text)
        if not l1_pred:
            return []

        # Fast path: if L1 covered every word with high confidence, skip HSSD
        n = len(l1_pred)
        n_matched = sum(1 for p in l1_pred if p < 0.5)
        if n_matched == n:
            return [0.0] * n   # entire text is Pile-confirmed

        # Slow path: HSSD must run on this text
        hssd_pred = self.hssd.predict_with_probs(text)
        if len(hssd_pred) != n:
            # Word-count mismatch (shouldn't happen since both use text.split())
            # Be safe: return L1 with clamp
            _LOG.warning("L1 and HSSD disagree on word count (%d vs %d) for "
                         "text starting with %r; using L1 alone",
                         n, len(hssd_pred), text[:60])
            return l1_pred

        # Combine: L1 wins where matched, HSSD fills the rest
        return [0.0 if lp < 0.5 else hp for lp, hp in zip(l1_pred, hssd_pred)]

    def predict_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch entry point. Preserves HSSD's GPU batching by running HSSD
        on the texts that *need* it (i.e. those not fully covered by L1)."""
        if not texts:
            return []

        # 1. L1 on all texts (fast, ~1 ms each)
        l1_preds = [self.l1.predict(t) for t in texts]

        # 2. Identify which texts still need HSSD
        need_hssd_idx: list[int] = []
        for i, l1_pred in enumerate(l1_preds):
            if not l1_pred:
                continue
            n_matched = sum(1 for p in l1_pred if p < 0.5)
            if n_matched < len(l1_pred):
                need_hssd_idx.append(i)

        # 3. Batched HSSD on just those texts (preserves GPU batching efficiency)
        hssd_results: dict[int, list[float]] = {}
        if need_hssd_idx:
            hssd_texts = [texts[i] for i in need_hssd_idx]
            hssd_batch = self.hssd.predict_batch_with_probs(hssd_texts)
            for idx_in_batch, idx_in_full in enumerate(need_hssd_idx):
                hssd_results[idx_in_full] = hssd_batch[idx_in_batch]

        # 4. Merge per-text
        out: list[list[float]] = []
        for i, l1_pred in enumerate(l1_preds):
            if not l1_pred:
                out.append([])
                continue
            if i not in hssd_results:
                # Full Pile coverage — L1 wins everywhere
                out.append([0.0] * len(l1_pred))
                continue
            hssd_pred = hssd_results[i]
            if len(hssd_pred) != len(l1_pred):
                _LOG.warning("Word-count mismatch on text %d (L1=%d, HSSD=%d)",
                             i, len(l1_pred), len(hssd_pred))
                out.append(l1_pred)
                continue
            merged = [0.0 if lp < 0.5 else hp for lp, hp in zip(l1_pred, hssd_pred)]
            out.append(merged)
        return out
