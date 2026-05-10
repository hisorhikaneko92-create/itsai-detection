"""L1 Pile retrieval predictor — the actual inference logic for the miner.

Wraps the 4 × L1 Bloom shards (built by build_pile_l1.py) and produces per-word
predictions for an arbitrary input text. Output shape matches what
neurons/miner.py expects: a list[float] of length `len(text.split())`.

The match rule is exactly the L1 logic explained to the user:

    1. Normalize text (lowercase, strip non-alphanumerics) → norm_words
    2. Slide a 5-word window across norm_words, check each window against any
       of the 4 Bloom shards (OR semantics)
    3. Apply run-length smoothing: only matched-windows that belong to a run
       of >= min_run consecutive hits survive (filters out coincidental
       common-English matches)
    4. Mark norm_words as "Pile" if any surviving smoothed window covers them
    5. Aggregate norm_word matches back to ORIGINAL words via the alignment
       tracked during normalization
    6. Per-original-word output: 0.0 if Pile (label = human), 0.99 if not
       Pile (label = AI). The 0.99 (instead of 1.0) is so np.round() still
       gives 1 but the validator's ap_score gets a meaningful continuous signal.

This file has no GPU dependencies. Loads ~108 GB Bloom filters on init; each
predict() call is microseconds. Used by the SN32 retrieval miner.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import xxhash
from rbloom import Bloom


# ----- Shared with build_pile_l1.py — MUST stay byte-compatible ---------------

_RE_NONALPHANUM = re.compile(r"[^a-z0-9 ]")
_SIGN_MASK = 1 << 127
_TWO_128   = 1 << 128


def hash_func(obj) -> int:
    b = obj.encode("utf-8") if isinstance(obj, str) else obj
    h = xxhash.xxh3_128_intdigest(b)
    return h - _TWO_128 if h & _SIGN_MASK else h


def _normalize_with_mapping(text: str) -> tuple[list[str], list[str], list[tuple[int, int]]]:
    """Return (orig_words, norm_words, orig_to_norm).

    orig_words: text.split() — preserves the miner-side word count contract.
    norm_words: alphanumeric-only lowercased tokens; the build's view of Pile.
    orig_to_norm: for each original word, a (start, end) range [start, end)
        into norm_words. start == end if the original word produced no
        alphanumeric content (e.g. "...", "---").
    """
    orig_words = text.split()
    norm_words: list[str] = []
    orig_to_norm: list[tuple[int, int]] = []
    for w in orig_words:
        nws = _RE_NONALPHANUM.sub(" ", w.lower()).split()
        start = len(norm_words)
        norm_words.extend(nws)
        end = len(norm_words)
        orig_to_norm.append((start, end))
    return orig_words, norm_words, orig_to_norm


def _smooth_hits(hits: list[bool], min_run: int) -> list[bool]:
    """Run-length smoother — survives only matched-windows that belong to a
    run of >= min_run consecutive True. Filters out lone matches that would
    otherwise be coincidental common-English n-grams.
    """
    if min_run <= 1:
        return list(hits)
    n = len(hits)
    out = [False] * n
    i = 0
    while i < n:
        if hits[i]:
            j = i
            while j < n and hits[j]:
                j += 1
            if j - i >= min_run:
                for k in range(i, j):
                    out[k] = True
            i = j
        else:
            i += 1
    return out


# ----- L1 predictor ----------------------------------------------------------

class L1Predictor:
    """Loads the 4 L1 shard Bloom filters and produces per-word predictions.

    Memory: ~108 GB resident (4 × 27 GB filters).
    Throughput: ~5000 grams/sec single-threaded (Python crossings dominate);
                a 120-text batch (~16k grams) finishes in ~300 ms.

    Args:
        bloom_paths: paths to the 4 L1 shard bloom files
        min_run: minimum consecutive matched-window run length for smoothing.
                 3 is conservative (allows 7-word matched spans); 1 disables.
        ai_prob: probability emitted for "not Pile" words (default 0.99 so the
                 validator's f1/fp scores round to AI but ap_score gets signal)
    """

    def __init__(
        self,
        bloom_paths: Iterable[Path | str],
        min_run: int = 3,
        ai_prob: float = 0.99,
    ):
        self.blooms = [Bloom.load(str(p), hash_func) for p in bloom_paths]
        self.min_run = min_run
        self.ai_prob = ai_prob

    def predict(self, text: str) -> list[float]:
        """Per-word AI probability. Output length == len(text.split())."""
        orig_words, norm_words, orig_to_norm = _normalize_with_mapping(text)
        n_orig = len(orig_words)
        n_norm = len(norm_words)
        if n_orig == 0:
            return []
        if n_norm < 5:
            return [self.ai_prob] * n_orig

        # 1. Check each 5-gram window against all 4 shards (OR)
        n_windows = n_norm - 4
        hits = [False] * n_windows
        for i in range(n_windows):
            gram = " ".join(norm_words[i:i + 5])
            for bf in self.blooms:
                if gram in bf:
                    hits[i] = True
                    break

        # 2. Run-length smoothing
        smoothed = _smooth_hits(hits, self.min_run)

        # 3. Each norm_word that any surviving window covers is "Pile-matched"
        norm_matched = [False] * n_norm
        for i in range(n_windows):
            if smoothed[i]:
                for j in range(i, i + 5):
                    if j < n_norm:
                        norm_matched[j] = True

        # 4. Map back to original words. An original word counts as Pile
        # if ANY of its norm_word sub-tokens are matched.
        out: list[float] = []
        for start, end in orig_to_norm:
            if start >= end:
                out.append(self.ai_prob)
            elif any(norm_matched[j] for j in range(start, end)):
                out.append(0.0)
            else:
                out.append(self.ai_prob)
        return out

    def predict_batch(self, texts: list[str]) -> list[list[float]]:
        """Convenience wrapper for batched calls (just loops predict())."""
        return [self.predict(t) for t in texts]