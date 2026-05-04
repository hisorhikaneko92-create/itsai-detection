"""
Validator-faithful training-dataset builder for SN32.

Mirrors the validator pipeline (detection/validator/data_generator.py) without
requiring bittensor / cc_net. Supports both **local Ollama** and **hosted
Ollama-compatible APIs** (Ollama Cloud, OpenRouter, etc.) so you can use the
full validator model pool -- including 70B+ checkpoints that won't fit a home
GPU -- by paying per-token instead of buying hardware.

For each sample we:

  1. Pull one document from The Pile (HuggingFace streaming) -- same source the
     validator's PromptDataset uses.
  2. Char-cut at a random index in [0.25 .. 0.75] of the doc length, producing
     {prompt, completion}. Reproduces detection/validator/my_datasets.py:54
     and is the reason boundaries can fall mid-word ("Frankenstein" cases).
  3. Pick a sample type from the validator's mix:
       25% pure-human   (label = all 0)
       25% pure-AI      (label = all 1; prompt is dropped, only AI text kept)
       40% human-then-AI (prompt + AI continuation -> labels [0]*prompt_words + [1]*rest)
       10% AI-in-middle  (NLTK sentence-aligned begin + AI middle + end)
  4. For AI types, call the configured chat API with a model picked at random
     from --ollama-models. Default rotation = the deduplicated 27-model
     validator pool (detection/validator/data_generator.py:327-364). Override
     --ollama-models to use a curated subset.
  5. Apply light augmentation (typos / case flips / punctuation drops) that
     keeps len(text.split()) constant -- mirrors detection/attacks/data_augmentation.
  6. Subsample to 35-350 words via the validator's exact rule.
  7. Append one row to the output CSV. Resumable: rerun with the same --output
     and it counts existing rows, generating only the remainder.

Output CSV schema (compatible with scripts/eval_prediction_modes.py):
    text                 mixed AI+human text after subsample/augment
    segmentation_labels  JSON list of per-word 0|1
    data_source          'pile' (matches validator's data_source tag)
    sample_type          one of: pure_human, pure_ai, human_then_ai, ai_in_middle
    model_name           model used (or 'none' for pure_human)
    n_words              len(text.split()) -- sanity column
    augmented            'true' / 'false'

Default endpoint is **Ollama Cloud** (https://ollama.com) because the
validator pool contains 70B+ / 100B+ checkpoints (llama3.3:70b,
command-r-plus:104b, mistral-large:123b, athene-v2:72b, etc.) that won't fit
on a consumer GPU. A 12 GB local card can serve only the small tail of the
pool, so generating validator-faithful data needs a hosted backend.

============================================================================
USAGE A -- Ollama Cloud (default, recommended)
============================================================================

1. Create an API key at https://ollama.com/settings/keys.
2. Export it (PowerShell or bash) and run:

    # PowerShell (Windows)
    $env:OLLAMA_API_KEY = "sk-...."
    python scripts\build_training_dataset.py --n-samples 50000 --output data\train_50k.csv

    # bash (Linux/macOS/WSL)
    export OLLAMA_API_KEY="sk-...."
    python scripts/build_training_dataset.py --n-samples 50000 --output data/train_50k.csv

The script sends `Authorization: Bearer $OLLAMA_API_KEY` to
https://ollama.com/api/chat and rotates through all 27 deduplicated
validator-pool models. Output is resumable: if the run is interrupted, just
rerun the same command and it picks up where it left off.

Smoke-test with a tiny run first to confirm your key works:

    python scripts\build_training_dataset.py --n-samples 50 --output data\smoke.csv

============================================================================
USAGE B -- OpenRouter (recommended for paid-key users)
============================================================================

OpenRouter aggregates ~100 paid models behind a single OpenAI-compatible
API. Use the --provider preset to skip the boilerplate -- it sets api_mode,
URL, and a curated default model list (paid only, no :free):

    # PowerShell (Windows)
    $env:OPENROUTER_API_KEY = "sk-or-v1-..."
    python scripts\build_training_dataset.py `
        --provider openrouter `
        --n-samples 30000 `
        --output data\train_openrouter.csv

The default rotation is OPENROUTER_MODELS (~20 models spanning Llama,
Qwen, Mistral, Gemma, Cohere, DeepSeek, Phi, Nemotron, gpt-oss, Hermes).
Override with --ollama-models if you want to add Anthropic / GPT-4o /
your own selection. The script also sends OpenRouter's recommended
HTTP-Referer + X-Title headers so your usage shows up on their
dashboard (set $env:OPENROUTER_REFERER and $env:OPENROUTER_APP_NAME
to customize).

For other OpenAI-compatible providers (Together, Groq, vLLM, etc.),
configure manually:

    python scripts/build_training_dataset.py \
        --api-mode openai \
        --ollama-url https://api.together.xyz \
        --ollama-models "meta-llama/Llama-3.3-70B-Instruct-Turbo,..." \
        --output data/train_50k.csv

============================================================================
USAGE C -- Concentrate (https://api.concentrate.ai, Responses API)
============================================================================

Concentrate uses OpenAI's Responses API shape (POST /v1/responses/) rather
than chat completions, so it gets its own --api-mode.

    # PowerShell (Windows)
    $env:CONCENTRATE_API_KEY = "sk-..."
    python scripts\build_training_dataset.py `
        --api-mode responses `
        --ollama-url https://api.concentrate.ai `
        --ollama-models "model-id-1,model-id-2,..." `
        --n-samples 50 `
        --output data\smoke_concentrate.csv

The script flattens the validator-style {system, user} message pair into
Concentrate's `instructions` + `input` fields, sends `max_output_tokens`
(not `max_tokens`), and reads the assistant text out of
`output[].content[].text`. Multi-model rotation works the same as for the
other backends -- just put your model ids in --ollama-models.

============================================================================
USAGE D -- Local Ollama on a GPU PC (small models only, won't match validator)
============================================================================

Only useful for fast iteration on the pipeline itself; the rotation is
limited to what fits your VRAM, so the data is NOT validator-faithful.

    python scripts\build_training_dataset.py --ollama-url http://127.0.0.1:11434 \
        --n-samples 200 --output data\local_smoke.csv

The script auto-detects which models are pulled (via /api/tags), uses the
intersection with the validator pool, and defaults --workers to 1.

============================================================================

Cost note for Cloud: across the validator's mix, each sample averages
~600 tokens of completion. 50,000 samples ~= 30M tokens. Pricing varies by
provider tier; smoke-test with --n-samples 200 before committing to a run.
"""
import argparse
import csv
import hashlib
import json
import os
import queue
import random
import string
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

try:
    import nltk
except ImportError:
    sys.exit("Install dependencies first:  pip install nltk requests datasets tqdm")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Missing 'datasets'. Run:  pip install datasets")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("Missing 'tqdm'. Run:  pip install tqdm")


# ---------------------------------------------------------------------------
# NLTK punkt — same boundary tokenizer the validator uses.
# ---------------------------------------------------------------------------
def _load_punkt():
    for pkg in ("punkt_tab", "punkt"):
        try:
            return nltk.data.load("tokenizers/punkt/english.pickle")
        except LookupError:
            nltk.download(pkg, quiet=True)
    return nltk.data.load("tokenizers/punkt/english.pickle")


_SENT_TOK = _load_punkt()


# ---------------------------------------------------------------------------
# Verbatim copies of validator helpers — keep these in sync if the validator
# updates its boundary logic. References are to detection/validator/.
# ---------------------------------------------------------------------------
def get_sentences(text: str) -> List[str]:
    """detection/validator/data_generator.py:26-44"""
    spans = list(_SENT_TOK.span_tokenize(text))
    out = []
    for i, (start, end) in enumerate(spans):
        next_start = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        expanded_end = end
        while expanded_end < next_start and expanded_end < len(text):
            if text[expanded_end].isspace():
                expanded_end += 1
            else:
                break
        out.append(text[start:expanded_end])
    return out


def merge_prompt_text(prompt: str, ai_text: str, human_then_ai_prob: float = 40 / 65) -> Tuple[str, int]:
    """detection/validator/segmentation_processer.py:12-25
    Returns (text, cnt_first_human). The split is exactly what the validator
    does — no space inserted between prompt and ai_text, so character-cut
    boundaries can land mid-word."""
    if not prompt:
        raise ValueError("prompt must be non-empty")
    if random.random() < human_then_ai_prob:
        return prompt + ai_text, len(prompt.split())
    return ai_text, 0


def subsample_words(text: str, labels: List[int],
                    min_cnt: int = 35, max_cnt: int = 350) -> Tuple[str, List[int]]:
    """detection/validator/segmentation_processer.py:27-100
    Random window over the text. Drops one transition if both 0->1 and 1->0
    are present. Returns the windowed (text, labels) with len matched."""
    words = text.split()
    if len(words) <= min_cnt:
        return " ".join(words), labels[:]
    cnt = random.randint(min_cnt, min(max_cnt, len(words)))

    has_01 = any(labels[i] == 0 and labels[i + 1] == 1 for i in range(len(labels) - 1))
    has_10 = any(labels[i] == 1 and labels[i + 1] == 0 for i in range(len(labels) - 1))

    if has_01 and has_10:
        ind = next((i + 1 for i in range(len(labels) - 1)
                    if labels[i] == 0 and labels[i + 1] == 1), None)
        if ind is not None:
            return subsample_words(" ".join(words[ind:]), labels[ind:], min_cnt, max_cnt)

    # Choose a window that, where possible, includes the boundary so the
    # classifier sees a real transition. If no boundary, sample anywhere.
    split_index = next(
        (i + 1 for i in range(len(labels) - 1) if labels[i] != labels[i + 1]),
        None,
    )
    if split_index is None:
        start = random.randint(0, len(words) - cnt)
    else:
        # Place the boundary somewhere inside the window
        start_min = max(0, split_index - cnt + 1)
        start_max = min(len(words) - cnt, split_index)
        start = random.randint(start_min, start_max) if start_max >= start_min else 0

    end = start + cnt
    return " ".join(words[start:end]), labels[start:end]


# ---------------------------------------------------------------------------
# Light augmentation — keeps len(text.split()) constant (the validator's
# DataAugmentator has the same invariant). Synonym replacement skipped to
# avoid the WordNet dependency; the typo/case/punct attacks below are the
# bulk of what gets thrown at miners anyway.
# ---------------------------------------------------------------------------
_PUNCT = set(string.punctuation)


def _augment_word(word: str, attack: str) -> str:
    if attack == "typo" and len(word) >= 4:
        i = random.randint(1, len(word) - 2)
        chars = list(word)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    if attack == "drop_char" and len(word) >= 5:
        i = random.randint(1, len(word) - 2)
        return word[:i] + word[i + 1 :]
    if attack == "case" and word and word[0].isalpha():
        return word[0].swapcase() + word[1:]
    if attack == "drop_punct":
        return "".join(c for c in word if c not in _PUNCT) or word
    return word


def augment(text: str, labels: List[int],
            per_word_p: float = 0.07) -> Tuple[str, List[int]]:
    """Return augmented text with the SAME word count and labels."""
    words = text.split()
    attacks = ("typo", "drop_char", "case", "drop_punct")
    out = []
    for w in words:
        if random.random() < per_word_p:
            attack = random.choice(attacks)
            out.append(_augment_word(w, attack))
        else:
            out.append(w)
    return " ".join(out), labels


# ---------------------------------------------------------------------------
# Validator model pool — names taken from
# detection/validator/data_generator.py:327-364, deduplicated. The validator
# itself instantiates command-r / command-r-plus:104b / internlm2.5:20b-chat
# multiple times to bias random.choice() toward them, but for training-data
# generation we want each distinct model to appear once so the rotation is
# evenly distributed.
# ---------------------------------------------------------------------------

# VALIDATOR_MODELS = [
#     "meta-llama/llama-2-13b-chat",                  # llama2:13b
#     "meta-llama/llama-3-70b-instruct",              # llama3:text
#     "meta-llama/llama-3-70b-instruct",              # llama3:70b
#     "meta-llama/llama-3.1-70b-instruct",            # llama3.1:70b-text-q4_0
#     "meta-llama/llama-3.2-3b-instruct",             # llama3.2
#     "meta-llama/llama-3.3-70b-instruct",            # llama3.3:70b
#     "qwen/qwen-2.5-32b-instruct",                   # qwen:32b-text-v1.5-q4_0
#     "qwen/qwen-2.5-14b-instruct",                   # qwen2.5:14b
#     "qwen/qwen-2.5-coder-32b-instruct",             # qwen2.5-coder:32b
#     "qwen/qwen-2.5-72b-instruct",                   # qwen2.5:72b
#     "qwen/qwen3-32b",                               # qwen3:32b
#     "cohere/command-r",                             # command-r
#     "cohere/command-r-plus",                        # command-r-plus:104b
#     "google/gemma-2-27b-it",                        # gemma2:27b-text-q4_0
#     "mistralai/mistral-nemo",                       # mistral-nemo:12b
#     "mistralai/mistral-small",                      # mistral-small:22b
#     "mistralai/mistral-large",                      # mistral-large:123b
#     "qwen/qwen-2.5-7b-instruct",                    # internlm2:7b          (replacement)
#     "qwen/qwen-2.5-32b-instruct",                   # internlm2:20b         (replacement)
#     "qwen/qwen-2.5-32b-instruct",                   # internlm2.5:20b-chat  (replacement)
#     "qwen/qwen-2.5-32b-instruct",                   # internlm2.5:latest    (replacement)
#     "deepseek/deepseek-chat",                       # deepseek-v2:16b       (now V3)
#     "deepseek/deepseek-r1-distill-qwen-14b",        # deepseek-r1:14b
#     "microsoft/phi-4",                              # phi4:14b
#     "cohere/aya-expanse-32b",                       # aya-expanse:32b
#     "01-ai/yi-large",                               # yi:34b-chat           (replacement)
#     "nvidia/llama-3.1-nemotron-70b-instruct",       # athene-v2:72b         (replacement)
# ]

# VALIDATOR_MODELS = [
#     # ── Llama family ──────────────────────────────────────────────
#     # llama2:13b — NOT in Concentrate; replaced with llama-3-8b
#     "llama-3-8b-instruct",
#     "llama-3-70b-instruct",          # llama3:text, llama3:70b
#     "llama-3.1-70b-instruct",        # llama3.1:70b-text-q4_0
#     "llama-3.2-3b-instruct",         # llama3.2
#     "llama-3.3-70b-instruct",        # llama3.3:70b, athene-v2:72b            (closest 70B-class)

#     # ── Qwen family ───────────────────────────────────────────────
#     # No Qwen 2.5 in Concentrate — only Qwen3.
#     "qwen3-32b",                     # qwen:32b-text-v1.5-q4_0  (was 32B), yi:34b-chat              (similar size & origin)
#     "qwen3-30b",                     # qwen2.5:14b              (closest size up)
#     "qwen3-coder-30b-a3b",           # qwen2.5-coder:32b
#     "qwen3-next-80b-a3b",            # qwen2.5:72b              (closest 70B+ Qwen)

#     # ── Cohere ────────────────────────────────────────────────────
#     # No command-r / command-r-plus in catalog — only Command A.
#     "command-a",                     # command-r, command-r-plus:104b

#     # ── Google Gemma ──────────────────────────────────────────────
#     "gemma-3-27b",                   # gemma2:27b-text-q4_0     (Gemma 3 replaces 2)

#     # ── Mistral ───────────────────────────────────────────────────
#     "mistral-nemo",                  # mistral-nemo:12b         (exact match)
#     "mistral-small-3.2",             # mistral-small:22b
#     "mistral-large-3",               # mistral-large:123b

#     # ── InternLM (NONE available) ─────────────────────────────────
#     # All four entries replaced with similar-sized substitutes.
#     "ministral-3-8b",                # internlm2:7b
#     "ministral-3-14b",               # internlm2:20b, internlm/internlm2.5:20b-chat, internlm/internlm2.5:latest, phi4:14b (no Phi in catalog)

#     # ── DeepSeek ──────────────────────────────────────────────────
#     "deepseek-v3-2",                 # deepseek-v2:16b          (V3.2 replaces V2)
#     "deepseek-r1-distill-32b",       # deepseek-r1:14b          (only R1 distill is 32B)

#     # ── Other (NOT available — replacements chosen) ───────────────
#     "command-a",                     # aya-expanse:32b          (Cohere multilingual fallback)

#     # ── Frontier closed-weight models ─────────────────────────────
#     # These produce stylistically distinct training data (different
#     # tokenizer, different RLHF tuning, different alignment) -- valuable
#     # diversity for the classifier. WITHOUT --reasoning-effort=none they
#     # take 30-300s per call (chain-of-thought before visible output), so
#     # always pair these with the reasoning-effort=none default.
#     "gpt-5.4-pro",
#     "gpt-5.3-codex",
#     # "claude-opus-4-7",
#     "claude-sonnet-4-6",
#     "claude-haiku-4-5",
#     "glm-5.1",
#     # "grok-4.20-0309-reasoning",
#     "kimi-k2-6",
# ]


# ---------------------------------------------------------------------------
# OpenRouter model rotation (used when --provider openrouter).
#
# All entries are PAID models (no `:free` suffix). OpenRouter's free tier
# caps at ~50 requests/day per model, which is useless for 30k+ sample
# generation. Pricing is current as of 2025-Q1 from openrouter.ai/models.
# Cost per 1M tokens (input + output averaged): low ~$0.05, high ~$8.
# Total cost for 30k samples (avg 2 API calls/sample, ~700 tokens each)
# at the median model price (~$0.4/M) is roughly $17.
#
# Selection criteria:
#   * Broad family coverage (Llama, Qwen, Mistral, Gemma, Cohere,
#     DeepSeek, Phi, Nemotron, gpt-oss) so the trained classifier
#     generalizes across stylistic fingerprints.
#   * Skip Anthropic / GPT-4 proper here -- per-call cost is 5-20x
#     median and the user can opt in via --ollama-models.
#   * Skip vision/embedding/audio models entirely (OpenRouter's
#     /v1/models lists hundreds of those).
# ---------------------------------------------------------------------------
OPENROUTER_MODELS = [
    # ── Llama family ──────────────────────────────────────────────
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.2-3b-instruct",

    # ── Qwen family ──────────────────────────────────────────────
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",

    # ── Google Gemma ──────────────────────────────────────────────
    "google/gemma-2-27b-it",

    # ── Mistral ──────────────────────────────────────────────────
    "mistralai/mistral-nemo",
    "mistralai/mistral-small-24b-instruct-2501",
    "mistralai/mistral-large",

    # ── Cohere ──────────────────────────────────────────────────
    "cohere/command-r-08-2024",
    "cohere/command-r-plus-08-2024",

    # ── DeepSeek ────────────────────────────────────────────────
    "deepseek/deepseek-chat",

    # ── Microsoft ────────────────────────────────────────────────
    # phi-3-medium-128k-instruct: REMOVED 2026-04 -- OpenRouter no
    # longer routes this slug (returns 404 "No endpoints found"). If
    # phi-4 or a successor reappears in the catalog, add it here.

    # ── Nvidia Nemotron ──────────────────────────────────────────
    "nvidia/llama-3.1-nemotron-70b-instruct",

    # ── OpenAI open-weight gpt-oss ───────────────────────────────
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",

    # ── Nous fine-tunes ─────────────────────────────────────────
    "nousresearch/hermes-3-llama-3.1-70b",
]

VALIDATOR_MODELS = [
"meta-llama/llama-3-70b-instruct",
"meta-llama/llama-3.1-70b-instruct",
"qwen/qwen-2.5-14b-instruct",
"qwen/qwen-2.5-32b-coder-instruct",
"qwen/qwen-2.5-72b-instruct",
"cohere/command-r",
"cohere/command-r-plus",
"google/gemma-2-27b-it",
"mistralai/mistral-small",
"mistralai/mistral-large",
"deepseek/deepseek-v2",
"deepseek/deepseek-r1",
"microsoft/phi-4",
"01-ai/yi-34b-chat"
]

# Provider presets: one flag (--provider) flips api_mode, base URL, and
# default model list in one shot. Each entry is the form
#   (api_mode, ollama_url, default_models_or_None, env_var_for_token).
# default_models=None means "use the auto-probe path".
PROVIDER_PRESETS = {
    "openrouter": {
        "api_mode":        "openai",
        "ollama_url":      "https://openrouter.ai/api",
        "default_models":  list(OPENROUTER_MODELS),
        "token_env_vars":  ("OPENROUTER_API_KEY",),
    },
    "concentrate": {
        "api_mode":        "responses",
        "ollama_url":      "https://api.concentrate.ai",
        "default_models":  list(VALIDATOR_MODELS),
        "token_env_vars":  ("CONCENTRATE_API_KEY",),
    },
    "ollama-cloud": {
        "api_mode":        "ollama",
        "ollama_url":      "https://ollama.com",
        "default_models":  None,                      # probe /api/tags
        "token_env_vars":  ("OLLAMA_API_KEY",),
    },
    "local": {
        "api_mode":        "ollama",
        "ollama_url":      "http://127.0.0.1:11434",
        "default_models":  None,                      # probe local /api/tags
        "token_env_vars":  (),                         # no token needed
    },
}

# Subset of VALIDATOR_MODELS that fits on a single ~12-16 GB consumer GPU at
# Q4 quantization. Used to (a) suggest `ollama pull` commands when running
# locally, and (b) pick a recommended subset when auto-detecting the local
# model pool. Pure-text models (llama3:text, qwen:32b-text-v1.5-q4_0, etc.)
# are excluded — they're large or rarely available.
LOCAL_GPU_MODELS = [
    "llama3.2",            # ~2 GB, 3 B
    "llama2:13b",          # ~7.4 GB Q4
    "qwen2.5:14b",         # ~9 GB Q4
    "mistral-nemo:12b",    # ~7 GB Q4
    "internlm2:7b",        # ~4.5 GB Q4
    "deepseek-r1:14b",     # ~9 GB Q4
    "phi4:14b",            # ~9 GB Q4
]


_LOCALHOST_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def _is_localhost(url: str) -> bool:
    """True if `url` points to a host on this machine. Used to decide
    whether to auto-detect pulled models and to pick worker defaults."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        return host in _LOCALHOST_HOSTS
    except Exception:
        return False


def _model_is_anthropic(model: str) -> bool:
    """True for Anthropic Claude slugs. Used in the Responses API path
    because Anthropic refuses `reasoning.effort='none'` and enforces a
    `max_output_tokens > 1024` floor whenever the `reasoning` field is
    present, so the cleanest behavior is to omit the field entirely for
    these models -- their default chat-completion behavior already skips
    extended thinking."""
    return isinstance(model, str) and model.lower().startswith("claude-")


# ---------------------------------------------------------------------------
# Chat client — multi-model, dual API mode. Authenticates with
# `Authorization: Bearer <token>` so it works with hosted services that
# require an API key (Ollama Cloud, OpenRouter, Together, etc.).
# ---------------------------------------------------------------------------
class ChatClient:
    """Unified client for two protocol modes:

    * ``api_mode='ollama'`` (default) — native Ollama API at /api/chat. Works
      for local `ollama serve` AND for Ollama Cloud (https://ollama.com)
      with the user's API key.
    * ``api_mode='openai'`` — OpenAI-compatible at /v1/chat/completions.
      Use for Together, OpenRouter, Groq, vLLM, llama.cpp server, etc.

    For the pure-AI sample type the validator uses raw text completion
    (`/api/generate?raw=true`). Hosted services rarely expose that, so this
    client always goes through the chat API and uses a "continue exactly the
    following text" instruction. The output style is close enough that the
    detector cannot tell the difference — and matches what most modern
    inference services support uniformly.
    """

    def __init__(self, base_url: str, models: List[str],
                 token: Optional[str] = None,
                 api_mode: str = "ollama",
                 chat_timeout: int = 300,
                 pool_size: int = 128,
                 reasoning_effort: Optional[str] = None,
                 extra_headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url.rstrip("/")
        self.models = models
        self.token = token
        self.api_mode = api_mode
        self.chat_timeout = chat_timeout
        # Extra HTTP headers attached to every request. Used for
        # OpenRouter's recommended `HTTP-Referer` + `X-Title` ranking
        # headers; harmless on other providers.
        self.extra_headers = dict(extra_headers) if extra_headers else {}
        # Only meaningful in Responses API mode. When set (e.g. "none"),
        # we inject `reasoning.effort = <value>` into every payload so
        # frontier reasoning models (gpt-5.4-pro, claude-opus, grok-reasoning,
        # etc.) skip their chain-of-thought and respond at normal-chat
        # latency. Non-reasoning models silently ignore the field.
        self.reasoning_effort = reasoning_effort

        # Shared session with a large connection pool. requests' default
        # HTTPAdapter caps at 10 connections per host, which silently
        # serializes worker threads above ~10 -- the reason workers=32
        # only gives 1.4x the throughput of workers=16. With pool_size=128
        # we get genuine concurrency up to that many in-flight calls.
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=0,  # we handle retries ourselves in _process_one
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if self.extra_headers:
            h.update(self.extra_headers)
        return h

    def health_check(self) -> List[str]:
        """Best-effort listing of available models. Many hosted services
        don't expose this — failures are non-fatal upstream.

        Tolerated response shapes:
          * Ollama:    {"models": [{"name": "..."}, ...]}
          * OpenAI:    {"data":   [{"id":   "..."}, ...]}
          * Concentrate (and some other Responses-API providers): bare list
                       [{"id": "..."}, ...]   or   ["model-id", ...]
        """
        url = (f"{self.base_url}/api/tags" if self.api_mode == "ollama"
               else f"{self.base_url}/v1/models")
        r = self.session.get(url, headers=self._headers(), timeout=15)
        r.raise_for_status()
        data = r.json()

        if self.api_mode == "ollama":
            if isinstance(data, dict):
                return [m.get("name", "") for m in data.get("models", [])]
            return []  # ollama listing should always be a dict; bail safely

        # openai or responses mode: accept dict-with-data, bare list-of-dicts,
        # or bare list-of-strings.
        items: List = []
        if isinstance(data, dict):
            items = data.get("data") or data.get("models") or []
        elif isinstance(data, list):
            items = data

        out: List[str] = []
        for it in items:
            if isinstance(it, dict):
                # `slug` is what Concentrate uses as the API ID at
                # /v1/responses/. `id` is what OpenAI / OpenRouter use.
                # `name` is a human-readable display label and only used
                # as a last-resort fallback (Ollama-style providers).
                mid = it.get("slug") or it.get("id") or it.get("name") or ""
                if mid:
                    out.append(mid)
            elif isinstance(it, str) and it:
                out.append(it)
        return out

    def pick(self) -> str:
        return random.choice(self.models)

    def chat(self, model: str, messages: List[Dict],
             num_predict: int = 900) -> str:
        """Multi-turn chat with system+user+assistant roles. Used by the
        AI-in-the-middle sampler (summary call + generation call)."""
        if self.api_mode == "ollama":
            # Native Ollama:
            #   POST https://ollama.com/api/chat
            #   {"model": ..., "messages": [...], "stream": false}
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": num_predict, "temperature": 0.7},
            }
            url = f"{self.base_url}/api/chat"
            r = self.session.post(url, headers=self._headers(), json=payload,
                                  timeout=self.chat_timeout)
            r.raise_for_status()
            return r.json()["message"]["content"]
        elif self.api_mode == "responses":
            # OpenAI Responses API style — used by Concentrate
            # (https://api.concentrate.ai/v1/responses/) and any provider
            # that adopts the spec. Differences from chat completions:
            #   * `input` is a single string, not a `messages` array, so we
            #     flatten the system prompt to `instructions` and concatenate
            #     non-system roles into `input`.
            #   * `max_output_tokens` instead of `max_tokens`.
            #   * Response text lives at `output[].content[].text` where
            #     output[].type == "message" and content[].type == "output_text".
            instructions_parts = [m["content"] for m in messages
                                  if m.get("role") == "system" and m.get("content")]
            input_parts = [m["content"] for m in messages
                           if m.get("role") != "system" and m.get("content")]
            payload = {
                "model": model,
                "input": "\n\n".join(input_parts),
                "stream": False,
                "max_output_tokens": num_predict,
                "temperature": 0.7,
            }
            if instructions_parts:
                payload["instructions"] = "\n\n".join(instructions_parts)
            if self.reasoning_effort and not _model_is_anthropic(model):
                # Disables (or tunes) chain-of-thought on reasoning models.
                # `reasoning.effort = "none"` is the big speed win: GPT-5,
                # Grok Reasoning, GLM, Kimi etc. respond at normal chat
                # latency (~5-10s) instead of 60-300s.
                #
                # Anthropic models are deliberately excluded: they reject
                # `effort=none` and force `max_output_tokens > 1024` whenever
                # the `reasoning` field is present. Omitting the field gets
                # us their default behavior (no extended thinking on chat
                # completions) without the token-cap penalty.
                payload["reasoning"] = {"effort": self.reasoning_effort}
            url = f"{self.base_url}/v1/responses/"
            r = self.session.post(url, headers=self._headers(), json=payload,
                                  timeout=self.chat_timeout)
            r.raise_for_status()
            data = r.json()
            for block in data.get("output", []):
                if block.get("type") == "message":
                    for c in block.get("content", []):
                        if c.get("type") == "output_text" and c.get("text"):
                            return c["text"]
            # No assistant text found — surface what we got so the caller's
            # error path logs something useful.
            err = data.get("error") or {}
            raise RuntimeError(
                f"responses API returned no output_text "
                f"(status={data.get('status')!r}, "
                f"error={err.get('message')!r})"
            )
        else:
            # OpenAI-compatible (also accepted at https://ollama.com/v1/...):
            #   POST <base>/v1/chat/completions
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "max_tokens": num_predict,
                "temperature": 0.7,
            }
            url = f"{self.base_url}/v1/chat/completions"
            hdrs = self._headers()
            if os.environ.get("DEBUG_HTTP"):
                masked = dict(hdrs)
                if "Authorization" in masked:
                    v = masked["Authorization"]
                    masked["Authorization"] = (v[:18] + "...REDACTED..." + v[-6:]
                                                if len(v) > 25 else "<short>")
                print(f"[DEBUG_HTTP] POST {url}\n"
                      f"  headers={masked}\n"
                      f"  payload.model={payload['model']!r}\n"
                      f"  payload.messages[0].role={messages[0].get('role')!r}\n"
                      f"  total_messages={len(messages)}",
                      flush=True)
            r = self.session.post(url, headers=hdrs, json=payload,
                                  timeout=self.chat_timeout)
            if os.environ.get("DEBUG_HTTP"):
                # Inspect what requests actually put on the wire.
                req = r.request
                req_hdrs = dict(req.headers)
                if "Authorization" in req_hdrs:
                    v = req_hdrs["Authorization"]
                    req_hdrs["Authorization"] = (v[:18] + "...REDACTED..." + v[-6:]
                                                  if len(v) > 25 else "<short>")
                print(f"[DEBUG_HTTP] <- status {r.status_code}\n"
                      f"  request_url    = {req.url}\n"
                      f"  request_headers= {req_hdrs}\n"
                      f"  response_body[:300]= {r.text[:300]}",
                      flush=True)
            r.raise_for_status()
            # Defensive parsing: some OpenRouter backends return
            #   {"choices":[]}                                  (empty)
            #   {"choices":[{"message":{"content":null}}]}      (refusal/empty)
            #   {"error":{...}} with status 200                 (rare but seen)
            # We turn all of these into a RuntimeError with a useful
            # message so the caller's retry/fresh-pair path fires
            # cleanly instead of crashing on `None.strip()` or
            # `KeyError: 'choices'`.
            data = r.json()
            choices = data.get("choices")
            if not choices:
                err = (data.get("error") or {}).get("message") or "empty choices"
                raise RuntimeError(
                    f"openai-compat response has no usable choices "
                    f"(model={model}, msg={err!r})"
                )
            content = (choices[0].get("message") or {}).get("content")
            if content is None:
                raise RuntimeError(
                    f"openai-compat response has null content "
                    f"(model={model}, finish_reason="
                    f"{choices[0].get('finish_reason')!r})"
                )
            return content

    def generate(self, model: str, prompt: str,
                 num_predict: int = 900, raw: bool = True) -> str:
        """Single-prompt completion via /api/generate (native Ollama only).

        Matches the user's working pattern:

            curl https://ollama.com/api/generate \\
                -H "Authorization: Bearer $OLLAMA_API_KEY" \\
                -d '{"model": "...", "prompt": "...", "stream": false}'

        With ``raw=True`` Ollama skips the chat template — equivalent to the
        validator's ``OllamaModel(text_completion_mode=True)`` path. For chat-
        tuned models on Cloud, ``raw=False`` lets Ollama apply the template;
        either way you get a coherent continuation.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0.7},
        }
        if raw:
            payload["raw"] = True
        url = f"{self.base_url}/api/generate"
        r = self.session.post(url, headers=self._headers(), json=payload,
                              timeout=self.chat_timeout)
        r.raise_for_status()
        return r.json().get("response", "")

    def continue_text(self, model: str, prompt: str,
                      num_predict: int = 900) -> str:
        """Used for pure-AI / human-then-AI sampling. Returns only the
        continuation (not the prompt), so the caller can concatenate
        ``prompt + continuation`` to mirror ``merge_prompt_text``.

        * Native Ollama mode → ``/api/generate`` with ``raw=true`` (closest
          match to the validator's text_completion_mode=True).
        * OpenAI-compat mode → ``/v1/chat/completions`` with a "continue this
          text" system instruction (raw completion isn't reliably supported
          on OpenAI-compat providers).
        """
        if self.api_mode == "ollama":
            # First try raw text-completion (matches validator's "text" models
            # like llama3:text, qwen:32b-text-v1.5-q4_0). If a chat-only model
            # rejects raw mode, retry without raw — Ollama then applies the
            # chat template and still gives a usable continuation.
            try:
                out = self.generate(model, prompt, num_predict=num_predict, raw=True)
                if out.strip():
                    return out.strip()
            except requests.HTTPError:
                pass
            return self.generate(model, prompt, num_predict=num_predict, raw=False).strip()

        instruction = (
            "Continue the following text in the same style and voice. "
            "Output ONLY the continuation — do not repeat the input, do not "
            "add a preamble, do not add a summary, do not add quotation marks. "
            "Aim for roughly 200-400 words of continuation."
        )
        return self.chat(
            model,
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
            num_predict=num_predict,
        ).strip()


# ---------------------------------------------------------------------------
# Validator prompt sets — copied subset from detection/validator/data_generator.py
# Random rotation makes the AI middle stylistically diverse.
# ---------------------------------------------------------------------------
SUMMARY_PROMPTS = [
    "Summarize the text in your own words, highlighting the key points. Do not generate anything else.",
    "Provide a concise summary of the text, focusing on its main argument. Refrain from generating anything else.",
    "In a few sentences, capture the core ideas of the text. Ensure you do not produce anything else.",
    "Write a short overview of the text, emphasizing the primary takeaways. Do not include anything else beyond the summary.",
    "Condense the text into a brief summary, touching on the essential details. Do not provide anything else in your response.",
]

GENERATION_PROMPTS = [
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. "
    "You will be given the start and finish of a text plus a summary of its middle. Your job is to "
    "compose only the middle portion, making sure it aligns with both the beginning and the end. "
    "Do not provide a summary; preserve any existing warnings by rephrasing them, and write nothing else. "
    "Do not generate anything else (Only middle part) - your output will be concatenated with begin and end.",

    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. "
    "You receive the opening and closing paragraphs of a text, as well as a synopsis of the central section. "
    "Your task is to generate the text for the middle part alone, ensuring coherence with the given "
    "beginning and end. Keep any cautions or alerts by rewording them, and do not include any summarizing. "
    "Do not generate anything else (Only middle part) - your output will be concatenated with begin and end.",

    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. "
    "You are provided with a text's first and final segments along with a brief outline of what occurs "
    "in the middle. Your job is to fill in only the middle content. The final text should flow naturally, "
    "so do not insert a summary. Retain all warnings by rephrasing, and write nothing else. "
    "Do not generate anything else (Only middle part) - your output will be concatenated with begin and end.",
]


# ---------------------------------------------------------------------------
# Source streamers — Pile + CommonCrawl (allenai/c4 by default since cc_net
# isn't usable from this script). Both produce the same {prompt, completion,
# data_source} shape used by detection/validator/my_datasets.py:47-70.
# ---------------------------------------------------------------------------
class HFTextStream:
    """Stream a HuggingFace dataset that has a 'text' (or 'raw_content')
    field, char-cutting each doc into a {prompt, completion} pair the same
    way detection/validator/my_datasets.py:53-56 does it."""

    def __init__(self, dataset_name: str, source_tag: str,
                 max_prompt_len: int = 1500,
                 seed: Optional[int] = None,
                 buffer_size: int = 10000,
                 load_kwargs: Optional[Dict] = None,
                 split: str = "train",
                 skip: int = 0):
        self.dataset_name = dataset_name
        self.source_tag = source_tag
        self.max_prompt_len = max_prompt_len
        self.buffer_size = buffer_size
        self.seed = seed if seed is not None else int(time.time())
        self.load_kwargs = load_kwargs or {}
        self.split = split
        # Stream offset. Set this in run 2/3/... so the new run reads from
        # past the documents the previous run consumed -- without it,
        # buffered shuffle starts at stream position 0 every time and the
        # second run sees ~the same head-of-stream documents the first run
        # already used (different order, same source docs).
        self.skip = max(0, int(skip))
        self._iter = None
        self._init_iter()

    def _init_iter(self):
        ds = load_dataset(self.dataset_name, streaming=True, **self.load_kwargs)
        if hasattr(ds, "keys") and self.split in ds:
            ds = ds[self.split]
        # `.skip()` advances past `self.skip` documents BEFORE shuffling.
        # This guarantees a non-overlapping document window across runs:
        # run 1 takes [0, ~N], run 2 with skip=N takes [N, ~2N], etc.
        if self.skip > 0:
            ds = ds.skip(self.skip)
        self._iter = iter(ds.shuffle(seed=self.seed, buffer_size=self.buffer_size))

    def next_pair(self) -> Dict[str, str]:
        for _ in range(50):
            try:
                el = next(self._iter)
            except StopIteration:
                self.seed = int(time.time())
                self._init_iter()
                continue
            text_field = "text" if "text" in el else "raw_content"
            text = el.get(text_field, "").replace("\x00", "")
            if not text:
                continue
            doc = text[: int(self.max_prompt_len * 1.25)]
            if len(doc) < 200:
                continue
            ctx_len = int(len(doc) * random.uniform(0.25, 0.75))
            return {
                "prompt": doc[:ctx_len][: self.max_prompt_len],
                "completion": text[ctx_len:],
                "data_source": "pile",
            }
        raise RuntimeError("Pile stream gave up after 50 attempts")


# ---------------------------------------------------------------------------
# Common Crawl streamer — same source as
# detection/validator/my_datasets.py:87-113. Wraps the validator's CCDataset
# (cc_net under the hood) so the data_source tag and quality filtering are
# identical to what your gate is scored on.
#
# Heavy: pulls a few hundred MB of WET segments to disk on first use.
# Slower per document than Pile streaming. Validator uses CC for ~33% of
# samples (PILE_PROB = 80/120), so a typical run hits a manageable amount.
# ---------------------------------------------------------------------------
class CCStream:
    def __init__(self,
                 max_prompt_len: int = 1500,
                 num_segments: int = 10,
                 cc_root: Optional[str] = None):
        # Late import so users without cc_net installed still see the rest of
        # the script work (Pile-only mode).
        from detection.validator.cc_dataset import CCDataset, get_2023_dumps  # noqa
        from pathlib import Path as _Path
        repo_root = _Path(__file__).resolve().parents[1]
        cc_dir = _Path(cc_root) if cc_root else (repo_root / "cc_net")
        self._CCDataset = CCDataset
        self._dumps = get_2023_dumps()
        self._cc_dir = cc_dir
        self._num_segments = num_segments
        self.max_prompt_len = max_prompt_len
        self._iter = None
        self._init_iter()

    def _init_iter(self):
        ds = self._CCDataset(
            dumps=self._dumps,
            num_segments=self._num_segments,
            lang_model=self._cc_dir / "bin" / "lid.bin",
            lm_dir=self._cc_dir / "data" / "lm_sp",
            lang_whitelist=["en"],
            lang_threshold=0.5,
            min_len=300,
            cache_dir=None,
            tmp_dir=self._cc_dir / "tmp_segments",
        )
        self._iter = iter(ds)

    def next_pair(self) -> Dict[str, str]:
        """Returns {'prompt', 'completion', 'data_source': 'common_crawl'} —
        same shape as PileStream so the rest of the pipeline is uniform."""
        for _ in range(50):
            try:
                el = next(self._iter)
            except StopIteration:
                self._init_iter()
                continue
            except Exception:
                # cc_net occasionally raises on a malformed segment — skip.
                time.sleep(0.5)
                continue
            text = el.get("raw_content", "").replace("\x00", "")
            if not text or len(text) < 200:
                continue
            doc = text[: int(self.max_prompt_len * 1.25)]
            ctx_len = int(len(doc) * random.uniform(0.25, 0.75))
            return {
                "prompt": doc[:ctx_len][: self.max_prompt_len],
                "completion": text[ctx_len:],
                "data_source": "common_crawl",
            }
        raise RuntimeError("CC stream gave up after 50 attempts")


# ---------------------------------------------------------------------------
# Source multiplexer — picks Pile vs CC on each `.next_pair()` according to
# `pile_prob`. Default 80/120 = 2/3 matches the validator's HumanDataset /
# PromptDataset ratio (detection/validator/my_datasets.py:18).
# ---------------------------------------------------------------------------
class SourceMux:
    def __init__(self, pile=None, cc: Optional[CCStream] = None,
                 pile_prob: float = 80 / 120):
        if pile is None and cc is None:
            raise ValueError("SourceMux requires at least one of pile or cc")
        self.pile = pile
        self.cc = cc
        # Force pile_prob=0 when there's no pile, =1 when there's no cc
        if pile is None:
            self.pile_prob = 0.0
        elif cc is None:
            self.pile_prob = 1.0
        else:
            self.pile_prob = pile_prob

    def next_pair(self) -> Dict[str, str]:
        if self.pile is None:
            return self.cc.next_pair()
        if self.cc is None or random.random() < self.pile_prob:
            return self.pile.next_pair()
        return self.cc.next_pair()


class DedupSource:
    """Wraps a source so the same prompt (root document) is never returned
    twice. Hashes are persisted to ``seen_path`` so A/B/C/D runs share the
    set across processes/restarts. Caller is single-threaded (the producer
    thread) so no lock is needed for the in-memory set; the file append is
    line-atomic on POSIX."""
    def __init__(self, inner, seen_path: Optional[str]):
        self.inner = inner
        self.seen_path = seen_path
        self.seen: set = set()
        self._fp = None
        self.skipped = 0
        if seen_path:
            p = Path(seen_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        h = line.strip()
                        if h:
                            self.seen.add(h)
            self._fp = open(p, "a", encoding="utf-8", buffering=1)
            print(f"Seen-roots dedup: loaded {len(self.seen)} hashes from {seen_path}")

    @staticmethod
    def _key(pair: Dict[str, str]) -> str:
        return hashlib.sha1(pair.get("prompt", "").encode("utf-8")).hexdigest()

    def next_pair(self) -> Dict[str, str]:
        # Bounded retry — if the source keeps recycling seen docs, surface it
        # rather than spinning forever.
        for _ in range(10000):
            pair = self.inner.next_pair()
            if not self.seen_path:
                return pair
            k = self._key(pair)
            if k in self.seen:
                self.skipped += 1
                continue
            self.seen.add(k)
            try:
                self._fp.write(k + "\n")
            except Exception:
                pass
            return pair
        raise RuntimeError("DedupSource: 10000 consecutive seen docs — "
                           "increase --cc-num-segments or clear --seen-roots")


# ---------------------------------------------------------------------------
# Sample builders
# ---------------------------------------------------------------------------
def build_pure_human(pair: Dict[str, str]) -> Tuple[str, List[int], str, str]:
    """The validator's HumanDataset returns el['real_completion'] as the
    text — i.e. the second half of the cut document. Labels = all 0."""
    text = pair["completion"].strip()
    labels = [0] * len(text.split())
    return text, labels, "pure_human", "none"


def build_pure_ai(pair: Dict[str, str], client: ChatClient,
                  model: Optional[str] = None) -> Tuple[str, List[int], str, str]:
    """generate_ai_data with cnt_first_human=0. Ask the model to continue
    the prompt; keep only its completion."""
    if model is None:
        model = client.pick()
    completion = client.continue_text(model, pair["prompt"])
    if not completion:
        raise RuntimeError(f"empty completion from {model}")
    labels = [1] * len(completion.split())
    return completion, labels, "pure_ai", model


def build_human_then_ai(pair: Dict[str, str], client: ChatClient,
                        model: Optional[str] = None) -> Tuple[str, List[int], str, str]:
    """generate_ai_data with the merge_prompt_text branch enabled. Final
    text is prompt + AI continuation, labels = [0]*prompt_words + [1]*rest."""
    if model is None:
        model = client.pick()
    completion = client.continue_text(model, pair["prompt"])
    if not completion:
        raise RuntimeError(f"empty completion from {model}")
    text = pair["prompt"] + completion
    cnt_first_human = len(pair["prompt"].split())
    labels = [0] * cnt_first_human + [1] * (len(text.split()) - cnt_first_human)
    return text, labels, "human_then_ai", model


def build_ai_in_middle(pair: Dict[str, str], client: ChatClient,
                       model: Optional[str] = None) -> Tuple[str, List[int], str, str]:
    """regenerated_in_the_middle — sentence-aligned begin/middle/end with
    middle replaced by Ollama. Labels = [0]*begin + [1]*middle + [0]*end."""
    full_text = pair["prompt"]
    sentences = get_sentences(full_text)
    if len(sentences) < 5:
        raise RuntimeError(f"need ≥5 sentences for ai-in-middle, got {len(sentences)}")

    lens = [len(x) for x in sentences]
    first_part = len(sentences) // 3
    second_part = 2 * len(sentences) // 3
    first_size = sum(lens[:first_part])
    second_size = sum(lens[first_part:second_part])
    third_size = sum(lens[second_part:])
    for _ in range(10):
        if first_part > 0 and first_size - lens[first_part - 1] > second_size + lens[first_part - 1]:
            first_part -= 1
        elif second_part > first_part and second_size - lens[second_part - 1] > third_size + lens[second_part - 1]:
            second_part -= 1
        elif first_part < second_part and first_part < len(lens) and first_size + lens[first_part] < second_size - lens[first_part]:
            first_part += 1
        elif second_part < len(sentences) and second_size + lens[second_part] < third_size - lens[second_part]:
            second_part += 1
        else:
            break
        first_size = sum(lens[:first_part])
        second_size = sum(lens[first_part:second_part])
        third_size = sum(lens[second_part:])

    if first_part == 0 or second_part >= len(sentences):
        raise RuntimeError("balance-thirds collapsed to a degenerate split")

    begin = "".join(sentences[:first_part])
    middle = "".join(sentences[first_part:second_part])
    end = "".join(sentences[second_part:])

    middle_stripped = middle.rstrip()
    diff = len(middle) - len(middle_stripped)
    if diff > 0:
        end = middle[-diff:] + end
    middle = middle_stripped

    if model is None:
        model = client.pick()
    summary = client.chat(model, [
        {"role": "system", "content": random.choice(SUMMARY_PROMPTS)},
        {"role": "user", "content": middle},
    ])
    middle_size = max(1, len(middle.split()))
    generated = client.chat(model, [
        {"role": "system", "content": random.choice(GENERATION_PROMPTS) +
            f" The middle should be about {middle_size} words long"},
        {"role": "user", "content": f"begin: {begin}\nend: {end}\nsummary: {summary}"},
    ]).strip()

    if not generated:
        raise RuntimeError(f"empty middle from {model}")

    text = begin + generated + end
    labels = ([0] * len(begin.split())
              + [1] * len(generated.split())
              + [0] * len(end.split()))
    return text, labels, "ai_in_middle", model


def build_human_in_middle(pair: Dict[str, str], client: ChatClient,
                          model: Optional[str] = None
                          ) -> Tuple[str, List[int], str, str]:
    """ai -> human -> ai pattern (2 seams: 1->0 then 0->1).

    The structural inverse of ai_in_middle. Splits the source doc into
    3 sentence-balanced thirds; keeps the MIDDLE as human and
    REGENERATES the FIRST and LAST as AI segments. Both AI segments
    are generated via summary+generate so they cover the same content
    as the original thirds (preserves topic coherence across the
    document) but with a stylistic discontinuity at each seam.

    Cost: 4 API calls per sample (vs ai_in_middle's 2). Pair with
    --no-subsample -- otherwise subsample_words collapses the 1->0->1
    pattern to a single-class window because its recursive 0->1 chop
    falls AFTER the 1->0 boundary.

    Labels = [1]*first_ai + [0]*human_middle + [1]*last_ai."""
    full_text = pair["prompt"]
    sentences = get_sentences(full_text)
    if len(sentences) < 5:
        raise RuntimeError(
            f"need >=5 sentences for human-in-middle, got {len(sentences)}"
        )

    ranges = _balance_n_segments(sentences, 3)
    raw_segments = ["".join(sentences[s:e]) for s, e in ranges]

    human_kept = raw_segments[1].strip()
    if not human_kept:
        raise RuntimeError("human_in_middle: middle segment empty after strip")

    if model is None:
        model = client.pick()

    # Regenerate FIRST segment (becomes the leading AI block).
    first_original = raw_segments[0].strip()
    if not first_original:
        raise RuntimeError("human_in_middle: first segment empty after strip")
    first_summary = client.chat(model, [
        {"role": "system", "content": random.choice(SUMMARY_PROMPTS)},
        {"role": "user",   "content": first_original},
    ])
    first_target_words = max(1, len(first_original.split()))
    first_generated = client.chat(model, [
        {"role": "system", "content": random.choice(GENERATION_PROMPTS) +
            f" The middle should be about {first_target_words} words long"},
        {"role": "user",
         "content": f"begin: \nend: {human_kept} {raw_segments[2]}\n"
                    f"summary: {first_summary}"},
    ]).strip()
    if not first_generated:
        raise RuntimeError(f"empty human_in_middle first segment from {model}")

    # Regenerate LAST segment (becomes the trailing AI block). Use the
    # JUST-GENERATED first AI segment as begin context, so the trailing
    # AI sees the "voice" of its sibling AI.
    last_original = raw_segments[2].strip()
    if not last_original:
        raise RuntimeError("human_in_middle: last segment empty after strip")
    last_summary = client.chat(model, [
        {"role": "system", "content": random.choice(SUMMARY_PROMPTS)},
        {"role": "user",   "content": last_original},
    ])
    last_target_words = max(1, len(last_original.split()))
    last_generated = client.chat(model, [
        {"role": "system", "content": random.choice(GENERATION_PROMPTS) +
            f" The middle should be about {last_target_words} words long"},
        {"role": "user",
         "content": f"begin: {first_generated} {human_kept}\nend: \n"
                    f"summary: {last_summary}"},
    ]).strip()
    if not last_generated:
        raise RuntimeError(f"empty human_in_middle last segment from {model}")

    # Word-level concat (same rationale as build_multi_seam: avoids
    # whitespace-mismatch run-ons between AI/human segments).
    all_words: List[str] = []
    all_labels: List[int] = []
    for seg, label in (
        (first_generated, 1),
        (human_kept,      0),
        (last_generated,  1),
    ):
        words = seg.split()
        if not words:
            raise RuntimeError("human_in_middle segment produced 0 words")
        all_words.extend(words)
        all_labels.extend([label] * len(words))

    text = " ".join(all_words)
    return text, all_labels, "human_in_middle", model


# ---------------------------------------------------------------------------
# Multi-seam (3+ transitions) -- alternating h/ai chunks
# ---------------------------------------------------------------------------
def _balance_n_segments(sentences: List[str], n: int) -> List[Tuple[int, int]]:
    """Split `sentences` into `n` ranges of roughly equal CHARACTER count.

    Returns a list of (start_idx, end_idx) where sentences[start:end]
    forms each chunk. Raises RuntimeError on degenerate splits (any
    chunk would be empty). Used by build_multi_seam; ai_in_middle has
    its own iterative-balance routine kept for backward compat."""
    if n < 2:
        raise ValueError(f"n must be >=2, got {n}")
    if len(sentences) < n:
        raise RuntimeError(
            f"need at least {n} sentences for {n}-segment balance, "
            f"got {len(sentences)}"
        )
    lens = [len(s) for s in sentences]
    total = sum(lens)
    if total == 0:
        raise RuntimeError("empty sentences")

    # Greedy: walk and place a cut when the running char total crosses
    # the next 1/n boundary. This produces roughly equal chunks; the
    # last chunk absorbs whatever rounding leaves over.
    cuts: List[int] = []
    running = 0
    for i, ln in enumerate(lens):
        running += ln
        if len(cuts) < n - 1:
            target = total * (len(cuts) + 1) / n
            if running >= target:
                cuts.append(i + 1)
    if len(cuts) < n - 1:
        raise RuntimeError(
            f"could not place {n - 1} cuts in {len(sentences)} sentences"
        )

    boundaries = [0] + cuts + [len(sentences)]
    ranges = list(zip(boundaries[:-1], boundaries[1:]))
    for s, e in ranges:
        if s == e:
            raise RuntimeError(
                f"degenerate multi_seam segment [{s}:{e}] -- chunks too uneven"
            )
    return ranges


def build_multi_seam(pair: Dict[str, str], client: ChatClient,
                     model: Optional[str] = None,
                     n_segments: int = 4) -> Tuple[str, List[int], str, str]:
    """h -> ai -> h -> ai ... pattern with `n_segments` alternating chunks.
    Even-indexed chunks are kept as the original human text; odd-indexed
    chunks are LLM-regenerated via the same summary+generate pattern as
    ai_in_middle.

    n_segments=4 (default) -> 3 transitions (h-ai-h-ai), ends in AI.
    n_segments=5           -> 4 transitions (h-ai-h-ai-h), ends in human.
    n_segments=6           -> 5 transitions, etc.

    n_segments<4 is rejected because n=3 is just ai_in_middle and n<3
    is single-transition (use human_then_ai instead). Cost per sample:
    ceil(n_segments/2) AI chunks * 2 API calls (summary + generate),
    so n=4 is 4 calls, n=6 is 6 calls. Pair this with
    --reasoning-effort=none on frontier models or you'll burn hours on
    chain-of-thought.

    Word-level concatenation (rather than character-level like
    ai_in_middle) ensures len(text.split()) == len(labels) regardless
    of how the LLM trims its output, which matters because subsample
    + length-sanity assert downstream rejects mismatches."""
    if n_segments < 4:
        raise ValueError(
            f"multi_seam needs n_segments>=4 (n=3 == ai_in_middle); got {n_segments}"
        )

    full_text = pair["prompt"]
    sentences = get_sentences(full_text)
    min_sents = n_segments + 2
    if len(sentences) < min_sents:
        raise RuntimeError(
            f"need >={min_sents} sentences for {n_segments}-segment "
            f"multi_seam, got {len(sentences)}"
        )

    ranges = _balance_n_segments(sentences, n_segments)
    raw_segments = ["".join(sentences[s:e]) for s, e in ranges]

    if model is None:
        model = client.pick()

    # Regenerate odd-indexed segments using surrounding human context.
    new_segments: List[str] = []
    for i, seg in enumerate(raw_segments):
        if i % 2 == 0:
            new_segments.append(seg)
            continue
        seg_stripped = seg.strip()
        if not seg_stripped:
            raise RuntimeError(f"multi_seam segment {i} is empty after strip")

        # Context = all surrounding HUMAN segments. We use the
        # already-finalized new_segments[:i] for the "begin" half so
        # the LLM sees the actual neighbouring human text it's bridging
        # into, even though the immediately preceding new_segments[i-1]
        # is an unfinalized placeholder for the next iteration.
        before = "".join(new_segments[:i]).rstrip()
        after  = "".join(raw_segments[i + 1:]).lstrip()

        summary = client.chat(model, [
            {"role": "system", "content": random.choice(SUMMARY_PROMPTS)},
            {"role": "user",   "content": seg_stripped},
        ])
        target_words = max(1, len(seg_stripped.split()))
        generated = client.chat(model, [
            {"role": "system", "content": random.choice(GENERATION_PROMPTS) +
                f" The middle should be about {target_words} words long"},
            {"role": "user",   "content": f"begin: {before}\nend: {after}\nsummary: {summary}"},
        ]).strip()

        if not generated:
            raise RuntimeError(f"empty multi_seam middle segment {i} from {model}")
        new_segments.append(generated)

    # Build text + labels at word level. We deliberately " ".join the
    # word arrays (instead of "".join the segments like ai_in_middle)
    # because the LLM's `.strip()` removes the trailing whitespace the
    # next human segment needs as its leading separator, which would
    # otherwise produce a word run-on at every h<->ai boundary and
    # break the labels==len(text.split()) sanity check.
    all_words: List[str] = []
    all_labels: List[int] = []
    for i, seg in enumerate(new_segments):
        words = seg.split()
        if not words:
            raise RuntimeError(f"multi_seam segment {i} produced 0 words")
        all_words.extend(words)
        label = 1 if (i % 2 == 1) else 0
        all_labels.extend([label] * len(words))

    text = " ".join(all_words)
    return text, all_labels, "multi_seam", model


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
# Even 25/25/25/25 across the four sample types so the trained classifier
# sees each label shape equally often. (The validator's actual mix is
# 25/25/40/10 -- VALIDATOR_SAMPLE_MIX below preserves that for reference and
# can be passed via build_task_plan(..., mix=VALIDATOR_SAMPLE_MIX) if you
# ever want a validator-faithful run instead.)
SAMPLE_MIX = [
    ("pure_human",     0.25),
    ("pure_ai",        0.25),
    ("human_then_ai",  0.25),
    ("ai_in_middle",   0.25),
    # multi_seam (3+ transitions) is opt-in only via --sample-types
    # because it's expensive (>=4 API calls/sample) and the validator's
    # default windows mostly produce 0/1/2-transition samples. Use it
    # to teach the v3 CRF's P(1->0)>=0.05 floor to actually fire.
    ("multi_seam",     0.00),
    # human_in_middle: 1->0->1 pattern (ai -> human -> ai), the
    # structural inverse of ai_in_middle. Opt-in via --sample-types
    # because it costs 4 API calls per sample and most validator
    # windows are 0->1 / 1->0, not 1->0->1. Pair with --no-subsample.
    ("human_in_middle", 0.00),
]

VALIDATOR_SAMPLE_MIX = [
    ("pure_human",     0.25),
    ("pure_ai",        0.25),
    ("human_then_ai",  0.40),
    ("ai_in_middle",   0.10),
]


def pick_sample_type() -> str:
    r = random.random()
    cum = 0.0
    for name, p in SAMPLE_MIX:
        cum += p
        if r <= cum:
            return name
    return SAMPLE_MIX[-1][0]


def count_existing_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        # subtract 1 for the header row
        return max(0, sum(1 for _ in f) - 1)


# ---------------------------------------------------------------------------
# Parallel pipeline — Pile producer thread + worker pool.
# ---------------------------------------------------------------------------
def build_task_plan(n_total: int, models: List[str],
                    mix=SAMPLE_MIX) -> List[Tuple[str, str]]:
    """Deterministic task list of (sample_type, model_name) tuples whose:
       - sample-type counts hit the validator's 25/25/40/10 mix exactly,
       - per-type model assignment is round-robin across `models` so each
         model is used roughly equally within each sample-type bucket.
    The list is shuffled before returning so workers don't pound one model
    in a burst (helps providers that rate-limit per-model)."""
    counts = {t: round(n_total * p) for t, p in mix}
    diff = n_total - sum(counts.values())
    if diff != 0:
        biggest = max(counts, key=counts.get)
        counts[biggest] += diff

    tasks: List[Tuple[str, str]] = []
    for t, n in counts.items():
        if t == "pure_human":
            tasks.extend([(t, "none")] * n)
            continue
        for i in range(n):
            tasks.append((t, models[i % len(models)]))

    random.shuffle(tasks)
    return tasks


def _pile_producer(source, q: "queue.Queue",
                   stop_event: threading.Event):
    """Single-thread source fetcher — keeps `q` topped up so workers never
    block on dataset I/O. ``source`` is a ``SourceMux`` (or any object with
    a ``next_pair()`` method). Pile / CC streamers aren't thread-safe, so
    all next_pair() calls must come from this one thread."""
    while not stop_event.is_set():
        try:
            pair = source.next_pair()
        except Exception:
            time.sleep(delay)
            delay = min(60.0, delay * 2)
            continue
        # Block until there's space; quit early if the run is stopping.
        while not stop_event.is_set():
            try:
                q.put(pair, timeout=1)
                break
            except queue.Full:
                continue


def _format_http_error(e: requests.HTTPError) -> str:
    code = getattr(e.response, "status_code", 0)
    body = ""
    try:
        body = e.response.text[:300]
    except Exception:
        pass
    return f"HTTP {code}: {body}".strip()


# Errors that mean "the connection itself dropped" rather than "the server
# said no". Retried separately from API-level errors with longer backoff so
# a flaky internet (5-minute outages, transient WiFi drops) doesn't burn
# through the regular retry budget.
_NETWORK_EXC: Tuple[type, ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _process_one(task: Tuple[str, str],
                 client: ChatClient,
                 pile_queue: "queue.Queue",
                 args,
                 max_retries: int = 3) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Generate one CSV-ready row for `task`. Retries on transient errors.
    Returns (row, None) on success, (None, error_str) on failure so the
    caller can surface the actual cause instead of a silent drop.

    Network errors (ConnectionError, Timeout, ChunkedEncodingError) are
    retried with their own larger budget (`args.network_max_retries`,
    default 20, exponential backoff capped at 5 minutes) so a flaky
    internet doesn't waste API-level retries."""
    sample_type, target_model = task
    model_for_call = None if sample_type == "pure_human" else target_model

    last_err: Optional[str] = None
    network_attempts = 0
    network_max = getattr(args, "network_max_retries", 20)
    fresh_pair_retries = getattr(args, "fresh_pair_retries", 3)

    # Outer loop: each iteration uses a fresh Pile/CC pair. We loop here on
    # *structural* failures (doc too short for ai_in_middle, balance-thirds
    # collapse, length-sanity fail after subsample). Same-pair API retries
    # happen in the inner while-loop below.
    pair: Optional[Dict[str, str]] = None
    for pair_iter in range(fresh_pair_retries + 1):
        try:
            pair = pile_queue.get(timeout=120)
        except Exception:
            return None, "pile queue timeout"

        # Cheap pre-screen for ai_in_middle / human_in_middle / multi_seam:
        # skip pairs whose prompt won't tokenize into enough sentences.
        # This kills the most common drop cause without burning an API
        # call. We only pre-screen while we still have pair budget left
        # -- on the final attempt we accept whatever we get and let the
        # downstream pure_human fallback handle a too-short doc.
        if pair_iter < fresh_pair_retries:
            min_sents = None
            if sample_type == "ai_in_middle":
                min_sents = 5
            elif sample_type == "human_in_middle":
                min_sents = 5
            elif sample_type == "multi_seam":
                min_sents = getattr(args, "multi_seam_segments", 4) + 2
            if min_sents is not None:
                try:
                    if len(get_sentences(pair["prompt"])) < min_sents:
                        last_err = f"pre-screen: doc has <{min_sents} sentences"
                        continue
                except Exception:
                    pass

        attempt = 0
        retry_with_fresh_pair = False
        while attempt < max_retries:
            try:
                if sample_type == "pure_human":
                    text, labels, st, model_name = build_pure_human(pair)
                elif sample_type == "pure_ai":
                    text, labels, st, model_name = build_pure_ai(pair, client, model=model_for_call)
                elif sample_type == "human_then_ai":
                    text, labels, st, model_name = build_human_then_ai(pair, client, model=model_for_call)
                elif sample_type == "human_in_middle":
                    # Same fresh-pair-then-pure_human fallback as
                    # ai_in_middle, with structural pattern matching the
                    # error messages raised by build_human_in_middle.
                    try:
                        text, labels, st, model_name = build_human_in_middle(
                            pair, client, model=model_for_call,
                        )
                    except RuntimeError as struct_err:
                        msg = str(struct_err)
                        structural = (
                            "sentences for human-in-middle" in msg
                            or "human_in_middle" in msg and "empty" in msg
                            or "human_in_middle segment produced 0 words" in msg
                        )
                        if structural:
                            last_err = msg
                            if pair_iter < fresh_pair_retries:
                                retry_with_fresh_pair = True
                                break
                            text, labels, st, model_name = build_pure_human(pair)
                        else:
                            raise
                elif sample_type == "multi_seam":
                    # Same fresh-pair-then-pure_human fallback as ai_in_middle.
                    # The structural patterns are SOURCE-DOCUMENT defects
                    # (not enough sentences, char-length-balance collapse,
                    # whitespace-only segment). On those a different Pile
                    # pair would help, so we recycle.
                    #
                    # API failures inside build_multi_seam ("empty middle
                    # segment", "openai-compat null content") are NOT
                    # structural -- they're transient model output issues.
                    # Re-raise them so the outer retry path retries the
                    # SAME pair with a different model rather than wasting
                    # the slot as pure_human.
                    try:
                        text, labels, st, model_name = build_multi_seam(
                            pair, client, model=model_for_call,
                            n_segments=getattr(args, "multi_seam_segments", 4),
                        )
                    except RuntimeError as struct_err:
                        msg = str(struct_err)
                        structural = (
                            "sentences for" in msg                       # too few sentences
                            or "could not place" in msg                  # balance failed
                            or "degenerate multi_seam segment" in msg    # empty range
                            or "is empty after strip" in msg             # AI seg whitespace
                            or "produced 0 words" in msg                 # human seg whitespace
                            or "empty sentences" in msg                  # nltk gave nothing
                            # NOTE: "empty multi_seam middle segment ... from <model>"
                            # is an API failure (model produced no output) and is
                            # deliberately NOT in this list. It re-raises and falls
                            # to the outer transient-retry handler below.
                        )
                        if structural:
                            last_err = msg
                            if pair_iter < fresh_pair_retries:
                                retry_with_fresh_pair = True
                                break
                            text, labels, st, model_name = build_pure_human(pair)
                        else:
                            raise
                else:
                    # ai_in_middle. Pre-screen above keeps most pairs valid,
                    # but a degenerate balance-thirds split can still trigger
                    # here. On structural failure, request a fresh pair
                    # (outer loop) rather than dropping. Final fallback is
                    # pure_human from the same pair so the slot isn't lost.
                    try:
                        text, labels, st, model_name = build_ai_in_middle(
                            pair, client, model=model_for_call,
                        )
                    except RuntimeError as struct_err:
                        msg = str(struct_err)
                        if ("sentences for ai-in-middle" in msg
                                or "balance-thirds" in msg):
                            last_err = msg
                            if pair_iter < fresh_pair_retries:
                                retry_with_fresh_pair = True
                                break  # → outer loop, new pair
                            text, labels, st, model_name = build_pure_human(pair)
                        else:
                            raise

                if not args.no_subsample:
                    text, labels = subsample_words(text, labels)
                if len(labels) != len(text.split()) or len(labels) < 20:
                    raise RuntimeError(
                        f"length sanity failed ({len(labels)} vs {len(text.split())})"
                    )

                augmented = False
                if not args.no_augment:
                    text, labels = augment(text, labels, per_word_p=args.augment_rate)
                    augmented = True
                    if len(labels) != len(text.split()):
                        raise RuntimeError("augmentation broke word count")

                return {
                    "text": text,
                    "segmentation_labels": json.dumps(labels),
                    "data_source": pair["data_source"],
                    "sample_type": st,
                    "model_name": model_name,
                    "n_words": len(labels),
                    "augmented": "true" if augmented else "false",
                }, None
            except requests.HTTPError as e:
                last_err = _format_http_error(e)
                code = getattr(e.response, "status_code", 0)
                # 429 = rate limit; 5xx = server error → backoff and retry.
                if code == 429 or 500 <= code < 600:
                    time.sleep(min(60, 2 ** (attempt + 1)))
                    attempt += 1
                    continue
                # 4xx other than 429 = our fault, don't burn retries
                return None, last_err
            except _NETWORK_EXC as e:
                # Connection-level blip — exponential backoff up to 5 min,
                # with its own retry budget so unstable internet doesn't
                # consume the API-level retry slots.
                last_err = f"{type(e).__name__}: {e}"
                if network_attempts >= network_max:
                    return None, last_err
                time.sleep(min(300, 5 * (2 ** network_attempts)))
                network_attempts += 1
                # Don't increment `attempt` -- network blips don't count.
                continue
            except RuntimeError as e:
                msg = str(e)
                last_err = f"{type(e).__name__}: {e}"

                # Transient API issues: a specific MODEL is misbehaving on
                # this prompt (returning null content, empty output, hitting
                # a moderation filter, etc.). With --even-mix the same
                # model would be reused on retry and almost certainly fail
                # the same way, so swap to a fresh model from the rotation
                # before retrying. This is the single biggest lever for
                # multi_seam success rate -- with 4 API calls per sample, a
                # single bad model in the rotation can sink ~5% of samples
                # without the swap.
                transient_api = (
                    "openai-compat response" in msg            # null content / no choices
                    or "empty multi_seam middle segment" in msg  # model produced empty
                    or "no output_text" in msg                 # Responses API empty
                )
                if transient_api:
                    model_for_call = client.pick()
                    last_err += f"  (next attempt: model={model_for_call})"

                # Pair-specific failures: same pair will keep failing, so
                # fetch a new one if we still have budget. We also bucket
                # "no output_text" here because a model refusal/empty
                # response on a particular prompt rarely changes on
                # retry but often clears with a different prompt.
                pair_specific = (
                    "length sanity failed" in msg
                    or "augmentation broke" in msg
                    or "no output_text" in msg
                    or "empty completion" in msg
                    or "empty middle" in msg
                )
                if pair_specific and pair_iter < fresh_pair_retries:
                    retry_with_fresh_pair = True
                    break
                time.sleep(1 + attempt)
                attempt += 1
                continue
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(1 + attempt)
                attempt += 1
                continue

        if not retry_with_fresh_pair:
            # Inner loop exhausted (max_retries on this pair) without a
            # fresh-pair signal. Don't keep grabbing new pairs forever --
            # each fresh pair has already had max_retries attempts above.
            break

    return None, last_err or "unknown error"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-samples", type=int, default=10000,
                    help="Total samples to produce (counts existing rows in --output).")
    ap.add_argument("--provider",
                    choices=tuple(PROVIDER_PRESETS.keys()),
                    default=None,
                    help="One-flag provider preset. Sets --api-mode, "
                         "--ollama-url, and a curated default --ollama-models "
                         "in one shot. Choices:\n"
                         "  openrouter   -> openai mode + https://openrouter.ai/api\n"
                         "                  + curated paid model list "
                         "(reads $OPENROUTER_API_KEY)\n"
                         "  concentrate  -> responses mode + https://api.concentrate.ai\n"
                         "                  + VALIDATOR_MODELS "
                         "(reads $CONCENTRATE_API_KEY)\n"
                         "  ollama-cloud -> ollama mode + https://ollama.com "
                         "(reads $OLLAMA_API_KEY)\n"
                         "  local        -> ollama mode + http://127.0.0.1:11434\n"
                         "Any explicit --api-mode / --ollama-url / "
                         "--ollama-models / --ollama-token still wins -- "
                         "the preset only fills in flags you didn't pass.")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "https://ollama.com"),
                    help="Base URL for the chat API. Default: https://ollama.com "
                         "(Ollama Cloud), which hosts the full validator pool including "
                         "70B+/100B+ models. For OpenAI-compatible providers (Together, "
                         "OpenRouter, Groq, vLLM) pass their endpoint and --api-mode openai. "
                         "For a local Ollama smoke test pass http://127.0.0.1:11434. "
                         "Falls back to $OLLAMA_URL.")
    ap.add_argument("--ollama-token",
                    default=os.environ.get("CONCENTRATE_API_KEY")
                            or os.environ.get("OPENROUTER_API_KEY")
                            or os.environ.get("OLLAMA_API_KEY"),
                    help="Bearer API key. Sent as `Authorization: Bearer <token>`. "
                         "Required for Concentrate / Ollama Cloud / OpenRouter / "
                         "most paid hosted services. Falls back to "
                         "$CONCENTRATE_API_KEY first, then $OPENROUTER_API_KEY, "
                         "then $OLLAMA_API_KEY. (The --provider preset will "
                         "still override this with the provider-specific env "
                         "var when set, so cross-contamination is prevented.)")
    ap.add_argument("--api-mode", choices=("ollama", "openai", "responses"),
                    default="ollama",
                    help="Wire protocol. 'ollama' uses /api/chat (works for local "
                         "ollama serve and for Ollama Cloud). 'openai' uses "
                         "/v1/chat/completions (Together, OpenRouter, Groq, vLLM, "
                         "etc.). 'responses' uses /v1/responses/ in OpenAI's newer "
                         "Responses-API shape (Concentrate at https://api.concentrate.ai).")
    ap.add_argument("--ollama-models", default=None,
                    help="Comma-separated list of model names to rotate through. "
                         "Default depends on --ollama-url: for localhost, the script "
                         "queries /api/tags and uses (pulled & validator pool), "
                         "falling back to whatever is pulled. For remote URLs, the "
                         "default is the deduplicated 27-model validator pool. Override to "
                         "pin a specific list, e.g. 'meta-llama/Llama-3.3-70B-Instruct-Turbo,...'.")
    ap.add_argument("--skip-health-check", action="store_true",
                    help="Don't probe /api/tags or /v1/models before starting. Some "
                         "hosted services don't expose model listing.")
    ap.add_argument("--output", default="data/train.csv",
                    help="Append-only CSV. Resumable.")
    ap.add_argument("--max-prompt-len", type=int, default=1500,
                    help="Max chars per Pile prompt (matches validator).")
    ap.add_argument("--no-augment", action="store_true",
                    help="Disable typo/case/punct attacks.")
    ap.add_argument("--augment-rate", type=float, default=0.07,
                    help="Per-word probability of augmentation (default 0.07).")
    ap.add_argument("--no-subsample", action="store_true",
                    help="Skip the 35-350 word window step (keep full text).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Reproducibility seed (default: time-based).")
    ap.add_argument("--ollama-timeout", type=int, default=300,
                    help="Per-call timeout (s). Increase for slow hosted services or "
                         "large models. Ollama Cloud: 300 is usually enough; vLLM with "
                         "70B+ uncached: try 600.")
    ap.add_argument("--report-every", type=int, default=50,
                    help="Print rolling stats every N samples.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of concurrent API workers. Default is 1 for "
                         "localhost (Ollama serves one request per model at a time, "
                         "so >1 just queues with no benefit on a single-GPU host) "
                         "and 8 for remote URLs. Increase to 16-32 on hosted services "
                         "that allow it; decrease if you hit 429 rate-limits.")
    ap.add_argument("--even-mix", action="store_true", default=True,
                    help="Use deterministic 25/25/40/10 sample-type counts and "
                         "round-robin model assignment per type, so the output is "
                         "evenly distributed by both shape and model. ON by default. "
                         "Use --no-even-mix to fall back to fully random sampling.")
    ap.add_argument("--no-even-mix", action="store_false", dest="even_mix",
                    help="Disable --even-mix.")
    ap.add_argument("--pile-buffer", type=int, default=64,
                    help="Number of source pairs the producer thread keeps pre-fetched "
                         "for the workers. Default 64.")
    ap.add_argument("--enable-cc", action="store_true", default=True,
                    help="Mix Common Crawl into the source pool (uses cc_net). "
                         "ON by default — the validator scores you with both Pile "
                         "and Common Crawl, so training data should match. Use "
                         "--no-cc to disable (Pile only) — useful for smoke tests "
                         "since CC is slower and downloads ~hundreds of MB.")
    ap.add_argument("--no-cc", action="store_false", dest="enable_cc",
                    help="Disable Common Crawl, use Pile only.")
    ap.add_argument("--pile-prob", type=float, default=80 / 120,
                    help="Probability of pulling from Pile vs CC. Default 80/120 = "
                         "0.667 — matches the validator's HumanDataset / PromptDataset "
                         "ratio. Set to 1.0 for Pile only, 0.0 for CC only.")
    ap.add_argument("--cc-num-segments", type=int, default=10,
                    help="Number of CC WET segments cc_net will sample from per "
                         "stream init. Default 10. Larger = more diversity but more "
                         "disk + bandwidth.")
    ap.add_argument("--cc-root", default=None,
                    help="Path to the cc_net data directory (containing bin/lid.bin, "
                         "data/lm_sp/, collinfo.json, tmp_segments/). Default: "
                         "<repo>/cc_net.")
    ap.add_argument("--sample-types", default=None,
                    help="Comma-separated list of sample types to generate. "
                         "Default uses the 25/25/40/10 mix across the first "
                         "four types (pure_human, pure_ai, human_then_ai, "
                         "ai_in_middle). multi_seam and human_in_middle are "
                         "opt-in only via this flag.\n"
                         "  --sample-types human_then_ai\n"
                         "    -> 100%% human_then_ai (1 seam: 0->1)\n"
                         "  --sample-types ai_in_middle\n"
                         "    -> 100%% ai_in_middle (2 seams: 0->1->0)\n"
                         "  --sample-types human_in_middle\n"
                         "    -> 100%% human_in_middle (2 seams: 1->0->1)\n"
                         "  --sample-types ai_in_middle,human_in_middle\n"
                         "    -> 50/50 mix of 0->1->0 and 1->0->1\n"
                         "  --sample-types multi_seam\n"
                         "    -> 100%% multi_seam (3+ seams @ default n=4)")
    ap.add_argument("--seen-roots", default="data/seen_roots.txt",
                    help="Path to a shared file of CC-prompt SHA1 hashes already "
                         "consumed. Pairs whose prompt-hash is in this file are "
                         "skipped, and successfully-emitted roots are appended. "
                         "Use this to keep runs A/B/C/D from sharing the same "
                         "human source documents. Pass '' to disable.")
    ap.add_argument("--multi-seam-segments", type=int, default=4,
                    help="Number of alternating h/ai chunks for multi_seam "
                         "samples. n=4 (default) -> 3 transitions; n=5 -> 4; "
                         "n=6 -> 5. Each AI chunk costs 2 API calls "
                         "(summary + generate), so cost scales as ceil(n/2)*2. "
                         "Requires >=n+2 sentences in the source doc. n<4 is "
                         "rejected (use --sample-types ai_in_middle for n=3).")
    args = ap.parse_args()

    # Apply --provider preset. Each flag is filled in ONLY if the user
    # didn't pass it explicitly on the command line (we check sys.argv
    # rather than comparing against argparse defaults because some
    # defaults are env-var-derived and we can't tell "default" from
    # "user passed the same value as default" any other way).
    if args.provider is not None:
        preset = PROVIDER_PRESETS[args.provider]
        cli_flags = set()
        for tok in sys.argv[1:]:
            if tok.startswith("--"):
                cli_flags.add(tok.split("=", 1)[0])

        def _passed(*flags) -> bool:
            return any(f in cli_flags for f in flags)

        if not _passed("--api-mode"):
            args.api_mode = preset["api_mode"]
        if not _passed("--ollama-url"):
            args.ollama_url = preset["ollama_url"]
        if not _passed("--ollama-models") and preset["default_models"]:
            args.ollama_models = ",".join(preset["default_models"])
        # Token resolution: when --ollama-token wasn't explicit, ALWAYS
        # prefer the provider-specific env var. The argparse default may
        # have already set args.ollama_token from a different env var
        # (e.g. CONCENTRATE_API_KEY before OPENROUTER_API_KEY), and that
        # cross-contamination would silently send the wrong key to the
        # provider -- which 401s with a misleading "Missing Authentication
        # header" error from OpenRouter / Cloudflare. Override it here so
        # the preset's intent always wins.
        if not _passed("--ollama-token"):
            preset_token: Optional[str] = None
            for env_var in preset["token_env_vars"]:
                v = os.environ.get(env_var)
                if v:
                    preset_token = v
                    break
            if preset_token:
                args.ollama_token = preset_token
            elif preset["token_env_vars"]:
                # The preset expects an env var but none was set. Wipe
                # the argparse default so we don't accidentally send a
                # wrong-provider key. The is_local check below will
                # then bail with a clean, actionable message instead of
                # letting every API call 401.
                if args.ollama_token:
                    print(f"WARNING: --provider {args.provider!r} expected "
                          f"${'/$'.join(preset['token_env_vars'])} but none "
                          f"are set. Discarding cross-provider env-var key "
                          f"that argparse loaded by default.")
                args.ollama_token = None
        print(f"Provider preset '{args.provider}': "
              f"api_mode={args.api_mode}  url={args.ollama_url}")

    if args.seed is not None:
        random.seed(args.seed)

    is_local = _is_localhost(args.ollama_url)

    # OpenRouter recommends `HTTP-Referer` + `X-Title` headers so calls
    # appear (anonymously aggregated) on the model leaderboard. Harmless
    # on every other provider -- they ignore unknown headers. Detected
    # by URL substring so it kicks in even without --provider openrouter.
    extra_headers: Dict[str, str] = {}
    if "openrouter.ai" in args.ollama_url.lower():
        extra_headers["HTTP-Referer"] = os.environ.get(
            "OPENROUTER_REFERER", "https://github.com/hisorhikaneko92-create/llm-detection",
        )
        extra_headers["X-Title"] = os.environ.get(
            "OPENROUTER_APP_NAME", "sn32-llm-detection-trainer",
        )

    # Cloud / hosted endpoints all require a Bearer token. Bail early with a
    # clear message instead of letting every API call 401.
    if not is_local and not args.ollama_token:
        sys.exit(
            "No API key set, but --ollama-url points to a remote endpoint "
            f"({args.ollama_url}).\n"
            "EASIEST -- pick a provider preset:\n"
            "  $env:OPENROUTER_API_KEY = '<key>';  python scripts/build_training_dataset.py --provider openrouter ...\n"
            "  $env:CONCENTRATE_API_KEY = '<key>'; python scripts/build_training_dataset.py --provider concentrate ...\n"
            "  $env:OLLAMA_API_KEY = '<key>';     python scripts/build_training_dataset.py --provider ollama-cloud ...\n"
            "Or set the env var manually and pass the matching --api-mode/--ollama-url:\n"
            "  $env:CONCENTRATE_API_KEY -- pair with --api-mode responses --ollama-url https://api.concentrate.ai\n"
            "  $env:OPENROUTER_API_KEY  -- pair with --api-mode openai    --ollama-url https://openrouter.ai/api\n"
            "  $env:OLLAMA_API_KEY      -- pair with --api-mode ollama    --ollama-url https://ollama.com\n"
            "For a local smoke test instead, use --provider local."
        )

    # Worker default: localhost → 1 (Ollama serializes), remote → 8.
    if args.workers is None:
        args.workers = 1 if is_local else 8

    # Resolve the model rotation. We always try to probe the provider's
    # model listing first (when --ollama-models wasn't set), so we never
    # ship requests for models the provider doesn't host.
    #
    #   1. --ollama-models was explicitly passed → split and use as-is.
    #      No probe — user knows what their provider offers.
    #   2. Otherwise:
    #        a. Probe /api/tags (or /v1/models for openai mode).
    #        b. Prefer intersection with the deduplicated validator pool,
    #           so the data is validator-faithful when possible.
    #        c. If 0 validator-pool models are hosted, fall back to the
    #           full provider listing — generated data will be diverse
    #           but flavored by whatever the provider hosts (e.g. Ollama
    #           Cloud's "*-cloud" models, OpenRouter's catalog, etc.).
    #        d. If both are empty, exit with actionable guidance.
    auto_detected = False
    if args.ollama_models is not None:
        models = [m.strip() for m in args.ollama_models.split(",") if m.strip()]
        if not models:
            sys.exit("--ollama-models cannot be empty")
    else:
        auto_detected = True
        try:
            probe = ChatClient(args.ollama_url, [], args.ollama_token,
                               api_mode=args.api_mode,
                               chat_timeout=args.ollama_timeout,
                               extra_headers=extra_headers)
            listed = probe.health_check()
        except Exception as e:
            if is_local:
                sys.exit(
                    f"Could not reach local Ollama at {args.ollama_url}: {e}\n"
                    "Start it with `ollama serve` (or open the Ollama app), then retry."
                )
            print(f"NOTE: provider {args.ollama_url} doesn't expose a model "
                  f"listing endpoint ({e}). Falling back to OPENROUTER_MODELS.")
            listed = []

        # Match OPENROUTER_MODELS against the provider listing. ":latest" is
        # an alias for the no-tag form (so "llama3.2" matches "llama3.2:latest").
        listed_set = set(listed)
        listed_set.update({p.rsplit(":latest", 1)[0] for p in listed if p.endswith(":latest")})
        intersect = []
        seen = set()
        for m in OPENROUTER_MODELS:
            if (m in listed_set or f"{m}:latest" in listed_set) and m not in seen:
                intersect.append(m)
                seen.add(m)

        if intersect:
            models = intersect
            print(f"Resolved {len(models)} OPENROUTER_MODELS hosted by the provider: {models}")
            if is_local:
                missing = [m for m in LOCAL_GPU_MODELS if m not in models]
                if missing:
                    print("To expand the local rotation, pull more GPU-friendly models:")
                    print("  ollama pull " + " ".join(missing))
        elif listed:
            # The listing returned entries but none matched OPENROUTER_MODELS.
            # Concentrate's /v1/models returns DISPLAY names ("GPT 5.4 Pro",
            # "Llama 3 70B Instruct" with spaces) -- those aren't valid API
            # IDs at /v1/responses/, so substituting them would 400 every
            # call. Detect display-name-shaped entries via spaces and fall
            # back to the curated OPENROUTER_MODELS instead.
            display_name_count = sum(1 for m in listed if " " in m)
            if display_name_count > len(listed) // 2:
                models = list(OPENROUTER_MODELS)
                print(f"WARNING: provider's /v1/models returned {len(listed)} "
                      f"entries but {display_name_count} of them contain "
                      "spaces, suggesting they are DISPLAY names rather than "
                      "API IDs. Using your curated OPENROUTER_MODELS instead. "
                      "First few listed display names for reference:")
                for m in listed[:8]:
                    print(f"  - {m}")
                if len(listed) > 8:
                    print(f"  ... and {len(listed) - 8} more")
                print("To override and use the listing literally, pass "
                      "--ollama-models <comma-list>.")
            else:
                models = list(dict.fromkeys(listed))  # preserve order, dedupe
                print(f"WARNING: provider doesn't host any of the "
                      f"{len(OPENROUTER_MODELS)} OPENROUTER_MODELS. "
                      "Falling back to whatever it does host:")
                for m in models[:20]:
                    print(f"  - {m}")
                if len(models) > 20:
                    print(f"  ... and {len(models) - 20} more")
                print("Generated data will be DIVERSE but won't mirror the "
                      "curated OPENROUTER_MODELS distribution. This is normal "
                      "for Ollama Cloud (which uses '*-cloud' suffixed names) "
                      "and most OpenAI-compatible providers. To pin the "
                      "rotation, pass --ollama-models <comma-list>.")
        elif is_local:
            sys.exit(
                "Local Ollama has no models pulled. Pull at least one and retry, e.g.:\n"
                "  ollama pull llama3.2\n"
                "Recommended GPU-friendly subset:\n"
                "  ollama pull " + " ".join(LOCAL_GPU_MODELS)
            )
        else:
            print("WARNING: provider listing was empty -- defaulting to "
                  "OPENROUTER_MODELS. If every API call fails, switch providers "
                  "or pass --ollama-models with names your provider actually hosts.")
            models = list(OPENROUTER_MODELS)

    # `reasoning_effort` is only injected into Responses API payloads.
    # For ollama / openai modes we leave it None so non-Responses
    # providers don't see an unexpected field.
    reasoning_effort = (args.reasoning_effort
                        if args.api_mode == "responses" else None)
    client = ChatClient(args.ollama_url, models, args.ollama_token,
                        api_mode=args.api_mode,
                        chat_timeout=args.ollama_timeout,
                        reasoning_effort=reasoning_effort,
                        extra_headers=extra_headers)

    print(f"API mode: {args.api_mode}")
    print(f"Endpoint: {args.ollama_url}")
    print(f"Auth:     {'Bearer ********' if args.ollama_token else '(none)'}")
    print(f"Models in rotation ({len(models)}): {models[:5]}{' ...' if len(models) > 5 else ''}")

    if not args.skip_health_check and not auto_detected:
        try:
            pulled = client.health_check()
            unique_requested = set(models)
            missing = [m for m in unique_requested
                       if m not in pulled and f"{m}:latest" not in pulled]
            if missing:
                print(f"WARNING: provider didn't list these models: {sorted(missing)[:10]}"
                      f"{' ...' if len(missing) > 10 else ''}")
                print(f"  ({len(missing)}/{len(unique_requested)} model names unrecognised — "
                      f"may still work if listing is incomplete on the provider side)")
        except Exception as e:
            print(f"NOTE: health check failed ({e}). Continuing — many hosted services "
                  f"don't expose a model listing endpoint. Use --skip-health-check to "
                  f"silence this.")

    # PileStream was removed in the cc_net refactor. CC-only mode now.
    # If the user explicitly asks for pile (pile_prob > 0), bail with a
    # clear message rather than silently dropping the request.
    pile = None
    if args.pile_prob > 0.0:
        sys.exit(
            f"--pile-prob={args.pile_prob} requires PileStream which has been "
            f"removed in the cc_net refactor. Pass --pile-prob 0.0 for "
            f"CC-only generation, or restore PileStream if you need Pile."
        )

    cc_stream: Optional[CCStream] = None
    if args.enable_cc:
        try:
            cc_stream = CCStream(
                max_prompt_len=args.max_prompt_len,
                num_segments=args.cc_num_segments,
                cc_root=args.cc_root,
            )
            print(f"Common Crawl: enabled  (pile_prob={args.pile_prob:.3f}, "
                  f"num_segments={args.cc_num_segments})")
        except Exception as e:
            print(f"WARNING: failed to init CCStream ({e}). Falling back to Pile-only. "
                  f"If you want CC, ensure cc_net is installed and the data files "
                  f"(bin/lid.bin, data/lm_sp/, collinfo.json) are present.")
            cc_stream = None
    else:
        print("Common Crawl: disabled (--no-cc).")

    source = SourceMux(pile=pile, cc=cc_stream, pile_prob=args.pile_prob)
    seen_path = args.seen_roots.strip() if args.seen_roots else ""
    source = DedupSource(source, seen_path or None)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fail fast if the output file is locked (e.g. open in Excel / IDE
    # viewer on Windows). Otherwise we'd waste 30+ seconds on Pile/CC
    # dataset initialization before the lock is discovered at the first
    # row-write attempt.
    try:
        with open(out_path, "a", encoding="utf-8"):
            pass
    except PermissionError as e:
        sys.exit(
            f"Cannot write to {out_path}: {e}\n"
            "It looks like the file is open in another program (Excel, "
            "IDE viewer, etc.) holding an exclusive lock. Close it and "
            "retry, or pass a different --output path."
        )
    except OSError as e:
        sys.exit(f"Cannot open {out_path} for append: {e}")

    already = count_existing_rows(out_path)
    if already >= args.n_samples:
        print(f"{out_path} already has {already} rows >= --n-samples {args.n_samples}. Done.")
        return

    new_file = not out_path.exists() or already == 0
    fieldnames = ["text", "segmentation_labels", "data_source",
                  "sample_type", "model_name", "n_words", "augmented"]
    remaining = args.n_samples - already

    print(f"Resuming: {already} existing -> producing {remaining} more")
    print(f"Workers: {args.workers}  |  even_mix: {args.even_mix}")

    # Resolve the active sample-type mix. --sample-types overrides the
    # default 25/25/25/25 with equal probability for each requested type.
    valid_types = [t for t, _ in SAMPLE_MIX]
    if args.sample_types:
        requested = [t.strip() for t in args.sample_types.split(",") if t.strip()]
        unknown = [t for t in requested if t not in valid_types]
        if unknown:
            sys.exit(
                f"--sample-types contains unknown types: {unknown}\n"
                f"Valid types: {valid_types}"
            )
        # De-dup while preserving order; equal weight per requested type.
        seen = set()
        requested = [t for t in requested if not (t in seen or seen.add(t))]
        per = 1.0 / len(requested)
        active_mix = [(t, per) for t in requested]
        print(f"Sample-type override: only {requested} "
              f"({per * 100:.1f}% each)")
    else:
        active_mix = list(SAMPLE_MIX)

    # Build the task plan. Even-mix → deterministic counts + round-robin
    # model assignment. Random → just one (sample_type, model) per slot, by
    # the same probability rules as before.
    if args.even_mix:
        tasks = build_task_plan(remaining, models, mix=active_mix)
    else:
        # Random sampling honours `active_mix` too -- weights drive the
        # per-slot probability without the deterministic round-robin.
        cum = []
        s = 0.0
        for t, p in active_mix:
            s += p
            cum.append((t, s))
        tasks = []
        for _ in range(remaining):
            r = random.random()
            st = next((t for t, c in cum if r <= c), active_mix[-1][0])
            tasks.append((st, "none" if st == "pure_human" else random.choice(models)))

    plan = Counter(t for t, _ in tasks)
    print(f"Planned distribution: " +
          ", ".join(f"{k}={v}" for k, v in plan.most_common()))

    # Pile producer thread keeps a queue topped up so workers never wait on
    # dataset I/O. Single-threaded fetching avoids races inside `datasets`.
    pile_queue: "queue.Queue[Dict[str, str]]" = queue.Queue(maxsize=args.pile_buffer)
    stop_event = threading.Event()
    pile_thread = threading.Thread(
        target=_pile_producer, args=(source, pile_queue, stop_event), daemon=True,
    )
    pile_thread.start()

    counts: Dict[str, int] = {n: 0 for n, _ in SAMPLE_MIX}
    fails: Counter = Counter()
    error_examples: List[Tuple[Tuple[str, str], str]] = []
    error_reason_counts: Counter = Counter()
    t_first = time.time()
    pbar = tqdm(total=args.n_samples, initial=already, dynamic_ncols=True)

    interrupted = False
    try:
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if new_file:
                writer.writeheader()
                f.flush()
            write_lock = threading.Lock()

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                future_to_task = {
                    pool.submit(_process_one, task, client, pile_queue, args): task
                    for task in tasks
                }
                produced = already
                for fut in as_completed(future_to_task):
                    task = future_to_task[fut]
                    try:
                        row, err = fut.result()
                    except Exception as e:
                        row, err = None, f"{type(e).__name__}: {e}"
                    if row is None:
                        fails[task[0]] += 1
                        if err:
                            # Bucket error reasons (e.g. "HTTP 404") so the user
                            # sees how many failures share the same root cause.
                            short = err.split("\n", 1)[0][:80]
                            error_reason_counts[short] += 1
                            if len(error_examples) < 5:
                                error_examples.append((task, err))
                        pbar.update(1)
                        produced += 1
                        continue

                    with write_lock:
                        writer.writerow(row)
                        f.flush()
                    counts[row["sample_type"]] = counts.get(row["sample_type"], 0) + 1
                    produced += 1
                    pbar.update(1)

                    if produced % args.report_every == 0:
                        elapsed = time.time() - t_first
                        rate = (produced - already) / elapsed if elapsed > 0 else 0
                        pbar.set_postfix(
                            rate=f"{rate:.2f}/s",
                            counts=",".join(f"{k[:2]}={v}" for k, v in counts.items()),
                            fails=sum(fails.values()),
                        )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted — output is up to date and resumable.")
    finally:
        stop_event.set()

    pbar.close()
    print("\nDone." if not interrupted else "\nStopped (resumable).")
    print(f"Output: {out_path} ({count_existing_rows(out_path)} rows total)")
    print(f"Per sample-type counts produced this run: {dict(counts)}")
    if fails:
        print(f"Failures (dropped after retries): {dict(fails)}")
    if error_reason_counts:
        print("\nFailure reasons (most common first):")
        for reason, n in error_reason_counts.most_common(10):
            print(f"  {n:>4}  {reason}")
        print("\nFirst few raw errors (for debugging):")
        for task, err in error_examples:
            print(f"  task={task} -> {err}")
    print(
        "\nNext step -- sanity-check with the eval script:\n"
        f"  python scripts/eval_prediction_modes.py --csv {out_path} "
        "--text-col text --labels-col segmentation_labels"
    )


if __name__ == "__main__":
    main()
