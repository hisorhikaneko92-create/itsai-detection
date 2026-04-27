"""
Validator-faithful training-dataset builder for SN32.

Mirrors the validator pipeline (detection/validator/data_generator.py) without
requiring bittensor / cc_net. Supports both **local Ollama** and **hosted
Ollama-compatible APIs** (Ollama Cloud, OpenRouter, etc.) so you can use the
full validator model pool — including 70B+ checkpoints that won't fit a home
GPU — by paying per-token instead of buying hardware.

For each sample we:

  1. Pull one document from The Pile (HuggingFace streaming) — same source the
     validator's PromptDataset uses.
  2. Char-cut at a random index in [0.25 .. 0.75] of the doc length, producing
     {prompt, completion}. Reproduces detection/validator/my_datasets.py:54
     and is the reason boundaries can fall mid-word ("Frankenstein" cases).
  3. Pick a sample type from the validator's mix:
       25% pure-human   (label = all 0)
       25% pure-AI      (label = all 1; prompt is dropped, only AI text kept)
       40% human-then-AI (prompt + AI continuation → labels [0]*prompt_words + [1]*rest)
       10% AI-in-middle  (NLTK sentence-aligned begin + AI middle + end)
  4. For AI types, call the configured chat API with a model picked at random
     from --ollama-models. Default rotation = the full 30-model validator pool
     (detection/validator/data_generator.py:327-364). Override --ollama-models
     to use a curated subset.
  5. Apply light augmentation (typos / case flips / punctuation drops) that
     keeps len(text.split()) constant — mirrors detection/attacks/data_augmentation.
  6. Subsample to 35-350 words via the validator's exact rule.
  7. Append one row to the output CSV. Resumable: rerun with the same --output
     and it counts existing rows, generating only the remainder.

Output CSV schema (compatible with scripts/eval_prediction_modes.py):
    text                 mixed AI+human text after subsample/augment
    segmentation_labels  JSON list of per-word 0|1
    data_source          'pile' (matches validator's data_source tag)
    sample_type          one of: pure_human, pure_ai, human_then_ai, ai_in_middle
    model_name           model used (or 'none' for pure_human)
    n_words              len(text.split()) — sanity column
    augmented            'true' / 'false'

============================================================================
USAGE A — Hosted Ollama API (the full 70B+ validator pool, recommended)
============================================================================

Set your API key once, then run with the validator model list. The script
sends `Authorization: Bearer <key>` to the configured base URL.

    export OLLAMA_API_KEY="sk-...."     # or pass --ollama-token
    export OLLAMA_URL="https://ollama.com"

    python scripts/build_training_dataset.py \
        --n-samples 50000 \
        --output data/train_50k.csv

By default --ollama-models is the validator pool (30 names). If your provider
exposes a different name format (e.g. `meta-llama/Llama-3.1-70B-Instruct`),
override it:

    python scripts/build_training_dataset.py \
        --api-mode openai \
        --ollama-url https://api.together.xyz \
        --ollama-models "meta-llama/Llama-3.3-70B-Instruct-Turbo,Qwen/Qwen2.5-72B-Instruct-Turbo,mistralai/Mistral-Large-2411" \
        --output data/train_50k.csv

============================================================================
USAGE B — Local Ollama on the GPU PC (smaller models only)
============================================================================

After `ollama pull llama3.2 qwen2.5:7b mistral:7b`:

    python scripts/build_training_dataset.py \
        --ollama-url http://127.0.0.1:11434 \
        --ollama-models llama3.2,qwen2.5:7b,mistral:7b \
        --n-samples 5000 \
        --output data/local_smoke.csv

============================================================================

Cost note for hosted: across the validator's mix, each sample averages
~600 tokens of completion. 50,000 samples ≈ 30M tokens — at the price of a
70B-tier model that's roughly $20-$60 depending on provider. Smoke-test with
--n-samples 200 before committing.
"""
import argparse
import csv
import json
import os
import random
import string
import sys
import time
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
# Validator's full model pool — copied from
# detection/validator/data_generator.py:327-364. Duplicated entries in the
# original (e.g. command-r-plus:104b appears 3 times) are preserved so that
# `random.choice()` reproduces the validator's biased model distribution.
# Override at the CLI with --ollama-models if your provider uses different
# model name strings.
# ---------------------------------------------------------------------------
VALIDATOR_MODELS = [
    "llama2:13b",
    "llama3:text",
    "llama3:70b",
    "llama3.1:70b-text-q4_0",
    "llama3.2",
    "llama3.3:70b",
    "qwen:32b-text-v1.5-q4_0",
    "qwen2.5:14b",
    "qwen2.5-coder:32b",
    "qwen2.5:72b",
    "qwen3:32b",
    "command-r",
    "command-r",
    "command-r-plus:104b",
    "command-r-plus:104b",
    "command-r-plus:104b",
    "gemma2:27b-text-q4_0",
    "mistral-nemo:12b",
    "mistral-small:22b",
    "mistral-large:123b",
    "internlm2:7b",
    "internlm2:20b",
    "internlm/internlm2.5:20b-chat",
    "internlm/internlm2.5:latest",
    "internlm/internlm2.5:20b-chat",
    "deepseek-v2:16b",
    "deepseek-r1:14b",
    "phi4:14b",
    "aya-expanse:32b",
    "yi:34b-chat",
    "athene-v2:72b",
]


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
                 chat_timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.models = models
        self.token = token
        self.api_mode = api_mode
        self.chat_timeout = chat_timeout

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def health_check(self) -> List[str]:
        """Best-effort listing of available models. Many hosted services
        don't expose this — failures are non-fatal upstream."""
        url = (f"{self.base_url}/api/tags" if self.api_mode == "ollama"
               else f"{self.base_url}/v1/models")
        r = requests.get(url, headers=self._headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        if self.api_mode == "ollama":
            return [m["name"] for m in data.get("models", [])]
        return [m.get("id", "") for m in data.get("data", [])]

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
            r = requests.post(url, headers=self._headers(), json=payload,
                              timeout=self.chat_timeout)
            r.raise_for_status()
            return r.json()["message"]["content"]
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
            r = requests.post(url, headers=self._headers(), json=payload,
                              timeout=self.chat_timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

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
        r = requests.post(url, headers=self._headers(), json=payload,
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
# Pile streamer — same source as detection/validator/my_datasets.py:73-84
# ---------------------------------------------------------------------------
class PileStream:
    def __init__(self, max_prompt_len: int = 1500, seed: Optional[int] = None,
                 dataset_name: str = "monology/pile-uncopyrighted",
                 buffer_size: int = 10000):
        self.max_prompt_len = max_prompt_len
        self.dataset_name = dataset_name
        self.buffer_size = buffer_size
        self.seed = seed if seed is not None else int(time.time())
        self._iter = None
        self._init_iter()

    def _init_iter(self):
        self._iter = iter(
            load_dataset(self.dataset_name, streaming=True)["train"]
            .shuffle(seed=self.seed, buffer_size=self.buffer_size)
        )

    def next_pair(self) -> Dict[str, str]:
        """Returns {'prompt', 'completion', 'data_source'} matching the
        validator's PromptDataset semantics."""
        for _ in range(50):  # try a few times if a doc fails the size check
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
# Sample builders
# ---------------------------------------------------------------------------
def build_pure_human(pair: Dict[str, str]) -> Tuple[str, List[int], str, str]:
    """The validator's HumanDataset returns el['real_completion'] as the
    text — i.e. the second half of the cut document. Labels = all 0."""
    text = pair["completion"].strip()
    labels = [0] * len(text.split())
    return text, labels, "pure_human", "none"


def build_pure_ai(pair: Dict[str, str], client: ChatClient) -> Tuple[str, List[int], str, str]:
    """generate_ai_data with cnt_first_human=0. Ask the model to continue
    the prompt; keep only its completion."""
    model = client.pick()
    completion = client.continue_text(model, pair["prompt"])
    if not completion:
        raise RuntimeError(f"empty completion from {model}")
    labels = [1] * len(completion.split())
    return completion, labels, "pure_ai", model


def build_human_then_ai(pair: Dict[str, str], client: ChatClient) -> Tuple[str, List[int], str, str]:
    """generate_ai_data with the merge_prompt_text branch enabled. Final
    text is prompt + AI continuation, labels = [0]*prompt_words + [1]*rest."""
    model = client.pick()
    completion = client.continue_text(model, pair["prompt"])
    if not completion:
        raise RuntimeError(f"empty completion from {model}")
    text = pair["prompt"] + completion
    cnt_first_human = len(pair["prompt"].split())
    labels = [0] * cnt_first_human + [1] * (len(text.split()) - cnt_first_human)
    return text, labels, "human_then_ai", model


def build_ai_in_middle(pair: Dict[str, str], client: ChatClient) -> Tuple[str, List[int], str, str]:
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
SAMPLE_MIX = [
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-samples", type=int, default=10000,
                    help="Total samples to produce (counts existing rows in --output).")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
                    help="Base URL for the chat API. For Ollama Cloud use "
                         "https://ollama.com. For OpenAI-compatible providers (Together, "
                         "OpenRouter, Groq, vLLM) use their endpoint and pair with "
                         "--api-mode openai. Falls back to $OLLAMA_URL.")
    ap.add_argument("--ollama-token", default=os.environ.get("OLLAMA_API_KEY"),
                    help="Bearer API key. Sent as `Authorization: Bearer <token>`. "
                         "Required for Ollama Cloud and most paid hosted services. "
                         "Falls back to $OLLAMA_API_KEY.")
    ap.add_argument("--api-mode", choices=("ollama", "openai"), default="ollama",
                    help="Wire protocol. 'ollama' uses /api/chat (works for local "
                         "ollama serve and for Ollama Cloud). 'openai' uses "
                         "/v1/chat/completions (Together, OpenRouter, Groq, vLLM, etc.).")
    ap.add_argument("--ollama-models", default=",".join(VALIDATOR_MODELS),
                    help="Comma-separated list of model names to rotate through. "
                         "Default: the full validator pool (30 models, with the same "
                         "duplicates the validator uses to bias toward command-r-plus, "
                         "etc). Override if your provider uses different name strings, "
                         "e.g. 'meta-llama/Llama-3.3-70B-Instruct-Turbo,Qwen/Qwen2.5-72B...'.")
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
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    models = [m.strip() for m in args.ollama_models.split(",") if m.strip()]
    if not models:
        sys.exit("--ollama-models cannot be empty")

    client = ChatClient(args.ollama_url, models, args.ollama_token,
                        api_mode=args.api_mode,
                        chat_timeout=args.ollama_timeout)

    print(f"API mode: {args.api_mode}")
    print(f"Endpoint: {args.ollama_url}")
    print(f"Auth:     {'Bearer ********' if args.ollama_token else '(none)'}")
    print(f"Models in rotation ({len(models)}): {models[:5]}{' ...' if len(models) > 5 else ''}")

    if not args.skip_health_check:
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

    pile = PileStream(max_prompt_len=args.max_prompt_len)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already = count_existing_rows(out_path)
    if already >= args.n_samples:
        print(f"{out_path} already has {already} rows ≥ --n-samples {args.n_samples}. Done.")
        return

    new_file = not out_path.exists() or already == 0
    fieldnames = ["text", "segmentation_labels", "data_source",
                  "sample_type", "model_name", "n_words", "augmented"]

    print(f"Resuming: {already} existing → producing {args.n_samples - already} more")
    pbar = tqdm(total=args.n_samples, initial=already, dynamic_ncols=True)

    counts = {n: 0 for n, _ in SAMPLE_MIX}
    fail_counts: Dict[str, int] = {}
    t_first = time.time()

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
            f.flush()

        produced = already
        while produced < args.n_samples:
            sample_type = pick_sample_type()
            try:
                pair = pile.next_pair()
                if sample_type == "pure_human":
                    text, labels, st, model_name = build_pure_human(pair)
                elif sample_type == "pure_ai":
                    text, labels, st, model_name = build_pure_ai(pair, client)
                elif sample_type == "human_then_ai":
                    text, labels, st, model_name = build_human_then_ai(pair, client)
                else:
                    text, labels, st, model_name = build_ai_in_middle(pair, client)

                if not args.no_subsample:
                    text, labels = subsample_words(text, labels)
                if len(labels) != len(text.split()) or len(labels) < 20:
                    raise RuntimeError(f"length sanity failed ({len(labels)} vs {len(text.split())})")

                augmented = False
                if not args.no_augment:
                    text, labels = augment(text, labels, per_word_p=args.augment_rate)
                    augmented = True
                    if len(labels) != len(text.split()):
                        raise RuntimeError("augmentation broke word count")

                writer.writerow({
                    "text": text,
                    "segmentation_labels": json.dumps(labels),  # JSON list, parses cleanly
                    "data_source": pair["data_source"],
                    "sample_type": st,
                    "model_name": model_name,
                    "n_words": len(labels),
                    "augmented": "true" if augmented else "false",
                })
                f.flush()
                counts[st] = counts.get(st, 0) + 1
                produced += 1
                pbar.update(1)
            except KeyboardInterrupt:
                print("\nInterrupted — output is up to date and resumable.")
                break
            except Exception as e:
                fail_counts[sample_type] = fail_counts.get(sample_type, 0) + 1
                # don't pollute the bar; only show in periodic reports
                continue

            if produced % args.report_every == 0:
                elapsed = time.time() - t_first
                rate = (produced - already) / elapsed if elapsed > 0 else 0
                pbar.set_postfix(
                    rate=f"{rate:.2f}/s",
                    counts=",".join(f"{k[:2]}={v}" for k, v in counts.items()),
                    fails=sum(fail_counts.values()),
                )

    pbar.close()
    print("\nDone.")
    print(f"Output: {out_path} ({count_existing_rows(out_path)} rows total)")
    print(f"Per sample-type counts produced this run: {counts}")
    if fail_counts:
        print(f"Failures (skipped, retried): {fail_counts}")
    print(
        "\nNext step — sanity-check with the eval script:\n"
        f"  python scripts/eval_prediction_modes.py --csv {out_path} "
        "--text-col text --labels-col segmentation_labels"
    )


if __name__ == "__main__":
    main()
