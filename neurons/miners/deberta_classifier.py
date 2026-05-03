# Pre-load pandas + sklearn before transformers. transformers 4.49 lazy-imports
# `transformers.generation.candidate_generator`, which imports sklearn, which
# imports pandas._libs. On Windows the cumulative depth of that nested DLL load
# chain overflows the 1 MB main-thread stack and segfaults the interpreter
# (https://github.com/huggingface/transformers/issues -- "stack overflow on
# Windows"). Importing them on a fresh shallow stack here populates sys.modules
# so transformers' later lazy chain hits cache and never recurses.
import pandas  # noqa: F401
import sklearn  # noqa: F401

import re

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset
from tqdm import tqdm

# Validator uses NLTK punkt for sentence splitting at
# detection/validator/data_generator.py:25. We mirror it so our sentence
# boundaries line up with the label boundaries the validator generates.
try:
    import nltk

    def _load_punkt_tokenizer():
        # Newer NLTK (>=3.8.2) redirects `punkt` to `punkt_tab` internally; the
        # old `punkt` download no longer satisfies the lookup. Try both.
        for pkg in ("punkt_tab", "punkt"):
            try:
                return nltk.data.load("tokenizers/punkt/english.pickle")
            except LookupError:
                nltk.download(pkg, quiet=True)
        return nltk.data.load("tokenizers/punkt/english.pickle")

    _NLTK_PUNKT = _load_punkt_tokenizer()
except Exception:
    _NLTK_PUNKT = None


class SimpleTestDataset(Dataset):
    def __init__(self, strings, tokenizer, max_sequence_length):
        self.Strings = strings
        self.Tokenizer = tokenizer
        self.MaxSequenceLength = max_sequence_length

    def __len__(self):
        return len(self.Strings)

    def __getitem__(self, idx):
        string = self.Strings[idx].strip()
        token_ids = self.Tokenizer(string, max_length=self.MaxSequenceLength, truncation=True).input_ids

        return {
            'input_ids': token_ids,
        }


def GeneratePredictions(model, tokenizer, test_dataset, device, batch_size=16):
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=DataCollatorWithPadding(tokenizer))

    # When the model is already half-precision we skip autocast (which would
    # otherwise insert redundant fp32<->fp16 casts) and rely on the model's
    # native dtype for the forward pass. softmax is still computed in fp32
    # for numerical stability regardless of the model dtype.
    model_dtype = next(model.parameters()).dtype
    is_half = model_dtype in (torch.float16, torch.bfloat16)
    on_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

    all_predictions = []
    with torch.inference_mode():
        for batch in data_loader:
            token_sequences = batch.input_ids.to(device)
            attention_masks = batch.attention_mask.to(device)

            if is_half or not on_cuda:
                raw_predictions = model(token_sequences, attention_masks).logits
            else:
                with torch.cuda.amp.autocast():
                    raw_predictions = model(token_sequences, attention_masks).logits

            scaled_predictions = raw_predictions.float().softmax(dim=1)[:, 1]
            all_predictions.append(scaled_predictions.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)

    return all_predictions


class DebertaClassifier:
    _PRECISION_DTYPES = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    def __init__(self, foundation_model_path, model_path, device,
                 precision="fp32", batch_size=16):
        if precision not in self._PRECISION_DTYPES:
            raise ValueError(
                f"precision must be one of {list(self._PRECISION_DTYPES)}, got {precision!r}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(foundation_model_path)
        self.max_length = 1024
        self.device = device
        self.batch_size = batch_size
        self.precision = precision
        self.dtype = self._PRECISION_DTYPES[precision]

        model = AutoModelForSequenceClassification.from_pretrained(
            foundation_model_path,
            state_dict=torch.load(model_path, map_location=device),
            attention_probs_dropout_prob=0,
            hidden_dropout_prob=0).to(device)

        # Cast weights down to fp16/bf16 for inference. DeBERTa-v3 forward
        # is well-conditioned in bf16 on Ada/Hopper GPUs and roughly halves
        # both the resident weight memory and the per-batch activation
        # memory vs. fp32.
        if self.dtype != torch.float32:
            model = model.to(dtype=self.dtype)

        self.model = model.eval()

    def predict_batch(self, texts):
        test_dataset = SimpleTestDataset(texts, self.tokenizer, self.max_length)
        return GeneratePredictions(self.model, self.tokenizer, test_dataset,
                                   self.device, batch_size=self.batch_size)

    def predict_word_probabilities_batch(
        self,
        texts,
        mode="hybrid",
        sentence_smoothing=0.15,
        window_size=96,
        window_stride=48,
        hybrid_window_weight=0.65,
        min_sentence_words=4,
    ):
        if mode == "document":
            return [
                [float(score)] * max(1, len(text.split()))
                for score, text in zip(self.predict_batch(texts), texts)
            ]

        predictions = []
        for text in texts:
            words = text.split()
            if not words:
                predictions.append([])
                continue

            if mode == "sentence":
                word_scores = self._predict_sentence_word_probabilities(
                    text, sentence_smoothing=sentence_smoothing
                )
            elif mode == "sentence_nltk":
                word_scores = self._predict_sentence_word_probabilities_nltk(
                    text,
                    sentence_smoothing=sentence_smoothing,
                    min_sentence_words=min_sentence_words,
                )
            elif mode == "window":
                word_scores = self._predict_window_word_probabilities(
                    words,
                    window_size=window_size,
                    window_stride=window_stride,
                )
            elif mode == "hybrid":
                sentence_scores = self._predict_sentence_word_probabilities(
                    text, sentence_smoothing=sentence_smoothing
                )
                window_scores = self._predict_window_word_probabilities(
                    words,
                    window_size=window_size,
                    window_stride=window_stride,
                )
                word_scores = [
                    float(
                        (1 - hybrid_window_weight) * sentence_score
                        + hybrid_window_weight * window_score
                    )
                    for sentence_score, window_score in zip(sentence_scores, window_scores)
                ]
            elif mode == "hybrid_nltk":
                sentence_scores = self._predict_sentence_word_probabilities_nltk(
                    text,
                    sentence_smoothing=sentence_smoothing,
                    min_sentence_words=min_sentence_words,
                )
                window_scores = self._predict_window_word_probabilities(
                    words,
                    window_size=window_size,
                    window_stride=window_stride,
                )
                word_scores = [
                    float(
                        (1 - hybrid_window_weight) * sentence_score
                        + hybrid_window_weight * window_score
                    )
                    for sentence_score, window_score in zip(sentence_scores, window_scores)
                ]
            else:
                raise ValueError(f"Unsupported prediction mode: {mode}")

            word_scores = self._align_word_scores(word_scores, len(words))
            predictions.append(word_scores)

        return predictions

    def __call__(self, text):
        return self.predict_batch([text])[0]

    def _predict_sentence_word_probabilities(self, text, sentence_smoothing=0.15):
        sentence_infos = self._split_into_sentence_word_spans(text)
        if not sentence_infos:
            return []

        sentence_texts = [sentence_text for sentence_text, _, _ in sentence_infos]
        sentence_scores = self.predict_batch(sentence_texts).tolist()
        sentence_scores = self._smooth_scores(sentence_scores, sentence_smoothing)

        word_scores = []
        for score, (_, start_word, end_word) in zip(sentence_scores, sentence_infos):
            word_scores.extend([float(score)] * (end_word - start_word))

        return word_scores

    def _predict_sentence_word_probabilities_nltk(
        self,
        text,
        sentence_smoothing=0.15,
        min_sentence_words=4,
    ):
        """Sentence-level scoring using the same NLTK punkt tokenizer the
        validator uses in detection/validator/data_generator.py:25. Because the
        validator's AI/human label transitions fall on these sentence
        boundaries, classifying each sentence independently lets the
        per-sentence score be assigned to every word of that sentence with
        (mostly) correct alignment against the validator's ground truth.

        Very short sentences (< min_sentence_words) are grouped with the next
        chunk so DeBERTa has enough context, but the resulting score is still
        attributed back to each original sentence's word count.
        """

        total_words = len(text.split())
        if total_words == 0:
            return []

        if _NLTK_PUNKT is None:
            return self._predict_sentence_word_probabilities(
                text, sentence_smoothing=sentence_smoothing
            )

        try:
            spans = list(_NLTK_PUNKT.span_tokenize(text))
        except Exception:
            spans = []

        if not spans:
            doc_score = float(self.predict_batch([text])[0])
            return [doc_score] * total_words

        # Mirror the validator's get_sentences(): each span expands to absorb
        # trailing whitespace up to the next span's start so the chunks cover
        # the whole text and match exactly how the validator sliced it.
        sentence_chunks = []
        word_counts = []
        for i, (start, end) in enumerate(spans):
            next_start = spans[i + 1][0] if i + 1 < len(spans) else len(text)
            expanded_end = end
            while expanded_end < next_start and text[expanded_end].isspace():
                expanded_end += 1
            chunk = text[start:expanded_end]
            wc = len(chunk.split())
            if wc == 0:
                continue
            sentence_chunks.append(chunk)
            word_counts.append(wc)

        if not sentence_chunks:
            doc_score = float(self.predict_batch([text])[0])
            return [doc_score] * total_words

        # Group very short chunks with the following ones so the classifier
        # sees enough context. The group's predicted score is attributed back
        # to every original chunk in the group, preserving per-word alignment.
        merged_texts = []
        orig_to_merged = [0] * len(sentence_chunks)

        i = 0
        while i < len(sentence_chunks):
            group_start = i
            combined_text = sentence_chunks[i]
            combined_words = word_counts[i]
            while combined_words < min_sentence_words and i + 1 < len(sentence_chunks):
                i += 1
                combined_text += sentence_chunks[i]
                combined_words += word_counts[i]
            merged_id = len(merged_texts)
            merged_texts.append(combined_text)
            for g in range(group_start, i + 1):
                orig_to_merged[g] = merged_id
            i += 1

        merged_scores = self.predict_batch(merged_texts).tolist()
        merged_scores = self._smooth_scores(merged_scores, sentence_smoothing)

        word_scores = []
        for orig_idx, wc in enumerate(word_counts):
            score = float(merged_scores[orig_to_merged[orig_idx]])
            word_scores.extend([score] * wc)

        return word_scores

    @staticmethod
    def _align_word_scores(word_scores, target_len):
        """Force the per-word score list to be exactly `target_len` long.
        NLTK span coverage can occasionally differ from a raw .split() by a
        word or two (leading whitespace, weird unicode); if so, pad or trim so
        the validator's `len(predictions[i]) == len(labels[i])` check passes.
        """
        if not word_scores:
            return [0.0] * target_len
        if len(word_scores) == target_len:
            return [float(s) for s in word_scores]
        if len(word_scores) < target_len:
            fill = float(word_scores[-1])
            return [float(s) for s in word_scores] + [fill] * (target_len - len(word_scores))
        return [float(s) for s in word_scores[:target_len]]

    def _predict_window_word_probabilities(self, words, window_size=96, window_stride=48):
        windows = []
        ranges = []
        n_words = len(words)

        if n_words == 0:
            return []

        if n_words <= window_size:
            return [float(self.predict_batch([" ".join(words)])[0])] * n_words

        for start in range(0, n_words, window_stride):
            end = min(start + window_size, n_words)
            windows.append(" ".join(words[start:end]))
            ranges.append((start, end))
            if end == n_words:
                break

        scores = self.predict_batch(windows).tolist()
        weighted_scores = np.zeros(n_words, dtype=np.float32)
        weights = np.zeros(n_words, dtype=np.float32)

        for score, (start, end) in zip(scores, ranges):
            span_len = end - start
            midpoint = (span_len - 1) / 2 if span_len > 1 else 0
            for offset, idx in enumerate(range(start, end)):
                distance = abs(offset - midpoint)
                weight = 1.0 - (distance / max(midpoint + 1.0, 1.0)) * 0.5
                weighted_scores[idx] += float(score) * weight
                weights[idx] += weight

        weights = np.where(weights == 0, 1.0, weights)
        return (weighted_scores / weights).tolist()

    def _split_into_sentence_word_spans(self, text):
        chunks = [chunk for chunk in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if chunk.strip()]
        if not chunks:
            chunks = [text.strip()]

        word_spans = []
        cursor = 0
        for chunk in chunks:
            word_count = len(chunk.split())
            if word_count == 0:
                continue
            start_word = cursor
            end_word = cursor + word_count
            word_spans.append((chunk, start_word, end_word))
            cursor = end_word

        if not word_spans:
            return []

        return word_spans

    def _smooth_scores(self, scores, smoothing):
        if len(scores) <= 1 or smoothing <= 0:
            return [float(score) for score in scores]

        smoothed = []
        for idx, score in enumerate(scores):
            cur_weight = 1.0
            total = float(score)

            if idx > 0:
                total += float(scores[idx - 1]) * smoothing
                cur_weight += smoothing
            if idx + 1 < len(scores):
                total += float(scores[idx + 1]) * smoothing
                cur_weight += smoothing

            smoothed.append(total / cur_weight)

        return smoothed
