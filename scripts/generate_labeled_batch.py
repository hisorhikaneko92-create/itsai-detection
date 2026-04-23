"""
Generate a labeled evaluation batch using the validator's EXACT AI-in-the-middle
logic, without any of the heavy validator dependencies.

The validator at detection/validator/data_generator.py pulls in bittensor,
CommonCrawl processing, fasttext, HuggingFace datasets, etc. — none of which
are easy to install on Windows. This script reproduces the one thing we need
for evaluation (building labeled mixed AI+human text) using only:

    pip install nltk requests

Everything else — sentence splitting, the balance-thirds loop, summary/generation
prompts, the `[0]*begin + [1]*middle + [0]*end` label formula — is copied
verbatim from detection/validator/data_generator.py so the output is
structurally identical to a validator batch.

Output CSV columns (compatible with scripts/eval_prediction_modes.py):
    text                 : the mixed AI+human text
    segmentation_labels  : Python-literal list of per-word labels (0|1)
    data_source          : marker for the evaluation script (default 'synthetic')

Typical usage (PowerShell on the GPU PC):

    python scripts\\generate_labeled_batch.py `
        --ollama-model llama3.2 `
        --n-ai-middle 20 `
        --n-pure-ai 5 `
        --n-pure-human 5 `
        --output data\\my_batch.csv

Then evaluate:

    python scripts\\eval_prediction_modes.py `
        --csv data\\my_batch.csv `
        --text-col text `
        --labels-col segmentation_labels
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import requests

import nltk

def _load_punkt():
    # Newer NLTK (>=3.8.2) redirects `punkt` internally to `punkt_tab`, which
    # needs a separate download. Try both packages in order.
    for pkg in ("punkt_tab", "punkt"):
        try:
            return nltk.data.load("tokenizers/punkt/english.pickle")
        except LookupError:
            nltk.download(pkg, quiet=True)
    return nltk.data.load("tokenizers/punkt/english.pickle")

_SENT_TOK = _load_punkt()


# ---------------------------------------------------------------------------
# Copied verbatim from detection/validator/data_generator.py:26-44
# ---------------------------------------------------------------------------
def get_sentences(text):
    spans = list(_SENT_TOK.span_tokenize(text))
    sentences_with_trailing = []
    for i, (start, end) in enumerate(spans):
        if i < len(spans) - 1:
            next_start = spans[i + 1][0]
        else:
            next_start = len(text)
        expanded_end = end
        while expanded_end < next_start and expanded_end < len(text):
            if text[expanded_end].isspace():
                expanded_end += 1
            else:
                break
        sentence_text = text[start:expanded_end]
        sentences_with_trailing.append(sentence_text)
    return sentences_with_trailing


# ---------------------------------------------------------------------------
# Prompt sets — copied from detection/validator/data_generator.py:47-81.
# Kept to a subset for brevity; randomized selection is still representative.
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
    "Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",

    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. "
    "You receive the opening and closing paragraphs of a text, as well as a synopsis of the central section. "
    "Your task is to generate the text for the middle part alone, ensuring coherence with the given "
    "beginning and end. Keep any cautions or alerts by rewording them, and do not include any summarizing. "
    "Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",

    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. "
    "You are provided with a text's first and final segments along with a brief outline of what occurs "
    "in the middle. Your job is to fill in only the middle content. The final text should flow naturally, "
    "so do not insert a summary. Retain all warnings by rephrasing, and write nothing else. "
    "Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
]


# ---------------------------------------------------------------------------
# Ollama HTTP client (replaces langchain_ollama + bittensor)
# ---------------------------------------------------------------------------
def ollama_chat(base_url, model, messages, num_predict=900, timeout=300):
    r = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": num_predict},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def ollama_generate(base_url, model, prompt, num_predict=900, timeout=300):
    """Raw-text completion — used to emulate the 'full AI' pure-generation case."""
    r = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "options": {"num_predict": num_predict},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["response"]


# ---------------------------------------------------------------------------
# The AI-in-the-middle logic — copied from
# detection/validator/data_generator.py:84-134 with the only change being that
# `model.classic_invoke([...])` is replaced with a direct Ollama HTTP call.
# ---------------------------------------------------------------------------
def regenerated_in_the_middle(base_url, model, text):
    sentences = get_sentences(text)
    if len(sentences) < 3:
        raise ValueError(f"Need at least 3 sentences to split into thirds, got {len(sentences)}")

    lens = [len(x) for x in sentences]
    first_part = len(sentences) // 3
    second_part = 2 * len(sentences) // 3

    first_size = sum(lens[:first_part])
    second_size = sum(lens[first_part:second_part])
    third_size = sum(lens[second_part:])

    for _ in range(10):
        if first_part > 0 and first_size - lens[first_part - 1] > second_size + lens[first_part - 1]:
            first_part -= 1
            first_size = sum(lens[:first_part])
            second_size = sum(lens[first_part:second_part])
        elif second_part > first_part and second_size - lens[second_part - 1] > third_size + lens[second_part - 1]:
            second_part -= 1
            second_size = sum(lens[first_part:second_part])
            third_size = sum(lens[second_part:])
        elif first_part < second_part and first_size + lens[first_part] < second_size - lens[first_part]:
            first_part += 1
            first_size = sum(lens[:first_part])
            second_size = sum(lens[first_part:second_part])
        elif second_part < len(sentences) and second_size + lens[second_part] < third_size - lens[second_part]:
            second_part += 1
            second_size = sum(lens[first_part:second_part])
            third_size = sum(lens[second_part:])
        else:
            break

    begin = "".join(sentences[:first_part])
    middle = "".join(sentences[first_part:second_part])
    end = "".join(sentences[second_part:])

    middle_stripped = middle.rstrip()
    diff = len(middle) - len(middle_stripped)
    if diff > 0:
        end = middle[-diff:] + end
    middle = middle_stripped

    summary_prompt = random.choice(SUMMARY_PROMPTS)
    generation_prompt = random.choice(GENERATION_PROMPTS)

    summary = ollama_chat(base_url, model, [
        {"role": "system", "content": summary_prompt},
        {"role": "user", "content": middle},
    ])

    middle_size = max(1, len(middle.split()))
    generated_middle = ollama_chat(base_url, model, [
        {"role": "system", "content": generation_prompt + f" The middle should be about {middle_size} words long"},
        {"role": "user", "content": f"begin: {begin}\nend: {end}\nsummary: {summary}"},
    ]).strip()

    full_text = begin + generated_middle + end
    labels = ([0] * len(begin.split())
              + [1] * len(generated_middle.split())
              + [0] * len(end.split()))
    return full_text, labels


def pure_ai_sample(base_url, model, seed_prompt):
    """Emulates 'full AI' — ask the model to continue a human prompt; the
    completion is labeled entirely as AI (1), matching the `AI_PERCENT` case
    from detection/validator/data_generator.py:222-275 when cnt_first_human==0.
    """
    completion = ollama_generate(base_url, model, seed_prompt).strip()
    labels = [1] * len(completion.split())
    return completion, labels


def pure_human_sample(text):
    text = text.strip()
    labels = [0] * len(text.split())
    return text, labels


# ---------------------------------------------------------------------------
# Bundled public-domain human source texts (Gutenberg etexts, far out of
# copyright). These give the AI-in-the-middle flow something to work from
# when no --human-text-file is supplied. Each is self-contained prose with
# >=5 clear sentences so the balance-thirds logic can split cleanly.
# ---------------------------------------------------------------------------
BUNDLED_HUMAN_TEXTS = [
    # Pride and Prejudice, Jane Austen (1813) — opening of Chapter 1
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. "
    "However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well "
    "fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their "
    "daughters. My dear Mr. Bennet, said his lady to him one day, have you heard that Netherfield Park is let at last? "
    "Mr. Bennet replied that he had not. But it is, returned she; for Mrs. Long has just been here, and she told me all about it. "
    "Mr. Bennet made no answer. Do not you want to know who has taken it? cried his wife impatiently. You want to tell me, and I "
    "have no objection to hearing it. This was invitation enough. Why, my dear, you must know, Mrs. Long says that Netherfield is "
    "taken by a young man of large fortune from the north of England.",

    # A Tale of Two Cities, Charles Dickens (1859) — opening
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the "
    "epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the "
    "spring of hope, it was the winter of despair. We had everything before us, we had nothing before us, we were all going "
    "direct to Heaven, we were all going direct the other way. In short, the period was so far like the present period, that "
    "some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of "
    "comparison only. There were a king with a large jaw and a queen with a plain face, on the throne of England; there were a "
    "king with a large jaw and a queen with a fair face, on the throne of France. In both countries it was clearer than crystal "
    "to the lords of the State preserves of loaves and fishes, that things in general were settled for ever.",

    # Alice's Adventures in Wonderland, Lewis Carroll (1865) — opening of Chapter 1
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. "
    "Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it. "
    "And what is the use of a book, thought Alice, without pictures or conversations? "
    "So she was considering in her own mind, whether the pleasure of making a daisy-chain would be worth the trouble of getting "
    "up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her. "
    "There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to "
    "itself, Oh dear! Oh dear! I shall be late! But when the Rabbit actually took a watch out of its waistcoat-pocket, and looked "
    "at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a "
    "rabbit with either a waistcoat-pocket, or a watch to take out of it. Burning with curiosity, she ran across the field after "
    "it, and was just in time to see it pop down a large rabbit-hole under the hedge.",

    # The Adventures of Sherlock Holmes, A. C. Doyle (1892) — from A Scandal in Bohemia
    "To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any other name. "
    "In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene "
    "Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind. "
    "He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have "
    "placed himself in a false position. He never spoke of the softer passions, save with a gibe and a sneer. They were admirable "
    "things for the observer, excellent for drawing the veil from men's motives and actions. But for the trained reasoner to "
    "admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which "
    "might throw a doubt upon all his mental results.",

    # Frankenstein, Mary Shelley (1818) — letter from Walton
    "You will rejoice to hear that no disaster has accompanied the commencement of an enterprise which you have regarded with "
    "such evil forebodings. I arrived here yesterday, and my first task is to assure my dear sister of my welfare and increasing "
    "confidence in the success of my undertaking. I am already far north of London, and as I walk in the streets of Petersburgh, "
    "I feel a cold northern breeze play upon my cheeks, which braces my nerves, and fills me with delight. Do you understand "
    "this feeling? This breeze, which has travelled from the regions towards which I am advancing, gives me a foretaste of those "
    "icy climes. Inspirited by this wind of promise, my daydreams become more fervent and vivid. I try in vain to be persuaded "
    "that the pole is the seat of frost and desolation; it ever presents itself to my imagination as the region of beauty and "
    "delight.",

    # Moby Dick, Herman Melville (1851) — opening
    "Call me Ishmael. Some years ago — never mind how long precisely — having little or no money in my purse, and nothing "
    "particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. "
    "It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the "
    "mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin "
    "warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, "
    "that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically "
    "knocking people's hats off — then, I account it high time to get to sea as soon as I can. This is my substitute for pistol "
    "and ball. With a philosophical flourish Cato throws himself upon his sword; I quietly take to the ship.",
]


def load_human_texts(path):
    if not path:
        return list(BUNDLED_HUMAN_TEXTS)

    raw = Path(path).read_text(encoding="utf-8")
    # split on blank lines (the validator's PromptDataset also serves one "sample" at a time)
    parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
    # keep only substantial samples so the three-way split works
    return [p for p in parts if len(p.split()) >= 120 and len(get_sentences(p)) >= 5]


def sanity_check_labels(text, labels):
    if len(labels) != len(text.split()):
        return False
    # text has to be non-trivial
    if len(labels) < 20:
        return False
    return True


def generate(args):
    random.seed(args.seed)

    sources = load_human_texts(args.human_text_file)
    sources = [s for s in sources if len(get_sentences(s)) >= 5]
    if not sources:
        raise SystemExit(
            "No viable human source texts (need at least 5 sentences each). "
            "Supply --human-text-file FILE with paragraphs separated by blank lines."
        )
    print(f"Using {len(sources)} human source texts")

    rows = []

    def add(category, text_getter):
        target = getattr(args, "n_" + category.replace("-", "_"))
        attempts = 0
        while len([r for r in rows if r["category"] == category]) < target and attempts < target * 3:
            attempts += 1
            i = len([r for r in rows if r["category"] == category]) + 1
            t0 = time.time()
            try:
                text, labels = text_getter()
            except Exception as e:
                print(f"  [{category} {i}/{target}] FAILED: {e}")
                continue
            if not sanity_check_labels(text, labels):
                print(f"  [{category} {i}/{target}] SKIP (label/word mismatch or too short)")
                continue
            rows.append({
                "category": category,
                "text": text,
                "segmentation_labels": str(labels),
                "data_source": args.data_source,
            })
            elapsed = time.time() - t0
            ai = sum(labels)
            print(
                f"  [{category} {i}/{target}] ok — {len(labels)} words "
                f"({ai} AI / {len(labels) - ai} human), {elapsed:.1f}s"
            )

    if args.n_ai_middle > 0:
        print(f"\nGenerating {args.n_ai_middle} AI-in-the-middle samples ...")
        add("ai-middle", lambda: regenerated_in_the_middle(
            args.ollama_url, args.ollama_model, random.choice(sources),
        ))

    if args.n_pure_ai > 0:
        print(f"\nGenerating {args.n_pure_ai} pure-AI samples ...")
        add("pure-ai", lambda: pure_ai_sample(
            args.ollama_url, args.ollama_model,
            # take the first 2 sentences of a human sample as a seed prompt
            " ".join(get_sentences(random.choice(sources))[:2]),
        ))

    if args.n_pure_human > 0:
        print(f"\nAdding {args.n_pure_human} pure-human samples ...")
        add("pure-human", lambda: pure_human_sample(random.choice(sources)))

    if not rows:
        raise SystemExit("No rows generated — check Ollama is reachable and the model is pulled.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "segmentation_labels", "data_source", "category"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} labeled rows to {out}")
    print(
        "\nNext step:\n"
        f"  python scripts/eval_prediction_modes.py --csv {out} "
        "--text-col text --labels-col segmentation_labels"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ollama-model", default="llama3.2",
                    help="Ollama model name. Must already be pulled. Default: llama3.2")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--n-ai-middle", type=int, default=20,
                    help="Number of AI-in-the-middle samples (hardest case; default 20).")
    ap.add_argument("--n-pure-ai", type=int, default=5,
                    help="Number of pure-AI samples (labels all 1).")
    ap.add_argument("--n-pure-human", type=int, default=5,
                    help="Number of pure-human samples (labels all 0).")
    ap.add_argument("--human-text-file", default=None,
                    help="Optional: path to a UTF-8 text file of human paragraphs separated "
                         "by blank lines. If omitted, uses a small bundled set of Gutenberg excerpts.")
    ap.add_argument("--data-source", default="synthetic",
                    help="Value for the CSV's data_source column. Set to 'common_crawl' if you want "
                         "the eval script to treat these rows as out-of-domain for the 0.9 gate check.")
    ap.add_argument("--output", default="data/labeled_batch.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Contacting Ollama at {args.ollama_url} with model='{args.ollama_model}'")
    try:
        # smoke-test the endpoint
        r = requests.get(f"{args.ollama_url.rstrip('/')}/api/tags", timeout=10)
        r.raise_for_status()
        pulled = [m.get("name") for m in r.json().get("models", [])]
        if args.ollama_model not in pulled and f"{args.ollama_model}:latest" not in pulled:
            print(
                f"WARNING: model '{args.ollama_model}' does not appear in ollama list. "
                f"Pulled models: {pulled}\n"
                f"Run: ollama pull {args.ollama_model}"
            )
    except Exception as e:
        raise SystemExit(
            f"Could not reach Ollama at {args.ollama_url}. Is Ollama running? ({e})"
        )

    generate(args)


if __name__ == "__main__":
    main()
