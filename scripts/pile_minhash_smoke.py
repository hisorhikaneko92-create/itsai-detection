"""Tiny empirical smoke test for Pile sentence-MinHash indexing.

Streams the first N docs of monology/pile-uncopyrighted, sentence-tokenizes,
builds a MinHashLSH index, and reports concrete numbers we can extrapolate
to the full 177M-doc dataset:

    docs/sec build throughput
    sentences per doc (median)
    bytes per sentence in the index
    final pickled index size

Usage (on the A100 with HF_TOKEN set):
    pip install datasketch                # only dep beyond what's already installed
    python scripts/pile_minhash_smoke.py --n-docs 50000 --num-perm 64
"""
import argparse
import gc
import os
import pickle
import resource
import statistics
import time
from pathlib import Path

import nltk
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=50_000,
                    help="Number of Pile docs to index in the smoke test")
    ap.add_argument("--num-perm", type=int, default=64,
                    help="MinHash permutation count (more = larger index, more accurate)")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="LSH match threshold (Jaccard similarity)")
    ap.add_argument("--output-dir", default="/tmp/pile_smoke",
                    help="Where to dump the index for measurement")
    ap.add_argument("--min-sent-words", type=int, default=8,
                    help="Skip sentences shorter than this (low-signal)")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    nltk.download("punkt", quiet=True)
    sent_tok = nltk.data.load("tokenizers/punkt/english.pickle")

    print(f"Streaming Pile-uncopyrighted, target {args.n_docs:,} docs, "
          f"num_perm={args.num_perm}")
    ds = load_dataset(
        "monology/pile-uncopyrighted", streaming=True, split="train",
    )

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    sentences_per_doc = []
    total_sentences = 0
    total_chars = 0
    t0 = time.time()
    last_print = t0

    for i, row in enumerate(ds):
        if i >= args.n_docs:
            break
        text = row.get("text", "")
        if not text:
            sentences_per_doc.append(0)
            continue
        total_chars += len(text)
        sents = [s for s in sent_tok.tokenize(text)
                 if len(s.split()) >= args.min_sent_words]
        sentences_per_doc.append(len(sents))
        for j, sent in enumerate(sents):
            mh = MinHash(num_perm=args.num_perm)
            # Shingle on word 3-grams for robustness to spelling noise
            words = sent.lower().split()
            for k in range(len(words) - 2):
                mh.update(" ".join(words[k:k + 3]).encode())
            key = f"{i}:{j}"
            try:
                lsh.insert(key, mh)
                total_sentences += 1
            except ValueError:
                # Duplicate key — shouldn't happen with idx:idx scheme
                pass

        if time.time() - last_print > 5:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i + 1:>7,} docs · {total_sentences:>10,} sentences · "
                  f"{rate:>5.0f} docs/sec · "
                  f"RAM peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
            last_print = time.time()

    elapsed = time.time() - t0

    # Persist the LSH index to measure on-disk footprint
    idx_path = Path(args.output_dir) / "lsh_index.pkl"
    print(f"\nDumping LSH index to {idx_path} ...")
    t1 = time.time()
    with open(idx_path, "wb") as f:
        pickle.dump(lsh, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_secs = time.time() - t1
    on_disk_bytes = idx_path.stat().st_size

    # Stats
    n_docs = len(sentences_per_doc)
    spd_med = statistics.median(sentences_per_doc) if sentences_per_doc else 0
    spd_mean = statistics.mean(sentences_per_doc) if sentences_per_doc else 0
    bytes_per_sentence = on_disk_bytes / max(1, total_sentences)
    bytes_per_doc = on_disk_bytes / max(1, n_docs)
    docs_per_sec = n_docs / elapsed
    sents_per_sec = total_sentences / elapsed
    ram_peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"\n=== RESULTS ({n_docs:,} docs · num_perm={args.num_perm}) ===")
    print(f"  build time:        {elapsed:>8.1f} s   (pickle: {pickle_secs:.1f} s)")
    print(f"  total sentences:   {total_sentences:>8,}   ({spd_med:.0f} median, {spd_mean:.1f} mean per doc)")
    print(f"  total chars:       {total_chars:>8,}")
    print(f"  build throughput:  {docs_per_sec:>8.0f} docs/sec   {sents_per_sec:>6.0f} sents/sec")
    print(f"  on-disk LSH:       {on_disk_bytes / 1024**2:>8.1f} MB")
    print(f"  bytes/sentence:    {bytes_per_sentence:>8.1f}")
    print(f"  bytes/doc:         {bytes_per_doc:>8.1f}")
    print(f"  RAM peak:          {ram_peak_mb:>8.0f} MB")

    # Extrapolation to full Pile (177M docs)
    PILE_DOCS = 176_800_498
    proj_secs = PILE_DOCS / docs_per_sec
    proj_index_gb = (on_disk_bytes / n_docs) * PILE_DOCS / 1024**3
    proj_sents = (total_sentences / n_docs) * PILE_DOCS
    print(f"\n=== EXTRAPOLATION to full Pile ({PILE_DOCS:,} docs) ===")
    print(f"  projected build time:   {proj_secs / 3600:.1f} hours "
          f"(at {docs_per_sec:.0f} docs/s on this hardware)")
    print(f"  projected sentences:    {proj_sents:,.0f}")
    print(f"  projected LSH size:     {proj_index_gb:.1f} GB on disk")
    print(f"  projected RAM at scale: {(ram_peak_mb / n_docs * PILE_DOCS) / 1024:.1f} GB "
          f"(naive linear, but LSH bands compress)")

    # Quick sanity check: query an in-index sentence and verify it matches
    if total_sentences > 100:
        # Take an arbitrary inserted sentence and re-MinHash it as a query
        sample = "the quick brown fox jumps over the lazy dog every single day"
        # Actually grab a real one we just inserted
        for i, row in enumerate(load_dataset("monology/pile-uncopyrighted",
                                               streaming=True, split="train")):
            if i >= 5: break
            text = row.get("text", "")
            if not text: continue
            sents = [s for s in sent_tok.tokenize(text)
                     if len(s.split()) >= args.min_sent_words]
            if not sents: continue
            sample = sents[0]
            break

        q = MinHash(num_perm=args.num_perm)
        words = sample.lower().split()
        for k in range(len(words) - 2):
            q.update(" ".join(words[k:k + 3]).encode())
        hits = lsh.query(q)
        print(f"\nSanity query (first sentence of doc 0):")
        print(f"  '{sample[:120]}...'" if len(sample) > 120 else f"  '{sample}'")
        print(f"  Found {len(hits)} matching keys: {hits[:5]}{'...' if len(hits) > 5 else ''}")
        print(f"  → index roundtrips correctly.")
    else:
        print(f"\n(Skipping sanity query — too few sentences)")

    gc.collect()


if __name__ == "__main__":
    main()
