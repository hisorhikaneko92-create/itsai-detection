"""Probe the SN32 HSSD inference server for the consistency-gate behavior
that gates miner reward via count_penalty in detection/validator/reward.py.

Two probes:

  Probe 1 — cache hit / same-batch determinism
    POST the same single-text twice. Both responses must be byte-identical
    after round(_, 2). If they differ, either the per-text cache is broken
    or the model is non-deterministic across calls.

  Probe 2 — cross-batch consistency (the actual gate 2 simulation)
    POST a small "check" batch (~5 texts), then a large "main" batch (~120
    texts) containing the same 5 texts plus padding. The validator's
    count_penalty rounds both responses to 2 decimals and requires them
    to match for shared texts. If a single word diverges, the entire
    batch's reward is zeroed.

Cross-platform: stdlib only (urllib + json). Run on the GPU PC where
the inference server is actually serving.

Usage:
    python scripts/probe_inference_determinism.py
    python scripts/probe_inference_determinism.py --url http://127.0.0.1:18091/predict
"""
import argparse
import json
import sys
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:18091/predict"


def post(url, payload, timeout=120):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, method="POST", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def diverge_at_r2(a, b):
    if len(a) != len(b):
        return -1
    return sum(1 for x, y in zip(a, b) if round(x, 2) != round(y, 2))


def probe1(url):
    text = ("Researchers studying large language models have observed that "
            "fluency alone is no longer a reliable signal of human authorship "
            "across many writing tasks of moderate length and complexity.")
    print("Probe 1 — same text, two calls, expect identical predictions")
    print(f"  url   : {url}")
    print(f"  words : {len(text.split())}")
    r1 = post(url, {"texts": [text]})
    time.sleep(2)
    r2 = post(url, {"texts": [text]})
    p1 = r1["predictions"][0]
    p2 = r2["predictions"][0]
    d = diverge_at_r2(p1, p2)
    if d == 0:
        print("  result: PASS — predictions identical at round(2)")
    elif d == -1:
        print(f"  result: FAIL — length mismatch ({len(p1)} vs {len(p2)})")
    else:
        print(f"  result: FAIL — {d} words diverge at round(2)")
        # Show first 3 diverging
        shown = 0
        for i, (x, y) in enumerate(zip(p1, p2)):
            if round(x, 2) != round(y, 2):
                print(f"    word[{i}]: r1={x:.4f}  r2={y:.4f}  r2-rounded={round(x,2)} vs {round(y,2)}")
                shown += 1
                if shown >= 3:
                    break
    return d == 0


def probe2(url):
    seed_texts = [
        "Machine learning models can produce surprisingly fluent text these days, with grammar and vocabulary that often pass for human-authored prose.",
        "The cat sat on the mat and watched the rain fall outside, occasionally tilting its head when a louder gust of wind rattled the old window frame.",
        "Researchers continue to investigate the limits of large language models, particularly with respect to long-context coherence and factual grounding.",
        "An investigation into the matter revealed several interesting findings that the committee felt warranted further independent review by a third party.",
        "However the data suggests a more nuanced interpretation is warranted before drawing strong conclusions about the underlying causal mechanism here.",
    ]
    # Build a 120-text main batch containing the seeds + filler
    filler_pool = [
        "Filler sentence number one is reasonably long and contains a few clauses to make it varied.",
        "A second filler sentence that talks about completely unrelated topics like weather and gardening.",
        "Another filler used to pad the batch up to a representative validator-style size for testing.",
    ]
    main_batch = []
    for i in range(120 - len(seed_texts)):
        main_batch.append(filler_pool[i % len(filler_pool)] + f" idx={i}")
    # Insert seeds at known positions
    for i, t in enumerate(seed_texts):
        main_batch.insert(20 + i * 18, t)

    print()
    print(f"Probe 2 — check batch ({len(seed_texts)}) then main batch ({len(main_batch)})")
    r_check = post(url, {"texts": seed_texts})
    time.sleep(2)
    r_main = post(url, {"texts": main_batch})

    all_pass = True
    for i, t in enumerate(seed_texts):
        check_p = r_check["predictions"][i]
        try:
            main_idx = main_batch.index(t)
        except ValueError:
            print(f"  [{i}] seed text not present in main batch (probe bug)")
            all_pass = False
            continue
        main_p = r_main["predictions"][main_idx]
        d = diverge_at_r2(check_p, main_p)
        if d == 0:
            print(f"  [{i}] words={len(check_p):3d} diverge=0  ok")
        elif d == -1:
            print(f"  [{i}] LENGTH MISMATCH check={len(check_p)} main={len(main_p)}")
            all_pass = False
        else:
            print(f"  [{i}] words={len(check_p):3d} diverge={d}  FAIL — gate 2 would zero reward")
            all_pass = False
            shown = 0
            for j, (x, y) in enumerate(zip(check_p, main_p)):
                if round(x, 2) != round(y, 2):
                    print(f"      word[{j}]: check={x:.4f} main={y:.4f}  r2={round(x,2)} vs {round(y,2)}")
                    shown += 1
                    if shown >= 3:
                        break

    print()
    if all_pass:
        print("  overall: PASS — count_penalty would NOT trip on a real validator round")
    else:
        print("  overall: FAIL — count_penalty WILL zero reward on a real validator round")
        print("           (the per-text cache is missing or batch-dependent drift is not being caught)")
    return all_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()
    print(f"SN32 inference-server determinism probe")
    print(f"=======================================")
    ok1 = probe1(args.url)
    ok2 = probe2(args.url)
    print()
    if ok1 and ok2:
        print("All probes PASS. Gate 2 should not be the blocker.")
        sys.exit(0)
    print("At least one probe FAILED. Likely cause of penalty=0 on validators.")
    sys.exit(2)


if __name__ == "__main__":
    main()
