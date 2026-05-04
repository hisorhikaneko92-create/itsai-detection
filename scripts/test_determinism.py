print("SCRIPT STARTED")

"""
Determinism test for HSSD seam detector.

Checks that the model produces bit-identical predictions when the same
document is sent:
  1. Three times alone (rules out random non-determinism like dropout)
  2. Alone vs. batched with other documents (rules out bf16 padding bugs)
  3. fp32 vs. bf16 (shows the impact of mixed precision)

If any of these FAIL, the SN32 determinism gate will reject your miner
and reward will be 0 regardless of F1.

Usage:
    python scripts/test_determinism.py \\
        --checkpoint models/seam_detector_v3/best \\
        --model-name models/deberta-v3-large
"""
import argparse
print("1. argparse OK")
import sys
print("2. sys OK")
from pathlib import Path
print("3. Path OK")

import torch
print("4. torch OK")
from transformers import AutoTokenizer
print("5. transformers OK")

sys.path.insert(0, str(Path(__file__).resolve().parent))
print("6. path inserted")

from train_seam_detector import SeamDetector
print("7. SeamDetector OK")

from peft import PeftModel
print("8. PeftModel OK")


def load_model(checkpoint_dir, model_name, device):
    base = SeamDetector(model_name=model_name)
    model = PeftModel.from_pretrained(
        base, str(Path(checkpoint_dir) / "lora_adapter"),
        is_trainable=False,
    ).to(device)
    model.eval()
    return model


def predict(model, tokenizer, texts, device, use_bf16=False, max_length=512):
    """Run inference on a list of texts. Returns list of word-level label lists."""
    encoded = tokenizer(
        [t.split() for t in texts],
        is_split_into_words=True,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else
        torch.amp.autocast(device_type="cuda", enabled=False)
    )

    with torch.no_grad(), autocast_ctx:
        paths = model(input_ids, attention_mask)

    # Map token-level paths back to word-level (matches validator scoring)
    results = []
    for i, path in enumerate(paths):
        word_ids = encoded.word_ids(i)
        n_words = len(texts[i].split())
        word_labels = [0] * n_words
        seen = set()
        for ti, wid in enumerate(word_ids):
            if wid is not None and wid not in seen and ti < len(path):
                word_labels[wid] = path[ti]
                seen.add(wid)
        results.append(word_labels)
    return results


def compare_predictions(p1, p2, label1, label2):
    """Compare two prediction lists; print first 5 differences if any."""
    if p1 == p2:
        print(f"  {label1} == {label2} : MATCH ✅")
        return True
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(p1, p2)) if a != b]
    print(f"  {label1} != {label2} : MISMATCH ❌  ({len(diffs)} word(s) differ)")
    for i, a, b in diffs[:5]:
        print(f"      word {i:3d}: {label1}={a}  {label2}={b}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to checkpoint dir containing lora_adapter/")
    ap.add_argument("--model-name", default="models/deberta-v3-large")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, args.model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Test documents of varying lengths
    target = (
        "The cat sat on the mat while reading a book in the morning. "
        "Then the assistant generated a coherent paragraph based on the "
        "user's request, providing detailed information about various "
        "topics including science, history, and current events."
    )
    short_doc = "Short doc filler " * 10           # ~30 words
    long_doc = "Long doc filler " * 100            # ~200 words

    print(f"\nTarget document: {len(target.split())} words")

    # ============================================================
    # Test 1: same doc, alone, 3 trials
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1 — Same doc, alone, 3 trials (catches dropout/random)")
    print("=" * 60)
    solo_results = []
    for trial in range(3):
        res = predict(model, tokenizer, [target], device, use_bf16=False)[0]
        solo_results.append(res)
        print(f"  Trial {trial+1}: first 25 labels = {res[:25]}")

    test1_pass = (solo_results[0] == solo_results[1] == solo_results[2])
    print(f"\n  Result: {'PASS ✅' if test1_pass else 'FAIL ❌'}")

    # ============================================================
    # Test 2: alone vs batched (fp32) — the critical SN32 test
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2 — Alone vs batched (fp32) — SN32 GATE TEST")
    print("=" * 60)

    alone_pred = predict(model, tokenizer, [target], device, use_bf16=False)[0]

    batched = [short_doc, target, long_doc]
    batched_pred = predict(model, tokenizer, batched, device, use_bf16=False)[1]

    test2_pass = compare_predictions(alone_pred, batched_pred,
                                        "alone(fp32)", "batched(fp32)")

    # ============================================================
    # Test 3: alone vs batched (bf16) — current default behavior
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3 — Alone vs batched (bf16) — current default")
    print("=" * 60)

    alone_bf16 = predict(model, tokenizer, [target], device, use_bf16=True)[0]
    batched_bf16 = predict(model, tokenizer, batched, device, use_bf16=True)[1]

    test3_pass = compare_predictions(alone_bf16, batched_bf16,
                                        "alone(bf16)", "batched(bf16)")

    # ============================================================
    # Test 4: same doc, different position in batch
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 4 — Same doc at different batch positions (bf16)")
    print("=" * 60)

    pos0 = predict(model, tokenizer, [target, long_doc, short_doc],
                    device, use_bf16=True)[0]
    pos2 = predict(model, tokenizer, [long_doc, short_doc, target],
                    device, use_bf16=True)[2]

    test4_pass = compare_predictions(pos0, pos2, "pos=0(bf16)", "pos=2(bf16)")

    # ============================================================
    # Final report
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    print(f"  Test 1 (random non-determinism): {'PASS ✅' if test1_pass else 'FAIL ❌'}")
    print(f"  Test 2 (alone vs batched, fp32): {'PASS ✅' if test2_pass else 'FAIL ❌'}")
    print(f"  Test 3 (alone vs batched, bf16): {'PASS ✅' if test3_pass else 'FAIL ❌'}")
    print(f"  Test 4 (batch position, bf16):   {'PASS ✅' if test4_pass else 'FAIL ❌'}")

    if all([test1_pass, test2_pass, test3_pass, test4_pass]):
        print("\n  All deterministic. The SN32 gate failure is NOT in the model.")
        print("  Investigate: server-side batching, response format, timeouts.")
    elif test2_pass and test3_pass:
        print("\n  Model is fully deterministic — issue is elsewhere.")
    elif test2_pass and not test3_pass:
        print("\n  Confirmed: bf16 + batching breaks determinism.")
        print("  Fix: disable bf16 in inference (run fp32 only).")
    elif not test2_pass:
        print("\n  Even fp32 has determinism issues — deeper problem.")
        print("  Try torch.use_deterministic_algorithms(True).")


if __name__ == "__main__":
    main()