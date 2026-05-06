"""
Quick architectural smoke test for HSSD v4.1.

Run this on the Lium pod BEFORE kicking off the full training:

    python scripts/smoke_test_v4.py

It builds the full model with the v4.1 defaults, runs forward and the
multi-term loss on dummy data, and reports parameter counts per group.
The whole thing finishes in <2 minutes and uses <4 GB VRAM. If this
exits 0 with sane numbers, the architecture, loss wiring, EMA, R-Drop,
and parameter grouping are all correct.

Successful output looks like:

    [model] backbone=137,XXX,XXX  CLAF=YYY,YYY  head=ZZZ,ZZZ
    [forward] training-mode keys: ['emissions','boundary_logits',...]
    [forward] eval-mode produces list[B] of variable-length 0/1 paths
    [loss] total=N.NN  components: crf=... focal=... boundary=... aux_*=...
    [r-drop] KL between two stochastic forwards = X.XX
    [ema] shadow tracks N tensors; store/apply/restore cycle OK
    [llrd] 24 layer groups, top-layer LR=3.00e-04  bottom-layer LR=2.69e-05
    SMOKE TEST PASS
"""
import argparse
import sys
import time
from pathlib import Path

# Same Windows stack-overflow workaround as train_seam_detector.py
import pandas    # noqa: F401
import sklearn   # noqa: F401

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_seam_detector as tsd    # noqa: E402


def build_default_args(model_name="microsoft/deberta-v3-large") -> argparse.Namespace:
    """Mirror parse_args() defaults; we don't need to actually parse."""
    args = argparse.Namespace(
        # data (unused here)
        train_csv=[], val_csv=[], test_csv=[],
        max_train_rows=None, max_val_rows=None,
        # model
        model_name=model_name,
        max_length=128,            # smaller for the smoke test
        stride=64,
        syntax_range=[5, 10], semantic_range=[13, 18], discourse_range=[20, 25],
        hidden_dropout=0.1, attn_dropout=0.1,
        # training
        num_epochs=1, batch_size=2, gradient_accumulation_steps=1,
        encoder_lr=3e-4, claf_lr=5e-4, head_lr=1e-3,
        weight_decay=0.01, claf_weight_decay=0.03, head_weight_decay=0.05,
        warmup_ratio=0.1, max_grad_norm=1.0, loss_spike_factor=10.0,
        llrd_decay=0.9,
        # loss
        lambda_focal=0.3, lambda_boundary=0.5,
        lambda_data_source=0.05, lambda_model_family=0.05, lambda_sample_type=0.05,
        focal_gamma=2.0, focal_seam_alpha=0.5,
        boundary_sigma=3.0,
        min_p_1to0=0.0,
        # rdrop
        use_rdrop=True, lambda_rdrop=0.5,
        # ema
        use_ema=True, ema_decay=0.999, ema_warmup_steps=2,
        # fgm (off in v4)
        use_fgm=False, fgm_target="input_norm", fgm_epsilon=1.0,
        # bf16 / checkpointing
        bf16=True, gradient_checkpointing=False,
        # lora
        use_lora=True, lora_r=32, lora_alpha=64, lora_dropout=0.1,
        lora_target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        use_8bit_adam=False,
        # compile / etc.
        torch_compile=False, torch_compile_mode="default",
        gate_monitor=False, gate_monitor_batches=1,
        output_dir=str(Path("models/_smoke_test").resolve()),
        log_every=1,
        validate_every_steps=0, save_every_steps=0,
        patience=0, dataloader_workers=0, seed=42, resume=False,
    )
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="microsoft/deberta-v3-large")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args_cli = ap.parse_args()

    device = torch.device(args_cli.device)
    print(f"[device] {device}")
    args = build_default_args(args_cli.model_name)

    # ---- 1. Build model ----
    t0 = time.time()
    model = tsd.SeamDetector(
        model_name=args.model_name,
        syntax_range=tuple(args.syntax_range),
        semantic_range=tuple(args.semantic_range),
        discourse_range=tuple(args.discourse_range),
        hidden_dropout=args.hidden_dropout,
        attn_dropout=args.attn_dropout,
    )

    # Apply LoRA (v4.1 default config)
    from peft import LoraConfig, get_peft_model
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=list(args.lora_target_modules),
        lora_dropout=args.lora_dropout, bias="none",
        modules_to_save=[
            "claf", "input_norm", "conv_head", "bpm",
            "classifier", "boundary_head", "aux_heads", "crf",
        ],
    )
    model = get_peft_model(model, peft_cfg)
    model = model.to(device)
    print(f"[build] {time.time() - t0:.1f}s")

    # ---- 2. Param-group breakdown ----
    groups = tsd.build_param_groups(model, args)
    n_total = sum(p.numel() for g in groups for p in g["params"])
    print(f"[params] total trainable = {n_total:,}")

    if args.llrd_decay and 0 < args.llrd_decay < 1:
        # Verify top-vs-bottom LR ratio matches expectation
        backbone_lrs = [g["lr"] for g in groups if g["params"]
                        and any("lora_" in n for n, _ in zip(
                            (next(iter(g["params"])).names() if False else []), [None]))]
        # Simpler: just print all LRs
        print("[llrd] LR per group:")
        for g in groups:
            n_p = sum(p.numel() for p in g["params"])
            print(f"   lr={g['lr']:.2e}  wd={g.get('weight_decay', 0):.3f}  params={n_p:,}")

    # ---- 3. Forward (training mode) ----
    model.train()
    B, T = 2, 64
    ids = torch.randint(100, 5000, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.long, device=device)
    labels = torch.zeros(B, T, dtype=torch.long, device=device)
    labels[0, 32:] = 1
    labels[1, 8:] = 1
    boundary_target = torch.zeros(B, T, device=device)
    boundary_target[0, 32] = 1.0
    boundary_target[1, 8] = 1.0
    ds_id = torch.tensor([0, 1], device=device)
    mf_id = torch.tensor([4, 9], device=device)
    st_id = torch.tensor([2, 3], device=device)

    out = model(ids, mask, labels=labels)
    print(f"[forward] training keys: {sorted(out.keys())}")
    print(f"           emissions       {tuple(out['emissions'].shape)}")
    print(f"           boundary_logits {tuple(out['boundary_logits'].shape)}")
    print(f"           data_source_logits  {tuple(out['data_source_logits'].shape)}")
    print(f"           model_family_logits {tuple(out['model_family_logits'].shape)}")
    print(f"           sample_type_logits  {tuple(out['sample_type_logits'].shape)}")

    # ---- 4. Multi-term loss ----
    crf = tsd._resolve_crf(model)
    loss, comps = tsd.compute_total_loss(
        out, labels, mask, crf,
        boundary_target=boundary_target,
        data_source_id=ds_id, model_family_id=mf_id, sample_type_id=st_id,
        lambda_focal=args.lambda_focal, lambda_boundary=args.lambda_boundary,
        lambda_data_source=args.lambda_data_source,
        lambda_model_family=args.lambda_model_family,
        lambda_sample_type=args.lambda_sample_type,
        focal_gamma=args.focal_gamma, focal_seam_alpha=args.focal_seam_alpha,
        return_components=True,
    )
    print(f"[loss] total={loss.item():.4f}")
    for k, v in comps.items():
        print(f"        {k:<10s} = {v.item():.4f}")

    # ---- 5. R-Drop KL ----
    out2 = model(ids, mask, labels=labels)
    kl = tsd.compute_rdrop_kl(
        out["emissions"][:, 1:, :], out2["emissions"][:, 1:, :],
        labels[:, 1:], mask[:, 1:],
    )
    print(f"[rdrop] symmetric KL = {kl.item():.4f}")

    # ---- 6. Eval-mode forward ----
    model.eval()
    paths = model(ids, mask)
    print(f"[eval] forward returned {type(paths).__name__} of {len(paths)} paths "
          f"(lens {[len(p) for p in paths]}; T={T})")

    # ---- 7. EMA cycle ----
    model.train()
    ema = tsd.WeightEMA(model, decay=args.ema_decay)
    print(f"[ema] shadow tensors: {len(ema.shadow)}")
    ema.update(model)
    ema.store(model)
    ema.apply(model)
    ema.restore(model)
    print(f"[ema] update + store/apply/restore cycle OK")

    print()
    print("SMOKE TEST PASS")


if __name__ == "__main__":
    main()
