"""
Train the Hybrid Semantic Seam Detector (HSSD) — v4.

v4 fixes four structural bugs in v3 (CLAF cross-range degeneracy,
per-sequence CRF / per-token focal scale mismatch, LoRA encoder LR set
at the full-finetune scale, hard-clamp emissions with zero gradient on
saturation) and adds capacity along three axes (per-position CLAF
attention, 2-layer dilated conv stack, BPM gate + boundary head + aux
heads). The cumulative effect is a strictly larger and properly-trained
model on the same data.

Architecture
  * DeBERTa-v3-Large backbone — hidden_dropout/attn_dropout = 0.1 at
    train(), auto-disabled in eval() (validator's determinism gate is
    checked in eval(), so the gate is unaffected).
  * LoRA r=16, alpha=32, dropout=0.1 on q,k,v projections (was q,v in
    v3). Encoder LR = 3e-4 (was 1e-5 — that legacy LR was for full FT,
    LoRA needs ~30x more to learn from zero-init).
  * CLAF v3 — per-position multi-head self-attention over each layer
    range (vs v2's single-query doc-level summary), three SEPARATE
    cross-range projections (vs v2's shared projection that produced
    near-identical streams and starved the gate).
  * Multi-scale dilated conv head — TWO layers stacked. Layer 1 is
    kernel-variety (k=3,5,7 d=1; k=3 d=8) over backbone features. Layer
    2 is dilation-variety (d=1,4,8,16) over layer-1 output. Output is
    layer1 + scale*layer2. Effective receptive field ≈ 65 tokens
    (vs v3's RF=17 ceiling).
  * Boundary Prototype Memory v2 — 64 prototypes (vs 16). Returns BOTH
    per-prototype similarities AND a per-position multiplicative gate
    that modulates the conv features before concatenation.
  * Soft tanh-clamp on emissions (gradient-passing) instead of v3's
    hard `clamp(-15,15)` which had zero gradient at saturation and
    trapped wrong-sign emissions in late training.
  * Boundary head — Linear(feat_dim → 192) + GELU + Linear(192 → 1)
    trained on a Gaussian-shaped target around the GT seam. Provides
    direct gradient on seam localization, which the per-token focal
    loss alone cannot.
  * Auxiliary heads — data_source / model_family / sample_type CE off
    the mean-pooled CLAF output. Free supervision from existing CSV
    columns; `data_source` in particular shapes pile-vs-CC discrimination
    (the validator's CC out-of-domain F1 gate).
  * CRF with NO transition floor (--min-p-1to0=0). Rebalanced data
    means the CRF can learn correct marginals; the v3 floor of 0.05
    fights start_transitions on `ai_then_human` rows.

Training loss
    L = L_CRF_per_token
      + λ_focal     * L_focal
      + λ_boundary  * L_boundary       (BCE vs Gaussian target)
      + λ_ds        * L_data_source    (CE)
      + λ_mf        * L_model_family   (CE)
      + λ_st        * L_sample_type    (CE)

Defaults: λ_focal=0.3, λ_boundary=0.5, λ_ds=λ_mf=λ_st=0.05.

Reads CSVs produced by scripts/build_training_dataset.py with columns
  text, segmentation_labels, data_source, sample_type, model_name, ...

Validation metrics (unchanged from v3)
  * Mean Seam Offset  (target: < 2.5 words)
  * F1 @ 5 Words      (target: > 0.92, real top-miner bar > 0.97)
  * Token-Level F1    (sanity check)

Usage
    pip install pytorch-crf peft
    pip install bitsandbytes        # optional, saves ~2 GB VRAM via 8-bit Adam

    python scripts\\train_seam_detector.py `
        --train-csv data\\Training_Dataset\\train_rebalanced_oversampled.csv `
        --val-csv   data\\Training_Dataset\\val_rebalanced.csv `
        --output-dir models\\seam_detector_v4 `
        --num-epochs 3
"""

# Pre-load pandas + sklearn before transformers. Same Windows stack-overflow
# workaround we already use in neurons/miners/deberta_classifier.py: when
# transformers later lazy-imports candidate_generator -> sklearn -> pandas
# in a deeply-nested call chain, the 1 MB main-thread stack on Windows
# blows up. Loading them on a fresh shallow stack here populates sys.modules
# so the later lazy chain hits cache and never recurses.
import pandas  # noqa: F401
import sklearn  # noqa: F401

# Silence "huggingface/tokenizers: process just got forked" noise.
# The fast tokenizers' rust workers detect DataLoader worker forks and
# disable themselves; we don't actually need rust parallelism here
# because all tokenization happens at __init__ time (cached) and
# __getitem__ is just a tensor slice. Setting this BEFORE the
# transformers import ensures it takes effect.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import json
import math
import os
import random
import sys
import time

# Some training rows have very long `text` fields (long Pile docs concatenated
# with AI continuations). The default 128 KB limit trips on these. Bump it
# unconditionally — we control the input files and parsing them is safe.
csv.field_size_limit(sys.maxsize)
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

try:
    from torchcrf import CRF
except ImportError:
    sys.exit(
        "Missing dependency 'pytorch-crf'.\n"
        "Install with:  pip install pytorch-crf"
    )

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError:
    sys.exit(
        "Missing dependency 'peft'.\n"
        "Install with:  pip install peft"
    )


# ---------------------------------------------------------------------------
# Model-family mapping for the auxiliary-head supervision.
# ---------------------------------------------------------------------------
# 10 buckets: keep this stable across runs. Aux head outputs match this
# index. Anything not matched falls into bucket 9 ("other / unknown").
MODEL_FAMILY_MAP = {
    # gemma
    "google/gemma": 0,
    # mistral / mixtral
    "mistralai/mistral": 1,
    "mistralai/mixtral": 1,
    # llama-derived (hermes, llama-3, openchat, etc.)
    "nousresearch/hermes": 2,
    "meta-llama/llama": 2,
    "openchat/": 2,
    # cohere command
    "cohere/command": 3,
    # microsoft phi
    "microsoft/phi": 4,
    # anthropic claude
    "anthropic/claude": 5,
    # openai gpt
    "openai/gpt": 6,
    # deepseek
    "deepseek/": 7,
    "deepseek-ai/": 7,
    # qwen / yi
    "qwen/": 8,
    "01-ai/yi": 8,
}
NUM_MODEL_FAMILIES = 10
NUM_DATA_SOURCES = 2          # 0=pile, 1=common_crawl
NUM_SAMPLE_TYPES = 4          # 0=pure_human, 1=pure_ai, 2=human_then_ai, 3=ai_then_human
SAMPLE_TYPE_MAP = {
    "pure_human":     0,
    "pure_ai":        1,
    "human_then_ai":  2,
    "ai_then_human":  3,
}
DATA_SOURCE_MAP = {
    "pile":          0,
    "common_crawl":  1,
}


def model_family_id(model_name: str) -> int:
    """Map a free-form model_name (e.g. 'google/gemma-2-27b-it') to one
    of the NUM_MODEL_FAMILIES family bucket ids. Substring match against
    MODEL_FAMILY_MAP keys; falls back to 9 (unknown)."""
    if not model_name:
        return NUM_MODEL_FAMILIES - 1
    n = model_name.strip().lower()
    for prefix, idx in MODEL_FAMILY_MAP.items():
        if n.startswith(prefix):
            return idx
    return NUM_MODEL_FAMILIES - 1


# ---------------------------------------------------------------------------
# CLAF v3 -- Cross-Layer Attention Fusion
# ---------------------------------------------------------------------------
# Differences from CLAF v2 (which had two structural bugs):
#   (Bug A) v2's cross-range layer applied the SAME projection output to
#       all three streams. After the residual the three streams were
#       nearly identical, so the downstream gate had no signal to
#       discriminate among. v3 uses THREE SEPARATE projections, one per
#       stream; the gate now sees three differentiated streams.
#   (Bug B) v2's per-range cross-attention used a single learned query
#       per range. Output was one vector per (batch, range), broadcast
#       to every position as a residual. So every token got the same
#       range-level context — CLAF was acting as a doc-level summarizer,
#       not a position-level fuser. v3 uses per-position queries (full
#       self-attention over each range's pooled hidden states), so the
#       range context at position i depends on what's around position i.
class CrossLayerAttentionFusionV3(nn.Module):
    """Per-position, per-range cross-attention with differentiated
    cross-range fusion and temperature-gated stream selection.

    Pipeline:
        h_{syn,sem,dis} = mean-pool over their hidden_state ranges
        a_{syn,sem,dis} = self-attention over each pooled stream
                          (per-position queries, key_padding_mask=attention_mask)
        h_*  ← LN(h_* + a_*)            # per-stream residual + LN
        h_*  ← h_* + GELU(W_*([h_syn;h_sem;h_dis]))
                                          # 3 separate cross-range projections
        gate = softmax(W_g([h_syn;h_sem;h_dis]) / tau,  dim=-1)
        fused = sum_k gate_k * h_k
    """

    def __init__(self, hidden_size: int = 1024,
                 num_heads: int = 8,
                 attn_dropout: float = 0.1,
                 syntax_range: Tuple[int, int] = (5, 10),
                 semantic_range: Tuple[int, int] = (13, 18),
                 discourse_range: Tuple[int, int] = (20, 25)):
        super().__init__()
        self.hidden_size = hidden_size
        self.syntax_range = syntax_range
        self.semantic_range = semantic_range
        self.discourse_range = discourse_range

        # Per-range self-attention. Three independent modules so each
        # stream's attention pattern can specialize.
        self.attn_syn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            batch_first=True, dropout=attn_dropout,
        )
        self.attn_sem = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            batch_first=True, dropout=attn_dropout,
        )
        self.attn_dis = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            batch_first=True, dropout=attn_dropout,
        )
        self.norm_syn = nn.LayerNorm(hidden_size)
        self.norm_sem = nn.LayerNorm(hidden_size)
        self.norm_dis = nn.LayerNorm(hidden_size)

        # Three SEPARATE cross-range projections (Bug A fix).
        self.xrange_proj_syn = nn.Linear(hidden_size * 3, hidden_size)
        self.xrange_proj_sem = nn.Linear(hidden_size * 3, hidden_size)
        self.xrange_proj_dis = nn.Linear(hidden_size * 3, hidden_size)

        # Temperature-scaled gate (kept from v2).
        self.gate_proj = nn.Linear(hidden_size * 3, 3)
        self.gate_temperature = nn.Parameter(torch.ones(1))

    @staticmethod
    def _pool_range(hidden_states, start: int, end: int) -> torch.Tensor:
        sliced = list(hidden_states[start:end])
        if not sliced:
            raise IndexError(
                f"CLAF range [{start}:{end}] produced no hidden states "
                f"(backbone has {len(hidden_states)} entries)"
            )
        stacked = torch.stack(sliced, dim=0)              # [L,B,T,H]
        return stacked.mean(dim=0)                         # [B,T,H]

    def forward(self, all_hidden_states,
                attention_mask: Optional[torch.Tensor] = None):
        """Returns (fused [B,T,H], gate_weights [B,T,3])."""
        h_syn = self._pool_range(all_hidden_states, *self.syntax_range)
        h_sem = self._pool_range(all_hidden_states, *self.semantic_range)
        h_dis = self._pool_range(all_hidden_states, *self.discourse_range)

        # Per-position self-attention. key_padding_mask masks out PAD
        # positions so attention rows do not weight padding tokens.
        # MultiheadAttention's key_padding_mask treats True=ignore.
        kpm = (~attention_mask.bool()) if attention_mask is not None else None

        # MHA's softmax in fp16/bf16 is unstable when entire rows are
        # masked (which can happen for short docs at right-edge of the
        # batch). Run attention in fp32 to side-step this. Cost is
        # negligible: attention is a tiny fraction of the per-step compute
        # vs. the backbone forward.
        with torch.amp.autocast(device_type=h_syn.device.type, enabled=False):
            a_syn, _ = self.attn_syn(h_syn.float(), h_syn.float(), h_syn.float(),
                                      key_padding_mask=kpm, need_weights=False)
            a_sem, _ = self.attn_sem(h_sem.float(), h_sem.float(), h_sem.float(),
                                      key_padding_mask=kpm, need_weights=False)
            a_dis, _ = self.attn_dis(h_dis.float(), h_dis.float(), h_dis.float(),
                                      key_padding_mask=kpm, need_weights=False)
        a_syn = a_syn.to(h_syn.dtype)
        a_sem = a_sem.to(h_sem.dtype)
        a_dis = a_dis.to(h_dis.dtype)

        h_syn = self.norm_syn(h_syn + a_syn)
        h_sem = self.norm_sem(h_sem + a_sem)
        h_dis = self.norm_dis(h_dis + a_dis)

        # Bug A fix: each stream gets ITS OWN cross-range summary.
        cross_input = torch.cat([h_syn, h_sem, h_dis], dim=-1)
        h_syn = h_syn + F.gelu(self.xrange_proj_syn(cross_input))
        h_sem = h_sem + F.gelu(self.xrange_proj_sem(cross_input))
        h_dis = h_dis + F.gelu(self.xrange_proj_dis(cross_input))

        gate_input = torch.cat([h_syn, h_sem, h_dis], dim=-1)
        gate_logits = self.gate_proj(gate_input)
        tau = self.gate_temperature.clamp(min=0.1, max=5.0)
        gate_weights = torch.softmax(gate_logits / tau, dim=-1)

        fused = (
            gate_weights[..., 0:1] * h_syn
            + gate_weights[..., 1:2] * h_sem
            + gate_weights[..., 2:3] * h_dis
        )
        return fused, gate_weights


# Backwards-compat alias for any external callers / __pycache__ files.
CrossLayerAttentionFusionV2 = CrossLayerAttentionFusionV3


# ---------------------------------------------------------------------------
# Boundary Prototype Memory v2 -- 64 prototypes + multiplicative gate
# ---------------------------------------------------------------------------
# Differences from v1 (which contributed 16/1296 = 1.2% of classifier
# input — drowned in conv features):
#   1. 64 prototypes instead of 16. ~5% of classifier input bandwidth,
#      enough capacity to capture seam archetypes across 4-5 generators
#      x 6 seam-position buckets x 2 domains.
#   2. Returns BOTH the per-prototype similarities AND a per-position
#      multiplicative gate. The gate (sigmoid of max-prototype-similarity)
#      directly modulates the conv features in SeamDetector, so BPM has
#      two ways to influence the classifier instead of one.
class BoundaryPrototypeMemoryV2(nn.Module):
    def __init__(self, input_dim: int, proto_dim: int = 384,
                 num_prototypes: int = 64):
        super().__init__()
        self.proj = nn.Linear(input_dim, proto_dim)
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, proto_dim) * 0.02,
        )
        # Learnable scale for the multiplicative gate. Init at 0.5 so
        # the gate provides modest modulation early; the optimizer can
        # raise it if the signal is informative.
        self.gate_scale = nn.Parameter(torch.tensor(0.5))
        self.num_prototypes = num_prototypes

    def forward(self, x: torch.Tensor):
        """x: [B,T,input_dim]
        Returns:
            sims [B,T,K]   per-prototype cosine similarities (concat to features)
            gate [B,T,1]   sigmoid(max sim), used for multiplicative modulation
        """
        embedded = F.gelu(self.proj(x))                                # [B,T,P]
        embedded_n = F.normalize(embedded, dim=-1)
        proto_n = F.normalize(self.prototypes, dim=-1)
        sims = torch.matmul(embedded_n, proto_n.T)                     # [B,T,K]
        gate = torch.sigmoid(sims.max(dim=-1, keepdim=True).values)    # [B,T,1]
        return sims, gate


# Backwards-compat alias.
BoundaryPrototypeMemory = BoundaryPrototypeMemoryV2


# ---------------------------------------------------------------------------
# Multi-scale dilated conv head -- 2 layers with residual skip
# ---------------------------------------------------------------------------
# Replaces the 1-layer head in v3. The old head capped receptive field at
# RF=17 (k=3, d=8). For seam detection across 350-word documents,
# paragraph-level context (RF≈30-65) is required and was missing.
#
# Layer 1 (kernel-variety) reads 'normed' [B,T,H]:
#   k=3 d=1 (RF=3), k=5 d=1 (RF=5), k=7 d=1 (RF=7), k=3 d=8 (RF=17)
#   → concat [B, 1280, T]
# Layer 2 (dilation-variety) reads layer-1 output [B,1280,T]:
#   d=1, d=4, d=8, d=16 (compounding RF up to ~65 tokens)
#   → concat [B, 1280, T]
# Output: layer1 + skip_scale * layer2  (learnable mix, init 0.5)
class MultiScaleConvHead(nn.Module):
    CHANNELS = 320
    OUT_DIM = CHANNELS * 4    # 1280

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        # Layer 1: kernel-variety branches over backbone features.
        self.l1_k3  = nn.Conv1d(hidden_size, self.CHANNELS, kernel_size=3, padding=1)
        self.l1_k5  = nn.Conv1d(hidden_size, self.CHANNELS, kernel_size=5, padding=2)
        self.l1_k7  = nn.Conv1d(hidden_size, self.CHANNELS, kernel_size=7, padding=3)
        self.l1_d8  = nn.Conv1d(hidden_size, self.CHANNELS, kernel_size=3, padding=8, dilation=8)
        # Layer 2: dilation-variety branches over layer-1 output.
        self.l2_d1  = nn.Conv1d(self.OUT_DIM, self.CHANNELS, kernel_size=3, padding=1)
        self.l2_d4  = nn.Conv1d(self.OUT_DIM, self.CHANNELS, kernel_size=3, padding=4, dilation=4)
        self.l2_d8  = nn.Conv1d(self.OUT_DIM, self.CHANNELS, kernel_size=3, padding=8, dilation=8)
        self.l2_d16 = nn.Conv1d(self.OUT_DIM, self.CHANNELS, kernel_size=3, padding=16, dilation=16)
        # Learnable skip-mix scalar.
        self.skip_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, H] -> [B, T, OUT_DIM]"""
        h = x.transpose(1, 2)                                           # [B,H,T]
        c1 = torch.cat([
            F.gelu(self.l1_k3(h)),
            F.gelu(self.l1_k5(h)),
            F.gelu(self.l1_k7(h)),
            F.gelu(self.l1_d8(h)),
        ], dim=1)                                                       # [B,1280,T]
        c2 = torch.cat([
            F.gelu(self.l2_d1(c1)),
            F.gelu(self.l2_d4(c1)),
            F.gelu(self.l2_d8(c1)),
            F.gelu(self.l2_d16(c1)),
        ], dim=1)                                                       # [B,1280,T]
        out = c1 + self.skip_scale * c2                                 # [B,1280,T]
        return out.transpose(1, 2)                                      # [B,T,1280]


# ---------------------------------------------------------------------------
# Auxiliary heads -- supervised by free CSV labels
# ---------------------------------------------------------------------------
# These heads are read from the mean-pooled CLAF output. They do not
# affect inference (the predict path ignores them), but at training time
# their gradient pulls representations apart along axes the per-token
# loss does not see directly. The data_source head in particular helps
# the model learn pile-vs-CC distinguishing features, which directly
# benefits the validator's CC out-of-domain F1 gate.
class AuxiliaryHeads(nn.Module):
    def __init__(self, hidden_size: int = 1024,
                 num_data_sources: int = NUM_DATA_SOURCES,
                 num_model_families: int = NUM_MODEL_FAMILIES,
                 num_sample_types: int = NUM_SAMPLE_TYPES):
        super().__init__()
        self.ds_head = nn.Linear(hidden_size, num_data_sources)
        self.mf_head = nn.Linear(hidden_size, num_model_families)
        self.st_head = nn.Linear(hidden_size, num_sample_types)

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "data_source":  self.ds_head(pooled),
            "model_family": self.mf_head(pooled),
            "sample_type":  self.st_head(pooled),
        }


# ---------------------------------------------------------------------------
# Architecture -- HSSD v4
# ---------------------------------------------------------------------------
class SeamDetector(nn.Module):
    """v4 pipeline (replaces v3 / Phase 2.5):

        DeBERTa-v3-Large  (output_hidden_states=True; backbone dropout
                           0.1 active in train(), off in eval())
          -> CLAF v3          (per-position cross-attention; 3 separate
                               cross-range projections)
          -> LayerNorm(1024)
          -> MultiScaleConvHead   (2-layer dilated stack, RF≈65)
          -> BPMv2  (64 prototypes, multiplicative gate + concat)
          -> Linear(1344 → 384) + GELU + Linear(384 → 2)   (classifier)
          -> CRF
        Side heads:
          -> boundary_head  Linear(1344 → 192) + GELU + Linear(192 → 1)
                            for per-position Gaussian-target boundary regression
          -> aux_heads      data_source / model_family / sample_type
                            from mean-pooled CLAF output

    forward(input_ids, attention_mask, labels=None):
        Eval (labels None):    list[list[int]] of Viterbi-decoded paths.
                               Same external behavior as v3 — predict_document.py
                               works without changes.
        Train (labels given):  dict with keys
            emissions         [B,T,2]   tanh-soft-clamped
            boundary_logits   [B,T]     per-position boundary score
            data_source_logits   [B,2]
            model_family_logits  [B,10]
            sample_type_logits   [B,4]

    Notes on bug fixes vs v3:
      * Soft tanh-clamp on emissions instead of hard `clamp(-15,15)` —
        the hard clamp had zero gradient outside [-15,15] and trapped
        emissions that saturated with the wrong sign. Soft clamp is
        monotone, smooth, saturates at ±15 with non-zero gradient.
      * Backbone hidden_dropout / attention_probs_dropout = 0.1 at
        training. eval() disables them — the validator's determinism
        gate runs in eval(), so this is invariant under the gate.
    """

    BPM_NUM_PROTOTYPES = 64
    BPM_PROTO_DIM = 384

    def __init__(self, model_name: str = "microsoft/deberta-v3-large",
                 syntax_range: Tuple[int, int] = (5, 10),
                 semantic_range: Tuple[int, int] = (13, 18),
                 discourse_range: Tuple[int, int] = (20, 25),
                 hidden_dropout: float = 0.1,
                 attn_dropout: float = 0.1,
                 num_model_families: int = NUM_MODEL_FAMILIES,
                 num_sample_types: int = NUM_SAMPLE_TYPES,
                 num_data_sources: int = NUM_DATA_SOURCES,
                 emission_clip: float = 15.0):
        super().__init__()
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        # Standard backbone dropout. Auto-disabled in eval(), so the
        # validator's determinism gate (which probes the miner in eval
        # mode) is unaffected.
        cfg.hidden_dropout_prob = float(hidden_dropout)
        cfg.attention_probs_dropout_prob = float(attn_dropout)
        # Force fp32 weights regardless of the source file's torch_dtype.
        # If the local model file was saved as fp16 (some checkpoints are),
        # loading it natively gives a backbone in fp16 while our custom
        # modules (LayerNorms, MHA, Conv, BPM, etc.) default to fp32 —
        # CLAF's mixed-precision attention path then trips
        # F.layer_norm with mismatched dtypes ("expected Half found Float").
        # Standard mixed-precision training keeps params in fp32 and uses
        # autocast to cast ACTIVATIONS to bf16 only inside the forward.
        self.backbone = AutoModel.from_pretrained(
            model_name, config=cfg, torch_dtype=torch.float32,
        )
        hidden_size = self.backbone.config.hidden_size

        self.claf = CrossLayerAttentionFusionV3(
            hidden_size=hidden_size,
            attn_dropout=float(attn_dropout),
            syntax_range=syntax_range,
            semantic_range=semantic_range,
            discourse_range=discourse_range,
        )
        self.input_norm = nn.LayerNorm(hidden_size)
        self.conv_head = MultiScaleConvHead(hidden_size=hidden_size)

        self.bpm = BoundaryPrototypeMemoryV2(
            input_dim=self.conv_head.OUT_DIM,
            proto_dim=self.BPM_PROTO_DIM,
            num_prototypes=self.BPM_NUM_PROTOTYPES,
        )

        # 1280 conv (gated by BPM) + 64 BPM similarities = 1344
        feat_dim = self.conv_head.OUT_DIM + self.BPM_NUM_PROTOTYPES
        self.feat_dim = feat_dim
        self.emission_clip = float(emission_clip)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 384),
            nn.GELU(),
            nn.Linear(384, 2),
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(feat_dim, 192),
            nn.GELU(),
            nn.Linear(192, 1),
        )
        self.aux_heads = AuxiliaryHeads(
            hidden_size=hidden_size,
            num_data_sources=num_data_sources,
            num_model_families=num_model_families,
            num_sample_types=num_sample_types,
        )

        self.crf = CRF(2, batch_first=True)
        self.num_labels = 2

        # Final guard: force the WHOLE SeamDetector to fp32. This is
        # idempotent if the backbone was already fp32 (the AutoModel
        # call above forced it), but it also covers any future submodule
        # path that might have inherited a non-fp32 dtype. With bf16
        # training, autocast handles activation dtype casts at forward
        # time — parameters stay fp32, which is the correct mixed-prec
        # convention.
        self.float()

    # -------------------------------------------------------------------
    # Internal forward helpers
    # -------------------------------------------------------------------
    def _features(self, input_ids: torch.Tensor,
                  attention_mask: torch.Tensor):
        """Returns:
            feat   [B,T,feat_dim]   classifier-input features
            normed [B,T,H]          CLAF + LN output (used for aux pooling)
        """
        out = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        fused, _gate = self.claf(out.hidden_states, attention_mask=attention_mask)
        normed = self.input_norm(fused)                                   # [B,T,H]
        conv = self.conv_head(normed)                                     # [B,T,1280]
        bpm_sims, bpm_gate = self.bpm(conv)                               # [B,T,K], [B,T,1]
        gated_conv = conv * (1.0 + self.bpm.gate_scale * bpm_gate)        # [B,T,1280]
        feat = torch.cat([gated_conv, bpm_sims], dim=-1)                  # [B,T,feat_dim]
        return feat, normed

    def _soft_clamp(self, emissions: torch.Tensor) -> torch.Tensor:
        """tanh-based saturation at ±emission_clip.

        Bug fix vs v3: hard `.clamp(-15,15)` had zero gradient outside the
        clamp range, so emissions that saturated with the WRONG sign
        could not be pulled back. Soft clamp is monotone, smooth, and
        saturates with non-zero gradient — late-training stability is
        preserved without trapping wrong predictions.
        """
        c = self.emission_clip
        return c * torch.tanh(emissions / c)

    def _emissions(self, input_ids: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns soft-clamped emissions [B, T, 2]."""
        feat, _ = self._features(input_ids, attention_mask)
        return self._soft_clamp(self.classifier(feat))

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    def compute_emissions(self, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """Used by HSSDPredictor's emission-aggregation path
        (sliding-window inference, global Viterbi)."""
        return self._emissions(input_ids, attention_mask)

    def compute_training_outputs(self, input_ids: torch.Tensor,
                                 attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single forward producing emissions + boundary + aux logits.

        Splitting into one call prevents the backbone from being run
        twice when both seam loss and aux loss are needed for the same
        batch."""
        feat, normed = self._features(input_ids, attention_mask)
        emissions = self._soft_clamp(self.classifier(feat))             # [B,T,2]
        boundary_logits = self.boundary_head(feat).squeeze(-1)          # [B,T]
        # Mean-pool CLAF output over valid positions (drop PAD and CLS).
        mask = attention_mask.float().clone()
        mask[:, 0] = 0.0    # exclude [CLS]
        mask = mask.unsqueeze(-1)                                        # [B,T,1]
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (normed * mask).sum(dim=1) / denom                      # [B,H]
        aux = self.aux_heads(pooled)
        return {
            "emissions":            emissions,
            "boundary_logits":      boundary_logits,
            "data_source_logits":   aux["data_source"],
            "model_family_logits":  aux["model_family"],
            "sample_type_logits":   aux["sample_type"],
        }

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        if labels is None:
            # Inference path: return Viterbi-decoded label sequences.
            # Same protocol as v3 so predict_document.py is unaffected.
            emissions = self._emissions(input_ids, attention_mask)
            em_no_cls = emissions[:, 1:, :]
            attn_no_cls = attention_mask[:, 1:]
            crf_mask = attn_no_cls.bool().clone()
            crf_mask[:, 0] = True
            with torch.amp.autocast(device_type=emissions.device.type,
                                     enabled=False):
                decoded_no_cls = self.crf.decode(
                    em_no_cls.float(), mask=crf_mask,
                )
            return [[0] + path for path in decoded_no_cls]
        # Training path: return ALL training outputs as a dict. The
        # train loop's loss function handles the dict.
        return self.compute_training_outputs(input_ids, attention_mask)


# ---------------------------------------------------------------------------
# Loss — per-token CRF + focal + boundary-Gaussian + aux CE
# ---------------------------------------------------------------------------
def compute_focal_loss(
    emissions: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    seam_alpha: float = 0.75,
    **_legacy_kwargs,
) -> torch.Tensor:
    """Per-token focal loss (Lin et al. 2017), normalised by valid count."""
    valid = (labels != -100)
    safe_labels = labels.clone()
    safe_labels[~valid] = 0

    log_probs = F.log_softmax(emissions.float(), dim=-1)
    log_pt = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()

    focal_weight = (1.0 - pt).pow(gamma)
    is_seam = (labels == 1).float()
    alpha_t = seam_alpha * is_seam + (1.0 - seam_alpha) * (1.0 - is_seam)
    loss_per_token = -(alpha_t * focal_weight * log_pt)

    valid_f = valid.float()
    return (loss_per_token * valid_f).sum() / valid_f.sum().clamp_min(1.0)


def compute_rdrop_kl(
    em1: torch.Tensor,
    em2: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Symmetric KL divergence between two emission tensors from twin
    forward passes (R-Drop, Liang et al. 2021).

    The two passes use the SAME inputs but DIFFERENT dropout masks
    (backbone dropout, attn dropout, LoRA dropout, CLAF attention
    dropout). KL(p1||p2)+KL(p2||p1) penalizes prediction inconsistency
    across stochastic forward passes — a strong implicit regularizer
    that consistently outperforms vanilla dropout for fine-tuning.

    Why we'd use R-Drop on top of EMA + dropout:
      * Dropout regularizes one forward pass; R-Drop regularizes the
        DIFFERENCE between two passes. They are complementary.
      * R-Drop's effect is largest on borderline tokens (where the
        two passes most often disagree) — the same tokens that drive
        seam-offset errors. Direct fix for our metric.
      * Cost: 2× forward + 2× backward (offset by removing FGM, which
        was 6× compute and is now off in v4 anyway).

    em1, em2          : [B, T, 2]
    labels            : [B, T]   uses -100 to mark ignored positions
    attention_mask    : [B, T]
    """
    valid = ((labels != -100) & attention_mask.bool()).float()           # [B,T]

    log_p1 = F.log_softmax(em1.float(), dim=-1)
    log_p2 = F.log_softmax(em2.float(), dim=-1)
    p1 = log_p1.exp()
    p2 = log_p2.exp()

    kl_12 = (p1 * (log_p1 - log_p2)).sum(dim=-1)                          # [B,T]
    kl_21 = (p2 * (log_p2 - log_p1)).sum(dim=-1)
    sym_kl = 0.5 * (kl_12 + kl_21)

    return (sym_kl * valid).sum() / valid.sum().clamp_min(1.0)


def compute_boundary_loss(
    boundary_logits: torch.Tensor,
    boundary_target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """BCE between sigmoid(boundary_logits) and a Gaussian-shaped target
    centered at the GT seam position.

    Why BCE on a [0,1] target (vs MSE):
      BCE has the correct curvature for a [0,1] target and a clean
      end-to-end gradient through sigmoid: (sigmoid(x) - target).
      MSE on sigmoid gives weak gradient at saturated outputs and is
      non-convex.

    Why a Gaussian target (vs hard one-hot at the GT seam):
      Seam labels in the CSV are word-accurate to within ~1 word.
      A Gaussian (sigma≈3 words) gives consistent gradient over the
      uncertainty band around the seam, and produces a smooth peak
      that is easier to localize at inference.
    """
    bce = F.binary_cross_entropy_with_logits(
        boundary_logits.float(), boundary_target.float(), reduction="none",
    )
    vm = valid_mask.float()
    return (bce * vm).sum() / vm.sum().clamp_min(1.0)


def compute_total_loss(
    outputs,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    crf_module: CRF,
    boundary_target: Optional[torch.Tensor] = None,
    data_source_id: Optional[torch.Tensor] = None,
    model_family_id: Optional[torch.Tensor] = None,
    sample_type_id: Optional[torch.Tensor] = None,
    lambda_focal: float = 0.3,
    lambda_boundary: float = 0.5,
    lambda_data_source: float = 0.05,
    lambda_model_family: float = 0.05,
    lambda_sample_type: float = 0.05,
    focal_gamma: float = 2.0,
    focal_seam_alpha: float = 0.75,
    return_components: bool = False,
    **_legacy_kwargs,
):
    """Multi-term training loss for HSSD v4.

    L_total = L_CRF_per_token
            + lambda_focal     * L_focal
            + lambda_boundary  * L_boundary
            + lambda_ds        * L_data_source     (if provided)
            + lambda_mf        * L_model_family    (if provided)
            + lambda_st        * L_sample_type     (if provided)

    Bug fix vs v3: L_CRF was averaged per-SEQUENCE (`reduction="mean"`),
    so for a 200-token sequence with 1 seam the CRF NLL was ~10 nats
    per sequence while the focal loss was ~0.1 per token. With
    lambda_focal=0.3, focal contributed 0.3% of the gradient. v4
    normalises CRF per-TOKEN by dividing by the valid-mask sum, which
    puts both terms on the same scale.

    `outputs` may be either:
      * a dict (HSSD v4) containing 'emissions' and optionally
        'boundary_logits' / '*_logits' for aux heads, or
      * a bare tensor (legacy v3) that becomes the emissions directly.
    Tolerating the second form makes the function safe for any
    external caller that hasn't migrated yet.
    """
    if torch.is_tensor(outputs):
        outputs = {"emissions": outputs}
    emissions = outputs["emissions"]
    device_type = emissions.device.type

    with torch.amp.autocast(device_type=device_type, enabled=False):
        emissions_fp32 = emissions.float()
        em_no_cls = emissions_fp32[:, 1:, :]
        labels_no_cls = labels[:, 1:]
        attn_no_cls = attention_mask[:, 1:]

        crf_mask = attn_no_cls.bool() & (labels_no_cls != -100)
        crf_mask = crf_mask.clone()
        crf_mask[:, 0] = True

        safe_labels = labels_no_cls.clone()
        safe_labels[labels_no_cls == -100] = 0

        # Per-token CRF NLL: divide sum NLL by valid-mask sum so the
        # term is on the same scale as the focal/boundary/aux losses.
        crf_nll_sum = -crf_module(
            em_no_cls, safe_labels, mask=crf_mask, reduction="sum",
        )
        n_valid = crf_mask.float().sum().clamp_min(1.0)
        crf_loss = crf_nll_sum / n_valid

        focal = compute_focal_loss(
            em_no_cls, labels_no_cls,
            gamma=focal_gamma,
            seam_alpha=focal_seam_alpha,
        )

    components = {"crf": crf_loss.detach(), "focal": focal.detach()}
    total = crf_loss + lambda_focal * focal

    if boundary_target is not None and "boundary_logits" in outputs:
        valid_mask = (labels != -100).float() * attention_mask.float()
        valid_mask = valid_mask.clone()
        valid_mask[:, 0] = 0.0
        bnd = compute_boundary_loss(
            outputs["boundary_logits"], boundary_target, valid_mask,
        )
        total = total + lambda_boundary * bnd
        components["boundary"] = bnd.detach()

    def _aux_ce(logits, ids, lam, key):
        nonlocal total
        if logits is None or ids is None or lam <= 0.0:
            return
        sel = ids >= 0
        if not sel.any():
            return
        ce = F.cross_entropy(logits[sel].float(), ids[sel].long())
        total = total + lam * ce
        components[key] = ce.detach()

    _aux_ce(outputs.get("data_source_logits"),  data_source_id,
            lambda_data_source,  "aux_ds")
    _aux_ce(outputs.get("model_family_logits"), model_family_id,
            lambda_model_family, "aux_mf")
    _aux_ce(outputs.get("sample_type_logits"),  sample_type_id,
            lambda_sample_type,  "aux_st")

    if return_components:
        return total, components
    return total


# ---------------------------------------------------------------------------
# Feature-Level FGM (Phase 2.5 -- replaces broken word-embedding FGM)
# ---------------------------------------------------------------------------
class FeatureFGM:
    """FGM on feature-producing parameters (default: input_norm).

    The Phase 2 FGM targeted word_embeddings, which are FROZEN under
    LoRA -- so .grad was always None and the attack was a silent
    no-op. The fix: target a parameter that IS trainable AND sits in
    the active feature path. `input_norm` (LayerNorm between CLAF and
    the conv head) is ideal:
      - It's in modules_to_save, so its weight/bias have requires_grad.
      - Perturbing those parameters distorts every position's features
        before they hit the conv head.
      - The 1024-dim feature space is directly relevant to seam
        detection (vs the 128K-dim embedding space where most
        dimensions are irrelevant).
    """

    def __init__(self, model: nn.Module, target_substring: str = "input_norm",
                 epsilon: float = 1.0):
        self.model = model
        self.target = target_substring
        self.epsilon = float(epsilon)
        self.backup: Dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if self.target not in name:
                continue
            if param.grad is None:
                continue
            norm = torch.norm(param.grad)
            if not torch.isfinite(norm) or norm.item() == 0.0:
                continue
            self.backup[name] = param.data.detach().clone()
            param.data.add_(self.epsilon * param.grad / norm)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# ---------------------------------------------------------------------------
# Weight EMA (Exponential Moving Average of trainable parameters)
# ---------------------------------------------------------------------------
# Why EMA helps:
#   * Late-stage SGD oscillates around the loss minimum. EMA averages
#     out that oscillation, producing a smoother, lower-variance final
#     model — typically +0.3-0.7 F1 on dense token-classification tasks
#     for free at inference time.
#   * For seam detection specifically, the seam-position prediction is
#     sensitive to small emission shifts. EMA reduces shift jitter
#     between adjacent steps and improves f1_at_5 in particular.
#   * Compatible with PEFT — we only track parameters with
#     requires_grad=True, so the frozen backbone is not duplicated.
#     Memory overhead = trainable-param count × dtype size, typically
#     ~150 MB for HSSD v4 in fp32 on CPU.
#
# Usage in train():
#   ema = WeightEMA(model, decay=args.ema_decay)
#   ... after every optimizer.step() ...
#       ema.update(model)
#   ... at validation / save ...
#       ema.store(model); ema.apply(model); evaluate; ema.restore(model)
class WeightEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        # CPU storage keeps GPU VRAM free for activations. The trade-off
        # is one host<->device copy per update; on a 5090 this is ~10ms
        # per step for a 5-10M trainable-param model, negligible vs the
        # ~500ms forward+backward.
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.detach().clone().cpu()
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if not p.requires_grad or n not in self.shadow:
                continue
            cpu_p = p.data.detach().to("cpu", copy=True, non_blocking=True)
            self.shadow[n].mul_(self.decay).add_(cpu_p, alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """Snapshot current model weights so we can apply EMA temporarily."""
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.detach().clone()

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        """Overwrite current model weights with EMA shadow weights."""
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.device))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore weights snapshotted by store()."""
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

    def state_dict(self) -> Dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.shadow = state.get("shadow", {})


# ---------------------------------------------------------------------------
# CRF transition constraint
# ---------------------------------------------------------------------------
def _resolve_crf(model: nn.Module) -> CRF:
    """Return the CRF module regardless of PEFT wrapping depth."""
    if isinstance(model, CRF):
        return model
    if hasattr(model, "crf"):
        return model.crf
    # PEFT wrapping: PeftModel -> base_model -> model -> SeamDetector
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model"):
            base = base.model
        if hasattr(base, "crf"):
            return base.crf
    raise AttributeError("Could not locate CRF module on this model")


def constrain_crf_transitions(model: nn.Module, min_prob: float = 0.05) -> None:
    """Clamp P(1->0) >= min_prob in the CRF transition matrix.

    Without this clamp the CRF can drive P(1->0) -> 0, which makes it
    practically impossible to predict a second seam in multi-seam docs
    (ai_in_middle samples, mostly). The constraint only floors one
    cell of the 2x2 transition matrix; legitimate transitions stay
    learnable."""
    crf = _resolve_crf(model)
    min_log = math.log(max(min_prob, 1e-9))
    with torch.no_grad():
        # transitions[i, j] = log P(j | i). We're constraining (1 -> 0).
        crf.transitions.data[1, 0].clamp_(min=min_log)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SeamDataset(Dataset):
    """Loads rows from build_training_dataset.py CSVs with TOKEN-AWARE
    word chunking.

    For each row, we batch-tokenize once at __init__ (no truncation) to
    compute the per-word token count, then split the word list at
    boundaries where the cumulative token count would exceed the
    budget. Each emitted chunk is guaranteed to tokenize to <=
    max_length tokens AFTER [CLS] and [SEP] are added by __getitem__,
    so the tokenizer's truncation flag never silently drops a token.

    Why this matters:
      * Default HuggingFace tokenizer with `truncation=True` silently
        cuts the tail of any chunk that tokenizes too long. URL/code/
        unicode-heavy text (350 words can be 600-3000 tokens) would
        lose the seam in the truncated tail without this fix.
      * Word-aware chunking alone (the old approach) only checks word
        count, not token count, so it inherits the truncation bug.

    Cost: a one-time batch-tokenization pass at __init__ (~30-60s for
    158k training rows; negligible compared to multi-hour training).
    """

    def __init__(self, csv_paths: List[Path], tokenizer,
                 max_length: int = 512, stride: int = 256,
                 shuffle_rows: bool = True,
                 max_rows: Optional[int] = None,
                 min_chunk_words: int = 20,
                 seed: int = 0,
                 boundary_sigma: float = 3.0):
        """boundary_sigma : Gaussian width (in TOKEN positions) for the
        boundary-head supervision target. ~3 tokens ≈ ~1.5 words on
        average for English DeBERTa, which matches the validator's
        ±5-word grading band."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = max(1, stride)         # kept for back-compat; unused now
        self.min_chunk_words = min_chunk_words
        self.boundary_sigma = float(boundary_sigma)

        rng = random.Random(seed)
        rows: List[Dict[str, str]] = []
        for path in csv_paths:
            with open(path, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    rows.append(r)
                    if max_rows and len(rows) >= max_rows:
                        break
            if max_rows and len(rows) >= max_rows:
                break

        if shuffle_rows:
            rng.shuffle(rows)

        # ---- Filter malformed rows up front ------------------------
        # Each entry: (words, word_labels, ds_id, mf_id, st_id).
        # ds/mf/st = -1 if the field is missing or unrecognised; the
        # loss function skips aux terms for any -1 row.
        valid_rows: List[Tuple[List[str], List[int], int, int, int]] = []
        skipped = 0
        for row in rows:
            text = row.get("text", "")
            label_field = row.get("segmentation_labels", "")
            if not text or not label_field:
                skipped += 1
                continue
            try:
                word_labels = json.loads(label_field)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(word_labels, list):
                skipped += 1
                continue
            words = text.split()
            if len(words) != len(word_labels):
                skipped += 1
                continue

            ds = (row.get("data_source") or "").strip()
            ds_id = DATA_SOURCE_MAP.get(ds, -1)
            st = (row.get("sample_type") or "").strip()
            st_id = SAMPLE_TYPE_MAP.get(st, -1)
            mn = (row.get("model_name") or "").strip()
            mf_id = model_family_id(mn) if mn else -1

            valid_rows.append(
                (words, list(map(int, word_labels)), ds_id, mf_id, st_id)
            )

        if skipped:
            print(f"  (skipped {skipped} malformed rows)")

        # ---- Batch tokenization (the speed-critical step) ---------
        # We tokenize WITHOUT special tokens so each chunk's content
        # tokens fit in `max_length - 2`; __getitem__ then re-tokenizes
        # WITH special tokens, bringing the total to exactly max_length.
        print(f"  Pre-tokenizing {len(valid_rows):,} rows for "
              f"token-aware chunking...")
        import time
        t0 = time.time()
        all_words = [vr[0] for vr in valid_rows]
        encodings = tokenizer(
            all_words,
            is_split_into_words=True,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        print(f"  pre-tokenization done in {time.time() - t0:.1f}s")

        # ---- Token-aware chunking -----------------------------------
        # Budget = max_length - 4. Reserves 2 tokens for [CLS] + [SEP]
        # plus a 2-token safety margin for "context drift": the first
        # tokenization pass measures word-token-count in the context of
        # the FULL row, but each chunk is re-tokenized standalone in the
        # cache build. SentencePiece's leading-space behavior on the
        # first word of a standalone chunk can change a word's token
        # count by 1 vs its in-row count. The +2 margin absorbs that
        # drift across the chunk so the cache's `truncation=True` never
        # silently chops off the chunk's tail (which is where seams
        # often live in late-seam rows).
        budget = max_length - 4
        # Each chunk: (words, labels, ds_id, mf_id, st_id).
        # Per-chunk aux IDs inherit from the parent row.
        self.chunks: List[Tuple[List[str], List[int], int, int, int]] = []
        n_split = 0
        n_dropped_huge_word = 0

        for idx, (words, word_labels, ds_id, mf_id, st_id) in enumerate(valid_rows):
            word_ids = encodings.word_ids(idx)
            n_words = len(words)
            if n_words == 0:
                continue

            # Per-word token count (how many sub-tokens each word
            # produced when tokenized standalone in this row).
            word_token_count = [0] * n_words
            for wid in word_ids:
                if wid is not None and 0 <= wid < n_words:
                    word_token_count[wid] += 1

            # Greedy split: walk words; if adding the next word's
            # tokens would exceed budget AND we already have at least
            # one word in the current chunk, emit and reset.
            start = 0
            cur_tokens = 0
            split_this_row = False
            for i in range(n_words):
                wt = word_token_count[i]
                # Defensive: if a single word tokenizes to > budget
                # (very rare -- huge URL or hex blob), we still emit it
                # as its own chunk. The tokenizer in __getitem__ WILL
                # truncate that one chunk, but it's the only path
                # forward; alternative is dropping the word entirely.
                if wt > budget:
                    # Emit any pending chunk first.
                    if start < i:
                        self._maybe_emit(words, word_labels,
                                          start, i, ds_id, mf_id, st_id)
                        n_split += 1
                    # Emit the huge word as its own chunk (will be
                    # tokenizer-truncated; rare).
                    self._maybe_emit(words, word_labels,
                                      i, i + 1, ds_id, mf_id, st_id)
                    n_dropped_huge_word += 1
                    start = i + 1
                    cur_tokens = 0
                    split_this_row = True
                    continue

                if cur_tokens + wt > budget and start < i:
                    self._maybe_emit(words, word_labels,
                                      start, i, ds_id, mf_id, st_id)
                    n_split += 1
                    start = i
                    cur_tokens = 0
                    split_this_row = True

                cur_tokens += wt

            # Final chunk
            if start < n_words:
                self._maybe_emit(words, word_labels,
                                  start, n_words, ds_id, mf_id, st_id)

        print(f"  Built {len(self.chunks):,} chunks "
              f"(token-budget split fired on {n_split:,} chunk boundaries)")
        if n_dropped_huge_word:
            print(f"  WARN: {n_dropped_huge_word:,} chunks contained a single "
                  f"word that tokenizes to > {budget} tokens. Those chunks "
                  f"WILL be tokenizer-truncated. Usually this is rare unicode "
                  f"or extremely long URLs/hex.")

        # Free the first-pass encodings (used only for word-token-count
        # bookkeeping during chunking).
        del encodings

        # ---- Pre-tokenize + cache the final chunks as tensors ------
        # Optimization C: tokenize the chunked (words, labels) pairs
        # ONCE at __init__ and store input_ids / attention_mask / labels
        # as fixed-shape tensors. __getitem__ then becomes a simple
        # tensor slice -- no per-batch tokenizer overhead during
        # training. This frees the dataloader workers to focus on disk
        # I/O and tensor pinning.
        #
        # Memory cost: N_chunks * max_length * 8 bytes * 3 tensors.
        # For 159k chunks at max_length=512 that's ~1.95 GB on CPU,
        # shared across DataLoader workers via copy-on-write fork
        # semantics on Linux (no per-worker duplication). Comfortably
        # fits in Colab's High-RAM (50 GB).
        print(f"  Pre-tokenizing {len(self.chunks):,} chunks for "
              f"cached __getitem__...")
        t1 = time.time()
        all_chunk_words = [c[0] for c in self.chunks]
        chunk_encs = tokenizer(
            all_chunk_words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,             # defensive only; chunking already fits the budget
            max_length=self.max_length,
            return_tensors="pt",
        )
        self._cached_input_ids: torch.Tensor = chunk_encs["input_ids"]          # [N, T]
        self._cached_attention_mask: torch.Tensor = chunk_encs["attention_mask"]  # [N, T]

        # Build labels + Gaussian boundary target in a single pass.
        # Boundary target: for chunks with exactly one 0/1 transition
        # in the valid (non-[-100]) positions, place a Gaussian peak at
        # the seam position with sigma=self.boundary_sigma. Otherwise,
        # all zeros — the BCE loss handles "no seam here" correctly.
        N = len(self.chunks)
        T = self.max_length
        cached_labels = torch.full((N, T), -100, dtype=torch.long)
        cached_boundary = torch.zeros((N, T), dtype=torch.float32)
        cached_ds = torch.full((N,), -1, dtype=torch.long)
        cached_mf = torch.full((N,), -1, dtype=torch.long)
        cached_st = torch.full((N,), -1, dtype=torch.long)

        positions = torch.arange(T, dtype=torch.float32)
        sigma2_2 = 2.0 * (self.boundary_sigma ** 2)

        for idx, chunk in enumerate(self.chunks):
            _words, word_labels, ds_id, mf_id, st_id = chunk
            row_word_ids = chunk_encs.word_ids(idx)
            row_labels = [
                -100 if wid is None else int(word_labels[wid])
                for wid in row_word_ids
            ]
            cached_labels[idx] = torch.tensor(row_labels, dtype=torch.long)
            cached_ds[idx] = int(ds_id)
            cached_mf[idx] = int(mf_id)
            cached_st[idx] = int(st_id)

            # Find the first 0/1 transition among VALID positions and the
            # number of transitions; place a Gaussian only if exactly one.
            seam_pos = None
            n_trans = 0
            prev = None
            for pos, lab in enumerate(row_labels):
                if lab == -100:
                    continue
                if prev is not None and lab != prev:
                    n_trans += 1
                    if seam_pos is None:
                        seam_pos = pos
                prev = lab
            if n_trans == 1 and seam_pos is not None:
                target = torch.exp(-((positions - float(seam_pos)) ** 2) / sigma2_2)
                # Zero out -100 positions in the target so the BCE mask
                # already handles them, but keep the rest crisp.
                for pos, lab in enumerate(row_labels):
                    if lab == -100:
                        target[pos] = 0.0
                cached_boundary[idx] = target

        self._cached_labels = cached_labels                              # [N,T] long
        self._cached_boundary = cached_boundary                          # [N,T] float
        self._cached_data_source_id = cached_ds                          # [N]
        self._cached_model_family_id = cached_mf                         # [N]
        self._cached_sample_type_id = cached_st                          # [N]

        cache_gb = (
            self._cached_input_ids.numel() * self._cached_input_ids.element_size()
            + self._cached_attention_mask.numel() * self._cached_attention_mask.element_size()
            + self._cached_labels.numel() * self._cached_labels.element_size()
            + self._cached_boundary.numel() * self._cached_boundary.element_size()
        ) / 1e9
        print(f"  cache built in {time.time() - t1:.1f}s "
              f"(memory: ~{cache_gb:.2f} GB CPU; shared across DataLoader "
              f"workers via fork copy-on-write)")

        # Quick diagnostic: distribution of aux IDs (how much aux
        # supervision the loss will actually see).
        with_seam = int((cached_boundary.sum(dim=1) > 0).sum().item())
        ds_known = int((cached_ds >= 0).sum().item())
        mf_known = int((cached_mf >= 0).sum().item())
        st_known = int((cached_st >= 0).sum().item())
        print(f"  chunks with single-seam (boundary target placed): "
              f"{with_seam}/{N}  ({100*with_seam/max(1,N):.1f}%)")
        print(f"  aux supervision available: data_source {ds_known}/{N}, "
              f"model_family {mf_known}/{N}, sample_type {st_known}/{N}")

    def _maybe_emit(self, words: List[str], labels: List[int],
                    start: int, end: int,
                    ds_id: int, mf_id: int, st_id: int) -> None:
        """Append (words[start:end], labels[start:end], ds_id, mf_id, st_id)
        to self.chunks if it meets the min_chunk_words floor."""
        if end - start < self.min_chunk_words:
            return
        self.chunks.append(
            (words[start:end], labels[start:end], ds_id, mf_id, st_id)
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # All work was done at __init__. This is now a simple tensor
        # slice -- no tokenizer call, no dictionary construction beyond
        # building the return dict.
        return {
            "input_ids":        self._cached_input_ids[idx],
            "attention_mask":   self._cached_attention_mask[idx],
            "labels":           self._cached_labels[idx],
            "boundary_target":  self._cached_boundary[idx],
            "data_source_id":   self._cached_data_source_id[idx],
            "model_family_id":  self._cached_model_family_id[idx],
            "sample_type_id":   self._cached_sample_type_id[idx],
        }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def _first_transition(arr: List[int]) -> Optional[int]:
    """Index of the first 0->1 or 1->0 transition (1-based: position of the
    second token). None if the entire array is one class."""
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            return i
    return None


# ---------------------------------------------------------------------------
# Resumable shuffle / sampler
# ---------------------------------------------------------------------------
class FixedOrderSampler(torch.utils.data.Sampler[int]):
    """Yields a pre-determined list of dataset indices in order. Used so we
    can deterministically reconstruct an epoch's permutation from a seed,
    optionally drop the first K already-consumed indices on resume, and let
    the DataLoader batch the remainder normally."""

    def __init__(self, indices: List[int]):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _epoch_permutation(num_samples: int, base_seed: int, epoch: int) -> List[int]:
    """Deterministic per-epoch permutation. Same (base_seed, epoch) -> same
    list. Built with a torch.Generator so the shuffle is reproducible
    across processes / Python versions."""
    g = torch.Generator()
    g.manual_seed(int(base_seed) + int(epoch) * 9973)  # 9973 is prime; mix epoch in
    return torch.randperm(num_samples, generator=g).tolist()


def _make_epoch_loader(train_ds, args: argparse.Namespace,
                      base_seed: int, epoch: int,
                      skip_samples: int = 0):
    """Build a DataLoader for `epoch`. Drops the first `skip_samples`
    indices from the deterministic permutation so resume can pick up at
    the exact mid-epoch point where the previous run was interrupted.

    Returns None if the entire epoch was already consumed (i.e. skip is
    >= dataset size); the caller should advance to the next epoch."""
    perm = _epoch_permutation(len(train_ds), base_seed, epoch)
    if skip_samples > 0:
        if skip_samples >= len(perm):
            return None
        perm = perm[skip_samples:]
    sampler = FixedOrderSampler(perm)
    return DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.dataloader_workers,
        pin_memory=True,
        drop_last=True,
    )


def compute_seam_metrics(gt_arrays: List[List[int]],
                         pred_arrays: List[List[int]]) -> Dict[str, float]:
    offsets: List[int] = []
    correct_at_5 = 0
    total = 0

    for gt, pred in zip(gt_arrays, pred_arrays):
        total += 1
        gt_seam = _first_transition(gt)
        pred_seam = _first_transition(pred)

        if gt_seam is None and pred_seam is None:
            correct_at_5 += 1
            continue
        if gt_seam is None or pred_seam is None:
            continue                                  # missed / hallucinated
        dist = abs(gt_seam - pred_seam)
        offsets.append(dist)
        if dist <= 5:
            correct_at_5 += 1

    return {
        "mean_seam_offset": float(np.mean(offsets)) if offsets else float("nan"),
        "f1_at_5":          correct_at_5 / max(1, total),
        "n_with_seam":      len(offsets),
        "n_total":          total,
    }


def compute_token_f1(gt_arrays: List[List[int]],
                     pred_arrays: List[List[int]]) -> float:
    from sklearn.metrics import f1_score
    flat_gt: List[int] = []
    flat_pr: List[int] = []
    for gt, pr in zip(gt_arrays, pred_arrays):
        n = min(len(gt), len(pr))
        flat_gt.extend(gt[:n])
        flat_pr.extend(pr[:n])
    if not flat_gt:
        return 0.0
    return float(f1_score(flat_gt, flat_pr, zero_division=0))


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device, autocast_ctx) -> Dict[str, float]:
    """Validation metrics, including the validator's exact reward components.

    The SN32 validator scores each miner with:
        reward = mean(fp_score, f1_score, ap_score)
        fp_score = 1 - FP / N            (low FP rate is good)
        f1_score = sklearn.f1_score(y_true, np.round(y_pred))
        ap_score = sklearn.average_precision_score(y_true, y_pred)

    The miner returns CONTINUOUS y_pred ∈ [0, 1] per word; the validator
    rounds for f1/fp and uses raw for AP. predict_document.predict_with_probs
    biases each per-word probability so np.round(p) == CRF Viterbi label
    (preserving optimal f1/fp) while keeping the unclamped portion as the
    AP ranking signal.

    To match the production scoring exactly during training-time validation
    we replicate that pipeline here:
       1. compute raw emissions for the batch (no autocast around CRF)
       2. softmax → P(AI) per token
       3. CRF Viterbi → discrete labels per token (with [CLS] sliced)
       4. clamp prob to side of 0.5 matching the Viterbi label
       5. compute fp_score / f1_score / ap_score / reward over flat tokens

    f1_at_5 / mean_seam_offset stay computed for diagnostic visibility but
    are NO LONGER the best-checkpoint criterion."""
    from sklearn.metrics import f1_score as _sk_f1
    from sklearn.metrics import average_precision_score as _sk_ap
    from sklearn.metrics import confusion_matrix as _sk_cm

    seam_det = _find_seam_detector(model)
    crf_module = _resolve_crf(model)

    model.eval()
    gt_arrays: List[List[int]] = []
    pred_arrays: List[List[int]] = []
    flat_y_true: List[int] = []
    flat_y_pred: List[float] = []    # continuous, validator-style

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]                  # CPU side, used for masking

            with autocast_ctx:
                # Raw emissions for the batch — needed for both CRF decode
                # and softmax probabilities.
                emissions = seam_det.compute_emissions(
                    input_ids, attention_mask,
                )

            # Slice off [CLS] (position 0) to match the train-time CRF
            # convention; prepend dummy 0 for output-length parity.
            em_no_cls = emissions[:, 1:, :]
            attn_no_cls = attention_mask[:, 1:]
            crf_mask = attn_no_cls.bool().clone()
            crf_mask[:, 0] = True
            with torch.amp.autocast(device_type=emissions.device.type,
                                     enabled=False):
                decoded_no_cls = crf_module.decode(
                    em_no_cls.float(), mask=crf_mask,
                )

            # softmax(emissions)[..., 1] → P(AI) per token, in fp32 for
            # determinism. Same path predict_document._decode_with_probs uses.
            with torch.amp.autocast(device_type=emissions.device.type,
                                     enabled=False):
                probs_ai = torch.softmax(emissions.float(), dim=-1)[..., 1]
            probs_ai_cpu = probs_ai.cpu().tolist()

            B = input_ids.shape[0]
            for i in range(B):
                # Reconstruct full path with the dummy CLS prepended.
                path = [0] + list(decoded_no_cls[i])
                row_probs = probs_ai_cpu[i]
                gt = labels[i].tolist()

                aligned_gt: List[int] = []
                aligned_pred: List[int] = []
                pi = 0
                for g in gt:
                    if pi >= len(path):
                        break
                    if g == -100:
                        pi += 1
                        continue
                    label = int(path[pi])
                    p = float(row_probs[pi])
                    # Clamp prob to match the Viterbi label (validator
                    # rounds at 0.5, so this preserves f1/fp).
                    if label == 1 and p < 0.51:
                        p = 0.51
                    elif label == 0 and p > 0.49:
                        p = 0.49
                    aligned_gt.append(int(g))
                    aligned_pred.append(label)
                    flat_y_true.append(int(g))
                    flat_y_pred.append(p)
                    pi += 1
                if aligned_gt:
                    gt_arrays.append(aligned_gt)
                    pred_arrays.append(aligned_pred)

    # ---- Diagnostic seam metrics (kept for visibility) ----
    metrics = compute_seam_metrics(gt_arrays, pred_arrays)
    metrics["token_f1"] = compute_token_f1(gt_arrays, pred_arrays)

    # ---- Validator-aligned scoring ----
    if flat_y_true:
        import numpy as _np
        y_true_a = _np.asarray(flat_y_true, dtype=_np.int32)
        y_pred_a = _np.asarray(flat_y_pred, dtype=_np.float32)
        preds_round = _np.round(y_pred_a).astype(_np.int32)

        # FP rate: needs both classes present in y_true for confusion_matrix
        # to return a 2x2 matrix. Defensive call in case the val batch
        # happens to have one class only (extremely unlikely on val_rebalanced).
        try:
            tn, fp, fn, tp = _sk_cm(y_true_a, preds_round, labels=[0, 1]).ravel()
            fp_score = 1.0 - fp / max(1, len(y_pred_a))
        except Exception:
            fp_score = float("nan")
        try:
            f1 = float(_sk_f1(y_true_a, preds_round, zero_division=0))
        except Exception:
            f1 = float("nan")
        try:
            ap = float(_sk_ap(y_true_a, y_pred_a))
        except Exception:
            ap = float("nan")
        # Same formula the validator uses (reward.py:49)
        reward = float(_np.nanmean([fp_score, f1, ap]))
    else:
        fp_score = f1 = ap = reward = float("nan")

    metrics["validator_fp_score"] = float(fp_score)
    metrics["validator_f1_score"] = float(f1)
    metrics["validator_ap_score"] = float(ap)
    metrics["validator_reward"]   = float(reward)
    return metrics


# ---------------------------------------------------------------------------
# CLAF gate diagnostic (optional)
# ---------------------------------------------------------------------------
def _find_seam_detector(model: nn.Module) -> Optional["SeamDetector"]:
    """Walk model.modules() to find the SeamDetector, regardless of how
    deeply PEFT has wrapped it."""
    for module in model.modules():
        if isinstance(module, SeamDetector):
            return module
    return None


def _find_trainable_claf(model: nn.Module) -> Optional[CrossLayerAttentionFusionV2]:
    """With modules_to_save, model.modules() yields TWO CLAF instances
    (the frozen original + the trainable adapter copy). We want the
    trainable one -- it's the one whose weights actually update."""
    fallback = None
    for module in model.modules():
        if isinstance(module, CrossLayerAttentionFusionV2):
            if module.gate_temperature.requires_grad:
                return module
            fallback = fallback or module
    return fallback


@torch.no_grad()
def log_gate_statistics(model: nn.Module, loader: DataLoader,
                        device: torch.device, num_batches: int = 5) -> None:
    """Print mean/std of CLAF gate weights at seam vs non-seam positions
    plus the current learnable temperature tau. Use this to verify
    CLAF is producing position-specific gating. Cheap (a few batches)
    -- safe to call every validation."""
    seam_det = _find_seam_detector(model)
    claf = _find_trainable_claf(model)
    if seam_det is None or claf is None:
        print("  (gate-monitor: SeamDetector / CLAF not found; skipping)")
        return

    backbone = seam_det.backbone

    model.eval()
    seam_gates: List[np.ndarray] = []
    non_seam_gates: List[np.ndarray] = []

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = backbone(input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        _, gw = claf(out.hidden_states)                              # [B,T,3]

        label_shift = torch.cat([labels[:, :1], labels[:, :-1]], dim=1)
        valid = labels != -100
        shifted_valid = torch.cat([valid[:, :1], valid[:, :-1]], dim=1)
        is_seam = (labels != label_shift) & valid & shifted_valid

        seam_gates.append(gw[is_seam].cpu().float().numpy())
        non_seam_gates.append(gw[~is_seam & valid].cpu().float().numpy())

    s = np.concatenate(seam_gates, axis=0) if seam_gates else None
    n = np.concatenate(non_seam_gates, axis=0) if non_seam_gates else None
    tau = float(claf.gate_temperature.detach().cpu().item())

    print(f"  [CLAF] tau={tau:.3f}  ", end="")
    if tau > 2.0:
        print("(soft -- CLAF may not be differentiating)")
    elif tau < 0.3:
        print("(very sharp -- check for instability)")
    else:
        print("(healthy range)")
    if s is not None and len(s) > 0:
        print(f"  [CLAF] seam     gates: "
              f"syn={s[:, 0].mean():.3f}  sem={s[:, 1].mean():.3f}  "
              f"dis={s[:, 2].mean():.3f}  (n={len(s)})")
    if n is not None and len(n) > 0:
        print(f"  [CLAF] non-seam gates: "
              f"syn={n[:, 0].mean():.3f}  sem={n[:, 1].mean():.3f}  "
              f"dis={n[:, 2].mean():.3f}  (n={len(n)})")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(model: nn.Module, save_dir: Path,
                    args: argparse.Namespace) -> None:
    """Save MODEL ONLY. With v4's `modules_to_save=[claf, input_norm,
    conv_head, bpm, classifier, boundary_head, aux_heads, crf]` the
    LoRA adapter directory bundles BOTH the LoRA delta weights AND the
    full state of those head modules, so a single save_pretrained call
    captures everything we need to reconstruct the model later.

    For a fully-resumable checkpoint that also captures
    optimizer / scheduler / RNG state, call save_training_state()
    instead -- it wraps this function and adds the rest."""
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.use_lora:
        model.save_pretrained(str(save_dir / "lora_adapter"))
    else:
        torch.save(model.state_dict(), save_dir / "full_model.pth")
    with open(save_dir / "training_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)


def save_training_state(model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler,
                        save_dir: Path,
                        args: argparse.Namespace,
                        progress: Dict,
                        ema: Optional["WeightEMA"] = None) -> None:
    """Save EVERYTHING needed to bit-exactly resume training: model,
    optimizer state, scheduler state, RNG states, EMA shadow (if any),
    and progress counters."""
    save_checkpoint(model, save_dir, args)

    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, save_dir / "optimizer.pth")

    if ema is not None:
        # EMA shadow saved separately so resume can pick up the running
        # average even if `best/` and the latest `step_*` diverge.
        torch.save(ema.state_dict(), save_dir / "ema_state.pth")

    rng = {
        "python":    random.getstate(),
        "numpy":     np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, save_dir / "rng_state.pth")

    with open(save_dir / "training_state.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def latest_step_dir(out_dir: Path) -> Optional[Path]:
    """Return the `step_NNNNNN/` directory with the highest step number,
    or None if none exist. Used by the resume path and the end-of-run
    summary to find the most recent crash-safety save when checkpoints
    are written to step-numbered folders rather than a single rolling
    `last/` directory."""
    best: Optional[Tuple[int, Path]] = None
    if not out_dir.is_dir():
        return None
    for p in out_dir.glob("step_*"):
        if not p.is_dir():
            continue
        if not (p / "training_state.json").exists():
            continue
        try:
            n = int(p.name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        if best is None or n > best[0]:
            best = (n, p)
    return best[1] if best else None


def load_training_state(load_dir: Path,
                        model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler,
                        device: torch.device,
                        use_lora: bool,
                        ema: Optional["WeightEMA"] = None) -> Tuple[nn.Module, Dict]:
    """Inverse of save_training_state. Returns (model, progress).

    For LoRA models, the caller has ALREADY wrapped the bare SeamDetector
    in `get_peft_model(...)` before this function runs. We therefore load
    adapter weights IN-PLACE into that existing PeftModel via
    `set_peft_model_state_dict`, NOT via `PeftModel.from_pretrained`.
    Re-wrapping with from_pretrained on an already-wrapped model would
    cause two failures:
      1. Doubled `base_model.model.base_model.model.` key prefix on every
         saved tensor → KeyError on every key in the checkpoint.
      2. Even if (1) were patched, from_pretrained creates fresh
         parameter objects, leaving the optimizer (built on the original
         params) with stale references → updates silently dropped.
    Only fall back to from_pretrained when the input is a bare base model.
    """
    if use_lora:
        adapter_dir = load_dir / "lora_adapter"
        if not adapter_dir.exists():
            sys.exit(f"Resume: LoRA adapter dir not found at {adapter_dir}")

        if isinstance(model, PeftModel):
            from peft import set_peft_model_state_dict

            st_path = adapter_dir / "adapter_model.safetensors"
            bin_path = adapter_dir / "adapter_model.bin"
            if st_path.exists():
                from safetensors.torch import load_file as _load_st
                adapter_state = _load_st(str(st_path))
            elif bin_path.exists():
                adapter_state = torch.load(str(bin_path), map_location=device)
            else:
                sys.exit(
                    f"Resume: no adapter_model.{{safetensors,bin}} found in "
                    f"{adapter_dir}"
                )

            result = set_peft_model_state_dict(
                model, adapter_state, adapter_name="default",
            )
            missing = list(getattr(result, "missing_keys", []) or [])
            unexpected = list(getattr(result, "unexpected_keys", []) or [])
            # The base-model frozen weights are intentionally absent from
            # the adapter state, so a long missing_keys list of plain
            # backbone tensor names is normal and not a warning. We only
            # surface keys that look like adapter / module-to-save paths.
            interesting_missing = [
                k for k in missing
                if "lora_" in k or "modules_to_save" in k
            ]
            if interesting_missing:
                print(f"  WARN: {len(interesting_missing)} adapter / "
                      f"modules_to_save key(s) missing from checkpoint "
                      f"(first 3): {interesting_missing[:3]}")
            if unexpected:
                print(f"  WARN: {len(unexpected)} unexpected key(s) in "
                      f"adapter checkpoint (first 3): {unexpected[:3]}")
            print(f"  loaded adapter weights into existing PeftModel "
                  f"({len(adapter_state):,} tensors)")
        else:
            model = PeftModel.from_pretrained(
                model, str(adapter_dir), is_trainable=True,
            )
    else:
        bb_path = load_dir / "full_model.pth"
        # weights_only=False because optimizer.pth / rng_state.pth /
        # full_model.pth are files WE wrote in save_training_state(),
        # so the unpickling code path is trusted. PyTorch 2.6+ changed
        # the default to True, which rejects numpy arrays (rng state),
        # scheduler state (Python ints/floats), and bnb optimizer
        # state -- all legitimate contents here.
        state = torch.load(bb_path, map_location=device, weights_only=False)
        model.load_state_dict(state)

    model = model.to(device)

    opt_path = load_dir / "optimizer.pth"
    if opt_path.exists():
        opt_state = torch.load(opt_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(opt_state["optimizer"])
        scheduler.load_state_dict(opt_state["scheduler"])
    else:
        print(f"WARN: no optimizer.pth in {load_dir}; resuming with fresh "
              f"optimizer / scheduler state")

    rng_path = load_dir / "rng_state.pth"
    if rng_path.exists():
        rng = torch.load(rng_path, map_location="cpu", weights_only=False)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if "torch_cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    if ema is not None:
        ema_path = load_dir / "ema_state.pth"
        if ema_path.exists():
            ema_state = torch.load(ema_path, map_location="cpu", weights_only=False)
            ema.load_state_dict(ema_state)
            print(f"  loaded EMA shadow ({len(ema.shadow):,} tensors, "
                  f"decay={ema.decay})")
        else:
            print(f"  no ema_state.pth in {load_dir}; EMA reinitialized "
                  f"from current weights")

    state_path = load_dir / "training_state.json"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            return model, json.load(f)
    print(f"WARN: no training_state.json in {load_dir}; using defaults")
    return model, {}


# ---------------------------------------------------------------------------
# Optimizer parameter grouping
# ---------------------------------------------------------------------------
import re as _re_layer  # used for parsing 'encoder.layer.<N>' from param names


def _backbone_layer_idx(name: str) -> Optional[int]:
    """Return the encoder layer index for a parameter name, or None
    if the parameter does not live inside a transformer layer (e.g.
    embeddings, position encoders, layer norms outside the layer
    stack)."""
    m = _re_layer.search(r"\.layer\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _detect_num_backbone_layers(model: nn.Module, fallback: int = 24) -> int:
    """Best-effort introspection of how many transformer layers the
    backbone has. DeBERTa-v3-Large = 24; this is also the v4 fallback.
    """
    seam = None
    for module in model.modules():
        if isinstance(module, SeamDetector):
            seam = module
            break
    if seam is None:
        return fallback
    try:
        return len(seam.backbone.encoder.layer)
    except AttributeError:
        return fallback


def build_param_groups(model: nn.Module, args: argparse.Namespace) -> List[Dict]:
    """v4.1 parameter grouping with optional LLRD (layer-wise learning
    rate decay) on backbone LoRA params.

    Without LLRD (--llrd-decay 0.0, the default for backwards compat):
        backbone (LoRA, all layers)      lr=encoder_lr, wd=weight_decay
        CLAF                              lr=claf_lr,    wd=claf_weight_decay
        head                              lr=head_lr,    wd=head_weight_decay

    With LLRD (--llrd-decay D in (0, 1), typical 0.9–0.95):
        For each transformer layer i ∈ [0..N-1]:
            backbone-layer-i                lr=encoder_lr * D^(N-1-i)
        Embedding/non-layer LoRA            lr=encoder_lr * D^N
        CLAF + head as before.

    Why LLRD: late layers carry the high-level semantics most relevant
    to seam detection, so they should adapt faster. Early layers
    encode general syntax already learned by DeBERTa pretraining and
    benefit from smaller updates. Standard LLM-finetuning trick;
    consistently +0.5-1.0 F1 on token-classification tasks at zero
    extra compute or memory.
    """
    llrd_decay = float(getattr(args, "llrd_decay", 0.0) or 0.0)
    use_llrd = 0.0 < llrd_decay < 1.0

    n_layers = _detect_num_backbone_layers(model) if use_llrd else 0
    if use_llrd:
        print(f"LLRD enabled  (decay={llrd_decay}, n_layers={n_layers})")

    # Buckets
    head_params: List[torch.nn.Parameter] = []
    claf_params: List[torch.nn.Parameter] = []
    # backbone_buckets: dict[layer_idx] -> list[param]; key=-1 for
    # non-layer (embedding/etc.) backbone params.
    backbone_buckets: Dict[int, List[torch.nn.Parameter]] = {}
    seen_ids = set()

    def _classify(name: str) -> str:
        n = name.lower()
        if "lora_" in n:
            return "backbone"
        if "claf" in n:
            return "claf"
        return "head"

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        bucket = _classify(name)
        if bucket == "backbone":
            li = _backbone_layer_idx(name) if use_llrd else None
            key = li if li is not None else -1
            backbone_buckets.setdefault(key, []).append(param)
        elif bucket == "claf":
            claf_params.append(param)
        else:
            head_params.append(param)

    backbone_total = sum(sum(p.numel() for p in v) for v in backbone_buckets.values())
    print(f"Param groups -- backbone(LoRA): {backbone_total:,} ; "
          f"CLAF: {sum(p.numel() for p in claf_params):,} ; "
          f"head: {sum(p.numel() for p in head_params):,}")

    groups: List[Dict] = []
    if use_llrd:
        # Per-layer groups with decayed LR. Top of stack (layer N-1) gets
        # full --encoder-lr; bottom of stack (layer 0) gets the most
        # decay. Embeddings get one more decay step.
        for li, params in sorted(backbone_buckets.items()):
            if not params:
                continue
            if li == -1:
                # Embeddings / other non-layer backbone LoRA: deepest decay.
                mult = llrd_decay ** n_layers
                tag = "emb"
            else:
                mult = llrd_decay ** (n_layers - 1 - li)
                tag = f"L{li}"
            groups.append({
                "params": params,
                "lr": args.encoder_lr * mult,
                "weight_decay": args.weight_decay,
                "_tag": f"backbone-{tag}",
            })
    else:
        # Single backbone group (all LoRA params at one LR).
        all_backbone = [p for v in backbone_buckets.values() for p in v]
        if all_backbone:
            groups.append({
                "params": all_backbone,
                "lr": args.encoder_lr,
                "weight_decay": args.weight_decay,
                "_tag": "backbone",
            })

    if claf_params:
        groups.append({
            "params": claf_params,
            "lr": args.claf_lr,
            "weight_decay": args.claf_weight_decay,
            "_tag": "claf",
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": args.head_lr,
            "weight_decay": args.head_weight_decay,
            "_tag": "head",
        })

    if use_llrd:
        # Print per-layer LR table for visibility.
        for g in groups:
            tag = g.get("_tag", "?")
            if tag.startswith("backbone-"):
                print(f"  {tag:<14s} lr={g['lr']:.2e}  "
                      f"params={sum(p.numel() for p in g['params']):,}")

    # Strip the diagnostic '_tag' field — torch.optim doesn't care about it
    # but rejects unknown kwargs in some bnb versions.
    for g in groups:
        g.pop("_tag", None)
    return groups


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  bf16={args.bf16}  |  LoRA={args.use_lora}  |  "
          f"checkpointing={args.gradient_checkpointing}  |  FGM={args.use_fgm}")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Datasets
    train_paths = [Path(p) for p in args.train_csv]
    val_paths = [Path(p) for p in (args.val_csv or [])]

    print(f"Loading train rows from: {[str(p) for p in train_paths]}")
    train_ds = SeamDataset(
        train_paths, tokenizer,
        max_length=args.max_length, stride=args.stride,
        shuffle_rows=True, max_rows=args.max_train_rows,
        seed=args.seed or 0,
        boundary_sigma=args.boundary_sigma,
    )
    print(f"  train chunks: {len(train_ds)}")

    val_ds = None
    if val_paths:
        print(f"Loading val rows from: {[str(p) for p in val_paths]}")
        val_ds = SeamDataset(
            val_paths, tokenizer,
            max_length=args.max_length, stride=args.stride,
            shuffle_rows=False, max_rows=args.max_val_rows,
            seed=(args.seed or 0) + 1,
            boundary_sigma=args.boundary_sigma,
        )
        print(f"  val chunks:   {len(val_ds)}")

    micro_batches_per_epoch = len(train_ds) // args.batch_size  # drop_last=True
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.dataloader_workers, pin_memory=True,
        )

    # Model
    print(f"Loading backbone: {args.model_name}")
    model = SeamDetector(
        model_name=args.model_name,
        syntax_range=tuple(args.syntax_range),
        semantic_range=tuple(args.semantic_range),
        discourse_range=tuple(args.discourse_range),
        hidden_dropout=args.hidden_dropout,
        attn_dropout=args.attn_dropout,
    )

    # Gradient checkpointing must be enabled BEFORE PEFT wrapping for
    # the underlying transformer to honor it. Also call
    # enable_input_require_grads() so gradients flow back through the
    # frozen embeddings into the LoRA adapters when checkpointing is
    # on (a subtle but well-documented PEFT + checkpointing gotcha).
    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        if hasattr(model.backbone, "enable_input_require_grads"):
            model.backbone.enable_input_require_grads()

    # LoRA wrapping over the WHOLE SeamDetector. modules_to_save lists
    # all v4 head components so save_pretrained captures them.
    if args.use_lora:
        # target_modules: which Linear layers inside the BACKBONE get a
        # LoRA adapter. v4 default is q,k,v,o (vs v3's q,v only) — same
        # rank, ~2x effective adaptation surface. The module names match
        # DeBERTa-v2/v3's attention block.
        target_modules = list(args.lora_target_modules)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            modules_to_save=[
                # CLAF v3 + LayerNorm
                "claf", "input_norm",
                # 2-layer dilated conv head (single nn.Module wrapping all branches)
                "conv_head",
                # Boundary Prototype Memory v2
                "bpm",
                # Classifier + boundary head + auxiliary heads + CRF
                "classifier", "boundary_head", "aux_heads", "crf",
            ],
        )
        model = get_peft_model(model, peft_config)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    model = model.to(device)

    # Trainable param report
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} / {total_params:,} "
          f"({100 * trainable / total_params:.2f}%)")

    # Optimization A: torch.compile
    # ------------------------------------------------------------------
    # Compiles the SeamDetector forward graph (DeBERTa + CLAF + conv +
    # BPM + classifier) into optimized CUDA kernels. Typical speedup on
    # A100 with bf16 + LoRA: 1.3-1.7x with no quality cost.
    #
    # mode="reduce-overhead" reuses CUDA graphs and minimizes Python
    # overhead between GPU calls. dynamic=False locks shapes for
    # max performance (we always use the same fixed [B, max_length]
    # tensor shapes thanks to padding="max_length" in SeamDataset).
    #
    # The CRF call is gated by autocast(enabled=False) and runs in
    # eager mode with graph breaks -- pytorch-crf uses Python control
    # flow that compile can't fuse. That's fine; the eager-mode CRF
    # is already only ~1-2% of total step time.
    #
    # First training step takes 60-90s extra for compilation. Cached
    # afterwards (same compiled graph reused for all subsequent steps).
    if args.torch_compile:
        try:
            t_compile = time.time()
            print(f"torch.compile: targeting SeamDetector._features "
                  f"(shape-stable hot path used by both train & eval). "
                  f"Adds ~60-90s to first step.")
            compile_mode = args.torch_compile_mode

            # Compile the shared _features method (DeBERTa + CLAF + conv
            # + BPM). Both _emissions (inference) and
            # compute_training_outputs (training) call _features, so
            # compiling it once benefits both code paths. The classifier
            # / boundary head / aux heads / CRF stay in eager mode —
            # they are tiny relative to the backbone and contain no
            # branchy Python control flow.
            seam_det: Optional[SeamDetector] = None
            for module in model.modules():
                if isinstance(module, SeamDetector):
                    seam_det = module
                    break

            if seam_det is None:
                # Couldn't find SeamDetector (shouldn't happen with our
                # wrapping). Fall back to compiling the whole model.
                model = torch.compile(model, mode=compile_mode, dynamic=False)
                print(f"torch.compile: SeamDetector not found inside "
                      f"wrapper; compiled whole model (mode={compile_mode}).")
            else:
                seam_det._features = torch.compile(
                    seam_det._features,
                    mode=compile_mode,
                    dynamic=False,
                )
                print(f"torch.compile: enabled on SeamDetector._features "
                      f"(mode={compile_mode}). setup={time.time() - t_compile:.1f}s")

            # Bump dynamo's cache and recompile limits as a safety net.
            # The default 8 can be too low when fast tokenizers + LoRA +
            # FGM combine to produce subtly varying graph traces during
            # the first few hundred steps. After warm-up, dynamo settles
            # and these limits are never touched again.
            try:
                import torch._dynamo as _td
                _td.config.cache_size_limit = 128
                _td.config.recompile_limit = 64
            except (ImportError, AttributeError):
                pass
        except Exception as e:
            print(f"torch.compile FAILED ({type(e).__name__}: {e}). "
                  f"Continuing in eager mode -- training quality is "
                  f"unaffected, just no compile speedup.")

    # Optimizer with three differential learning rates.
    optim_groups = build_param_groups(model, args)

    optimizer = None
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(optim_groups)
            print("Optimizer: AdamW8bit (bitsandbytes)")
        except ImportError:
            print("bitsandbytes not installed; falling back to torch.optim.AdamW. "
                  "Install with `pip install bitsandbytes` to save ~2 GB VRAM.")
    if optimizer is None:
        optimizer = torch.optim.AdamW(optim_groups)
        print("Optimizer: torch.optim.AdamW")

    # LR schedule
    update_steps_per_epoch = max(
        1, micro_batches_per_epoch // args.gradient_accumulation_steps,
    )
    total_update_steps = update_steps_per_epoch * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_update_steps),
        num_training_steps=total_update_steps,
    )
    print(f"Total update steps: {total_update_steps}  "
          f"(effective batch = {args.batch_size * args.gradient_accumulation_steps})")

    # Feature-level FGM regularizer (off by default in v4)
    fgm = (FeatureFGM(model, target_substring=args.fgm_target,
                      epsilon=args.fgm_epsilon)
           if args.use_fgm else None)
    if fgm:
        print(f"FGM enabled  (target='{args.fgm_target}', "
              f"epsilon={args.fgm_epsilon})  -- adds 1 forward+backward "
              f"per micro-batch")

    # Weight EMA — averages trainable parameters across optimizer steps
    # to produce a smoother final model. Active only when --use-ema is
    # set; the EMA copy is what gets evaluated and saved as 'best'.
    ema = (WeightEMA(model, decay=args.ema_decay)
           if args.use_ema else None)
    if ema is not None:
        print(f"Weight EMA enabled  (decay={args.ema_decay})  -- "
              f"validation runs against EMA weights, raw weights are "
              f"restored before the next training step.")

    # Autocast context
    if args.bf16 and device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save cadence: independent of validation. If --save-every-steps is
    # not set (0), fall back to --validate-every-steps so saves continue
    # at the validation cadence (preserves the prior behaviour where
    # save and validation were locked together).
    effective_save_every = (
        args.save_every_steps
        if args.save_every_steps and args.save_every_steps > 0
        else args.validate_every_steps
    )
    if args.save_every_steps and args.save_every_steps > 0:
        print(f"Save cadence: every {effective_save_every} steps "
              f"(validation: every {args.validate_every_steps} steps).")
    else:
        print(f"Save cadence: every {effective_save_every} steps "
              f"(== validation cadence; pass --save-every-steps to decouple).")

    best_seam_offset = float("inf")
    best_f1_at_5 = -1.0
    best_reward = -1.0           # the validator's composite reward; primary criterion
    epochs_since_improvement = 0
    global_update = 0
    start_epoch = 0
    resume_skip_samples_in_first_epoch = 0
    base_shuffle_seed = (args.seed if args.seed is not None else 0) ^ 0xCAFE

    # ----- Resume from the most recent `step_NNNNNN/` if requested ------
    # Each validation save now lives in its own `step_<global_update>/`
    # folder rather than overwriting a single `last/` directory. This
    # keeps the full save history on disk so a collapse at step N can be
    # recovered by deleting the corrupted later folders and resuming
    # from any earlier good step. The user manages disk cleanup.
    if args.resume:
        resume_dir = latest_step_dir(out_dir)
        if resume_dir is None:
            print(f"--resume requested but no `step_*/` checkpoint found "
                  f"in {out_dir}. Starting from scratch.")
        else:
            print(f"Resuming from {resume_dir}")
            model, progress = load_training_state(
                resume_dir, model, optimizer, scheduler, device,
                use_lora=args.use_lora,
                ema=ema,
            )
            if fgm:
                fgm = FeatureFGM(model, target_substring=args.fgm_target,
                                 epsilon=args.fgm_epsilon)
            start_epoch = int(progress.get("epoch_completed", 0))
            global_update = int(progress.get("global_update", 0))
            best_seam_offset = float(progress.get("best_seam_offset", float("inf")))
            best_f1_at_5 = float(progress.get("best_f1_at_5", -1.0))
            best_reward = float(progress.get("best_reward", -1.0))
            epochs_since_improvement = int(progress.get("epochs_since_improvement", 0))
            saved_step_in_epoch = int(progress.get("step_within_epoch", 0))
            if saved_step_in_epoch > 0:
                resume_skip_samples_in_first_epoch = saved_step_in_epoch * args.batch_size
                print(f"  resumed mid-epoch {start_epoch+1}: skipping first "
                      f"{saved_step_in_epoch} mini-batches "
                      f"({resume_skip_samples_in_first_epoch} samples) of the "
                      f"epoch's permutation; global_update={global_update}; "
                      f"best_seam_offset={best_seam_offset:.2f}")
            else:
                print(f"  resumed at epoch {start_epoch+1}/step {global_update}; "
                      f"best_seam_offset={best_seam_offset:.2f}")

    optimizer.zero_grad()

    # Helper: compute the multi-term loss given a fresh model output dict.
    crf_module = _resolve_crf(model)

    def _step_loss(outputs,
                   labels: torch.Tensor,
                   attention_mask: torch.Tensor,
                   boundary_target: Optional[torch.Tensor] = None,
                   data_source_id: Optional[torch.Tensor] = None,
                   model_family_id: Optional[torch.Tensor] = None,
                   sample_type_id: Optional[torch.Tensor] = None):
        return compute_total_loss(
            outputs, labels, attention_mask, crf_module,
            boundary_target=boundary_target,
            data_source_id=data_source_id,
            model_family_id=model_family_id,
            sample_type_id=sample_type_id,
            lambda_focal=args.lambda_focal,
            lambda_boundary=args.lambda_boundary,
            lambda_data_source=args.lambda_data_source,
            lambda_model_family=args.lambda_model_family,
            lambda_sample_type=args.lambda_sample_type,
            focal_gamma=args.focal_gamma,
            focal_seam_alpha=args.focal_seam_alpha,
            return_components=True,
        )

    def _do_validation(epoch_idx: int, step_within_epoch: int,
                       end_of_epoch: bool) -> bool:
        """Returns True if early-stop was triggered."""
        nonlocal best_seam_offset, best_f1_at_5, best_reward, epochs_since_improvement
        if val_loader is None:
            return False

        # If EMA is enabled AND past warmup, evaluate and save against
        # the EMA weights — those are the weights we ultimately serve.
        # The raw weights are snapshotted now so the optimizer can
        # resume against the un-averaged parameters once we're done.
        # During warmup the shadow is just the initial weights, so
        # using it would regress validation; fall back to raw weights.
        ema_active_now = (
            ema is not None and global_update >= int(args.ema_warmup_steps)
        )
        if ema_active_now:
            ema.store(model)
            ema.apply(model)

        # Pre-initialize so the patience block downstream still has a
        # value to test against if evaluate() raises and we propagate.
        improved = False
        try:
            model.eval()
            metrics = evaluate(model, val_loader, device, autocast_ctx)
            if args.gate_monitor:
                log_gate_statistics(model, val_loader, device,
                                    num_batches=args.gate_monitor_batches)
            model.train()
            cur_offset = metrics["mean_seam_offset"]
            cur_f1_at_5 = metrics["f1_at_5"]
            cur_token_f1 = metrics["token_f1"]
            cur_fp = metrics["validator_fp_score"]
            cur_ap = metrics["validator_ap_score"]
            cur_val_f1 = metrics["validator_f1_score"]
            cur_reward = metrics["validator_reward"]
            ema_tag = " [EMA]" if ema_active_now else (" [raw]" if ema is not None else "")
            # Validator-aligned line — this is the number we optimize for.
            print(f"  [val{ema_tag} @ epoch {epoch_idx+1} step {global_update}] "
                  f"REWARD={cur_reward:.4f}  "
                  f"f1={cur_val_f1:.4f}  fp={cur_fp:.4f}  ap={cur_ap:.4f}")
            # Diagnostics underneath.
            print(f"          diag: seam_offset={cur_offset:.2f}  "
                  f"f1_at_5={cur_f1_at_5:.4f}  token_f1={cur_token_f1:.4f}  "
                  f"({metrics['n_with_seam']}/{metrics['n_total']} had a seam)")

            # Collapse guard: refuses to save 'best' if any of the
            # following indicate divergence rather than improvement:
            #   1. token_f1 < 0.50 — random chance, broken model
            #   2. seam coverage < 25% — model has stopped predicting
            #      seams entirely (everything labeled all-0 or all-1)
            #   3. validator_reward has dropped more than 30% from best —
            #      sharp regression (looser bound than f1_at_5 since
            #      reward is more stable across distribution shifts).
            coverage = metrics["n_with_seam"] / max(1, metrics["n_total"])
            reward_floor = 0.7 * max(best_reward, 0.0)
            collapse_suspected = (
                coverage < 0.25
                or cur_token_f1 < 0.50
                or (best_reward > 0 and cur_reward < reward_floor)
            )
            # Best criterion: maximize the validator's exact composite
            # reward. f1_at_5 / mean_seam_offset stay as diagnostics.
            improved = (
                (not math.isnan(cur_reward))
                and cur_reward > best_reward
                and not collapse_suspected
            )
            if collapse_suspected and not math.isnan(cur_reward) and cur_reward > best_reward:
                print(f"  COLLAPSE GUARD tripped: refusing to save best "
                      f"(coverage={coverage:.1%}, token_f1={cur_token_f1:.3f}, "
                      f"reward={cur_reward:.3f} vs floor={reward_floor:.3f}). "
                      f"The reward improvement is an artifact of degenerate "
                      f"predictions, not a real gain.")
            if improved:
                best_reward = cur_reward
                best_seam_offset = cur_offset
                best_f1_at_5 = cur_f1_at_5
                # Save EMA weights (currently active in `model`) as best.
                save_training_state(
                    model, optimizer, scheduler, out_dir / "best", args,
                    progress={
                        "epoch_completed":          epoch_idx,
                        "step_within_epoch":        step_within_epoch,
                        "global_update":            global_update,
                        "best_reward":              best_reward,
                        "best_seam_offset":         best_seam_offset,
                        "best_f1_at_5":             best_f1_at_5,
                        "validator_f1_score":       cur_val_f1,
                        "validator_fp_score":       cur_fp,
                        "validator_ap_score":       cur_ap,
                        "epochs_since_improvement": 0,
                        "saved_at":                 "validation_improved",
                        "saved_with_ema":           ema is not None,
                    },
                    ema=ema,
                )
                print(f"  saved new best "
                      f"(REWARD={best_reward:.4f}, f1={cur_val_f1:.4f}, "
                      f"fp={cur_fp:.4f}, ap={cur_ap:.4f}) to {out_dir / 'best'}")
        finally:
            # Always restore raw weights so the optimizer keeps stepping
            # against the un-averaged parameters, even if evaluate() or
            # save raised. Only needed if we actually applied EMA.
            if ema_active_now:
                ema.restore(model)

        if end_of_epoch:
            if improved:
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if args.patience and epochs_since_improvement >= args.patience:
                    print(f"  early stop: no val improvement for "
                          f"{epochs_since_improvement} epoch(s) (--patience "
                          f"{args.patience}). Halting.")
                    return True
        return False

    def _save_last(epoch_idx: int, step_within_epoch: int, reason: str) -> None:
        # Each save lives in its own `step_<global_update>/` folder.
        # Width 6 digits handles up to 999,999 optimizer updates;
        # lexicographic sort matches numeric order so `ls -1` shows
        # the save history in training order. Saves the RAW (non-EMA)
        # weights here so resume picks up exactly where training was;
        # the `best/` folder holds the EMA snapshot separately.
        step_dir = out_dir / f"step_{global_update:06d}"
        save_training_state(
            model, optimizer, scheduler, step_dir, args,
            progress={
                "epoch_completed":          epoch_idx,
                "step_within_epoch":        step_within_epoch,
                "global_update":            global_update,
                "best_reward":              best_reward,
                "best_seam_offset":         best_seam_offset,
                "best_f1_at_5":             best_f1_at_5,
                "epochs_since_improvement": epochs_since_improvement,
                "saved_at":                 reason,
            },
            ema=ema,
        )

    early_stopped = False

    # Loss-spike circuit breaker. Tracks an exponential moving average
    # (EMA) of recent training losses; when a single batch's loss
    # exceeds spike_factor * EMA after warm-up, we skip the optimizer
    # update for that batch instead of letting 8-bit Adam absorb a
    # corrupted moment estimate. The 8-bit moment quantization is
    # what makes a single bad step compound into the runaway cascade
    # we saw at step 3760: one giant gradient corrupts the second-
    # moment buffer, the next ~50 steps apply wrongly-scaled updates,
    # and the model collapses to "predict no seam everywhere".
    loss_ema = None
    loss_ema_decay = 0.99
    spike_warmup_steps = 200          # let EMA settle before guarding
    spike_factor = float(args.loss_spike_factor)

    for epoch in range(start_epoch, args.num_epochs):
        if early_stopped:
            break

        skip_samples = (resume_skip_samples_in_first_epoch
                        if epoch == start_epoch else 0)
        skip_at_start = skip_samples
        train_loader = _make_epoch_loader(
            train_ds, args, base_shuffle_seed, epoch, skip_samples=skip_samples,
        )
        if train_loader is None:
            print(f"  epoch {epoch+1}: nothing left to do, advancing.")
            resume_skip_samples_in_first_epoch = 0
            continue
        resume_skip_samples_in_first_epoch = 0

        model.train()
        running_loss = 0.0
        running_n = 0
        last_components: Optional[Dict[str, torch.Tensor]] = None

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            boundary_target = batch.get("boundary_target")
            data_source_id  = batch.get("data_source_id")
            model_family_id = batch.get("model_family_id")
            sample_type_id  = batch.get("sample_type_id")
            if boundary_target is not None:
                boundary_target = boundary_target.to(device, non_blocking=True)
            if data_source_id is not None:
                data_source_id = data_source_id.to(device, non_blocking=True)
            if model_family_id is not None:
                model_family_id = model_family_id.to(device, non_blocking=True)
            if sample_type_id is not None:
                sample_type_id = sample_type_id.to(device, non_blocking=True)

            # ---- Standard forward + backward --------------------------
            with autocast_ctx:
                outputs = model(input_ids, attention_mask, labels=labels)
                loss, components = _step_loss(
                    outputs, labels, attention_mask,
                    boundary_target=boundary_target,
                    data_source_id=data_source_id,
                    model_family_id=model_family_id,
                    sample_type_id=sample_type_id,
                )

                # ---- R-Drop: second stochastic forward pass ------------
                # Same inputs, different dropout masks. We add a symmetric
                # KL between the two emission distributions on top of
                # averaging the two task losses. Cost: 2× fwd + 2× bwd.
                if args.use_rdrop and args.lambda_rdrop > 0:
                    outputs2 = model(input_ids, attention_mask, labels=labels)
                    loss2, _ = _step_loss(
                        outputs2, labels, attention_mask,
                        boundary_target=boundary_target,
                        data_source_id=data_source_id,
                        model_family_id=model_family_id,
                        sample_type_id=sample_type_id,
                    )
                    em_no_cls_1 = outputs["emissions"][:, 1:, :]
                    em_no_cls_2 = outputs2["emissions"][:, 1:, :]
                    rdrop_kl = compute_rdrop_kl(
                        em_no_cls_1, em_no_cls_2,
                        labels[:, 1:], attention_mask[:, 1:],
                    )
                    # Average the two task losses; add KL with its λ.
                    loss = 0.5 * (loss + loss2) + args.lambda_rdrop * rdrop_kl
                    components["rdrop_kl"] = rdrop_kl.detach()

            if not torch.isfinite(loss):
                print(f"  WARN: non-finite loss at step {step+1} "
                      f"(loss={loss.item()}). Skipping batch and "
                      f"zeroing pending gradients.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_value = loss.item()

            # Loss-spike circuit breaker. After warm-up, if a single
            # batch's loss exceeds spike_factor * EMA, drop the batch.
            # Catches the runaway cascade BEFORE 8-bit Adam absorbs
            # the corrupted moment estimate.
            if (loss_ema is not None
                    and global_update >= spike_warmup_steps
                    and loss_value > spike_factor * loss_ema):
                print(f"  SPIKE GUARD tripped at step {step+1}: "
                      f"loss={loss_value:.2f} > {spike_factor:.1f}x "
                      f"EMA={loss_ema:.2f}. Skipping batch (no backward, "
                      f"no optimizer step).")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Update EMA only with healthy batches.
            if loss_ema is None:
                loss_ema = loss_value
            else:
                loss_ema = loss_ema_decay * loss_ema + (1 - loss_ema_decay) * loss_value

            (loss / args.gradient_accumulation_steps).backward()

            # ---- Feature-level FGM adversarial pass -------------------
            adv_loss_value = 0.0
            if fgm is not None:
                fgm.attack()
                with autocast_ctx:
                    outputs_adv = model(input_ids, attention_mask, labels=labels)
                    loss_adv, _ = _step_loss(
                        outputs_adv, labels, attention_mask,
                        boundary_target=boundary_target,
                        data_source_id=data_source_id,
                        model_family_id=model_family_id,
                        sample_type_id=sample_type_id,
                    )
                if torch.isfinite(loss_adv):
                    (loss_adv / args.gradient_accumulation_steps).backward()
                    adv_loss_value = loss_adv.item()
                else:
                    print(f"  WARN: non-finite FGM adversarial loss at "
                          f"step {step+1}; skipping adversarial backward")
                fgm.restore()

            running_loss += loss.item() + adv_loss_value
            running_n += 1 + (1 if fgm is not None else 0)

            # Latest-step component breakdown for the periodic log. We
            # record the LAST micro-batch's components rather than averaging
            # because: (a) they are dominated by per-batch noise on the
            # short scale of one accumulation window anyway, (b) showing
            # raw values is more honest about scale, and (c) avoiding a
            # running average keeps memory & complexity tiny.
            last_components = components

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # Enforce P(1->0) >= min_prob in the CRF transition matrix
                # only when the floor is set (>0). Default v4 is 0 (no
                # constraint), so this becomes a no-op cheaply.
                if args.min_p_1to0 and args.min_p_1to0 > 0.0:
                    constrain_crf_transitions(model, min_prob=args.min_p_1to0)
                # EMA: update shadow weights AFTER the optimizer step has
                # written the new param values. Inexpensive (host<->dev
                # copy of trainable params only).
                if ema is not None:
                    if global_update + 1 == int(args.ema_warmup_steps):
                        # End of warmup: re-snapshot the shadow to the
                        # current weights so EMA starts averaging from
                        # this point onward (rather than averaging in
                        # the very-early random-ish weights).
                        ema = WeightEMA(model, decay=args.ema_decay)
                        print(f"  EMA shadow re-initialized at step "
                              f"{global_update + 1} "
                              f"(--ema-warmup-steps={args.ema_warmup_steps})")
                    elif global_update + 1 > int(args.ema_warmup_steps):
                        ema.update(model)
                global_update += 1

                if global_update % args.log_every == 0:
                    avg_loss = running_loss / max(1, running_n)
                    lrs = [g["lr"] for g in optimizer.param_groups]
                    lr_str = "/".join(f"{lr:.2e}" for lr in lrs)
                    comp_str = ""
                    if last_components is not None:
                        parts = []
                        for k in ("crf", "focal", "boundary", "aux_ds",
                                  "aux_mf", "aux_st", "rdrop_kl"):
                            if k in last_components:
                                parts.append(f"{k}={last_components[k].item():.3f}")
                        if parts:
                            comp_str = "  [" + " ".join(parts) + "]"
                    print(f"  step {global_update:5d}/{total_update_steps}  "
                          f"loss={avg_loss:.4f}  lr={lr_str}{comp_str}")
                    running_loss, running_n = 0.0, 0

                # ---- Mid-epoch validation (independent) ----------------
                if (args.validate_every_steps
                        and global_update % args.validate_every_steps == 0):
                    abs_mb = (skip_at_start // args.batch_size) + (step + 1)
                    early_stopped_now = _do_validation(
                        epoch, abs_mb, end_of_epoch=False,
                    )
                    if early_stopped_now:
                        early_stopped = True
                        break

                # ---- Crash-safety save (independent) ------------------
                # Saves a `step_NNNNNN/` checkpoint every save_every_steps
                # updates, regardless of whether validation fired. Set on
                # its own cadence -- typically more frequent than
                # validation (validation is expensive, saving is cheap).
                # When save_every_steps == 0 the loop falls back to the
                # validate cadence for save (current default behaviour).
                if (effective_save_every
                        and global_update % effective_save_every == 0):
                    abs_mb = (skip_at_start // args.batch_size) + (step + 1)
                    _save_last(epoch, abs_mb, "periodic_save")

        if early_stopped:
            break

        # Final flush (in case the epoch ended mid-accumulation)
        if (step + 1) % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if args.min_p_1to0 and args.min_p_1to0 > 0.0:
                constrain_crf_transitions(model, min_prob=args.min_p_1to0)
            if ema is not None and global_update + 1 > int(args.ema_warmup_steps):
                ema.update(model)
            global_update += 1

        print(f"Epoch {epoch+1}/{args.num_epochs} done.")

        early_stopped = _do_validation(epoch, step + 1, end_of_epoch=True)

        _save_last(epoch + 1, 0, "end_of_epoch")
        save_training_state(
            model, optimizer, scheduler, out_dir / f"epoch_{epoch+1}", args,
            progress={
                "epoch_completed":          epoch + 1,
                "step_within_epoch":        0,
                "global_update":            global_update,
                "best_reward":              best_reward,
                "best_seam_offset":         best_seam_offset,
                "best_f1_at_5":             best_f1_at_5,
                "epochs_since_improvement": epochs_since_improvement,
                "saved_at":                 "end_of_epoch",
            },
            ema=ema,
        )

    print("\nTraining complete.")
    if best_reward >= 0:
        print(f"Best val REWARD: {best_reward:.4f}  "
              f"(seam_offset={best_seam_offset:.2f}, "
              f"f1@5={best_f1_at_5:.4f})  saved to {out_dir / 'best'}")
    final_latest = latest_step_dir(out_dir)
    if final_latest is not None:
        print(f"Latest resumable checkpoint: {final_latest}")
    else:
        print(f"No `step_*/` checkpoints written (training likely "
              f"halted before the first validation save).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    p.add_argument("--train-csv", nargs="+", required=True,
                   help="One or more CSVs from build_training_dataset.py.")
    p.add_argument("--val-csv", nargs="*", default=None,
                   help="Validation CSV(s). Held out from --train-csv. "
                        "Optional but strongly recommended.")
    p.add_argument("--test-csv", nargs="*", default=None,
                   help="True hold-out test CSV(s). Path is recorded in "
                        "training_args.json so future evaluation scripts "
                        "can pick it up; the file itself is NEVER read "
                        "during training.")
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--max-val-rows", type=int, default=None)
    p.add_argument("--patience", type=int, default=0,
                   help="Early-stop after N validation rounds without "
                        "improvement in mean_seam_offset. Default 0 = "
                        "disabled. The doc recommends 3 for no-dropout "
                        "training.")
    p.add_argument("--validate-every-steps", type=int, default=0,
                   help="Run validation every N optimizer updates in "
                        "addition to the end-of-epoch validation.")
    p.add_argument("--save-every-steps", type=int, default=0,
                   help="Save a step_NNNNNN/ crash-safety checkpoint every "
                        "N optimizer updates. Independent of validation -- "
                        "set smaller than --validate-every-steps to get "
                        "finer recovery granularity without paying the "
                        "validation cost. Default 0 means inherit the "
                        "validation cadence (legacy behaviour where save "
                        "and validation were locked together).")
    p.add_argument("--resume", action="store_true",
                   help="If <output-dir>/last/ exists from a previous run, "
                        "load model + optimizer + scheduler + RNG state from "
                        "it and continue training.")

    # Architecture
    p.add_argument("--model-name", default="microsoft/deberta-v3-large",
                   help="HuggingFace model id for the backbone. CLAF "
                        "requires >= 20 layers, so DeBERTa-v3-Large is "
                        "the recommended floor.")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--syntax-range", type=int, nargs=2, default=[5, 10],
                   help="hidden_states slice [start, end) for the syntax "
                        "stream of CLAF. Default [5, 10] (HF's index 0 "
                        "is embeddings, so layer N lives at index N).")
    p.add_argument("--semantic-range", type=int, nargs=2, default=[13, 18],
                   help="hidden_states slice for the semantic stream. "
                        "Default [13, 18].")
    p.add_argument("--discourse-range", type=int, nargs=2, default=[20, 25],
                   help="hidden_states slice for the discourse stream. "
                        "Default [20, 25] = layers 20..24 (5 layers, "
                        "matching syntax/semantic). Note: DeBERTa-v3-Large "
                        "has 24 layers + embedding output = 25 hidden "
                        "states, so [21,26] silently truncates to 4 layers.")

    # Training
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Per-device micro-batch. Combined with "
                        "--gradient-accumulation-steps it sets the effective "
                        "batch size. Default 2 keeps DeBERTa-Large + CLAF + "
                        "4-branch 320ch conv head + BPM + classifier + "
                        "512-token sequences inside 12 GB VRAM with "
                        "checkpointing + LoRA + FGM.")
    p.add_argument("--gradient-accumulation-steps", type=int, default=8,
                   help="Effective batch = batch_size * accum_steps. "
                        "Default 2 * 8 = 16, the doc-recommended value.")
    p.add_argument("--encoder-lr", type=float, default=3e-4,
                   help="LR for backbone (LoRA) parameters. v4 default is "
                        "3e-4 — appropriate for LoRA adapters initialized "
                        "from zero. The legacy 1e-5 was a full-fine-tune "
                        "LR and left LoRA effectively unused.")
    p.add_argument("--llrd-decay", type=float, default=0.9,
                   help="Layer-wise learning rate decay (LLRD) for backbone "
                        "LoRA params. Layer i gets lr = encoder_lr * "
                        "decay^(N-1-i), so deeper (semantic) layers learn "
                        "faster than shallow (syntactic) layers. v4.1 "
                        "default 0.9 = aggressive but stable. Set to 0.0 "
                        "to disable LLRD (single-LR backbone group).")
    p.add_argument("--claf-lr",    type=float, default=5e-4,
                   help="LR for CLAF parameters.")
    p.add_argument("--head-lr",    type=float, default=1e-3,
                   help="LR for conv / dilated conv / BPM / classifier / "
                        "LayerNorm / boundary / aux / CRF.")
    p.add_argument("--weight-decay",      type=float, default=0.01,
                   help="Weight decay for backbone (LoRA) params.")
    p.add_argument("--claf-weight-decay", type=float, default=0.03,
                   help="Weight decay for CLAF params.")
    p.add_argument("--head-weight-decay", type=float, default=0.05,
                   help="Weight decay for conv/BPM/classifier/LN/CRF. "
                        "The doc recommends 0.05 to compensate for "
                        "no-dropout.")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--loss-spike-factor", type=float, default=5.0,
                   help="Loss-spike circuit breaker. After --validate-every-steps "
                        "warm-up, batches whose loss exceeds this multiple of the "
                        "EMA-smoothed running loss are dropped (no backward, no "
                        "optimizer step). Catches the runaway cascade that 8-bit "
                        "Adam's quantized moments otherwise absorb and propagate. "
                        "Default 5.0 -- typical training noise stays well under "
                        "2x EMA so 5x is comfortably above false-positive range.")

    # Legacy v3 flags — kept so old launch commands still parse.
    # NOTE: v3's --lambda-boundary was a DIFFERENT loss (boundary-CE
    # window). v4 reuses --lambda-boundary for the Gaussian-target
    # BCE head defined in the "Boundary head" group below; the v3
    # version of this flag has been removed to avoid an argparse
    # collision. --boundary-weight and --boundary-radius remain as
    # accepted-but-ignored aliases for backward CLI compatibility.
    p.add_argument("--boundary-weight", type=float, default=3.0,
                   help="DEPRECATED (v3 boundary-CE). Ignored in v4.")
    p.add_argument("--boundary-radius", type=int, default=2,
                   help="DEPRECATED (v3 boundary-CE). Ignored in v4.")

    # Focal loss (per-token)
    p.add_argument("--lambda-focal", type=float, default=0.3,
                   dest="lambda_focal",
                   help="Weight of the focal loss term. With v4's "
                        "per-token CRF normalization, 0.3 means focal "
                        "contributes ~30% of the gradient (was ~0.3% in "
                        "v3 due to per-sequence CRF / per-token focal "
                        "scale mismatch).")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   dest="focal_gamma",
                   help="Focal modulation exponent. gamma=2 is standard.")
    p.add_argument("--focal-seam-alpha", type=float, default=0.5,
                   dest="focal_seam_alpha",
                   help="Class-weight for label=1 (AI tokens). With v4's "
                        "rebalanced data the class frequency is roughly "
                        "balanced (25% pure_ai, 25% pure_human, plus "
                        "balanced seam directions), so 0.5 (no class "
                        "re-weighting) is the right default. Setting "
                        "this to 0.75 over-weights pure_ai docs by 3× "
                        "relative to pure_human docs because EVERY "
                        "token in a pure_ai doc is label=1 and gets "
                        "the up-weight.")

    # Boundary head (Gaussian target)
    p.add_argument("--lambda-boundary", type=float, default=0.5,
                   dest="lambda_boundary",
                   help="Weight of the boundary-head BCE term. The "
                        "boundary head outputs a per-position score, "
                        "trained against a Gaussian-shaped target "
                        "centered at the GT seam (sigma=--boundary-sigma).")
    p.add_argument("--boundary-sigma", type=float, default=3.0,
                   help="Gaussian width (in token positions) for the "
                        "boundary supervision target. Default 3 ≈ 1.5 "
                        "words for English DeBERTa.")

    # R-Drop (two stochastic forward passes per micro-batch)
    p.add_argument("--use-rdrop", action="store_true", default=False,
                   help="Enable R-Drop: two forward passes with different "
                        "dropout masks per micro-batch, with symmetric KL "
                        "between emission distributions. Strong implicit "
                        "regularizer; doubles fwd+bwd cost. v4.1 default "
                        "OFF — turn on for the final 'super-max' run.")
    p.add_argument("--no-rdrop", action="store_false", dest="use_rdrop")
    p.add_argument("--lambda-rdrop", type=float, default=0.5,
                   help="Weight of the R-Drop KL term. 0.5 is the canonical "
                        "value from the R-Drop paper; range is typically "
                        "0.1-1.0. Higher values force tighter agreement "
                        "between the two stochastic forward passes.")

    # Auxiliary heads (free supervision from CSV columns)
    p.add_argument("--lambda-data-source", type=float, default=0.05,
                   dest="lambda_data_source",
                   help="Weight of the data_source aux CE (pile vs CC). "
                        "Helps the CC out-of-domain F1 gate.")
    p.add_argument("--lambda-model-family", type=float, default=0.05,
                   dest="lambda_model_family",
                   help="Weight of the model-family aux CE.")
    p.add_argument("--lambda-sample-type", type=float, default=0.05,
                   dest="lambda_sample_type",
                   help="Weight of the sample_type aux CE "
                        "(pure_human / pure_ai / h_then_a / a_then_h).")

    # CRF transition constraint
    p.add_argument("--min-p-1to0", type=float, default=0.0,
                   help="Floor on the CRF P(1->0) transition probability. "
                        "v4 default is 0.0 (disabled): with the rebalanced "
                        "data, the CRF can learn correct transitions from "
                        "the marginal frequencies. The legacy 0.05 floor "
                        "fights with start_transitions on `ai_then_human` "
                        "rows and produces an early-seam bias.")

    # Feature-level FGM (legacy; OFF by default in v4 — replaced by
    # standard backbone dropout, which regularizes ~400M backbone params
    # at zero compute overhead vs FGM's 6× compute on 2K LN params).
    p.add_argument("--use-fgm", action="store_true", default=False,
                   help="Enable feature-level FGM. Off by default in v4. "
                        "FGM perturbs LayerNorm parameters (≈2K params) "
                        "and doubles fwd/bwd cost; v4 uses backbone "
                        "dropout instead (0 compute, much broader "
                        "regularization, eval()-disabled).")
    p.add_argument("--no-fgm", action="store_false", dest="use_fgm")
    p.add_argument("--fgm-target", type=str, default="input_norm",
                   help="Substring used to select FGM-targeted parameters.")
    p.add_argument("--fgm-epsilon", type=float, default=1.0,
                   help="FGM perturbation magnitude.")

    # Weight EMA (v4)
    p.add_argument("--use-ema", action="store_true", default=True,
                   help="Maintain an exponential-moving-average copy of "
                        "trainable parameters. Validation runs against "
                        "the EMA weights, and `best/` is saved with EMA "
                        "weights. The optimizer continues stepping on "
                        "the un-averaged weights.")
    p.add_argument("--no-ema", action="store_false", dest="use_ema")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay. 0.999 ≈ averaging over the last ~1000 "
                        "optimizer steps. For shorter runs (e.g. <5K "
                        "steps), drop to 0.995. For very long runs, can "
                        "go to 0.9995.")
    p.add_argument("--ema-warmup-steps", type=int, default=500,
                   help="Skip EMA updates for the first N optimizer steps. "
                        "Early-training weights are far from the final "
                        "minimum, so a shadow that includes them will lag "
                        "the live weights for thousands of steps. Skipping "
                        "the warmup and re-snapshotting at step N gives "
                        "EMA a clean starting point.")

    # Backbone dropout (v4 — applied at training, auto-disabled in eval)
    p.add_argument("--hidden-dropout", type=float, default=0.1,
                   help="DeBERTa hidden_dropout_prob. Applied during "
                        "training only — eval() disables it, so the "
                        "validator's determinism gate is unaffected.")
    p.add_argument("--attn-dropout", type=float, default=0.1,
                   help="DeBERTa attention_probs_dropout_prob (training-only).")

    # Memory / hardware
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_false", dest="bf16")
    # Gradient checkpointing — OFF by default in v4. With LoRA + 5090 +
    # bf16, checkpointing is unnecessary and adds an extra forward pass
    # per step (1.5-2× slowdown). Enable only if VRAM-constrained.
    p.add_argument("--gradient-checkpointing", action="store_true",
                   default=False,
                   help="Recompute activations during backward to save "
                        "VRAM at the cost of ~1.5x compute. Off by "
                        "default in v4.")
    p.add_argument("--no-gradient-checkpointing", action="store_false",
                   dest="gradient_checkpointing")

    # LoRA
    p.add_argument("--use-lora", action="store_true", default=True)
    p.add_argument("--no-lora", action="store_false", dest="use_lora")
    p.add_argument("--lora-r", type=int, default=32,
                   help="LoRA rank. v4.1 default 32 (was 16 in v3/v4). "
                        "Combined with the expanded --lora-target-modules "
                        "(q,k,v + dense), gives ~9× the LoRA parameter "
                        "count of v3 (~14M trainable adapter params). The "
                        "extra capacity is what lets the model encode "
                        "model-name-specific stylistic subtleties — the "
                        "discriminative signal the >0.97 miners exploit. "
                        "Memory cost: ~50 MB extra on GPU.")
    p.add_argument("--lora-alpha", type=int, default=64,
                   help="LoRA alpha. Convention is α/r ≈ 2.0 for "
                        "stability; default 64 keeps the same scaling "
                        "as v3's α=32, r=16.")
    p.add_argument("--lora-dropout", type=float, default=0.1,
                   help="LoRA dropout. Active only at .train() time — "
                        "PEFT respects eval() so the validator's "
                        "determinism gate is unaffected. v4 default "
                        "0.1 (was 0.0 in v3 under a misapplied 'no "
                        "dropout' rule).")
    p.add_argument("--lora-target-modules", type=str, nargs="+",
                   default=["query_proj", "key_proj", "value_proj", "dense"],
                   help="DeBERTa modules to attach LoRA to (PEFT matches "
                        "by suffix). v4.1 default is q,k,v + every Linear "
                        "named 'dense' in the backbone — that captures "
                        "the attention output projection AND both FFN "
                        "linear layers (intermediate.dense, output.dense). "
                        "~4× the LoRA parameter count of v4, but on the "
                        "5090 the memory headroom and training time both "
                        "tolerate it. To go back to attention-only LoRA: "
                        "--lora-target-modules query_proj key_proj value_proj")

    # 8-bit Adam
    p.add_argument("--use-8bit-adam", action="store_true", default=False,
                   help="Use bitsandbytes' AdamW8bit. Saves ~2-3 GB VRAM.")

    # torch.compile
    p.add_argument("--torch-compile", action="store_true", default=True,
                   help="Compile the SeamDetector forward graph via "
                        "torch.compile. Typical speedup on A100 + bf16 "
                        "+ LoRA: 1.3-1.7x with no quality cost. First "
                        "step adds 60-90s for graph capture; cached "
                        "for all subsequent steps. Default ON. Use "
                        "--no-torch-compile to disable if you hit a "
                        "compile error.")
    p.add_argument("--no-torch-compile", action="store_false",
                   dest="torch_compile")
    p.add_argument("--torch-compile-mode", default="default",
                   choices=("default", "reduce-overhead", "max-autotune"),
                   help="torch.compile mode. Default is 'default', which "
                        "is the only safe choice when --use-fgm is on: "
                        "FGM does TWO forward+backward passes per "
                        "micro-batch and the CUDA-graph buffer reuse in "
                        "'reduce-overhead' overwrites tensors that the "
                        "first backward still needs. Use 'reduce-overhead' "
                        "only with --no-fgm. 'max-autotune' is slower to "
                        "compile but marginally faster at runtime; same "
                        "FGM caveat applies.")

    # Diagnostics
    p.add_argument("--gate-monitor", action="store_true", default=False,
                   help="Print CLAF gate distributions + tau at every "
                        "validation. Useful for verifying CLAF is "
                        "learning position-specific gating.")
    p.add_argument("--gate-monitor-batches", type=int, default=5,
                   help="Number of validation batches sampled for the "
                        "CLAF gate diagnostic.")

    # Logging / checkpointing
    p.add_argument("--output-dir", default="models/seam_detector",
                   help="Where checkpoints + training_args.json get written.")
    p.add_argument("--log-every", type=int, default=10,
                   help="Print loss every N optimizer updates.")

    # Misc
    p.add_argument("--dataloader-workers", type=int, default=0,
                   help="DataLoader num_workers. Keep 0 on Windows unless "
                        "you've set up the multiprocessing spawn context.")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    # NOTE (v4.1): the v3-era guard that auto-zeroed --lora-dropout has
    # been removed. The "no dropout" rule applies to INFERENCE — the
    # validator runs the miner in eval() mode, where PyTorch + PEFT both
    # auto-disable dropout. Training-time dropout (backbone, CLAF MHA,
    # LoRA adapters) is invisible to the determinism gate and provides
    # meaningful regularization, so we leave whatever the user passes.
    return args


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
