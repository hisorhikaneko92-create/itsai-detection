"""
Train the Hybrid Semantic Seam Detector (HSSD) -- v3 / Phase 2.5.

Architecture (per HSSD_Architecture_Document.md, "SOTA: HSSD v3"):
  * DeBERTa-v3-Large backbone (LoRA r=16, output_hidden_states=True)
  * CLAF v2 -- Cross-Layer Attention Fusion with:
      - corrected layer indices: hidden_states[5:10] / [13:18] / [21:26]
      - per-range learnable query + multi-head cross-attention
      - cross-range interaction via Linear(3H -> H) + GELU residual
      - temperature-scaled softmax gate (learnable tau, clamped 0.1..5.0)
  * LayerNorm(1024) input stabilization
  * Multi-Scale DILATED Conv head, 4 branches x 320ch:
      - Conv1d(k=3,  d=1)  trigram
      - Conv1d(k=5,  d=1)  phrase rhythm
      - Conv1d(k=7,  d=1)  clause structure
      - Conv1d(k=3,  d=8)  sentence-level pattern (RF=17)
      - GELU activations throughout, concat -> 1280-dim
  * Boundary Prototype Memory -- 16 learnable seam archetypes:
      Linear(1280 -> 384) + GELU -> normalize -> cosine vs 16 prototypes
      -> 16-dim similarity feature, concatenated to conv features (1296)
  * Two-layer classifier: Linear(1296 -> 384) + GELU + Linear(384 -> 2)
  * CRF (2 labels) with constrained transitions: P(1->0) >= 0.05 enforced
    after each optimizer step

Training strategy:
  * NO DROPOUT anywhere (project constraint -- breaks the validator's
    determinism / batch-consistency gate). lora_dropout=0.0 enforced.
  * Feature-level FGM (replaces word-embedding FGM, which was a silent
    no-op under LoRA). Perturbs the input_norm LayerNorm parameters
    along the gradient direction (epsilon=1.0). The perturbation
    propagates through the conv head + classifier on the adversarial
    forward pass, then is restored before the optimizer step.
  * Two-term loss applied OUTSIDE model.forward (model returns raw
    emissions in train mode):
        L_total = L_CRF  +  lambda * L_boundary
    L_boundary = focused cross-entropy with `boundary_weight` multiplier
    inside a 5-word radius around each ground-truth seam.
  * Three-group differential LR + weight decay:
        backbone (LoRA)         lr=1e-5  wd=0.01
        CLAF (cross-attn+gate)  lr=5e-4  wd=0.03
        head (conv+BPM+cls+LN+CRF) lr=1e-3  wd=0.05
  * LoRA on attention q/v projections (r=16, alpha=32, lora_dropout=0.0).
    `modules_to_save` keeps CLAF / conv / dilated conv / BPM / classifier
    / CRF / LayerNorm fully trainable AND bundled with the adapter on
    save_pretrained.
  * Cosine LR schedule with 10% warmup, BF16 autocast, gradient
    checkpointing, optional 8-bit AdamW (bitsandbytes).

Reads CSVs produced by scripts/build_training_dataset.py:
  columns: text, segmentation_labels (JSON list of 0/1), data_source, ...

Validation metrics:
  * Mean Seam Offset  (target: < 2.5 words)
  * F1 @ 5 Words      (target: > 0.92)
  * Token-Level F1    (sanity check)

Usage:
    pip install pytorch-crf peft
    pip install bitsandbytes        # optional, saves ~2 GB VRAM via 8-bit Adam

    python scripts\\train_seam_detector.py `
        --train-csv data\\train_50k.csv `
        --val-csv   data\\probe_w32_v8.csv `
        --output-dir models\\seam_detector `
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
# CLAF v2 -- Cross-Layer Attention Fusion (fixed + cross-range + temperature)
# ---------------------------------------------------------------------------
class CrossLayerAttentionFusionV2(nn.Module):
    """Phase 2.5 CLAF.

    Differences from the Phase 2 CLAF:
      1. Corrected layer indices. HuggingFace returns
         (num_layers + 1) tensors with index 0 = embedding output, so
         "Layer N" really lives at hidden_states[N]. The doc's intended
         5-layer ranges are encoded as [5:10] / [13:18] / [21:26].
      2. Cross-range interaction layer (Stage 2.5): after per-range
         cross-attention residuals, all three streams are concatenated
         and projected back to hidden_size, providing each stream with
         a compressed summary of the others. This converts "uninformed
         disagreement" into "meaningful disagreement" before gating.
      3. Learnable temperature on the softmax gate, replacing the old
         LayerNorm(3) -> softmax. tau is clamped to [0.1, 5.0]:
            tau < 1 -> sharper (CLAF is confident in level selection)
            tau > 1 -> softer  (CLAF hedges)
         tau is also a useful diagnostic for whether CLAF is doing
         anything position-specific."""

    def __init__(self, hidden_size: int = 1024,
                 syntax_range: Tuple[int, int] = (5, 10),
                 semantic_range: Tuple[int, int] = (13, 18),
                 discourse_range: Tuple[int, int] = (20, 25)):
        super().__init__()
        self.hidden_size = hidden_size
        self.syntax_range = syntax_range
        self.semantic_range = semantic_range
        self.discourse_range = discourse_range

        # Stage 2: per-range learnable queries + cross-attention.
        self.syntax_query    = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.semantic_query  = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.discourse_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.0,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Stage 2.5: cross-range interaction (NEW in v3).
        self.cross_range_proj = nn.Linear(hidden_size * 3, hidden_size)

        # Stage 3: temperature-scaled gating (replaces gate LayerNorm).
        self.gate_proj = nn.Linear(hidden_size * 3, 3)
        self.gate_temperature = nn.Parameter(torch.ones(1))    # tau

    @staticmethod
    def _pool_range(hidden_states, start: int, end: int) -> torch.Tensor:
        """Mean-pool hidden_states[start:end] along the layer axis.
        Python slicing tolerates an out-of-range upper bound, so this
        gracefully handles backbones with fewer hidden states than the
        configured range."""
        sliced = list(hidden_states[start:end])
        if not sliced:
            raise IndexError(
                f"CLAF range [{start}:{end}] produced no hidden states "
                f"(backbone has {len(hidden_states)} entries)"
            )
        stacked = torch.stack(sliced, dim=0)              # [L,B,T,H]
        return stacked.mean(dim=0)                         # [B,T,H]

    def forward(self, all_hidden_states):
        """all_hidden_states: tuple (num_layers + 1) of [B, T, H] tensors.

        Returns:
            fused        [B, T, H]
            gate_weights [B, T, 3]   (kept for diagnostic logging)
        """
        h_syn = self._pool_range(all_hidden_states, *self.syntax_range)
        h_sem = self._pool_range(all_hidden_states, *self.semantic_range)
        h_dis = self._pool_range(all_hidden_states, *self.discourse_range)

        B, T, H = h_syn.shape

        # Per-range cross-attention: each level's learnable query attends
        # to its OWN pooled representation (level-disentangled). The 3*B
        # batched dim shapes the attention into one global vector per
        # level per batch row.
        ctx = torch.cat([h_syn, h_sem, h_dis], dim=0)                       # [3B,T,H]
        q = torch.cat([
            self.syntax_query.expand(B, -1, -1),     # [B,1,H]
            self.semantic_query.expand(B, -1, -1),
            self.discourse_query.expand(B, -1, -1),
        ], dim=0)                                                            # [3B,1,H]

        attended, _ = self.cross_attn(q, ctx, ctx)                          # [3B,1,H]
        attended = self.attn_norm(attended)                                  # [3B,1,H]
        a_syn, a_sem, a_dis = attended.split(B, dim=0)                       # each [B,1,H]

        # Residual: add per-level global context to every position.
        h_syn = h_syn + a_syn                                                # [B,T,H]
        h_sem = h_sem + a_sem
        h_dis = h_dis + a_dis

        # Stage 2.5: cross-range interaction. Each level gets a
        # compressed summary of the other two -- so the gate's
        # disagreement signal becomes meaningful.
        cross_input = torch.cat([h_syn, h_sem, h_dis], dim=-1)               # [B,T,3H]
        cross_context = F.gelu(self.cross_range_proj(cross_input))           # [B,T,H]
        h_syn = h_syn + cross_context
        h_sem = h_sem + cross_context
        h_dis = h_dis + cross_context

        # Stage 3: temperature-scaled gated fusion.
        gate_input = torch.cat([h_syn, h_sem, h_dis], dim=-1)                # [B,T,3H]
        gate_logits = self.gate_proj(gate_input)                             # [B,T,3]
        tau = self.gate_temperature.clamp(min=0.1, max=5.0)
        gate_weights = torch.softmax(gate_logits / tau, dim=-1)              # [B,T,3]

        fused = (
            gate_weights[..., 0:1] * h_syn
            + gate_weights[..., 1:2] * h_sem
            + gate_weights[..., 2:3] * h_dis
        )                                                                     # [B,T,H]
        return fused, gate_weights


# ---------------------------------------------------------------------------
# Boundary Prototype Memory (BPM) -- v3 addition
# ---------------------------------------------------------------------------
class BoundaryPrototypeMemory(nn.Module):
    """16 learnable seam archetype vectors. Each token's conv-features
    are projected to `proto_dim` and compared (cosine similarity)
    against every prototype. The 16 similarity scalars become extra
    classification features alongside the conv outputs.

    Why this helps: convs detect local patterns. BPM adds a metric-
    learning channel that asks "does this position resemble any known
    seam archetype?" -- a complementary detection mechanism."""

    def __init__(self, input_dim: int = 1280, proto_dim: int = 384,
                 num_prototypes: int = 16):
        super().__init__()
        self.proj = nn.Linear(input_dim, proto_dim)
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, proto_dim) * 0.02,
        )
        self.num_prototypes = num_prototypes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, input_dim]  ->  similarities [B, T, num_prototypes]."""
        embedded = F.gelu(self.proj(x))                                       # [B,T,P]
        embedded_n = F.normalize(embedded, dim=-1)
        proto_n = F.normalize(self.prototypes, dim=-1)
        return torch.matmul(embedded_n, proto_n.T)                            # [B,T,K]


# ---------------------------------------------------------------------------
# Architecture -- HSSD v3 (Phase 2.5)
# ---------------------------------------------------------------------------
class SeamDetector(nn.Module):
    """v3 pipeline:
        DeBERTa-v3-Large (output_hidden_states=True)
            -> CLAF v2
            -> LayerNorm(1024)
            -> 4-branch dilated multi-scale Conv (k=3,5,7 d=1; k=3 d=8) at 320ch
            -> BPM (16 prototypes)
            -> Linear(1296 -> 384) + GELU + Linear(384 -> 2)
            -> CRF

    forward(input_ids, attention_mask, labels=None):
        Train (labels given):  emissions tensor [B, T, 2] (raw)
            The two-term loss is computed by compute_total_loss() in
            the training loop, NOT inside the model. This keeps the
            loss flexible (caller can mix in extra terms) and lets
            FGM perturb features and recompute emissions cleanly.
        Eval (labels None):    list[list[int]] of Viterbi-decoded paths
    """

    CONV_CHANNELS = 320
    CONV_OUT_DIM = CONV_CHANNELS * 4    # 1280
    BPM_PROTO_DIM = 384
    BPM_NUM_PROTOS = 16
    CLASSIFIER_INPUT = CONV_OUT_DIM + BPM_NUM_PROTOS    # 1296
    CLASSIFIER_HIDDEN = 384

    def __init__(self, model_name: str = "microsoft/deberta-v3-large",
                 syntax_range: Tuple[int, int] = (5, 10),
                 semantic_range: Tuple[int, int] = (13, 18),
                 discourse_range: Tuple[int, int] = (20, 25)):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.claf = CrossLayerAttentionFusionV2(
            hidden_size=hidden_size,
            syntax_range=syntax_range,
            semantic_range=semantic_range,
            discourse_range=discourse_range,
        )
        self.input_norm = nn.LayerNorm(hidden_size)

        # Multi-scale dilated conv head, 4 branches x 320ch (Phase 2.5).
        self.conv3        = nn.Conv1d(hidden_size, self.CONV_CHANNELS,
                                       kernel_size=3, padding=1)
        self.conv5        = nn.Conv1d(hidden_size, self.CONV_CHANNELS,
                                       kernel_size=5, padding=2)
        self.conv7        = nn.Conv1d(hidden_size, self.CONV_CHANNELS,
                                       kernel_size=7, padding=3)
        self.conv3_dilated = nn.Conv1d(hidden_size, self.CONV_CHANNELS,
                                        kernel_size=3, padding=8, dilation=8)

        # Boundary Prototype Memory.
        self.bpm = BoundaryPrototypeMemory(
            input_dim=self.CONV_OUT_DIM,
            proto_dim=self.BPM_PROTO_DIM,
            num_prototypes=self.BPM_NUM_PROTOS,
        )

        # Two-layer classifier (GELU throughout).
        self.classifier = nn.Sequential(
            nn.Linear(self.CLASSIFIER_INPUT, self.CLASSIFIER_HIDDEN),
            nn.GELU(),
            nn.Linear(self.CLASSIFIER_HIDDEN, 2),
        )

        self.crf = CRF(2, batch_first=True)
        self.num_labels = 2

    def _emissions(self, input_ids: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns raw emissions [B, T, 2]."""
        out = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,    # required for CLAF
        )
        fused, _gate = self.claf(out.hidden_states)             # [B,T,H]
        normed = self.input_norm(fused)                          # [B,T,H]

        x = normed.transpose(1, 2)                               # [B,H,T]
        c3  = F.gelu(self.conv3(x))                              # [B,320,T]
        c5  = F.gelu(self.conv5(x))                              # [B,320,T]
        c7  = F.gelu(self.conv7(x))                              # [B,320,T]
        c3d = F.gelu(self.conv3_dilated(x))                      # [B,320,T]
        cat = torch.cat([c3, c5, c7, c3d], dim=1)                # [B,1280,T]
        cat = cat.transpose(1, 2)                                # [B,T,1280]

        bpm_sims = self.bpm(cat)                                 # [B,T,16]
        feat = torch.cat([cat, bpm_sims], dim=-1)                # [B,T,1296]
        # Bounded emissions in [-15, 15]. Healthy training logits sit
        # well inside this range; the clamp is a hard ceiling that
        # prevents BF16-saturated or FGM-perturbed forwards from
        # producing inf/extreme logits that the CRF's logsumexp would
        # turn into astronomical NLL. Bounded loss → bounded gradient
        # → no single bad batch can blow up the optimizer state.
        return self.classifier(feat).clamp(-15.0, 15.0)          # [B,T,2]

    def compute_emissions(self, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """Public access to raw emission scores. Used by HSSDPredictor's
        emission-aggregation path: gather emissions across overlapping
        windows, average them per token, then run a single global
        Viterbi decode over the full doc. This is more principled than
        per-window Viterbi + OR-voting because the CRF's transition
        matrix sees the whole sequence at decode time, producing a
        single globally-coherent label sequence with no per-window
        inconsistencies on overlap regions."""
        return self._emissions(input_ids, attention_mask)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        emissions = self._emissions(input_ids, attention_mask)
        if labels is None:
            # Inference: Viterbi decode. We SLICE OFF [CLS] (always
            # position 0) before the CRF so the CRF's mandatory
            # "position 0 must be True" requirement applies to the
            # first real content token, not the [CLS] dummy. This
            # removes the silent label-0 bias the CRF would otherwise
            # learn at [CLS]. The backbone still sees [CLS] -- we just
            # don't include its emission in the sequence likelihood.
            #
            # Autocast is disabled around the CRF because pytorch-crf's
            # logsumexp is unstable in bf16 (the cast to fp32 alone
            # isn't enough -- autocast would still downcast inputs to
            # its matmul/exp/log calls).
            em_no_cls = emissions[:, 1:, :]
            attn_no_cls = attention_mask[:, 1:]
            crf_mask = attn_no_cls.bool().clone()
            crf_mask[:, 0] = True
            with torch.amp.autocast(device_type=emissions.device.type,
                                     enabled=False):
                decoded_no_cls = self.crf.decode(
                    em_no_cls.float(), mask=crf_mask,
                )
            # Prepend a dummy label at position 0 ([CLS]) so the output
            # length matches input_ids length. Callers that map
            # predictions back to words use `word_ids()`, where index 0
            # is None ([CLS]) and gets skipped -- so this dummy 0 is
            # never read but keeps the alignment one-to-one.
            return [[0] + path for path in decoded_no_cls]
        # Training: return RAW emissions; caller computes loss.
        return emissions


# ---------------------------------------------------------------------------
# Focal + CRF loss (replaces the boundary-CE family)
# ---------------------------------------------------------------------------
def compute_focal_loss(
    emissions: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    seam_alpha: float = 0.75,
) -> torch.Tensor:
    """Focal loss for seam token detection.

    Why focal instead of boundary CE:
      - No buffer_radius to tune. Focal modulation (1-p)^gamma
        automatically concentrates gradient on hard examples — which
        are exactly the tokens near the boundary — without needing to
        manually define a neighbourhood.
      - Stable scale. CE normalised by total tokens swings wildly when
        seam density varies batch to batch. Focal normalises by valid
        tokens only and is O(1) per position regardless of seam density.
      - Handles class imbalance correctly. seam_alpha=0.75 explicitly
        up-weights the rare label-1 class. Combined with the focal term,
        easy non-seam predictions (p≈0.99) contribute virtually zero
        gradient while hard boundary-adjacent positions drive the update.

    gamma=2.0   : standard focal modulation (Lin et al. 2017)
    seam_alpha  : weight for the rare seam label (label=1).
                  Non-seam tokens receive weight (1 - seam_alpha).
                  0.75 means seam positions get 3× the base weight
                  of non-seam positions before focal modulation.
    """
    B, T, C = emissions.shape
    valid = (labels != -100)

    safe_labels = labels.clone()
    safe_labels[~valid] = 0

    # Stable log-softmax in fp32 (emissions already clamped upstream)
    log_probs = F.log_softmax(emissions.float(), dim=-1)   # [B, T, C]

    # Log probability of the ground-truth class at each position
    log_pt = log_probs.gather(
        dim=-1, index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)                                           # [B, T]
    pt = log_pt.exp()

    # Focal modulation: (1-p)^gamma → near-zero for easy correct preds
    focal_weight = (1.0 - pt).pow(gamma)                   # [B, T]

    # Alpha correction: up-weights the minority seam label
    is_seam = (labels == 1).float()
    alpha_t = seam_alpha * is_seam + (1.0 - seam_alpha) * (1.0 - is_seam)

    # Per-token focal loss; zero at -100 padding positions
    loss_per_token = -(alpha_t * focal_weight * log_pt)     # [B, T]

    valid_f = valid.float()
    return (loss_per_token * valid_f).sum() / valid_f.sum().clamp_min(1.0)


def compute_total_loss(
    emissions: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    crf_module: CRF,
    lambda_focal: float = 0.3,
    focal_gamma: float = 2.0,
    focal_seam_alpha: float = 0.75,
    # Legacy kwargs accepted but ignored — kept so any external callers
    # that pass boundary_weight / lambda_boundary don't raise TypeError.
    **_legacy_kwargs,
) -> torch.Tensor:
    """L_total = L_CRF  +  lambda_focal * L_focal

    L_CRF  (weight=1.0): CRF negative log-likelihood.
           Optimises the JOINT sequence probability through the
           transition matrix. Responsible for global coherence —
           no isolated 1-labels, no runaway multi-seam predictions.

    L_focal (weight=lambda_focal): per-token focal loss.
           Concentrates gradient at the hard positions (near the seam)
           without contradicting the CRF's joint-probability framework,
           because lambda_focal=0.3 keeps it as a secondary signal.
           The CRF loss always dominates; focal merely steers the
           emission head toward boundary precision.

    lambda_focal=0.3: deliberately conservative. The old value of 0.5
        was too large relative to the boundary CE's miscalibrated scale.
        0.3 with correctly-scaled focal loss delivers more boundary
        gradient than 0.5 with the old CE, because focal's
        self-weighting suppresses the flood of easy non-seam tokens
        that were drowning the boundary signal before.
    """
    device_type = emissions.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        emissions_fp32 = emissions.float()

        # Slice off [CLS] — same rationale as the old loss: pytorch-crf
        # requires position 0's mask to be True, so we exclude the
        # [CLS] dummy and let the first CONTENT token take that slot.
        em_no_cls     = emissions_fp32[:, 1:, :]
        labels_no_cls = labels[:, 1:]
        attn_no_cls   = attention_mask[:, 1:]

        crf_mask = attn_no_cls.bool() & (labels_no_cls != -100)
        crf_mask = crf_mask.clone()
        crf_mask[:, 0] = True

        safe_labels = labels_no_cls.clone()
        safe_labels[labels_no_cls == -100] = 0

        crf_loss = -crf_module(
            em_no_cls, safe_labels, mask=crf_mask, reduction="mean",
        )

        focal = compute_focal_loss(
            em_no_cls, labels_no_cls,
            gamma=focal_gamma,
            seam_alpha=focal_seam_alpha,
        )

        return crf_loss + lambda_focal * focal


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
                 seed: int = 0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = max(1, stride)         # kept for back-compat; unused now
        self.min_chunk_words = min_chunk_words

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
        valid_rows: List[Tuple[List[str], List[int]]] = []
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
            valid_rows.append((words, list(map(int, word_labels))))

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
        # Budget = max_length - 2 to reserve room for [CLS] + [SEP] in
        # __getitem__. Every chunk's tokenized length is guaranteed to
        # be <= budget here, so tokenizer truncation never fires.
        budget = max_length - 2
        self.chunks: List[Tuple[List[str], List[int]]] = []
        n_split = 0
        n_dropped_huge_word = 0

        for idx, (words, word_labels) in enumerate(valid_rows):
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
                        self._maybe_emit(words, word_labels, start, i)
                        n_split += 1
                    # Emit the huge word as its own chunk (will be
                    # tokenizer-truncated; rare).
                    self._maybe_emit(words, word_labels, i, i + 1)
                    n_dropped_huge_word += 1
                    start = i + 1
                    cur_tokens = 0
                    split_this_row = True
                    continue

                if cur_tokens + wt > budget and start < i:
                    self._maybe_emit(words, word_labels, start, i)
                    n_split += 1
                    start = i
                    cur_tokens = 0
                    split_this_row = True

                cur_tokens += wt

            # Final chunk
            if start < n_words:
                self._maybe_emit(words, word_labels, start, n_words)

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

        # Build labels in a single loop (using each row's word_ids).
        cached_labels: List[List[int]] = []
        for idx, (_words, word_labels) in enumerate(self.chunks):
            row_word_ids = chunk_encs.word_ids(idx)
            row_labels = [
                -100 if wid is None else int(word_labels[wid])
                for wid in row_word_ids
            ]
            cached_labels.append(row_labels)
        self._cached_labels: torch.Tensor = torch.tensor(
            cached_labels, dtype=torch.long,
        )                                                                       # [N, T]

        cache_gb = (
            self._cached_input_ids.numel() * self._cached_input_ids.element_size()
            + self._cached_attention_mask.numel() * self._cached_attention_mask.element_size()
            + self._cached_labels.numel() * self._cached_labels.element_size()
        ) / 1e9
        print(f"  cache built in {time.time() - t1:.1f}s "
              f"(memory: ~{cache_gb:.2f} GB CPU; shared across DataLoader "
              f"workers via fork copy-on-write)")

        # We keep self.chunks for backward inspection (e.g. logging),
        # but __getitem__ now reads from the tensor cache instead.

    def _maybe_emit(self, words: List[str], labels: List[int],
                    start: int, end: int) -> None:
        """Append (words[start:end], labels[start:end]) to self.chunks
        if it meets the min_chunk_words floor."""
        if end - start < self.min_chunk_words:
            return
        self.chunks.append((words[start:end], labels[start:end]))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # All work was done at __init__. This is now a simple tensor
        # slice -- no tokenizer call, no dictionary construction beyond
        # building the return dict.
        return {
            "input_ids":      self._cached_input_ids[idx],
            "attention_mask": self._cached_attention_mask[idx],
            "labels":         self._cached_labels[idx],
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
    model.eval()
    gt_arrays: List[List[int]] = []
    pred_arrays: List[List[int]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]                  # CPU side, used for masking

            with autocast_ctx:
                paths = model(input_ids, attention_mask)

            # `paths` is a list (length B) of variable-length 0/1 lists.
            for i, path in enumerate(paths):
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
                    aligned_gt.append(g)
                    aligned_pred.append(path[pi])
                    pi += 1
                if aligned_gt:
                    gt_arrays.append(aligned_gt)
                    pred_arrays.append(aligned_pred)

    metrics = compute_seam_metrics(gt_arrays, pred_arrays)
    metrics["token_f1"] = compute_token_f1(gt_arrays, pred_arrays)
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
    """Save MODEL ONLY. With Phase 2.5's `modules_to_save=[claf,
    input_norm, conv*, conv3_dilated, bpm, classifier, crf]` the LoRA
    adapter directory bundles BOTH the LoRA delta weights AND the full
    state of those head modules, so a single save_pretrained call
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
                        progress: Dict) -> None:
    """Save EVERYTHING needed to bit-exactly resume training: model,
    optimizer state, scheduler state, RNG states, and progress counters."""
    save_checkpoint(model, save_dir, args)

    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, save_dir / "optimizer.pth")

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
                        use_lora: bool) -> Tuple[nn.Module, Dict]:
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

    state_path = load_dir / "training_state.json"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            return model, json.load(f)
    print(f"WARN: no training_state.json in {load_dir}; using defaults")
    return model, {}


# ---------------------------------------------------------------------------
# Optimizer parameter grouping
# ---------------------------------------------------------------------------
def build_param_groups(model: nn.Module, args: argparse.Namespace) -> List[Dict]:
    """Three groups, matching the HSSD doc:
        backbone (LoRA / encoder)        slow LR, low weight decay
        CLAF (cross-attn + gate + xrng)  intermediate LR, moderate decay
        head (conv + dilated + BPM + classifier + LayerNorm + CRF)
                                         fast LR, high weight decay
    """
    backbone_params, claf_params, head_params = [], [], []
    seen_ids = set()

    def _classify(name: str) -> str:
        n = name.lower()
        if "lora_" in n:
            return "backbone"
        if "claf" in n:
            return "claf"
        # Anything else trainable is a head module.
        return "head"

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        bucket = _classify(name)
        if bucket == "backbone":
            backbone_params.append(param)
        elif bucket == "claf":
            claf_params.append(param)
        else:
            head_params.append(param)

    print(f"Param groups -- backbone(LoRA): "
          f"{sum(p.numel() for p in backbone_params):,} ; "
          f"CLAF: {sum(p.numel() for p in claf_params):,} ; "
          f"head: {sum(p.numel() for p in head_params):,}")

    groups: List[Dict] = []
    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": args.encoder_lr,
            "weight_decay": args.weight_decay,
        })
    if claf_params:
        groups.append({
            "params": claf_params,
            "lr": args.claf_lr,
            "weight_decay": args.claf_weight_decay,
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": args.head_lr,
            "weight_decay": args.head_weight_decay,
        })
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
    # all v3 head components so save_pretrained captures them.
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["query_proj", "value_proj"],
            lora_dropout=args.lora_dropout,                # default 0.0
            bias="none",
            modules_to_save=[
                "claf", "input_norm",
                "conv3", "conv5", "conv7", "conv3_dilated",
                "bpm",
                "classifier", "crf",
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
            print(f"torch.compile: targeting SeamDetector._emissions "
                  f"(shape-stable hot path). Adds ~60-90s to first step.")
            compile_mode = args.torch_compile_mode

            # Compile only the _emissions method -- not the full forward.
            # _emissions is the heavy compute (DeBERTa + CLAF + conv +
            # BPM + classifier) and has fixed input/output shapes
            # thanks to padding="max_length". The branchy CRF code in
            # forward (which calls .item() during decode) stays in
            # eager mode -- compile can't fuse Python control flow
            # anyway, and trying to trace it just causes recompile
            # churn from the variable-length list comprehension on
            # the inference path.
            #
            # Walk through any PEFT wrapping to find the actual
            # SeamDetector instance whose method we want to wrap.
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
                # Replace the bound method with a compiled version.
                # This compiles the heavy compute path WITHOUT trying
                # to trace the CRF decode branch.
                seam_det._emissions = torch.compile(
                    seam_det._emissions,
                    mode=compile_mode,
                    dynamic=False,
                )
                print(f"torch.compile: enabled on SeamDetector._emissions "
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

    # Feature-level FGM regularizer (replaces dropout)
    fgm = (FeatureFGM(model, target_substring=args.fgm_target,
                      epsilon=args.fgm_epsilon)
           if args.use_fgm else None)
    if fgm:
        print(f"FGM enabled  (target='{args.fgm_target}', "
              f"epsilon={args.fgm_epsilon})  -- adds 1 forward+backward "
              f"per micro-batch")

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
            )
            if fgm:
                fgm = FeatureFGM(model, target_substring=args.fgm_target,
                                 epsilon=args.fgm_epsilon)
            start_epoch = int(progress.get("epoch_completed", 0))
            global_update = int(progress.get("global_update", 0))
            best_seam_offset = float(progress.get("best_seam_offset", float("inf")))
            best_f1_at_5 = float(progress.get("best_f1_at_5", -1.0))
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

    # Helper: compute the two-term loss given a fresh emissions tensor.
    crf_module = _resolve_crf(model)

    def _two_term_loss(emissions: torch.Tensor,
                       labels: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        return compute_total_loss(
            emissions, labels, attention_mask, crf_module,
            lambda_focal=args.lambda_focal,
            focal_gamma=args.focal_gamma,
            focal_seam_alpha=args.focal_seam_alpha,
        )

    def _do_validation(epoch_idx: int, step_within_epoch: int,
                       end_of_epoch: bool) -> bool:
        """Returns True if early-stop was triggered."""
        nonlocal best_seam_offset, best_f1_at_5, epochs_since_improvement
        if val_loader is None:
            return False

        model.eval()
        metrics = evaluate(model, val_loader, device, autocast_ctx)
        if args.gate_monitor:
            log_gate_statistics(model, val_loader, device,
                                num_batches=args.gate_monitor_batches)
        model.train()
        cur_offset = metrics["mean_seam_offset"]
        cur_f1 = metrics["f1_at_5"]
        print(f"  [val @ epoch {epoch_idx+1} step {global_update}] "
              f"mean_seam_offset={cur_offset:.2f}  "
              f"f1_at_5={cur_f1:.4f}  "
              f"token_f1={metrics['token_f1']:.4f}  "
              f"({metrics['n_with_seam']}/{metrics['n_total']} had a seam)")

        # Collapse guard: when training implodes, the model predicts
        # "no seam" almost everywhere, n_with_seam drops to ~0, and
        # mean_seam_offset becomes the average over a handful of lucky
        # docs -- numerically lower than any healthy run. Without this
        # guard the script overwrites the healthy `best/` with the
        # collapsed model. We require that:
        #   1. f1_at_5 has not dropped catastrophically (>=0.5x best),
        #   2. seam coverage is at least 25% of the validation set
        #      (real seams typically appear in 50-60% of val docs).
        # Both are loose floors; they fire only on actual collapse.
        coverage = metrics["n_with_seam"] / max(1, metrics["n_total"])
        f1_floor = 0.5 * max(best_f1_at_5, 0.0)
        collapse_suspected = (coverage < 0.25) or (
            best_f1_at_5 > 0 and cur_f1 < f1_floor
        )
        improved = (
            (not math.isnan(cur_offset))
            and cur_offset < best_seam_offset
            and not collapse_suspected
        )
        if collapse_suspected and not math.isnan(cur_offset) and cur_offset < best_seam_offset:
            print(f"  COLLAPSE GUARD tripped: refusing to save best "
                  f"(coverage={coverage:.1%}, cur_f1={cur_f1:.3f} vs "
                  f"f1_floor={f1_floor:.3f}). Model has likely diverged; "
                  f"the seam_offset improvement is an artifact of empty "
                  f"predictions, not a real gain.")
        if improved:
            best_seam_offset = cur_offset
            best_f1_at_5 = cur_f1
            save_training_state(
                model, optimizer, scheduler, out_dir / "best", args,
                progress={
                    "epoch_completed":          epoch_idx,
                    "step_within_epoch":        step_within_epoch,
                    "global_update":            global_update,
                    "best_seam_offset":         best_seam_offset,
                    "best_f1_at_5":             best_f1_at_5,
                    "epochs_since_improvement": 0,
                    "saved_at":                 "validation_improved",
                },
            )
            print(f"  saved new best "
                  f"(seam_offset={best_seam_offset:.2f}, f1@5={best_f1_at_5:.4f}) "
                  f"to {out_dir / 'best'}")

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
        # the save history in training order.
        step_dir = out_dir / f"step_{global_update:06d}"
        save_training_state(
            model, optimizer, scheduler, step_dir, args,
            progress={
                "epoch_completed":          epoch_idx,
                "step_within_epoch":        step_within_epoch,
                "global_update":            global_update,
                "best_seam_offset":         best_seam_offset,
                "best_f1_at_5":             best_f1_at_5,
                "epochs_since_improvement": epochs_since_improvement,
                "saved_at":                 reason,
            },
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

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # ---- Standard forward + backward --------------------------
            with autocast_ctx:
                emissions = model(input_ids, attention_mask, labels=labels)
                loss = _two_term_loss(emissions, labels, attention_mask)

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
                    emissions_adv = model(input_ids, attention_mask, labels=labels)
                    loss_adv = _two_term_loss(emissions_adv, labels, attention_mask)
                if torch.isfinite(loss_adv):
                    (loss_adv / args.gradient_accumulation_steps).backward()
                    adv_loss_value = loss_adv.item()
                else:
                    print(f"  WARN: non-finite FGM adversarial loss at "
                          f"step {step+1}; skipping adversarial backward")
                fgm.restore()

            running_loss += loss.item() + adv_loss_value
            running_n += 1 + (1 if fgm is not None else 0)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # Enforce P(1->0) >= min_prob in the CRF transition matrix.
                # Done after every optimizer step (cost: 2 tensor ops).
                constrain_crf_transitions(model, min_prob=args.min_p_1to0)
                global_update += 1

                if global_update % args.log_every == 0:
                    avg_loss = running_loss / max(1, running_n)
                    lrs = [g["lr"] for g in optimizer.param_groups]
                    lr_str = "/".join(f"{lr:.2e}" for lr in lrs)
                    print(f"  step {global_update:5d}/{total_update_steps}  "
                          f"loss={avg_loss:.4f}  lr={lr_str}")
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
            constrain_crf_transitions(model, min_prob=args.min_p_1to0)
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
                "best_seam_offset":         best_seam_offset,
                "best_f1_at_5":             best_f1_at_5,
                "epochs_since_improvement": epochs_since_improvement,
                "saved_at":                 "end_of_epoch",
            },
        )

    print("\nTraining complete.")
    if best_seam_offset != float("inf"):
        print(f"Best val mean_seam_offset: {best_seam_offset:.2f}  "
              f"(f1@5={best_f1_at_5:.4f})  saved to {out_dir / 'best'}")
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
    p.add_argument("--encoder-lr", type=float, default=1e-5,
                   help="LR for backbone (LoRA) parameters.")
    p.add_argument("--claf-lr",    type=float, default=5e-4,
                   help="LR for CLAF parameters.")
    p.add_argument("--head-lr",    type=float, default=1e-3,
                   help="LR for conv / dilated conv / BPM / classifier / "
                        "LayerNorm / CRF.")
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

    # Loss (boundary-aware, two-term) -- DEPRECATED.
    # These flags are kept so old launch commands still parse, but the
    # active loss path is focal + CRF (see --lambda-focal etc. below).
    # The values here are not read by the training loop.
    p.add_argument("--boundary-weight", type=float, default=3.0,
                   help="DEPRECATED. Replaced by focal loss; ignored.")
    p.add_argument("--boundary-radius", type=int, default=2,
                   help="DEPRECATED. Replaced by focal loss; ignored.")
    p.add_argument("--lambda-boundary", dest="lambda_boundary",
                   type=float, default=0.5,
                   help="DEPRECATED. Replaced by focal loss; ignored.")

    # Focal loss (replaces boundary CE)
    p.add_argument("--lambda-focal", type=float, default=0.3,
                   dest="lambda_focal",
                   help="Weight of the focal loss term. 0.3 keeps CRF "
                        "dominant while focal steers the emission head "
                        "toward boundary precision.")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   dest="focal_gamma",
                   help="Focal modulation exponent. gamma=2 is standard. "
                        "Higher values suppress easy examples more "
                        "aggressively.")
    p.add_argument("--focal-seam-alpha", type=float, default=0.75,
                   dest="focal_seam_alpha",
                   help="Base weight for the rare seam label (label=1). "
                        "Non-seam tokens receive (1 - alpha). 0.75 means "
                        "seam positions start with 3x the gradient weight "
                        "of non-seam before focal modulation is applied.")

    # CRF transition constraint
    p.add_argument("--min-p-1to0", type=float, default=0.05,
                   help="Floor on the CRF P(1->0) transition probability. "
                        "Prevents the CRF from suppressing multi-seam "
                        "predictions. Set to 0 to disable. Default 0.05.")

    # Feature-level FGM adversarial training (replaces dropout)
    p.add_argument("--use-fgm", action="store_true", default=True,
                   help="Enable feature-level FGM (default target = "
                        "input_norm). Doubles forward+backward cost but "
                        "is the doc's primary regularizer in place of "
                        "dropout.")
    p.add_argument("--no-fgm", action="store_false", dest="use_fgm")
    p.add_argument("--fgm-target", type=str, default="input_norm",
                   help="Substring used to select FGM-targeted "
                        "parameters. Default 'input_norm' targets the "
                        "LayerNorm between CLAF and the conv head.")
    p.add_argument("--fgm-epsilon", type=float, default=1.0,
                   help="FGM perturbation magnitude.")

    # Memory / hardware
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_false", dest="bf16")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", action="store_false",
                   dest="gradient_checkpointing")

    # LoRA
    p.add_argument("--use-lora", action="store_true", default=True)
    p.add_argument("--no-lora", action="store_false", dest="use_lora")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0,
                   help="Doc rule: NO DROPOUT. Setting this above 0 will "
                        "violate the SN32 validator's determinism gate "
                        "and is not recommended. Default 0.0.")

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

    if args.lora_dropout and args.lora_dropout > 0.0:
        print(f"WARNING: --lora-dropout={args.lora_dropout} > 0. The HSSD "
              f"doc forbids dropout (it breaks the validator's "
              f"determinism gate). Setting it to 0.0.")
        args.lora_dropout = 0.0
    return args


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
