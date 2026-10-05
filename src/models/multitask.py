"""Shared-encoder multi-task architecture.

One visual encoder feeds four heads:

    health   -> 1 logit   (P(diseased))
    attrs    -> 3 logits  (black_gill, wssv, yellow_head)  [DISEASE_ATTRS order]
    biomass  -> 3 values  (log_weight, length_cm, cephalothorax_cm)
    domain   -> optional, only for the domain-generalization ablation

The same class serves the single-task experiments: heads that a run does not
supervise simply receive no loss (their masks are zero), so E1/E2/E4 and E6 share
one code path and differ only in configuration. That keeps the single-task vs
multi-task comparison honest -- identical architecture, identical head capacity,
only the loss mixture changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.labels import N_ATTRS

BACKBONES = {
    "convnext_tiny": "convnext_tiny",
    "mobilenetv3_large": "mobilenetv3_large_100",
    # Round-1 revision: cross-dataset backbone sweep (reviewer A7). vit_b_16 is
    # deliberately absent -- its pretrained normalisation is mean=std=0.5 while
    # src/data/transforms.py hard-codes ImageNet statistics.
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "swin_t": "swin_tiny_patch4_window7_224",
}


@dataclass
class ModelConfig:
    backbone: str = "convnext_tiny"
    pretrained: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    head_hidden: int = 256
    enable_health: bool = True
    enable_attrs: bool = True
    enable_biomass: bool = True
    enable_domain: bool = False
    n_domains: int = 2
    biomass_outputs: tuple[str, ...] = ("log_weight", "length_cm", "cephalothorax_cm")
    pooling: str = "mean"           # multi-view aggregation: mean | attention
    mixstyle: bool = False
    mixstyle_p: float = 0.5
    mixstyle_alpha: float = 0.1
    # Fixed-resolution backbones (Swin) must be created at the training size and
    # probed at that size; convolutional ones accept any size and are probed at 64.
    img_size: int | None = None
    backbone_kwargs: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def build_encoder(cfg: ModelConfig) -> tuple[nn.Module, int]:
    """Create a timm backbone with the classifier stripped, and measure its real
    output width.

    `model.num_features` is NOT reliable across families: for MobileNetV3 it
    reports 960 (the width before `conv_head`) while `forward()` with
    num_classes=0 actually emits 1280. ConvNeXt has no such head, so the two agree
    there and the discrepancy hides until a second backbone is added. A single
    dummy forward pass settles it for any architecture.
    """
    if cfg.backbone not in BACKBONES:
        raise KeyError(f"Unknown backbone {cfg.backbone!r}; known: {sorted(BACKBONES)}")
    kw = dict(cfg.backbone_kwargs)
    if cfg.img_size is not None:
        kw["img_size"] = cfg.img_size
    model = timm.create_model(
        BACKBONES[cfg.backbone],
        pretrained=cfg.pretrained,
        num_classes=0,          # strip classifier, keep pooled features
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
        **kw,
    )
    probe = cfg.img_size or 64
    was_training = model.training
    model.eval()
    with torch.no_grad():
        feat_dim = int(model(torch.zeros(1, 3, probe, probe)).shape[-1])
    model.train(was_training)
    return model, feat_dim


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPool(nn.Module):
    """Gated attention pooling over the views of one individual (Ilse et al., 2018).

    Learns which photographs of a shrimp are informative -- e.g. down-weighting
    frames where the body is occluded or curled -- rather than averaging blindly.
    """

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden)
        self.U = nn.Linear(in_dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """feats (B, V, D), mask (B, V) with 1 for real views. Returns (B, D), (B, V)."""
        a = self.w(torch.tanh(self.V(feats)) * torch.sigmoid(self.U(feats))).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(mask <= 0, torch.finfo(a.dtype).min)
        w = torch.softmax(a, dim=1)
        if mask is not None:
            # a row with zero valid views would produce NaN from softmax over -inf
            empty = (mask.sum(dim=1, keepdim=True) <= 0)
            w = torch.where(empty, torch.zeros_like(w), w)
        return torch.einsum("bv,bvd->bd", w, feats), w


def masked_mean_pool(feats: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return feats.mean(dim=1)
    m = mask.unsqueeze(-1).to(feats.dtype)
    return (feats * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


class ShrimpMultiTaskModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder, self.feat_dim = build_encoder(cfg)
        self.mixstyle_names: list[str] = []

        if cfg.mixstyle:
            from src.models.mixstyle import (
                apply_mixstyle_to_convnext, apply_mixstyle_to_mobilenetv3,
            )
            fn = (apply_mixstyle_to_convnext if cfg.backbone == "convnext_tiny"
                  else apply_mixstyle_to_mobilenetv3)
            self.mixstyle_names = fn(self.encoder, p=cfg.mixstyle_p, alpha=cfg.mixstyle_alpha)

        d = self.feat_dim
        self.health_head = MLPHead(d, 1, cfg.head_hidden) if cfg.enable_health else None
        self.attr_head = MLPHead(d, N_ATTRS, cfg.head_hidden) if cfg.enable_attrs else None
        self.biomass_head = (MLPHead(d, len(cfg.biomass_outputs), cfg.head_hidden)
                             if cfg.enable_biomass else None)
        self.domain_head = (MLPHead(d, cfg.n_domains, cfg.head_hidden)
                            if cfg.enable_domain else None)
        self.attn_pool = AttentionPool(d) if cfg.pooling == "attention" else None

    # ---------------------------------------------------------------- #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def heads_from_features(self, feats: torch.Tensor, domain_lambda: float = 0.0) -> dict:
        out: dict = {"features": feats}
        if self.health_head is not None:
            out["health_logit"] = self.health_head(feats).squeeze(-1)
        if self.attr_head is not None:
            out["attr_logits"] = self.attr_head(feats)
        if self.biomass_head is not None:
            out["biomass"] = self.biomass_head(feats)
        if self.domain_head is not None:
            z = GradientReversal.apply(feats, domain_lambda) if domain_lambda else feats
            out["domain_logits"] = self.domain_head(z)
        return out

    def forward(self, x: torch.Tensor, domain_lambda: float = 0.0) -> dict:
        """x (B,3,H,W) -> head outputs."""
        return self.heads_from_features(self.encode(x), domain_lambda=domain_lambda)

    def forward_multiview(self, x: torch.Tensor, view_mask: torch.Tensor | None = None
                          ) -> dict:
        """x (B, V, 3, H, W): encode every view, aggregate, then predict once per
        individual. Returns head outputs plus per-view attention weights."""
        b, v = x.shape[:2]
        feats = self.encode(x.flatten(0, 1)).view(b, v, -1)
        if self.attn_pool is not None:
            pooled, w = self.attn_pool(feats, view_mask)
        else:
            pooled, w = masked_mean_pool(feats, view_mask), None
        out = self.heads_from_features(pooled)
        out["view_weights"] = w
        out["view_features"] = feats
        return out

    # ---------------------------------------------------------------- #
    def set_mixstyle_active(self, flag: bool) -> None:
        for m in self.modules():
            if hasattr(m, "set_active") and m.__class__.__name__ == "MixStyle":
                m.set_active(flag)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> dict:
        self.eval()
        o = self.forward(x)
        res = {}
        if "health_logit" in o:
            res["p_diseased"] = torch.sigmoid(o["health_logit"])
        if "attr_logits" in o:
            res["p_attrs"] = torch.sigmoid(o["attr_logits"])
        if "biomass" in o:
            res["biomass"] = o["biomass"]
        return res


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(train),
            "size_mb_fp32": round(total * 4 / 1e6, 2)}
