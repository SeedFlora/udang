"""Training loop shared by every experiment (single-task and multi-task).

A single loop serves E1-E9. The only thing that changes between experiments is
which tasks are enabled and how their losses are weighted -- that is what makes
"single-task vs multi-task" a controlled comparison rather than two different
codebases producing two different numbers.

Multi-task batching uses a DETERMINISTIC interleave schedule: for each epoch the
per-task step counts are fixed up front and the visiting order is generated from
a seeded permutation, so a run is reproducible and the task mixture is auditable.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.losses import (
    FixedTaskWeighting, HomoscedasticUncertaintyWeighting, coral_loss,
    health_attribute_consistency, masked_asymmetric_loss, masked_bce_with_logits,
    masked_focal_with_logits, masked_huber,
)

LOSS_FNS = {
    "bce": masked_bce_with_logits,
    "focal": masked_focal_with_logits,
    "asl": masked_asymmetric_loss,
}


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    min_lr_factor: float = 0.01
    grad_accum: int = 1
    amp: bool = True
    early_stop_patience: int = 12
    # Minimum epochs before early stopping may fire. The validation score is a
    # composite of balanced accuracy and macro-F1 and is genuinely noisy during
    # warm-up, so without a floor a run can latch onto an early lucky epoch and
    # terminate at ~epoch 6 of 14 while a sibling fold trains to convergence.
    # Observed directly: E1 fold 0 reached val 0.903 at epoch 13 while fold 1
    # stopped at epoch 6 having peaked at 0.779 on epoch 1.
    min_epochs: int = 0
    clip_grad_norm: float = 5.0
    attr_loss: str = "bce"                 # bce | focal | asl
    lambda_attribute: float = 1.0
    lambda_consistency: float = 0.1
    lambda_biomass: float = 1.0
    biomass_component_weights: tuple[float, float, float] = (1.0, 0.3, 0.3)
    huber_delta: float = 1.0
    task_weighting: str = "fixed"          # fixed | uncertainty
    fixed_task_weights: dict = field(default_factory=lambda: {"disease": 1.0, "biomass": 1.0})
    coral_lambda: float = 0.0
    domain_lambda: float = 0.0
    monitor: str = "val_primary"
    monitor_mode: str = "max"
    log_every: int = 20
    seed: int = 42
    # Best-checkpoints are only ever reloaded for test inference (weights) --
    # nothing resumes mid-run, the resume guard skips whole folds -- so the AdamW
    # moments (2x the model) are dead weight: 78 runs cost 23 GB on a disk with
    # 35 GB free before the revision grid of ~230 runs. Off by default.
    save_optimizer_state: bool = False


def cosine_warmup(step: int, total: int, warmup: int, min_factor: float) -> float:
    if total <= 0:
        return 1.0
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


class MultiTaskTrainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig, device: torch.device,
                 out_dir: Path, run_name: str):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name

        if cfg.task_weighting == "uncertainty":
            self.weighter = HomoscedasticUncertaintyWeighting(["disease", "biomass"]).to(device)
        else:
            self.weighter = FixedTaskWeighting(cfg.fixed_task_weights).to(device)

        params = list(self.model.parameters()) + list(self.weighter.parameters())
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
        self.history: list[dict] = []
        self.best_metric = -float("inf") if cfg.monitor_mode == "max" else float("inf")
        self.best_epoch = -1
        self.epochs_without_improve = 0

    # ------------------------------------------------------------------ #
    def disease_loss(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        health = batch["health"].to(self.device).float()
        attrs = batch["attrs"].to(self.device).float()
        amask = batch["attr_mask"].to(self.device).float()

        hl = out["health_logit"]
        l_health = nn.functional.binary_cross_entropy_with_logits(hl, health)
        fn = LOSS_FNS[self.cfg.attr_loss]
        l_attr = fn(out["attr_logits"], attrs, amask)
        l_cons = health_attribute_consistency(hl, out["attr_logits"], amask)

        total = (l_health
                 + self.cfg.lambda_attribute * l_attr
                 + self.cfg.lambda_consistency * l_cons)
        # The component dict is logged, never back-propagated: detach before
        # converting, or every step warns about reading a graph tensor as a scalar.
        return total, {"health": float(l_health.detach()), "attr": float(l_attr.detach()),
                       "consistency": float(l_cons.detach())}

    def biomass_loss(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        t = batch["biomass"].to(self.device).float()
        m = batch["biomass_mask"].to(self.device).float()
        w = torch.tensor(self.cfg.biomass_component_weights, device=self.device).float()
        pred = out["biomass"]
        per = []
        for k in range(t.shape[1]):
            per.append(masked_huber(pred[:, k:k + 1], t[:, k:k + 1], m[:, k:k + 1],
                                    delta=self.cfg.huber_delta))
        stacked = torch.stack(per)
        total = (stacked * w[: len(per)]).sum()
        return total, {"log_weight": float(per[0].detach()), "length": float(per[1].detach()),
                       "ceph": float(per[2].detach()) if len(per) > 2 else 0.0}

    # ------------------------------------------------------------------ #
    def _schedule(self, loaders: dict[str, DataLoader], epoch: int) -> list[str]:
        """Deterministic interleave of task batches for one epoch."""
        counts = {k: len(v) for k, v in loaders.items() if v is not None}
        order: list[str] = []
        for k, n in counts.items():
            order += [k] * n
        rng = np.random.default_rng(self.cfg.seed * 1000 + epoch)
        return list(rng.permutation(np.array(order, dtype=object)))

    def train_epoch(self, loaders: dict[str, DataLoader], epoch: int,
                    total_epochs: int) -> dict:
        self.model.train()
        iters = {k: iter(v) for k, v in loaders.items() if v is not None}
        order = self._schedule(loaders, epoch)
        n_steps = len(order)

        agg: dict[str, list[float]] = {}
        gnorms: dict[str, list[float]] = {}
        self.opt.zero_grad(set_to_none=True)
        t0 = time.time()

        for step, task in enumerate(order):
            try:
                batch = next(iters[task])
            except StopIteration:
                continue

            f = cosine_warmup(epoch * n_steps + step, total_epochs * n_steps,
                              self.cfg.warmup_epochs * n_steps, self.cfg.min_lr_factor)
            for g in self.opt.param_groups:
                g["lr"] = self.cfg.lr * f

            with torch.amp.autocast("cuda", enabled=self.cfg.amp and self.device.type == "cuda"):
                if task == "disease":
                    out = self.model(batch["image"].to(self.device, non_blocking=True))
                    loss, parts = self.disease_loss(out, batch)
                elif task == "biomass":
                    if "views" in batch:
                        v = batch["views"].to(self.device, non_blocking=True)
                        out = self.model.forward_multiview(v, batch["view_mask"].to(self.device))
                    else:
                        out = self.model(batch["image"].to(self.device, non_blocking=True))
                    loss, parts = self.biomass_loss(out, batch)
                else:
                    continue

                if self.cfg.coral_lambda > 0 and "features" in out and "domain" in batch:
                    d = batch["domain"].to(self.device)
                    uq = torch.unique(d)
                    if len(uq) >= 2:
                        a = out["features"][d == uq[0]]
                        b = out["features"][d == uq[1]]
                        loss = loss + self.cfg.coral_lambda * coral_loss(a, b)

                total, weights = self.weighter({task: loss})
                total = total / self.cfg.grad_accum

            self.scaler.scale(total).backward()

            if (step + 1) % self.cfg.grad_accum == 0 or step == n_steps - 1:
                self.scaler.unscale_(self.opt)
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    self.cfg.clip_grad_norm)
                gnorms.setdefault(task, []).append(float(gn))
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)

            agg.setdefault(f"{task}_loss", []).append(float(loss))
            for k, v in parts.items():
                agg.setdefault(f"{task}_{k}", []).append(v)

        res = {k: float(np.mean(v)) for k, v in agg.items() if v}
        for k, v in gnorms.items():
            res[f"grad_norm_{k}"] = float(np.mean(v))
        res["epoch"] = epoch
        res["lr"] = self.opt.param_groups[0]["lr"]
        res["train_sec"] = round(time.time() - t0, 1)
        res["n_steps"] = n_steps
        return res

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, epoch: int, extra: dict | None = None) -> Path:
        p = self.out_dir / f"{self.run_name}_best.pt"
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "weighter_state": self.weighter.state_dict(),
            "best_metric": self.best_metric,
            "config": self.cfg.__dict__,
            "extra": extra or {},
        }
        if self.cfg.save_optimizer_state:
            payload["optimizer_state"] = self.opt.state_dict()
            payload["scaler_state"] = self.scaler.state_dict()
        torch.save(payload, p)
        return p

    def load_checkpoint(self, path: Path, strict: bool = True) -> dict:
        # weights_only=True: checkpoints written by save_checkpoint contain only
        # tensors and plain containers, so the unpickling escape hatch is not
        # needed and is left off deliberately.
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ck["model_state"], strict=strict)
        if "optimizer_state" in ck:
            try:
                self.opt.load_state_dict(ck["optimizer_state"])
            except Exception:  # noqa: BLE001
                pass
        self.best_metric = ck.get("best_metric", self.best_metric)
        return ck

    def update_best(self, epoch: int, metric: float, extra: dict | None = None) -> bool:
        better = (metric > self.best_metric) if self.cfg.monitor_mode == "max" else (metric < self.best_metric)
        if better and np.isfinite(metric):
            self.best_metric = metric
            self.best_epoch = epoch
            self.epochs_without_improve = 0
            self.save_checkpoint(epoch, extra)
            return True
        self.epochs_without_improve += 1
        return False

    def should_stop(self, epoch: int = 10 ** 9) -> bool:
        """Early stopping, floored at cfg.min_epochs completed epochs."""
        if epoch + 1 < self.cfg.min_epochs:
            return False
        return self.epochs_without_improve >= self.cfg.early_stop_patience

    def save_history(self) -> Path:
        p = self.out_dir / f"{self.run_name}_history.json"
        p.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        return p
