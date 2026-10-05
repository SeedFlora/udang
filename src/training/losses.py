"""Masked multi-task losses for ShrimpVision-MTL.

Every objective in this project has to cope with *partial supervision*: a batch
mixes images from datasets that annotate different label subsets, so an entry may
be supervised (mask 1.0) or UNKNOWN (mask 0.0). Unknown entries must contribute
exactly zero loss and zero gradient -- they are emphatically **not** negatives.

Two invariants are load-bearing and are enforced by the self-test at the bottom
of this module:

1. A loss evaluated with an all-zero mask returns a finite ``0.0`` tensor, never
   ``NaN``. Batches legitimately contain no supervised examples for some labels
   (e.g. ``cephalothorax_cm`` is absent for the whole of DB1).
2. The *value* stored at a masked-out position is irrelevant -- on **both** sides.
   Predictions and targets alike are sanitised to zero wherever the mask is zero,
   before any arithmetic happens. Sentinel targets (``NaN``, ``-1``) written by
   the dataloader into unlabelled slots therefore cannot poison the forward pass,
   and neither can a non-finite *prediction* in a column nobody supervises: the
   reduction multiplies by the mask, and ``inf * 0`` and ``nan * 0`` are both
   ``NaN``, so sanitising only the targets would leave an unsupervised column
   able to destroy the entire loss.

Non-finite predictions at *supervised* positions are left untouched, so a model
that has genuinely diverged still reports ``NaN`` instead of being masked over.

Conventions used throughout (see ``src.data.labels``):

* Health head: single logit per sample, positive class = DISEASED.
* Attribute head: logits ``(B, 3)`` in ``DISEASE_ATTRS`` order
  ``("black_gill", "wssv", "yellow_head")``.
* Biomass head: ``(B, 3)`` = ``[log_weight, length_cm, cephalothorax_cm]``.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "masked_bce_with_logits",
    "masked_focal_with_logits",
    "masked_asymmetric_loss",
    "masked_huber",
    "health_attribute_consistency",
    "HomoscedasticUncertaintyWeighting",
    "FixedTaskWeighting",
    "coral_loss",
]

_VALID_REDUCTIONS = ("mean", "sum")


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _check_shapes(name: str, *tensors: Tensor) -> None:
    """Raise if the supplied tensors do not all share the same shape.

    Parameters
    ----------
    name : str
        Name of the calling loss, used in the error message.
    *tensors : torch.Tensor
        Tensors whose shapes must agree.

    Raises
    ------
    ValueError
        If any two tensors disagree in shape.
    """
    ref = tensors[0].shape
    for t in tensors[1:]:
        if t.shape != ref:
            shapes = [tuple(x.shape) for x in tensors]
            raise ValueError(f"{name}: all inputs must share a shape, got {shapes}")


def _prepare(
    name: str,
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate shapes and neutralise *both* sides at unsupervised positions.

    Predictions **and** targets are replaced by ``0`` wherever ``mask == 0``. The
    target side guards against sentinel placeholders (``NaN``, ``-1``) written by
    the dataloader into unlabelled slots. The prediction side guards against a
    head that has emitted a non-finite value in a column nobody supervises: the
    reduction multiplies by the mask, and ``inf * 0`` and ``nan * 0`` are both
    ``NaN``, so an unsupervised column would otherwise destroy the whole loss.
    That is not hypothetical -- ``cephalothorax_cm`` is unsupervised for every
    DB1 sample, and an unbounded regression head can overflow there while the
    supervised columns are still perfectly healthy.

    ``torch.where`` is used rather than a multiply because its backward pass
    *selects* rather than scales: masked-out entries receive an exact ``0``
    gradient instead of ``0 * inf == NaN``.

    Parameters
    ----------
    name : str
        Name of the calling loss, used in error messages.
    pred : torch.Tensor
        Predictions (logits or regression outputs), shape ``(B, K)``.
    target : torch.Tensor
        Targets, shape ``(B, K)``. Values where ``mask == 0`` are discarded.
    mask : torch.Tensor
        Supervision mask, shape ``(B, K)``; ``1.0`` supervised, ``0.0`` unknown.

    Returns
    -------
    pred_safe : torch.Tensor
        Prediction with every unsupervised entry replaced by ``0``.
    target_safe : torch.Tensor
        Target with every unsupervised entry replaced by ``0``.
    mask_f : torch.Tensor
        Mask cast to ``pred.dtype``.

    Notes
    -----
    Non-finite predictions at *supervised* positions are deliberately left alone,
    so a genuinely diverged model still surfaces as a ``NaN`` loss rather than
    being silently swept under the rug.
    """
    _check_shapes(name, pred, target, mask)
    mask_f = mask.to(dtype=pred.dtype)
    keep = mask_f > 0
    zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
    pred_safe = torch.where(keep, pred, zero)
    target_safe = torch.where(keep, target.to(dtype=pred.dtype), zero)
    return pred_safe, target_safe, mask_f


def _reduce(per_elem: Tensor, mask: Tensor, reduction: str) -> Tensor:
    """Apply the supervision mask and reduce.

    Parameters
    ----------
    per_elem : torch.Tensor
        Elementwise loss, shape ``(B, K)``.
    mask : torch.Tensor
        Supervision mask, same shape, already cast to the loss dtype.
    reduction : {"mean", "sum"}
        ``"mean"`` divides by ``clamp(mask.sum(), min=1.0)`` so an all-zero mask
        yields a finite ``0.0``; ``"sum"`` returns the masked sum.

    Returns
    -------
    torch.Tensor
        Scalar loss.

    Raises
    ------
    ValueError
        If ``reduction`` is not one of ``"mean"`` or ``"sum"``.
    """
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(f"reduction must be one of {_VALID_REDUCTIONS}, got {reduction!r}")
    masked = per_elem * mask
    total = masked.sum()
    if reduction == "sum":
        return total
    return total / mask.sum().clamp(min=1.0)


# --------------------------------------------------------------------------- #
# Classification losses
# --------------------------------------------------------------------------- #
def masked_bce_with_logits(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Binary cross-entropy over supervised entries only.

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits, shape ``(B, K)``.
    targets : torch.Tensor
        Binary targets in ``{0, 1}``, shape ``(B, K)``. Values at positions where
        ``mask == 0`` are ignored entirely.
    mask : torch.Tensor
        Supervision mask, shape ``(B, K)``; ``1.0`` supervised, ``0.0`` unknown.
    reduction : {"mean", "sum"}, optional
        ``"mean"`` normalises by the number of supervised entries (clamped to a
        minimum of 1.0), ``"sum"`` returns the masked sum. Default ``"mean"``.

    Returns
    -------
    torch.Tensor
        Scalar loss. Exactly ``0.0`` when ``mask`` is all zeros.

    Notes
    -----
    With an all-ones mask and ``reduction="mean"`` this is numerically identical
    to :func:`torch.nn.functional.binary_cross_entropy_with_logits`.
    """
    lgt, tgt, mask_f = _prepare("masked_bce_with_logits", logits, targets, mask)
    per_elem = F.binary_cross_entropy_with_logits(lgt, tgt, reduction="none")
    return _reduce(per_elem, mask_f, reduction)


def masked_focal_with_logits(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    gamma: float = 2.0,
    alpha: float | None = 0.25,
    reduction: str = "mean",
) -> Tensor:
    """Focal loss (Lin et al., 2017) restricted to supervised entries.

    The elementwise objective is ``alpha_t * (1 - p_t) ** gamma * CE`` where
    ``p_t`` is the predicted probability of the ground-truth class and
    ``alpha_t = alpha`` for positives, ``1 - alpha`` for negatives.

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits, shape ``(B, K)``.
    targets : torch.Tensor
        Binary targets in ``{0, 1}``, shape ``(B, K)``.
    mask : torch.Tensor
        Supervision mask, shape ``(B, K)``.
    gamma : float, optional
        Focusing parameter; ``0`` recovers (alpha-weighted) BCE. Default ``2.0``.
    alpha : float or None, optional
        Positive-class weight in ``[0, 1]``. ``None`` disables alpha balancing.
        Default ``0.25``.
    reduction : {"mean", "sum"}, optional
        See :func:`masked_bce_with_logits`. Default ``"mean"``.

    Returns
    -------
    torch.Tensor
        Scalar loss. Exactly ``0.0`` when ``mask`` is all zeros.
    """
    lgt, tgt, mask_f = _prepare("masked_focal_with_logits", logits, targets, mask)
    ce = F.binary_cross_entropy_with_logits(lgt, tgt, reduction="none")
    p = torch.sigmoid(lgt)
    p_t = p * tgt + (1.0 - p) * (1.0 - tgt)
    per_elem = ce * (1.0 - p_t).pow(gamma)
    if alpha is not None:
        alpha_t = alpha * tgt + (1.0 - alpha) * (1.0 - tgt)
        per_elem = per_elem * alpha_t
    return _reduce(per_elem, mask_f, reduction)


def masked_asymmetric_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
    reduction: str = "mean",
) -> Tensor:
    """Asymmetric Loss (ASL, Ridnik et al., 2021) over supervised entries.

    ASL decouples the focusing strength of positives and negatives and applies
    *probability shifting* to negatives, ``p_neg = max(p - clip, 0)``, which hard-
    thresholds very easy negatives to exactly zero loss. This suits the heavily
    imbalanced, partially-labelled disease-attribute head.

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits, shape ``(B, K)``.
    targets : torch.Tensor
        Binary targets in ``{0, 1}``, shape ``(B, K)``.
    mask : torch.Tensor
        Supervision mask, shape ``(B, K)``.
    gamma_neg : float, optional
        Focusing exponent applied to negatives. Default ``4.0``.
    gamma_pos : float, optional
        Focusing exponent applied to positives. Default ``1.0``.
    clip : float, optional
        Probability margin subtracted from negative probabilities before the
        loss is formed. ``0`` disables shifting. Default ``0.05``.
    reduction : {"mean", "sum"}, optional
        See :func:`masked_bce_with_logits`. Default ``"mean"``.

    Returns
    -------
    torch.Tensor
        Scalar loss. Exactly ``0.0`` when ``mask`` is all zeros.
    """
    lgt, tgt, mask_f = _prepare("masked_asymmetric_loss", logits, targets, mask)
    eps = torch.finfo(logits.dtype).tiny

    p_pos = torch.sigmoid(lgt)
    p_neg = 1.0 - p_pos
    if clip is not None and clip > 0:
        # Shift negative probabilities; easy negatives collapse to zero loss.
        p_neg = (p_neg + clip).clamp(max=1.0)

    log_pos = torch.log(p_pos.clamp(min=eps))
    log_neg = torch.log(p_neg.clamp(min=eps))

    loss_pos = tgt * log_pos
    loss_neg = (1.0 - tgt) * log_neg

    # Asymmetric focusing: pt is the probability of the *wrong* class.
    pt_pos = 1.0 - p_pos
    pt_neg = 1.0 - p_neg
    pt = pt_pos * tgt + pt_neg * (1.0 - tgt)
    gamma = gamma_pos * tgt + gamma_neg * (1.0 - tgt)
    focus = pt.clamp(min=0.0).pow(gamma)

    per_elem = -(loss_pos + loss_neg) * focus
    return _reduce(per_elem, mask_f, reduction)


# --------------------------------------------------------------------------- #
# Regression loss
# --------------------------------------------------------------------------- #
def masked_huber(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    delta: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Huber (smooth-L1) regression loss restricted to supervised entries.

    Used for the biomass head, whose targets are
    ``[log_weight, length_cm, cephalothorax_cm]`` and whose mask marks
    ``cephalothorax_cm`` unsupervised for every DB1 sample.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values, shape ``(B, K)``.
    target : torch.Tensor
        Regression targets, shape ``(B, K)``. Entries where ``mask == 0`` may hold
        any placeholder, including ``NaN``; they are zeroed before use.
    mask : torch.Tensor
        Supervision mask, shape ``(B, K)``.
    delta : float, optional
        Transition point between the quadratic and linear regimes. Default ``1.0``.
    reduction : {"mean", "sum"}, optional
        See :func:`masked_bce_with_logits`. Default ``"mean"``.

    Returns
    -------
    torch.Tensor
        Scalar loss. Exactly ``0.0`` when ``mask`` is all zeros.
    """
    prd, tgt, mask_f = _prepare("masked_huber", pred, target, mask)
    per_elem = F.huber_loss(prd, tgt, reduction="none", delta=delta)
    return _reduce(per_elem, mask_f, reduction)


# --------------------------------------------------------------------------- #
# Cross-head consistency
# --------------------------------------------------------------------------- #
def health_attribute_consistency(
    health_logit: Tensor,
    attr_logits: Tensor,
    attr_mask: Tensor | None = None,
) -> Tensor:
    """Penalise simultaneously confident "healthy" and "disease present" outputs.

    The health head predicts ``P(diseased)``; the attribute head predicts a
    per-disease probability. A sample cannot coherently be healthy *and* carry a
    disease attribute, so the product
    ``P(healthy) * max_k P(attribute_k)`` is driven towards zero.

    Parameters
    ----------
    health_logit : torch.Tensor
        Health logits, shape ``(B,)`` or ``(B, 1)``. Positive class = diseased.
    attr_logits : torch.Tensor
        Attribute logits, shape ``(B, K)`` in ``DISEASE_ATTRS`` order.
    attr_mask : torch.Tensor or None, optional
        Supervision mask, shape ``(B, K)``. Unsupervised attributes are excluded
        from the maximum via ``masked_fill(-inf)``, so a row with no supervised
        attribute contributes exactly ``0``. Default ``None`` (all supervised).

    Returns
    -------
    torch.Tensor
        Scalar penalty in ``[0, 1]``, always finite -- including when every row
        has zero supervised attributes.
    """
    if attr_logits.dim() != 2:
        raise ValueError(f"attr_logits must be 2-D (B, K), got {tuple(attr_logits.shape)}")

    health = health_logit.reshape(-1)
    if health.shape[0] != attr_logits.shape[0]:
        raise ValueError(
            "health_logit and attr_logits must agree on batch size, got "
            f"{health.shape[0]} vs {attr_logits.shape[0]}"
        )

    if attr_mask is not None:
        _check_shapes("health_attribute_consistency", attr_logits, attr_mask)

    if health.numel() == 0:
        # An empty batch would make ``.mean()`` return NaN. Task-scheduled steps
        # legitimately hand a head zero samples, so return a graph-attached 0.
        return attr_logits.sum() * 0.0 + health.sum() * 0.0

    p_healthy = 1.0 - torch.sigmoid(health)

    if attr_mask is None:
        max_logit = attr_logits.max(dim=1).values
        p_dis_max = torch.sigmoid(max_logit)
    else:
        mask_bool = attr_mask.to(dtype=torch.bool)
        neg_inf = torch.tensor(float("-inf"), dtype=attr_logits.dtype, device=attr_logits.device)
        filled = attr_logits.masked_fill(~mask_bool, neg_inf)
        # sigmoid(-inf) == 0 exactly, so fully-unsupervised rows already score 0;
        # torch.where makes that structural rather than a limit artefact.
        max_logit = filled.max(dim=1).values
        p_dis_max = torch.sigmoid(max_logit)
        has_any = mask_bool.any(dim=1)
        p_dis_max = torch.where(has_any, p_dis_max, torch.zeros_like(p_dis_max))

    return (p_healthy * p_dis_max).mean()


# --------------------------------------------------------------------------- #
# Task weighting
# --------------------------------------------------------------------------- #
class HomoscedasticUncertaintyWeighting(nn.Module):
    """Learned multi-task weighting via homoscedastic uncertainty (Kendall & Gal, 2018).

    Each task ``i`` owns a learnable log-variance ``s_i``; the combined objective is

    .. math:: \\sum_i \\exp(-s_i) \\, L_i + s_i

    The ``+ s_i`` term stops the optimiser from driving every weight to zero. The
    effective weight of task ``i`` is ``exp(-s_i)``.

    Tasks missing from the ``losses`` dict on a given step are skipped entirely:
    neither their weighted loss nor their ``s_i`` regulariser is accumulated. This
    matters because the deterministic task schedule sends single-task batches, and
    charging the log-variance penalty for a task that produced no loss would
    otherwise let the model shrink its weight for free.

    Parameters
    ----------
    task_names : list of str
        Task identifiers, defining the index order of ``log_vars``.

    Attributes
    ----------
    task_names : list of str
        Ordered task identifiers.
    log_vars : torch.nn.Parameter
        Learnable log-variances, shape ``(n_tasks,)``, initialised to zeros.
    """

    def __init__(self, task_names: list[str]) -> None:
        super().__init__()
        names = list(task_names)
        if len(names) == 0:
            raise ValueError("task_names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError(f"task_names must be unique, got {names}")
        self.task_names: list[str] = names
        self._index: dict[str, int] = {n: i for i, n in enumerate(names)}
        self.log_vars = nn.Parameter(torch.zeros(len(names)))

    def forward(self, losses: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Combine per-task losses into a single scalar.

        Parameters
        ----------
        losses : dict of str to torch.Tensor
            Scalar loss per task. Tasks absent from this dict are ignored for
            this step; unknown task names raise.

        Returns
        -------
        total : torch.Tensor
            Scalar weighted total, differentiable w.r.t. ``log_vars``.
        weights : dict of str to float
            Current effective weight ``exp(-s_i)`` for each contributing task.

        Raises
        ------
        KeyError
            If ``losses`` contains a task not declared in ``task_names``.
        """
        unknown = set(losses) - set(self._index)
        if unknown:
            raise KeyError(f"unknown task(s) {sorted(unknown)}; declared: {self.task_names}")

        # Zero base that stays attached to the graph, so backward() is safe even
        # when no task contributed on this step.
        total = self.log_vars.sum() * 0.0
        weights: dict[str, float] = {}
        for name, loss in losses.items():
            log_var = self.log_vars[self._index[name]]
            precision = torch.exp(-log_var)
            total = total + precision * loss + log_var
            weights[name] = float(precision.detach())
        return total, weights

    def current_weights(self) -> dict[str, float]:
        """Return ``exp(-s_i)`` for every declared task.

        Returns
        -------
        dict of str to float
            Effective weight per task, detached from the graph.
        """
        with torch.no_grad():
            w = torch.exp(-self.log_vars)
        return {n: float(w[i]) for n, i in self._index.items()}

    def extra_repr(self) -> str:
        return f"task_names={self.task_names}"


class FixedTaskWeighting(nn.Module):
    """Static per-task loss weighting, interchangeable with the learned variant.

    Parameters
    ----------
    weights : dict of str to float
        Fixed multiplier per task name.

    Attributes
    ----------
    weights : dict of str to float
        The configured multipliers.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        if len(weights) == 0:
            raise ValueError("weights must not be empty")
        self.weights: dict[str, float] = {k: float(v) for k, v in weights.items()}

    def forward(self, losses: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Combine per-task losses using the configured static weights.

        Parameters
        ----------
        losses : dict of str to torch.Tensor
            Scalar loss per task. Tasks absent from this dict are ignored.

        Returns
        -------
        total : torch.Tensor
            Scalar weighted total.
        weights_used : dict of str to float
            The weight applied to each contributing task.

        Raises
        ------
        KeyError
            If ``losses`` contains a task with no configured weight.
        """
        unknown = set(losses) - set(self.weights)
        if unknown:
            raise KeyError(
                f"unknown task(s) {sorted(unknown)}; configured: {sorted(self.weights)}"
            )

        total: Tensor | None = None
        weights_used: dict[str, float] = {}
        for name, loss in losses.items():
            w = self.weights[name]
            term = loss * w
            total = term if total is None else total + term
            weights_used[name] = w
        if total is None:
            # No task contributed on this step. A plain ``torch.zeros(())`` is a
            # detached leaf, so the caller's unconditional ``total.backward()``
            # would raise "does not require grad". Hand back a differentiable
            # zero instead, matching HomoscedasticUncertaintyWeighting, which
            # anchors its zero on ``log_vars``. Backward is then a safe no-op:
            # the gradient lands on this throwaway tensor and no parameter is
            # touched.
            total = torch.zeros((), requires_grad=True) * 1.0
        return total, weights_used

    def extra_repr(self) -> str:
        return f"weights={self.weights}"


# --------------------------------------------------------------------------- #
# Domain alignment
# --------------------------------------------------------------------------- #
def coral_loss(source_feats: Tensor, target_feats: Tensor) -> Tensor:
    """CORAL feature alignment (Sun & Saenko, 2016).

    Aligns the second-order statistics of two feature sets by penalising the
    squared Frobenius norm between their covariance matrices, normalised by
    ``4 * d ** 2``.

    Parameters
    ----------
    source_feats : torch.Tensor
        Source-domain features, shape ``(N_s, d)``.
    target_feats : torch.Tensor
        Target-domain features, shape ``(N_t, d)``.

    Returns
    -------
    torch.Tensor
        Scalar alignment loss, or a graph-attached ``0.0`` when either side holds
        fewer than two samples (covariance is undefined there).

    Raises
    ------
    ValueError
        If the two feature sets do not share a feature dimension, or are not 2-D.
    """
    if source_feats.dim() != 2 or target_feats.dim() != 2:
        raise ValueError(
            "coral_loss expects 2-D (N, d) features, got "
            f"{tuple(source_feats.shape)} and {tuple(target_feats.shape)}"
        )
    if source_feats.shape[1] != target_feats.shape[1]:
        raise ValueError(
            "coral_loss: feature dimension mismatch, "
            f"{source_feats.shape[1]} vs {target_feats.shape[1]}"
        )

    n_s, d = source_feats.shape
    n_t = target_feats.shape[0]
    if n_s < 2 or n_t < 2:
        return source_feats.sum() * 0.0 + target_feats.sum() * 0.0

    def _cov(x: Tensor) -> Tensor:
        n = x.shape[0]
        ones = torch.ones(1, n, dtype=x.dtype, device=x.device)
        sums = ones @ x                       # (1, d)
        outer = sums.t() @ sums / n           # (d, d)
        return (x.t() @ x - outer) / (n - 1)

    diff = _cov(source_feats) - _cov(target_feats)
    return (diff * diff).sum() / (4.0 * float(d) ** 2)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import math

    _results: list[tuple[str, bool, str]] = []

    def _check(name: str, fn) -> None:
        """Run one self-test block, recording PASS/FAIL."""
        try:
            fn()
            _results.append((name, True, ""))
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001 - self-test reporting
            _results.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {name} -> {type(exc).__name__}: {exc}")

    torch.manual_seed(0)
    B, K = 8, 3

    _MASKED_LOSSES = {
        "masked_bce_with_logits": masked_bce_with_logits,
        "masked_focal_with_logits": masked_focal_with_logits,
        "masked_asymmetric_loss": masked_asymmetric_loss,
        "masked_huber": masked_huber,
    }

    # --- 1. all-zero mask -> finite exact 0.0 --------------------------------
    def _t1() -> None:
        zero_mask = torch.zeros(B, K)
        logits = torch.randn(B, K) * 5.0
        bin_tgt = torch.randint(0, 2, (B, K)).float()
        reg_tgt = torch.randn(B, K) * 100.0
        # Poison the unsupervised slots the way a real dataloader might.
        bin_tgt_bad = torch.full((B, K), float("nan"))
        reg_tgt_bad = torch.full((B, K), float("nan"))
        for name, fn in _MASKED_LOSSES.items():
            tgt = reg_tgt if name == "masked_huber" else bin_tgt
            bad = reg_tgt_bad if name == "masked_huber" else bin_tgt_bad
            for red in ("mean", "sum"):
                for t in (tgt, bad):
                    out = fn(logits, t, zero_mask, reduction=red)
                    assert torch.isfinite(out), f"{name}[{red}] not finite: {out}"
                    assert out.item() == 0.0, f"{name}[{red}] expected 0.0, got {out.item()}"
        # consistency loss with a fully-unsupervised attribute mask
        c = health_attribute_consistency(torch.randn(B), torch.randn(B, K) * 8.0, zero_mask)
        assert torch.isfinite(c), f"consistency not finite: {c}"
        assert c.item() == 0.0, f"consistency expected 0.0, got {c.item()}"
        # coral with <2 samples
        cl = coral_loss(torch.randn(1, 16), torch.randn(5, 16))
        assert torch.isfinite(cl) and cl.item() == 0.0, f"coral expected 0.0, got {cl.item()}"

    _check("1. all-zero mask -> finite exact 0.0 (all masked losses + consistency + coral)", _t1)

    # --- 2. masked_bce == F.binary_cross_entropy_with_logits (mask all ones) --
    def _t2() -> None:
        logits = torch.randn(B, K) * 3.0
        tgt = torch.randint(0, 2, (B, K)).float()
        ones = torch.ones(B, K)
        ref_mean = F.binary_cross_entropy_with_logits(logits, tgt, reduction="mean")
        got_mean = masked_bce_with_logits(logits, tgt, ones, reduction="mean")
        assert torch.allclose(got_mean, ref_mean, atol=1e-7), f"{got_mean} != {ref_mean}"
        ref_sum = F.binary_cross_entropy_with_logits(logits, tgt, reduction="sum")
        got_sum = masked_bce_with_logits(logits, tgt, ones, reduction="sum")
        assert torch.allclose(got_sum, ref_sum, atol=1e-5), f"{got_sum} != {ref_sum}"

    _check("2. masked_bce == F.binary_cross_entropy_with_logits when mask is all ones", _t2)

    # --- 3. masked-out target VALUES are irrelevant --------------------------
    def _t3() -> None:
        mask = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0],
             [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        )
        logits = torch.randn(B, K) * 2.5
        bin_a = torch.randint(0, 2, (B, K)).float()
        reg_a = torch.randn(B, K)
        # Perturb ONLY the masked-out slots, with hostile values.
        flip = torch.where(mask > 0, torch.zeros(B, K), torch.ones(B, K))
        bin_b = torch.where(flip > 0, 1.0 - bin_a, bin_a)
        bin_c = torch.where(flip > 0, torch.full((B, K), float("nan")), bin_a)
        reg_b = torch.where(flip > 0, reg_a + 1e6, reg_a)
        reg_c = torch.where(flip > 0, torch.full((B, K), float("nan")), reg_a)
        for name, fn in _MASKED_LOSSES.items():
            if name == "masked_huber":
                a, variants = reg_a, (reg_b, reg_c)
            else:
                a, variants = bin_a, (bin_b, bin_c)
            base = fn(logits, a, mask)
            assert torch.isfinite(base), f"{name} base not finite"
            for i, v in enumerate(variants):
                out = fn(logits, v, mask)
                assert torch.isfinite(out), f"{name} variant {i} not finite: {out}"
                assert torch.allclose(base, out, atol=1e-7), (
                    f"{name} changed with masked-out targets (variant {i}): "
                    f"{base.item()} vs {out.item()}"
                )

    _check("3. masked losses ignore the VALUE of targets at masked-out positions", _t3)

    # --- 4. gradients flow, and are finite -----------------------------------
    def _t4() -> None:
        mask = torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
        )
        bin_tgt = torch.randint(0, 2, (B, K)).float()
        reg_tgt = torch.where(mask > 0, torch.randn(B, K), torch.full((B, K), float("nan")))
        for name, fn in _MASKED_LOSSES.items():
            x = (torch.randn(B, K) * 2.0).requires_grad_(True)
            tgt = reg_tgt if name == "masked_huber" else bin_tgt
            loss = fn(x, tgt, mask)
            loss.backward()
            assert x.grad is not None, f"{name}: grad is None"
            assert torch.isfinite(x.grad).all(), f"{name}: non-finite grad {x.grad}"
            assert x.grad.abs().sum() > 0, f"{name}: grad is identically zero"
            # gradient must be exactly zero at unsupervised positions
            assert (x.grad * (1.0 - mask)).abs().max().item() == 0.0, (
                f"{name}: non-zero grad leaked into masked-out positions"
            )

        # consistency loss gradients, including an all-unsupervised row
        cm = mask.clone()
        cm[3] = 0.0
        hl = torch.randn(B).requires_grad_(True)
        al = (torch.randn(B, K) * 3.0).requires_grad_(True)
        c = health_attribute_consistency(hl, al, cm)
        c.backward()
        assert hl.grad is not None and al.grad is not None, "consistency: grad is None"
        assert torch.isfinite(hl.grad).all(), f"consistency: non-finite health grad {hl.grad}"
        assert torch.isfinite(al.grad).all(), f"consistency: non-finite attr grad {al.grad}"

        # coral gradients
        sf = torch.randn(12, 16).requires_grad_(True)
        tf = torch.randn(9, 16).requires_grad_(True)
        cl = coral_loss(sf, tf)
        cl.backward()
        assert sf.grad is not None and torch.isfinite(sf.grad).all(), "coral: bad source grad"
        assert tf.grad is not None and torch.isfinite(tf.grad).all(), "coral: bad target grad"

        # weighting modules
        for mod in (
            HomoscedasticUncertaintyWeighting(["health", "attr", "biomass"]),
            FixedTaskWeighting({"health": 1.0, "attr": 0.5, "biomass": 2.0}),
        ):
            y = (torch.randn(B, K)).requires_grad_(True)
            total, _ = mod({"health": y.pow(2).mean(), "attr": y.abs().mean()})
            total.backward()
            assert y.grad is not None and torch.isfinite(y.grad).all(), (
                f"{type(mod).__name__}: bad grad"
            )

    _check("4. gradients flow: non-None, finite, zero at masked-out positions", _t4)

    # --- 5. consistency semantics -------------------------------------------
    def _t5() -> None:
        attrs_high = torch.full((B, K), 8.0)
        m = torch.ones(B, K)
        diseased = torch.full((B,), 8.0)     # P(diseased) ~ 1 -> P(healthy) ~ 0
        healthy = torch.full((B,), -8.0)     # P(diseased) ~ 0 -> P(healthy) ~ 1
        c_ok = health_attribute_consistency(diseased, attrs_high, m)
        c_bad = health_attribute_consistency(healthy, attrs_high, m)
        assert c_ok.item() < 1e-3, f"expected ~0 for diseased+attrs-high, got {c_ok.item()}"
        assert c_bad.item() > 0.99, f"expected ~1 for healthy+attrs-high, got {c_bad.item()}"
        assert c_bad.item() > 100.0 * c_ok.item(), "contradiction not penalised much harder"
        # (B, 1) health logits must behave identically
        c_col = health_attribute_consistency(healthy.reshape(B, 1), attrs_high, m)
        assert torch.allclose(c_col, c_bad), "(B,1) health logit shape mismatch"
        # masking out the high attributes removes the penalty
        c_masked = health_attribute_consistency(healthy, attrs_high, torch.zeros(B, K))
        assert c_masked.item() == 0.0, f"expected 0.0 with no supervised attrs, got {c_masked}"
        # partial: only an unsupervised attribute is high -> no penalty
        attrs_mixed = torch.tensor([[-8.0, -8.0, 9.0]]).repeat(B, 1)
        part_mask = torch.tensor([[1.0, 1.0, 0.0]]).repeat(B, 1)
        c_part = health_attribute_consistency(healthy, attrs_mixed, part_mask)
        assert c_part.item() < 1e-3, f"unsupervised attr leaked into max: {c_part.item()}"
        print(f"       c(diseased,attrs_high)={c_ok.item():.3e}  "
              f"c(healthy,attrs_high)={c_bad.item():.6f}")

    _check("5. consistency ~0 when diseased+attrs-high, large when healthy+attrs-high", _t5)

    # --- 6. uncertainty weighting reduces to a plain sum at log_var == 0 -----
    def _t6() -> None:
        names = ["health", "attr", "biomass", "domain"]
        w = HomoscedasticUncertaintyWeighting(names)
        assert w.log_vars.shape == (4,), f"log_vars shape {tuple(w.log_vars.shape)}"
        assert torch.all(w.log_vars == 0), "log_vars must init to zeros"

        losses = {
            "health": torch.tensor(0.7),
            "attr": torch.tensor(1.3),
            "biomass": torch.tensor(0.25),
            "domain": torch.tensor(2.0),
        }
        total, weights = w(losses)
        expected = sum(v.item() for v in losses.values())
        assert math.isclose(total.item(), expected, rel_tol=1e-6), (
            f"expected sum {expected}, got {total.item()}"
        )
        assert all(math.isclose(v, 1.0, rel_tol=1e-6) for v in weights.values()), weights
        assert set(weights) == set(names), weights

        # Absent tasks contribute NOTHING -- neither loss nor log_var penalty.
        subset = {"attr": torch.tensor(1.3)}
        t_sub, w_sub = w(subset)
        assert math.isclose(t_sub.item(), 1.3, rel_tol=1e-6), t_sub.item()
        assert set(w_sub) == {"attr"}, w_sub

        # With a non-zero log_var the absent task still must not be charged.
        with torch.no_grad():
            w.log_vars.copy_(torch.tensor([0.5, -0.5, 1.0, 2.0]))
        t2, w2 = w({"attr": torch.tensor(1.3)})
        exp2 = math.exp(0.5) * 1.3 + (-0.5)
        assert math.isclose(t2.item(), exp2, rel_tol=1e-5), f"{t2.item()} != {exp2}"
        assert math.isclose(w2["attr"], math.exp(0.5), rel_tol=1e-6), w2

        # Empty dict is safe and backward-able.
        t_empty, w_empty = w({})
        assert t_empty.item() == 0.0 and w_empty == {}, (t_empty, w_empty)
        t_empty.backward()

        # FixedTaskWeighting parity of interface
        f = FixedTaskWeighting({"health": 1.0, "attr": 1.0, "biomass": 1.0, "domain": 1.0})
        tf_total, tf_w = f(losses)
        assert math.isclose(tf_total.item(), expected, rel_tol=1e-6), tf_total.item()
        assert tf_w == {k: 1.0 for k in losses}, tf_w
        f2 = FixedTaskWeighting({"health": 2.0, "attr": 0.5})
        t3, _ = f2({"health": torch.tensor(1.0), "attr": torch.tensor(4.0)})
        assert math.isclose(t3.item(), 2.0 * 1.0 + 0.5 * 4.0, rel_tol=1e-6), t3.item()

    _check("6. HomoscedasticUncertaintyWeighting == sum of losses when log_vars == 0", _t6)

    # --- 7. non-finite PREDICTIONS at unsupervised positions are inert --------
    def _t7() -> None:
        # Regression guard: `per_elem * mask` makes inf*0 == nan*0 == NaN, so a
        # blown-up unsupervised column used to wipe out the whole loss.
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        bin_tgt = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        reg_tgt = torch.tensor([[2.5, 12.0, 0.0], [3.1, 10.4, 0.0]])
        clean = torch.tensor([[0.7, -1.2, 0.4], [-0.3, 0.9, -0.8]])
        for bad in (float("inf"), float("-inf"), float("nan")):
            dirty = torch.where(mask > 0, clean, torch.full_like(clean, bad))
            for name, fn in _MASKED_LOSSES.items():
                tgt = reg_tgt if name == "masked_huber" else bin_tgt
                ref = fn(clean, tgt, mask)
                got = fn(dirty, tgt, mask)
                assert torch.isfinite(got), f"{name}: pred={bad} at masked slot -> {got}"
                assert torch.allclose(ref, got, atol=1e-7), (
                    f"{name}: unsupervised pred={bad} changed the loss "
                    f"{ref.item()} vs {got.item()}"
                )
        # ... and with an all-zero mask the loss is still exactly 0.0
        zero = torch.zeros(2, 3)
        for bad in (float("inf"), float("nan")):
            dirty = torch.full((2, 3), bad)
            for name, fn in _MASKED_LOSSES.items():
                tgt = reg_tgt if name == "masked_huber" else bin_tgt
                out = fn(dirty, tgt, zero)
                assert torch.isfinite(out) and out.item() == 0.0, f"{name}[{bad}] -> {out}"
        # gradients stay finite when an unsupervised column overflows
        x = torch.tensor([[2.4, 11.5, 1e30], [3.0, 10.0, 1e30]], requires_grad=True)
        masked_huber(x * x, reg_tgt, mask).backward()
        assert torch.isfinite(x.grad).all(), f"overflowing masked column poisoned grad: {x.grad}"

    _check("7. non-finite PREDICTIONS at unsupervised positions cannot poison the loss", _t7)

    # --- 8. empty batches and empty loss dicts -------------------------------
    def _t8() -> None:
        # B == 0 happens whenever the task schedule hands a head no samples.
        for name, fn in _MASKED_LOSSES.items():
            for red in ("mean", "sum"):
                out = fn(torch.zeros(0, K), torch.zeros(0, K), torch.zeros(0, K), reduction=red)
                assert torch.isfinite(out) and out.item() == 0.0, f"{name}[{red}] -> {out}"
        for am in (torch.zeros(0, K), None):
            c = health_attribute_consistency(torch.zeros(0), torch.zeros(0, K), am)
            assert torch.isfinite(c), f"consistency on empty batch -> {c}"
            assert c.item() == 0.0, f"consistency on empty batch -> {c.item()}"
        for s, t in (((0, 8), (5, 8)), ((5, 8), (0, 8)), ((1, 8), (1, 8))):
            cl = coral_loss(torch.randn(*s), torch.randn(*t))
            assert torch.isfinite(cl) and cl.item() == 0.0, f"coral{s}{t} -> {cl}"
        # Both weighting modules must survive an empty step AND its backward.
        for mod in (
            HomoscedasticUncertaintyWeighting(["health", "attr"]),
            FixedTaskWeighting({"health": 1.0, "attr": 0.5}),
        ):
            total, used = mod({})
            assert torch.isfinite(total) and total.item() == 0.0, (type(mod).__name__, total)
            assert used == {}, used
            total.backward()  # must not raise

    _check("8. empty batches / empty loss dicts stay finite and backward-able", _t8)

    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_total = len(_results)
    print("-" * 72)
    print(f"SELF-TEST SUMMARY: {n_pass}/{n_total} passed")
    for name, ok, err in _results:
        if not ok:
            print(f"  FAILED: {name} :: {err}")
    print("ALL ASSERTIONS PASSED" if n_pass == n_total else "SOME ASSERTIONS FAILED")
    raise SystemExit(0 if n_pass == n_total else 1)
