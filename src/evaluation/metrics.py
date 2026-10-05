"""Evaluation metrics for the ShrimpVision-MTL multi-task model.

This module is deliberately free of torch: every entry point consumes numpy
arrays so that it can be reused from training callbacks, offline evaluation
scripts and notebook analyses alike.

Three families of metrics live here.

Binary health classification
    :func:`disease_health_metrics` scores the single-logit health head, where the
    positive class is DISEASED (1 = diseased, 0 = healthy). It reports the usual
    discrimination metrics plus calibration (Brier score and expected calibration
    error) and the raw confusion-matrix counts, so downstream reports never have
    to recompute them from probabilities.

Multi-label disease attributes
    :func:`disease_attribute_metrics` scores the ``(N, 3)`` attribute head in the
    canonical ``DISEASE_ATTRS`` order. Every computation is restricted to entries
    whose supervision mask is 1. An unknown label is *not* a negative: it is
    simply absent from the evaluation. Attributes with no supervised entries at
    all yield ``nan`` and are excluded from the macro aggregates rather than
    silently contributing a zero.

Biomass regression
    :func:`biomass_metrics` and :func:`bland_altman_stats` score a single
    continuous target (weight in grams, length in cm, or cephalothorax length in
    cm). The caller is responsible for exponentiating log-space weight
    predictions before passing them here, and for telling us the ``unit``.

Conventions
-----------
Every function ignores entries where the supervision mask is 0, and additionally
drops any pair in which the target or the prediction is not finite. A metric
computed over an empty selection returns ``float('nan')`` instead of raising, and
metrics that are mathematically undefined (AUROC with a single class present,
correlation with zero variance) likewise return ``nan``. This is load-bearing:
batches and even whole evaluation folds legitimately contain no supervised
examples for some labels.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "disease_health_metrics",
    "expected_calibration_error",
    "disease_attribute_metrics",
    "select_thresholds",
    "biomass_metrics",
    "bland_altman_stats",
    "THRESHOLD_OBJECTIVES",
    "DEFAULT_THRESHOLD_GRID",
]

NAN = float("nan")

#: Objectives accepted by :func:`select_thresholds`.
THRESHOLD_OBJECTIVES: tuple[str, ...] = ("f1", "balanced_accuracy", "youden", "mcc")

#: Default sweep used when ``select_thresholds`` is called without a grid.
#: Marked read-only: it is shared by every caller that omits ``grid``, so an
#: in-place edit anywhere would silently change threshold selection everywhere.
DEFAULT_THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
DEFAULT_THRESHOLD_GRID.flags.writeable = False

#: Scores within this tolerance of the maximum count as tied.
_TIE_TOL = 1e-12


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _as_1d_float(a: np.ndarray) -> np.ndarray:
    """Return ``a`` as a contiguous 1-D float64 array."""
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _valid_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the scorable subset of a binary problem.

    Entries are kept when the supervision mask is non-zero (when a mask is given)
    and both the label and the probability are finite.

    Parameters
    ----------
    y_true : ndarray
        Binary labels, any shape broadcastable to 1-D.
    y_prob : ndarray
        Predicted probabilities of the positive class.
    mask : ndarray, optional
        Supervision mask; 1 marks a supervised entry, 0 marks UNKNOWN.

    Returns
    -------
    y : ndarray
        Kept labels, float64, 1-D.
    p : ndarray
        Kept probabilities, float64, 1-D.
    """
    y = _as_1d_float(y_true)
    p = _as_1d_float(y_prob)
    if y.shape != p.shape:
        raise ValueError(f"y_true shape {y.shape} != y_prob shape {p.shape}")
    keep = np.isfinite(y) & np.isfinite(p)
    if mask is not None:
        m = _as_1d_float(mask)
        if m.shape != y.shape:
            raise ValueError(f"mask shape {m.shape} != y_true shape {y.shape}")
        keep &= m != 0.0
    return y[keep], p[keep]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall and F1 from confusion counts, 0.0 on zero division."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _confusion(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(tn, fp, fn, tp)`` for binary 0/1 arrays."""
    yb = y > 0.5
    pb = pred > 0.5
    tp = int(np.count_nonzero(yb & pb))
    tn = int(np.count_nonzero(~yb & ~pb))
    fp = int(np.count_nonzero(~yb & pb))
    fn = int(np.count_nonzero(yb & ~pb))
    return tn, fp, fn, tp


def _safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC, or ``nan`` when fewer than two classes are present."""
    if y.size == 0 or np.unique(y > 0.5).size < 2:
        return NAN
    return float(roc_auc_score((y > 0.5).astype(np.int64), p))


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision, or ``nan`` when fewer than two classes are present."""
    if y.size == 0 or np.unique(y > 0.5).size < 2:
        return NAN
    return float(average_precision_score((y > 0.5).astype(np.int64), p))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error between probability and binary outcome."""
    if y.size == 0:
        return NAN
    return float(np.mean((p - (y > 0.5).astype(np.float64)) ** 2))


def _nanmean_of(values: Sequence[float]) -> float:
    """Mean over the finite entries of ``values``; ``nan`` when none are finite."""
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else NAN


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Equal-width binned expected calibration error.

    Probabilities are partitioned into ``n_bins`` equal-width bins over
    ``[0, 1]``. Within each bin the mean predicted probability (confidence) is
    compared with the observed positive rate (accuracy), and the absolute gaps
    are averaged weighted by bin population:

    .. math::
        \\mathrm{ECE} = \\sum_b \\frac{|B_b|}{N}
                        \\left| \\mathrm{acc}(B_b) - \\mathrm{conf}(B_b) \\right|

    This is the reliability-diagram form of ECE for a probability of the positive
    class, which is the quantity the health and attribute heads emit.

    Parameters
    ----------
    y_true : ndarray
        Binary labels.
    y_prob : ndarray
        Predicted probability of the positive class, in ``[0, 1]``.
    n_bins : int, default=15
        Number of equal-width bins.

    Returns
    -------
    float
        The expected calibration error, or ``nan`` if no finite pair is present.

    Notes
    -----
    Empty bins are skipped, so the result is invariant to bins that no sample
    ever falls into.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    y, p = _valid_binary(y_true, y_prob)
    n = y.size
    if n == 0:
        return NAN
    y = (y > 0.5).astype(np.float64)
    pc = np.clip(p, 0.0, 1.0)
    # floor(p * n_bins) puts each probability in an equal-width bin; the top edge
    # (p == 1) is folded back into the last bin.
    idx = np.minimum((pc * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    conf_sum = np.bincount(idx, weights=pc, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=y, minlength=n_bins)
    nonempty = counts > 0
    gaps = np.abs(acc_sum[nonempty] - conf_sum[nonempty]) / counts[nonempty]
    return float(np.sum(counts[nonempty] * gaps) / n)


# --------------------------------------------------------------------------- #
# Binary health head
# --------------------------------------------------------------------------- #
def disease_health_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    mask: np.ndarray | None = None,
    n_bins: int = 15,
) -> dict:
    """Score the binary health head (positive class = DISEASED).

    Parameters
    ----------
    y_true : ndarray of shape (N,)
        Binary labels, 1 = diseased, 0 = healthy.
    y_prob : ndarray of shape (N,)
        Predicted probability that the sample is diseased.
    threshold : float, default=0.5
        Decision threshold; a sample is predicted diseased when
        ``y_prob >= threshold``.
    mask : ndarray of shape (N,), optional
        Supervision mask. Entries equal to 0 are ignored entirely. The health
        label is supervised for every record in this project, so the default of
        ``None`` (score everything) is the normal case.
    n_bins : int, default=15
        Bin count for the expected calibration error.

    Returns
    -------
    dict
        Keys ``accuracy``, ``balanced_accuracy``, ``precision``, ``recall``,
        ``f1``, ``auroc``, ``average_precision``, ``brier``, ``ece``, ``tn``,
        ``fp``, ``fn``, ``tp``, ``n``. Threshold-free metrics (``auroc``,
        ``average_precision``) are ``nan`` when only one class is present;
        ``balanced_accuracy`` is ``nan`` for the same reason, since one of its
        two component rates would be undefined.

    Notes
    -----
    Confusion counts are ints; every other value is a float. With an empty
    selection all rates are ``nan`` and all counts are 0.
    """
    y, p = _valid_binary(y_true, y_prob, mask)
    n = int(y.size)
    if n == 0:
        return {
            "accuracy": NAN, "balanced_accuracy": NAN, "precision": NAN,
            "recall": NAN, "f1": NAN, "auroc": NAN, "average_precision": NAN,
            "brier": NAN, "ece": NAN,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0, "n": 0,
        }

    pred = (p >= threshold).astype(np.float64)
    tn, fp, fn, tp = _confusion(y, pred)
    precision, recall, f1 = _prf(tp, fp, fn)

    n_pos = tp + fn
    n_neg = tn + fp
    if n_pos > 0 and n_neg > 0:
        balanced_accuracy = 0.5 * (tp / n_pos + tn / n_neg)
    else:
        balanced_accuracy = NAN

    return {
        "accuracy": float((tp + tn) / n),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": _safe_auroc(y, p),
        "average_precision": _safe_ap(y, p),
        "brier": _brier(y, p),
        "ece": expected_calibration_error(y, p, n_bins=n_bins),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Multi-label disease attributes
# --------------------------------------------------------------------------- #
def disease_attribute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    thresholds: np.ndarray,
    attr_names: Sequence[str],
    n_bins: int = 15,
) -> dict:
    """Score the multi-label disease-attribute head, masking UNKNOWN entries.

    Parameters
    ----------
    y_true : ndarray of shape (N, K)
        Binary attribute labels in ``DISEASE_ATTRS`` order. Values at masked-out
        positions are meaningless and are never read.
    y_prob : ndarray of shape (N, K)
        Predicted per-attribute probabilities.
    mask : ndarray of shape (N, K)
        Supervision mask; 1.0 marks a supervised entry, 0.0 marks UNKNOWN. An
        UNKNOWN entry is excluded from every metric and is never counted as a
        negative.
    thresholds : ndarray of shape (K,)
        Per-attribute decision thresholds, typically from
        :func:`select_thresholds` fitted on validation data.
    attr_names : sequence of str
        Attribute names, length ``K``, in the same order as the columns.
    n_bins : int, default=15
        Bin count for the per-attribute expected calibration error.

    Returns
    -------
    dict
        One entry per attribute name mapping to a dict with keys ``precision``,
        ``recall``, ``f1``, ``average_precision``, ``auroc``, ``support_pos``,
        ``support_supervised``, ``brier`` and ``ece``. Plus the aggregate keys:

        ``macro_f1``
            Mean F1 over attributes with at least one supervised entry.
        ``micro_f1``
            F1 over all supervised ``(sample, attribute)`` entries pooled.
        ``macro_map``
            Mean average precision over attributes with at least one supervised
            positive and a computable AP (both classes present).
        ``hamming_loss``
            Fraction of supervised entries whose thresholded prediction differs
            from the label.
        ``exact_match_ratio``
            Fraction of *fully supervised* rows -- rows where every attribute is
            supervised -- whose predictions match the label on all attributes.
            Rows with any UNKNOWN attribute are excluded, because a partial row
            cannot be exactly matched.
        ``n_rows``, ``n_supervised``, ``n_rows_fully_supervised``
            Population counts backing the aggregates above.

    Notes
    -----
    An attribute with zero supervised entries yields ``nan`` for all of its
    metrics and is excluded from ``macro_f1`` and ``macro_map``; it does not
    drag the macro averages towards zero.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)

    if y_true.ndim != 2:
        raise ValueError(f"y_true must be 2-D (N, K), got shape {y_true.shape}")
    if y_prob.shape != y_true.shape or mask.shape != y_true.shape:
        raise ValueError(
            "y_true, y_prob and mask must share shape; got "
            f"{y_true.shape}, {y_prob.shape}, {mask.shape}"
        )
    n_rows, n_attrs = y_true.shape
    if thresholds.shape[0] != n_attrs:
        raise ValueError(
            f"thresholds has {thresholds.shape[0]} entries, expected {n_attrs}"
        )
    if len(attr_names) != n_attrs:
        raise ValueError(
            f"attr_names has {len(attr_names)} entries, expected {n_attrs}"
        )

    out: dict = {}
    f1_per_attr: list[float] = []
    ap_per_attr: list[float] = []
    pooled_y: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []

    for k, name in enumerate(attr_names):
        yk, pk = _valid_binary(y_true[:, k], y_prob[:, k], mask[:, k])
        n_sup = int(yk.size)
        if n_sup == 0:
            out[name] = {
                "precision": NAN, "recall": NAN, "f1": NAN,
                "average_precision": NAN, "auroc": NAN,
                "support_pos": 0, "support_supervised": 0,
                "brier": NAN, "ece": NAN,
            }
            continue

        predk = (pk >= thresholds[k]).astype(np.float64)
        _, fp, fn, tp = _confusion(yk, predk)
        precision, recall, f1 = _prf(tp, fp, fn)
        ap = _safe_ap(yk, pk)

        out[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_precision": ap,
            "auroc": _safe_auroc(yk, pk),
            "support_pos": int(np.count_nonzero(yk > 0.5)),
            "support_supervised": n_sup,
            "brier": _brier(yk, pk),
            "ece": expected_calibration_error(yk, pk, n_bins=n_bins),
        }

        f1_per_attr.append(f1)
        if out[name]["support_pos"] > 0:
            ap_per_attr.append(ap)
        pooled_y.append((yk > 0.5).astype(np.float64))
        pooled_pred.append(predk)

    # ---- pooled (micro) aggregates over supervised entries only ----------- #
    if pooled_y:
        all_y = np.concatenate(pooled_y)
        all_pred = np.concatenate(pooled_pred)
        _, m_fp, m_fn, m_tp = _confusion(all_y, all_pred)
        _, _, micro_f1 = _prf(m_tp, m_fp, m_fn)
        n_supervised = int(all_y.size)
        hamming = float(np.mean(all_y != all_pred))
    else:
        micro_f1 = NAN
        n_supervised = 0
        hamming = NAN

    # ---- exact match over FULLY supervised rows only ---------------------- #
    row_finite = np.all(np.isfinite(y_true) & np.isfinite(y_prob), axis=1)
    full_rows = np.all(mask != 0.0, axis=1) & row_finite
    n_full = int(np.count_nonzero(full_rows))
    if n_full > 0:
        yf = y_true[full_rows] > 0.5
        pf = y_prob[full_rows] >= thresholds[None, :]
        exact_match = float(np.mean(np.all(yf == pf, axis=1)))
    else:
        exact_match = NAN

    out["macro_f1"] = _nanmean_of(f1_per_attr)
    out["micro_f1"] = float(micro_f1)
    out["macro_map"] = _nanmean_of(ap_per_attr)
    out["hamming_loss"] = hamming
    out["exact_match_ratio"] = exact_match
    out["n_rows"] = int(n_rows)
    out["n_supervised"] = n_supervised
    out["n_rows_fully_supervised"] = n_full
    return out


# --------------------------------------------------------------------------- #
# Threshold selection
# --------------------------------------------------------------------------- #
def _objective_score(objective: str, tp: int, fp: int, fn: int, tn: int) -> float:
    """Evaluate a threshold-selection objective from confusion counts."""
    if objective == "f1":
        return _prf(tp, fp, fn)[2]
    if objective in ("balanced_accuracy", "youden"):
        n_pos = tp + fn
        n_neg = tn + fp
        if n_pos == 0 or n_neg == 0:
            return NAN
        sensitivity = tp / n_pos
        specificity = tn / n_neg
        # Youden's J = sens + spec - 1 = 2 * balanced_accuracy - 1, so both
        # objectives are maximised at the same threshold; report each on its
        # own natural scale.
        if objective == "youden":
            return sensitivity + specificity - 1.0
        return 0.5 * (sensitivity + specificity)
    if objective == "mcc":
        num = (tp * tn) - (fp * fn)
        den = float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if den <= 0.0:
            # A degenerate confusion matrix (an empty predicted or actual class)
            # leaves MCC undefined. Report nan rather than the conventional 0.0:
            # inside a threshold sweep a 0.0 would outrank every genuinely
            # anti-correlated threshold and hand back a constant predictor.
            return NAN
        return float(num / np.sqrt(den))
    raise ValueError(
        f"Unknown objective {objective!r}; expected one of {THRESHOLD_OBJECTIVES}"
    )


def _plateau_centre(candidates: np.ndarray, scores: np.ndarray) -> float | None:
    """Return the centre of the widest plateau of maximal-scoring candidates.

    The objective, viewed as a function of the threshold, is a step function that
    only changes where the threshold crosses a predicted probability. Its maximum
    is therefore attained on one or more *intervals*, and on cleanly separable
    data the interval is wide. Picking the middle of the widest such interval is
    more robust under transfer to unseen data than picking its edge.

    Crucially, the value returned is always one of ``candidates`` and is always a
    maximiser. Taking the median of the whole tied set is *not* safe: when the
    maximum is attained on two or more disjoint plateaus the median falls into
    the valley between them and scores strictly below the maximum.

    Parameters
    ----------
    candidates : ndarray of shape (M,)
        Candidate thresholds, in any order.
    scores : ndarray of shape (M,)
        Objective value at each candidate; ``nan`` marks an undefined objective.

    Returns
    -------
    float or None
        The chosen threshold, or ``None`` when no candidate scored finitely.
    """
    finite = np.isfinite(scores)
    if not np.any(finite):
        return None

    best = float(np.max(scores[finite]))
    order = np.argsort(candidates, kind="stable")
    cand_sorted = candidates[order]
    score_sorted = scores[order]
    tied = np.isfinite(score_sorted) & (score_sorted >= best - _TIE_TOL)

    idx = np.flatnonzero(tied)
    # Split the tied positions into maximal runs of consecutive grid points; each
    # run is one plateau.
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    widest = max(
        runs,
        key=lambda r: (float(cand_sorted[r[-1]] - cand_sorted[r[0]]), int(r.size)),
    )
    return float(cand_sorted[widest[widest.size // 2]])


def select_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    objective: str = "f1",
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Pick per-attribute decision thresholds maximising an objective.

    Intended to be run on the VALIDATION split; the resulting vector is then
    frozen and applied to the test split by :func:`disease_attribute_metrics`.
    UNKNOWN entries are excluded from the sweep, so a threshold is never tuned
    against fabricated negatives.

    Parameters
    ----------
    y_true : ndarray of shape (N, K)
        Binary attribute labels.
    y_prob : ndarray of shape (N, K)
        Predicted per-attribute probabilities.
    mask : ndarray of shape (N, K)
        Supervision mask; 0 marks UNKNOWN.
    objective : {'f1', 'balanced_accuracy', 'youden', 'mcc'}, default='f1'
        Quantity to maximise. ``balanced_accuracy``, ``youden`` and ``mcc`` are
        undefined for a degenerate confusion matrix (an empty predicted or actual
        class); such thresholds score ``nan`` and are skipped rather than
        competing for the maximum.
    grid : ndarray, optional
        Candidate thresholds. Defaults to ``np.linspace(0.01, 0.99, 99)``.

    Returns
    -------
    ndarray of shape (K,)
        The selected threshold per attribute, float64. Every returned value is
        drawn from ``grid`` and is guaranteed to attain the maximum objective
        over ``grid`` (except for the ``0.5`` fallbacks described below).

    Notes
    -----
    An attribute with no supervised entries, or with no supervised positives,
    falls back to ``0.5``: there is no signal to tune against, and any other
    choice would be an artefact of the negatives alone. The same fallback applies
    when the objective is undefined at every candidate.

    When several thresholds tie for the maximum -- the norm on cleanly separable
    data, where every threshold inside the margin scores identically -- the
    candidate at the middle of the *widest contiguous plateau* of maximisers is
    returned. That lands in the middle of the separating gap rather than on its
    edge, which transfers better to unseen data.

    Note that the median of the tied set as a whole would be wrong: the maximum
    can be attained on several disjoint plateaus, and the median then falls into
    the valley between them, returning a threshold that scores strictly below the
    maximum. Selecting within a single plateau avoids that.
    """
    if objective not in THRESHOLD_OBJECTIVES:
        raise ValueError(
            f"Unknown objective {objective!r}; expected one of {THRESHOLD_OBJECTIVES}"
        )
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if y_true.ndim != 2:
        raise ValueError(f"y_true must be 2-D (N, K), got shape {y_true.shape}")
    if y_prob.shape != y_true.shape or mask.shape != y_true.shape:
        raise ValueError(
            "y_true, y_prob and mask must share shape; got "
            f"{y_true.shape}, {y_prob.shape}, {mask.shape}"
        )

    candidates = (
        DEFAULT_THRESHOLD_GRID if grid is None
        else np.asarray(grid, dtype=np.float64).reshape(-1)
    )
    if candidates.size == 0:
        raise ValueError("grid must contain at least one candidate threshold")

    n_attrs = y_true.shape[1]
    chosen = np.full(n_attrs, 0.5, dtype=np.float64)

    for k in range(n_attrs):
        yk, pk = _valid_binary(y_true[:, k], y_prob[:, k], mask[:, k])
        if yk.size == 0 or np.count_nonzero(yk > 0.5) == 0:
            continue  # keep the 0.5 fallback

        yb = yk > 0.5
        scores = np.empty(candidates.size, dtype=np.float64)
        for i, t in enumerate(candidates):
            pb = pk >= t
            tp = int(np.count_nonzero(yb & pb))
            fp = int(np.count_nonzero(~yb & pb))
            fn = int(np.count_nonzero(yb & ~pb))
            tn = int(np.count_nonzero(~yb & ~pb))
            scores[i] = _objective_score(objective, tp, fp, fn, tn)

        centre = _plateau_centre(candidates, scores)
        if centre is not None:
            chosen[k] = centre
        # else: no threshold scored finitely -> keep the 0.5 fallback

    return chosen


# --------------------------------------------------------------------------- #
# Biomass regression
# --------------------------------------------------------------------------- #
def biomass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit: str = "g",
    mape_floor: float | None = None,
    mask: np.ndarray | None = None,
) -> dict:
    """Score one continuous biomass target in its natural (non-log) units.

    Weight is modelled in natural-log space by the network, so the caller must
    exponentiate predictions and targets back to grams before calling this;
    otherwise MAPE and MAE would describe log-grams, which no reader expects.

    Parameters
    ----------
    y_true : ndarray of shape (N,)
        Ground-truth values in ``unit``.
    y_pred : ndarray of shape (N,)
        Predicted values in ``unit``.
    unit : str, default='g'
        Unit label, echoed back in the result for report rendering.
    mape_floor : float, optional
        Samples with ``|y_true| < mape_floor`` are excluded from MAPE only, to
        stop a near-zero denominator from exploding the percentage. Defaults to
        ``1e-6``; callers scoring weight in grams should pass something
        physically meaningful such as ``0.5``.
    mask : ndarray of shape (N,), optional
        Supervision mask; 0 marks a missing target. Cephalothorax length is
        missing for DB1, so that column arrives fully masked for those rows.

    Returns
    -------
    dict
        Keys ``mae``, ``rmse``, ``mape``, ``r2``, ``medae``, ``pearson_r``,
        ``spearman_r``, ``n``, ``unit``, plus ``n_mape_excluded`` and
        ``mape_floor``. ``mape`` is a percentage (0-100 scale, unbounded above).

    Notes
    -----
    ``r2`` is ``nan`` when fewer than two samples survive or when the targets
    have zero variance (the denominator of the coefficient of determination
    vanishes). ``pearson_r`` and ``spearman_r`` are ``nan`` when fewer than three
    samples survive or when either input has zero variance.
    """
    floor = 1e-6 if mape_floor is None else float(mape_floor)
    y, p = _valid_binary(y_true, y_pred, mask)  # generic finite+mask selection
    n = int(y.size)

    result: dict = {
        "mae": NAN, "rmse": NAN, "mape": NAN, "r2": NAN, "medae": NAN,
        "pearson_r": NAN, "spearman_r": NAN,
        "n": n, "unit": unit,
        "n_mape_excluded": 0, "mape_floor": floor,
    }
    if n == 0:
        return result

    err = p - y
    abs_err = np.abs(err)
    result["mae"] = float(np.mean(abs_err))
    result["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    result["medae"] = float(np.median(abs_err))

    # MAPE with a guarded denominator.
    ok = np.abs(y) >= floor
    result["n_mape_excluded"] = int(n - np.count_nonzero(ok))
    if np.any(ok):
        result["mape"] = float(np.mean(abs_err[ok] / np.abs(y[ok])) * 100.0)

    # R^2 needs a non-degenerate target.
    if n >= 2:
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot > 0.0:
            ss_res = float(np.sum(err ** 2))
            result["r2"] = float(1.0 - ss_res / ss_tot)

    # Correlations need >= 3 points and non-zero variance on both sides.
    if n >= 3 and np.std(y) > 0.0 and np.std(p) > 0.0:
        result["pearson_r"] = float(stats.pearsonr(y, p)[0])
        result["spearman_r"] = float(stats.spearmanr(y, p).statistic)

    return result


def bland_altman_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    """Bland-Altman agreement statistics for a continuous target.

    Quantifies systematic bias and the spread of disagreement between the model
    and the reference measurement, which is the standard way to report whether an
    automated biomass estimate could stand in for manual weighing.

    Parameters
    ----------
    y_true : ndarray of shape (N,)
        Reference measurements.
    y_pred : ndarray of shape (N,)
        Model predictions.
    mask : ndarray of shape (N,), optional
        Supervision mask; 0 marks a missing target.

    Returns
    -------
    dict
        Keys ``bias`` (mean of ``pred - true``), ``sd_diff`` (sample standard
        deviation of the differences, ``ddof=1``), ``loa_lower`` and
        ``loa_upper`` (the 95% limits of agreement, ``bias +/- 1.96 * sd``), and
        ``n``.

    Notes
    -----
    With fewer than two samples the standard deviation and both limits are
    ``nan``; with exactly one sample the bias is still reported.
    """
    y, p = _valid_binary(y_true, y_pred, mask)
    n = int(y.size)
    if n == 0:
        return {"bias": NAN, "sd_diff": NAN, "loa_lower": NAN,
                "loa_upper": NAN, "n": 0}

    diff = p - y
    bias = float(np.mean(diff))
    if n < 2:
        return {"bias": bias, "sd_diff": NAN, "loa_lower": NAN,
                "loa_upper": NAN, "n": n}

    sd = float(np.std(diff, ddof=1))
    return {
        "bias": bias,
        "sd_diff": sd,
        "loa_lower": float(bias - 1.96 * sd),
        "loa_upper": float(bias + 1.96 * sd),
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Run the module self-test, printing one line per checked property."""
    import math

    rng = np.random.default_rng(0)
    attrs = ("black_gill", "wssv", "yellow_head")

    # -- 1. perfect predictions -------------------------------------------- #
    y = np.array([0, 1, 0, 1, 1, 0], dtype=np.float64)
    h = disease_health_metrics(y, y.copy())
    assert h["f1"] == 1.0, h
    assert h["accuracy"] == 1.0 and h["balanced_accuracy"] == 1.0, h
    assert h["auroc"] == 1.0 and h["average_precision"] == 1.0, h
    assert h["brier"] == 0.0, h
    assert (h["tp"], h["tn"], h["fp"], h["fn"], h["n"]) == (3, 3, 0, 0, 6), h
    print(f"[1a] health perfect: f1={h['f1']:.3f} auroc={h['auroc']:.3f} "
          f"brier={h['brier']:.3f} bal_acc={h['balanced_accuracy']:.3f}")

    bt = np.array([12.0, 25.0, 31.5, 40.0, 8.25], dtype=np.float64)
    b = biomass_metrics(bt, bt.copy(), unit="g", mape_floor=0.5)
    assert b["mae"] == 0.0 and b["rmse"] == 0.0 and b["medae"] == 0.0, b
    assert b["r2"] == 1.0, b
    assert abs(b["mape"]) < 1e-12, b
    assert abs(b["pearson_r"] - 1.0) < 1e-12, b
    assert abs(b["spearman_r"] - 1.0) < 1e-12, b
    print(f"[1b] biomass perfect: mae={b['mae']:.3f} r2={b['r2']:.3f} "
          f"mape={b['mape']:.3f}% pearson={b['pearson_r']:.3f} n={b['n']}")

    # single-class health -> auroc/ap nan, not a crash
    one = disease_health_metrics(np.ones(4), np.array([0.9, 0.8, 0.7, 0.6]))
    assert math.isnan(one["auroc"]) and math.isnan(one["average_precision"]), one
    assert math.isnan(one["balanced_accuracy"]), one
    assert one["recall"] == 1.0, one
    print(f"[1c] single-class health: auroc={one['auroc']} ap="
          f"{one['average_precision']} recall={one['recall']:.3f} (no crash)")

    # -- 2. fully masked attribute -> nan, excluded from macro_f1 ---------- #
    yt = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 1, 1],
                   [0, 0, 0]], dtype=np.float64)
    yp = np.array([[0.9, 0.1, 0.9],
                   [0.1, 0.9, 0.1],
                   [0.8, 0.7, 0.8],
                   [0.2, 0.3, 0.2]], dtype=np.float64)
    mk = np.ones_like(yt)
    mk[:, 2] = 0.0  # yellow_head never supervised (the DB1 situation)
    thr = np.array([0.5, 0.5, 0.5])
    m = disease_attribute_metrics(yt, yp, mk, thr, attrs)
    assert math.isnan(m["yellow_head"]["f1"]), m["yellow_head"]
    assert math.isnan(m["yellow_head"]["auroc"]), m["yellow_head"]
    assert math.isnan(m["yellow_head"]["ece"]), m["yellow_head"]
    assert m["yellow_head"]["support_supervised"] == 0, m["yellow_head"]
    assert m["black_gill"]["f1"] == 1.0 and m["wssv"]["f1"] == 1.0, m
    assert m["macro_f1"] == 1.0, m["macro_f1"]          # nan attr excluded
    assert m["macro_map"] == 1.0, m["macro_map"]
    assert m["micro_f1"] == 1.0, m["micro_f1"]
    assert m["hamming_loss"] == 0.0, m["hamming_loss"]
    assert m["n_supervised"] == 8, m["n_supervised"]     # 4 rows x 2 attrs
    assert math.isnan(m["exact_match_ratio"]), m["exact_match_ratio"]
    print(f"[2] masked attr: yellow_head.f1={m['yellow_head']['f1']} "
          f"macro_f1={m['macro_f1']:.3f} (2 of 3 attrs) "
          f"n_supervised={m['n_supervised']} emr={m['exact_match_ratio']}")

    # -- 3. exact_match_ratio counts only fully supervised rows ------------ #
    yt3 = np.array([[1, 0, 1],    # fully supervised, all correct
                    [1, 1, 0],    # fully supervised, wrong on wssv
                    [1, 0, 1],    # partial (yellow_head unknown), correct
                    [0, 1, 1]],   # partial (yellow_head unknown), correct
                   dtype=np.float64)
    yp3 = np.array([[0.9, 0.1, 0.9],
                    [0.9, 0.1, 0.1],   # wssv predicted 0 but label is 1
                    [0.9, 0.1, 0.9],
                    [0.1, 0.9, 0.9]],
                   dtype=np.float64)
    mk3 = np.ones_like(yt3)
    mk3[2, 2] = 0.0
    mk3[3, 2] = 0.0
    m3 = disease_attribute_metrics(yt3, yp3, mk3, thr, attrs)
    assert m3["n_rows_fully_supervised"] == 2, m3["n_rows_fully_supervised"]
    assert m3["exact_match_ratio"] == 0.5, m3["exact_match_ratio"]
    # naive "all rows" EMR would be 3/4 = 0.75; masked rows must not inflate it
    assert m3["exact_match_ratio"] != 0.75
    # only one supervised entry is wrong, out of 3+3+2+2 = 10
    assert abs(m3["hamming_loss"] - 0.1) < 1e-12, m3["hamming_loss"]
    print(f"[3] emr={m3['exact_match_ratio']:.3f} over "
          f"{m3['n_rows_fully_supervised']}/4 fully-supervised rows "
          f"(naive all-rows value would be 0.750); "
          f"hamming={m3['hamming_loss']:.3f}")

    # -- 4. ECE of a perfectly calibrated synthetic set is ~0 -------------- #
    n_cal = 200_000
    p_cal = rng.uniform(0.0, 1.0, size=n_cal)
    y_cal = (rng.uniform(0.0, 1.0, size=n_cal) < p_cal).astype(np.float64)
    ece_good = expected_calibration_error(y_cal, p_cal, n_bins=15)
    assert ece_good < 0.01, ece_good
    # a deliberately miscalibrated (overconfident) version must score much worse
    p_bad = np.clip(p_cal * 1.6 - 0.3, 0.0, 1.0)
    ece_bad = expected_calibration_error(y_cal, p_bad, n_bins=15)
    assert ece_bad > 10 * ece_good, (ece_good, ece_bad)
    assert math.isnan(expected_calibration_error(np.array([]), np.array([])))
    print(f"[4] ECE calibrated={ece_good:.5f} (<0.01), "
          f"miscalibrated={ece_bad:.5f}, empty=nan")

    # -- 5. select_thresholds recovers a known optimal threshold ----------- #
    # Separable design: negatives in [0.00, 0.30), positives in [0.70, 1.00].
    # Every threshold inside the (0.30, 0.70] gap is optimal; the centre of that
    # plateau of tied candidates should land at the middle of the gap, ~0.50.
    n_each = 400
    neg = rng.uniform(0.00, 0.30, size=n_each)
    pos = rng.uniform(0.70, 1.00, size=n_each)
    yv = np.concatenate([np.zeros(n_each), np.ones(n_each)])
    pv = np.concatenate([neg, pos])
    yv2 = np.stack([yv, yv, yv], axis=1)
    pv2 = np.stack([pv, pv, pv], axis=1)
    mv2 = np.ones_like(yv2)
    mv2[:, 2] = 0.0            # third attribute entirely unsupervised
    t_sel = select_thresholds(yv2, pv2, mv2, objective="f1")
    assert t_sel.shape == (3,), t_sel.shape
    assert abs(t_sel[0] - 0.5) < 0.05, t_sel
    assert abs(t_sel[1] - 0.5) < 0.05, t_sel
    assert t_sel[2] == 0.5, t_sel        # unsupervised -> 0.5 fallback
    # the recovered thresholds must actually be perfect on this data
    mv2_all = np.ones_like(yv2)
    m5 = disease_attribute_metrics(yv2, pv2, mv2_all, t_sel, attrs)
    assert m5["macro_f1"] == 1.0, m5["macro_f1"]

    # a shifted margin must move the threshold with it (not a hard-coded 0.5)
    neg_s = rng.uniform(0.00, 0.10, size=n_each)
    pos_s = rng.uniform(0.30, 0.60, size=n_each)
    pv_s = np.concatenate([neg_s, pos_s])
    t_shift = select_thresholds(
        yv[:, None], pv_s[:, None], np.ones((2 * n_each, 1)), objective="f1"
    )
    assert 0.10 < t_shift[0] < 0.30, t_shift
    # no supervised positives at all -> 0.5 fallback
    t_nopos = select_thresholds(
        np.zeros((10, 1)), rng.uniform(size=(10, 1)), np.ones((10, 1))
    )
    assert t_nopos[0] == 0.5, t_nopos

    # REGRESSION: the maximum may be attained on two DISJOINT plateaus. Here F1
    # peaks at 0.6667 on both {0.32, 0.33} and {0.81, 0.82}; the median of the
    # tied set as a whole is 0.57, which sits in the valley between them and
    # scores only 0.5714. The returned threshold must be a genuine maximiser.
    y_dis = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    p_dis = np.array([0.31, 0.83, 0.93, 0.33, 0.88, 0.34, 0.80])
    t_dis = float(select_thresholds(
        y_dis[:, None], p_dis[:, None], np.ones((7, 1)), objective="f1"
    )[0])

    def _f1_at(t: float) -> float:
        yb = y_dis > 0.5
        pb = p_dis >= t
        return _prf(
            int(np.count_nonzero(yb & pb)),
            int(np.count_nonzero(~yb & pb)),
            int(np.count_nonzero(yb & ~pb)),
        )[2]

    best_f1 = max(_f1_at(float(t)) for t in DEFAULT_THRESHOLD_GRID)
    assert abs(best_f1 - 2 / 3) < 1e-12, best_f1
    assert _f1_at(t_dis) >= best_f1 - 1e-9, (t_dis, _f1_at(t_dis), best_f1)
    assert abs(t_dis - 0.57) > 1e-9, t_dis   # the old median-of-tied answer
    print(f"[5] select_thresholds: separable={t_sel[0]:.3f},{t_sel[1]:.3f} "
          f"(gap centre 0.5) unsup={t_sel[2]:.3f} "
          f"shifted-margin={t_shift[0]:.3f} (in 0.10-0.30) "
          f"no-positives={t_nopos[0]:.3f} "
          f"disjoint-plateau={t_dis:.2f} (f1={_f1_at(t_dis):.4f}, "
          f"max={best_f1:.4f}; median-of-tied 0.57 would score 0.5714)")

    # -- 6. degenerate inputs stay finite ---------------------------------- #
    empty_attr = disease_attribute_metrics(
        np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), thr, attrs
    )
    assert math.isnan(empty_attr["macro_f1"]), empty_attr["macro_f1"]
    assert math.isnan(empty_attr["hamming_loss"]), empty_attr["hamming_loss"]
    assert empty_attr["n_supervised"] == 0
    empty_bio = biomass_metrics(np.array([]), np.array([]), unit="cm")
    assert empty_bio["n"] == 0 and math.isnan(empty_bio["mae"]), empty_bio
    assert empty_bio["unit"] == "cm"
    const = biomass_metrics(np.full(5, 7.0), np.full(5, 7.0), unit="g")
    assert math.isnan(const["r2"]), const          # zero target variance
    assert math.isnan(const["pearson_r"]), const   # zero variance
    assert const["mae"] == 0.0, const
    tiny = biomass_metrics(
        np.array([0.0, 0.1, 10.0]), np.array([1.0, 0.2, 11.0]),
        unit="g", mape_floor=0.5,
    )
    assert tiny["n_mape_excluded"] == 2, tiny
    assert abs(tiny["mape"] - 10.0) < 1e-9, tiny   # only the 10.0 -> 11.0 pair
    print(f"[6] degenerate: macro_f1={empty_attr['macro_f1']} "
          f"empty_bio.n={empty_bio['n']} const.r2={const['r2']} "
          f"mape_excluded={tiny['n_mape_excluded']} mape={tiny['mape']:.1f}%")

    # -- 7. Bland-Altman ---------------------------------------------------- #
    ba_t = np.array([10.0, 20.0, 30.0, 40.0])
    ba_p = ba_t + np.array([1.0, 1.0, 3.0, 3.0])
    ba = bland_altman_stats(ba_t, ba_p)
    assert abs(ba["bias"] - 2.0) < 1e-12, ba
    expected_sd = float(np.std(np.array([1.0, 1.0, 3.0, 3.0]), ddof=1))
    assert abs(ba["sd_diff"] - expected_sd) < 1e-12, ba
    assert abs(ba["loa_lower"] - (2.0 - 1.96 * expected_sd)) < 1e-12, ba
    assert abs(ba["loa_upper"] - (2.0 + 1.96 * expected_sd)) < 1e-12, ba
    assert ba["n"] == 4, ba
    ba1 = bland_altman_stats(np.array([5.0]), np.array([6.0]))
    assert ba1["bias"] == 1.0 and math.isnan(ba1["sd_diff"]), ba1
    print(f"[7] bland-altman: bias={ba['bias']:.3f} sd={ba['sd_diff']:.3f} "
          f"loa=[{ba['loa_lower']:.3f}, {ba['loa_upper']:.3f}] n={ba['n']}")

    # -- 8. unknown is never a negative ------------------------------------ #
    # Identical predictions, but in the second call the true positives are
    # UNKNOWN rather than supervised. Masking must remove them, not score them
    # as negatives that the model got wrong.
    y8 = np.array([[1.0], [1.0], [0.0], [0.0]])
    p8 = np.array([[0.9], [0.9], [0.1], [0.1]])
    full8 = disease_attribute_metrics(y8, p8, np.ones_like(y8), np.array([0.5]),
                                      ("black_gill",))
    part8 = disease_attribute_metrics(
        y8, p8, np.array([[0.0], [0.0], [1.0], [1.0]]), np.array([0.5]),
        ("black_gill",),
    )
    assert full8["black_gill"]["support_pos"] == 2, full8
    assert part8["black_gill"]["support_pos"] == 0, part8
    assert part8["black_gill"]["support_supervised"] == 2, part8
    assert part8["hamming_loss"] == 0.0, part8   # the 2 kept negatives are right
    assert math.isnan(part8["macro_map"]), part8  # no positives -> no AP
    print(f"[8] masking: support_pos full={full8['black_gill']['support_pos']} "
          f"masked={part8['black_gill']['support_pos']} "
          f"hamming={part8['hamming_loss']:.3f} macro_map={part8['macro_map']}")

    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    _selftest()
