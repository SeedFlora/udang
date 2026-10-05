"""Grouped (cluster) bootstrap confidence intervals and paired model comparison.

Why grouping is mandatory here
------------------------------
The shrimp datasets contain multiple photographs of the *same* individual and,
after offline augmentation, multiple derived copies of the *same* original
image. Those rows are strongly positively correlated: they share the animal,
the pond, the lighting rig and the acquisition session. Treating them as
independent observations inflates the effective sample size and produces
confidence intervals that are far too narrow -- the classic cluster-sampling
variance underestimation.

Every resampling routine in this module therefore resamples *groups* (clusters)
with replacement and then concatenates **all** rows belonging to each drawn
group. The number of drawn groups always equals the number of observed groups,
so the replicate design mirrors the original sampling design. Replicate row
counts vary between replicates (groups differ in size), which is the correct
behaviour for an unbalanced cluster bootstrap.

The self-test at the bottom of this file empirically demonstrates that ignoring
the grouping shrinks the interval by a large factor on clustered data.

References
----------
Davison, A. C. and Hinkley, D. V. (1997). *Bootstrap Methods and Their
Application*, chapter 3 (resampling for dependent / clustered data).
Field, C. A. and Welsh, A. H. (2007). Bootstrapping clustered data.
*Journal of the Royal Statistical Society: Series B*, 69(3), 369-390.
"""
from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "grouped_bootstrap_ci",
    "paired_grouped_bootstrap",
    "wilcoxon_across_folds",
    "summarize_seeds",
]

MetricFn = Callable[[np.ndarray, np.ndarray], float]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _as_1d(name: str, arr: np.ndarray | Sequence[float]) -> np.ndarray:
    """Coerce input to a 1-D numpy array, preserving dtype where possible.

    Parameters
    ----------
    name : str
        Argument name, used only for the error message.
    arr : array_like
        Input sequence.

    Returns
    -------
    numpy.ndarray
        1-D view/copy of ``arr``.

    Raises
    ------
    ValueError
        If ``arr`` is not 1-D after conversion.
    """
    out = np.asarray(arr)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    return out


def _group_row_indices(groups: np.ndarray) -> list[np.ndarray]:
    """Build the row-index array of every unique group.

    Uses a single stable argsort rather than one boolean scan per group, so the
    cost is ``O(n log n)`` instead of ``O(n * n_groups)``.

    Parameters
    ----------
    groups : numpy.ndarray
        Group / cluster identifier per row. Any hashable-sortable dtype.

    Returns
    -------
    list of numpy.ndarray
        ``out[g]`` holds the row indices belonging to the ``g``-th unique group.
        Groups are ordered by ``numpy.unique`` order (irrelevant for the
        bootstrap, but makes the result deterministic).
    """
    _, codes = np.unique(groups, return_inverse=True)
    codes = np.asarray(codes).ravel()
    n_groups = int(codes.max()) + 1 if codes.size else 0
    order = np.argsort(codes, kind="stable")
    # Boundaries of each code block inside the sorted order.
    edges = np.searchsorted(codes[order], np.arange(n_groups + 1))
    return [order[edges[g]:edges[g + 1]] for g in range(n_groups)]


def _safe_metric(
    metric_fn: MetricFn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Evaluate ``metric_fn``, mapping failures and non-finite results to nan.

    Degenerate bootstrap replicates are normal: a resample can legitimately
    contain a single class, which makes ROC-AUC undefined and makes several
    sklearn metrics raise. Such replicates are dropped from the percentile
    interval rather than crashing the run.

    Warnings raised by ``metric_fn`` are suppressed. This is not cosmetic: a
    metric that *warns* instead of raising on a degenerate replicate (notably
    :func:`sklearn.metrics.roc_auc_score`, which emits an
    ``UndefinedMetricWarning`` and returns nan when a resample is single-class)
    would otherwise emit one warning per replicate. On a rare attribute such as
    WSSV that is hundreds of lines of stderr per call and thousands per
    evaluation sweep, which drowns out genuine warnings. The structured
    replacement is the ``n_valid`` field of the returned dict: compare it with
    ``n_boot`` to see how many replicates were degenerate.

    Parameters
    ----------
    metric_fn : callable
        Maps ``(y_true, y_pred)`` to a scalar.
    y_true, y_pred : numpy.ndarray
        Aligned arrays for this replicate.

    Returns
    -------
    float
        The metric, or ``nan`` if it raised, warned its way to a non-finite
        result, or was not finite.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = float(metric_fn(y_true, y_pred))
    except Exception:  # noqa: BLE001 - degenerate replicates must not abort the run
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _percentile_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Two-sided percentile interval of a replicate distribution.

    Parameters
    ----------
    values : numpy.ndarray
        Finite replicate statistics.
    alpha : float
        Total tail mass, e.g. ``0.05`` for a 95 percent interval.

    Returns
    -------
    tuple of float
        ``(lo, hi)``, or ``(nan, nan)`` when ``values`` is empty.
    """
    if values.size == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(values, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _draw_replicate_rows(
    group_rows: list[np.ndarray],
    n_groups: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one cluster-bootstrap replicate as a row-index array.

    Parameters
    ----------
    group_rows : list of numpy.ndarray
        Row indices per group, from :func:`_group_row_indices`.
    n_groups : int
        Number of groups to draw (equals ``len(group_rows)``).
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    numpy.ndarray
        Concatenated row indices of the drawn groups. Rows of a group that is
        drawn twice appear twice, as required.
    """
    picked = rng.integers(0, n_groups, size=n_groups)
    return np.concatenate([group_rows[g] for g in picked])


def _validate_alpha(alpha: float) -> None:
    """Raise if ``alpha`` is outside the open interval (0, 1)."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def grouped_bootstrap_ci(
    metric_fn: MetricFn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Cluster-bootstrap confidence interval for an arbitrary metric.

    Unique group identifiers are resampled **with replacement**; every row of a
    drawn group enters the replicate. The metric is then recomputed on the
    concatenated rows. This propagates within-group correlation into the
    interval width, which a naive row-level bootstrap cannot do.

    Parameters
    ----------
    metric_fn : callable
        ``metric_fn(y_true, y_pred) -> float``. May raise on degenerate
        replicates; such replicates are skipped.
    y_true : numpy.ndarray, shape (n,)
        Ground-truth values.
    y_pred : numpy.ndarray, shape (n,)
        Model outputs (labels, probabilities or regression values -- whatever
        ``metric_fn`` expects).
    groups : numpy.ndarray, shape (n,)
        Cluster identifier per row, e.g. shrimp individual id or original-image
        id. Rows sharing an id are never split across a replicate.
    n_boot : int, default 2000
        Number of bootstrap replicates.
    alpha : float, default 0.05
        Total tail mass; ``0.05`` yields a 95 percent percentile interval.
    seed : int, default 42
        Seed for :class:`numpy.random.Generator`; identical seeds reproduce
        identical intervals.

    Returns
    -------
    dict
        ``point`` -- metric on the full, unresampled data.
        ``lo``, ``hi`` -- percentile interval bounds.
        ``mean``, ``std`` -- mean and standard deviation of valid replicates.
        ``n_boot`` -- replicates requested.
        ``n_groups`` -- number of unique groups.
        ``n_valid`` -- replicates that produced a finite metric. Warnings from
        ``metric_fn`` are suppressed, so ``n_valid < n_boot`` is the signal that
        replicates were degenerate; a large shortfall means the interval rests
        on a filtered subsample and should be treated with suspicion.

    Raises
    ------
    ValueError
        If the inputs have mismatched lengths, are not 1-D, ``n_boot`` is
        negative, or ``alpha`` is outside ``(0, 1)``.

    Notes
    -----
    All-nan or empty inputs return ``nan`` for every statistic instead of
    raising, so a batch with no supervised rows for some label does not abort an
    evaluation sweep.
    """
    _validate_alpha(alpha)
    if n_boot < 0:
        raise ValueError(f"n_boot must be non-negative, got {n_boot}")

    y_true = _as_1d("y_true", y_true)
    y_pred = _as_1d("y_pred", y_pred)
    groups = _as_1d("groups", groups)
    if not (y_true.shape[0] == y_pred.shape[0] == groups.shape[0]):
        raise ValueError(
            "y_true, y_pred and groups must have equal length, got "
            f"{y_true.shape[0]}, {y_pred.shape[0]}, {groups.shape[0]}"
        )

    n = y_true.shape[0]
    if n == 0:
        return {
            "point": float("nan"), "lo": float("nan"), "hi": float("nan"),
            "mean": float("nan"), "std": float("nan"),
            "n_boot": int(n_boot), "n_groups": 0, "n_valid": 0,
        }

    point = _safe_metric(metric_fn, y_true, y_pred)
    group_rows = _group_row_indices(groups)
    n_groups = len(group_rows)

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        rows = _draw_replicate_rows(group_rows, n_groups, rng)
        replicates[b] = _safe_metric(metric_fn, y_true[rows], y_pred[rows])

    valid = replicates[np.isfinite(replicates)]
    lo, hi = _percentile_ci(valid, alpha)
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "mean": float(valid.mean()) if valid.size else float("nan"),
        "std": float(valid.std(ddof=1)) if valid.size > 1 else float("nan"),
        "n_boot": int(n_boot),
        "n_groups": int(n_groups),
        "n_valid": int(valid.size),
    }


def paired_grouped_bootstrap(
    metric_fn: MetricFn,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Paired cluster bootstrap for the difference between two models.

    Both models are scored on the **same** resampled rows in every replicate.
    Pairing cancels the replicate-to-replicate variation that is common to both
    models, so the interval on the difference is much tighter -- and far more
    honest -- than differencing two independently bootstrapped intervals.

    Parameters
    ----------
    metric_fn : callable
        ``metric_fn(y_true, y_pred) -> float``. Higher is assumed better only
        for the sign convention of ``diff``; nothing else depends on it.
    y_true : numpy.ndarray, shape (n,)
        Ground-truth values.
    pred_a : numpy.ndarray, shape (n,)
        Predictions of model A (the candidate).
    pred_b : numpy.ndarray, shape (n,)
        Predictions of model B (the baseline).
    groups : numpy.ndarray, shape (n,)
        Cluster identifier per row.
    n_boot : int, default 2000
        Number of bootstrap replicates.
    alpha : float, default 0.05
        Total tail mass for the interval on the difference.
    seed : int, default 42
        Seed for :class:`numpy.random.Generator`.

    Returns
    -------
    dict
        ``metric_a``, ``metric_b`` -- point metrics on the full data.
        ``diff`` -- ``metric_a - metric_b`` on the full data.
        ``lo``, ``hi`` -- percentile interval for the difference. An interval
        excluding 0 is the evidence that the models differ.
        ``p_two_sided`` -- bootstrap two-sided p-value,
        ``2 * min(frac(diff <= 0), frac(diff >= 0))`` clipped to ``[0, 1]``,
        computed over the replicate differences.
        ``n_boot``, ``n_groups``.

    Raises
    ------
    ValueError
        If the inputs have mismatched lengths, are not 1-D, ``n_boot`` is
        negative, or ``alpha`` is outside ``(0, 1)``.

    Notes
    -----
    A replicate is discarded only if *either* model yields a non-finite metric,
    which keeps the difference well defined and the pairing intact.
    """
    _validate_alpha(alpha)
    if n_boot < 0:
        raise ValueError(f"n_boot must be non-negative, got {n_boot}")

    y_true = _as_1d("y_true", y_true)
    pred_a = _as_1d("pred_a", pred_a)
    pred_b = _as_1d("pred_b", pred_b)
    groups = _as_1d("groups", groups)
    lengths = {y_true.shape[0], pred_a.shape[0], pred_b.shape[0], groups.shape[0]}
    if len(lengths) != 1:
        raise ValueError(
            "y_true, pred_a, pred_b and groups must have equal length, got "
            f"{y_true.shape[0]}, {pred_a.shape[0]}, {pred_b.shape[0]}, {groups.shape[0]}"
        )

    n = y_true.shape[0]
    if n == 0:
        return {
            "metric_a": float("nan"), "metric_b": float("nan"), "diff": float("nan"),
            "lo": float("nan"), "hi": float("nan"), "p_two_sided": float("nan"),
            "n_boot": int(n_boot), "n_groups": 0,
        }

    metric_a = _safe_metric(metric_fn, y_true, pred_a)
    metric_b = _safe_metric(metric_fn, y_true, pred_b)
    group_rows = _group_row_indices(groups)
    n_groups = len(group_rows)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        rows = _draw_replicate_rows(group_rows, n_groups, rng)
        yt = y_true[rows]
        m_a = _safe_metric(metric_fn, yt, pred_a[rows])
        m_b = _safe_metric(metric_fn, yt, pred_b[rows])
        diffs[i] = m_a - m_b  # nan propagates if either side failed

    valid = diffs[np.isfinite(diffs)]
    lo, hi = _percentile_ci(valid, alpha)

    if valid.size:
        frac_le = float(np.mean(valid <= 0.0))
        frac_ge = float(np.mean(valid >= 0.0))
        p_two_sided = float(np.clip(2.0 * min(frac_le, frac_ge), 0.0, 1.0))
    else:
        p_two_sided = float("nan")

    return {
        "metric_a": metric_a,
        "metric_b": metric_b,
        "diff": metric_a - metric_b,
        "lo": lo,
        "hi": hi,
        "p_two_sided": p_two_sided,
        "n_boot": int(n_boot),
        "n_groups": int(n_groups),
    }


def wilcoxon_across_folds(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired Wilcoxon signed-rank test across folds or seeds.

    Use this for the *outer* comparison -- one paired score per cross-validation
    fold or per training seed. It is the standard non-parametric companion to the
    bootstrap, which operates within a single evaluation set.

    Parameters
    ----------
    a : sequence of float
        Per-fold (or per-seed) scores of model A.
    b : sequence of float
        Per-fold scores of model B, aligned element-wise with ``a``.

    Returns
    -------
    dict
        ``statistic`` -- Wilcoxon statistic, or ``nan`` when the test cannot run.
        ``pvalue`` -- two-sided p-value, or ``nan``.
        ``n`` -- number of pairs supplied.
        ``median_diff`` -- median of ``a - b`` (``nan`` for empty input).
        ``note`` -- present **only** when the test was not run, explaining why.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different lengths.

    Notes
    -----
    Degenerate inputs never raise. With fewer than 3 pairs the signed-rank test
    has no power at any conventional level, and with all-zero differences the
    statistic is undefined; both cases return ``nan`` plus a ``note``.
    """
    arr_a = _as_1d("a", np.asarray(a, dtype=float))
    arr_b = _as_1d("b", np.asarray(b, dtype=float))
    if arr_a.shape[0] != arr_b.shape[0]:
        raise ValueError(
            f"a and b must have equal length, got {arr_a.shape[0]} and {arr_b.shape[0]}"
        )

    n = int(arr_a.shape[0])
    diff = arr_a - arr_b
    median_diff = float(np.median(diff)) if n else float("nan")
    base = {"statistic": float("nan"), "pvalue": float("nan"),
            "n": n, "median_diff": median_diff}

    if n < 3:
        return {**base, "note": f"n={n} < 3 pairs: signed-rank test not meaningful"}
    if not np.all(np.isfinite(diff)):
        return {**base, "note": "non-finite differences present: test not run"}
    if np.allclose(diff, 0.0):
        return {**base, "note": "all paired differences are zero: statistic undefined"}

    try:
        result = stats.wilcoxon(arr_a, arr_b)
    except ValueError as exc:
        return {**base, "note": f"scipy.stats.wilcoxon failed: {exc}"}

    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n": n,
        "median_diff": median_diff,
    }


def summarize_seeds(values: Sequence[float]) -> dict:
    """Summarize a handful of per-seed scores with a t-based 95 percent CI.

    Seed counts in this project are small (typically 3-5), so the normal
    approximation is not appropriate; the interval uses the Student-t quantile
    with ``n - 1`` degrees of freedom.

    Parameters
    ----------
    values : sequence of float
        One score per training seed.

    Returns
    -------
    dict
        ``mean``, ``std`` (sample std, ``ddof=1``), ``min``, ``max``, ``n``,
        ``ci95_lo``, ``ci95_hi``. With ``n < 2`` the standard deviation and both
        interval bounds are ``nan``; with ``n == 0`` every field except ``n`` is
        ``nan``.

    Notes
    -----
    A zero-variance sample (all seeds identical) yields a zero-width interval
    centred on the mean, which is the correct degenerate limit.
    """
    arr = _as_1d("values", np.asarray(values, dtype=float))
    n = int(arr.shape[0])
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "n": 0,
                "ci95_lo": float("nan"), "ci95_hi": float("nan")}

    mean = float(arr.mean())
    out = {
        "mean": mean,
        "std": float("nan"),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": n,
        "ci95_lo": float("nan"),
        "ci95_hi": float("nan"),
    }
    if n < 2:
        return out

    std = float(arr.std(ddof=1))
    out["std"] = std
    half_width = float(stats.t.ppf(0.975, df=n - 1)) * std / float(np.sqrt(n))
    out["ci95_lo"] = mean - half_width
    out["ci95_hi"] = mean + half_width
    return out


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plain accuracy, used as the self-test metric."""
    return float(np.mean(y_true == y_pred))


def _self_test() -> None:
    """Run all self-test assertions, printing a short report."""
    rng = np.random.default_rng(0)
    print("=" * 70)
    print("src.evaluation.bootstrap self-test")
    print("=" * 70)

    # -- 1. Identical predictions -> degenerate, zero-width interval ---------- #
    n_groups, per_group = 30, 8
    groups = np.repeat(np.arange(n_groups), per_group)
    y = rng.integers(0, 2, size=n_groups * per_group)
    res = grouped_bootstrap_ci(_accuracy, y, y.copy(), groups, n_boot=500, seed=42)
    width = res["hi"] - res["lo"]
    print(f"[1] identical preds : point={res['point']:.4f} "
          f"CI=[{res['lo']:.4f}, {res['hi']:.4f}] width={width:.2e} "
          f"n_groups={res['n_groups']} n_valid={res['n_valid']}")
    assert abs(res["point"] - 1.0) < 1e-12, "perfect predictions must score 1.0"
    assert width < 1e-9, f"identical predictions must give zero-width CI, got {width}"
    assert res["lo"] <= res["point"] <= res["hi"], "point must lie inside its CI"
    assert res["n_groups"] == n_groups and res["n_valid"] == 500

    # -- 2. Model A strictly better -> diff > 0 and CI excluding 0 ----------- #
    n_groups, per_group = 40, 10
    groups = np.repeat(np.arange(n_groups), per_group)
    n = n_groups * per_group
    y = rng.integers(0, 2, size=n)
    pred_a = y.copy()                                    # perfect
    flip = rng.random(n) < 0.30
    pred_b = np.where(flip, 1 - y, y)                    # ~70 percent accurate
    res = paired_grouped_bootstrap(_accuracy, y, pred_a, pred_b, groups,
                                   n_boot=1000, seed=42)
    print(f"[2] A strictly better: a={res['metric_a']:.4f} b={res['metric_b']:.4f} "
          f"diff={res['diff']:.4f} CI=[{res['lo']:.4f}, {res['hi']:.4f}] "
          f"p={res['p_two_sided']:.4f}")
    assert res["diff"] > 0.0, "model A must win"
    assert res["lo"] > 0.0, f"CI must exclude 0, got lo={res['lo']}"
    assert res["p_two_sided"] < 0.05, f"expected significant p, got {res['p_two_sided']}"
    assert res["n_groups"] == n_groups

    # -- 3. Identical models -> diff exactly 0, p == 1 ----------------------- #
    res = paired_grouped_bootstrap(_accuracy, y, pred_b, pred_b.copy(), groups,
                                   n_boot=1000, seed=42)
    print(f"[3] identical models : diff={res['diff']:.4f} "
          f"CI=[{res['lo']:.4f}, {res['hi']:.4f}] p={res['p_two_sided']:.4f}")
    assert res["diff"] == 0.0, f"identical models must tie, got {res['diff']}"
    assert res["lo"] == 0.0 and res["hi"] == 0.0, "difference must be identically 0"
    assert abs(res["p_two_sided"] - 1.0) < 1e-12, f"p must be 1, got {res['p_two_sided']}"

    # -- 4. Grouping widens the interval on clustered data ------------------- #
    # 40 clusters of 25 rows. Every row inside a cluster shares one outcome, so
    # the effective sample size is 40, not 1000. A row-level bootstrap cannot
    # see this and produces a badly over-confident interval.
    n_groups, per_group = 40, 25
    cluster_correct = rng.random(n_groups) < 0.5          # cluster-level outcome
    y_c = np.repeat(1, n_groups * per_group)
    pred_c = np.repeat(cluster_correct.astype(int), per_group)
    groups_c = np.repeat(np.arange(n_groups), per_group)
    ignored = np.arange(n_groups * per_group)             # one row == one "group"

    res_grouped = grouped_bootstrap_ci(_accuracy, y_c, pred_c, groups_c,
                                       n_boot=1000, seed=42)
    res_ignored = grouped_bootstrap_ci(_accuracy, y_c, pred_c, ignored,
                                       n_boot=1000, seed=42)
    w_grouped = res_grouped["hi"] - res_grouped["lo"]
    w_ignored = res_ignored["hi"] - res_ignored["lo"]
    print(f"[4] clustered data  : point={res_grouped['point']:.4f}")
    print(f"    grouped  CI=[{res_grouped['lo']:.4f}, {res_grouped['hi']:.4f}] "
          f"width={w_grouped:.4f} (n_groups={res_grouped['n_groups']})")
    print(f"    ignoring CI=[{res_ignored['lo']:.4f}, {res_ignored['hi']:.4f}] "
          f"width={w_ignored:.4f} (n_groups={res_ignored['n_groups']})")
    print(f"    width ratio grouped/ignored = {w_grouped / w_ignored:.2f}x")
    assert w_grouped > w_ignored, "grouped CI must be wider than the naive one"
    assert w_grouped > 3.0 * w_ignored, (
        f"grouped CI should be much wider under strong clustering, "
        f"got {w_grouped:.4f} vs {w_ignored:.4f}"
    )

    # -- 5. Reproducibility --------------------------------------------------- #
    r1 = grouped_bootstrap_ci(_accuracy, y_c, pred_c, groups_c, n_boot=300, seed=7)
    r2 = grouped_bootstrap_ci(_accuracy, y_c, pred_c, groups_c, n_boot=300, seed=7)
    r3 = grouped_bootstrap_ci(_accuracy, y_c, pred_c, groups_c, n_boot=300, seed=8)
    p1 = paired_grouped_bootstrap(_accuracy, y, pred_a, pred_b, groups_c[:n],
                                  n_boot=300, seed=7)
    p2 = paired_grouped_bootstrap(_accuracy, y, pred_a, pred_b, groups_c[:n],
                                  n_boot=300, seed=7)
    print(f"[5] reproducibility : seed7==seed7 -> {r1 == r2}, "
          f"paired seed7==seed7 -> {p1 == p2}, "
          f"seed7!=seed8 -> {r1 != r3}")
    assert r1 == r2, "same seed must reproduce identical grouped results"
    assert p1 == p2, "same seed must reproduce identical paired results"
    assert r1 != r3, "different seeds should not give byte-identical intervals"

    # -- 6. Wilcoxon across folds -------------------------------------------- #
    fold_a = [0.81, 0.83, 0.79, 0.85, 0.82]
    fold_b = [0.76, 0.78, 0.75, 0.80, 0.77]
    w = wilcoxon_across_folds(fold_a, fold_b)
    print(f"[6] wilcoxon        : stat={w['statistic']:.1f} p={w['pvalue']:.4f} "
          f"n={w['n']} median_diff={w['median_diff']:.4f}")
    assert w["n"] == 5 and w["median_diff"] > 0 and np.isfinite(w["pvalue"])
    assert "note" not in w

    w_small = wilcoxon_across_folds([0.8, 0.9], [0.7, 0.6])
    w_zero = wilcoxon_across_folds([0.8] * 5, [0.8] * 5)
    print(f"    n<3   -> pvalue={w_small['pvalue']} note={w_small['note']!r}")
    print(f"    zeros -> pvalue={w_zero['pvalue']} note={w_zero['note']!r}")
    assert np.isnan(w_small["pvalue"]) and "note" in w_small
    assert np.isnan(w_zero["pvalue"]) and "note" in w_zero

    # -- 7. summarize_seeds --------------------------------------------------- #
    s = summarize_seeds([0.81, 0.83, 0.79])
    print(f"[7] summarize_seeds : mean={s['mean']:.4f} std={s['std']:.4f} "
          f"CI=[{s['ci95_lo']:.4f}, {s['ci95_hi']:.4f}] n={s['n']}")
    assert abs(s["mean"] - 0.81) < 1e-9 and s["n"] == 3
    assert s["ci95_lo"] < s["mean"] < s["ci95_hi"]
    s1 = summarize_seeds([0.8])
    print(f"    n=1 -> std={s1['std']} ci=[{s1['ci95_lo']}, {s1['ci95_hi']}]")
    assert np.isnan(s1["std"]) and np.isnan(s1["ci95_lo"]) and np.isnan(s1["ci95_hi"])
    s0 = summarize_seeds([])
    assert s0["n"] == 0 and np.isnan(s0["mean"])

    # -- 8. Degenerate metric never crashes ----------------------------------- #
    def _explodes(_yt: np.ndarray, _yp: np.ndarray) -> float:
        raise RuntimeError("metric unavailable for this replicate")

    bad = grouped_bootstrap_ci(_explodes, y_c, pred_c, groups_c, n_boot=50, seed=1)
    empty = grouped_bootstrap_ci(_accuracy, np.array([]), np.array([]), np.array([]))
    print(f"[8] robustness      : failing metric -> n_valid={bad['n_valid']} "
          f"point={bad['point']}; empty input -> n_groups={empty['n_groups']}")
    assert bad["n_valid"] == 0 and np.isnan(bad["lo"]) and np.isnan(bad["point"])
    assert empty["n_groups"] == 0 and np.isnan(empty["point"])

    # -- 9. Degenerate replicates stay silent -------------------------------- #
    # A metric that WARNS (rather than raises) on a degenerate replicate must
    # not emit one warning per replicate; n_valid is the diagnostic instead.
    def _warns_on_single_class(yt_: np.ndarray, yp_: np.ndarray) -> float:
        if np.unique(yt_).size < 2:
            warnings.warn("only one class present", UserWarning, stacklevel=2)
            return float("nan")
        return float(np.mean(yt_ == yp_))

    n_g, per = 20, 5
    g_rare = np.repeat(np.arange(n_g), per)
    y_rare = np.zeros(n_g * per, dtype=int)
    y_rare[:per] = 1                      # a single positive cluster
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet = grouped_bootstrap_ci(_warns_on_single_class, y_rare, y_rare.copy(),
                                     g_rare, n_boot=400, seed=42)
    n_degenerate = quiet["n_boot"] - quiet["n_valid"]
    print(f"[9] warning silence : n_valid={quiet['n_valid']}/{quiet['n_boot']} "
          f"degenerate={n_degenerate} warnings_escaped={len(caught)}")
    assert n_degenerate > 0, "test is vacuous unless some replicates are degenerate"
    assert len(caught) == 0, (
        f"metric warnings must not escape _safe_metric, {len(caught)} did"
    )

    print("-" * 70)
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    _self_test()
