"""Figure generation for disease and biomass results.

All figures are written from SAVED PREDICTIONS, so a figure can never disagree
with the reported numbers. No absolute filesystem path is ever drawn onto a
figure -- only basenames and record ids.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc, confusion_matrix, precision_recall_curve, roc_curve,
)

from src.data.labels import DISEASE_ATTRS  # noqa: E402

DPI = 130


def _save(fig, out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out.name


# --------------------------------------------------------------------------- #
# disease
# --------------------------------------------------------------------------- #
def plot_health_confusion(pred: pd.DataFrame, out: Path, title: str = "",
                          threshold: float = 0.5) -> str:
    y = pred["y_health"].values.astype(int)
    p = (pred["p_health"].values >= threshold).astype(int)
    cm = confusion_matrix(y, p, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1], ["healthy", "diseased"])
    ax.set_yticks([0, 1], ["healthy", "diseased"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Health confusion matrix\n{title}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out)


def plot_pr_roc(pred: pd.DataFrame, out_pr: Path, out_roc: Path,
                title: str = "") -> tuple[str, str]:
    """Precision-recall and ROC for the health head and each supervised attribute."""
    series = [("health (diseased)", pred["y_health"].values, pred["p_health"].values)]
    for a in DISEASE_ATTRS:
        m = pred[f"m_{a}"].values > 0
        if m.sum() > 5 and len(np.unique(pred[f"y_{a}"].values[m])) > 1:
            series.append((a, pred[f"y_{a}"].values[m], pred[f"p_{a}"].values[m]))

    fig, ax = plt.subplots(figsize=(5, 4.2))
    for name, y, p in series:
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=f"{name} (AP={auc(rc, pr):.3f})", lw=1.6)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(f"Precision-Recall\n{title}", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    a1 = _save(fig, out_pr)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    for name, y, p in series:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", lw=1.6)
    ax.plot([0, 1], [0, 1], "k--", lw=.8, alpha=.5)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC\n{title}", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    a2 = _save(fig, out_roc)
    return a1, a2


def plot_reliability(pred: pd.DataFrame, out: Path, n_bins: int = 12,
                     title: str = "") -> str:
    """Reliability diagram for the health head and each supervised attribute."""
    fig, axes = plt.subplots(1, 1 + len(DISEASE_ATTRS),
                             figsize=(3.1 * (1 + len(DISEASE_ATTRS)), 3.0))
    items = [("health", pred["y_health"].values, pred["p_health"].values,
              np.ones(len(pred), bool))]
    for a in DISEASE_ATTRS:
        items.append((a, pred[f"y_{a}"].values, pred[f"p_{a}"].values,
                      pred[f"m_{a}"].values > 0))

    for ax, (name, y, p, m) in zip(np.atleast_1d(axes), items):
        y, p = y[m], p[m]
        if len(y) < 5:
            ax.text(.5, .5, f"{name}\nno supervised data", ha="center", va="center",
                    fontsize=8)
            ax.set_axis_off()
            continue
        edges = np.linspace(0, 1, n_bins + 1)
        xs, ys, ws = [], [], []
        for i in range(n_bins):
            sel = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= 1)
            if sel.sum() > 0:
                xs.append(p[sel].mean()); ys.append(y[sel].mean()); ws.append(sel.sum())
        ax.plot([0, 1], [0, 1], "k--", lw=.8, alpha=.6)
        ax.plot(xs, ys, "o-", lw=1.4, ms=4)
        ax.set_title(f"{name} (n={len(y)})", fontsize=8)
        ax.set_xlabel("predicted probability", fontsize=7)
        ax.set_ylabel("observed frequency", fontsize=7)
        ax.grid(alpha=.3)
    fig.suptitle(f"Reliability diagrams  {title}", fontsize=9)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# biomass
# --------------------------------------------------------------------------- #
def _individual_table(pred: pd.DataFrame) -> pd.DataFrame:
    d = pred.copy()
    d["pred_weight_g"] = np.exp(d["pred_log_weight"])
    return d.groupby("individual_id").agg(
        pred_weight_g=("pred_weight_g", "median"),
        pred_length_cm=("pred_length_cm", "median"),
        y_weight_g=("y_weight_g", "first"),
        y_length_cm=("y_length_cm", "first"),
        db=("db", "first"), n_views=("pred_weight_g", "count")).reset_index()


def plot_pred_vs_obs(pred: pd.DataFrame, out: Path, title: str = "") -> str:
    t = _individual_table(pred).dropna(subset=["y_weight_g", "pred_weight_g"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, (yc, pc, lab) in zip(axes, [("y_weight_g", "pred_weight_g", "weight (g)"),
                                        ("y_length_cm", "pred_length_cm", "length (cm)")]):
        s = t.dropna(subset=[yc, pc])
        if len(s) < 2:
            ax.set_axis_off(); continue
        for db, g in s.groupby("db"):
            ax.scatter(g[yc], g[pc], s=26, alpha=.75, label=str(db))
        lo = min(s[yc].min(), s[pc].min()); hi = max(s[yc].max(), s[pc].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=.9, alpha=.6)
        r = np.corrcoef(s[yc], s[pc])[0, 1] if len(s) > 2 else np.nan
        ax.set_xlabel(f"observed {lab}"); ax.set_ylabel(f"predicted {lab}")
        ax.set_title(f"{lab}  (n={len(s)} individuals, r={r:.3f})", fontsize=9)
        ax.legend(fontsize=7, title="held-out DB", title_fontsize=7)
        ax.grid(alpha=.3)
    fig.suptitle(f"Predicted vs observed, individual level  {title}", fontsize=9)
    return _save(fig, out)


def plot_residuals_and_bland_altman(pred: pd.DataFrame, out: Path,
                                    title: str = "") -> str:
    t = _individual_table(pred).dropna(subset=["y_weight_g", "pred_weight_g"])
    if len(t) < 3:
        fig, ax = plt.subplots(); ax.text(.5, .5, "insufficient data", ha="center")
        return _save(fig, out)
    resid = t["pred_weight_g"] - t["y_weight_g"]
    mean_wt = (t["pred_weight_g"] + t["y_weight_g"]) / 2
    bias, sd = resid.mean(), resid.std(ddof=1)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    axes[0].scatter(t["y_weight_g"], resid, s=24, alpha=.75)
    axes[0].axhline(0, color="k", lw=.9, ls="--")
    axes[0].set_xlabel("true weight (g)"); axes[0].set_ylabel("residual (g)")
    axes[0].set_title("Error vs true weight", fontsize=9); axes[0].grid(alpha=.3)

    s = t.dropna(subset=["y_length_cm"])
    if len(s) > 2:
        axes[1].scatter(s["y_length_cm"], s["pred_weight_g"] - s["y_weight_g"],
                        s=24, alpha=.75, color="tab:orange")
        axes[1].axhline(0, color="k", lw=.9, ls="--")
    axes[1].set_xlabel("true length (cm)"); axes[1].set_ylabel("weight residual (g)")
    axes[1].set_title("Error vs true length", fontsize=9); axes[1].grid(alpha=.3)

    axes[2].scatter(mean_wt, resid, s=24, alpha=.75, color="tab:green")
    axes[2].axhline(bias, color="tab:red", lw=1.2,
                    label=f"bias={bias:.2f} g")
    axes[2].axhline(bias + 1.96 * sd, color="tab:red", ls="--", lw=.9,
                    label=f"+1.96sd={bias + 1.96 * sd:.2f}")
    axes[2].axhline(bias - 1.96 * sd, color="tab:red", ls="--", lw=.9,
                    label=f"-1.96sd={bias - 1.96 * sd:.2f}")
    axes[2].set_xlabel("mean of predicted and true (g)")
    axes[2].set_ylabel("difference (g)")
    axes[2].set_title("Bland-Altman", fontsize=9)
    axes[2].legend(fontsize=7); axes[2].grid(alpha=.3)
    fig.suptitle(f"Biomass error structure  {title}", fontsize=9)
    return _save(fig, out)


def plot_by_db(pred: pd.DataFrame, out: Path, title: str = "") -> str:
    t = _individual_table(pred).dropna(subset=["y_weight_g", "pred_weight_g"])
    if not len(t):
        fig, ax = plt.subplots(); ax.text(.5, .5, "no data", ha="center")
        return _save(fig, out)
    t["abs_err"] = (t["pred_weight_g"] - t["y_weight_g"]).abs()
    grp = t.groupby("db")["abs_err"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    labels = list(grp.groups)
    # Set tick labels manually rather than via boxplot's label kwarg, which
    # matplotlib renamed (labels -> tick_labels) across versions.
    ax.boxplot([grp.get_group(k).values for k in labels])
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([str(k) for k in labels])
    for i, k in enumerate(labels, 1):
        v = grp.get_group(k)
        ax.scatter(np.random.default_rng(0).normal(i, .05, len(v)), v,
                   s=16, alpha=.55)
    ax.set_ylabel("absolute weight error (g)")
    ax.set_xlabel("held-out DB")
    ax.set_title(f"Individual-level error by held-out DB  {title}", fontsize=9)
    ax.grid(alpha=.3, axis="y")
    return _save(fig, out)


def plot_views_curve(view_results: pd.DataFrame, out: Path, title: str = "") -> str:
    """MAE as a function of the number of views available at test time."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for model, g in view_results.groupby("model"):
        g = g.sort_values("n_views")
        ax.errorbar(g["n_views"], g["mae"], yerr=g.get("mae_std"),
                    marker="o", capsize=3, lw=1.5, label=model)
    ax.set_xlabel("number of views used at test time")
    ax.set_ylabel("individual-level weight MAE (g)")
    ax.set_title(f"Effect of view count  {title}", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    return _save(fig, out)


def plot_training_curves(history: list[dict], out: Path, title: str = "") -> str:
    if not history:
        fig, ax = plt.subplots(); ax.text(.5, .5, "no history", ha="center")
        return _save(fig, out)
    h = pd.DataFrame(history)
    loss_cols = [c for c in h.columns if c.endswith("_loss")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for c in loss_cols:
        axes[0].plot(h["epoch"], h[c], marker="o", ms=3, label=c)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Training losses", fontsize=9)
    axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)

    if "val_primary" in h:
        axes[1].plot(h["epoch"], h["val_primary"], marker="o", ms=3, color="tab:green")
        best = h["val_primary"].idxmax()
        axes[1].axvline(h.loc[best, "epoch"], color="tab:red", ls="--", lw=1,
                        label=f"best epoch {int(h.loc[best, 'epoch'])}")
        axes[1].legend(fontsize=7)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation score")
    axes[1].set_title("Validation (model selection)", fontsize=9); axes[1].grid(alpha=.3)

    gcols = [c for c in h.columns if c.startswith("grad_norm_")]
    for c in gcols:
        axes[2].plot(h["epoch"], h[c], marker="o", ms=3, label=c)
    if "lr" in h:
        ax2 = axes[2].twinx()
        ax2.plot(h["epoch"], h["lr"], color="gray", ls=":", lw=1, label="lr")
        ax2.set_ylabel("learning rate", fontsize=7)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("gradient norm")
    axes[2].set_title("Per-task gradient norms", fontsize=9)
    axes[2].legend(fontsize=7); axes[2].grid(alpha=.3)
    fig.suptitle(title, fontsize=9)
    return _save(fig, out)


def plot_results_bar(summary: pd.DataFrame, metric: str, out: Path,
                     title: str = "", lower_is_better: bool = False) -> str:
    col = f"{metric}_mean"
    if col not in summary.columns:
        fig, ax = plt.subplots(); ax.text(.5, .5, f"{col} absent", ha="center")
        return _save(fig, out)
    s = summary.dropna(subset=[col]).sort_values(col, ascending=lower_is_better)
    fig, ax = plt.subplots(figsize=(8.5, max(3.0, .38 * len(s) + 1.2)))
    err = s.get(f"{metric}_std")
    ax.barh(s["config_name"], s[col], xerr=err, capsize=3, alpha=.85)
    ax.set_xlabel(metric + (" (lower is better)" if lower_is_better else ""))
    ax.set_title(title or metric, fontsize=9)
    ax.grid(alpha=.3, axis="x")
    ax.invert_yaxis()
    return _save(fig, out)
