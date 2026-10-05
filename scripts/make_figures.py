"""Generate all result figures from saved predictions.

Usage:
    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import (  # noqa: E402
    CHECKPOINTS, FIGURES, METRICS, PREDICTIONS, ensure_dirs,
)
from src.visualization.figures import (  # noqa: E402
    plot_by_db, plot_health_confusion, plot_pr_roc, plot_pred_vs_obs,
    plot_reliability, plot_residuals_and_bland_altman, plot_results_bar,
    plot_training_curves,
)


def group_predictions(pattern: str, exclude: str | None = None) -> dict[str, pd.DataFrame]:
    out: dict[str, list[pd.DataFrame]] = {}
    for f in sorted(PREDICTIONS.glob(pattern)):
        if exclude and exclude in f.name:
            continue
        out.setdefault(f.name.rsplit("_f", 1)[0], []).append(pd.read_parquet(f))
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def main() -> int:
    ensure_dirs()
    made: list[str] = []

    dis = group_predictions("*_test.parquet", exclude="biomass")
    dis = {k: v for k, v in dis.items() if "p_health" in v.columns}
    for tag, df in dis.items():
        print(f"disease figures: {tag} ({len(df)} predictions)")
        made.append(plot_health_confusion(
            df, FIGURES / f"{tag}_confusion.png", title=tag))
        a, b = plot_pr_roc(df, FIGURES / f"{tag}_pr.png",
                           FIGURES / f"{tag}_roc.png", title=tag)
        made += [a, b]
        made.append(plot_reliability(df, FIGURES / f"{tag}_reliability.png", title=tag))

    bio = group_predictions("*_biomass_test.parquet")
    for tag, df in bio.items():
        print(f"biomass figures: {tag} ({len(df)} predictions)")
        made.append(plot_pred_vs_obs(df, FIGURES / f"{tag}_pred_vs_obs.png", title=tag))
        made.append(plot_residuals_and_bland_altman(
            df, FIGURES / f"{tag}_residuals_bland_altman.png", title=tag))
        made.append(plot_by_db(df, FIGURES / f"{tag}_error_by_db.png", title=tag))

    # training curves
    for h in sorted(CHECKPOINTS.glob("*_history.json")):
        try:
            hist = json.loads(h.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        made.append(plot_training_curves(
            hist, FIGURES / f"{h.stem.replace('_history', '')}_curves.png",
            title=h.stem.replace("_history", "")))

    # cross-model summary bars
    rs = METRICS / "results_summary.csv"
    if rs.exists():
        s = pd.read_csv(rs)
        for metric, lower in (("health_bal_acc", False), ("attr_macro_f1", False),
                              ("ind_weight_mae_g", True)):
            if f"{metric}_mean" in s.columns:
                made.append(plot_results_bar(
                    s, metric, FIGURES / f"summary_{metric}.png",
                    title=f"{metric} across configurations (mean +/- sd over folds)",
                    lower_is_better=lower))

    made = [m for m in made if m]
    print(f"\nwrote {len(made)} figures to outputs/figures/")
    for m in sorted(set(made)):
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
