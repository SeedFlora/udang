"""Select the best model per role using VALIDATION metrics only.

Section 18 of the specification forbids selecting a model on test performance.
This script therefore reads `best_val_metric` from the experiment registry --
the value recorded at checkpoint time from the validation split -- and never
opens a test prediction file.

Writes outputs/metrics/model_selection.json, consumed by scripts/run_e9_seeds.py.

Usage:
    python scripts/select_models.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import METRICS  # noqa: E402

# role -> (predicate over the registry row, human description)
#
# E4 and E5 are deliberately SEPARATE roles rather than one "best biomass" role.
# Ranking them against each other on the recorded validation score would be
# invalid if the two were scored at different granularities -- aggregating several
# views averages noise away, so a multi-view model posts a lower MAE by
# construction. (_validate now scores both at the individual level, which removes
# the granularity mismatch, but keeping the roles separate means E9 repeats the
# best of EACH family and the head-to-head comparison is made where it belongs:
# in evaluate.py's paired bootstrap over shared individuals.)
ROLES = {
    "disease_single": (lambda r: str(r["experiment"]) == "E1",
                       "best single-dataset disease model"),
    "disease_pooled": (lambda r: str(r["experiment"]) == "E2",
                       "best pooled hierarchical disease model"),
    "biomass_single_image": (lambda r: str(r["experiment"]) == "E4",
                             "best single-image biomass model"),
    "biomass_multiview": (lambda r: str(r["experiment"]) == "E5",
                          "best multi-view biomass model"),
    # Multitask role is E6 only: the genuine shared-encoder multi-task runs. E7 is
    # domain generalization, and its cross-dataset variants are single-fold with a
    # validation metric that is not comparable to E6's, so including them here would
    # both mis-select and break E9 (their fold 1 does not exist).
    "multitask": (lambda r: str(r["experiment"]) == "E6",
                  "best shared-encoder multi-task model"),
    "lightweight": (lambda r: str(r["backbone"]) == "mobilenetv3_large",
                    "best lightweight model"),
}


def main() -> int:
    reg_path = METRICS / "experiment_registry.csv"
    if not reg_path.exists():
        print("BLOCKER: experiment_registry.csv not found.")
        return 2
    reg = pd.read_csv(reg_path)
    done = reg[reg["status"] == "completed"].copy()
    if done.empty:
        print("BLOCKER: no completed runs.")
        return 2
    done["best_val_metric"] = pd.to_numeric(done["best_val_metric"], errors="coerce")

    selected: dict[str, str] = {}
    detail: dict[str, dict] = {}
    for role, (pred, desc) in ROLES.items():
        sub = done[done.apply(pred, axis=1)]
        sub = sub[sub["best_val_metric"].notna()]
        if sub.empty:
            print(f"  {role:16s} -> (no completed runs)")
            continue
        # average the validation metric across folds, then take the best config
        agg = (sub.groupby("run_name")["best_val_metric"].mean().reset_index())
        agg["config"] = agg["run_name"].str.rsplit("_f", n=1).str[0]
        by_cfg = agg.groupby("config")["best_val_metric"].agg(["mean", "count"])
        by_cfg = by_cfg.sort_values("mean", ascending=False)
        best = by_cfg.index[0]
        selected[role] = best
        detail[role] = {
            "description": desc,
            "selected": best,
            "val_metric_mean": float(by_cfg.iloc[0]["mean"]),
            "n_folds": int(by_cfg.iloc[0]["count"]),
            "ranking": [{"config": c, "val_metric_mean": float(v["mean"]),
                         "n_folds": int(v["count"])}
                        for c, v in by_cfg.iterrows()],
        }
        print(f"  {role:16s} -> {best}  (val={by_cfg.iloc[0]['mean']:.4f}, "
              f"{int(by_cfg.iloc[0]['count'])} folds)")

    out = {
        "criterion": "mean best_val_metric across folds, recorded at checkpoint time "
                     "from the validation split. No test metric was consulted.",
        "selected": selected,
        "detail": detail,
    }
    (METRICS / "model_selection.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> outputs/metrics/model_selection.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
