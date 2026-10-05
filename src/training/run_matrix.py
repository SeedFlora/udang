"""Drive a subset of the experiment matrix and aggregate results.

Shared by scripts/train_disease.py, train_biomass.py, train_multitask.py and
run_ablations.py so every experiment goes through one code path.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.labels import DISEASE_ATTRS
from src.training.experiment import ExperimentConfig, run_experiment
from src.training.registry import build_e9, build_matrix
from src.utils.paths import MANIFESTS, METRICS, SPLITS

RESULTS_SUMMARY = METRICS / "results_summary.csv"
FOLD_METRICS = METRICS / "fold_metrics.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    man = pd.read_parquet(MANIFESTS / "unified_manifest.parquet")
    spl = pd.read_parquet(SPLITS / "splits.parquet")
    return man, spl


def flatten_fold_result(r: dict) -> dict:
    """One flat row per completed fold, for fold_metrics.csv."""
    row = {"run_name": r.get("run_name"), "experiment": r.get("experiment"),
           "config_name": r.get("config_name"), "fold": r.get("fold"),
           "seed": r.get("seed"), "status": r.get("status", "completed")}
    d = r.get("disease")
    if d:
        h = d["health"]
        a = d["attributes"]
        row.update({
            "health_bal_acc": h.get("balanced_accuracy"),
            "health_f1": h.get("f1"),
            "health_auroc": h.get("auroc"),
            "health_precision": h.get("precision"),
            "health_recall": h.get("recall"),
            "health_brier": h.get("brier"),
            "health_ece": h.get("ece"),
            "attr_macro_f1": a.get("macro_f1"),
            "attr_micro_f1": a.get("micro_f1"),
            "attr_macro_map": a.get("macro_map"),
            "attr_hamming": a.get("hamming_loss"),
            "attr_exact_match": a.get("exact_match_ratio"),
            "n_test": d.get("n"),
        })
        for at in DISEASE_ATTRS:
            s = a.get(at, {})
            row[f"{at}_f1"] = s.get("f1")
            row[f"{at}_ap"] = s.get("average_precision")
            row[f"{at}_support_pos"] = s.get("support_pos")
        for src, sv in (d.get("by_source") or {}).items():
            tag = "sd" if src.startswith("Shrimp") else "tg"
            row[f"src_{tag}_health_bal_acc"] = sv["health"].get("balanced_accuracy")
            row[f"src_{tag}_attr_macro_f1"] = sv["attributes"].get("macro_f1")
            row[f"src_{tag}_n"] = sv.get("n")
    b = r.get("biomass")
    if b:
        iw = b.get("individual_level_weight_g", {})
        il = b.get("individual_level_length_cm", {})
        gw = b.get("image_level_weight_g", {})
        row.update({
            "ind_weight_mae_g": iw.get("mae"), "ind_weight_rmse_g": iw.get("rmse"),
            "ind_weight_mape": iw.get("mape"), "ind_weight_r2": iw.get("r2"),
            "ind_weight_medae_g": iw.get("medae"),
            "ind_weight_pearson": iw.get("pearson_r"),
            "ind_weight_spearman": iw.get("spearman_r"),
            "ind_length_mae_cm": il.get("mae"), "ind_length_r2": il.get("r2"),
            "img_weight_mae_g": gw.get("mae"), "img_weight_r2": gw.get("r2"),
            "n_individuals": b.get("n_individuals"),
        })
    return row


def _append_csv(path: Path, df: pd.DataFrame, key_cols: list[str]) -> None:
    if path.exists():
        prev = pd.read_csv(path)
        merged = pd.concat([prev, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_cols, keep="last")
    else:
        merged = df
    merged.to_csv(path, index=False)


def run_configs(cfgs: list[ExperimentConfig], verbose: bool = True) -> pd.DataFrame:
    man, spl = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows: list[dict] = []
    for i, c in enumerate(cfgs, 1):
        print(f"\n{'#' * 78}\n# [{i}/{len(cfgs)}] {c.name}  ({c.experiment})  "
              f"protocol={c.protocol} folds={c.folds} seed={c.seed}\n"
              f"# {c.notes}\n{'#' * 78}", flush=True)
        res = run_experiment(c, man, spl, device=device, verbose=verbose)
        for r in res:
            r.setdefault("experiment", c.experiment)
            r.setdefault("config_name", c.name)
            r.setdefault("seed", c.seed)
            all_rows.append(flatten_fold_result(r))
        if all_rows:
            _append_csv(FOLD_METRICS, pd.DataFrame(all_rows),
                        ["run_name"])
    df = pd.DataFrame(all_rows)
    if len(df):
        _append_csv(FOLD_METRICS, df, ["run_name"])
        summarize()
    return df


def summarize() -> pd.DataFrame:
    """Aggregate fold_metrics.csv into results_summary.csv (mean/sd across folds)."""
    if not FOLD_METRICS.exists():
        return pd.DataFrame()
    df = pd.read_csv(FOLD_METRICS)
    df = df[df["status"] == "completed"] if "status" in df else df
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    num = [c for c in num if c not in ("fold", "seed")]
    g = df.groupby(["experiment", "config_name"])
    rows = []
    for (exp, name), sub in g:
        row = {"experiment": exp, "config_name": name,
               "n_folds": int(sub["fold"].nunique()),
               "n_runs": int(len(sub)),
               "seeds": ";".join(str(s) for s in sorted(sub["seed"].unique()))}
        for c in num:
            v = sub[c].dropna()
            if len(v):
                row[f"{c}_mean"] = float(v.mean())
                row[f"{c}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["experiment", "config_name"])
    out.to_csv(RESULTS_SUMMARY, index=False)
    return out


def select_matrix(profile: str, only: list[str] | None, seed: int = 42,
                  e9: bool = False) -> list[ExperimentConfig]:
    if e9:
        return build_e9(profile)
    return build_matrix(profile, seed=seed, only=only)
