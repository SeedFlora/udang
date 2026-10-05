"""Recompute all metrics FROM SAVED PREDICTIONS and run statistical comparisons.

This script never loads a model. That is deliberate: section 24 requires that
every reported number be reproducible from the prediction files alone, and the
only way to prove it is to compute the headline table without touching a
checkpoint.

Confidence intervals use a CLUSTER bootstrap. Disease predictions are resampled
by fused group (so an augmentation family counts once), biomass predictions by
individual shrimp (so 30 photos of one animal are not 30 observations).

Usage:
    python scripts/evaluate.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.labels import DISEASE_ATTRS  # noqa: E402
from src.evaluation.bootstrap import (  # noqa: E402
    grouped_bootstrap_ci, paired_grouped_bootstrap, summarize_seeds,
    wilcoxon_across_folds,
)
from src.evaluation.metrics import (  # noqa: E402
    biomass_metrics, disease_attribute_metrics, disease_health_metrics,
)
from src.utils.paths import MANIFESTS, METRICS, PREDICTIONS, REPO_ROOT  # noqa: E402


# --------------------------------------------------------------------------- #
def _balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    pred = (p >= 0.5).astype(int)
    out = []
    for c in (0, 1):
        m = y == c
        if m.sum() == 0:
            return float("nan")
        out.append((pred[m] == c).mean())
    return float(np.mean(out))


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def _is_reportable(name: str) -> bool:
    """Exclude pipeline-test runs (preflight / smoke) from reported results."""
    low = name.lower()
    return not (low.startswith("preflight") or "smoke" in low)


def load_disease_predictions() -> dict[str, pd.DataFrame]:
    out: dict[str, list[pd.DataFrame]] = {}
    for f in sorted(PREDICTIONS.glob("*_test.parquet")):
        if "biomass" in f.name or not _is_reportable(f.name):
            continue
        df = pd.read_parquet(f)
        if "p_health" not in df.columns:
            continue
        cfg = f.name.rsplit("_f", 1)[0]
        out.setdefault(cfg, []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def load_biomass_predictions() -> dict[str, pd.DataFrame]:
    out: dict[str, list[pd.DataFrame]] = {}
    for f in sorted(PREDICTIONS.glob("*_biomass_test.parquet")):
        if not _is_reportable(f.name):
            continue
        df = pd.read_parquet(f)
        cfg = f.name.rsplit("_f", 1)[0]
        out.setdefault(cfg, []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def attach_groups(df: pd.DataFrame, man: pd.DataFrame) -> pd.DataFrame:
    g = man.set_index("record_id")["group_id"]
    df = df.copy()
    df["group_id"] = df["record_id"].map(g).fillna(df["record_id"])
    return df


# --------------------------------------------------------------------------- #
def fold_thresholds(cfg: str) -> dict[int, np.ndarray]:
    """Per-fold thresholds as recorded at checkpoint time from that fold's own
    validation split.

    Each fold selects its own operating point, so pooling every fold's predictions
    and scoring them with a single fold's thresholds would evaluate four of the
    five folds at an operating point they never chose. Attribute metrics are
    therefore computed per fold with that fold's thresholds and then aggregated.
    """
    out: dict[int, np.ndarray] = {}
    for p in sorted((REPO_ROOT / "outputs" / "logs").glob(f"{cfg}_f*_meta.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
            f = int(m.get("fold", -1))
            if m.get("thresholds"):
                out[f] = np.asarray(m["thresholds"], dtype=float)
        except Exception:  # noqa: BLE001
            continue
    return out


def _agg_across_folds(per_fold: list[dict], keys: list[str]) -> dict:
    """mean / sd across folds for a set of scalar metric keys."""
    out: dict = {}
    for k in keys:
        v = np.array([d.get(k, np.nan) for d in per_fold], dtype=float)
        v = v[np.isfinite(v)]
        out[k] = float(v.mean()) if len(v) else float("nan")
        out[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return out


def _restrict_cross_dataset_attrs(cfg: str, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """For cross-dataset runs, drop Yellow Head from attribute scoring.

    A model trained on ShrimpDiseaseImageBD has never seen a supervised Yellow
    Head label (that dataset has no such category), and the target test subset is
    restricted to the common Healthy / BG / WSSV space, so Yellow Head has zero
    positives there. Leaving it in would push a meaningless 0 or nan into
    macro-F1. Only Black Gill and WSSV are logically valid cross-dataset.
    """
    if "cross" not in cfg.lower():
        return df, ""
    out = df.copy()
    out["m_yellow_head"] = 0.0
    return out, ("Yellow Head excluded from attribute metrics: it is not a category "
                 "in ShrimpDiseaseImageBD and has zero positives in the common-class "
                 "target subset. Scored on black_gill and wssv only.")


def evaluate_disease(preds: dict[str, pd.DataFrame], man: pd.DataFrame,
                     n_boot: int) -> dict:
    res: dict = {}
    for cfg, df in preds.items():
        df = attach_groups(df, man)
        df, cross_note = _restrict_cross_dataset_attrs(cfg, df)
        thr_by_fold = fold_thresholds(cfg)
        default = np.full(len(DISEASE_ATTRS), 0.5)

        # --- attribute metrics: per fold, each with its OWN thresholds ---
        per_fold: list[dict] = []
        for f, g in df.groupby("fold") if "fold" in df.columns else [(0, df)]:
            t = thr_by_fold.get(int(f), default)
            m = disease_attribute_metrics(
                g[[f"y_{a}" for a in DISEASE_ATTRS]].values,
                g[[f"p_{a}" for a in DISEASE_ATTRS]].values,
                g[[f"m_{a}" for a in DISEASE_ATTRS]].values, t, DISEASE_ATTRS)
            flat = {"fold": int(f), "thresholds": [float(x) for x in t],
                    "macro_f1": m.get("macro_f1"), "micro_f1": m.get("micro_f1"),
                    "macro_map": m.get("macro_map"),
                    "hamming_loss": m.get("hamming_loss"),
                    "exact_match_ratio": m.get("exact_match_ratio")}
            for a in DISEASE_ATTRS:
                flat[f"{a}_f1"] = m.get(a, {}).get("f1")
                flat[f"{a}_ap"] = m.get(a, {}).get("average_precision")
            per_fold.append(flat)

        agg_keys = (["macro_f1", "micro_f1", "macro_map", "hamming_loss",
                     "exact_match_ratio"]
                    + [f"{a}_f1" for a in DISEASE_ATTRS]
                    + [f"{a}_ap" for a in DISEASE_ATTRS])

        entry = {
            "n_predictions": int(len(df)),
            "n_groups": int(df["group_id"].nunique()),
            "n_folds": len(per_fold),
            "thresholds_per_fold": {str(d["fold"]): d["thresholds"] for d in per_fold},
            # health uses a fixed 0.5 operating point, so pooling is well defined
            "health": disease_health_metrics(df["y_health"].values,
                                             df["p_health"].values),
            "attributes_mean_across_folds": _agg_across_folds(per_fold, agg_keys),
            "attributes_per_fold": per_fold,
            "health_balanced_accuracy_ci": grouped_bootstrap_ci(
                _balanced_accuracy, df["y_health"].values, df["p_health"].values,
                df["group_id"].values, n_boot=n_boot),
        }
        if cross_note:
            entry["cross_dataset_note"] = cross_note
        # threshold-free ranking metrics are safe to pool
        entry["attributes"] = disease_attribute_metrics(
            df[[f"y_{a}" for a in DISEASE_ATTRS]].values,
            df[[f"p_{a}" for a in DISEASE_ATTRS]].values,
            df[[f"m_{a}" for a in DISEASE_ATTRS]].values,
            np.asarray(list(thr_by_fold.values())).mean(axis=0)
            if thr_by_fold else default, DISEASE_ATTRS)
        entry["attributes"]["_note"] = (
            "Pooled across folds using the mean of the per-fold thresholds. "
            "Threshold-dependent values here are indicative; the per-fold values in "
            "attributes_mean_across_folds are the ones to quote. AP and AUROC are "
            "threshold-free and unaffected.")

        by_src = {}
        for src, g in df.groupby("dataset"):
            mean_thr = (np.asarray(list(thr_by_fold.values())).mean(axis=0)
                        if thr_by_fold else default)
            by_src[src] = {
                "n": int(len(g)),
                "health": disease_health_metrics(g["y_health"].values,
                                                 g["p_health"].values),
                "attributes": disease_attribute_metrics(
                    g[[f"y_{a}" for a in DISEASE_ATTRS]].values,
                    g[[f"p_{a}" for a in DISEASE_ATTRS]].values,
                    g[[f"m_{a}" for a in DISEASE_ATTRS]].values,
                    mean_thr, DISEASE_ATTRS),
            }
        entry["by_source"] = by_src

        # E9 runs the same config at three seeds and their prediction files share a
        # config key, so pooling them would silently merge three independent models
        # into one number that disagrees with the per-seed rows in fold_metrics.csv.
        if "seed" in df.columns and df["seed"].nunique() > 1:
            per_seed = {}
            for sd, g in df.groupby("seed"):
                per_seed[str(int(sd))] = {
                    "n": int(len(g)),
                    "health": disease_health_metrics(g["y_health"].values,
                                                     g["p_health"].values),
                }
            entry["by_seed"] = per_seed
            entry["n_seeds"] = int(df["seed"].nunique())
            entry["_seed_note"] = (
                "This configuration was run at multiple seeds. Pooled values mix "
                "them; use by_seed, or the E9 seed summary, for seed variance.")
        res[cfg] = entry
    return res


def evaluate_biomass(preds: dict[str, pd.DataFrame], n_boot: int) -> dict:
    res: dict = {}
    for cfg, df in preds.items():
        df = df.copy()
        df["pred_weight_g"] = np.exp(df["pred_log_weight"])
        # individual-level is the PRIMARY result
        ind = df.groupby("individual_id").agg(
            pred_weight_g=("pred_weight_g", "median"),
            pred_length_cm=("pred_length_cm", "median"),
            y_weight_g=("y_weight_g", "first"),
            y_length_cm=("y_length_cm", "first"),
            db=("db", "first"), n_views=("pred_weight_g", "count")).reset_index()
        ok = np.isfinite(ind["y_weight_g"]) & np.isfinite(ind["pred_weight_g"])
        ind_ok = ind[ok]

        entry = {
            "n_images": int(len(df)),
            "n_individuals": int(len(ind)),
            "individual_level_weight_g": biomass_metrics(
                ind_ok["y_weight_g"].values, ind_ok["pred_weight_g"].values,
                unit="g", mape_floor=0.5),
            "image_level_weight_g": biomass_metrics(
                df["y_weight_g"].values, df["pred_weight_g"].values,
                unit="g", mape_floor=0.5),
            # bootstrap resamples INDIVIDUALS, never images
            "individual_weight_mae_ci": grouped_bootstrap_ci(
                _mae, ind_ok["y_weight_g"].values, ind_ok["pred_weight_g"].values,
                ind_ok["individual_id"].values, n_boot=n_boot),
        }
        lok = np.isfinite(ind["y_length_cm"]) & np.isfinite(ind["pred_length_cm"])
        if lok.sum() > 2:
            entry["individual_level_length_cm"] = biomass_metrics(
                ind[lok]["y_length_cm"].values, ind[lok]["pred_length_cm"].values,
                unit="cm", mape_floor=0.1)
        by_db = {}
        for db, g in ind_ok.groupby("db"):
            by_db[str(db)] = {"n": int(len(g)),
                              **biomass_metrics(g["y_weight_g"].values,
                                                g["pred_weight_g"].values,
                                                unit="g", mape_floor=0.5)}
        entry["by_held_out_db"] = by_db
        res[cfg] = entry
    return res


# --------------------------------------------------------------------------- #
PAIRED_COMPARISONS = [
    ("E2_pooled_bce_ordinary", "E1_disease_single_tiger",
     "pooled vs single-dataset (TigerShrimpBD)"),
    ("E2_pooled_bce_ordinary", "E1_disease_single_sdibd",
     "pooled vs single-dataset (ShrimpDiseaseImageBD)"),
    ("E2_pooled_asl_ordinary", "E2_pooled_bce_ordinary",
     "masked asymmetric loss vs BCE"),
    ("E2_pooled_focal_ordinary", "E2_pooled_bce_ordinary",
     "masked focal loss vs BCE"),
    ("E2_pooled_bce_balanced", "E2_pooled_bce_ordinary",
     "source-balanced vs ordinary sampling"),
    ("E2_pooled_bce_withaug", "E2_pooled_bce_ordinary",
     "with published augmented copies vs originals only"),
    ("E2_pooled_bce_strongcolor", "E2_pooled_bce_ordinary",
     "aggressive colour augmentation vs disease-safe"),
    ("E6_multitask_fixed", "E2_pooled_bce_ordinary",
     "multi-task vs single-task disease"),
    ("E6_multitask_uncertainty", "E6_multitask_fixed",
     "uncertainty weighting vs fixed task weights"),
    ("E7_mtl_mixstyle", "E6_multitask_fixed", "MixStyle vs ERM multi-task"),
    ("E7_mtl_coral", "E6_multitask_fixed", "CORAL vs ERM multi-task"),
    ("E8_lightweight_disease", "E2_pooled_bce_ordinary",
     "MobileNetV3-Large vs ConvNeXt-Tiny"),
]

BIOMASS_COMPARISONS = [
    ("E5_biomass_multiview_mean", "E4_biomass_single_image",
     "multi-view mean pooling vs single image"),
    ("E5_biomass_multiview_attention", "E5_biomass_multiview_mean",
     "attention pooling vs mean pooling"),
    ("E6_multitask_fixed", "E4_biomass_single_image",
     "multi-task biomass vs single-task biomass"),
    ("E8_lightweight_biomass", "E4_biomass_single_image",
     "MobileNetV3-Large vs ConvNeXt-Tiny (biomass)"),
    ("E13_biomass_within_db", "E4_biomass_single_image",
     "within-database CV vs leave-one-DB-out (same model; protocol effect)"),
    ("E12_biomass_loso", "E4_biomass_single_image",
     "leave-one-site-out vs leave-one-DB-out"),
]


def paired_tests(dis: dict[str, pd.DataFrame], bio: dict[str, pd.DataFrame],
                 man: pd.DataFrame, n_boot: int) -> list[dict]:
    out: list[dict] = []
    for a, b, label in PAIRED_COMPARISONS:
        if a not in dis or b not in dis:
            out.append({"comparison": label, "model_a": a, "model_b": b,
                        "status": "skipped", "reason": "one or both runs absent"})
            continue
        da, db = attach_groups(dis[a], man), attach_groups(dis[b], man)
        common = sorted(set(da["record_id"]) & set(db["record_id"]))
        if len(common) < 20:
            out.append({"comparison": label, "model_a": a, "model_b": b,
                        "status": "skipped",
                        "reason": f"only {len(common)} shared test records "
                                  f"(different protocols are not directly comparable)"})
            continue
        ia = da.set_index("record_id").loc[common]
        ib = db.set_index("record_id").loc[common]
        r = paired_grouped_bootstrap(
            _balanced_accuracy, ia["y_health"].values, ia["p_health"].values,
            ib["p_health"].values, ia["group_id"].values, n_boot=n_boot)
        out.append({"comparison": label, "model_a": a, "model_b": b,
                    "metric": "health_balanced_accuracy",
                    "n_paired_records": len(common), "status": "ok", **r})

    for a, b, label in BIOMASS_COMPARISONS:
        if a not in bio or b not in bio:
            out.append({"comparison": label, "model_a": a, "model_b": b,
                        "status": "skipped", "reason": "one or both runs absent"})
            continue

        def ind_table(df):
            d = df.copy()
            d["pred_weight_g"] = np.exp(d["pred_log_weight"])
            return d.groupby("individual_id").agg(
                pred=("pred_weight_g", "median"), y=("y_weight_g", "first")).reset_index()

        ta, tb = ind_table(bio[a]), ind_table(bio[b])
        common = sorted(set(ta["individual_id"]) & set(tb["individual_id"]))
        if len(common) < 10:
            out.append({"comparison": label, "model_a": a, "model_b": b,
                        "status": "skipped",
                        "reason": f"only {len(common)} shared individuals"})
            continue
        ia = ta.set_index("individual_id").loc[common]
        ib = tb.set_index("individual_id").loc[common]
        r = paired_grouped_bootstrap(_mae, ia["y"].values, ia["pred"].values,
                                     ib["pred"].values, np.asarray(common),
                                     n_boot=n_boot)
        out.append({"comparison": label, "model_a": a, "model_b": b,
                    "metric": "individual_weight_MAE_g (lower is better)",
                    "n_paired_individuals": len(common), "status": "ok", **r})
    return out


def seed_summary() -> list[dict]:
    """Mean / sd / t-CI across seeds for every configuration run at >1 seed.

    (Round 1 keyed this on experiment == "E9"; the revision names multi-seed runs
    plainly, so any config with several seeds qualifies.)
    """
    fm = METRICS / "fold_metrics.csv"
    if not fm.exists():
        return []
    df = pd.read_csv(fm)
    out = []
    for cfg, g in df.groupby("config_name"):
        if g["seed"].nunique() < 2:
            continue
        for metric in ("health_bal_acc", "attr_macro_f1",
                       "ind_weight_mae_g", "ind_weight_r2"):
            if metric in g and g[metric].notna().any():
                per_seed = g.groupby("seed")[metric].mean().values
                out.append({"config_name": cfg, "metric": metric,
                            "n_seeds": len(per_seed),
                            "per_seed": [round(float(v), 5) for v in per_seed],
                            **summarize_seeds(per_seed)})
    return out


def fold_wilcoxon() -> list[dict]:
    fm = METRICS / "fold_metrics.csv"
    if not fm.exists():
        return []
    df = pd.read_csv(fm)
    out = []
    for a, b, label in PAIRED_COMPARISONS:
        ga = df[df["config_name"] == a].set_index("fold")
        gb = df[df["config_name"] == b].set_index("fold")
        common = sorted(set(ga.index) & set(gb.index))
        if len(common) < 3 or "health_bal_acc" not in df:
            continue
        va = ga.loc[common, "health_bal_acc"].values
        vb = gb.loc[common, "health_bal_acc"].values
        if np.isnan(va).any() or np.isnan(vb).any():
            continue
        out.append({"comparison": label, "model_a": a, "model_b": b,
                    "n_folds": len(common), **wilcoxon_across_folds(va, vb)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    man = pd.read_parquet(MANIFESTS / "unified_manifest.parquet")
    dis = load_disease_predictions()
    bio = load_biomass_predictions()
    print(f"disease prediction sets: {len(dis)}  biomass prediction sets: {len(bio)}")
    if not dis and not bio:
        print("BLOCKER: no prediction files found in outputs/predictions/")
        return 2

    report = {
        "disease": evaluate_disease(dis, man, args.n_boot),
        "biomass": evaluate_biomass(bio, args.n_boot),
        "paired_comparisons": paired_tests(dis, bio, man, args.n_boot),
        "seed_summary": seed_summary(),
        "fold_wilcoxon": fold_wilcoxon(),
        "n_boot": args.n_boot,
        "note": "All metrics recomputed from saved prediction files without "
                "re-running inference. Disease CIs resample fused groups; biomass "
                "CIs resample individual shrimp.",
    }
    p = METRICS / "evaluation_report.json"
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    rows = []
    for cfg, r in report["disease"].items():
        ci = r["health_balanced_accuracy_ci"]
        af = r["attributes_mean_across_folds"]
        rows.append({"config": cfg, "task": "disease", "n": r["n_predictions"],
                     "n_folds": r.get("n_folds"),
                     "health_bal_acc": round(r["health"]["balanced_accuracy"], 4),
                     "ci_lo": round(ci["lo"], 4), "ci_hi": round(ci["hi"], 4),
                     "health_f1": round(r["health"]["f1"], 4),
                     "health_auroc": round(r["health"]["auroc"], 4),
                     # per-fold thresholds, averaged across folds
                     "attr_macro_f1": round(af["macro_f1"], 4),
                     "attr_macro_f1_sd": round(af["macro_f1_std"], 4),
                     "attr_macro_map": round(af["macro_map"], 4)})
    for cfg, r in report["biomass"].items():
        w = r["individual_level_weight_g"]
        ci = r["individual_weight_mae_ci"]
        rows.append({"config": cfg, "task": "biomass", "n": r["n_individuals"],
                     "ind_mae_g": round(w["mae"], 4),
                     "ci_lo": round(ci["lo"], 4), "ci_hi": round(ci["hi"], 4),
                     "ind_r2": round(w["r2"], 4)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(METRICS / "evaluation_headline.csv", index=False)
    print("\n" + tbl.to_string(index=False))
    print(f"\n-> {p.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
