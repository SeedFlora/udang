"""Biomass performance as a function of the number of views available at test time.

Specification section 13.2 requires reporting individual-level performance for
one randomly selected view, three views, five views, and all available views,
with random-view selection repeated several times under fixed seeds.

This script does that WITHOUT retraining: it loads each biomass checkpoint once
and re-runs test-time inference under different view budgets. Aggregation is
median-of-per-view-predictions for the single-image model (the aggregation the
main evaluation uses) and true multi-view pooling for the multi-view models.

Usage:
    python scripts/eval_view_counts.py [--repeats 3] [--views 1 3 5 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.datasets import BiomassImageDataset  # noqa: E402
from src.data.transforms import build_biomass_transforms  # noqa: E402
from src.evaluation.bootstrap import grouped_bootstrap_ci  # noqa: E402
from src.evaluation.metrics import biomass_metrics  # noqa: E402
from src.models.multitask import ModelConfig, ShrimpMultiTaskModel  # noqa: E402
from src.utils.paths import (  # noqa: E402
    CHECKPOINTS, FIGURES, MANIFESTS, METRICS, REPO_ROOT, SPLITS,
)
from src.visualization.figures import plot_views_curve  # noqa: E402


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


@torch.no_grad()
def per_image_predictions(model, man, ids, device, image_size) -> pd.DataFrame:
    ds = BiomassImageDataset(man, ids, build_biomass_transforms(image_size, train=False))
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    rows = []
    for b in dl:
        out = model(b["image"].to(device))
        pred = out["biomass"].float().cpu().numpy()
        for i in range(len(pred)):
            rows.append({"individual_id": b["individual_id"][i],
                         "db": b["db"][i],
                         "record_id": b["record_id"][i],
                         "pred_weight_g": float(np.exp(pred[i][0])),
                         "pred_length_cm": float(pred[i][1]),
                         "y_weight_g": float(b["weight_g"][i]),
                         "y_length_cm": float(b["length_cm"][i])})
    return pd.DataFrame(rows)


def subsample_and_score(per_img: pd.DataFrame, n_views: int, seed: int) -> dict:
    """n_views == 0 means use every available view."""
    rng = np.random.default_rng(seed)
    picked = []
    for ind, g in per_img.groupby("individual_id"):
        g = g.reset_index(drop=True)
        k = len(g) if n_views == 0 else min(n_views, len(g))
        idx = rng.permutation(len(g))[:k]
        picked.append(g.iloc[idx])
    sub = pd.concat(picked, ignore_index=True)
    agg = sub.groupby("individual_id").agg(
        pred=("pred_weight_g", "median"), y=("y_weight_g", "first"),
        pred_len=("pred_length_cm", "median"), y_len=("y_length_cm", "first"),
        n=("pred_weight_g", "count")).reset_index()
    ok = np.isfinite(agg["y"]) & np.isfinite(agg["pred"])
    a = agg[ok]
    m = biomass_metrics(a["y"].values, a["pred"].values, unit="g", mape_floor=0.5)
    lok = np.isfinite(agg["y_len"]) & np.isfinite(agg["pred_len"])
    ml = (biomass_metrics(agg[lok]["y_len"].values, agg[lok]["pred_len"].values,
                          unit="cm", mape_floor=0.1) if lok.sum() > 2 else {})
    return {"n_individuals": int(len(a)),
            "mean_views_used": float(agg["n"].mean()),
            "weight": m, "length": ml,
            "_ind": a[["individual_id", "y", "pred"]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--views", type=int, nargs="*", default=[1, 3, 5, 0])
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    man = pd.read_parquet(MANIFESTS / "unified_manifest.parquet")
    spl = pd.read_parquet(SPLITS / "splits.parquet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cks = sorted(CHECKPOINTS.glob("E4_biomass_single_image_f*_best.pt")) + \
        sorted(CHECKPOINTS.glob("E8_lightweight_biomass_f*_best.pt"))
    if not cks:
        print("BLOCKER: no biomass checkpoints found. Run E4 first.")
        return 2

    results: list[dict] = []
    for ck_path in cks:
        stem = ck_path.stem.replace("_best", "")
        cfg_name = stem.rsplit("_f", 1)[0]
        fold = int(stem.rsplit("_f", 1)[1].split("_s")[0])
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        mcfg = ModelConfig(pretrained=False, enable_health=False, enable_attrs=False)
        # The checkpoint stores only the TrainConfig, not the model config, so the
        # backbone (and pooling) must come from the run's meta.json -- otherwise a
        # MobileNet checkpoint (feat 1280) is loaded into a default ConvNeXt model
        # (feat 768) and the biomass head size mismatches.
        import json as _json
        meta_p = (REPO_ROOT / "outputs" / "logs" / f"{stem}_meta.json")
        if meta_p.exists():
            try:
                mm = _json.loads(meta_p.read_text(encoding="utf-8"))
                model_cfg = (mm.get("config") or {}).get("model") or {}
                for k, v in model_cfg.items():
                    if hasattr(mcfg, k) and k != "pretrained":
                        setattr(mcfg, k, v)
                mcfg.pretrained = False
            except Exception:  # noqa: BLE001
                pass
        model = ShrimpMultiTaskModel(mcfg).to(device).eval()
        missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
        # a size-consistent load may still legitimately skip the attention pooler
        # when comparing a mean-pooling checkpoint; only fail on a real mismatch
        if any("biomass_head" in k for k in list(missing) + list(unexpected)):
            print(f"   WARN {stem}: biomass head keys mismatched, skipping")
            continue

        ids = spl[(spl["protocol"] == "biomass_lodo") & (spl["fold"] == fold)
                  & (spl["split"] == "test")]["record_id"].tolist()
        if not ids:
            continue
        per_img = per_image_predictions(model, man, ids, device, args.image_size)
        print(f"{stem}: {len(per_img)} test images, "
              f"{per_img['individual_id'].nunique()} individuals")

        for nv in args.views:
            reps = []
            last_ind = None
            for rep in range(1 if nv == 0 else args.repeats):
                r = subsample_and_score(per_img, nv, seed=1000 + rep)
                reps.append(r)
                last_ind = r["_ind"]
            maes = [r["weight"]["mae"] for r in reps]
            r2s = [r["weight"]["r2"] for r in reps]
            ci = grouped_bootstrap_ci(_mae, last_ind["y"].values,
                                      last_ind["pred"].values,
                                      last_ind["individual_id"].values,
                                      n_boot=args.n_boot)
            results.append({
                "config_name": cfg_name, "fold": fold,
                "n_views": nv if nv else int(round(reps[0]["mean_views_used"])),
                "n_views_requested": "all" if nv == 0 else nv,
                "repeats": len(reps),
                "mae": float(np.mean(maes)), "mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
                "r2": float(np.mean(r2s)),
                "mae_ci_lo": ci["lo"], "mae_ci_hi": ci["hi"],
                "n_individuals": reps[0]["n_individuals"],
                "mean_views_used": reps[0]["mean_views_used"],
            })
            print(f"   views={nv if nv else 'all':>3}  "
                  f"MAE={np.mean(maes):.3f} g (sd over repeats {np.std(maes, ddof=1) if len(maes) > 1 else 0:.3f}), "
                  f"R2={np.mean(r2s):.3f}")
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(METRICS / "view_count_results.csv", index=False)
    (METRICS / "view_count_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    agg = df.groupby(["config_name", "n_views_requested"]).agg(
        mae=("mae", "mean"), mae_std=("mae", "std"), r2=("r2", "mean"),
        folds=("fold", "nunique")).reset_index()
    print("\n=== mean across held-out DBs ===")
    print(agg.round(4).to_string(index=False))

    plot_df = agg.rename(columns={"config_name": "model"}).copy()
    plot_df["n_views"] = pd.to_numeric(
        plot_df["n_views_requested"].replace("all", 99), errors="coerce")
    plot_views_curve(plot_df, FIGURES / "biomass_view_count_curve.png",
                     title="(99 = all available views)")
    print(f"\n-> outputs/metrics/view_count_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
