"""End-to-end experiment runner shared by E1-E9.

One entry point (`run_experiment`) drives every mandatory experiment. Which task
heads are supervised, which protocol/fold is used, and how losses are weighted are
all configuration -- so "single-task vs multi-task" and "ConvNeXt vs MobileNet"
differ in exactly one variable at a time.

Predictions are always written before metrics are computed, and metrics are then
computed FROM those saved predictions. That guarantees the section-24 requirement
that every number in the report can be regenerated without rerunning inference.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.datasets import (
    BiomassImageDataset, BiomassIndividualDataset, DiseaseDataset, SourceBalancedSampler,
)
from src.data.labels import DISEASE_ATTRS
from src.data.transforms import build_biomass_transforms, build_disease_transforms
from src.evaluation.metrics import (
    biomass_metrics, disease_attribute_metrics, disease_health_metrics, select_thresholds,
)
from src.models.multitask import ModelConfig, ShrimpMultiTaskModel, count_parameters
from src.training.trainer import MultiTaskTrainer, TrainConfig
from src.utils.hashing import hash_dataframe
from src.utils.paths import CHECKPOINTS, LOGS, METRICS, PREDICTIONS, REPO_ROOT
from src.utils.runmeta import RunMeta, set_global_seed


@dataclass
class ExperimentConfig:
    name: str
    experiment: str                       # E1 .. E9
    tasks: tuple[str, ...] = ("disease",)
    protocol: str = "disease_cv"
    # Disease and biomass folds come from different split tables, so a multi-task
    # run needs both named explicitly.
    biomass_protocol: str = "biomass_lodo"
    folds: tuple[int, ...] = (0,)
    seed: int = 42
    image_size: int = 320
    batch_size: int = 24
    biomass_batch_size: int = 16
    num_workers: int = 6
    include_published_aug: bool = False
    source_balanced: bool = False
    strong_color_aug: bool = False
    multiview: bool = False
    max_views: int = 4
    eval_view_counts: tuple[int, ...] = (1, 3, 5, 0)   # 0 == all available
    view_eval_repeats: int = 3
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    limit_train_batches: int | None = None
    extra_force_rerun: bool = False        # ignore on-disk results and retrain
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["model"] = asdict(self.model)
        d["train"] = asdict(self.train)
        return d


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def _subset(manifest: pd.DataFrame, splits: pd.DataFrame, protocol: str, fold: int,
            split: str) -> list[str]:
    s = splits[(splits["protocol"] == protocol) & (splits["fold"] == fold)
               & (splits["split"] == split)]
    return list(s["record_id"])


def build_disease_loaders(cfg: ExperimentConfig, manifest: pd.DataFrame,
                          splits: pd.DataFrame, fold: int) -> dict[str, DataLoader]:
    tf_tr = build_disease_transforms(cfg.image_size, train=True,
                                     strong_color=cfg.strong_color_aug)
    tf_ev = build_disease_transforms(cfg.image_size, train=False)
    out: dict[str, DataLoader] = {}
    for split, tf, shuffle in (("train", tf_tr, True), ("val", tf_ev, False),
                               ("test", tf_ev, False)):
        ids = _subset(manifest, splits, cfg.protocol, fold, split)
        if not ids:
            continue
        ds = DiseaseDataset(manifest, ids, tf)
        sampler = None
        if shuffle and cfg.source_balanced:
            sampler = SourceBalancedSampler(list(ds.df["dataset"]), seed=cfg.seed)
            shuffle = False
        # persistent_workers=False deliberately. With it True, each fold's three
        # loaders held num_workers live processes each, and on Windows those are not
        # promptly reaped when the loader is rebuilt next fold; across ~10 sequential
        # runs the worker processes accumulated until RAM was exhausted and OpenBLAS
        # could no longer allocate. The image cache makes decode cheap enough that
        # respawning workers each epoch costs little.
        out[split] = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=(split == "train"),
            persistent_workers=False, prefetch_factor=2 if cfg.num_workers else None,
        )
    return out


def build_biomass_loaders(cfg: ExperimentConfig, manifest: pd.DataFrame,
                          splits: pd.DataFrame, fold: int) -> dict[str, DataLoader]:
    tf_tr = build_biomass_transforms(cfg.image_size, train=True)
    tf_ev = build_biomass_transforms(cfg.image_size, train=False)
    out: dict[str, DataLoader] = {}
    proto = cfg.biomass_protocol
    for split, tf, shuffle in (("train", tf_tr, True), ("val", tf_ev, False),
                               ("test", tf_ev, False)):
        ids = _subset(manifest, splits, proto, fold, split)
        if not ids:
            continue
        persistent = cfg.num_workers > 0
        if cfg.multiview:
            ds = BiomassIndividualDataset(manifest, ids, tf, max_views=cfg.max_views,
                                          seed=cfg.seed,
                                          deterministic_first_n=(split != "train"))
            bs = max(2, cfg.biomass_batch_size // max(1, cfg.max_views // 2))
            # Persistent workers pickle the dataset once and never see the parent's
            # later mutations, so dataset.set_epoch(epoch) would silently never
            # reach them and every epoch would resample the SAME views -- turning
            # multi-view training into a fixed max_views-per-shrimp subset.
            # Verified: with persistent_workers=True the worker-side epoch stayed 0
            # across set_epoch(0/1/2); with it False the selection changed each epoch.
            persistent = False
        else:
            ds = BiomassImageDataset(manifest, ids, tf)
            bs = cfg.biomass_batch_size
        out[split] = DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=cfg.num_workers,
            pin_memory=True, drop_last=(split == "train"),
            persistent_workers=False,
            prefetch_factor=2 if cfg.num_workers else None,
        )
    return out


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_disease(model: torch.nn.Module, loader: DataLoader,
                    device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict] = []
    for b in loader:
        out = model(b["image"].to(device, non_blocking=True))
        ph = torch.sigmoid(out["health_logit"]).float().cpu().numpy()
        pa = torch.sigmoid(out["attr_logits"]).float().cpu().numpy()
        for i in range(len(ph)):
            r = {
                "record_id": b["record_id"][i],
                "dataset": b["dataset"][i],
                "raw_class": b["raw_class"][i],
                "y_health": float(b["health"][i]),
                "p_health": float(ph[i]),
            }
            for k, a in enumerate(DISEASE_ATTRS):
                r[f"y_{a}"] = float(b["attrs"][i][k])
                r[f"m_{a}"] = float(b["attr_mask"][i][k])
                r[f"p_{a}"] = float(pa[i][k])
            rows.append(r)
    return pd.DataFrame(rows)


@torch.no_grad()
def predict_biomass(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                    multiview: bool) -> pd.DataFrame:
    model.eval()
    rows: list[dict] = []
    for b in loader:
        if multiview:
            out = model.forward_multiview(b["views"].to(device),
                                          b["view_mask"].to(device))
            n = len(b["individual_id"])
            pred = out["biomass"].float().cpu().numpy()
            vw = out.get("view_weights")
            for i in range(n):
                rows.append({
                    "individual_id": b["individual_id"][i],
                    "db": b["db"][i],
                    "record_id": f"IND::{b['individual_id'][i]}",
                    "pred_log_weight": float(pred[i][0]),
                    "pred_length_cm": float(pred[i][1]),
                    "pred_ceph_cm": float(pred[i][2]),
                    "y_weight_g": float(b["weight_g"][i]),
                    "y_length_cm": float(b["length_cm"][i]),
                    "y_log_weight": float(b["biomass"][i][0]),
                    "m_log_weight": float(b["biomass_mask"][i][0]),
                    "m_length": float(b["biomass_mask"][i][1]),
                    "m_ceph": float(b["biomass_mask"][i][2]),
                    "y_ceph_cm": float(b["biomass"][i][2]),
                    "n_views_used": int(b["view_mask"][i].sum()),
                    "n_available_views": int(b["n_available_views"][i]),
                    "view_weights": json.dumps(
                        [round(float(x), 4) for x in vw[i].cpu().numpy()]) if vw is not None else "",
                })
        else:
            out = model(b["image"].to(device, non_blocking=True))
            pred = out["biomass"].float().cpu().numpy()
            for i in range(len(pred)):
                rows.append({
                    "record_id": b["record_id"][i],
                    "individual_id": b["individual_id"][i],
                    "db": b["db"][i],
                    "pred_log_weight": float(pred[i][0]),
                    "pred_length_cm": float(pred[i][1]),
                    "pred_ceph_cm": float(pred[i][2]),
                    "y_weight_g": float(b["weight_g"][i]),
                    "y_length_cm": float(b["length_cm"][i]),
                    "y_log_weight": float(b["biomass"][i][0]),
                    "y_ceph_cm": float(b["biomass"][i][2]),
                    "m_log_weight": float(b["biomass_mask"][i][0]),
                    "m_length": float(b["biomass_mask"][i][1]),
                    "m_ceph": float(b["biomass_mask"][i][2]),
                    "complete_body": float(b["complete_body"][i]),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# metrics from saved predictions
# --------------------------------------------------------------------------- #
def disease_metrics_from_predictions(pred: pd.DataFrame, thresholds: np.ndarray,
                                     health_threshold: float = 0.5) -> dict:
    res = {"health": disease_health_metrics(pred["y_health"].values,
                                            pred["p_health"].values,
                                            threshold=health_threshold)}
    y = pred[[f"y_{a}" for a in DISEASE_ATTRS]].values
    p = pred[[f"p_{a}" for a in DISEASE_ATTRS]].values
    m = pred[[f"m_{a}" for a in DISEASE_ATTRS]].values
    res["attributes"] = disease_attribute_metrics(y, p, m, np.asarray(thresholds),
                                                  DISEASE_ATTRS)
    res["n"] = int(len(pred))
    by_src = {}
    for src, g in pred.groupby("dataset"):
        gy = g[[f"y_{a}" for a in DISEASE_ATTRS]].values
        gp = g[[f"p_{a}" for a in DISEASE_ATTRS]].values
        gm = g[[f"m_{a}" for a in DISEASE_ATTRS]].values
        by_src[src] = {
            "n": int(len(g)),
            "health": disease_health_metrics(g["y_health"].values, g["p_health"].values,
                                             threshold=health_threshold),
            "attributes": disease_attribute_metrics(gy, gp, gm, np.asarray(thresholds),
                                                    DISEASE_ATTRS),
        }
    res["by_source"] = by_src
    return res


def biomass_metrics_from_predictions(pred: pd.DataFrame, mape_floor: float = 0.5) -> dict:
    """Image-level AND individual-level. Individual level is the primary result."""
    out: dict = {}
    pw = np.exp(pred["pred_log_weight"].values)          # back to grams
    yw = pred["y_weight_g"].values
    ok = np.isfinite(yw) & np.isfinite(pw) & (pred["m_log_weight"].values > 0)
    out["image_level_weight_g"] = biomass_metrics(yw[ok], pw[ok], unit="g",
                                                  mape_floor=mape_floor)
    lm = pred["m_length"].values > 0
    out["image_level_length_cm"] = biomass_metrics(
        pred["y_length_cm"].values[lm], pred["pred_length_cm"].values[lm],
        unit="cm", mape_floor=0.1)

    agg = pred.copy()
    agg["pred_weight_g"] = pw
    ind = agg.groupby("individual_id").agg(
        pred_weight_g=("pred_weight_g", "median"),
        pred_length_cm=("pred_length_cm", "median"),
        y_weight_g=("y_weight_g", "first"),
        y_length_cm=("y_length_cm", "first"),
        db=("db", "first"),
        n_views=("record_id", "count"),
    ).reset_index()
    iw = np.isfinite(ind["y_weight_g"].values) & np.isfinite(ind["pred_weight_g"].values)
    out["individual_level_weight_g"] = biomass_metrics(
        ind["y_weight_g"].values[iw], ind["pred_weight_g"].values[iw],
        unit="g", mape_floor=mape_floor)
    il = np.isfinite(ind["y_length_cm"].values) & np.isfinite(ind["pred_length_cm"].values)
    out["individual_level_length_cm"] = biomass_metrics(
        ind["y_length_cm"].values[il], ind["pred_length_cm"].values[il],
        unit="cm", mape_floor=0.1)
    out["n_individuals"] = int(len(ind))
    out["individual_table"] = ind.to_dict(orient="list")
    return out


# --------------------------------------------------------------------------- #
# main runner
# --------------------------------------------------------------------------- #
def run_experiment(cfg: ExperimentConfig, manifest: pd.DataFrame, splits: pd.DataFrame,
                   device: torch.device | None = None, verbose: bool = True) -> list[dict]:
    device = device or pick_device()
    results: list[dict] = []
    man_hash = hash_dataframe(manifest, ["record_id", "sha256"])
    spl_hash = hash_dataframe(splits, ["record_id", "protocol", "fold", "split"])

    for fold in cfg.folds:
        run_name = f"{cfg.name}_f{fold}_s{cfg.seed}"

        # Resume support: skip a fold whose checkpoint, predictions and metrics are
        # already on disk. Lets the whole matrix be relaunched after an interruption
        # (e.g. the OOM that killed the first attempt) without redoing finished work.
        done_ck = (CHECKPOINTS / f"{run_name}_best.pt").exists()
        done_metrics = (METRICS / f"{run_name}.json").exists()
        if done_ck and done_metrics and not cfg.extra_force_rerun:
            if verbose:
                print(f"\n=== {run_name}: already complete, skipping ===", flush=True)
            try:
                results.append(json.loads(
                    (METRICS / f"{run_name}.json").read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
            continue

        set_global_seed(cfg.seed + fold)
        meta = RunMeta(experiment=cfg.experiment, run_name=run_name, seed=cfg.seed,
                       config=cfg.to_dict(), manifest_hash=man_hash, split_hash=spl_hash,
                       protocol=cfg.protocol, fold=fold, notes=cfg.notes)
        dis = bio = model = trainer = None
        try:
            loaders: dict[str, DataLoader | None] = {}
            dis = bio = None
            if "disease" in cfg.tasks:
                dis = build_disease_loaders(cfg, manifest, splits, fold)
            if "biomass" in cfg.tasks:
                bio = build_biomass_loaders(cfg, manifest, splits, fold)

            model = ShrimpMultiTaskModel(cfg.model)
            trainer = MultiTaskTrainer(model, cfg.train, device,
                                       CHECKPOINTS, run_name)
            if verbose:
                print(f"\n=== {run_name} | device={device} | "
                      f"{count_parameters(model)} ===", flush=True)

            train_loaders = {}
            if dis:
                train_loaders["disease"] = dis.get("train")
            if bio:
                train_loaders["biomass"] = bio.get("train")

            for epoch in range(cfg.train.epochs):
                if bio and cfg.multiview and bio.get("train") is not None:
                    bio["train"].dataset.set_epoch(epoch)
                stats = trainer.train_epoch(train_loaders, epoch, cfg.train.epochs)

                val_metric, val_detail = _validate(cfg, model, dis, bio, device)
                stats.update({f"val_{k}": v for k, v in val_detail.items()})
                stats["val_primary"] = val_metric
                trainer.history.append(stats)
                improved = trainer.update_best(epoch, val_metric,
                                               extra={"val_detail": val_detail})
                if verbose:
                    print(f"  ep{epoch:02d} " +
                          " ".join(f"{k}={v:.4f}" for k, v in stats.items()
                                   if isinstance(v, float) and k.endswith("loss")) +
                          f" | val_primary={val_metric:.4f}"
                          f"{' *' if improved else ''}", flush=True)
                if trainer.should_stop(epoch):
                    if verbose:
                        print(f"  early stop at epoch {epoch} "
                              f"(best {trainer.best_epoch})", flush=True)
                    break

            # restore best weights before any test-set inference
            best_ck = CHECKPOINTS / f"{run_name}_best.pt"
            if best_ck.exists():
                trainer.load_checkpoint(best_ck)
            meta.best_epoch = trainer.best_epoch
            meta.best_val_metric = float(trainer.best_metric)
            meta.best_val_metric_name = cfg.train.monitor
            meta.checkpoint_path = str(best_ck.relative_to(REPO_ROOT).as_posix()) if best_ck.exists() else ""
            meta.curve_path = str(trainer.save_history().relative_to(REPO_ROOT).as_posix())

            fold_res = {"run_name": run_name, "fold": fold, "seed": cfg.seed,
                        "experiment": cfg.experiment, "config_name": cfg.name}

            # ---- disease: thresholds on VAL only, then test ----
            if dis and "test" in dis:
                thr = np.full(len(DISEASE_ATTRS), 0.5)
                if "val" in dis:
                    vp = predict_disease(model, dis["val"], device)
                    y = vp[[f"y_{a}" for a in DISEASE_ATTRS]].values
                    p = vp[[f"p_{a}" for a in DISEASE_ATTRS]].values
                    m = vp[[f"m_{a}" for a in DISEASE_ATTRS]].values
                    thr = select_thresholds(y, p, m, objective="f1")
                    vp.to_parquet(PREDICTIONS / f"{run_name}_val.parquet", index=False)
                tp = predict_disease(model, dis["test"], device)
                tp["fold"] = fold
                tp["seed"] = cfg.seed
                tp["run_name"] = run_name
                pth = PREDICTIONS / f"{run_name}_test.parquet"
                tp.to_parquet(pth, index=False)
                meta.predictions_path = str(pth.relative_to(REPO_ROOT).as_posix())
                meta.thresholds = [float(t) for t in thr]
                fold_res["disease"] = disease_metrics_from_predictions(tp, thr)
                fold_res["thresholds"] = [float(t) for t in thr]

            # ---- biomass ----
            if bio and "test" in bio:
                bp = predict_biomass(model, bio["test"], device, cfg.multiview)
                bp["fold"] = fold
                bp["seed"] = cfg.seed
                bp["run_name"] = run_name
                pth = PREDICTIONS / f"{run_name}_biomass_test.parquet"
                bp.to_parquet(pth, index=False)
                meta.predictions_path = (meta.predictions_path + ";" if meta.predictions_path else "") + \
                    str(pth.relative_to(REPO_ROOT).as_posix())
                fold_res["biomass"] = biomass_metrics_from_predictions(bp)

            mpath = METRICS / f"{run_name}.json"
            mpath.write_text(json.dumps(fold_res, indent=2, default=str), encoding="utf-8")
            meta.metrics_path = str(mpath.relative_to(REPO_ROOT).as_posix())
            meta.finish("completed")
            results.append(fold_res)

        except Exception as e:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            (LOGS / f"{run_name}_FAILED.log").write_text(tb, encoding="utf-8")
            meta.finish("failed", error=f"{type(e).__name__}: {e}")
            print(f"!!! RUN FAILED {run_name}: {type(e).__name__}: {e}", flush=True)
            results.append({"run_name": run_name, "fold": fold, "status": "failed",
                            "error": f"{type(e).__name__}: {e}"})
        finally:
            meta.to_json(LOGS / f"{run_name}_meta.json")
            meta.register()
            # Explicit teardown each fold. Dropping references and forcing a
            # collection reaps any DataLoader worker processes now rather than
            # letting them accumulate across the ~40 folds of the full matrix,
            # which is what exhausted RAM on the first attempt.
            del dis, bio, model, trainer
            import gc
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results


@torch.no_grad()
def _validate(cfg: ExperimentConfig, model: torch.nn.Module,
              dis: dict | None, bio: dict | None, device: torch.device
              ) -> tuple[float, dict]:
    """Validation score used for model selection. Never touches test data."""
    model.eval()
    detail: dict = {}
    parts: list[float] = []

    if dis and "val" in dis:
        vp = predict_disease(model, dis["val"], device)
        hm = disease_health_metrics(vp["y_health"].values, vp["p_health"].values)
        y = vp[[f"y_{a}" for a in DISEASE_ATTRS]].values
        p = vp[[f"p_{a}" for a in DISEASE_ATTRS]].values
        m = vp[[f"m_{a}" for a in DISEASE_ATTRS]].values
        thr = select_thresholds(y, p, m, objective="f1")
        am = disease_attribute_metrics(y, p, m, thr, DISEASE_ATTRS)
        detail["health_bal_acc"] = hm["balanced_accuracy"]
        detail["attr_macro_f1"] = am["macro_f1"]
        score = np.nanmean([hm["balanced_accuracy"], am["macro_f1"]])
        parts.append(float(score) if np.isfinite(score) else 0.0)

    if bio and "val" in bio:
        bp = predict_biomass(model, bio["val"], device, cfg.multiview)
        bp = bp.copy()
        bp["pred_weight_g"] = np.exp(bp["pred_log_weight"])
        # ALWAYS score validation at the INDIVIDUAL level, for both the
        # single-image and multi-view models. Scoring the single-image model
        # per-image and the multi-view model per-individual would compare two
        # different quantities: aggregating several views averages noise away, so
        # the multi-view arm would post a lower MAE by construction and win model
        # selection regardless of whether it is actually better. Individual level
        # is also the primary metric the specification asks for.
        agg = bp.groupby("individual_id").agg(
            pred=("pred_weight_g", "median"), y=("y_weight_g", "first")).reset_index()
        ok = np.isfinite(agg["y"].values) & np.isfinite(agg["pred"].values)
        bm = biomass_metrics(agg["y"].values[ok], agg["pred"].values[ok],
                             unit="g", mape_floor=0.5)
        detail["biomass_mae_g"] = bm["mae"]
        detail["biomass_r2"] = bm["r2"]
        detail["biomass_n_individuals"] = int(ok.sum())
        # convert to a "higher is better" score bounded in (0,1]
        parts.append(1.0 / (1.0 + bm["mae"]) if np.isfinite(bm["mae"]) else 0.0)

    return (float(np.mean(parts)) if parts else 0.0), detail
