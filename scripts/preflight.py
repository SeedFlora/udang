"""Pre-training checks (specification section 18).

Runs, in order:
  1. the unit-test suite
  2. data-loader smoke tests (real images, real batching, shape/finiteness)
  3. a 20-sample overfitting test  -- proves the loss and optimiser can learn
  4. a one-epoch smoke test        -- proves an epoch completes end to end
  5. a split-leakage test on the ACTUAL split tables on disk

Any failure exits non-zero. The point is to spend GPU hours only on a pipeline
that has already demonstrated it can learn, rather than discovering a broken
loss after a multi-hour run.

Usage:
    python scripts/preflight.py [--profile standard] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.datasets import (  # noqa: E402
    BiomassImageDataset, BiomassIndividualDataset, DiseaseDataset,
)
from src.data.splits import verify_no_leakage  # noqa: E402
from src.data.transforms import build_biomass_transforms, build_disease_transforms  # noqa: E402
from src.models.multitask import ModelConfig, ShrimpMultiTaskModel  # noqa: E402
from src.training.losses import masked_bce_with_logits, masked_huber  # noqa: E402
from src.utils.paths import MANIFESTS, METRICS, REPO_ROOT, SPLITS  # noqa: E402

RESULTS: list[dict] = []


def record(name: str, ok: bool, detail: str = "", extra: dict | None = None,
           skipped: bool = False) -> bool:
    status = "SKIP" if skipped else ("PASS" if ok else "FAIL")
    RESULTS.append({"check": name, "status": status,
                    "detail": detail, "extra": extra or {}})
    print(f"[{status}] {name}: {detail}", flush=True)
    # A skipped check does not gate, but it is recorded so the report can state
    # exactly which checks did not actually run.
    return True if skipped else ok


# --------------------------------------------------------------------------- #
def check_unit_tests() -> bool:
    print("\n=== 1. unit tests ===", flush=True)
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    tail = (r.stdout or "").strip().splitlines()[-1:] or ["no output"]
    return record("unit_tests", r.returncode == 0, tail[-1])


def check_dataloaders(man: pd.DataFrame, spl: pd.DataFrame, device: torch.device) -> bool:
    print("\n=== 2. data loaders (real images) ===", flush=True)
    ok_all = True

    d = spl[(spl["protocol"] == "disease_cv") & (spl["fold"] == 0) &
            (spl["split"] == "train")]["record_id"].tolist()[:48]
    if d:
        ds = DiseaseDataset(man, d, build_disease_transforms(224, train=True))
        b = torch.utils.data.default_collate([ds[i] for i in range(min(8, len(ds)))])
        ok = (b["image"].ndim == 4 and b["image"].shape[1] == 3
              and torch.isfinite(b["image"]).all()
              and b["attrs"].shape[1] == 3 and b["attr_mask"].shape[1] == 3)
        ok_all &= record("disease_loader", ok,
                         f"batch image {tuple(b['image'].shape)}, "
                         f"attrs {tuple(b['attrs'].shape)}, finite={bool(torch.isfinite(b['image']).all())}")
        # masks must actually contain unknowns for ShrimpDiseaseImageBD
        sd = man[(man["dataset"] == "ShrimpDiseaseImageBD")]
        unknown = (sd["label_mask_yellow_head"].fillna(0) == 0).all()
        ok_all &= record("mask_semantics_in_manifest", bool(unknown),
                         "every ShrimpDiseaseImageBD row has yellow_head masked as unknown")
    else:
        ok_all &= record("disease_loader", False, "no disease split rows found")

    bm = spl[(spl["protocol"] == "biomass_lodo") & (spl["fold"] == 0) &
             (spl["split"] == "train")]["record_id"].tolist()[:64]
    if bm:
        ds = BiomassImageDataset(man, bm, build_biomass_transforms(224, train=True))
        b = torch.utils.data.default_collate([ds[i] for i in range(min(8, len(ds)))])
        ok = (b["biomass"].shape[1] == 3 and b["biomass_mask"].shape[1] == 3
              and torch.isfinite(b["image"]).all())
        ok_all &= record("biomass_loader", ok,
                         f"targets {tuple(b['biomass'].shape)}, "
                         f"mask mean {float(b['biomass_mask'].mean()):.3f}")

        mv = BiomassIndividualDataset(man, bm, build_biomass_transforms(224, train=True),
                                      max_views=4, seed=0)
        if len(mv):
            s = mv[0]
            ok = (s["views"].shape[0] == 4 and s["view_mask"].sum() >= 1
                  and torch.isfinite(s["views"]).all())
            ok_all &= record("multiview_batching", ok,
                             f"views {tuple(s['views'].shape)}, "
                             f"valid={int(s['view_mask'].sum())}/{s['views'].shape[0]}, "
                             f"available={s['n_available_views']}")
    else:
        # Distinguish "the biomass archive is not present yet" from "biomass
        # loading is broken". Only the latter should gate.
        has_biomass = int((man["task"] == "biomass").sum())
        ok_all &= record("biomass_loader", has_biomass == 0,
                         f"no biomass split rows; manifest holds {has_biomass} biomass "
                         f"records. Biomass experiments (E4-E7) cannot run until the "
                         f"archive is present.",
                         skipped=(has_biomass == 0))
    return ok_all


def check_overfit_20(man: pd.DataFrame, spl: pd.DataFrame, device: torch.device) -> bool:
    """Memorise 20 samples. If the model cannot drive the loss down on 20 images,
    the loss/optimiser wiring is broken and no amount of real training will help."""
    print("\n=== 3. 20-sample overfitting test ===", flush=True)
    ids = spl[(spl["protocol"] == "disease_cv") & (spl["fold"] == 0) &
              (spl["split"] == "train")]["record_id"].tolist()[:20]
    if len(ids) < 20:
        return record("overfit_20_samples", False, "fewer than 20 training records")

    ds = DiseaseDataset(man, ids, build_disease_transforms(192, train=False))
    batch = torch.utils.data.default_collate([ds[i] for i in range(20)])
    x = batch["image"].to(device)
    yh = batch["health"].to(device).float()
    ya = batch["attrs"].to(device).float()
    ma = batch["attr_mask"].to(device).float()

    model = ShrimpMultiTaskModel(ModelConfig(backbone="convnext_tiny", pretrained=True,
                                             enable_biomass=False)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    first = last = None
    for step in range(60):
        o = model(x)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(o["health_logit"], yh)
                + masked_bce_with_logits(o["attr_logits"], ya, ma))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss)
        last = float(loss)

    model.eval()
    with torch.no_grad():
        o = model(x)
        acc = float(((torch.sigmoid(o["health_logit"]) >= .5).float() == yh).float().mean())
    ok = np.isfinite(last) and last < first * 0.35 and acc >= 0.9
    return record("overfit_20_samples", ok,
                  f"loss {first:.4f} -> {last:.4f}, health accuracy {acc:.2f} "
                  f"(must drop >65% and reach >=0.90)",
                  {"loss_first": first, "loss_last": last, "train_accuracy": acc})


def check_masked_loss_zero() -> bool:
    print("\n=== 3b. masked-loss invariants ===", flush=True)
    z = torch.zeros(4, 3)
    a = masked_bce_with_logits(torch.randn(4, 3), z, z)
    b = masked_huber(torch.randn(4, 3), z, z)
    ok = bool(torch.isfinite(a) and torch.isfinite(b) and float(a) == 0 and float(b) == 0)
    # a non-finite prediction in a fully-masked column must not propagate
    pred = torch.tensor([[1.0, 2.0, float("inf")]])
    c = masked_huber(pred, torch.zeros(1, 3), torch.tensor([[1.0, 1.0, 0.0]]))
    ok = ok and bool(torch.isfinite(c))
    return record("masked_losses_zero_and_finite", ok,
                  "all-zero mask gives exact 0.0; inf at a masked slot does not propagate")


def check_one_epoch(man: pd.DataFrame, spl: pd.DataFrame, device: torch.device) -> bool:
    print("\n=== 4. one-epoch smoke test ===", flush=True)
    from src.training.experiment import ExperimentConfig, run_experiment
    from src.training.trainer import TrainConfig

    cfg = ExperimentConfig(
        name="PREFLIGHT_smoke", experiment="E0", tasks=("disease",),
        protocol="disease_cv", folds=(0,), image_size=160, batch_size=16,
        num_workers=0, seed=42,
        model=ModelConfig(backbone="convnext_tiny", pretrained=True, enable_biomass=False),
        train=TrainConfig(epochs=1, early_stop_patience=1, warmup_epochs=0),
        notes="preflight smoke test - not a reportable result")
    t0 = time.time()
    try:
        res = run_experiment(cfg, man, spl, device=device, verbose=True)
    except Exception as e:  # noqa: BLE001
        return record("one_epoch_smoke", False, f"{type(e).__name__}: {e}")
    ok = bool(res) and res[0].get("status") != "failed" and "disease" in res[0]
    detail = "epoch completed, predictions and metrics written"
    if ok:
        h = res[0]["disease"]["health"]
        detail += f", health bal_acc={h['balanced_accuracy']:.3f}"
    return record("one_epoch_smoke", ok, f"{detail} ({time.time() - t0:.0f}s)")


def check_split_leakage(spl: pd.DataFrame) -> bool:
    print("\n=== 5. split leakage (actual tables on disk) ===", flush=True)
    is_naive = spl["protocol"].str.contains("naive", regex=False)
    gated = spl[~is_naive]
    try:
        rep = verify_no_leakage(gated)
    except AssertionError as e:
        return record("split_leakage", False, str(e)[:220])
    # An empty gated table yields an empty report, which would pass vacuously:
    # "no group straddles a split" is trivially true when nothing was checked.
    if not rep["protocols"]:
        return record("split_leakage", False, "no gated protocol found in the split table")
    n = sum(len(v) for v in rep["protocols"].values())
    n_naive = int(is_naive.sum())
    return record("split_leakage", True,
                  f"{len(rep['protocols'])} gated protocols / {n} folds: no group_id "
                  f"appears in more than one split ({n_naive} rows of the deliberately "
                  f"leaky naive control arm excluded from the gate)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="standard")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"preflight on {device}")

    man = pd.read_parquet(MANIFESTS / "unified_manifest.parquet")
    spl = pd.read_parquet(SPLITS / "splits.parquet")

    ok = True
    if not args.skip_tests:
        ok &= check_unit_tests()
    ok &= check_dataloaders(man, spl, device)
    ok &= check_masked_loss_zero()
    ok &= check_overfit_20(man, spl, device)
    ok &= check_one_epoch(man, spl, device)
    ok &= check_split_leakage(spl)

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "preflight.json").write_text(
        json.dumps({"all_passed": ok, "checks": RESULTS}, indent=2, default=str),
        encoding="utf-8")
    print(f"\n{'=' * 60}\nPREFLIGHT: {'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
