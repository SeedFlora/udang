"""Shortcut probes: how much of each headline number is recoverable WITHOUT
reading pathology -- from JPEG geometry and file size, from six colour moments,
or from an 8x8 thumbnail. Logistic regression, trained on train+val and scored on
test of the SAME split tables the CNNs used, fold by fold.

The metadata probe is the sharpest statement in the paper: every diseased
TigerShrimpBD original is 780x1040 or 1040x780 and no healthy original is, so
(width, height, aspect, log file size) classifies health perfectly.

Usage:
    python scripts/shortcut_probes.py [--splits-dir data/splits]
        [--out outputs/metrics/shortcut_probes.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cache import resolve_image_path  # noqa: E402
from src.utils.paths import MANIFESTS, METRICS, REPO_ROOT, SPLITS, rel_to_repo  # noqa: E402

DISEASE_PROTOCOLS = ["disease_cv_tiger", "disease_cv_sdibd", "disease_cv",
                     "disease_cv_withaug", "cross_sd2tg", "cross_tg2sd"]

# Frozen BEFORE any number is quoted. The metadata probe's balanced accuracy moves
# by up to +-0.08 on the pooled protocol with regularisation and class weighting,
# so the paper quotes this setting and reports the sensitivity range beside it.
SETTINGS = {"scaler": "StandardScaler", "C": 1.0, "class_weight": "balanced",
            "train_on": "train+val", "threshold": 0.5, "max_iter": 5000, "seed": 42}
SENSITIVITY = [(C, cw) for C in (0.1, 1.0, 10.0, 100.0) for cw in (None, "balanced")]
# Every diseased TigerShrimpBD ORIGINAL is 780x1040 or 1040x780; no healthy original
# is (healthy originals are 16:9 sizes: 1280x720, 720x1280, 900x1600, ...). A lookup
# on image size alone -- no learning, no hyperparameters -- is the cleanest statement.
PORTRAIT_RULE = {(780, 1040), (1040, 780)}
# The rule above is a constant read off the FULL dataset (test folds included),
# not fit on train and scored on held-out test like every other probe here. Every
# geometry_rule_780x1040 cell carries this note so it is never read as an
# out-of-sample estimate.
GEOMETRY_RULE_NOTE = ("fixed lookup rule; constant read off the full dataset; "
                      "descriptive, not a held-out estimate")


def probe(features: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
          test_mask: np.ndarray, seed: int = 42, C: float = 1.0,
          class_weight: str | None = "balanced", multiclass: bool = False) -> dict:
    Xtr = features[train_mask]
    # A feature with zero variance on the training side (e.g. width/height when the
    # source dataset has a single resolution, as SDIBD does at 2048x2048) carries no
    # information and makes the fitted direction arbitrary on the target side.
    # Reported so a reader can discount the cross-dataset metadata cells.
    degenerate = [int(i) for i in np.flatnonzero(Xtr.std(axis=0) == 0)]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=SETTINGS["max_iter"], C=C,
                                           class_weight=class_weight, random_state=seed))
    clf.fit(Xtr, y[train_mask])
    yt = y[test_mask]
    pred = clf.predict(features[test_mask])
    out = {"balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
           "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
           "degenerate_train_features": degenerate}
    # multiclass=True (DB identity) skips AUROC outright rather than guessing a
    # one-vs-rest scheme from whatever classes happen to land in this fold; a
    # binary fold with a single class in the test split is equally undefined and
    # reported the same way. JSON has no NaN token, so both write None.
    if multiclass:
        out["auroc"] = None
    elif len(np.unique(yt)) == 2:
        p = clf.predict_proba(features[test_mask])[:, 1]
        out["auroc"] = float(roc_auc_score(yt, p))
    else:
        out["auroc"] = None
    return out


def geometry_rule(man: pd.DataFrame, y: np.ndarray, test_rows: np.ndarray) -> dict:
    """Predict 'diseased' iff (width, height) is in PORTRAIT_RULE. No training."""
    wh = list(zip(man["width"].astype(int), man["height"].astype(int)))
    pred = np.array([wh[i] in PORTRAIT_RULE for i in test_rows]).astype(int)
    return {"balanced_accuracy": float(balanced_accuracy_score(y[test_rows], pred)),
            "n_test": int(len(test_rows)), "protocol": GEOMETRY_RULE_NOTE}


def metadata_features(man: pd.DataFrame) -> np.ndarray:
    w = man["width"].astype(float).values
    h = man["height"].astype(float).values
    s = man["file_size"].astype(float).values
    return np.c_[w, h, w / np.maximum(h, 1.0), np.log(np.maximum(s, 1.0))]


def _load_small(rel: str, side: int) -> np.ndarray:
    with Image.open(resolve_image_path(rel, REPO_ROOT)) as im:
        return np.asarray(im.convert("RGB").resize((side, side), Image.BILINEAR), dtype=np.float32) / 255.0


def thumbnail_features(man: pd.DataFrame, side: int = 8) -> np.ndarray:
    return np.stack([_load_small(r, side).reshape(-1) for r in man["relative_image_path"]])


def colour_moment_features(man: pd.DataFrame) -> np.ndarray:
    feats = []
    for r in man["relative_image_path"]:
        a = _load_small(r, 32).reshape(-1, 3)
        feats.append(np.r_[a.mean(0), a.std(0)])
    return np.stack(feats)


def _mean_or_none(vals: list) -> float | None:
    """Mean over non-None values; None (never NaN) if there is nothing to average."""
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def run_protocol(spl: pd.DataFrame, man: pd.DataFrame, proto: str, target: np.ndarray,
                 feats: dict[str, np.ndarray], idx: dict[str, int],
                 multiclass: bool = False) -> dict:
    res: dict = {"per_fold": {}, "mean": {}}
    g = spl[spl["protocol"] == proto]
    for fold, f in g.groupby("fold"):
        rows = f["record_id"].map(idx)
        ok = rows.notna().values
        if (~ok).sum() > 0:
            print(f"  {proto} fold {fold}: {int((~ok).sum())} split records not in manifest, dropped")
        rows = rows[ok].astype(int).values
        split = f["split"].values[ok]
        tr = np.isin(split, ["train", "val"])
        te = split == "test"
        y = target[rows]
        fold_res = {}
        for name, F in feats.items():
            fold_res[name] = probe(F[rows], y, tr, te, seed=SETTINGS["seed"],
                                   C=SETTINGS["C"], class_weight=SETTINGS["class_weight"],
                                   multiclass=multiclass)
        fold_res["metadata_sensitivity_ba"] = {
            f"C={C}|cw={cw}": probe(feats["metadata_whasize"][rows], y, tr, te,
                                    seed=SETTINGS["seed"], C=C, class_weight=cw,
                                    multiclass=multiclass)["balanced_accuracy"]
            for C, cw in SENSITIVITY}
        if proto == "disease_cv_tiger":
            fold_res["geometry_rule_780x1040"] = geometry_rule(man, target, rows[te])
        res["per_fold"][str(int(fold))] = fold_res
    for name in feats:
        for k in ("balanced_accuracy", "auroc"):
            vals = [d[name][k] for d in res["per_fold"].values()]
            res["mean"].setdefault(name, {})[k] = _mean_or_none(vals)
    sens = [v for d in res["per_fold"].values() for v in d["metadata_sensitivity_ba"].values()]
    res["metadata_sensitivity_ba_range"] = [float(min(sens)), float(max(sens))] if sens else None
    if res["per_fold"] and all("geometry_rule_780x1040" in d for d in res["per_fold"].values()):
        res["mean"]["geometry_rule_780x1040"] = {
            "balanced_accuracy": float(np.mean(
                [d["geometry_rule_780x1040"]["balanced_accuracy"] for d in res["per_fold"].values()])),
            "protocol": GEOMETRY_RULE_NOTE}
    return res


def _fmt_mean_cell(v: dict) -> str:
    ba = v.get("balanced_accuracy")
    ba_str = f"{ba:.3f}" if ba is not None else "None"
    auroc = v.get("auroc", float("nan"))
    auroc_str = f"{auroc:.3f}" if isinstance(auroc, float) else str(auroc)
    return f"BA={ba_str} AUROC={auroc_str}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default=str(SPLITS))
    ap.add_argument("--manifest", default=str(MANIFESTS / "unified_manifest.parquet"))
    ap.add_argument("--out", default=str(METRICS / "shortcut_probes.json"))
    ap.add_argument("--protocols", nargs="*", default=DISEASE_PROTOCOLS)
    args = ap.parse_args()

    man = pd.read_parquet(args.manifest)
    man = man[~man["exclude"].astype(bool)].reset_index(drop=True)
    spl = pd.read_parquet(Path(args.splits_dir) / "splits.parquet")
    idx = {rid: i for i, rid in enumerate(man["record_id"])}

    print("computing features (metadata from manifest; pixels from the 448 px cache) ...")
    feats = {"metadata_whasize": metadata_features(man),
             "colour_moments6": colour_moment_features(man),
             "thumbnail_8x8": thumbnail_features(man, 8)}

    diseased = (man["health_status"].astype(str) == "diseased").astype(int).values
    is_tiger = (man["dataset"] == "TigerShrimpBD").astype(int).values
    db_id = pd.factorize(man["pond_or_db_id"].astype(str))[0]

    out = {"splits_dir": rel_to_repo(args.splits_dir), "settings": SETTINGS,
           "sensitivity_grid": [f"C={C}|cw={cw}" for C, cw in SENSITIVITY],
           "features": list(feats), "protocols": {}}
    for proto in args.protocols:
        if proto not in set(spl["protocol"]):
            print(f"  skip {proto}: not in split table")
            continue
        out["protocols"][proto] = {"target": "health(diseased)",
                                   **run_protocol(spl, man, proto, diseased, feats, idx)}
        r = out["protocols"][proto]
        print(f"  {proto:26s} " + "  ".join(
            f"{n}: {_fmt_mean_cell(v)}" for n, v in r["mean"].items())
            + f"  | metadata BA range over settings {r['metadata_sensitivity_ba_range']}")
    if "disease_cv" in set(spl["protocol"]):
        out["protocols"]["disease_cv::dataset_identity"] = {
            "target": "dataset==TigerShrimpBD", **run_protocol(spl, man, "disease_cv", is_tiger, feats, idx)}
    if "biomass_lodo" in set(spl["protocol"]):
        out["protocols"]["biomass_within_db::db_identity"] = (
            {"target": "pond_or_db_id (multiclass)",
             **run_protocol(spl, man, "biomass_within_db", db_id, feats, idx, multiclass=True)}
            if "biomass_within_db" in set(spl["protocol"]) else
            {"note": "biomass_within_db protocol absent; DB identity needs a within-DB split"})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {rel_to_repo(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
