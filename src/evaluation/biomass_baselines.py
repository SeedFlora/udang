"""Non-neural biomass baselines (specification section 12).

These exist to make the image-based results interpretable. In particular the
allometric and tabular baselines consume the GROUND-TRUTH manual length
measurement, which an image-only model never sees -- so they are not a fair
comparison, they are an upper reference. Every result carries an explicit
`information_advantage` field saying what the model was given, so the report
cannot accidentally present them as like-for-like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.evaluation.metrics import biomass_metrics

MAPE_FLOOR_G = 0.5


def individual_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per shrimp individual (measurements are per-individual, not per-image)."""
    bm = manifest[(manifest["task"] == "biomass") & (~manifest["exclude"].astype(bool))]
    cols = ["individual_id", "pond_or_db_id", "weight_g", "length_cm",
            "cephalothorax_cm", "complete_body"]
    t = (bm[cols].dropna(subset=["individual_id"])
         .drop_duplicates("individual_id").reset_index(drop=True))
    n_views = bm.groupby("individual_id")["record_id"].count().rename("n_views")
    return t.merge(n_views, on="individual_id", how="left")


# --------------------------------------------------------------------------- #
def mean_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Always predict the training-fold mean. The floor any model must beat."""
    mw = float(train["weight_g"].mean())
    ml = float(train["length_cm"].mean()) if train["length_cm"].notna().any() else np.nan
    out = {
        "model": "mean_baseline",
        "information_advantage": "none (constant prediction)",
        "train_mean_weight_g": mw,
        "weight": biomass_metrics(test["weight_g"].values,
                                  np.full(len(test), mw), unit="g",
                                  mape_floor=MAPE_FLOOR_G),
    }
    if np.isfinite(ml) and test["length_cm"].notna().any():
        m = test["length_cm"].notna().values
        out["length"] = biomass_metrics(test["length_cm"].values[m],
                                        np.full(int(m.sum()), ml), unit="cm",
                                        mape_floor=0.1)
    return out


def allometric_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Fit W = a * L^b on training individuals, via the log-linear form.

    log(W) = log(a) + b*log(L) is fitted by ordinary least squares, which is the
    standard estimator for allometric scaling and is what the specification asks
    to evaluate alongside the direct power form.
    """
    tr = train.dropna(subset=["weight_g", "length_cm"])
    tr = tr[(tr["weight_g"] > 0) & (tr["length_cm"] > 0)]
    te = test.dropna(subset=["weight_g", "length_cm"])
    te = te[(te["weight_g"] > 0) & (te["length_cm"] > 0)]
    if len(tr) < 3 or len(te) < 1:
        return {"model": "allometric", "error": "insufficient paired length/weight data",
                "n_train": int(len(tr)), "n_test": int(len(te))}

    lx = np.log(tr["length_cm"].values)
    ly = np.log(tr["weight_g"].values)
    b, log_a = np.polyfit(lx, ly, 1)
    a = float(np.exp(log_a))

    pred_log = log_a + b * np.log(te["length_cm"].values)
    pred = np.exp(pred_log)
    r2_log = float(np.corrcoef(ly, log_a + b * lx)[0, 1] ** 2) if len(tr) > 2 else np.nan
    return {
        "model": "allometric_W_eq_a_L_pow_b",
        "information_advantage": "GROUND-TRUTH manual length (not available to image-only models)",
        "a": a, "b": float(b), "fit_r2_log_space": r2_log,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "weight": biomass_metrics(te["weight_g"].values, pred, unit="g",
                                  mape_floor=MAPE_FLOOR_G),
        "weight_log_space": biomass_metrics(np.log(te["weight_g"].values), pred_log,
                                            unit="log_g", mape_floor=1e-6),
    }


def tabular_baselines(train: pd.DataFrame, test: pd.DataFrame,
                      use_db_metadata: bool = False, seed: int = 42) -> list[dict]:
    """Linear / random forest / gradient boosting on manual measurements."""
    feats = ["length_cm"]
    if train["cephalothorax_cm"].notna().sum() > 3 and test["cephalothorax_cm"].notna().sum() > 0:
        feats.append("cephalothorax_cm")

    tr = train.dropna(subset=feats + ["weight_g"])
    te = test.dropna(subset=feats + ["weight_g"])
    if len(tr) < 5 or len(te) < 1:
        return [{"model": "tabular", "error": "insufficient rows",
                 "n_train": int(len(tr)), "n_test": int(len(te)), "features": feats}]

    Xtr, Xte = tr[feats].values, te[feats].values
    if use_db_metadata:
        # one-hot the DB id; only meaningful when explicitly evaluated as a
        # metadata model, and it cannot generalise to an unseen pond
        dbs = sorted(set(tr["pond_or_db_id"]) | set(te["pond_or_db_id"]))
        Xtr = np.hstack([Xtr, np.array([[1.0 * (d == x) for d in dbs]
                                        for x in tr["pond_or_db_id"]])])
        Xte = np.hstack([Xte, np.array([[1.0 * (d == x) for d in dbs]
                                        for x in te["pond_or_db_id"]])])
    ytr, yte = tr["weight_g"].values, te["weight_g"].values

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=seed,
                                               min_samples_leaf=2, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }
    out = []
    adv = ("GROUND-TRUTH manual " + " + ".join(feats)
           + (" + DB identity" if use_db_metadata else ""))
    for name, mdl in models.items():
        try:
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xte)
            r = {"model": name, "features": feats + (["db_onehot"] if use_db_metadata else []),
                 "information_advantage": adv,
                 "n_train": int(len(tr)), "n_test": int(len(te)),
                 "weight": biomass_metrics(yte, pred, unit="g", mape_floor=MAPE_FLOOR_G)}
            if hasattr(mdl, "feature_importances_"):
                r["feature_importances"] = [round(float(v), 4)
                                            for v in mdl.feature_importances_]
            out.append(r)
        except Exception as e:  # noqa: BLE001
            out.append({"model": name, "error": f"{type(e).__name__}: {e}"})
    return out


def run_all_baselines(manifest: pd.DataFrame, splits: pd.DataFrame,
                      protocol: str = "biomass_lodo", seed: int = 42) -> dict:
    """Leave-one-DB-out evaluation of every non-neural baseline."""
    ind = individual_table(manifest)
    sp = splits[splits["protocol"] == protocol]
    results: dict = {"protocol": protocol, "folds": {}, "n_individuals": int(len(ind))}

    man_ind = manifest.set_index("record_id")["individual_id"]
    for fold, g in sp.groupby("fold"):
        tr_ind = set(man_ind.loc[g[g["split"] == "train"]["record_id"]].dropna())
        te_ind = set(man_ind.loc[g[g["split"] == "test"]["record_id"]].dropna())
        tr = ind[ind["individual_id"].isin(tr_ind)]
        te = ind[ind["individual_id"].isin(te_ind)]
        held = sorted(set(te["pond_or_db_id"]))
        results["folds"][int(fold)] = {
            "held_out_db": held,
            "n_train_individuals": int(len(tr)),
            "n_test_individuals": int(len(te)),
            "mean": mean_baseline(tr, te),
            "allometric": allometric_baseline(tr, te),
            "tabular": tabular_baselines(tr, te, use_db_metadata=False, seed=seed),
        }
    return results


def summarize_baselines(res: dict) -> pd.DataFrame:
    rows = []
    for fold, f in res["folds"].items():
        for key in ("mean", "allometric"):
            r = f.get(key, {})
            if "weight" in r:
                rows.append({"fold": fold, "held_out_db": ";".join(f["held_out_db"]),
                             "model": r["model"],
                             "information_advantage": r.get("information_advantage", ""),
                             **{f"weight_{k}": v for k, v in r["weight"].items()
                                if isinstance(v, (int, float))}})
        for r in f.get("tabular", []):
            if "weight" in r:
                rows.append({"fold": fold, "held_out_db": ";".join(f["held_out_db"]),
                             "model": r["model"],
                             "information_advantage": r.get("information_advantage", ""),
                             **{f"weight_{k}": v for k, v in r["weight"].items()
                                if isinstance(v, (int, float))}})
    return pd.DataFrame(rows)
