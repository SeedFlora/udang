"""Leakage-safe split construction.

Six protocols, all built from the manifest's fused `group_id`:

  disease_cv            5-fold stratified group CV over the harmonized disease data
  cross_dataset         train on one disease dataset, test on the other (both directions)
  biomass_lodo          leave-one-DB-out, grouped strictly by individual shrimp
  biomass_loso          leave-one-SITE-out ({DB1} | {DB2,DB3} | {DB4,DB5}), the
                        defensible generalisation estimate behind leave-one-DB-out
  biomass_within_db     within-database grouped CV -- the protocol control for
                        leave-one-DB-out (same model, same data, only the held-out
                        unit changes)
  disease_cv_naive_withaug
                        a deliberately leaky control arm, measured not gated: an
                        image-level random split with no grouping at all, kept so
                        the cost of the naive protocol can be reported rather than
                        merely asserted

The single invariant every protocol must satisfy: no group_id may appear in more
than one of {train, val, test} within a fold. `verify_no_leakage` enforces it and
is called by the build script and by the test-suite; a violation is a hard failure,
not a warning, because a leaked augmentation family inflates every downstream
number in the report. The naive control arm is the one protocol never passed
through that gate -- `leakage_report` (the same code, minus the raise) measures it
instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, StratifiedKFold

from src.data.labels import is_cross_dataset_eligible

SPLIT_COLUMNS = ["record_id", "protocol", "fold", "split", "group_id",
                 "dataset", "stratum"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def disease_pool(manifest: pd.DataFrame, include_published_aug: bool = False,
                 include_annotated_copies: bool = False) -> pd.DataFrame:
    """Rows eligible for disease modelling.

    Defaults implement the specification's primary protocol: original images only
    (published TigerShrimpBD augmentations are an ABLATION, not the headline
    result) and no annotated duplicates of raw images (an annotated copy is the
    same photograph, not an extra sample).
    """
    df = manifest[(manifest["task"] == "disease") & (~manifest["exclude"].astype(bool))].copy()
    if not include_annotated_copies:
        df = df[df["record_role"] != "annotated"]
    if not include_published_aug:
        df = df[~df["is_published_augmentation"].astype(bool)]
    return df


def biomass_pool(manifest: pd.DataFrame, require_weight: bool = True) -> pd.DataFrame:
    df = manifest[(manifest["task"] == "biomass") & (~manifest["exclude"].astype(bool))].copy()
    if require_weight:
        df = df[df["weight_g"].notna() & (df["weight_g"] > 0)]
    return df


# The five biomass "databases" are three acquisition SITES. Source: the provider's
# own spreadsheet names under data/raw/biomass_vannamei/extracted/DATASET/ --
# DB1_INDUSTRIAL_FARM_1, DB2_INDUSTRIAL_FARM_2_C1, DB3_INDUSTRIAL_FARM_2_C2,
# DB4_ACADEMIC_POND_S1, DB5_ACADEMIC_POND_S2. (C1/C2 and S1/S2 are the provider's
# suffixes; "cycle" / "sampling" is our reading of them and is worded as such in
# the manuscript.) Leave-one-DB-out therefore leaves a same-rig sibling in
# training; leave-one-site-out is the defensible generalisation estimate.
BIOMASS_SITES = {
    "DB1": "site1_industrial_farm_1",
    "DB2": "site2_industrial_farm_2", "DB3": "site2_industrial_farm_2",
    "DB4": "site3_academic_pond", "DB5": "site3_academic_pond",
}


def _stratum(row) -> str:
    """Stratify approximately by dataset source AND disease combination."""
    return f"{row['dataset']}|{row['raw_class']}"


def _inner_val_split(train_df: pd.DataFrame, seed: int, val_frac: float = 0.2
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Carve a validation set out of the TRAINING folds only, still group-safe."""
    n_inner = max(2, int(round(1.0 / val_frac)))
    y = train_df["stratum"].values
    g = train_df["group_id"].values
    # Fall back to plain GroupKFold when a stratum is too rare to stratify on.
    try:
        splitter = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        tr_idx, va_idx = next(iter(splitter.split(np.zeros(len(train_df)), y, g)))
    except ValueError:
        splitter = GroupKFold(n_splits=n_inner)
        tr_idx, va_idx = next(iter(splitter.split(np.zeros(len(train_df)), y, g)))
    return tr_idx, va_idx


# --------------------------------------------------------------------------- #
# protocols
# --------------------------------------------------------------------------- #
def make_disease_cv(manifest: pd.DataFrame, n_splits: int = 5, seed: int = 42,
                    include_published_aug: bool = False,
                    protocol_name: str = "disease_cv") -> pd.DataFrame:
    df = disease_pool(manifest, include_published_aug=include_published_aug).copy()
    df["stratum"] = df.apply(_stratum, axis=1)

    y = df["stratum"].values
    g = df["group_id"].values
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows: list[dict] = []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(df)), y, g)):
        tr_df = df.iloc[tr]
        inner_tr, inner_va = _inner_val_split(tr_df, seed=seed + fold)
        assign = {}
        for i in df.index[te]:
            assign[i] = "test"
        for i in tr_df.index[inner_tr]:
            assign[i] = "train"
        for i in tr_df.index[inner_va]:
            assign[i] = "val"
        for i, s in assign.items():
            rows.append({
                "record_id": df.at[i, "record_id"], "protocol": protocol_name,
                "fold": fold, "split": s, "group_id": df.at[i, "group_id"],
                "dataset": df.at[i, "dataset"], "stratum": df.at[i, "stratum"],
            })
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def make_disease_cv_withaug(manifest: pd.DataFrame, base: pd.DataFrame,
                            base_protocol: str = "disease_cv",
                            protocol_name: str = "disease_cv_withaug") -> pd.DataFrame:
    """Published-augmentation ablation that changes exactly ONE variable.

    A naive re-run of the CV over the augmented pool puts published augmented
    copies into the TEST split, so the ablation would be scored on synthetic
    images while its baseline is scored on real ones -- three things changing at
    once (what is trained on, what is tested on, and which real photographs are in
    test at all). Measured on the first attempt: 62% of the with-augmentation test
    fold was synthetic, and it shared only 88 of ~430 real test records with the
    baseline fold.

    Instead, fold membership is INHERITED from the primary disease_cv assignment
    via group_id, and val/test stay original-images-only. Augmented copies are
    added to TRAIN only, and only to the folds where their source original is
    already training data. The test set is then byte-identical to the baseline's,
    so the comparison isolates the effect of training on published augmentations.

    Augmented images whose source original was never published (438 images in 52
    families, confirmed in the audit) belong to groups absent from the baseline.
    They can never be test data, so they are added to train in every fold.
    """
    aug_pool = disease_pool(manifest, include_published_aug=True)
    aug_only = aug_pool[aug_pool["is_published_augmentation"].astype(bool)]

    b = base[base["protocol"] == base_protocol]
    rows: list[dict] = []
    for fold, g in b.groupby("fold"):
        # inherit the baseline assignment verbatim (originals only)
        for _, r in g.iterrows():
            rows.append({"record_id": r["record_id"], "protocol": protocol_name,
                         "fold": int(fold), "split": r["split"],
                         "group_id": r["group_id"], "dataset": r["dataset"],
                         "stratum": r["stratum"]})
        train_groups = set(g[g["split"] == "train"]["group_id"])
        known_groups = set(g["group_id"])
        for _, r in aug_only.iterrows():
            gid = r["group_id"]
            # an augmented copy joins TRAIN iff its source original trains in this
            # fold, or its group has no published original at all
            if gid in train_groups or gid not in known_groups:
                rows.append({"record_id": r["record_id"], "protocol": protocol_name,
                             "fold": int(fold), "split": "train", "group_id": gid,
                             "dataset": r["dataset"],
                             "stratum": f"{r['dataset']}|{r['raw_class']}"})
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def make_disease_cv_single_dataset(manifest: pd.DataFrame, dataset: str,
                                   protocol_name: str, n_splits: int = 5,
                                   seed: int = 42) -> pd.DataFrame:
    """E1: the same grouped CV restricted to one source dataset.

    Kept identical to the pooled protocol in every other respect so that
    "single-dataset vs pooled" is a one-variable comparison.
    """
    sub = manifest[manifest["dataset"] == dataset]
    if len(sub) == 0:
        return pd.DataFrame(columns=SPLIT_COLUMNS)
    return make_disease_cv(sub, n_splits=n_splits, seed=seed,
                           include_published_aug=False, protocol_name=protocol_name)


def make_cross_dataset(manifest: pd.DataFrame, seed: int = 42,
                       include_published_aug: bool = False) -> pd.DataFrame:
    """Train on one disease dataset, test on the other. Both directions.

    Scored only on the semantically aligned Healthy / Black Gill / WSSV space --
    Yellow Head is absent from ShrimpDiseaseImageBD and BG_WSSV co-infection is
    inexpressible in TigerShrimpBD, so neither can be scored in a shared class space.
    Thresholds are selected on a validation split carved from the SOURCE dataset;
    the target dataset is never touched during training or tuning.
    """
    df = disease_pool(manifest, include_published_aug=include_published_aug).copy()
    df["stratum"] = df.apply(_stratum, axis=1)
    df["cross_eligible"] = [
        is_cross_dataset_eligible(r["dataset"], r["raw_class"]) for _, r in df.iterrows()
    ]

    rows: list[dict] = []
    pairs = [("ShrimpDiseaseImageBD", "TigerShrimpBD"),
             ("TigerShrimpBD", "ShrimpDiseaseImageBD")]
    for fold, (src, tgt) in enumerate(pairs):
        name = f"cross_{'sd2tg' if src.startswith('Shrimp') else 'tg2sd'}"
        src_df = df[df["dataset"] == src]
        # target scored only on the common class space
        tgt_df = df[(df["dataset"] == tgt) & (df["cross_eligible"])]

        inner_tr, inner_va = _inner_val_split(src_df, seed=seed + fold)
        for i in src_df.index[inner_tr]:
            rows.append({"record_id": src_df.at[i, "record_id"], "protocol": name,
                         "fold": 0, "split": "train", "group_id": src_df.at[i, "group_id"],
                         "dataset": src, "stratum": src_df.at[i, "stratum"]})
        for i in src_df.index[inner_va]:
            rows.append({"record_id": src_df.at[i, "record_id"], "protocol": name,
                         "fold": 0, "split": "val", "group_id": src_df.at[i, "group_id"],
                         "dataset": src, "stratum": src_df.at[i, "stratum"]})
        for i in tgt_df.index:
            rows.append({"record_id": tgt_df.at[i, "record_id"], "protocol": name,
                         "fold": 0, "split": "test", "group_id": tgt_df.at[i, "group_id"],
                         "dataset": tgt, "stratum": tgt_df.at[i, "stratum"]})
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def _biomass_holdout(df: pd.DataFrame, unit: pd.Series, protocol_name: str,
                     seed: int, val_individual_frac: float) -> pd.DataFrame:
    """Hold out one UNIT (a DB or a site) per fold. Validation individuals are
    drawn from the REMAINING DBs only; grouping is by individual shrimp."""
    rows: list[dict] = []
    units = sorted(unit.dropna().unique())
    rng = np.random.default_rng(seed)
    for fold, held in enumerate(units):
        te = df[unit == held]
        tr_all = df[unit != held]

        # choose validation INDIVIDUALS, stratified by source DB
        val_inds: set[str] = set()
        for _db, grp in tr_all.groupby("pond_or_db_id"):
            inds = np.array(sorted(grp["individual_id"].unique()))
            k = max(1, int(round(len(inds) * val_individual_frac)))
            val_inds.update(rng.permutation(inds)[:k].tolist())

        for i in te.index:
            rows.append({"record_id": te.at[i, "record_id"], "protocol": protocol_name,
                         "fold": fold, "split": "test", "group_id": te.at[i, "individual_id"],
                         "dataset": "BiomassVannamei", "stratum": str(held)})
        for i in tr_all.index:
            s = "val" if tr_all.at[i, "individual_id"] in val_inds else "train"
            rows.append({"record_id": tr_all.at[i, "record_id"], "protocol": protocol_name,
                         "fold": fold, "split": s, "group_id": tr_all.at[i, "individual_id"],
                         "dataset": "BiomassVannamei",
                         "stratum": str(tr_all.at[i, "pond_or_db_id"])})
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def make_biomass_lodo(manifest: pd.DataFrame, seed: int = 42,
                      val_individual_frac: float = 0.2) -> pd.DataFrame:
    """Leave-one-DB-out. Test = the entire held-out DB."""
    df = biomass_pool(manifest).copy()
    df["stratum"] = df["pond_or_db_id"].astype(str)
    return _biomass_holdout(df, df["pond_or_db_id"], "biomass_lodo", seed, val_individual_frac)


def make_biomass_loso(manifest: pd.DataFrame, seed: int = 42,
                      val_individual_frac: float = 0.2) -> pd.DataFrame:
    """Leave-one-SITE-out: {DB1} / {DB2, DB3} / {DB4, DB5}. Three folds."""
    df = biomass_pool(manifest).copy()
    df["stratum"] = df["pond_or_db_id"].astype(str)
    site = df["pond_or_db_id"].map(BIOMASS_SITES)
    if site.isna().any():
        raise KeyError(f"unmapped DB ids: {sorted(df.loc[site.isna(), 'pond_or_db_id'].unique())}")
    return _biomass_holdout(df, site, "biomass_loso", seed, val_individual_frac)


def make_biomass_within_db(manifest: pd.DataFrame, n_splits: int = 5, seed: int = 42,
                           val_individual_frac: float = 0.2) -> pd.DataFrame:
    """Within-database grouped CV: every DB in train AND test of every fold,
    individuals never split. The protocol control for leave-one-DB-out -- same
    model, same data, only the held-out unit changes (reviewer A4/A8)."""
    df = biomass_pool(manifest).copy()
    ind = (df.drop_duplicates("individual_id")[["individual_id", "pond_or_db_id"]]
           .sort_values("individual_id").reset_index(drop=True))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for fold, (tr_i, te_i) in enumerate(skf.split(np.zeros(len(ind)), ind["pond_or_db_id"].values)):
        te_inds = set(ind.iloc[te_i]["individual_id"])
        val_inds: set[str] = set()
        for _db, grp in ind.iloc[tr_i].groupby("pond_or_db_id"):
            inds = np.array(sorted(grp["individual_id"]))
            k = max(1, int(round(len(inds) * val_individual_frac)))
            val_inds.update(rng.permutation(inds)[:k].tolist())
        for i in df.index:
            iid = df.at[i, "individual_id"]
            s = "test" if iid in te_inds else ("val" if iid in val_inds else "train")
            rows.append({"record_id": df.at[i, "record_id"], "protocol": "biomass_within_db",
                         "fold": fold, "split": s, "group_id": iid,
                         "dataset": "BiomassVannamei",
                         "stratum": str(df.at[i, "pond_or_db_id"])})
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def make_disease_naive(manifest: pd.DataFrame, n_splits: int = 5, seed: int = 42,
                       include_published_aug: bool = True,
                       protocol_name: str = "disease_cv_naive_withaug") -> pd.DataFrame:
    """DELIBERATELY LEAKY control arm (reviewer A4, protocol-ladder rung 1).

    An image-level stratified random split over the augmented pool, with NO
    grouping: group_id is set to record_id so the self-consistency check is
    blind to it by construction, and the manifest cross-check will report it.
    This protocol is built to be measured, never to be trusted -- build_splits.py
    records its leakage instead of gating on it, and the manuscript says so.
    """
    df = disease_pool(manifest, include_published_aug=include_published_aug).copy()
    df["stratum"] = df.apply(_stratum, axis=1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: list[dict] = []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(df)), df["stratum"].values)):
        tr_df = df.iloc[tr]
        # a fixed 5 would warn on a stratum with fewer than 5 members; on the real
        # data every stratum has >= 39, so this stays the intended 20% carve
        n_inner = max(2, min(5, int(tr_df["stratum"].value_counts().min())))
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + fold)
        itr, iva = next(iter(inner.split(np.zeros(len(tr_df)), tr_df["stratum"].values)))
        assign = {i: "test" for i in df.index[te]}
        assign.update({i: "train" for i in tr_df.index[itr]})
        assign.update({i: "val" for i in tr_df.index[iva]})
        for i, s in assign.items():
            rows.append({"record_id": df.at[i, "record_id"], "protocol": protocol_name,
                         "fold": fold, "split": s, "group_id": df.at[i, "record_id"],
                         "dataset": df.at[i, "dataset"], "stratum": df.at[i, "stratum"]})
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def naive_test_composition(naive: pd.DataFrame, manifest: pd.DataFrame) -> list[dict]:
    """What a random image-level split actually puts in TEST: the fraction of
    test records whose true fused group also appears in train/val of that fold,
    and the fraction that are published synthetic copies."""
    mg = manifest.set_index("record_id")["group_id"]
    syn = manifest.set_index("record_id")["is_published_augmentation"].astype(bool)
    out = []
    for fold, g in naive.groupby("fold"):
        te = g[g["split"] == "test"]
        trv_groups = set(g[g["split"] != "test"]["record_id"].map(mg))
        te_groups = te["record_id"].map(mg)
        out.append({
            "fold": int(fold), "n_test": int(len(te)),
            "frac_test_group_leaked": float(te_groups.isin(trv_groups).mean()),
            "frac_test_synthetic": float(te["record_id"].map(syn).mean()),
        })
    return out


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def leakage_report(splits: pd.DataFrame, manifest: pd.DataFrame | None = None
                   ) -> dict:
    """Measure leakage. Returns a report and NEVER raises.

    This is the shared core: `verify_no_leakage` is this function plus a raise,
    and the deliberately leaky naive control arm is measured with this one
    directly (gating it would abort the build, which is the opposite of the point).

    Two independent checks:

    1. Self-consistency: no group_id written by the protocol appears in more than
       one split of a fold.
    2. Manifest cross-check (when `manifest` is supplied): re-derive grouping from
       the manifest's own fused `group_id` and verify THAT does not straddle
       splits either. Check 1 alone is circular -- a protocol that wrote the wrong
       grouping key would pass it while leaking badly. This catches that.
    """
    report: dict = {"protocols": {}, "violations": [], "manifest_cross_check": None}
    for (proto, fold), grp in splits.groupby(["protocol", "fold"]):
        by_split = {s: set(g["group_id"]) for s, g in grp.groupby("split")}
        overlaps = {}
        names = sorted(by_split)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                inter = by_split[names[a]] & by_split[names[b]]
                if inter:
                    overlaps[f"{names[a]}&{names[b]}"] = sorted(inter)[:20]
                    report["violations"].append({
                        "protocol": proto, "fold": int(fold),
                        "pair": f"{names[a]}&{names[b]}", "n_shared_groups": len(inter),
                        "examples": sorted(inter)[:10],
                    })
        # also verify no record_id appears twice in one fold
        dup_rec = grp["record_id"].duplicated().sum()
        if dup_rec:
            report["violations"].append({
                "protocol": proto, "fold": int(fold),
                "pair": "duplicate_record_in_fold", "n_shared_groups": int(dup_rec),
                "examples": [],
            })
        report["protocols"].setdefault(proto, {})[int(fold)] = {
            "n_records": int(len(grp)),
            "n_groups": int(grp["group_id"].nunique()),
            "by_split": {s: int(len(g)) for s, g in grp.groupby("split")},
            "groups_by_split": {s: int(g["group_id"].nunique()) for s, g in grp.groupby("split")},
            "overlaps": overlaps,
        }

    # ---- completeness: an empty or truncated split must not pass as "clean" ----
    # A fold with no validation set, or a protocol that silently dropped records,
    # trivially satisfies the no-overlap property. Absence of overlap is not
    # evidence of a usable split.
    for (proto, fold), grp in splits.groupby(["protocol", "fold"]):
        by = grp.groupby("split")["record_id"].count().to_dict()
        missing = [s for s in ("train", "test") if by.get(s, 0) == 0]
        if missing:
            report["violations"].append({
                "protocol": proto, "fold": int(fold),
                "pair": f"empty_split:{','.join(missing)}",
                "n_shared_groups": 0, "examples": [],
            })
        # cross_* protocols legitimately have no separate val fold only if the
        # source split was carved; flag a missing val anywhere else
        if by.get("val", 0) == 0:
            report["violations"].append({
                "protocol": proto, "fold": int(fold), "pair": "empty_val_split",
                "n_shared_groups": 0, "examples": [],
            })

    # ---- independent cross-check against the manifest's own fused grouping ----
    if manifest is not None and "group_id" in manifest.columns:
        mg = manifest.set_index("record_id")["group_id"]
        mind = (manifest.set_index("record_id")["individual_id"]
                if "individual_id" in manifest.columns else None)
        cross: dict = {"checked_protocols": [], "violations": []}
        for (proto, fold), grp in splits.groupby(["protocol", "fold"]):
            g = grp.copy()
            g["manifest_group"] = g["record_id"].map(mg)
            for key in ["manifest_group"] + (["manifest_individual"] if mind is not None else []):
                if key == "manifest_individual":
                    g[key] = g["record_id"].map(mind)
                sub = g[g[key].notna()]
                spans = sub.groupby(key)["split"].nunique()
                bad = spans[spans > 1]
                if len(bad):
                    cross["violations"].append({
                        "protocol": proto, "fold": int(fold), "key": key,
                        "n_straddling": int(len(bad)),
                        "examples": [str(x) for x in list(bad.index)[:8]],
                    })
            cross["checked_protocols"].append(f"{proto}/f{fold}")
        report["manifest_cross_check"] = cross
        report["violations"] += cross["violations"]

    return report


def verify_no_leakage(splits: pd.DataFrame, manifest: pd.DataFrame | None = None) -> dict:
    """Hard leakage gate: `leakage_report` plus a raise on any violation."""
    report = leakage_report(splits, manifest)
    if report["violations"]:
        raise AssertionError(
            f"LEAKAGE DETECTED in {len(report['violations'])} protocol/fold combinations: "
            f"{report['violations'][:3]}")
    return report
