"""Mandatory data-audit checks (specification section 6).

Each check returns a uniform record. Checks marked `leakage_critical` gate the
whole project: if any of them FAILs, training must not start, because a leaked
duplicate or a mis-grouped augmentation family silently inflates every downstream
metric and the resulting report would be wrong rather than merely incomplete.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit.dedup import cross_block_near_duplicates, normalize_original_stem
from src.data.labels import DISEASE_ATTRS
from src.utils.paths import REPO_ROOT


def _check(name: str, status: str, severity: str, count: int, message: str,
           examples: list | None = None, extra: dict | None = None) -> dict:
    return {"check": name, "status": status, "severity": severity, "count": int(count),
            "message": message, "examples": (examples or [])[:15], "extra": extra or {}}


# --------------------------------------------------------------------------- #
def check_missing_files(df: pd.DataFrame, root: Path = REPO_ROOT) -> dict:
    missing = [r for r in df["relative_image_path"] if not (root / r).exists()]
    return _check("missing_files", "fail" if missing else "pass", "leakage_critical",
                  len(missing),
                  f"{len(missing)} manifest rows point at files that do not exist",
                  missing)


def check_unreadable(df: pd.DataFrame) -> dict:
    bad = df[df["exclude"].astype(bool)]
    return _check("unreadable_or_corrupt", "warn" if len(bad) else "pass",
                  "informational", len(bad),
                  f"{len(bad)} images could not be decoded and were excluded",
                  bad[["record_id", "relative_image_path", "exclude_reason"]]
                  .to_dict("records"))


def check_exact_duplicates(df: pd.DataFrame) -> dict:
    ok = df[~df["exclude"].astype(bool)]
    counts = ok.groupby("sha256")["record_id"].count()
    dups = counts[counts > 1]
    n_extra = int((dups - 1).sum())
    ex = []
    for sha in list(dups.index)[:10]:
        rows = ok[ok["sha256"] == sha]
        ex.append({"sha256": sha[:16], "n": int(len(rows)),
                   "paths": list(rows["relative_image_path"])[:4],
                   "datasets": sorted(set(rows["dataset"]))})
    return _check("exact_duplicate_sha256", "warn" if len(dups) else "pass",
                  "leakage_critical", len(dups),
                  f"{len(dups)} SHA-256 values occur more than once "
                  f"({n_extra} redundant copies). These are fused into one group so "
                  f"they cannot straddle a split.", ex,
                  {"n_redundant_copies": n_extra})


def check_near_duplicates(diag: dict) -> dict:
    d = diag.get("dedup", {})
    n = d.get("n_near_dup_links", 0)
    return _check("near_duplicates_phash", "pass", "informational", n,
                  f"{n} within-block perceptual near-duplicate links fused into groups",
                  d.get("near_pairs_sample", [])[:10],
                  {"group_size_histogram": d.get("group_size_histogram", {}),
                   "largest_group": d.get("largest_group")})


def check_cross_dataset_duplicates(df: pd.DataFrame, threshold: int = 6) -> dict:
    ok = df[~df["exclude"].astype(bool)].reset_index(drop=True)
    ph = ok["phash"].fillna("0")
    h = np.array([np.uint64(int(str(x), 16)) for x in ph], dtype=np.uint64)
    blocks = ok["dataset"].values
    pairs = cross_block_near_duplicates(h, blocks, threshold=threshold)
    ex = [{"a": ok.at[i, "relative_image_path"], "b": ok.at[j, "relative_image_path"],
           "hamming": d, "datasets": [ok.at[i, "dataset"], ok.at[j, "dataset"]]}
          for i, j, d in pairs[:15]]
    # exact cross-dataset duplicates are the severe case
    sha_cross = (ok.groupby("sha256")["dataset"].nunique())
    n_exact_cross = int((sha_cross > 1).sum())
    status = "fail" if n_exact_cross else ("warn" if pairs else "pass")
    return _check("cross_dataset_duplicates", status, "leakage_critical",
                  len(pairs),
                  f"{n_exact_cross} images are byte-identical across datasets; "
                  f"{len(pairs)} cross-dataset perceptual near-duplicates at Hamming<={threshold}",
                  ex, {"n_exact_cross_dataset": n_exact_cross})


def check_raw_annotated_pairs(df: pd.DataFrame) -> dict:
    sd = df[df["dataset"] == "ShrimpDiseaseImageBD"]
    raw = sd[sd["record_role"] == "raw"]
    ann = sd[sd["record_role"] == "annotated"]
    if len(ann) == 0:
        return _check("raw_annotated_pairing", "pass", "leakage_critical", 0,
                      "no annotated copies present")
    raw_stems = {Path(p).stem.lower() for p in raw["original_filename"]}
    ann_stems = {Path(p).stem.lower() for p in ann["original_filename"]}
    paired = ann_stems & raw_stems
    # every annotated copy must land in the same fused group as its raw twin
    grouped_together = 0
    for stem in list(paired):
        g_raw = set(raw[raw["original_filename"].str.lower().str.startswith(stem)]["group_id"])
        g_ann = set(ann[ann["original_filename"].str.lower().str.startswith(stem)]["group_id"])
        if g_raw and g_ann and g_raw & g_ann:
            grouped_together += 1
    frac = grouped_together / max(1, len(paired))
    return _check("raw_annotated_pairing",
                  "pass" if frac >= 0.95 else "warn", "leakage_critical",
                  len(paired),
                  f"{len(ann)} annotated copies, {len(paired)} share a stem with a raw "
                  f"image; {grouped_together} ({frac:.1%}) are fused into the same group. "
                  f"Annotated copies are excluded from the primary disease pool so they "
                  f"are never counted as independent samples.",
                  sorted(paired)[:10],
                  {"n_annotated": int(len(ann)), "n_raw": int(len(raw)),
                   "fraction_grouped": round(frac, 4)})


def check_dimensions(df: pd.DataFrame) -> dict:
    ok = df[~df["exclude"].astype(bool)]
    bad = ok[(ok["width"].fillna(0) <= 0) | (ok["height"].fillna(0) <= 0)]
    return _check("invalid_dimensions", "fail" if len(bad) else "pass",
                  "leakage_critical", len(bad),
                  f"{len(bad)} images report a non-positive dimension",
                  bad[["record_id", "width", "height"]].to_dict("records"))


def check_aspect_ratios(df: pd.DataFrame, lo: float = 0.2, hi: float = 5.0) -> dict:
    ok = df[~df["exclude"].astype(bool)].copy()
    ok["ar"] = ok["width"] / ok["height"].replace(0, np.nan)
    ext = ok[(ok["ar"] < lo) | (ok["ar"] > hi)]
    return _check("extreme_aspect_ratio", "warn" if len(ext) else "pass",
                  "informational", len(ext),
                  f"{len(ext)} images fall outside aspect ratio [{lo}, {hi}]; "
                  f"letterbox resizing preserves them without distortion",
                  ext[["record_id", "width", "height", "ar"]].round(3).to_dict("records"),
                  {"ar_min": float(ok["ar"].min()) if len(ok) else None,
                   "ar_max": float(ok["ar"].max()) if len(ok) else None})


def check_empty_classes(df: pd.DataFrame) -> dict:
    dis = df[(df["task"] == "disease") & (~df["exclude"].astype(bool))]
    counts = dis.groupby(["dataset", "raw_class"])["record_id"].count()
    empty = [f"{a}/{b}" for (a, b), c in counts.items() if c == 0]
    return _check("empty_classes", "fail" if empty else "pass", "leakage_critical",
                  len(empty), "classes present in the schema but with zero images",
                  empty, {"class_counts": {f"{a}|{b}": int(c) for (a, b), c in counts.items()}})


def check_label_spelling(df: pd.DataFrame) -> dict:
    dis = df[df["task"] == "disease"]
    variants: dict[str, set] = {}
    for ds, g in dis.groupby("dataset"):
        variants[ds] = set(g["raw_class"].dropna().unique())
    # harmonization already canonicalised these; report what survived
    return _check("label_spelling_variants", "pass", "informational",
                  sum(len(v) for v in variants.values()),
                  "canonical raw classes observed after normalisation",
                  [f"{k}: {sorted(v)}" for k, v in variants.items()])


def check_filename_collisions(df: pd.DataFrame) -> dict:
    ok = df[~df["exclude"].astype(bool)]
    c = Counter(ok["original_filename"])
    coll = {k: v for k, v in c.items() if v > 1}
    ex = []
    for k in list(coll)[:10]:
        rows = ok[ok["original_filename"] == k]
        ex.append({"filename": k, "n": int(len(rows)),
                   "distinct_sha256": int(rows["sha256"].nunique()),
                   "datasets": sorted(set(rows["dataset"]))})
    return _check("filename_collisions", "warn" if coll else "pass", "informational",
                  len(coll),
                  f"{len(coll)} filenames occur more than once across the corpus "
                  f"(identical names in different class folders are expected)",
                  ex)


def check_published_augmentations(df: pd.DataFrame) -> dict:
    tg = df[(df["dataset"] == "TigerShrimpBD") & (~df["exclude"].astype(bool))]
    if len(tg) == 0:
        return _check("published_augmentations", "pass", "leakage_critical", 0,
                      "TigerShrimpBD not present")
    aug = tg[tg["is_published_augmentation"].astype(bool)]
    org = tg[~tg["is_published_augmentation"].astype(bool)]
    org_groups = set(org["group_id"])
    orphans = aug[~aug["group_id"].isin(org_groups)]
    frac = 1 - len(orphans) / max(1, len(aug))

    # Why orphans exist: their normalised stems name source photographs that the
    # publisher did NOT include in the originals folders. Verified by checking
    # every orphan stem against the set of published original stems.
    org_stems = {str(Path(f).stem).lower() for f in org["original_filename"]}
    orphan_stems = {normalize_original_stem(str(f))[0].lower()
                    for f in orphans["original_filename"]}
    resolvable = orphan_stems & org_stems
    orphan_families = len(orphan_stems)

    # Leakage safety does not depend on finding the original: siblings derived
    # from the SAME unpublished source still share a family key and fuse together.
    sib_ok = True
    for gid, g in orphans.groupby("group_id"):
        if g["augmentation_family_id"].nunique() > 1:
            sib_ok = False
            break

    status = "pass" if (frac >= 0.85 and not resolvable and sib_ok) else "warn"
    return _check("published_augmentations", status, "leakage_critical",
                  int(len(aug)),
                  f"{len(org)} originals and {len(aug)} published augmented copies. "
                  f"{len(aug) - len(orphans)} augmented images ({frac:.1%}) share a fused "
                  f"group with a published original. The remaining {len(orphans)} form "
                  f"{orphan_families} families whose source photographs were NOT published "
                  f"in the originals folders ({len(resolvable)} of those stems match a "
                  f"published original, i.e. this is a property of the source data, not a "
                  f"tracing failure). Those families are still fused among themselves, so "
                  f"no augmented sibling can straddle a split.",
                  sorted(orphan_stems)[:8],
                  {"n_original": int(len(org)), "n_augmented": int(len(aug)),
                   "n_orphan_augmented": int(len(orphans)),
                   "n_orphan_families": int(orphan_families),
                   "n_orphan_stems_matching_a_published_original": int(len(resolvable)),
                   "orphan_siblings_correctly_fused": bool(sib_ok),
                   "fraction_traced_to_published_original": round(frac, 4)})


def check_biomass_individuals(df: pd.DataFrame, diag: dict) -> list[dict]:
    out = []
    bm = df[(df["task"] == "biomass") & (~df["exclude"].astype(bool))]
    d = diag.get("BiomassVannamei", {})

    if len(bm) == 0:
        # Reported explicitly rather than silently skipped: a manifest with no
        # biomass rows means the biomass experiments cannot run at all.
        return [_check("biomass_dataset_present", "fail", "leakage_critical", 0,
                       "No biomass records in the manifest. The biomass archive is "
                       "absent or was not extracted, so E4-E7 cannot be executed.")]

    no_meas = bm[bm["weight_g"].isna()]
    out.append(_check("images_without_measurements",
                      "warn" if len(no_meas) else "pass", "informational",
                      len(no_meas),
                      f"{len(no_meas)} biomass images have no joined weight measurement",
                      list(no_meas["relative_image_path"])[:10]))

    mwi = d.get("measurements_without_images", [])
    out.append(_check("measurements_without_images",
                      "warn" if mwi else "pass", "informational", len(mwi),
                      f"{len(mwi)} spreadsheet rows have no matching image",
                      mwi[:15]))

    unparsed = d.get("unparsed_names", [])
    out.append(_check("unparsable_image_names",
                      "fail" if unparsed else "pass", "leakage_critical", len(unparsed),
                      f"{len(unparsed)} biomass filenames do not match the s<m>_<n> "
                      f"pattern, so their individual identity cannot be recovered",
                      unparsed[:15]))

    imp = d.get("implausible", [])
    out.append(_check("implausible_measurements", "warn" if imp else "pass",
                      "informational", len(imp),
                      f"{len(imp)} individuals carry a physically implausible "
                      f"measurement flag (retained, flagged, not deleted)",
                      imp[:15]))

    per_ind = bm.groupby("individual_id")["record_id"].count()
    out.append(_check("biomass_individual_coverage", "pass", "informational",
                      int(bm["individual_id"].nunique()),
                      f"{bm['individual_id'].nunique()} individuals, "
                      f"{len(bm)} images, median {per_ind.median():.0f} views/individual",
                      [], {"views_per_individual": {
                          "min": int(per_ind.min()), "median": float(per_ind.median()),
                          "max": int(per_ind.max())},
                          "by_db": d.get("by_db", {})}))
    return out


def check_sdibd_individual_names(diag: dict) -> list[dict]:
    """ShrimpDiseaseImageBD animal identity is recovered from filenames
    (<CLASS>-<individual>-img-<view>). A name that does not parse leaves an animal
    ungroupable -- the same failure the biomass check above gates on -- so it is
    leakage-critical. A filename class token that disagrees with its folder is
    only a warning: the id is scoped by the folder class, which over-merges
    (conservative) rather than splits."""
    d = diag.get("ShrimpDiseaseImageBD", {})
    unparsed = d.get("unparsed_names", [])
    mismatch = d.get("class_token_mismatch", [])
    return [
        _check("sdibd_unparsable_image_names",
               "fail" if unparsed else "pass", "leakage_critical", len(unparsed),
               f"{len(unparsed)} ShrimpDiseaseImageBD filenames do not match "
               f"<CLASS>-<n>-img-<v>, so their animal identity cannot be recovered",
               unparsed[:15]),
        _check("sdibd_class_token_mismatch",
               "warn" if mismatch else "pass", "informational", len(mismatch),
               f"{len(mismatch)} ShrimpDiseaseImageBD filenames carry a class token that "
               f"differs from their folder class (id scoped by folder)",
               mismatch[:15]),
    ]


def check_biomass_measurement_outliers(df: pd.DataFrame, z_k: float = 3.5) -> dict:
    """Detect biomass measurements that are INTERNALLY INCONSISTENT.

    The test that matters is allometric: fit log(W) = a + b*log(L) over all
    individuals and flag large standardised residuals, i.e. animals whose weight
    does not match their length. A second check compares the cephalothorax/length
    ratio against the population.

    A marginal within-pond z-score is deliberately NOT used, because it cannot
    distinguish a bad number from a small animal. Worked example from this dataset:
    DB2_s043 (L=4.00 cm, W=1.0 g, C=1.30 cm) sits 5.7 marginal SD below the DB2
    length mean and looks alarming, but W/L^3 = 0.0156 (z = -1.33) and C/L = 0.325
    (z = +0.29, essentially the population mean), and its allometric residual is not
    in the top eight. All three of its measurements corroborate each other, which a
    transcription slip does not do -- it is a genuinely small shrimp. Flagging it
    would have put a false claim in the report. The pond size distributions
    legitimately differ (mean length: DB1 12.06, DB2 11.17, DB3 11.25, DB4 8.21,
    DB5 8.27 cm), which is what makes the marginal test misleading.

    Anything flagged is RETAINED, never dropped; the point is visibility, since a
    genuinely wrong measurement carries a large residual on the log-weight target
    and biomass intervals bootstrap over individuals.
    """
    bm = df[(df["task"] == "biomass") & (~df["exclude"].astype(bool))]
    if len(bm) == 0:
        return _check("biomass_measurement_outliers", "pass", "informational", 0,
                      "no biomass records")
    ind = bm.drop_duplicates("individual_id")[
        ["individual_id", "pond_or_db_id", "weight_g", "length_cm",
         "cephalothorax_cm"]].copy()
    n_img = bm.groupby("individual_id")["record_id"].count()

    flagged: list[dict] = []
    extra: dict = {}

    # --- allometric residual: does weight match length? ---
    m = ind[(ind["weight_g"] > 0) & (ind["length_cm"] > 0)].copy()
    if len(m) >= 10:
        lx = np.log(m["length_cm"].to_numpy(dtype=float))
        ly = np.log(m["weight_g"].to_numpy(dtype=float))
        b, a = np.polyfit(lx, ly, 1)
        resid = ly - (a + b * lx)
        sd = float(resid.std(ddof=2)) or 1.0
        z = resid / sd
        extra["allometric_fit"] = {"a": round(float(a), 4), "b": round(float(b), 4),
                                   "residual_sd_log": round(sd, 4), "n": int(len(m))}
        for k, iid in enumerate(m["individual_id"].tolist()):
            if abs(z[k]) > z_k:
                flagged.append({
                    "individual_id": str(iid), "test": "allometric_residual",
                    "db": str(m.iloc[k]["pond_or_db_id"]),
                    "length_cm": float(m.iloc[k]["length_cm"]),
                    "weight_g": float(m.iloc[k]["weight_g"]),
                    "expected_weight_g": round(float(np.exp(a + b * lx[k])), 2),
                    "residual_z": round(float(z[k]), 2),
                    "n_images_affected": int(n_img.get(iid, 0)),
                })

    # --- shape consistency: cephalothorax relative to total length ---
    c = ind[(ind["cephalothorax_cm"] > 0) & (ind["length_cm"] > 0)].copy()
    if len(c) >= 10:
        ratio = (c["cephalothorax_cm"] / c["length_cm"]).to_numpy(dtype=float)
        mu, sd = float(ratio.mean()), float(ratio.std(ddof=1)) or 1.0
        extra["cephalothorax_length_ratio"] = {"mean": round(mu, 4),
                                               "sd": round(sd, 4), "n": int(len(c))}
        for k, iid in enumerate(c["individual_id"].tolist()):
            zz = (ratio[k] - mu) / sd
            if abs(zz) > z_k:
                flagged.append({
                    "individual_id": str(iid), "test": "cephalothorax_length_ratio",
                    "db": str(c.iloc[k]["pond_or_db_id"]),
                    "ratio": round(float(ratio[k]), 4), "population_mean": round(mu, 4),
                    "residual_z": round(float(zz), 2),
                    "n_images_affected": int(n_img.get(iid, 0)),
                })

    uniq = sorted({f["individual_id"] for f in flagged})
    n_img_total = sum({f["individual_id"]: f["n_images_affected"]
                       for f in flagged}.values())
    extra.update({"n_individuals_flagged": len(uniq),
                  "n_images_affected": int(n_img_total),
                  "z_threshold": z_k, "individuals": uniq[:20],
                  "method": "allometric residual on log(W) ~ log(L), plus "
                            "cephalothorax/length ratio; marginal within-pond "
                            "z-scores deliberately not used"})
    return _check("biomass_measurement_outliers",
                  "warn" if flagged else "pass", "informational", len(uniq),
                  f"{len(uniq)} individual(s) have measurements inconsistent with the "
                  f"population length-weight relationship (|z| > {z_k}), affecting "
                  f"{n_img_total} images. Retained and flagged, never deleted.",
                  sorted(flagged, key=lambda f: -abs(f["residual_z"]))[:12], extra)


_CAPTURE_DATE_RE = re.compile(r"IMG-(\d{8})-", re.I)


def capture_session(filename: str) -> str | None:
    """Capture session recovered from the filename timestamp, where present.

    TigerShrimpBD filenames are WhatsApp-style `IMG-YYYYMMDD-WAnnnn.jpg`, so the
    acquisition date is recoverable and can be tested as a confound.
    """
    m = _CAPTURE_DATE_RE.search(str(filename))
    return m.group(1) if m else None


def check_class_session_confound(df: pd.DataFrame) -> dict:
    """Is the health label predictable from the CAPTURE SESSION alone?

    If a dataset photographs all its healthy animals on one set of days and all
    its diseased animals on another, then session-level artefacts -- lighting,
    background, camera, water colour, season -- separate the classes perfectly
    without any disease sign being visible. A model can then score 100% while
    having learned nothing about disease, and no split strategy can remove the
    confound because it is a property of how the data was collected.

    This is reported, not repaired: the only honest response is to state it and to
    weight cross-dataset transfer (which does not share session artefacts) more
    heavily than within-dataset accuracy.
    """
    out_rows: list[dict] = []
    worst = 0.0
    for ds, g in df[(df["task"] == "disease")
                    & (~df["exclude"].astype(bool))].groupby("dataset"):
        g = g[g["record_role"].isin(["raw", "original"])].copy()
        g["session"] = g["original_filename"].map(capture_session)
        known = g[g["session"].notna()]
        if len(known) == 0:
            out_rows.append({"dataset": str(ds), "sessions_recoverable": False,
                             "note": "no timestamp in filenames; confound not testable"})
            continue
        per = known.groupby("session")["health_status"].nunique()
        pure = set(per[per == 1].index)
        n_pure = int(known["session"].isin(pure).sum())
        frac = n_pure / max(1, len(known))
        worst = max(worst, frac)
        by_sess = (known.groupby(["session", "health_status"])["record_id"]
                   .count().unstack(fill_value=0))
        out_rows.append({
            "dataset": str(ds), "sessions_recoverable": True,
            "n_sessions": int(known["session"].nunique()),
            "n_sessions_single_health_class": int(len(pure)),
            "n_images_in_single_class_sessions": n_pure,
            "n_images_with_session": int(len(known)),
            "fraction_confounded": round(frac, 4),
            "sessions": {str(k): {str(c): int(v) for c, v in row.items()}
                         for k, row in by_sess.iterrows()},
        })

    status = "fail" if worst >= 0.999 else ("warn" if worst > 0.5 else "pass")
    msg = ("Health label vs capture session. "
           + "; ".join(
               f"{r['dataset']}: "
               + (f"{r['fraction_confounded']:.1%} of images sit in sessions "
                  f"containing only ONE health class "
                  f"({r['n_sessions_single_health_class']}/{r['n_sessions']} sessions)"
                  if r.get("sessions_recoverable") else r.get("note", ""))
               for r in out_rows))
    if worst >= 0.999:
        msg += (". COMPLETE CONFOUND: the health label is perfectly determined by "
                "acquisition session, so within-dataset healthy-vs-diseased scores "
                "on that dataset cannot be attributed to disease recognition.")
    return _check("class_session_confound", status, "informational",
                  int(round(worst * 100)), msg, out_rows,
                  {"max_fraction_confounded": round(worst, 4),
                   "interpretation": "1.0 means every image was captured in a "
                                     "session containing only one health class"})


def check_mask_integrity(df: pd.DataFrame) -> dict:
    """No unknown label may have been silently converted to a negative."""
    dis = df[(df["task"] == "disease") & (~df["exclude"].astype(bool))]
    problems = []
    sd = dis[dis["dataset"] == "ShrimpDiseaseImageBD"]
    if len(sd) and (sd["label_mask_yellow_head"].fillna(0) > 0).any():
        problems.append("ShrimpDiseaseImageBD rows claim supervised yellow_head labels, "
                        "but that dataset has no Yellow Head category")
    tg = dis[dis["dataset"] == "TigerShrimpBD"]
    for a in DISEASE_ATTRS:
        if len(tg) and (tg[f"label_mask_{a}"].fillna(0) == 0).any():
            problems.append(f"TigerShrimpBD rows have unsupervised {a}, but all four "
                            f"classes are annotated in that dataset")
    stats = {a: {"supervised": int((dis[f"label_mask_{a}"].fillna(0) > 0).sum()),
                 "positive": int(((dis[f"label_mask_{a}"].fillna(0) > 0) &
                                  (dis[a].fillna(0) > 0)).sum())}
             for a in DISEASE_ATTRS}
    return _check("label_mask_integrity", "fail" if problems else "pass",
                  "leakage_critical", len(problems),
                  "supervision masks must mark unknown labels, never encode them as 0",
                  problems, stats)


def run_all_checks(df: pd.DataFrame, diag: dict) -> tuple[list[dict], dict]:
    checks = [
        check_missing_files(df),
        check_unreadable(df),
        check_exact_duplicates(df),
        check_near_duplicates(diag),
        check_cross_dataset_duplicates(df),
        check_raw_annotated_pairs(df),
        check_dimensions(df),
        check_aspect_ratios(df),
        check_empty_classes(df),
        check_label_spelling(df),
        check_filename_collisions(df),
        check_published_augmentations(df),
        check_mask_integrity(df),
        check_class_session_confound(df),
        check_biomass_measurement_outliers(df),
    ]
    checks += check_biomass_individuals(df, diag)
    checks += check_sdibd_individual_names(diag)

    crit_fail = [c for c in checks if c["severity"] == "leakage_critical"
                 and c["status"] == "fail"]
    summary = {
        "n_checks": len(checks),
        "n_pass": sum(c["status"] == "pass" for c in checks),
        "n_warn": sum(c["status"] == "warn" for c in checks),
        "n_fail": sum(c["status"] == "fail" for c in checks),
        "leakage_critical_failures": [c["check"] for c in crit_fail],
        "gate": "BLOCKED" if crit_fail else "CLEAR",
    }
    return checks, summary
