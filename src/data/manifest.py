"""Construction of the single unified manifest that every later stage consumes.

One row per image across all three datasets. Missing labels are represented as
NaN plus an explicit supervision mask -- never as a zero. The manifest is the only
place dataset-specific layout knowledge lives; models, splitters and evaluators
read the manifest and nothing else.
"""
from __future__ import annotations

import hashlib
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit.dedup import build_fused_groups, normalize_original_stem
from src.data.biomass_parse import (
    individual_id, parse_image_name, parse_measurement_sheet, plausibility_flags,
)
from src.data.labels import DISEASE_ATTRS, harmonize

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

MANIFEST_COLUMNS = [
    "record_id", "dataset", "dataset_version", "source_domain", "source_subdomain",
    "species_if_known", "task", "relative_image_path", "original_filename",
    "width", "height", "file_size", "sha256", "phash",
    "duplicate_cluster_id", "augmentation_family_id", "individual_id", "pond_or_db_id",
    "health_status", "black_gill", "wssv", "yellow_head",
    "label_mask_black_gill", "label_mask_wssv", "label_mask_yellow_head",
    "length_cm", "weight_g", "cephalothorax_cm", "complete_body",
    "annotation_path", "split", "fold",
    # additional bookkeeping columns (superset of the required schema)
    "raw_class", "record_role", "is_published_augmentation", "group_id",
    "exclude", "exclude_reason", "quality_flags", "view_index",
]

SPECIES = {
    # The Bangladesh disease datasets are farmed black tiger shrimp; TigerShrimpBD
    # states this explicitly, ShrimpDiseaseImageBD does not name a species, so it is
    # recorded as unspecified rather than assumed.
    "ShrimpDiseaseImageBD": "unspecified",
    "TigerShrimpBD": "Penaeus monodon",
    "BiomassVannamei": "Litopenaeus vannamei",
}


# --------------------------------------------------------------------------- #
# Per-image statistics (parallel)
# --------------------------------------------------------------------------- #
def _image_stats(path_str: str) -> dict:
    """width/height/sha256/phash for one image. Never raises; reports errors."""
    from PIL import Image
    import imagehash

    out = {"path": path_str, "width": None, "height": None, "sha256": None,
           "phash": None, "file_size": None, "readable": False, "error": None}
    p = Path(path_str)
    try:
        out["file_size"] = p.stat().st_size
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            while True:
                b = fh.read(1 << 20)
                if not b:
                    break
                h.update(b)
        out["sha256"] = h.hexdigest()
        with Image.open(p) as im:
            im.verify()                       # catches truncated / corrupt files
        with Image.open(p) as im:
            im = im.convert("RGB")
            out["width"], out["height"] = im.size
            out["phash"] = str(imagehash.phash(im))
        out["readable"] = True
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def compute_image_stats(paths: list[Path], workers: int = 10) -> pd.DataFrame:
    strs = [str(p) for p in paths]
    if not strs:
        return pd.DataFrame(columns=["path", "width", "height", "sha256", "phash",
                                     "file_size", "readable", "error"])
    chunk = max(1, len(strs) // (workers * 8))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_image_stats, strs, chunksize=chunk))
    return pd.DataFrame(rows)


# Archive junk that must never enter the manifest. The ShrimpDiseaseImageBD zips
# were created on macOS and carry a __MACOSX/ tree that MIRRORS every image path
# with a `._`-prefixed AppleDouble stub of the same extension. Including it would
# exactly double the dataset with unreadable 4 KB files, and the duplicate-hash
# check would then be flooded with meaningless pairs.
_JUNK_DIRS = {"__macosx", ".git", ".ipynb_checkpoints"}
_JUNK_PREFIXES = ("._",)
_JUNK_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}


def is_junk_path(p: Path, root: Path) -> bool:
    rel = p.relative_to(root)
    if any(part.lower() in _JUNK_DIRS for part in rel.parts):
        return True
    if p.name.lower() in _JUNK_NAMES:
        return True
    return p.name.startswith(_JUNK_PREFIXES)


def _iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not is_junk_path(p, root):
            yield p


def _norm(s: str) -> str:
    """Lower-case, collapse separators, and strip a leading ordinal prefix.

    Source folders are named '1. Healthy', '4. WSSV_BG' etc., so the numbering
    has to come off before the class name can be matched.
    """
    s = str(s).strip()
    s = re.sub(r"^\s*\d+\s*[.)-]\s*", "", s)          # '4. WSSV_BG' -> 'WSSV_BG'
    return " ".join(s.lower().replace("-", " ").replace("_", " ").split())


# --------------------------------------------------------------------------- #
# ShrimpDiseaseImageBD
# --------------------------------------------------------------------------- #
# The publisher writes the co-infection folder as 'WSSV_BG' under Annotated and
# 'WSSV_BG' under Raw, while the documentation calls the class 'BG_WSSV'. Both
# orderings map to the same canonical class.
_SDIBD_CLASSES = {"healthy": "Healthy", "bg": "BG", "wssv": "WSSV",
                  "bg wssv": "BG_WSSV", "bgwssv": "BG_WSSV",
                  "wssv bg": "BG_WSSV", "wssvbg": "BG_WSSV"}


# ShrimpDiseaseImageBD names encode the ANIMAL: `<CLASS>-<individual>-img-<view>`,
# e.g. BG-34-img-3.jpg is the third photograph of black-gill animal 34. The data
# descriptor (Islam et al., Data in Brief 60:111553) states 1,149 images from
# 416 individual shrimp. Numbering restarts at 1 in every class (118 of the 142
# numbers recur), so the key is scoped by the folder's canonical class.
_SDIBD_NAME_RE = re.compile(
    r"^(?P<cls>[A-Za-z]+(?:_[A-Za-z]+)?)-(?P<ind>\d+)-img-(?P<view>\d+)$")


def parse_sdibd_name(filename: str) -> tuple[str | None, int | None, int | None]:
    """`BG-34-img-3.jpg` -> ("BG", 34, 3). Returns (None, None, None) when the
    pattern does not hold (AppleDouble stubs, augmented names, other datasets)."""
    stem = Path(str(filename).strip()).stem
    m = _SDIBD_NAME_RE.match(stem)
    if not m:
        return None, None, None
    return m.group("cls"), int(m.group("ind")), int(m.group("view"))


def sdibd_individual_id(raw_class: str, ind: int) -> str:
    """Class-scoped animal key. Mirrors the DB-scoping rule for biomass individuals
    in src/data/biomass_parse.py::individual_id."""
    return f"sdibd::{raw_class}::{int(ind):03d}"


def forced_individual_links(usable: pd.DataFrame) -> list[tuple[str, str]]:
    """Unite every record that carries an individual_id with every other record of
    the same individual: all views of one biomass shrimp and, since the round-1
    revision, all views of one ShrimpDiseaseImageBD animal. Before the revision
    this loop was filtered to task == "biomass", which is why 84.8-92.2 % of SDIBD
    test images shared an animal with train/val under the submitted splits."""
    extra: list[tuple[str, str]] = []
    has = usable[usable["individual_id"].notna()]
    for _ind, grp in has.groupby("individual_id"):
        ids = list(grp["record_id"])
        extra += [(ids[0], o) for o in ids[1:]]
    return extra


def scan_shrimp_disease_image_bd(root: Path, version: str) -> tuple[list[dict], dict]:
    """Walk the extracted tree. Raw and annotated copies are BOTH recorded but the
    annotated copies are marked so they are never counted as independent samples."""
    recs: list[dict] = []
    diag = {"unclassified_paths": [], "n_raw": 0, "n_annotated": 0, "n_labels": 0,
            "unparsed_names": [], "class_token_mismatch": []}

    label_files: dict[str, Path] = {}
    for p in sorted(root.rglob("*.txt")):
        if is_junk_path(p, root):
            continue
        label_files.setdefault(p.stem.lower(), p)
    diag["n_labels"] = len(label_files)

    for p in _iter_images(root):
        parts = [_norm(x) for x in p.relative_to(root).parts]
        raw_class = None
        for part in parts:
            if part in _SDIBD_CLASSES:
                raw_class = _SDIBD_CLASSES[part]
        if raw_class is None:
            diag["unclassified_paths"].append(str(p.relative_to(root)))
            continue

        joined = " ".join(parts)
        is_annotated = "annotated" in joined
        role = "annotated" if is_annotated else "raw"
        if is_annotated:
            diag["n_annotated"] += 1
        else:
            diag["n_raw"] += 1

        lab = label_files.get(p.stem.lower()) if is_annotated else None
        tok, ind, view = parse_sdibd_name(p.name)
        rel = str(p.relative_to(root).as_posix())
        if ind is None:
            diag["unparsed_names"].append(rel)
        elif _SDIBD_CLASSES.get(_norm(tok)) != raw_class:
            diag["class_token_mismatch"].append(rel)
        recs.append({
            "path": p,
            "dataset": "ShrimpDiseaseImageBD",
            "dataset_version": version,
            "source_domain": "ShrimpDiseaseImageBD",
            "source_subdomain": role,
            "task": "disease",
            "raw_class": raw_class,
            "record_role": role,
            "is_published_augmentation": False,
            "annotation_path": str(lab.relative_to(root).as_posix()) if lab else None,
            "pond_or_db_id": None,
            "individual_id": sdibd_individual_id(raw_class, ind) if ind is not None else None,
            "view_index": view,
        })
    return recs, diag


# --------------------------------------------------------------------------- #
# TigerShrimpBD
# --------------------------------------------------------------------------- #
_TIGER_CLASS_TO_RAW = {
    "healthy": "Healthy",
    "black_gill": "Black Gill",
    "wssv": "WSSV",
    "yellow_head": "Yellow Head",
}


def scan_tiger_shrimp_bd(root: Path, version: str) -> tuple[list[dict], dict]:
    """Local layout is <class>/<role>/<file>.jpg, written by download_data.py from
    the verified folder->class contract."""
    recs: list[dict] = []
    diag = {"unclassified_paths": [], "by_class_role": {}}

    for p in _iter_images(root):
        rel = p.relative_to(root).parts
        if len(rel) < 3:
            diag["unclassified_paths"].append(str(p.relative_to(root)))
            continue
        cls_dir, role = rel[0], rel[1]
        if cls_dir not in _TIGER_CLASS_TO_RAW or role not in ("original", "augmented"):
            diag["unclassified_paths"].append(str(p.relative_to(root)))
            continue

        key = f"{cls_dir}/{role}"
        diag["by_class_role"][key] = diag["by_class_role"].get(key, 0) + 1

        base_stem, applied = normalize_original_stem(p.name)
        recs.append({
            "path": p,
            "dataset": "TigerShrimpBD",
            "dataset_version": version,
            "source_domain": "TigerShrimpBD",
            "source_subdomain": role,
            "task": "disease",
            "raw_class": _TIGER_CLASS_TO_RAW[cls_dir],
            "record_role": role,
            "is_published_augmentation": role == "augmented" or bool(applied),
            # family key is class-scoped: identical base stems in different classes
            # are different photographs, not one family
            "augmentation_family_key": f"tiger::{cls_dir}::{base_stem.lower()}",
            "annotation_path": None,
            "pond_or_db_id": None,
            "individual_id": None,
            "view_index": None,
        })
    return recs, diag


# --------------------------------------------------------------------------- #
# Biomass
# --------------------------------------------------------------------------- #
def scan_biomass(root: Path, version: str) -> tuple[list[dict], dict]:
    """Images live under DB1..DB5; measurements in one spreadsheet per DB."""
    recs: list[dict] = []
    diag: dict = {"sheets": [], "unparsed_names": [], "by_db": {},
                  "images_without_measurement": [], "measurements_without_images": [],
                  "implausible": []}

    db_dirs: dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        if p.is_dir() and _norm(p.name).replace(" ", "") in {f"db{i}" for i in range(1, 6)}:
            db_dirs[p.name.upper()] = p
    diag["db_dirs_found"] = {k: str(v.relative_to(root).as_posix()) for k, v in db_dirs.items()}

    sheets: dict[str, pd.DataFrame] = {}
    for p in sorted(root.rglob("*.xls*")):
        if p.name.startswith("~$"):
            continue
        m = [f"DB{i}" for i in range(1, 6) if _norm(p.stem).replace(" ", "").startswith(f"db{i}")]
        if not m:
            continue
        try:
            df, d = parse_measurement_sheet(p)
            sheets[m[0]] = df
            d["db"] = m[0]
            diag["sheets"].append(d)
        except Exception as e:  # noqa: BLE001
            diag["sheets"].append({"db": m[0], "file": p.name, "error": str(e)})

    for db, d in sorted(db_dirs.items()):
        sheet = sheets.get(db)
        meas = {}
        if sheet is not None:
            for _, r in sheet.iterrows():
                meas[int(r["sample"])] = r
        seen_samples: set[int] = set()
        n_img = 0
        for p in _iter_images(d):
            ind, view = parse_image_name(p.name)
            if ind is None:
                diag["unparsed_names"].append(f"{db}/{p.name}")
                continue
            n_img += 1
            seen_samples.add(ind)
            row = meas.get(ind)
            if row is None:
                diag["images_without_measurement"].append(f"{db}/{p.name}")
            flags = plausibility_flags(row) if row is not None else ["no_measurement_row"]
            if flags and row is not None:
                diag["implausible"].append({"db": db, "sample": ind, "flags": flags})
            recs.append({
                "path": p,
                "dataset": "BiomassVannamei",
                "dataset_version": version,
                "source_domain": "BiomassVannamei",
                "source_subdomain": db,
                "task": "biomass",
                "raw_class": None,
                "record_role": "raw",
                "is_published_augmentation": False,
                "annotation_path": None,
                "pond_or_db_id": db,
                "individual_id": individual_id(db, ind),
                "view_index": view,
                "length_cm": float(row["length_cm"]) if row is not None and "length_cm" in row and pd.notna(row.get("length_cm")) else np.nan,
                "weight_g": float(row["weight_g"]) if row is not None and "weight_g" in row and pd.notna(row.get("weight_g")) else np.nan,
                "cephalothorax_cm": float(row["cephalothorax_cm"]) if row is not None and "cephalothorax_cm" in row and pd.notna(row.get("cephalothorax_cm")) else np.nan,
                "complete_body": float(row["complete_body"]) if row is not None and "complete_body" in row and pd.notna(row.get("complete_body")) else np.nan,
                "quality_flags": ";".join(flags) if flags else "",
            })
        diag["by_db"][db] = {"n_images": n_img, "n_individuals_in_images": len(seen_samples),
                             "n_individuals_in_sheet": int(len(sheet)) if sheet is not None else 0}
        if sheet is not None:
            missing = sorted(set(sheet["sample"].astype(int)) - seen_samples)
            diag["measurements_without_images"].extend(f"{db}/s{s}" for s in missing)
    return recs, diag


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_unified_manifest(raw_root: Path, versions: dict[str, str],
                           workers: int = 10, phash_threshold: int = 6,
                           ) -> tuple[pd.DataFrame, dict]:
    diagnostics: dict = {}
    records: list[dict] = []

    sd_root = raw_root / "shrimp_disease_image_bd" / "extracted"
    if sd_root.exists():
        r, d = scan_shrimp_disease_image_bd(sd_root, versions.get("ShrimpDiseaseImageBD", "3"))
        records += r
        diagnostics["ShrimpDiseaseImageBD"] = d

    tg_root = raw_root / "tiger_shrimp_bd"
    if tg_root.exists():
        r, d = scan_tiger_shrimp_bd(tg_root, versions.get("TigerShrimpBD", "1"))
        records += r
        diagnostics["TigerShrimpBD"] = d

    bm_root = raw_root / "biomass_vannamei" / "extracted"
    if bm_root.exists():
        r, d = scan_biomass(bm_root, versions.get("BiomassVannamei", "2"))
        records += r
        diagnostics["BiomassVannamei"] = d

    if not records:
        raise RuntimeError(f"No images discovered under {raw_root}. Refusing to build an empty manifest.")

    df = pd.DataFrame(records)
    stats = compute_image_stats([r["path"] for r in records], workers=workers)
    df["path_str"] = [str(p) for p in df["path"]]
    df = df.merge(stats, left_on="path_str", right_on="path", how="left",
                  suffixes=("", "_stat"))

    df["relative_image_path"] = [
        Path(p).resolve().relative_to(raw_root.resolve().parents[1]).as_posix()
        for p in df["path_str"]
    ]
    df["original_filename"] = [Path(p).name for p in df["path_str"]]
    df["species_if_known"] = df["dataset"].map(SPECIES)

    # Deterministic record ids: sorted by (dataset, relative path)
    df = df.sort_values(["dataset", "relative_image_path"]).reset_index(drop=True)
    code = {"ShrimpDiseaseImageBD": "SD", "TigerShrimpBD": "TG", "BiomassVannamei": "BM"}
    df["record_id"] = [f"{code[d]}-{i:06d}" for i, d in enumerate(df["dataset"])]

    # ---- disease label harmonization (masked, never invented) ----
    # Built as whole columns rather than cell-by-cell: pandas 2.x refuses to write
    # a string into a float64 column, and a 6k-row .at loop is needlessly slow.
    df["health_status"] = pd.Series([None] * len(df), dtype=object)
    for a in DISEASE_ATTRS:
        df[a] = np.nan
        df[f"label_mask_{a}"] = np.nan

    dis_mask = (df["task"] == "disease").values
    health = np.array([None] * len(df), dtype=object)
    attrs = {a: np.full(len(df), np.nan) for a in DISEASE_ATTRS}
    masks = {a: np.full(len(df), np.nan) for a in DISEASE_ATTRS}

    for i in np.nonzero(dis_mask)[0]:
        h = harmonize(df.iat[i, df.columns.get_loc("dataset")],
                      df.iat[i, df.columns.get_loc("raw_class")])
        health[i] = h.health_status
        for a in DISEASE_ATTRS:
            attrs[a][i] = h.attrs[a]
            masks[a][i] = h.masks[a]

    df["health_status"] = health
    for a in DISEASE_ATTRS:
        df[a] = attrs[a]
        df[f"label_mask_{a}"] = masks[a]

    for c in ("length_cm", "weight_g", "cephalothorax_cm", "complete_body"):
        if c not in df.columns:
            df[c] = np.nan
    if "quality_flags" not in df.columns:
        df["quality_flags"] = ""
    df["quality_flags"] = df["quality_flags"].fillna("")

    # ---- exclusions: unreadable images cannot be used, and are recorded ----
    df["exclude"] = ~df["readable"].fillna(False)
    df["exclude_reason"] = np.where(df["exclude"], "unreadable_or_corrupt:" + df["error"].fillna(""), "")

    # ---- grouping: exact dup + augmentation family + BLOCKED near-dup ----
    usable = df[~df["exclude"]].copy()
    fam = []
    for _, r in usable.iterrows():
        if r["dataset"] == "TigerShrimpBD":
            fam.append(r.get("augmentation_family_key"))
        elif r["dataset"] == "ShrimpDiseaseImageBD":
            # a raw image and its annotated copy share a stem -> one family
            fam.append(f"sdibd::{r['raw_class']}::{Path(r['original_filename']).stem.lower()}")
        else:
            fam.append(None)

    # Near-duplicate fusion is blocked by dataset+class (disease) or by individual
    # (biomass) so unrelated images that merely share a background never merge.
    blocks = np.array([
        f"{r['dataset']}|{r['raw_class'] if r['task'] == 'disease' else r['individual_id']}"
        for _, r in usable.iterrows()
    ])

    # every view of one INDIVIDUAL (biomass shrimp or SDIBD animal) is forced into
    # a single group
    extra = forced_individual_links(usable)

    ph = usable["phash"].fillna("0")
    ph_u64 = np.array([np.uint64(int(str(x), 16)) for x in ph], dtype=np.uint64)
    mapping, dedup_diag = build_fused_groups(
        record_ids=list(usable["record_id"]),
        sha256s=list(usable["sha256"].fillna("")),
        phash_u64=ph_u64,
        family_keys=fam,
        phash_threshold=phash_threshold,
        near_dup_blocks=blocks,
        extra_links=extra,
    )
    diagnostics["dedup"] = dedup_diag
    df["group_id"] = df["record_id"].map(mapping)

    # exact-duplicate cluster id (sha256 identity only) -- reported separately
    sha_to_cluster = {}
    for k, s in enumerate(sorted(df.loc[~df["exclude"], "sha256"].dropna().unique())):
        sha_to_cluster[s] = f"dup_{k:06d}"
    df["duplicate_cluster_id"] = df["sha256"].map(sha_to_cluster)
    df["augmentation_family_id"] = pd.Series(fam, index=usable.index).reindex(df.index)

    df["split"] = ""
    df["fold"] = -1

    for c in MANIFEST_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    out = df[MANIFEST_COLUMNS].copy()
    return out, diagnostics
