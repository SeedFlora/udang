"""Parsing of the Litopenaeus vannamei biomass dataset (images + Excel measurements).

Two things must be exactly right here or the biomass protocol silently breaks:

  1. individual identity -- images are named `s<m>_<n>` where m indexes the shrimp
     and n indexes one photo of that shrimp. Every image of one shrimp must land in
     the same fold, so `individual_id` is the grouping key for all biomass splits.
  2. measurement joining -- the Excel SAMPLE column indexes the same m. A join that
     silently drops rows would fabricate a smaller dataset, so unmatched images and
     unmatched measurements are both reported rather than dropped quietly.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# `s12_3.png` -> individual 12, view 3. Tolerates upper case and stray spaces.
IMAGE_NAME_RE = re.compile(r"^\s*s\s*(?P<ind>\d+)\s*[_\-]\s*(?P<view>\d+)\s*$", re.I)

# Header aliases seen across the five DB spreadsheets.
_COLUMN_ALIASES = {
    "sample": "sample",
    "muestra": "sample",
    "length": "length_cm",
    "length (cm)": "length_cm",
    "longitud": "length_cm",
    "largo": "length_cm",
    "weight": "weight_g",
    "weight (g)": "weight_g",
    "peso": "weight_g",
    "cephalothorax": "cephalothorax_cm",
    "cephalothorax (cm)": "cephalothorax_cm",
    "cefalotorax": "cephalothorax_cm",
    "complete shrimp images": "complete_body",
    "complete shrimp image": "complete_body",
    "complete": "complete_body",
    "imagenes completas": "complete_body",
}


def _norm_header(h) -> str:
    s = " ".join(str(h).strip().lower().replace("_", " ").split())
    s = s.replace("(cm)", "(cm)").replace("(g)", "(g)")
    return s


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map spreadsheet headers onto canonical names. Unknown headers are ignored
    but reported by the caller through `unmapped`."""
    mapping: dict[str, str] = {}
    for col in df.columns:
        key = _norm_header(col)
        if key in _COLUMN_ALIASES:
            mapping[col] = _COLUMN_ALIASES[key]
            continue
        # tolerate "LENGTH (cm)" style by stripping the unit parenthetical
        base = re.sub(r"\s*\(.*?\)\s*", "", key).strip()
        if base in _COLUMN_ALIASES:
            mapping[col] = _COLUMN_ALIASES[base]
    return mapping


def parse_measurement_sheet(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read one DB spreadsheet into canonical columns.

    Returns (dataframe, diagnostics). The dataframe always has a `sample` column
    plus whichever of length_cm / weight_g / cephalothorax_cm / complete_body exist.
    """
    raw = pd.read_excel(path)
    # Some sheets carry a blank leading row or an index column; drop all-NaN cols.
    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")
    mapping = resolve_columns(raw)
    unmapped = [c for c in raw.columns if c not in mapping]

    if "sample" not in mapping.values():
        raise ValueError(
            f"{path.name}: no SAMPLE column found (headers={list(raw.columns)}). "
            f"Individual identity cannot be reconstructed -- refusing to guess."
        )

    df = raw.rename(columns=mapping)[list(dict.fromkeys(mapping.values()))].copy()
    df["sample"] = pd.to_numeric(df["sample"], errors="coerce")
    df = df.dropna(subset=["sample"])
    df["sample"] = df["sample"].astype(int)

    for c in ("length_cm", "weight_g", "cephalothorax_cm"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "complete_body" in df.columns:
        df["complete_body"] = pd.to_numeric(df["complete_body"], errors="coerce")

    diag = {
        "file": path.name,
        "n_rows": int(len(df)),
        "columns_mapped": mapping,
        "columns_unmapped": unmapped,
        "n_duplicate_samples": int(df["sample"].duplicated().sum()),
        "has_cephalothorax": "cephalothorax_cm" in df.columns,
        "has_complete_body": "complete_body" in df.columns,
    }
    return df, diag


def parse_image_name(filename: str) -> tuple[int | None, int | None]:
    """`s12_3.png` -> (12, 3). Returns (None, None) when the pattern does not hold."""
    m = IMAGE_NAME_RE.match(Path(filename).stem)
    if not m:
        return None, None
    return int(m.group("ind")), int(m.group("view"))


def individual_id(db: str, sample: int) -> str:
    """Globally unique individual key. DB-scoped because sample numbering restarts
    at 1 in every DB -- a bare sample number would silently merge different shrimp
    across ponds."""
    return f"{db}_s{int(sample):03d}"


def plausibility_flags(row: pd.Series) -> list[str]:
    """Physically implausible measurements, flagged rather than deleted."""
    flags: list[str] = []
    w = row.get("weight_g")
    ln = row.get("length_cm")
    ce = row.get("cephalothorax_cm")
    if pd.notna(w) and (w <= 0 or w > 200):
        flags.append("weight_out_of_range")
    if pd.notna(ln) and (ln <= 0 or ln > 40):
        flags.append("length_out_of_range")
    if pd.notna(ce) and (ce <= 0 or ce > 20):
        flags.append("cephalothorax_out_of_range")
    if pd.notna(ce) and pd.notna(ln) and ce >= ln:
        flags.append("cephalothorax_ge_length")
    if pd.notna(w) and pd.notna(ln) and ln > 0:
        # crude allometric sanity band: W = a * L^b with generous a, b bounds
        ratio = w / (ln ** 3)
        if ratio < 1e-4 or ratio > 0.2:
            flags.append("weight_length_ratio_extreme")
    return flags
