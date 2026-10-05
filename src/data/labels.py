"""Hierarchical disease-label harmonization across the two disease datasets.

Design rule (from the project specification, section 7): an image that a dataset
does not annotate for a given disease is UNKNOWN, never a confirmed negative.
Unknown entries carry a supervision mask of 0 and contribute exactly zero loss.

ShrimpDiseaseImageBD annotates Black Gill and WSSV only -- it has no Yellow Head
category at all, so Yellow Head is unknown for every one of its images. Treating
those as yellow_head=0 would inject ~1.1k fabricated negatives into a label the
dataset never assessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Canonical attribute order. Every model head, metric and threshold vector in the
# project indexes attributes in exactly this order.
DISEASE_ATTRS: tuple[str, ...] = ("black_gill", "wssv", "yellow_head")
N_ATTRS = len(DISEASE_ATTRS)

# Health head is binary: 1 == diseased (positive class = "disease present").
# Encoding disease as the positive class keeps precision/recall oriented towards
# the clinically important event.
HEALTH_POSITIVE_IS_DISEASED = True


@dataclass(frozen=True)
class HarmonizedLabel:
    health_status: str                 # "healthy" | "diseased"
    diseased: int                      # 1 if diseased
    attrs: dict[str, int] = field(default_factory=dict)       # 0/1, meaningless where mask==0
    masks: dict[str, int] = field(default_factory=dict)       # 1 == supervised, 0 == unknown
    raw_class: str = ""
    source: str = ""

    def as_row(self) -> dict:
        row = {
            "health_status": self.health_status,
            "diseased": self.diseased,
            "raw_class": self.raw_class,
        }
        for a in DISEASE_ATTRS:
            row[a] = self.attrs.get(a, 0)
            row[f"label_mask_{a}"] = self.masks.get(a, 0)
        return row


# --------------------------------------------------------------------------- #
# Raw-class normalisation (guards against spelling / case / separator variants)
# --------------------------------------------------------------------------- #
_SDIBD_ALIASES = {
    "healthy": "Healthy",
    "bg": "BG",
    "wssv": "WSSV",
    "bg_wssv": "BG_WSSV",
    "bgwssv": "BG_WSSV",
    "bg wssv": "BG_WSSV",
    "bg-wssv": "BG_WSSV",
    "blackgill": "BG",
    "black_gill": "BG",
}

_TIGER_ALIASES = {
    "healthy": "Healthy",
    "black gill": "Black Gill",
    "black_gill": "Black Gill",
    "blackgill": "Black Gill",
    "bg": "Black Gill",
    "white spot syndrome virus": "WSSV",
    "white_spot_syndrome_virus": "WSSV",
    "whitespotsyndromevirus": "WSSV",
    "wssv": "WSSV",
    "white spot": "WSSV",
    "yellow head": "Yellow Head",
    "yellow_head": "Yellow Head",
    "yellowhead": "Yellow Head",
    "yh": "Yellow Head",
}


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("-", " ").replace("_", " ").split())


def normalize_raw_class(source: str, raw: str) -> str:
    key = _norm(raw)
    table = _SDIBD_ALIASES if source == "ShrimpDiseaseImageBD" else _TIGER_ALIASES
    # try both the spaced and underscored forms
    for candidate in (key, key.replace(" ", "_"), key.replace(" ", "")):
        if candidate in table:
            return table[candidate]
    raise KeyError(f"Unrecognised raw class {raw!r} for source {source!r}. "
                   f"Refusing to guess a label mapping.")


# --------------------------------------------------------------------------- #
# Dataset-specific mappings (specification section 7.2)
# --------------------------------------------------------------------------- #
# ShrimpDiseaseImageBD: Yellow Head is NOT a category in this dataset -> unknown.
_SDIBD_MAP = {
    #              diseased  bg  wssv  yh     mask_bg mask_wssv mask_yh
    "Healthy":    (0,        0,  0,    0,     1,      1,        0),
    "BG":         (1,        1,  0,    0,     1,      1,        0),
    "WSSV":       (1,        0,  1,    0,     1,      1,        0),
    "BG_WSSV":    (1,        1,  1,    0,     1,      1,        0),
}

# TigerShrimpBD: all four categories present -> every attribute is supervised.
_TIGER_MAP = {
    "Healthy":     (0, 0, 0, 0,   1, 1, 1),
    "Black Gill":  (1, 1, 0, 0,   1, 1, 1),
    "WSSV":        (1, 0, 1, 0,   1, 1, 1),
    "Yellow Head": (1, 0, 0, 1,   1, 1, 1),
}

_TABLES = {
    "ShrimpDiseaseImageBD": _SDIBD_MAP,
    "TigerShrimpBD": _TIGER_MAP,
}


def harmonize(source: str, raw_class: str) -> HarmonizedLabel:
    """Map a dataset-native class string to the shared hierarchical label space."""
    if source not in _TABLES:
        raise KeyError(f"No harmonization table for source {source!r}")
    canonical = normalize_raw_class(source, raw_class)
    diseased, bg, wssv, yh, m_bg, m_wssv, m_yh = _TABLES[source][canonical]
    return HarmonizedLabel(
        health_status="diseased" if diseased else "healthy",
        diseased=diseased,
        attrs={"black_gill": bg, "wssv": wssv, "yellow_head": yh},
        masks={"black_gill": m_bg, "wssv": m_wssv, "yellow_head": m_yh},
        raw_class=canonical,
        source=source,
    )


# Categories that are semantically aligned across BOTH disease datasets. Used for
# the cross-dataset protocol (section 8.2), which must not be scored on classes
# that only one dataset can express.
CROSS_DATASET_COMMON_CLASSES = ("Healthy", "Black Gill", "WSSV")
CROSS_DATASET_COMMON_ATTRS = ("black_gill", "wssv")


def is_cross_dataset_eligible(source: str, raw_class: str) -> bool:
    """True when a record belongs to the common Healthy / BG / WSSV space.

    BG_WSSV co-infection is excluded because TigerShrimpBD has no co-infection
    category and so cannot express it; Yellow Head is excluded because
    ShrimpDiseaseImageBD never assesses it.
    """
    canonical = normalize_raw_class(source, raw_class)
    if source == "ShrimpDiseaseImageBD":
        return canonical in ("Healthy", "BG", "WSSV")
    return canonical in ("Healthy", "Black Gill", "WSSV")
