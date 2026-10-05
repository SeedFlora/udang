"""What 'superseded' means for the round-1 revision: runs trained on the
file-level SDIBD grouping (their protocol's assignment changed) or runs under
the retired E9_ naming scheme. Shared by the archive tool and the phase driver
so the two can never disagree.

The protocol alone does not decide it: the affected protocols were re-run on the
rebuilt grouping, so a row in one of them is superseded only if the split hash it
recorded differs from the hash `data/splits/splits_hash.txt` carries now. Keying
on the protocol alone made the driver's stale-artefact interlock flag the freshly
completed Phase B runs and refuse the next launch.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import SPLITS

SUPERSEDED_PROTOCOLS = frozenset({"disease_cv", "disease_cv_withaug", "disease_cv_sdibd",
                                  "cross_sd2tg", "cross_tg2sd"})
RETIRED_PREFIXES = ("E9_",)
SPLITS_HASH_FILE = SPLITS / "splits_hash.txt"


def read_current_split_hash() -> str | None:
    """The hash of the splits table on disk, or None when it cannot be read."""
    p = Path(SPLITS_HASH_FILE)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip() or None


def superseded_mask(registry: pd.DataFrame, current_split_hash: str | None = None,
                    protocols: frozenset[str] | set[str] = SUPERSEDED_PROTOCOLS,
                    prefixes: tuple[str, ...] = RETIRED_PREFIXES) -> pd.Series:
    """Boolean mask over registry rows that are superseded.

    A row is superseded iff its protocol is one whose assignment changed AND the
    split hash it was trained under is not the current one, or its run_name uses
    a retired naming scheme. A missing/NaN split hash counts as superseded: a run
    that did not record which grouping it saw cannot be shown to be current. When
    no current hash is available at all, every row in a superseded protocol is
    treated as superseded (the conservative direction: it blocks, never permits).
    """
    cur = current_split_hash if current_split_hash is not None else read_current_split_hash()

    by_proto = registry["protocol"].isin(protocols)
    if cur is None or "split_hash" not in registry.columns:
        stale_hash = pd.Series(True, index=registry.index)
    else:
        col = registry["split_hash"]
        stale_hash = ~(col.notna() & col.astype(str).str.strip().eq(cur))

    if prefixes:
        by_name = registry["run_name"].astype(str).str.startswith(tuple(prefixes))
    else:
        by_name = pd.Series(False, index=registry.index)
    return (by_proto & stale_hash) | by_name
