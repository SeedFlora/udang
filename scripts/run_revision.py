"""Drive the round-1 revision grid in stoppable phases.

  B  seed 42, disease, profile standard3   (corrected grouping; backbone sweep; naive arm)
  C  seed 42, biomass, profile standard5   (LODO folds 3-4, LOSO, within-DB)
  D  seeds 2026 + 3407 for everything above

Resume is free: run_experiment skips any fold whose checkpoint AND metrics exist,
so a phase can be relaunched after an interruption. --dry-run prints the exact
(run, exists, estimated seconds) table without touching the GPU.

A run is also refused before launch if its protocol has no rows yet in
data/splits/splits.parquet (split_ok column) -- this catches the case where a
new protocol (e.g. biomass_loso) hasn't been built into the split table yet.

A run is also refused if it is still named/protocol-matched to a superseded
(file-grouping-era) registry row that has not been archived yet -- this catches
the case where scripts/archive_superseded_runs.py has not been run for real, so
a stale checkpoint would otherwise be silently reused as "already complete".

Usage:
    python scripts/run_revision.py --phase B --dry-run
    python scripts/run_revision.py --phase B
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.registry import build_matrix  # noqa: E402
from src.training.run_matrix import run_configs  # noqa: E402
from src.training.superseded import superseded_mask  # noqa: E402
from src.utils.paths import CHECKPOINTS, METRICS, OUTPUTS, SPLITS  # noqa: E402

DISEASE = ["E1", "E2", "E3", "E6", "E7", "E8_lightweight_disease", "E10", "E11"]
BIOMASS = ["E4", "E5", "E8_lightweight_biomass", "E12", "E13"]
PHASES = {
    "B": [("standard3", DISEASE, [42])],
    "C": [("standard5", BIOMASS, [42])],
    "D": [("standard3", DISEASE, [2026, 3407]), ("standard5", BIOMASS, [2026, 3407])],
}
_RX = re.compile(r"_f(\d+)_s(\d+)$")


def _family_seconds() -> dict[str, float]:
    """Mean duration per config family from the registry, base rows only.

    Reads the live registry AND, once it exists, the archived-rows CSV that
    scripts/archive_superseded_runs.py writes -- otherwise every estimate for a
    family whose base rows got archived away would silently fall back to the
    generic 500s/family default.

    E9_ rows are excluded on purpose: the same (config, fold, seed) took 63 s as
    E9_E5_biomass_multiview_mean and 623 s as E5_biomass_multiview_mean, so a mean
    over both under-prices the family ~3x. Estimates are rough either way."""
    frames = []
    p = METRICS / "experiment_registry.csv"
    if p.exists():
        frames.append(pd.read_csv(p))
    archived = OUTPUTS / "_archive" / "round1_submitted" / "archived_registry_rows.csv"
    if archived.exists():
        frames.append(pd.read_csv(archived))
    if not frames:
        return {}
    reg = pd.concat(frames, ignore_index=True)
    reg = reg[~reg["run_name"].astype(str).str.startswith("E9_")]
    fam = reg["run_name"].astype(str).str.replace(_RX, "", regex=True)
    return reg.assign(fam=fam).groupby("fam")["duration_sec"].mean().to_dict()


def _split_pairs() -> set[tuple[str, int]]:
    """Every (protocol, fold) pair with train, val AND test rows in splits.parquet."""
    p = SPLITS / "splits.parquet"
    if not p.exists():
        return set()
    df = pd.read_parquet(p, columns=["protocol", "fold", "split"])
    need = {"train", "val", "test"}
    pairs: set[tuple[str, int]] = set()
    for (proto, fold), sub in df.groupby(["protocol", "fold"]):
        if need <= set(sub["split"].unique()):
            pairs.add((str(proto), int(fold)))
    return pairs


def _stale_planned(df: pd.DataFrame) -> list[str]:
    """Planned run_names that are also superseded rows in the live registry."""
    p = METRICS / "experiment_registry.csv"
    if not p.exists():
        return []
    reg = pd.read_csv(p)
    stale_names = set(reg.loc[superseded_mask(reg), "run_name"])
    return sorted(set(df["run_name"]) & stale_names)


def plan_runs(phase: str) -> pd.DataFrame:
    secs = _family_seconds()
    pairs = _split_pairs()
    fallback = {"E10": 600.0, "E11": 542.0, "E12": 505.0, "E13": 505.0}
    rows = []
    for profile, only, seeds in PHASES[phase]:
        for seed in seeds:
            for c in build_matrix(profile, seed=seed, only=only):
                for fold in c.folds:
                    rn = f"{c.name}_f{fold}_s{seed}"
                    exists = (CHECKPOINTS / f"{rn}_best.pt").exists() and (METRICS / f"{rn}.json").exists()
                    est_from_fallback = c.name not in secs
                    est = secs.get(c.name, fallback.get(c.experiment, 500.0))
                    if c.tasks == ("biomass",):
                        needed = [(c.biomass_protocol, fold)]
                    else:
                        needed = [(c.protocol, fold)]
                        if "biomass" in c.tasks:
                            needed.append((c.biomass_protocol, fold))
                    split_ok = all(pair in pairs for pair in needed)
                    rows.append({"run_name": rn, "config": c.name, "experiment": c.experiment,
                                 "fold": fold, "seed": seed, "profile": profile,
                                 "exists": bool(exists), "split_ok": bool(split_ok),
                                 "est_sec": float(est),
                                 "est_from_fallback": bool(est_from_fallback)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    df = plan_runs(args.phase)
    todo = df[~df["exists"]]
    print(df.to_string(index=False))
    print(f"\nphase {args.phase}: {len(df)} run keys, {len(todo)} to run, "
          f"{len(df) - len(todo)} already complete; "
          f"estimated {todo['est_sec'].sum() / 3600:.1f} h (registry means)")

    # Both guards must run before any training call. In --dry-run we report every
    # blocking condition we find (so a single dry run surfaces everything wrong
    # before someone reaches for the real command); for a real launch we fail
    # fast and raise on whichever guard trips first -- either is sufficient to
    # stop before run_configs.
    blocked = False

    missing = df[~df["exists"] & ~df["split_ok"]]
    if len(missing):
        blocked = True
        print("\nmissing split rows for:\n  " + "\n  ".join(missing["run_name"]))
        msg = (f"refusing to launch: {len(missing)} planned runs have no split "
               "rows — rebuild splits first")
        if args.dry_run:
            print(msg)
        else:
            raise SystemExit(msg)

    stale = _stale_planned(df)
    if stale:
        blocked = True
        print("\nplanned runs that still carry file-grouping-era artefacts:\n  "
              + "\n  ".join(stale))
        if args.dry_run:
            print(f"NOTE: {len(stale)} planned runs still carry file-grouping-era "
                  "artefacts; run scripts/archive_superseded_runs.py before launching")
        else:
            raise SystemExit(f"refusing to launch: {len(stale)} planned runs still carry "
                             "file-grouping-era artefacts; run "
                             "scripts/archive_superseded_runs.py before launching")

    if blocked:
        return 1   # only reachable in --dry-run: a real launch already raised above

    if args.dry_run:
        return 0
    for profile, only, seeds in PHASES[args.phase]:
        for seed in seeds:
            cfgs = build_matrix(profile, seed=seed, only=only)
            print(f"\n### profile={profile} seed={seed}: {len(cfgs)} configurations")
            run_configs(cfgs, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
