"""Move superseded run artefacts out of outputs/ so they cannot pollute the
revision's evaluation, and free the disk their checkpoints occupy.

Superseded = trained on the submitted (file-level) SDIBD grouping, or belonging
to the retired E9_ naming scheme. Every artefact is committed under the git tag
round1-submitted (except *.pt, which git ignores), so this is disk hygiene, not
deletion of evidence. Registry rows are dropped in place and saved beside the
archive; fold_metrics.csv / results_summary.csv are regenerated.

Re-running after a partial failure converges: files already moved are counted as
absent and the registry is rewritten only after all moves succeed.

Usage:
    python scripts/archive_superseded_runs.py --dry-run
    python scripts/archive_superseded_runs.py --delete-checkpoints
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.run_matrix import FOLD_METRICS, summarize  # noqa: E402
from src.training.superseded import (RETIRED_PREFIXES, SUPERSEDED_PROTOCOLS,  # noqa: E402
                                     read_current_split_hash, superseded_mask)
from src.utils.paths import METRICS, REPO_ROOT  # noqa: E402

REGISTRY = METRICS / "experiment_registry.csv"
DEFAULT_PROTOCOLS = sorted(SUPERSEDED_PROTOCOLS)


def select_superseded(registry: pd.DataFrame, protocols: set[str],
                      prefixes: tuple[str, ...]) -> pd.DataFrame:
    """Rows to archive. Shares `superseded_mask` with the phase driver, so the
    tool and the launch interlock can never disagree about what is stale: a row
    in one of `protocols` counts only when its split hash is not the current one."""
    return registry[superseded_mask(registry, protocols=protocols, prefixes=prefixes)].copy()


def artefact_paths(row: pd.Series, run_name: str) -> list[Path]:
    paths: set[str] = set()
    for col in ("checkpoint_path", "metrics_path"):
        v = str(row.get(col, "") or "")
        if v and v != "nan":
            paths.add(v)
    for v in str(row.get("predictions_path", "") or "").split(";"):
        if v and v != "nan":
            paths.add(v)
    paths |= {
        f"outputs/checkpoints/{run_name}_history.json",
        f"outputs/predictions/{run_name}_val.parquet",
        f"outputs/logs/{run_name}_meta.json",
        f"outputs/logs/{run_name}_FAILED.log",
    }
    return [Path(p) for p in sorted(paths)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="outputs/_archive/round1_submitted")
    ap.add_argument("--protocols", nargs="*", default=DEFAULT_PROTOCOLS)
    ap.add_argument("--prefix", nargs="*", default=list(RETIRED_PREFIXES))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--delete-checkpoints", action="store_true",
                    help="delete *.pt instead of moving them (frees disk; they are "
                         "superseded and git never tracked them)")
    args = ap.parse_args()

    reg = pd.read_csv(REGISTRY)
    if read_current_split_hash() is None:
        # Without a current hash every superseded-protocol row would count as stale;
        # the driver may refuse to launch on that, but the archiver must never move
        # the live grid on nothing worse than a missing text file.
        raise SystemExit("refusing to archive: data/splits/splits_hash.txt is missing or "
                         "empty; every superseded-protocol row would count as stale")
    sel = select_superseded(reg, set(args.protocols), tuple(args.prefix))
    keep = reg.drop(sel.index)
    dest = REPO_ROOT / args.dest
    outputs_root = (REPO_ROOT / "outputs").resolve()
    try:
        dest.resolve().relative_to(outputs_root)
    except ValueError:
        raise SystemExit(f"--dest must resolve under outputs/: {args.dest}")
    print(f"registry rows: {len(reg)}  superseded: {len(sel)}  kept: {len(keep)}")
    print("kept runs:\n  " + "\n  ".join(sorted(keep["run_name"])))

    moved = deleted = missing = 0
    freed = 0
    for _, row in sel.iterrows():
        for rel in artefact_paths(row, str(row["run_name"])):
            if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "outputs":
                raise SystemExit(f"refusing to touch a path outside outputs/: {rel}")
            src = REPO_ROOT / rel
            if not src.exists():
                missing += 1
                continue
            if src.suffix == ".pt" and args.delete_checkpoints:
                freed += src.stat().st_size
                if not args.dry_run:
                    src.unlink()
                deleted += 1
                continue
            if not args.dry_run:
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest / rel))
            moved += 1
    print(f"{'would move' if args.dry_run else 'moved'} {moved} files, "
          f"{'would delete' if args.dry_run else 'deleted'} {deleted} checkpoints "
          f"({freed / 1e9:.1f} GB), {missing} listed files were already absent")

    if args.dry_run:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    sel.to_csv(dest / "archived_registry_rows.csv", index=False)
    keep.to_csv(REGISTRY, index=False)
    if FOLD_METRICS.exists():
        fm = pd.read_csv(FOLD_METRICS)
        fm[~fm["run_name"].isin(set(sel["run_name"]))].to_csv(FOLD_METRICS, index=False)
    summarize()
    print(f"registry now {len(keep)} rows; archived rows -> {dest / 'archived_registry_rows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
