# Reviewer guide

This repository contains the ShrimpVision-MTL implementation, automated tests,
dataset provenance and grouping, split assignments, saved predictions, and
numerical experiment results. It is a clean export of the execution package;
historical commit IDs in the run registry describe the original experiments and
are retained as provenance, rather than as available commits in this repository.

## Install

Use Python 3.11 or later; the recorded runs used Python 3.13.5. Create and activate
a virtual environment, then install the pinned dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-lock.txt
```

The recorded training configuration uses CUDA 12.6. Tests and saved-prediction
analyses do not require training or the original dataset images.

## Verify the implementation

```bash
python -m pytest tests/ -q
```

The tests exercise label masks, leakage-safe group and individual splits, model
outputs, losses, metrics, checkpoint round-trips, and experiment management.

## Inspect and recompute saved results

Start with `reports/DATASET_AUDIT.md`, `reports/EXPERIMENT_REPORT.md`, and
`reports/REPRODUCIBILITY.md`. Machine-readable results are in `outputs/metrics/`;
the run registry identifies the protocol, fold, seed, hashes, and run provenance.

```bash
python scripts/evaluate.py --n-boot 2000
```

This command reads the saved prediction Parquet files, the unified manifest, and
the per-run validation thresholds. It rewrites the generated evaluation outputs
in `outputs/metrics/`. A quick execution check can use `--n-boot 10`; its intervals
are only a smoke check and should not be used as reported results.

Per-run metadata retains scientific configuration, thresholds, hashes, software
versions, and GPU model information needed to interpret the experiments. Private
system metadata, free-text run notes, errors, and console logs are omitted.
`outputs/checkpoints/` contains training-history JSON only. Raw dataset images,
cached images, trained weights, environment files, internal correspondence,
manuscripts, templates, and the original Git history are excluded.

The standalone analyses in `scripts/reviewer/` provide the final seed-aware
biomass and statistical comparisons from the same saved artifacts. Their numeric
outputs are kept separately in `outputs/reviewer/`; the generic pooled evaluation
above and those final analyses have different aggregation rules.

```bash
python scripts/reviewer/biomass_analysis.py --n-boot 2000 --seed 42 --pred-seeds "42,2026,3407"
python scripts/reviewer/stats_rehab.py --n-boot 2000 --seed 42 --seeds 42 2026 3407
```

These commands generate `t_biomass_protocols`, `t_tabular_tier`, and
`t_ablation_stats` as CSV and JSON tables. The JSON records aggregation rules,
bootstrap parameters, per-seed results, and source artifact paths.

The published copies of these three tables are collected in
[`supplement/`](supplement/) together with their definitions, provenance, and
checksums. CSV files match `outputs/reviewer/` byte for byte; supplement JSON
preserves finite values and represents unavailable values as `null`.

## Reproduce training

Download and verify the public datasets before running image-based stages:

```bash
python scripts/download_data.py
python scripts/build_manifests.py --workers 10
python scripts/audit_data.py
python scripts/build_splits.py --seed 42
python scripts/build_image_cache.py
```

The manifests and splits in this repository permit inspection of grouping and
assignments without redistributing the source images. The previous split snapshot
in `data/splits_round1_submitted/` supports comparison with the corrected grouping;
current experiments use `data/splits/`.

The current experiment driver covers the corrected multi-seed grid, including
backbone comparisons (E10), the augmentation leakage control (E11), leave-one-site
out biomass evaluation (E12), and within-database biomass evaluation (E13):

```bash
python scripts/run_revision.py --phase B
python scripts/run_revision.py --phase C
python scripts/run_revision.py --phase D
```

Use `python scripts/run_revision.py --help` for phases and dry-run options. The
original E0-E9 matrix remains documented in `README.md` and `Makefile`. On Linux
or macOS, invoke Make targets with `PY=.venv/bin/python`.

## Reproducibility limits

Read the provenance caveats in `reports/REPRODUCIBILITY.md`: some historical runs
were made from modified working trees, and the saved E5 multi-view results used
an earlier unstable sampling seed. The current implementation uses a stable
SHA-256 seed; exact regeneration of every historical training result is therefore
not claimed. Saved-prediction analyses can be checked without retraining.

Dataset acquisition contracts and citations are in `configs/data/sources.yaml`
and `LICENSE_NOTES.md`. Dataset files remain available from their original
publishers and are not redistributed here.
