# Reproducibility

This is the original generated experiment record. The reviewer export retains
configuration, validation thresholds, software versions, and GPU model metadata,
but omits private system fields and free-text runtime notes. See
[`REVIEWER_GUIDE.md`](../REVIEWER_GUIDE.md) for the current reproduction routes.

Generated: 2026-09-15T10:39:42.724349+00:00


## Recorded per run

Every run writes `outputs/logs/<run>_meta.json` containing: configuration, random seed, git commit and dirty flag, manifest hash, split hash, full environment and hardware description, start/end timestamps, duration, best epoch, best validation metric, selected per-label thresholds, and paths to the checkpoint, predictions, metrics and training curve. One row per completed run is appended to `outputs/metrics/experiment_registry.csv`.


Registry currently holds **240 runs** (240 completed).

| Field | Across all runs |
|---|---|
| runs | 240 |
| manifest_hash | f9c5ee92c2301668 (all 240 runs identical) |
| split_hash | 2 DISTINCT: 8cea4f217d48, b3943dc4ec3f |
| git_dirty | False: 228, True: 12 |
| git_commit | 55 distinct: 6cfd72bx43, bf541bbx27, b186108x23, 511bb0ax11, 1b2c732x11, dfeb985x8, 8452663x6, 7dad1c5x6, bcbf9e8x5, 2f92d24x5, 6ddd09ex5, c5cb87bx5, 6c3a631x4, 6820018x4, e78fba5x4, 6afd72bx4, 3eeac0fx3, a767f03x3, 1989b8ax3, b8e6e9ex3, 9cd3b15x3, 7e6171fx3, 284d117x3, 89ba362x3, 71ed2bbx3, fd933a2x3, 6af6d75x3, ce57c87x2, e64ffb9x2, 3b544a0x2, ddff9c1x2, e70838cx2, e68c569x2, 5f61920x2, fcf3660x2, 5f1ebcfx1, f6f88afx1, 4bfe9d0x1, 1ec9488x1, 8665528x1, 58ab437x1, f53a33cx1, 18510e6x1, 65d34e0x1, a6e4285x1, e9da6f9x1, 201d0a4x1, 36edeeax1, 5a96ccbx1, 8ba2e74x1, 8d0b74cx1, 82226d5x1, 203d099x1, 19f1703x1, 39b815bx1 |
| torch | 2.13.0+cu126 |
| gpu | NVIDIA GeForce RTX 3070 Ti Laptop GPU |

**Provenance caveat.** All 240 runs consumed the data recorded above. However 12 of 240 runs were trained from a git-DIRTY working tree spanning 55 commits, so their exact training code is not pinned to a clean commit. Every run's commit and dirty flag is recorded in the committed `experiment_registry.csv` (so any run is traceable), but checking out a single commit will not regenerate the whole matrix bit-for-bit.


## Determinism

`set_global_seed` seeds Python, NumPy and torch (CPU and CUDA) and enables cuDNN deterministic mode. Per-fold seeds are `seed + fold`. The multi-task batch interleave is generated from a seeded permutation, so the task schedule is reproducible. Multi-view sampling is seeded by a stable SHA-256 of (seed, epoch, individual) via `stable_seed`, so the per-epoch view draw reproduces across processes.


**Historical note (multi-view / E5):** the E5 predictions currently on disk were produced by an earlier build that seeded the multi-view draw from Python's per-process-salted `hash()`, which is NOT stable across interpreters; those specific numbers are therefore not regenerable by a re-run. The seed was replaced with `stable_seed` (SHA-256); regenerate E5 to obtain reproducible multi-view results.


Residual non-determinism: some cuDNN/cuBLAS kernels and atomics are not bit-reproducible even in deterministic mode, and DataLoader worker ordering can vary. Metrics should reproduce to within a few units in the third decimal, not bitwise.


## Metrics are recomputable without inference

`scripts/evaluate.py` loads no checkpoint. It reads the saved prediction parquet files and recomputes every headline number, including confidence intervals and paired comparisons. This is the check that the reported numbers are properties of saved artefacts rather than of a live model.


Prediction files on disk: **432**.


## Data integrity

| Dataset | DOI | Files | SHA-256 vs publisher |
|---|---|---|---|
| BiomassVannamei | 10.17632/h8tcn6ykky.2 | 1 | all match |
| ShrimpDiseaseImageBD | 10.17632/jhrtdj9txm.3 | 4 | all match |
| TigerShrimpBD | 10.17632/9dj4sk5d55.1 | 4575 | all match |

## Preflight

| Status | Check | Detail |
|---|---|---|
| PASS | disease_loader | batch image (8, 3, 224, 224), attrs (8, 3), finite=True |
| PASS | mask_semantics_in_manifest | every ShrimpDiseaseImageBD row has yellow_head masked as unknown |
| SKIP | biomass_loader | no biomass split rows; manifest holds 0 biomass records. Biomass experiments (E4-E7) cannot run unti |
| PASS | masked_losses_zero_and_finite | all-zero mask gives exact 0.0; inf at a masked slot does not propagate |
| PASS | overfit_20_samples | loss 1.4353 -> 0.0000, health accuracy 1.00 (must drop >65% and reach >=0.90) |
| PASS | one_epoch_smoke | epoch completed, predictions and metrics written, health bal_acc=0.500 (190s) |
| PASS | split_leakage | 6 protocols / 22 folds: no group_id appears in more than one split |

## Environment

Exact pinned versions are in `requirements-lock.txt`. Key versions:

- Python 3.13.5
- torch 2.13.0+cu126 / CUDA 12.6
- timm 1.0.28, albumentations 2.0.8, scikit-learn 1.9.0
