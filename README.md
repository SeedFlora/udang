# ShrimpVision-MTL

Reviewer package for code and numerical reproduction. Start with
[`REVIEWER_GUIDE.md`](REVIEWER_GUIDE.md) for saved-result verification and the
current E10-E13 experiment route. The commands below also document the original
E0-E9 pipeline.

The numerical supplement is available in [`supplement/`](supplement/), with
three result tables in CSV and JSON, provenance, and SHA-256 checksums.

Cross-dataset multi-task learning for shrimp disease recognition and non-invasive
biomass estimation.

A single shared visual encoder is trained to support four things at once:
healthy-versus-diseased recognition, multi-label disease identification
(Black Gill / WSSV / Yellow Head), shrimp length estimation, and shrimp weight
estimation - across three independently published datasets that do not share a
label space.

This repository is an experiment execution package, not a paper. Results live in
[`reports/EXPERIMENT_REPORT.md`](reports/EXPERIMENT_REPORT.md).

---

## The problem this repository actually solves

The three source datasets cannot simply be concatenated:

| | ShrimpDiseaseImageBD | TigerShrimpBD | Biomass (L. vannamei) |
|---|---|---|---|
| Task | disease | disease | regression |
| Classes | Healthy, BG, WSSV, BG_WSSV | Healthy, Black Gill, WSSV, Yellow Head | none |
| Yellow Head annotated? | **no** | yes | n/a |
| Co-infection expressible? | yes | **no** | n/a |
| Images | 1,149 raw (+746 duplicate annotated copies) | 1,001 originals + 3,574 published augmentations | 5,507 images of **170 animals** |

Three properties of this data make naive training silently wrong, and each is
handled explicitly:

1. **Unknown is not negative.** ShrimpDiseaseImageBD never assesses Yellow Head.
   Encoding those images as `yellow_head = 0` would fabricate ~1.1k negatives for
   a label the dataset never looked at. Every label carries a supervision mask;
   masked entries contribute exactly zero loss and zero gradient.

2. **Duplicates and derivatives must not straddle a split.** All 746 images in
   ShrimpDiseaseImageBD's "Annotated" folder are **byte-identical** (same SHA-256)
   to images in "Raw Images". TigerShrimpBD publishes 3,574 augmented derivatives
   of 1,001 originals. Splits are grouped on the transitive closure of
   exact-duplicate, augmentation-family and perceptual near-duplicate relations.

3. **5,507 biomass images are only 170 shrimp.** Treating images as independent
   would shrink every confidence interval by roughly the square root of 30.
   Biomass grouping is by individual, and all biomass intervals bootstrap over
   individuals.

## Quick start

```bash
make setup        # venv + pinned deps (CUDA 12.6 wheels)
make data         # download all three datasets, verify every SHA-256
make manifest     # unified manifest: hashes, pHash, groups, individuals
make audit        # mandatory audit - EXITS NON-ZERO on leakage-critical failure
make splits       # leakage-safe splits - RAISES if any group straddles a split
make preflight    # tests + 20-sample overfit + 1-epoch smoke + leakage check
make all          # everything, E0 through E9
```

`make audit` and `make splits` are gates, not formalities: the audit blocks the
pipeline when a leakage-critical check fails, and split construction raises
rather than returning a split that leaks.

## Layout

```
configs/     dataset registry with verification contracts asserted at download time
src/data/    labels + masks, manifest construction, splits, datasets, transforms, cache
src/audit/   duplicate/near-duplicate/augmentation-family clustering, audit checks
src/models/  shared-encoder multi-task model, attention pooling, MixStyle
src/training/masked losses, trainer, experiment runner, experiment registry
src/evaluation/ metrics, cluster bootstrap, non-neural biomass baselines
scripts/     one entry point per pipeline stage
reports/     DATASET_AUDIT.md, EXPERIMENT_REPORT.md, REPRODUCIBILITY.md
outputs/     checkpoints, predictions, metrics, figures, error analysis, logs
```

## Experiment matrix

| ID | Experiment |
|---|---|
| E0 | Data and leakage audit (no training) |
| E1 | Disease single-dataset baselines (ConvNeXt-Tiny, each dataset alone) |
| E2 | Pooled hierarchical disease model + loss / sampling / augmentation ablations |
| E3 | Cross-dataset transfer, both directions |
| E4 | Biomass single-image regression, leave-one-DB-out |
| E5 | Biomass multi-view (mean vs attention pooling) |
| E6 | Shared-encoder multi-task (fixed vs learned uncertainty weighting) |
| E7 | Domain generalization (MixStyle, CORAL, source-balanced sampling) |
| E8 | Lightweight MobileNetV3-Large + efficiency measurements |
| E9 | Final repeated runs at seeds 42 / 2026 / 3407 with bootstrap CIs |

Single-task and multi-task runs share one model class and one training loop; only
the loss mixture changes. That keeps "does multi-task help?" a controlled
one-variable comparison rather than a comparison of two codebases.

## Design decisions worth knowing

- **Metrics are recomputed from saved predictions.** `scripts/evaluate.py` loads
  no checkpoint. If a number in the report cannot be regenerated from the parquet
  files in `outputs/predictions/`, that is a bug.
- **Thresholds are selected on validation only** and saved with the model.
- **Non-neural baselines come first.** Mean, allometric (`W = a * L^b`) and
  tabular models are evaluated before any deep model. The allometric and tabular
  baselines consume ground-truth manual length, which an image-only model never
  sees; every result records that information advantage explicitly so they are
  never presented as like-for-like.
- **Augmentations are chosen to preserve the disease sign.** Black Gill, WSSV and
  Yellow Head are colour- and texture-dependent, so aggressive colour transforms
  are an ablation, never a default. Biomass augmentation never crops body
  endpoints and never distorts aspect ratio.
- **Images are pre-resized to a 448 px cache.** ShrimpDiseaseImageBD ships every
  image at 2048x2048, so JPEG decode - not the GPU - dominated epoch time.

## Data and licensing

All three datasets are CC BY 4.0 and are **not** redistributed here;
`scripts/download_data.py` fetches them from their DOIs and verifies each file
against the publisher-declared SHA-256. Licence terms, required citations and two
recorded conflicts (a public-domain claim contradicting a CC BY licence field, and
a dataset version that is not retrievable at the requested version) are documented
in [`LICENSE_NOTES.md`](LICENSE_NOTES.md).

## Tests

```bash
.venv/Scripts/python -m pytest tests/ -q
```

Covers label mapping and mask semantics, duplicate/individual group splitting,
the no-overlap guarantee (including a test that the leakage guard actually fires
on injected leakage), spreadsheet parsing, model output dimensions, masked
losses, metric calculation, threshold selection, multi-view batching and
checkpoint round-trip.
