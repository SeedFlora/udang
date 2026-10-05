# Numerical supplement

This supplement contains numerical results for ShrimpVision-MTL. Each table is
provided as CSV for analysis and as JSON with technical definitions, per-seed
results, and input provenance. The tables come from the verified artifacts in
[`outputs/reviewer/`](../outputs/reviewer/). CSV files are unchanged copies.
JSON preserves every finite numeric value and uses `null` for unavailable or
undefined values that the source JSON represented as `NaN` or infinity. This
keeps the supplement JSON readable by strict JSON parsers; CSV missing values
remain blank.

The source code and inputs are pinned to commit
[`b12bacd`](https://github.com/SeedFlora/udang/tree/b12bacda98dab5ba8e1d47b86698262f153accfe).
`provenance.json` records the full source commit, table shapes, analysis parameters,
and source and supplement file hashes. `CHECKSUMS.sha256` covers all files except the
checksum list itself. The separately generated JSON metadata may report
`git_head: unknown`, because those results were produced before the initial
export commit; the pinned commit in `provenance.json` identifies their published
code and input snapshot.

## Tables

| Table | Rows | Contents |
| --- | ---: | --- |
| [t_biomass_protocols.csv](t_biomass_protocols.csv) / [JSON](t_biomass_protocols.json) | 13 | Animal-level biomass results by held-out unit for leave-one-database-out (LODO), leave-one-site-out (LOSO), and within-database protocols. |
| [t_tabular_tier.csv](t_tabular_tier.csv) / [JSON](t_tabular_tier.json) | 5 | Length-only random forests compared with length plus out-of-fold image features for each held-out database. |
| [t_ablation_stats.csv](t_ablation_stats.csv) / [JSON](t_ablation_stats.json) | 18 | Paired disease and biomass comparisons, seed variation, cluster bootstrap intervals, and multiplicity-adjusted significance. |

## Reading the results

The prediction seeds are **42, 2026, and 3407**. Bootstrap and random-forest
calculations use seed **42**; bootstrap intervals use **2,000 replicates**.

For biomass, each animal's prediction is the median of its test-view weight
predictions in grams. Image metrics average the three per-seed results. MAE is in
grams and lower is better; higher R-squared is better. The database-centered
R-squared uses within-database target variation in its denominator. Sampling
intervals resample individuals and condition on the fitted runs; seed standard
deviation is reported separately. The `db_mean` predictor uses held-out labels
and is an oracle reference. Forest and allometric references use manual
measurements, which the image-only model does not receive.

In the tabular tier, `delta_mae` equals the length-plus-image MAE minus the
length-only MAE. A negative value favors the added image features. This table
contains no confidence intervals or p-values; its JSON includes per-seed results.

In the paired comparisons, the delta is **metric(A) minus metric(B)**. A positive
delta favors A for disease balanced accuracy, evaluated at threshold 0.50. A
negative delta favors A for biomass MAE. `ci_pooled_lo` and `ci_pooled_hi` describe
the seed-mean sampling interval; `ci_union_lo` and `ci_union_hi` form the envelope
of the per-seed intervals. Holm correction is applied within the disease and
biomass families to a directional worst-seed test. The pooled interval and
adjusted p-value target different uncertainty and can disagree. Inspect
`p_floor` and `p_holm_at_floor` before treating a reported p-value as exact.
Interpret the secondary Wilcoxon result only where `wilcoxon_valid` is true.

## Recompute

From the repository root, after installing the pinned dependencies:

```bash
python scripts/reviewer/biomass_analysis.py --n-boot 2000 --seed 42 --pred-seeds "42,2026,3407"
python scripts/reviewer/stats_rehab.py --n-boot 2000 --seed 42 --seeds 42 2026 3407
```

These commands regenerate the tables under `outputs/reviewer/`. Installation,
saved-input locations, and historical training reproducibility limits are
documented in [`REVIEWER_GUIDE.md`](../REVIEWER_GUIDE.md). Code and dataset
licensing are described in [`LICENSE_NOTES.md`](../LICENSE_NOTES.md).

This folder contains numerical tables and documentation. Manuscripts,
supplementary PDFs or Word documents, reviewer correspondence, credentials,
environment files, source images, and model weights are excluded.
