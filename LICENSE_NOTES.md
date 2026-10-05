# Licence notes

This file records the licence terms of every dataset used, the source of each
statement, and any conflicts found. Licence values were captured live from the
Mendeley Data public API at acquisition time and stored verbatim in
`data/manifests/acquisition_record.json`; nothing here is assumed or inferred.

## Code

The code in this repository is released under the MIT licence. This does **not**
extend to the datasets, which carry their own terms below.

## Datasets

All three datasets are published on Mendeley Data under
**Creative Commons Attribution 4.0 International (CC BY 4.0)**
(<http://creativecommons.org/licenses/by/4.0>).

CC BY 4.0 permits use, redistribution and adaptation, including commercially,
provided attribution is given and changes are indicated. Research use as
performed here is permitted. Attribution is therefore **required** for every
dataset, and the citations below must accompany any reuse of results.

| Dataset | Version used | DOI | Licence | Licence source |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 3 | `10.17632/jhrtdj9txm.3` | CC BY 4.0 | Mendeley Data API field `data_licence` |
| TigerShrimpBD | 1 | `10.17632/9dj4sk5d55.1` | CC BY 4.0 | Mendeley Data API field `data_licence` |
| Litopenaeus vannamei biomass dataset | 2 | `10.17632/h8tcn6ykky.2` | CC BY 4.0 | Mendeley Data API field `data_licence` |

### Required citations

- Islam, M. M.; Sarker, A.; Choudhury, A.; Ahmed, N.; Rasel, A. A. S.
  *ShrimpDiseaseImageBD: An Image Dataset for Computer Vision-Based Detection of
  Shrimp Diseases in Bangladesh*, Mendeley Data, V3, `10.17632/jhrtdj9txm.3`.
- Ahmed, S. I.; Farid, D. M. *TigerShrimpBD: A Tiger Shrimp Image Dataset*,
  Mendeley Data, V1, `10.17632/9dj4sk5d55.1`.
- Ramirez-Coronel, F. J.; Esquer-Miranda, E.; Rodriguez-Elias, O. M.;
  Garcia-Hinostro, P.; Parra-Salazar, G. C. *A Litopenaeus vannamei shrimp
  dataset with images and corresponding weight-size measurements for the
  development of artificial intelligence-based biomass estimation and organism
  detection algorithms*, Mendeley Data, V2, `10.17632/h8tcn6ykky.2`.

## Recorded conflicts

### 1. ShrimpDiseaseImageBD - licence field vs description text

- **Repository licence field**: `CC BY 4.0`.
- **Dataset description text**: "The dataset is released to the public domain to
  encourage further research and development in this domain."

These two statements disagree. A public-domain dedication (e.g. CC0) would waive
the attribution requirement that CC BY 4.0 imposes.

**Resolution applied**: the more restrictive interpretation is adopted. This
project treats ShrimpDiseaseImageBD as **CC BY 4.0**, attributes it in full, and
does not rely on any public-domain waiver. Use is limited to research
experimentation, which both interpretations permit.

### 2. Biomass dataset - requested version vs retrievable version

- The project specification cites DOI `10.17632/h8tcn6ykky.1` (version 1).
- The Mendeley public API serves a file listing only for the **current** version.
  Querying `?version=1` returns an empty file list with `size: 0`, so version 1
  is not retrievable programmatically. Version 2 (`10.17632/h8tcn6ykky.2`,
  published 2024-07-01) is the version actually obtained.

**Resolution applied**: version 2 was used and the substitution is recorded
here, in `configs/data/sources.yaml`, and in `reports/DATASET_AUDIT.md`. This is
**not** a silent dataset replacement: it is the same dataset and DOI stem at a
later version. The structural expectations stated in the specification for this
dataset (5,507 images, 170 individuals, DB1-DB5, `s<m>_<n>` naming) were verified
against version 2 and hold.

## Notes on redistribution

This repository does **not** redistribute any dataset. `data/raw/` is excluded
from version control; `scripts/download_data.py` retrieves the data from the
DOI records and verifies every file against the publisher-declared SHA-256.
Anyone reproducing this work obtains the data directly from Mendeley Data under
its own CC BY 4.0 terms.
