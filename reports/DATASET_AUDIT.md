# ShrimpVision-MTL - Dataset Audit (E0)

Generated: 2026-09-14T15:09:43.748950+00:00  
Manifest: `data/manifests/unified_manifest.parquet` (11977 rows, 11977 usable)  
Manifest content hash: `f9c5ee92c230166852b2fb149bfee1e2f8ac2ada99ec770e1d28f87ce8c4ced0`


**Leakage gate: CLEAR** (16 pass, 5 warn, 1 fail of 22 checks)


## 1. Acquisition, provenance and licence

| Dataset | Ver | DOI | Retrieved (UTC) | Files | Size | Licence |
|---|---|---|---|---|---|---|
| BiomassVannamei | 2 | `10.17632/h8tcn6ykky.2` | 2026-07-22 18:00:21 | 1 | 1465.2 MB | CC BY 4.0 |
| ShrimpDiseaseImageBD | 3 | `10.17632/jhrtdj9txm.3` | 2026-07-22 18:00:05 | 4 | 494.2 MB | CC BY 4.0 |
| TigerShrimpBD | 1 | `10.17632/9dj4sk5d55.1` | 2026-07-22 18:00:20 | 4575 | 209.5 MB | CC BY 4.0 |

### Version conflicts

- **BiomassVannamei**: requested `10.17632/h8tcn6ykky.1` (v1), resolved to `10.17632/h8tcn6ykky.2` (v2). Reason: Mendeley public API returns zero files for ?version=1; only the current version (2) exposes a downloadable file listing.

### Integrity verification

Every downloaded file's SHA-256 was recomputed locally and compared with the publisher-declared hash from the Mendeley API.

| Dataset | Files hashed | SHA-256 matches | Verdict |
|---|---|---|---|
| BiomassVannamei | 1 | 1 | ALL MATCH |
| ShrimpDiseaseImageBD | 4 | 4 | ALL MATCH |
| TigerShrimpBD | 4575 | 4575 | ALL MATCH |

### Archive checksums (top-level files)

| Dataset | File | Size | SHA-256 (truncated) |
|---|---|---|---|
| BiomassVannamei | DATASET.zip | 1465.19 MB | `89aebcb122d7054ec451af70f6250572...` |
| ShrimpDiseaseImageBD | Readme.docx | 0.06 MB | `5b41b5603f5d1912bdcf7bd4c1ac9d38...` |
| ShrimpDiseaseImageBD | ShrimpDiseaseBD_Summary.xlsx | 0.03 MB | `90a016a62e83e7e3d3cab8158f514069...` |
| ShrimpDiseaseImageBD | Annotated Diseased Shrimp Images.zip | 202.60 MB | `8d90317d116b8304200255e30abb5fc6...` |
| ShrimpDiseaseImageBD | Raw Images.zip | 291.46 MB | `f0525447d7c1331c3871aba919c57ac8...` |

### Licence conflicts

- **shrimp_disease_image_bd**: repository field says `CC BY 4.0`, description text says "The dataset is released to the public domain". Conflict between a CC BY 4.0 licence field and a prose "public domain" claim. The more restrictive interpretation (CC BY 4.0, attribution required) is adopted per project policy. See `LICENSE_NOTES.md`.

## 2. Verified counts after exclusions

| Dataset | Task | Role | Images | Fused groups |
|---|---|---|---|---|
| BiomassVannamei | biomass | raw | 5507 | 170 |
| ShrimpDiseaseImageBD | disease | annotated | 746 | 271 |
| ShrimpDiseaseImageBD | disease | raw | 1149 | 412 |
| TigerShrimpBD | disease | augmented | 3574 | 931 |
| TigerShrimpBD | disease | original | 1001 | 881 |

### Disease class distribution (usable images)

| Dataset | Raw class | Role | Images |
|---|---|---|---|
| ShrimpDiseaseImageBD | BG | annotated | 198 |
| ShrimpDiseaseImageBD | BG | raw | 198 |
| ShrimpDiseaseImageBD | BG_WSSV | annotated | 220 |
| ShrimpDiseaseImageBD | BG_WSSV | raw | 220 |
| ShrimpDiseaseImageBD | Healthy | raw | 403 |
| ShrimpDiseaseImageBD | WSSV | annotated | 328 |
| ShrimpDiseaseImageBD | WSSV | raw | 328 |
| TigerShrimpBD | Black Gill | augmented | 854 |
| TigerShrimpBD | Black Gill | original | 39 |
| TigerShrimpBD | Healthy | augmented | 846 |
| TigerShrimpBD | Healthy | original | 423 |
| TigerShrimpBD | WSSV | augmented | 978 |
| TigerShrimpBD | WSSV | original | 489 |
| TigerShrimpBD | Yellow Head | augmented | 896 |
| TigerShrimpBD | Yellow Head | original | 50 |

### Harmonized label supervision (masked, unknown never coerced to negative)

| Attribute | Supervised | Unknown (masked) | Positive | Negative |
|---|---|---|---|---|
| black_gill | 6470 | 0 | 1729 | 4741 |
| wssv | 6470 | 0 | 2563 | 3907 |
| yellow_head | 4575 | 1895 | 946 | 3629 |

`yellow_head` is unknown for every ShrimpDiseaseImageBD image because that dataset has no Yellow Head category. Those rows carry mask=0 and contribute zero loss; they are **not** treated as yellow_head-negative.


### Biomass structure

| DB | Images | Individuals | Views/individual | With weight | With cephalothorax |
|---|---|---|---|---|---|
| DB1 | 490 | 22 | 22.3 | 490 | 0 |
| DB2 | 2002 | 58 | 34.5 | 2002 | 2002 |
| DB3 | 1719 | 50 | 34.4 | 1719 | 1719 |
| DB4 | 635 | 20 | 31.8 | 635 | 635 |
| DB5 | 661 | 20 | 33.0 | 661 | 661 |

Total: **5507 images**, **170 individuals**.

Weight (g) per individual: min 1.00, median 30.00, max 41.00, mean 25.42, sd 10.47  
Length (cm) per individual: min 4.00, median 11.30, max 14.40


## 3. Audit checks

| Status | Check | Severity | Count | Detail |
|---|---|---|---|---|
| PASS | missing_files | leakage_critical | 0 | 0 manifest rows point at files that do not exist |
| PASS | unreadable_or_corrupt | informational | 0 | 0 images could not be decoded and were excluded |
| WARN | exact_duplicate_sha256 | leakage_critical | 747 | 747 SHA-256 values occur more than once (747 redundant copies). These are fused into one group so they cannot straddle a split. |
| PASS | near_duplicates_phash | informational | 38520 | 38520 within-block perceptual near-duplicate links fused into groups |
| WARN | cross_dataset_duplicates | leakage_critical | 31 | 0 images are byte-identical across datasets; 31 cross-dataset perceptual near-duplicates at Hamming<=6 |
| PASS | raw_annotated_pairing | leakage_critical | 746 | 746 annotated copies, 746 share a stem with a raw image; 746 (100.0%) are fused into the same group. Annotated copies are excluded from the primary disease pool |
| PASS | invalid_dimensions | leakage_critical | 0 | 0 images report a non-positive dimension |
| PASS | extreme_aspect_ratio | informational | 0 | 0 images fall outside aspect ratio [0.2, 5.0]; letterbox resizing preserves them without distortion |
| PASS | empty_classes | leakage_critical | 0 | classes present in the schema but with zero images |
| PASS | label_spelling_variants | informational | 8 | canonical raw classes observed after normalisation |
| WARN | filename_collisions | informational | 2489 | 2489 filenames occur more than once across the corpus (identical names in different class folders are expected) |
| WARN | published_augmentations | leakage_critical | 3574 | 1001 originals and 3574 published augmented copies. 3136 augmented images (87.7%) share a fused group with a published original. The remaining 438 form 52 famil |
| PASS | label_mask_integrity | leakage_critical | 0 | supervision masks must mark unknown labels, never encode them as 0 |
| FAIL | class_session_confound | informational | 100 | Health label vs capture session. ShrimpDiseaseImageBD: no timestamp in filenames; confound not testable; TigerShrimpBD: 100.0% of images sit in sessions contain |
| WARN | biomass_measurement_outliers | informational | 2 | 2 individual(s) have measurements inconsistent with the population length-weight relationship (|z| > 3.5), affecting 50 images. Retained and flagged, never dele |
| PASS | images_without_measurements | informational | 0 | 0 biomass images have no joined weight measurement |
| PASS | measurements_without_images | informational | 0 | 0 spreadsheet rows have no matching image |
| PASS | unparsable_image_names | leakage_critical | 0 | 0 biomass filenames do not match the s<m>_<n> pattern, so their individual identity cannot be recovered |
| PASS | implausible_measurements | informational | 0 | 0 individuals carry a physically implausible measurement flag (retained, flagged, not deleted) |
| PASS | biomass_individual_coverage | informational | 170 | 170 individuals, 5507 images, median 30 views/individual |
| PASS | sdibd_unparsable_image_names | leakage_critical | 0 | 0 ShrimpDiseaseImageBD filenames do not match <CLASS>-<n>-img-<v>, so their animal identity cannot be recovered |
| PASS | sdibd_class_token_mismatch | informational | 0 | 0 ShrimpDiseaseImageBD filenames carry a class token that differs from their folder class (id scoped by folder) |

### Check detail


#### PASS - `missing_files` (leakage_critical)

0 manifest rows point at files that do not exist

#### PASS - `unreadable_or_corrupt` (informational)

0 images could not be decoded and were excluded

#### WARN - `exact_duplicate_sha256` (leakage_critical)

747 SHA-256 values occur more than once (747 redundant copies). These are fused into one group so they cannot straddle a split.

```json
{
  "n_redundant_copies": 747
}
```

Examples:
```
{'sha256': '0047e595287c2d87', 'n': 2, 'paths': ['data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/1. BG/images/BG-28-img-2.jpg', 'data/raw/shrimp_disease_image_bd/extracted/Raw Images/2. BG/BG
{'sha256': '00602351ac33ad7c', 'n': 2, 'paths': ['data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/4. WSSV_BG/images/WSSV_BG-64-img-2.jpg', 'data/raw/shrimp_disease_image_bd/extracted/Raw Image
{'sha256': '011cc6d717abea76', 'n': 2, 'paths': ['data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/2. WSSV/images/WSSV-100-img-2.jpg', 'data/raw/shrimp_disease_image_bd/extracted/Raw Images/3. 
{'sha256': '018bb309a3bf982f', 'n': 2, 'paths': ['data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/1. BG/images/BG-23-img-1.jpg', 'data/raw/shrimp_disease_image_bd/extracted/Raw Images/2. BG/BG
{'sha256': '0245cb5829159b0d', 'n': 2, 'paths': ['data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/2. WSSV/images/WSSV-45-img-2.jpg', 'data/raw/shrimp_disease_image_bd/extracted/Raw Images/3. W
{'sha256': '024c705bc13b0c72', 'n': 2, 'paths': ['data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/4. WSSV_BG/images/WSSV_BG-56-img-1.jpg', 'data/raw/shrimp_disease_image_bd/extracted/Raw Image
{'sha256': '02606560b4582d66', 'n': 2, 'paths': ['data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/4. WSSV_BG/images/WSSV_BG-75-img-1.jpg', 'data/raw/shrimp_disease_image_bd/extracted/Raw Image
{'sha256': '02698fb8be95af35', 'n': 2, 'paths': ['data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/2. WSSV/images/WSSV-8-img-3.jpg', 'data/raw/shrimp_disease_image_bd/extracted/Raw Images/3. WS
```

#### PASS - `near_duplicates_phash` (informational)

38520 within-block perceptual near-duplicate links fused into groups

```json
{
  "group_size_histogram": {
    "1": 26,
    "2": 27,
    "3": 834,
    "4": 44,
    "5": 1,
    "6": 276,
    "9": 13,
    "12": 7,
    "13": 28,
    "14": 2,
    "15": 66,
    "16": 8,
    "20": 20,
    "24": 1,
    "25": 1,
    "26": 1,
    "28": 2,
    "29": 8,
    "30": 100,
    "31": 1,
    "32": 1,
    "35": 3,
    "39": 2,
    "40": 32,
    "43": 1,
    "45": 1,
    "50": 1,
    "60": 2,
    "65": 1,
    "75": 1,
    "80": 2
  },
  "largest_group": 80
}
```

Examples:
```
[201, 212, 2]
[201, 223, 4]
[201, 234, 4]
[201, 245, 4]
[201, 256, 4]
[201, 267, 6]
[201, 274, 6]
[201, 275, 6]
```

#### WARN - `cross_dataset_duplicates` (leakage_critical)

0 images are byte-identical across datasets; 31 cross-dataset perceptual near-duplicates at Hamming<=6

```json
{
  "n_exact_cross_dataset": 0
}
```

Examples:
```
{'a': 'data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/1. BG/images/BG-38-img-2.jpg', 'b': 'data/raw/tiger_shrimp_bd/black_gill/augmented/aug_1_IMG-20240426-WA0269.jpg', 'hamming': 6, 'dataset
{'a': 'data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/1. BG/images/BG-38-img-2.jpg', 'b': 'data/raw/tiger_shrimp_bd/black_gill/augmented/aug_4_IMG-20240426-WA0269.jpg', 'hamming': 6, 'dataset
{'a': 'data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/1. BG/images/BG-38-img-2.jpg', 'b': 'data/raw/tiger_shrimp_bd/black_gill/augmented/aug_6_IMG-20240426-WA0269.jpg', 'hamming': 6, 'dataset
{'a': 'data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/2. WSSV/images/WSSV-70-img-1.jpg', 'b': 'data/raw/tiger_shrimp_bd/healthy/augmented/resized_IMG-20240707-WA0181.jpg', 'hamming': 4, 'data
{'a': 'data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/2. WSSV/images/WSSV-70-img-1.jpg', 'b': 'data/raw/tiger_shrimp_bd/healthy/augmented/resized_IMG-20240707-WA0184.jpg', 'hamming': 6, 'data
{'a': 'data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/2. WSSV/images/WSSV-70-img-1.jpg', 'b': 'data/raw/tiger_shrimp_bd/healthy/augmented/resized_IMG-20240710-WA0070.jpg', 'hamming': 6, 'data
{'a': 'data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/2. WSSV/images/WSSV-70-img-1.jpg', 'b': 'data/raw/tiger_shrimp_bd/healthy/original/IMG-20240707-WA0181.jpg', 'hamming': 4, 'datasets': ['
{'a': 'data/raw/shrimp_disease_image_bd/extracted/Annotated Diseased Shrimp Images/2. WSSV/images/WSSV-70-img-1.jpg', 'b': 'data/raw/tiger_shrimp_bd/healthy/original/IMG-20240707-WA0184.jpg', 'hamming': 6, 'datasets': ['
```

#### PASS - `raw_annotated_pairing` (leakage_critical)

746 annotated copies, 746 share a stem with a raw image; 746 (100.0%) are fused into the same group. Annotated copies are excluded from the primary disease pool so they are never counted as independent samples.

```json
{
  "n_annotated": 746,
  "n_raw": 1149,
  "fraction_grouped": 1.0
}
```

Examples:
```
bg-1-img-1
bg-1-img-2
bg-1-img-3
bg-10-img-1
bg-10-img-2
bg-10-img-3
bg-11-img-1
bg-11-img-2
```

#### PASS - `invalid_dimensions` (leakage_critical)

0 images report a non-positive dimension

#### PASS - `extreme_aspect_ratio` (informational)

0 images fall outside aspect ratio [0.2, 5.0]; letterbox resizing preserves them without distortion

```json
{
  "ar_min": 0.5625,
  "ar_max": 1.7777777777777777
}
```

#### PASS - `empty_classes` (leakage_critical)

classes present in the schema but with zero images

```json
{
  "class_counts": {
    "ShrimpDiseaseImageBD|BG": 396,
    "ShrimpDiseaseImageBD|BG_WSSV": 440,
    "ShrimpDiseaseImageBD|Healthy": 403,
    "ShrimpDiseaseImageBD|WSSV": 656,
    "TigerShrimpBD|Black Gill": 893,
    "TigerShrimpBD|Healthy": 1269,
    "TigerShrimpBD|WSSV": 1467,
    "TigerShrimpBD|Yellow Head": 946
  }
}
```

#### PASS - `label_spelling_variants` (informational)

canonical raw classes observed after normalisation

Examples:
```
ShrimpDiseaseImageBD: ['BG', 'BG_WSSV', 'Healthy', 'WSSV']
TigerShrimpBD: ['Black Gill', 'Healthy', 'WSSV', 'Yellow Head']
```

#### WARN - `filename_collisions` (informational)

2489 filenames occur more than once across the corpus (identical names in different class folders are expected)

Examples:
```
{'filename': 's10_1.png', 'n': 5, 'distinct_sha256': 5, 'datasets': ['BiomassVannamei']}
{'filename': 's10_10.png', 'n': 5, 'distinct_sha256': 5, 'datasets': ['BiomassVannamei']}
{'filename': 's10_11.png', 'n': 5, 'distinct_sha256': 5, 'datasets': ['BiomassVannamei']}
{'filename': 's10_12.png', 'n': 5, 'distinct_sha256': 5, 'datasets': ['BiomassVannamei']}
{'filename': 's10_13.png', 'n': 5, 'distinct_sha256': 5, 'datasets': ['BiomassVannamei']}
{'filename': 's10_14.png', 'n': 5, 'distinct_sha256': 5, 'datasets': ['BiomassVannamei']}
{'filename': 's10_15.png', 'n': 5, 'distinct_sha256': 5, 'datasets': ['BiomassVannamei']}
{'filename': 's10_16.png', 'n': 5, 'distinct_sha256': 5, 'datasets': ['BiomassVannamei']}
```

#### WARN - `published_augmentations` (leakage_critical)

1001 originals and 3574 published augmented copies. 3136 augmented images (87.7%) share a fused group with a published original. The remaining 438 form 52 families whose source photographs were NOT published in the originals folders (0 of those stems match a published original, i.e. this is a property of the source data, not a tracing failure). Those families are still fused among themselves, so no augmented sibling can straddle a split.

```json
{
  "n_original": 1001,
  "n_augmented": 3574,
  "n_orphan_augmented": 438,
  "n_orphan_families": 52,
  "n_orphan_stems_matching_a_published_original": 0,
  "orphan_siblings_correctly_fused": false,
  "fraction_traced_to_published_original": 0.8774
}
```

Examples:
```
img-20240414-wa0084
img-20240414-wa0133
img-20240414-wa0139
img-20240414-wa0190
img-20240414-wa0191
img-20240414-wa0192
img-20240414-wa0214
img-20240414-wa0217
```

#### PASS - `label_mask_integrity` (leakage_critical)

supervision masks must mark unknown labels, never encode them as 0

```json
{
  "black_gill": {
    "supervised": 6470,
    "positive": 1729
  },
  "wssv": {
    "supervised": 6470,
    "positive": 2563
  },
  "yellow_head": {
    "supervised": 4575,
    "positive": 946
  }
}
```

#### FAIL - `class_session_confound` (informational)

Health label vs capture session. ShrimpDiseaseImageBD: no timestamp in filenames; confound not testable; TigerShrimpBD: 100.0% of images sit in sessions containing only ONE health class (13/13 sessions). COMPLETE CONFOUND: the health label is perfectly determined by acquisition session, so within-dataset healthy-vs-diseased scores on that dataset cannot be attributed to disease recognition.

```json
{
  "max_fraction_confounded": 1.0,
  "interpretation": "1.0 means every image was captured in a session containing only one health class"
}
```

Examples:
```
{'dataset': 'ShrimpDiseaseImageBD', 'sessions_recoverable': False, 'note': 'no timestamp in filenames; confound not testable'}
{'dataset': 'TigerShrimpBD', 'sessions_recoverable': True, 'n_sessions': 13, 'n_sessions_single_health_class': 13, 'n_images_in_single_class_sessions': 1001, 'n_images_with_session': 1001, 'fraction_confounded': 1.0, 'se
```

#### WARN - `biomass_measurement_outliers` (informational)

2 individual(s) have measurements inconsistent with the population length-weight relationship (|z| > 3.5), affecting 50 images. Retained and flagged, never deleted.

```json
{
  "allometric_fit": {
    "a": -4.8948,
    "b": 3.4026,
    "residual_sd_log": 0.1414,
    "n": 170
  },
  "cephalothorax_length_ratio": {
    "mean": 0.3162,
    "sd": 0.0298,
    "n": 148
  },
  "n_individuals_flagged": 2,
  "n_images_affected": 50,
  "z_threshold": 3.5,
  "individuals": [
    "DB1_s002",
    "DB5_s005"
  ],
  "method": "allometric residual on log(W) ~ log(L), plus cephalothorax/length ratio; marginal within-pond z-scores deliberately not used"
}
```

Examples:
```
{'individual_id': 'DB1_s002', 'test': 'allometric_residual', 'db': 'DB1', 'length_cm': 14.4, 'weight_g': 33.0, 'expected_weight_g': 65.42, 'residual_z': -4.84, 'n_images_affected': 20}
{'individual_id': 'DB5_s005', 'test': 'allometric_residual', 'db': 'DB5', 'length_cm': 8.34, 'weight_g': 6.0, 'expected_weight_g': 10.2, 'residual_z': -3.75, 'n_images_affected': 30}
```

#### PASS - `images_without_measurements` (informational)

0 biomass images have no joined weight measurement

#### PASS - `measurements_without_images` (informational)

0 spreadsheet rows have no matching image

#### PASS - `unparsable_image_names` (leakage_critical)

0 biomass filenames do not match the s<m>_<n> pattern, so their individual identity cannot be recovered

#### PASS - `implausible_measurements` (informational)

0 individuals carry a physically implausible measurement flag (retained, flagged, not deleted)

#### PASS - `biomass_individual_coverage` (informational)

170 individuals, 5507 images, median 30 views/individual

```json
{
  "views_per_individual": {
    "min": 15,
    "median": 30.0,
    "max": 80
  },
  "by_db": {
    "DB1": {
      "n_images": 490,
      "n_individuals_in_images": 22,
      "n_individuals_in_sheet": 22
    },
    "DB2": {
      "n_images": 2002,
      "n_individuals_in_images": 58,
      "n_individuals_in_sheet": 58
    },
    "DB3": {
      "n_images": 1719,
      "n_individuals_in_images": 50,
      "n_individuals_in_sheet": 50
    },
    "DB4": {
      "n_images": 635,
      "n_individuals_in_images": 20,
      "n_individuals_in_sheet": 20
    },
    "DB5": {
      "n_images": 661,
      "n_individuals_in_images": 20,
      "n_individuals_in_sheet": 20
    }
  }
}
```

#### PASS - `sdibd_unparsable_image_names` (leakage_critical)

0 ShrimpDiseaseImageBD filenames do not match <CLASS>-<n>-img-<v>, so their animal identity cannot be recovered

#### PASS - `sdibd_class_token_mismatch` (informational)

0 ShrimpDiseaseImageBD filenames carry a class token that differs from their folder class (id scoped by folder)

## 4. Grouping and leakage control

- records grouped: **11977**
- fused groups: **1513**
- exact-duplicate links: 747
- augmentation-family links: 3245
- perceptual near-duplicate links (within-block): 38520
- forced links (individual identity (biomass shrimp; SDIBD animals)): 6816
- largest group: 80 images

The grouping key is the transitive closure of exact-duplicate, augmentation-family and within-block perceptual near-duplicate relations, plus forced links binding every image of one individual (biomass shrimp or SDIBD animal). All splits are constructed over this key, so an augmented copy can never appear in a different split from its source original, and no shrimp can appear in two folds.


Perceptual fusion is deliberately **blocked** by dataset+class (disease) and by individual (biomass). Unblocked fusion would chain the 170 biomass shrimp photographed on one rig into a single component and collapse the splits.


## 5. Excluded records

0 records excluded. See `data/manifests/excluded_records.csv`.