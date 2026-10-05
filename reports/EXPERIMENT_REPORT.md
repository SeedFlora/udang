# ShrimpVision-MTL - Technical Experiment Report

Generated: 2026-09-15T10:39:42.470016+00:00  
This is a factual execution report. It documents only experiments that were actually run, from artefacts written to disk during those runs.


## 1. What was executed

- **240 runs completed**, 0 failed or incomplete.

| ID | Experiment | Runs completed | Failed | Seeds | Wall-clock |
|---|---|---|---|---|---|
| E1 | Disease single-dataset baselines | 18 | 0 | 2026;3407;42 | 32.7 min |
| E10 |  | 18 | 0 | 2026;3407;42 | 52.8 min |
| E11 |  | 9 | 0 | 2026;3407;42 | 91.5 min |
| E12 |  | 9 | 0 | 2026;3407;42 | 70.3 min |
| E13 |  | 15 | 0 | 2026;3407;42 | 142.8 min |
| E2 | Pooled hierarchical disease model + ablations | 54 | 0 | 2026;3407;42 | 242.4 min |
| E3 | Cross-dataset disease evaluation | 6 | 0 | 2026;3407;42 | 16.9 min |
| E4 | Biomass single-image baseline | 15 | 0 | 2026;3407;42 | 149.6 min |
| E5 | Biomass multi-view model | 30 | 0 | 2026;3407;42 | 64.5 min |
| E6 | Shared-encoder multi-task model | 18 | 0 | 2026;3407;42 | 231.4 min |
| E7 | Domain-generalized multi-task model | 24 | 0 | 2026;3407;42 | 262.6 min |
| E8 | Lightweight model (MobileNetV3-Large) | 24 | 0 | 2026;3407;42 | 156.7 min |

E0 (audit) is reported in `reports/DATASET_AUDIT.md`; its gate status is **CLEAR**.

## 2. Dataset counts after exclusions

| Dataset | Task | Role | Images | Fused groups |
|---|---|---|---|---|
| BiomassVannamei | biomass | raw | 5507 | 170 |
| ShrimpDiseaseImageBD | disease | annotated | 746 | 271 |
| ShrimpDiseaseImageBD | disease | raw | 1149 | 412 |
| TigerShrimpBD | disease | augmented | 3574 | 931 |
| TigerShrimpBD | disease | original | 1001 | 881 |

Total manifest rows: **11977**; usable: **11977**; excluded: **0**.


**Modelling pools actually used**

- Disease primary pool (originals only, annotated duplicates removed): **2150 images** (1149 ShrimpDiseaseImageBD + 1001 TigerShrimpBD).
- Disease with-augmentation pool (ablation only): **6470 images**.
- Biomass: **5507 images / 170 individuals** across 5 DBs.

## 3. Leakage findings

- **missing_files** [PASS] - 0 manifest rows point at files that do not exist
- **exact_duplicate_sha256** [WARN] - 747 SHA-256 values occur more than once (747 redundant copies). These are fused into one group so they cannot straddle a split.
- **cross_dataset_duplicates** [WARN] - 0 images are byte-identical across datasets; 31 cross-dataset perceptual near-duplicates at Hamming<=6
- **raw_annotated_pairing** [PASS] - 746 annotated copies, 746 share a stem with a raw image; 746 (100.0%) are fused into the same group. Annotated copies are excluded from the primary disease pool so they are never counted as independent samples.
- **invalid_dimensions** [PASS] - 0 images report a non-positive dimension
- **empty_classes** [PASS] - classes present in the schema but with zero images
- **published_augmentations** [WARN] - 1001 originals and 3574 published augmented copies. 3136 augmented images (87.7%) share a fused group with a published original. The remaining 438 form 52 families whose source photographs were NOT published in the originals folders (0 of those stems match a published original, i.e. this is a property of the source data, not a tracing failure). Those families are still fused among themselves, so no augmented sibling can straddle a split.
- **label_mask_integrity** [PASS] - supervision masks must mark unknown labels, never encode them as 0
- **unparsable_image_names** [PASS] - 0 biomass filenames do not match the s<m>_<n> pattern, so their individual identity cannot be recovered
- **sdibd_unparsable_image_names** [PASS] - 0 ShrimpDiseaseImageBD filenames do not match <CLASS>-<n>-img-<v>, so their animal identity cannot be recovered

The three most consequential findings:

1. **All 746 'Annotated Diseased Shrimp Images' in ShrimpDiseaseImageBD are byte-identical (identical SHA-256) to images in 'Raw Images'.** The annotated folder contributes no new photographs, only YOLO label files beside duplicate copies. Counting it as extra data would have inflated that dataset by 65% with duplicates and split the same photograph across folds. Those copies are fused into their raw image's group and excluded from the primary disease pool.
2. **TigerShrimpBD ships 3,574 published augmented derivatives of 1,001 originals.** Augmented copies are grouped with their source original so they cannot straddle a split. The primary protocol uses originals only; the with-augmentation run is reported strictly as an ablation.
3. **Biomass images are 5,507 photographs of only 170 animals** (median 30 views each). Every image of one individual is forced into a single group, and all biomass confidence intervals bootstrap over individuals, never images.

### 3.1 Acquisition-session confound (read this before any accuracy number)

Health label vs capture session. ShrimpDiseaseImageBD: no timestamp in filenames; confound not testable; TigerShrimpBD: 100.0% of images sit in sessions containing only ONE health class (13/13 sessions). COMPLETE CONFOUND: the health label is perfectly determined by acquisition session, so within-dataset healthy-vs-diseased scores on that dataset cannot be attributed to disease recognition.

- **ShrimpDiseaseImageBD**: no timestamp in filenames; confound not testable
- **TigerShrimpBD**: 13 of 13 capture sessions contain only one health class, covering 1001/1001 images (100.0%).

| Capture session | healthy | diseased |
|---|---|---|
| 20240414 | 0 | 36 |
| 20240421 | 0 | 68 |
| 20240422 | 0 | 18 |
| 20240423 | 0 | 159 |
| 20240424 | 0 | 56 |
| 20240425 | 0 | 59 |
| 20240426 | 0 | 56 |
| 20240427 | 0 | 126 |
| 20240624 | 35 | 0 |
| 20240625 | 145 | 0 |
| 20240707 | 112 | 0 |
| 20240709 | 61 | 0 |
| 20240710 | 70 | 0 |

In TigerShrimpBD every healthy shrimp was photographed in June-July 2024 and every diseased shrimp in April 2024, on entirely disjoint days. Session artefacts -- lighting, background, camera, water colour, season -- therefore separate healthy from diseased perfectly, with no disease sign needing to be visible. This is a property of how the data was collected and **no split strategy can remove it**: grouping by session still leaves the April/June appearance difference aligned with the label.

**Consequence for interpretation.** A within-dataset healthy-versus-diseased score on TigerShrimpBD is an upper bound that cannot be attributed to disease recognition. Where such a model reaches a perfect or near-perfect score, the correct reading is that the task was separable by acquisition artefact, not that the disease was recognised. Evidence that does bear on disease recognition, and should be weighted instead:

- the **cross-dataset protocol (E3)**, where the target dataset does not share the source's session artefacts;
- the **per-attribute** results (Black Gill / WSSV / Yellow Head), which require discriminating between diseased classes captured in the SAME April sessions and so are not separable by session;
- **ShrimpDiseaseImageBD**, whose filenames carry no timestamp, so the same confound could not be tested there and is neither confirmed nor excluded.

## 4. Final split design

| Protocol | Folds | Train rows | Val rows | Test rows | Distinct groups |
|---|---|---|---|---|---|
| biomass_lodo | 5 | 17690 | 4338 | 5507 | 170 |
| biomass_loso | 3 | 8896 | 2118 | 5507 | 170 |
| biomass_within_db | 5 | 17628 | 4400 | 5507 | 170 |
| cross_sd2tg | 1 | 919 | 230 | 951 | 1243 |
| cross_tg2sd | 1 | 801 | 200 | 929 | 1212 |
| disease_cv | 5 | 6878 | 1722 | 2150 | 1293 |
| disease_cv_naive_withaug | 5 | 18316 | 4580 | 5724 | 5724 |
| disease_cv_sdibd | 5 | 3676 | 920 | 1149 | 412 |
| disease_cv_tiger | 5 | 3202 | 802 | 1001 | 881 |
| disease_cv_withaug | 5 | 19107 | 1722 | 2150 | 1343 |

- Disease: 5-fold **stratified group** CV, stratified by (dataset x raw class), grouped by the fused duplicate/augmentation-family key. Validation is carved from the training folds only.
- Cross-dataset: train on one disease dataset, test on the other, scored only on the semantically aligned Healthy / Black Gill / WSSV space. Thresholds are selected on a validation split of the SOURCE dataset; the target is never used for tuning.
- Biomass: leave-one-DB-out over DB1-DB5, grouped strictly by individual shrimp; validation individuals come from the remaining DBs only.

Manifest hash: `f9c5ee92c230166852b2fb149bfee1e2f8ac2ada99ec770e1d28f87ce8c4ced0`  
Split hash: `8cea4f217d4881079bb09715c5702024befe9e97798469711059cd96269b5fac`

## 5. Models and hyperparameters

Architecture: one shared visual encoder feeding four heads - a binary health head (1 logit, positive class = diseased), a 3-output disease-attribute head (black_gill, wssv, yellow_head), a 3-output biomass head (log weight, length cm, cephalothorax cm), and an optional domain classifier. Single-task and multi-task runs use the SAME class and the same head capacity; only the loss mixture differs, which keeps the comparison controlled.

Losses:

```
L_disease = BCE(health) + lambda_attr * masked_attr_loss + lambda_cons * consistency
L_biomass = sum_k w_k * masked_Huber(component_k)
```
Unknown labels carry a supervision mask of 0 and contribute exactly zero loss and zero gradient. `yellow_head` is unknown for every ShrimpDiseaseImageBD image; `cephalothorax_cm` is unknown for all of DB1.

| Experiment | Backbone | torch | GPU |
|---|---|---|---|
| E1 | convnext_tiny | 2.13.0+cu126 | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| E4 | convnext_tiny | 2.13.0+cu126 | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| E5 | convnext_tiny | 2.13.0+cu126 | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| E8 | mobilenetv3_large | 2.13.0+cu126 | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| E2 | convnext_tiny | 2.13.0+cu126 | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| E3 | convnext_tiny | 2.13.0+cu126 | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| E6 | convnext_tiny | 2.13.0+cu126 | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| E7 | convnext_tiny | 2.13.0+cu126 | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| E10 | resnet50 | 2.13.0+cu126 | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| E10 | efficientnet_b0 | 2.13.0+cu126 | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| E10 | swin_t | 2.13.0+cu126 | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| E11 | convnext_tiny | 2.13.0+cu126 | NVIDIA GeForce RTX 3070 Ti Laptop GPU |

Compute profile used for the mandatory matrix: **standard** (epochs=12, early-stopping patience=4, image size=320, folds=3, ablation folds=2). Optimiser AdamW, lr 3e-4, weight decay 1e-4, cosine decay with warm-up, mixed precision on GPU.

Images are pre-resized to a max side of 448 px (5507 resized, 0 copied unchanged, 456.1 MB). ShrimpDiseaseImageBD ships every image at 2048x2048, so decoding dominated epoch time; the cache removes that bottleneck without affecting what the network sees at 320 px.

## 6. Completed experiment matrix

| ID | Configuration | Folds | Runs | Seeds |
|---|---|---|---|---|
| E1 | E1_disease_single_sdibd | 3 | 9 | 2026;3407;42 |
| E1 | E1_disease_single_tiger | 3 | 9 | 2026;3407;42 |
| E10 | E10_efficientnet_b0_sd2tg | 1 | 3 | 2026;3407;42 |
| E10 | E10_efficientnet_b0_tg2sd | 1 | 3 | 2026;3407;42 |
| E10 | E10_resnet50_sd2tg | 1 | 3 | 2026;3407;42 |
| E10 | E10_resnet50_tg2sd | 1 | 3 | 2026;3407;42 |
| E10 | E10_swin_t_sd2tg | 1 | 3 | 2026;3407;42 |
| E10 | E10_swin_t_tg2sd | 1 | 3 | 2026;3407;42 |
| E11 | E11_naive_withaug | 3 | 9 | 2026;3407;42 |
| E12 | E12_biomass_loso | 3 | 9 | 2026;3407;42 |
| E13 | E13_biomass_within_db | 5 | 15 | 2026;3407;42 |
| E2 | E2_pooled_asl_ordinary | 3 | 9 | 2026;3407;42 |
| E2 | E2_pooled_bce_balanced | 3 | 9 | 2026;3407;42 |
| E2 | E2_pooled_bce_ordinary | 3 | 9 | 2026;3407;42 |
| E2 | E2_pooled_bce_strongcolor | 3 | 9 | 2026;3407;42 |
| E2 | E2_pooled_bce_withaug | 3 | 9 | 2026;3407;42 |
| E2 | E2_pooled_focal_ordinary | 3 | 9 | 2026;3407;42 |
| E3 | E3_cross_sd2tg | 1 | 3 | 2026;3407;42 |
| E3 | E3_cross_tg2sd | 1 | 3 | 2026;3407;42 |
| E4 | E4_biomass_single_image | 5 | 15 | 2026;3407;42 |
| E5 | E5_biomass_multiview_attention | 5 | 15 | 2026;3407;42 |
| E5 | E5_biomass_multiview_mean | 5 | 15 | 2026;3407;42 |
| E6 | E6_multitask_fixed | 3 | 9 | 2026;3407;42 |
| E6 | E6_multitask_uncertainty | 3 | 9 | 2026;3407;42 |
| E7 | E7_cross_mixstyle_sd2tg | 1 | 3 | 2026;3407;42 |
| E7 | E7_cross_mixstyle_tg2sd | 1 | 3 | 2026;3407;42 |
| E7 | E7_mtl_coral | 3 | 9 | 2026;3407;42 |
| E7 | E7_mtl_mixstyle | 3 | 9 | 2026;3407;42 |
| E8 | E8_lightweight_biomass | 5 | 15 | 2026;3407;42 |
| E8 | E8_lightweight_disease | 3 | 9 | 2026;3407;42 |

## 7. Quantitative results

### 7.1 Disease (test folds, thresholds selected on validation)

| config name | n folds | health bal acc | health bal acc std | health f1 | health auroc | attr macro f1 | attr macro map | attr exact match |
|---|---|---|---|---|---|---|---|---|
| E1_disease_single_sdibd | 3 | 0.8124 | 0.0515 | 0.8725 | 0.9014 | 0.7494 | 0.8090 | n/a |
| E1_disease_single_tiger | 3 | 0.9977 | 0.0046 | 0.9981 | 1.0000 | 0.7247 | 0.7619 | 0.9290 |
| E10_efficientnet_b0_sd2tg | 1 | 0.6149 | 0.1125 | 0.7213 | 0.7506 | 0.3170 | 0.5080 | 0.2780 |
| E10_efficientnet_b0_tg2sd | 1 | 0.4394 | 0.0166 | 0.6377 | 0.3680 | 0.1681 | 0.4225 | n/a |
| E10_resnet50_sd2tg | 1 | 0.5528 | 0.0294 | 0.7248 | 0.8343 | 0.2748 | 0.4864 | 0.0943 |
| E10_resnet50_tg2sd | 1 | 0.4649 | 0.0294 | 0.6798 | 0.3900 | 0.3820 | 0.3612 | n/a |
| E10_swin_t_sd2tg | 1 | 0.5738 | 0.0398 | 0.7284 | 0.7337 | 0.3232 | 0.5003 | 0.0301 |
| E10_swin_t_tg2sd | 1 | 0.4499 | 0.0313 | 0.6595 | 0.3050 | 0.3286 | 0.4866 | n/a |
| E11_naive_withaug | 3 | 0.9756 | 0.0074 | 0.9881 | 0.9978 | 0.9717 | 0.9951 | 0.9983 |
| E2_pooled_asl_ordinary | 3 | 0.9093 | 0.0264 | 0.9329 | 0.9699 | 0.6601 | 0.7216 | 0.8810 |
| E2_pooled_bce_balanced | 3 | 0.8935 | 0.0750 | 0.9218 | 0.9768 | 0.8294 | 0.8834 | 0.9391 |
| E2_pooled_bce_ordinary | 3 | 0.9155 | 0.0384 | 0.9389 | 0.9760 | 0.8120 | 0.8687 | 0.9148 |
| E2_pooled_bce_strongcolor | 3 | 0.9117 | 0.0329 | 0.9341 | 0.9751 | 0.7824 | 0.8546 | 0.9336 |
| E2_pooled_bce_withaug | 3 | 0.9342 | 0.0131 | 0.9500 | 0.9821 | 0.8833 | 0.9298 | 0.9867 |
| E2_pooled_focal_ordinary | 3 | 0.9004 | 0.0470 | 0.9316 | 0.9754 | 0.7028 | 0.7484 | 0.9097 |
| E3_cross_sd2tg | 1 | 0.4942 | 0.0307 | 0.6724 | 0.4594 | 0.2475 | 0.3155 | 0.1385 |
| E3_cross_tg2sd | 1 | 0.3803 | 0.0176 | 0.4233 | 0.2706 | 0.1636 | 0.2940 | n/a |
| E6_multitask_fixed | 3 | 0.8712 | 0.0695 | 0.9167 | 0.9681 | 0.7345 | 0.7861 | 0.9158 |
| E6_multitask_uncertainty | 3 | 0.9065 | 0.0445 | 0.9279 | 0.9675 | 0.7648 | 0.8000 | 0.9191 |
| E7_cross_mixstyle_sd2tg | 1 | 0.5435 | 0.0701 | 0.6782 | 0.5856 | 0.2033 | 0.3561 | 0.0796 |
| E7_cross_mixstyle_tg2sd | 1 | 0.4440 | 0.0502 | 0.4115 | 0.3082 | 0.0912 | 0.2921 | n/a |
| E7_mtl_coral | 3 | 0.9075 | 0.0327 | 0.9228 | 0.9733 | 0.7526 | 0.8214 | 0.8775 |
| E7_mtl_mixstyle | 3 | 0.8313 | 0.1063 | 0.8847 | 0.9331 | 0.6609 | 0.6925 | 0.8727 |
| E8_lightweight_disease | 3 | 0.9516 | 0.0173 | 0.9612 | 0.9922 | 0.8788 | 0.9457 | 0.9701 |

#### Per-source breakdown (how each model does on each dataset's images)

| config name | SDIBD health bal acc | SDIBD attr macro f1 | SDIBD n | Tiger health bal acc | Tiger attr macro f1 | Tiger n |
|---|---|---|---|---|---|---|
| E1_disease_single_sdibd | 0.8124 | 0.7494 | 229.6667 | n/a | n/a | n/a |
| E1_disease_single_tiger | n/a | n/a | n/a | 0.9977 | 0.7247 | 200.3333 |
| E10_efficientnet_b0_sd2tg | n/a | n/a | n/a | 0.6149 | 0.3170 | 951.0000 |
| E10_efficientnet_b0_tg2sd | 0.4394 | 0.1681 | 929.0000 | n/a | n/a | n/a |
| E10_resnet50_sd2tg | n/a | n/a | n/a | 0.5528 | 0.2748 | 951.0000 |
| E10_resnet50_tg2sd | 0.4649 | 0.3820 | 929.0000 | n/a | n/a | n/a |
| E10_swin_t_sd2tg | n/a | n/a | n/a | 0.5738 | 0.3232 | 951.0000 |
| E10_swin_t_tg2sd | 0.4499 | 0.3286 | 929.0000 | n/a | n/a | n/a |
| E11_naive_withaug | 0.8979 | 0.8723 | 230.0000 | 0.9992 | 0.9982 | 915.0000 |
| E2_pooled_asl_ordinary | 0.8299 | 0.7253 | 229.6667 | 0.9904 | 0.4985 | 200.6667 |
| E2_pooled_bce_balanced | 0.8248 | 0.7951 | 229.6667 | 0.9661 | 0.7150 | 200.6667 |
| E2_pooled_bce_ordinary | 0.8424 | 0.8003 | 229.6667 | 0.9908 | 0.6842 | 200.6667 |
| E2_pooled_bce_strongcolor | 0.8351 | 0.7793 | 229.6667 | 0.9914 | 0.6307 | 200.6667 |
| E2_pooled_bce_withaug | 0.8718 | 0.8213 | 229.6667 | 0.9995 | 0.9379 | 200.6667 |
| E2_pooled_focal_ordinary | 0.8104 | 0.7234 | 229.6667 | 0.9915 | 0.5433 | 200.6667 |
| E3_cross_sd2tg | n/a | n/a | n/a | 0.4942 | 0.2475 | 951.0000 |
| E3_cross_tg2sd | 0.3803 | 0.1636 | 929.0000 | n/a | n/a | n/a |
| E6_multitask_fixed | 0.7654 | 0.7518 | 229.6667 | 0.9765 | 0.5587 | 200.6667 |
| E6_multitask_uncertainty | 0.8395 | 0.7761 | 229.6667 | 0.9728 | 0.5883 | 200.6667 |
| E7_cross_mixstyle_sd2tg | n/a | n/a | n/a | 0.5435 | 0.2033 | 951.0000 |
| E7_cross_mixstyle_tg2sd | 0.4440 | 0.0912 | 929.0000 | n/a | n/a | n/a |
| E7_mtl_coral | 0.8292 | 0.7703 | 229.6667 | 0.9918 | 0.5940 | 200.6667 |
| E7_mtl_mixstyle | 0.7121 | 0.6947 | 229.6667 | 0.9532 | 0.5132 | 200.6667 |
| E8_lightweight_disease | 0.9102 | 0.8507 | 229.6667 | 0.9960 | 0.8848 | 200.6667 |

`sd` = ShrimpDiseaseImageBD, `tg` = TigerShrimpBD. This is the like-for-like view of pooled versus single-dataset training: it shows what a pooled model achieves on each source's own images. Note that the Tiger health column inherits the acquisition-session confound of section 3.1 and should not be read as disease recognition.

### 7.1b Cross-dataset transfer (E3) -- the test that matters

| Direction | N | health bal-acc | health F1 | health AUROC | health AP |
|---|---|---|---|---|---|
| E3_cross_sd2tg | 2853 | 0.4942 | 0.6742 | 0.5025 | 0.5828 |
| E3_cross_tg2sd | 2787 | 0.3803 | 0.4636 | 0.3237 | 0.4545 |
| E7_cross_mixstyle_sd2tg | 2853 | 0.5435 | 0.6835 | 0.5421 | 0.5739 |
| E7_cross_mixstyle_tg2sd | 2787 | 0.4440 | 0.5015 | 0.3786 | 0.4834 |

Mean within-dataset health balanced accuracy across E1/E2 is **0.791**; mean cross-dataset is **0.466** (chance = 0.5), at a mean AUROC of **0.437**.

**This is the empirical confirmation of the section-3.1 confound.** A model that scores near 1.0 within a dataset but at or below chance on the other dataset has learned dataset-specific appearance -- background, lighting, session -- not disease. An AUROC below 0.5 on the target domain means the features are actively anti-correlated with the true label there, which a genuine disease detector could not produce. The within-dataset numbers in sections 7.1-7.2 must be read in this light: they are upper bounds reflecting separability by acquisition artefact, and the cross-dataset numbers here are the honest estimate of transferable disease recognition. Where a single disease sign is strong and local (WSSV white spots) some signal does transfer; where it is subtle (Black Gill discolouration) almost none does.

### 7.2 Per-disease-attribute detail

| config name | black gill f1 | black gill ap | wssv f1 | wssv ap | yellow head f1 | yellow head ap |
|---|---|---|---|---|---|---|
| E1_disease_single_sdibd | 0.6416 | 0.7514 | 0.8573 | 0.8667 | n/a | n/a |
| E1_disease_single_tiger | 0.3918 | 0.4234 | 0.9542 | 0.9634 | 0.8281 | 0.8991 |
| E10_efficientnet_b0_sd2tg | 0.2048 | 0.1595 | 0.7461 | 0.8565 | 0.0000 | n/a |
| E10_efficientnet_b0_tg2sd | 0.2483 | 0.2807 | 0.0878 | 0.5642 | n/a | n/a |
| E10_resnet50_sd2tg | 0.1133 | 0.1286 | 0.7111 | 0.8443 | 0.0000 | n/a |
| E10_resnet50_tg2sd | 0.3623 | 0.2369 | 0.4016 | 0.4856 | n/a | n/a |
| E10_swin_t_sd2tg | 0.2757 | 0.2539 | 0.6937 | 0.7467 | 0.0000 | n/a |
| E10_swin_t_tg2sd | 0.2989 | 0.3420 | 0.3583 | 0.6313 | n/a | n/a |
| E11_naive_withaug | 0.9435 | 0.9895 | 0.9727 | 0.9959 | 0.9988 | 1.0000 |
| E2_pooled_asl_ordinary | 0.5998 | 0.6259 | 0.8668 | 0.9037 | 0.5138 | 0.6352 |
| E2_pooled_bce_balanced | 0.6918 | 0.7332 | 0.9141 | 0.9535 | 0.8821 | 0.9634 |
| E2_pooled_bce_ordinary | 0.6807 | 0.7112 | 0.9203 | 0.9510 | 0.8349 | 0.9440 |
| E2_pooled_bce_strongcolor | 0.6666 | 0.7221 | 0.9048 | 0.9465 | 0.7758 | 0.8953 |
| E2_pooled_bce_withaug | 0.7763 | 0.8296 | 0.9288 | 0.9747 | 0.9449 | 0.9851 |
| E2_pooled_focal_ordinary | 0.5783 | 0.6102 | 0.8877 | 0.9294 | 0.6425 | 0.7057 |
| E3_cross_sd2tg | 0.1151 | 0.1023 | 0.6276 | 0.5287 | 0.0000 | n/a |
| E3_cross_tg2sd | 0.2007 | 0.2257 | 0.1265 | 0.3624 | n/a | n/a |
| E6_multitask_fixed | 0.6141 | 0.6160 | 0.8980 | 0.9294 | 0.6912 | 0.8130 |
| E6_multitask_uncertainty | 0.6500 | 0.6779 | 0.9040 | 0.9321 | 0.7404 | 0.7899 |
| E7_cross_mixstyle_sd2tg | 0.1090 | 0.0933 | 0.5009 | 0.6188 | 0.0000 | n/a |
| E7_cross_mixstyle_tg2sd | 0.0619 | 0.2312 | 0.1206 | 0.3529 | n/a | n/a |
| E7_mtl_coral | 0.6513 | 0.6810 | 0.8988 | 0.9390 | 0.7078 | 0.8440 |
| E7_mtl_mixstyle | 0.5561 | 0.5291 | 0.8565 | 0.8951 | 0.5702 | 0.6532 |
| E8_lightweight_disease | 0.7835 | 0.8831 | 0.9484 | 0.9804 | 0.9044 | 0.9736 |

`yellow_head` is supervised only in TigerShrimpBD (50 original images of that class), so its interval is wide and its estimate is fragile.

### 7.3 Biomass

Individual-level results are primary; image-level results are reported for completeness only. Weight is predicted in log space and the per-individual prediction is the MEDIAN over that animal's views (median is invariant to the log/exp transform, so there is no Jensen bias). R2 is referenced to the test mean, so a negative value means the image-only model does not beat predicting the held-out DB's mean weight.


**Headline reading.** Under strict leave-one-DB-out the image-only models reach individual-level weight R2 at or below zero: they predict close to the training-pond mean and do not track individual size. Multi-view aggregation does not rescue this, and learned attention pooling is worse than both mean pooling and the single-image model, consistent with over-parameterisation on ~120 training individuals across two ablation folds. The informative contrast is with the non-neural baselines (section 7.4): the allometric and tabular models reach R2 up to ~0.7-0.8, but only because they are given the ground-truth manual LENGTH. The bottleneck is therefore not the regressor but the inference of absolute size from pixels with no scale reference, across ponds whose camera, lighting and water differ. This is a genuine negative result: non-invasive image-only biomass estimation that generalises to an unseen pond is not achieved here. E5 uses 2 folds and E4 uses 3, so the paired comparison in section 9 is computed only over individuals both models evaluated.

| config | N ind | N img | held-out DBs | wt MAE (g) | wt RMSE (g) | wt R2 | wt MAPE % | len MAE (cm) | img wt MAE (g) |
|---|---|---|---|---|---|---|---|---|---|
| E12_biomass_loso | 170 | 16521 | DB1, DB2, DB3, DB4, DB5 | 9.620 | 12.438 | -0.420 | 90.3 | 1.366 | 9.484 |
| E13_biomass_within_db | 170 | 16521 | DB1, DB2, DB3, DB4, DB5 | 3.092 | 4.363 | 0.825 | 20.0 | 0.521 | 3.425 |
| E4_biomass_single_image | 170 | 16521 | DB1, DB2, DB3, DB4, DB5 | 6.231 | 9.257 | 0.214 | 67.1 | 1.104 | 6.577 |
| E5_biomass_multiview_attention | 170 | 510 | DB1, DB2, DB3, DB4, DB5 | 7.239 | 10.212 | 0.043 | 76.4 | 1.081 | 8.380 |
| E5_biomass_multiview_mean | 170 | 510 | DB1, DB2, DB3, DB4, DB5 | 6.868 | 9.044 | 0.249 | 50.5 | 0.910 | 7.987 |
| E6_multitask_fixed | 130 | 12633 | DB1, DB2, DB3 | 4.231 | 5.899 | -0.054 | 36.2 | 0.841 | 6.032 |
| E6_multitask_uncertainty | 130 | 12633 | DB1, DB2, DB3 | 3.995 | 5.300 | 0.149 | 33.6 | 0.821 | 5.251 |
| E7_mtl_coral | 130 | 12633 | DB1, DB2, DB3 | 3.693 | 5.267 | 0.160 | 34.8 | 0.773 | 4.664 |
| E7_mtl_mixstyle | 130 | 12633 | DB1, DB2, DB3 | 4.106 | 5.634 | 0.039 | 35.6 | 0.750 | 5.314 |
| E8_lightweight_biomass | 170 | 16521 | DB1, DB2, DB3, DB4, DB5 | 3.680 | 5.014 | 0.769 | 28.2 | 0.836 | 4.536 |

**Per held-out DB (individual-level weight).** The pooled points above average over each model's held-out DBs, and E4 and E5 do NOT cover the same DBs (E4 ran DB1-DB3, E5 ran DB1-DB2), so their pooled MAEs are not over the same population and must not be read as a head-to-head. On DB1 multi-view is in fact better; the pooled gap is driven by DB2. The like-for-like comparison is the paired test in section 9, restricted to shared individuals.

| config | held-out DB | N ind | MAE (g) | RMSE (g) | R2 |
|---|---|---|---|---|---|
| E12_biomass_loso | DB1 | 22 | 3.842 | 4.916 | 0.040 |
| E12_biomass_loso | DB2 | 58 | 3.714 | 6.168 | 0.059 |
| E12_biomass_loso | DB3 | 50 | 9.963 | 10.918 | -3.294 |
| E12_biomass_loso | DB4 | 20 | 21.219 | 21.256 | -219.662 |
| E12_biomass_loso | DB5 | 20 | 20.645 | 20.694 | -202.924 |
| E13_biomass_within_db | DB1 | 22 | 3.560 | 4.644 | 0.143 |
| E13_biomass_within_db | DB2 | 58 | 3.539 | 4.817 | 0.426 |
| E13_biomass_within_db | DB3 | 50 | 3.813 | 5.134 | 0.051 |
| E13_biomass_within_db | DB4 | 20 | 1.333 | 1.618 | -0.279 |
| E13_biomass_within_db | DB5 | 20 | 1.240 | 1.510 | -0.086 |
| E4_biomass_single_image | DB1 | 22 | 3.938 | 5.024 | -0.002 |
| E4_biomass_single_image | DB2 | 58 | 3.712 | 6.303 | 0.017 |
| E4_biomass_single_image | DB3 | 50 | 4.073 | 5.536 | -0.104 |
| E4_biomass_single_image | DB4 | 20 | 21.123 | 21.152 | -217.518 |
| E4_biomass_single_image | DB5 | 20 | 6.562 | 7.829 | -28.188 |
| E5_biomass_multiview_attention | DB1 | 22 | 5.258 | 6.727 | -0.798 |
| E5_biomass_multiview_attention | DB2 | 58 | 4.141 | 6.483 | -0.040 |
| E5_biomass_multiview_attention | DB3 | 50 | 3.906 | 5.528 | -0.101 |
| E5_biomass_multiview_attention | DB4 | 20 | 22.010 | 22.189 | -239.476 |
| E5_biomass_multiview_attention | DB5 | 20 | 11.962 | 12.079 | -68.479 |
| E5_biomass_multiview_mean | DB1 | 22 | 5.178 | 6.468 | -0.662 |
| E5_biomass_multiview_mean | DB2 | 58 | 4.717 | 7.246 | -0.299 |
| E5_biomass_multiview_mean | DB3 | 50 | 12.041 | 13.212 | -5.287 |
| E5_biomass_multiview_mean | DB4 | 20 | 1.606 | 2.387 | -1.782 |
| E5_biomass_multiview_mean | DB5 | 20 | 7.297 | 7.409 | -25.139 |
| E6_multitask_fixed | DB1 | 22 | 5.779 | 7.510 | -1.241 |
| E6_multitask_fixed | DB2 | 58 | 4.268 | 6.142 | 0.067 |
| E6_multitask_fixed | DB3 | 50 | 3.508 | 4.679 | 0.212 |
| E6_multitask_uncertainty | DB1 | 22 | 4.069 | 5.227 | -0.085 |
| E6_multitask_uncertainty | DB2 | 58 | 3.999 | 5.620 | 0.219 |
| E6_multitask_uncertainty | DB3 | 50 | 3.958 | 4.937 | 0.122 |
| E7_mtl_coral | DB1 | 22 | 3.710 | 5.157 | -0.057 |
| E7_mtl_coral | DB2 | 58 | 3.770 | 5.773 | 0.175 |
| E7_mtl_coral | DB3 | 50 | 3.595 | 4.666 | 0.216 |
| E7_mtl_mixstyle | DB1 | 22 | 3.557 | 4.687 | 0.127 |
| E7_mtl_mixstyle | DB2 | 58 | 3.417 | 5.557 | 0.236 |
| E7_mtl_mixstyle | DB3 | 50 | 5.148 | 6.086 | -0.334 |
| E8_lightweight_biomass | DB1 | 22 | 6.608 | 7.841 | -1.442 |
| E8_lightweight_biomass | DB2 | 58 | 3.200 | 4.666 | 0.461 |
| E8_lightweight_biomass | DB3 | 50 | 4.254 | 5.207 | 0.024 |
| E8_lightweight_biomass | DB4 | 20 | 1.436 | 1.854 | -0.679 |
| E8_lightweight_biomass | DB5 | 20 | 2.664 | 3.425 | -4.587 |

Image-only biomass regression does not beat the per-DB mean under leave-one-DB-out (R2 <= 0 on every fold): appearance differs enough between ponds/collections that a model trained on other DBs cannot calibrate absolute weight on a held-out DB from 170 animals across 5 DBs.

### 7.4 Non-neural biomass baselines (leave-one-DB-out)

| Fold | Held-out DB | Model | MAE (g) | RMSE (g) | R2 | Information advantage |
|---|---|---|---|---|---|---|
| 0 | DB1 | mean_baseline | 7.215 | 8.304 | -1.739 | none (constant prediction) |
| 0 | DB1 | allometric_W_eq_a_L_pow_b | 7.610 | 10.607 | -3.469 | GROUND-TRUTH manual length (not available to image-o |
| 0 | DB1 | linear_regression | 4.116 | 5.377 | -0.149 | GROUND-TRUTH manual length_cm |
| 0 | DB1 | random_forest | 3.402 | 4.109 | 0.329 | GROUND-TRUTH manual length_cm |
| 0 | DB1 | gradient_boosting | 4.859 | 5.623 | -0.256 | GROUND-TRUTH manual length_cm |
| 1 | DB2 | mean_baseline | 8.860 | 9.718 | -1.336 | none (constant prediction) |
| 1 | DB2 | allometric_W_eq_a_L_pow_b | 2.946 | 3.666 | 0.667 | GROUND-TRUTH manual length (not available to image-o |
| 1 | DB2 | linear_regression | 2.554 | 3.410 | 0.712 | GROUND-TRUTH manual length_cm + cephalothorax_cm |
| 1 | DB2 | random_forest | 2.196 | 2.708 | 0.819 | GROUND-TRUTH manual length_cm + cephalothorax_cm |
| 1 | DB2 | gradient_boosting | 2.287 | 2.816 | 0.804 | GROUND-TRUTH manual length_cm + cephalothorax_cm |
| 2 | DB3 | mean_baseline | 8.410 | 9.158 | -2.021 | none (constant prediction) |
| 2 | DB3 | allometric_W_eq_a_L_pow_b | 3.380 | 4.093 | 0.397 | GROUND-TRUTH manual length (not available to image-o |
| 2 | DB3 | linear_regression | 2.523 | 3.041 | 0.667 | GROUND-TRUTH manual length_cm + cephalothorax_cm |
| 2 | DB3 | random_forest | 2.281 | 2.905 | 0.696 | GROUND-TRUTH manual length_cm + cephalothorax_cm |
| 2 | DB3 | gradient_boosting | 2.673 | 3.403 | 0.583 | GROUND-TRUTH manual length_cm + cephalothorax_cm |
| 3 | DB4 | mean_baseline | 18.512 | 18.568 | -167.381 | none (constant prediction) |
| 3 | DB4 | allometric_W_eq_a_L_pow_b | 1.327 | 1.505 | -0.107 | GROUND-TRUTH manual length (not available to image-o |
| 3 | DB4 | linear_regression | 1.972 | 2.236 | -1.442 | GROUND-TRUTH manual length_cm + cephalothorax_cm |
| 3 | DB4 | random_forest | 0.581 | 0.740 | 0.732 | GROUND-TRUTH manual length_cm + cephalothorax_cm |
| 3 | DB4 | gradient_boosting | 0.536 | 0.826 | 0.667 | GROUND-TRUTH manual length_cm + cephalothorax_cm |
| 4 | DB5 | mean_baseline | 18.517 | 18.573 | -163.270 | none (constant prediction) |
| 4 | DB5 | allometric_W_eq_a_L_pow_b | 1.560 | 1.831 | -0.596 | GROUND-TRUTH manual length (not available to image-o |
| 4 | DB5 | linear_regression | 2.480 | 2.768 | -2.649 | GROUND-TRUTH manual length_cm + cephalothorax_cm |
| 4 | DB5 | random_forest | 0.614 | 0.945 | 0.575 | GROUND-TRUTH manual length_cm + cephalothorax_cm |
| 4 | DB5 | gradient_boosting | 0.770 | 1.126 | 0.396 | GROUND-TRUTH manual length_cm + cephalothorax_cm |

**These baselines are not like-for-like with the image models.** The allometric and tabular models receive the ground-truth manual length (and cephalothorax) measurement, which an image-only model never sees. They are an upper reference, not a competitor.

Fitted allometric relationships:
```
fold 0: W = 0.00559 * L^3.535 (log-space R2=0.963)
fold 1: W = 0.00707 * L^3.411 (log-space R2=0.939)
fold 2: W = 0.00795 * L^3.364 (log-space R2=0.953)
fold 3: W = 0.00987 * L^3.296 (log-space R2=0.949)
fold 4: W = 0.01108 * L^3.242 (log-space R2=0.944)
```

## 8. Confidence intervals

Disease intervals resample **fused groups**; biomass intervals resample **individual shrimp**. Neither treats correlated images as independent.

**Disease - health balanced accuracy, 95% cluster bootstrap**

| Configuration | N predictions | N groups | Point | 95% CI |
|---|---|---|---|---|
| E10_efficientnet_b0_sd2tg | 2853 | 831 | 0.6149 | [0.5945, 0.6360] |
| E10_efficientnet_b0_tg2sd | 2787 | 331 | 0.4394 | [0.4124, 0.4645] |
| E10_resnet50_sd2tg | 2853 | 831 | 0.5528 | [0.5379, 0.5678] |
| E10_resnet50_tg2sd | 2787 | 331 | 0.4649 | [0.4485, 0.4812] |
| E10_swin_t_sd2tg | 2853 | 831 | 0.5738 | [0.5572, 0.5920] |
| E10_swin_t_tg2sd | 2787 | 331 | 0.4499 | [0.4303, 0.4681] |
| E11_naive_withaug | 10305 | 1266 | 0.9756 | [0.9668, 0.9828] |
| E1_disease_single_sdibd | 2067 | 246 | 0.8123 | [0.7693, 0.8522] |
| E1_disease_single_tiger | 1803 | 528 | 0.9977 | [0.9953, 0.9995] |
| E2_pooled_asl_ordinary | 3873 | 777 | 0.9092 | [0.8876, 0.9298] |
| E2_pooled_bce_balanced | 3873 | 777 | 0.8938 | [0.8732, 0.9132] |
| E2_pooled_bce_ordinary | 3873 | 777 | 0.9156 | [0.8959, 0.9336] |
| E2_pooled_bce_strongcolor | 3873 | 777 | 0.9116 | [0.8906, 0.9311] |
| E2_pooled_bce_withaug | 3873 | 777 | 0.9342 | [0.9168, 0.9522] |
| E2_pooled_focal_ordinary | 3873 | 777 | 0.9005 | [0.8796, 0.9200] |
| E3_cross_sd2tg | 2853 | 831 | 0.4942 | [0.4797, 0.5101] |
| E3_cross_tg2sd | 2787 | 331 | 0.3803 | [0.3504, 0.4098] |
| E6_multitask_fixed | 3873 | 777 | 0.8713 | [0.8487, 0.8929] |
| E6_multitask_uncertainty | 3873 | 777 | 0.9065 | [0.8858, 0.9257] |
| E7_cross_mixstyle_sd2tg | 2853 | 831 | 0.5435 | [0.5298, 0.5563] |
| E7_cross_mixstyle_tg2sd | 2787 | 331 | 0.4440 | [0.4220, 0.4651] |
| E7_mtl_coral | 3873 | 777 | 0.9075 | [0.8876, 0.9255] |
| E7_mtl_mixstyle | 3873 | 777 | 0.8313 | [0.8098, 0.8510] |
| E8_lightweight_disease | 3873 | 777 | 0.9516 | [0.9369, 0.9651] |

**Biomass - individual-level weight MAE (g), 95% cluster bootstrap**

| Configuration | N individuals | Point | 95% CI |
|---|---|---|---|
| E12_biomass_loso | 170 | 9.6198 | [8.4661, 10.7814] |
| E13_biomass_within_db | 170 | 3.0924 | [2.6363, 3.5721] |
| E4_biomass_single_image | 170 | 6.2308 | [5.2220, 7.2739] |
| E5_biomass_multiview_attention | 170 | 7.2388 | [6.2103, 8.3178] |
| E5_biomass_multiview_mean | 170 | 6.8681 | [5.9846, 7.7929] |
| E6_multitask_fixed | 130 | 4.2315 | [3.5524, 4.9523] |
| E6_multitask_uncertainty | 130 | 3.9952 | [3.4104, 4.6096] |
| E7_mtl_coral | 130 | 3.6925 | [3.0543, 4.3714] |
| E7_mtl_mixstyle | 130 | 4.1062 | [3.4757, 4.7870] |
| E8_lightweight_biomass | 170 | 3.6804 | [3.1632, 4.2031] |

**Across seeds 42, 2026, 3407**

| Configuration | Metric | Seeds | Mean | SD | 95% t-CI |
|---|---|---|---|---|---|
| E10_efficientnet_b0_sd2tg | health_bal_acc | 3 | 0.6149 | 0.1125 | [0.3355, 0.8943] |
| E10_efficientnet_b0_sd2tg | attr_macro_f1 | 3 | 0.3170 | 0.0186 | [0.2707, 0.3632] |
| E10_efficientnet_b0_tg2sd | health_bal_acc | 3 | 0.4394 | 0.0166 | [0.3982, 0.4805] |
| E10_efficientnet_b0_tg2sd | attr_macro_f1 | 3 | 0.1681 | 0.0105 | [0.1419, 0.1943] |
| E10_resnet50_sd2tg | health_bal_acc | 3 | 0.5528 | 0.0294 | [0.4797, 0.6260] |
| E10_resnet50_sd2tg | attr_macro_f1 | 3 | 0.2748 | 0.0349 | [0.1881, 0.3615] |
| E10_resnet50_tg2sd | health_bal_acc | 3 | 0.4649 | 0.0294 | [0.3920, 0.5379] |
| E10_resnet50_tg2sd | attr_macro_f1 | 3 | 0.3820 | 0.1042 | [0.1232, 0.6408] |
| E10_swin_t_sd2tg | health_bal_acc | 3 | 0.5738 | 0.0398 | [0.4748, 0.6727] |
| E10_swin_t_sd2tg | attr_macro_f1 | 3 | 0.3232 | 0.0373 | [0.2304, 0.4159] |
| E10_swin_t_tg2sd | health_bal_acc | 3 | 0.4499 | 0.0313 | [0.3720, 0.5277] |
| E10_swin_t_tg2sd | attr_macro_f1 | 3 | 0.3286 | 0.0970 | [0.0876, 0.5697] |
| E11_naive_withaug | health_bal_acc | 3 | 0.9756 | 0.0021 | [0.9703, 0.9809] |
| E11_naive_withaug | attr_macro_f1 | 3 | 0.9717 | 0.0005 | [0.9704, 0.9730] |
| E12_biomass_loso | ind_weight_mae_g | 3 | 9.8291 | 0.8757 | [7.6539, 12.0044] |
| E12_biomass_loso | ind_weight_r2 | 3 | -62.2776 | 20.9357 | [-114.2848, -10.2704] |
| E13_biomass_within_db | ind_weight_mae_g | 3 | 3.1779 | 0.1279 | [2.8601, 3.4957] |
| E13_biomass_within_db | ind_weight_r2 | 3 | 0.8192 | 0.0121 | [0.7891, 0.8492] |
| E1_disease_single_sdibd | health_bal_acc | 3 | 0.8124 | 0.0209 | [0.7605, 0.8643] |
| E1_disease_single_sdibd | attr_macro_f1 | 3 | 0.7494 | 0.0343 | [0.6642, 0.8346] |
| E1_disease_single_tiger | health_bal_acc | 3 | 0.9977 | 0.0039 | [0.9880, 1.0075] |
| E1_disease_single_tiger | attr_macro_f1 | 3 | 0.7247 | 0.0371 | [0.6325, 0.8169] |
| E2_pooled_asl_ordinary | health_bal_acc | 3 | 0.9093 | 0.0138 | [0.8751, 0.9434] |
| E2_pooled_asl_ordinary | attr_macro_f1 | 3 | 0.6601 | 0.0583 | [0.5153, 0.8049] |
| E2_pooled_bce_balanced | health_bal_acc | 3 | 0.8935 | 0.0489 | [0.7721, 1.0150] |
| E2_pooled_bce_balanced | attr_macro_f1 | 3 | 0.8294 | 0.0274 | [0.7614, 0.8973] |
| E2_pooled_bce_ordinary | health_bal_acc | 3 | 0.9155 | 0.0303 | [0.8401, 0.9909] |
| E2_pooled_bce_ordinary | attr_macro_f1 | 3 | 0.8120 | 0.0403 | [0.7119, 0.9120] |
| E2_pooled_bce_strongcolor | health_bal_acc | 3 | 0.9117 | 0.0076 | [0.8927, 0.9307] |
| E2_pooled_bce_strongcolor | attr_macro_f1 | 3 | 0.7824 | 0.0516 | [0.6543, 0.9105] |
| E2_pooled_bce_withaug | health_bal_acc | 3 | 0.9342 | 0.0033 | [0.9259, 0.9424] |
| E2_pooled_bce_withaug | attr_macro_f1 | 3 | 0.8833 | 0.0186 | [0.8372, 0.9294] |
| E2_pooled_focal_ordinary | health_bal_acc | 3 | 0.9004 | 0.0270 | [0.8334, 0.9674] |
| E2_pooled_focal_ordinary | attr_macro_f1 | 3 | 0.7028 | 0.0725 | [0.5228, 0.8829] |
| E3_cross_sd2tg | health_bal_acc | 3 | 0.4942 | 0.0307 | [0.4179, 0.5704] |
| E3_cross_sd2tg | attr_macro_f1 | 3 | 0.2475 | 0.0348 | [0.1610, 0.3341] |
| E3_cross_tg2sd | health_bal_acc | 3 | 0.3803 | 0.0176 | [0.3365, 0.4241] |
| E3_cross_tg2sd | attr_macro_f1 | 3 | 0.1636 | 0.1207 | [-0.1363, 0.4635] |
| E4_biomass_single_image | ind_weight_mae_g | 3 | 8.0164 | 3.0123 | [0.5334, 15.4993] |
| E4_biomass_single_image | ind_weight_r2 | 3 | -52.9670 | 38.4015 | [-148.3616, 42.4275] |
| E5_biomass_multiview_attention | ind_weight_mae_g | 3 | 10.2561 | 3.4288 | [1.7384, 18.7738] |
| E5_biomass_multiview_attention | ind_weight_r2 | 3 | -66.0421 | 41.3355 | [-168.7251, 36.6409] |
| E5_biomass_multiview_mean | ind_weight_mae_g | 3 | 7.9901 | 2.3115 | [2.2481, 13.7321] |
| E5_biomass_multiview_mean | ind_weight_r2 | 3 | -27.5793 | 22.9495 | [-84.5891, 29.4305] |
| E6_multitask_fixed | health_bal_acc | 3 | 0.8712 | 0.0427 | [0.7651, 0.9773] |
| E6_multitask_fixed | attr_macro_f1 | 3 | 0.7345 | 0.1140 | [0.4514, 1.0175] |
| E6_multitask_fixed | ind_weight_mae_g | 3 | 5.7090 | 0.9551 | [3.3364, 8.0816] |
| E6_multitask_fixed | ind_weight_r2 | 3 | -1.1356 | 0.4493 | [-2.2516, -0.0196] |
| E6_multitask_uncertainty | health_bal_acc | 3 | 0.9065 | 0.0138 | [0.8722, 0.9409] |
| E6_multitask_uncertainty | attr_macro_f1 | 3 | 0.7648 | 0.0548 | [0.6286, 0.9009] |
| E6_multitask_uncertainty | ind_weight_mae_g | 3 | 4.7785 | 0.2166 | [4.2405, 5.3164] |
| E6_multitask_uncertainty | ind_weight_r2 | 3 | -0.2668 | 0.1210 | [-0.5674, 0.0338] |
| E7_cross_mixstyle_sd2tg | health_bal_acc | 3 | 0.5435 | 0.0701 | [0.3694, 0.7176] |
| E7_cross_mixstyle_sd2tg | attr_macro_f1 | 3 | 0.2033 | 0.0387 | [0.1073, 0.2993] |
| E7_cross_mixstyle_tg2sd | health_bal_acc | 3 | 0.4440 | 0.0502 | [0.3193, 0.5688] |
| E7_cross_mixstyle_tg2sd | attr_macro_f1 | 3 | 0.0912 | 0.0764 | [-0.0985, 0.2810] |
| E7_mtl_coral | health_bal_acc | 3 | 0.9075 | 0.0216 | [0.8539, 0.9611] |
| E7_mtl_coral | attr_macro_f1 | 3 | 0.7526 | 0.0694 | [0.5801, 0.9252] |
| E7_mtl_coral | ind_weight_mae_g | 3 | 4.2664 | 0.8383 | [2.1839, 6.3490] |
| E7_mtl_coral | ind_weight_r2 | 3 | -0.1678 | 0.4853 | [-1.3733, 1.0377] |
| E7_mtl_mixstyle | health_bal_acc | 3 | 0.8313 | 0.0772 | [0.6396, 1.0229] |
| E7_mtl_mixstyle | attr_macro_f1 | 3 | 0.6609 | 0.0623 | [0.5062, 0.8156] |
| E7_mtl_mixstyle | ind_weight_mae_g | 3 | 4.5841 | 0.8315 | [2.5185, 6.6498] |
| E7_mtl_mixstyle | ind_weight_r2 | 3 | -0.2440 | 0.2624 | [-0.8960, 0.4079] |
| E8_lightweight_biomass | ind_weight_mae_g | 3 | 4.0781 | 0.2232 | [3.5236, 4.6326] |
| E8_lightweight_biomass | ind_weight_r2 | 3 | -1.9416 | 0.8604 | [-4.0791, 0.1958] |
| E8_lightweight_disease | health_bal_acc | 3 | 0.9516 | 0.0078 | [0.9322, 0.9710] |
| E8_lightweight_disease | attr_macro_f1 | 3 | 0.8788 | 0.0194 | [0.8306, 0.9269] |

## 9. Statistical comparisons

Paired cluster bootstrap on shared test records/individuals. A difference is only called meaningful when its 95% interval excludes zero.


**How to read this table.** Each row is a paired cluster bootstrap: both models are scored on the records (or individuals) they share, aligned by `record_id`, and every replicate feeds the SAME resampled clusters to both, so the interval is on the within-pair difference. A row shows an interval only when both runs exist and share at least 20 test records (10 individuals for biomass); otherwise it is marked absent. The multi-task, domain-generalization and lightweight rows (E6-E8) were not run in this matrix and therefore appear as `one or both runs absent`, not as computed intervals.


**Pooled vs single-dataset (the first two rows).** E2 (pooled `disease_cv`) and E1 (`disease_cv_tiger` / `disease_cv_sdibd`) use different cross-validation partitions, but every image still appears exactly once as an out-of-fold test prediction in each, so pairing by `record_id` compares two genuinely held-out predictions of the same image. These rows are computed on 367 and 424 shared records respectively. Read them alongside the unpaired **per-source breakdown** in section 7.1, which restricts the pooled model to each source's own test images. Caveat: the within-set bootstrap resamples records and so yields a tight interval, whereas the across-fold Wilcoxon below rests on only 2-3 folds and is underpowered; where the two disagree, treat the pooled-vs-single effect as suggestive, not settled.

| Comparison (A vs B) | Diff (A-B) | 95% CI | p | CI excludes 0 |
|---|---|---|---|---|
| pooled vs single-dataset (TigerShrimpBD) | -0.0080 | [-0.0153, -0.0018] | 0.0070 | yes |
| pooled vs single-dataset (ShrimpDiseaseImageBD) | 0.0245 | [-0.0036, 0.0532] | 0.0850 | no |
| masked asymmetric loss vs BCE | -0.0063 | [-0.0171, 0.0048] | 0.2770 | no |
| masked focal loss vs BCE | -0.0150 | [-0.0255, -0.0049] | 0.0050 | yes |
| source-balanced vs ordinary sampling | -0.0218 | [-0.0329, -0.0110] | 0.0020 | yes |
| with published augmented copies vs originals only | 0.0186 | [0.0073, 0.0304] | 0.0020 | yes |
| aggressive colour augmentation vs disease-safe | -0.0039 | [-0.0143, 0.0068] | 0.4780 | no |
| multi-task vs single-task disease | -0.0442 | [-0.0606, -0.0272] | 0.0000 | yes |
| uncertainty weighting vs fixed task weights | 0.0352 | [0.0225, 0.0485] | 0.0000 | yes |
| MixStyle vs ERM multi-task | -0.0400 | [-0.0584, -0.0216] | 0.0000 | yes |
| CORAL vs ERM multi-task | 0.0361 | [0.0211, 0.0513] | 0.0000 | yes |
| MobileNetV3-Large vs ConvNeXt-Tiny | 0.0360 | [0.0230, 0.0494] | 0.0000 | yes |
| multi-view mean pooling vs single image | 0.6373 | [-0.8440, 1.9941] | 0.3590 | no |
| attention pooling vs mean pooling | 0.3707 | [-1.0667, 1.8644] | 0.6270 | no |
| multi-task biomass vs single-task biomass | 0.3427 | [-0.1377, 0.8987] | 0.1730 | no |
| MobileNetV3-Large vs ConvNeXt-Tiny (biomass) | -2.5504 | [-3.6684, -1.4714] | 0.0000 | yes |
| within-database CV vs leave-one-DB-out (same model; protocol effect) | -3.1384 | [-4.1918, -2.1768] | 0.0000 | yes |
| leave-one-site-out vs leave-one-DB-out | 3.3890 | [2.5266, 4.3391] | 0.0000 | yes |

**Wilcoxon signed-rank across folds**

| Comparison | Folds | W | p | Median diff |
|---|---|---|---|---|
| pooled vs single-dataset (TigerShrimpBD) | 3 | 0.00 | 0.0039 | -0.0701 |
| pooled vs single-dataset (ShrimpDiseaseImageBD) | 3 | 1.00 | 0.0078 | 0.0935 |
| masked asymmetric loss vs BCE | 3 | 17.00 | 0.5703 | -0.0077 |
| masked focal loss vs BCE | 3 | 18.00 | 0.6523 | -0.0057 |
| source-balanced vs ordinary sampling | 3 | 21.00 | 0.9102 | 0.0045 |
| with published augmented copies vs originals only | 3 | 12.00 | 0.2500 | 0.0151 |
| aggressive colour augmentation vs disease-safe | 3 | 10.00 | 0.1641 | -0.0132 |
| multi-task vs single-task disease | 3 | 7.00 | 0.1484 | -0.0181 |
| uncertainty weighting vs fixed task weights | 3 | 12.00 | 0.2500 | 0.0034 |
| MixStyle vs ERM multi-task | 3 | 11.00 | 0.2031 | -0.0218 |
| CORAL vs ERM multi-task | 3 | 13.00 | 0.3008 | 0.0092 |
| MobileNetV3-Large vs ConvNeXt-Tiny | 3 | 1.00 | 0.0078 | 0.0271 |

## 10. Runtime and hardware

- Platform: Windows-11-10.0.26300-SP0
- CPU: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel (20 logical cores)
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU (8.59 GB, capability 8.6)
- Python 3.13.5, torch 2.13.0+cu126 (CUDA 12.6), timm 1.0.28
- Total recorded training wall-clock: **25.24 hours** across 240 runs.

Longest runs:

| Run | Duration | Best epoch |
|---|---|---|
| E6_multitask_uncertainty_f0_s2026 | 18.6 min | 10 |
| E7_mtl_coral_f0_s2026 | 18.1 min | 9 |
| E7_mtl_mixstyle_f0_s3407 | 17.8 min | 11 |
| E6_multitask_fixed_f0_s2026 | 17.5 min | 10 |
| E6_multitask_uncertainty_f0_s42 | 17.3 min | 10 |

## 11. Error analysis


**E10_efficientnet_b0_sd2tg** (2853 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 255 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 773 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 65 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 40 | true WSSV scored higher for Black Gill |
| missed_co_infection | 0 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 1600 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 268 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| TigerShrimpBD | 2853 | 0.3603 | 255 | 773 |

**E10_efficientnet_b0_tg2sd** (2787 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 318 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 1112 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 87 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 666 | true WSSV scored higher for Black Gill |
| missed_co_infection | 0 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 0 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 896 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2787 | 0.5131 | 318 | 1112 |

**E10_resnet50_sd2tg** (2853 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 71 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 1078 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 96 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 45 | true WSSV scored higher for Black Gill |
| missed_co_infection | 0 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 1839 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 464 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| TigerShrimpBD | 2853 | 0.4027 | 71 | 1078 |

**E10_resnet50_tg2sd** (2787 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 159 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 1172 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 168 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 452 | true WSSV scored higher for Black Gill |
| missed_co_infection | 0 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 0 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 992 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2787 | 0.4776 | 159 | 1172 |

**E10_swin_t_sd2tg** (2853 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 107 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 996 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 46 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 143 | true WSSV scored higher for Black Gill |
| missed_co_infection | 0 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 2575 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 936 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| TigerShrimpBD | 2853 | 0.3866 | 107 | 996 |

**E10_swin_t_tg2sd** (2787 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 230 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 1154 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 28 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 630 | true WSSV scored higher for Black Gill |
| missed_co_infection | 0 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 0 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 1238 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2787 | 0.4966 | 230 | 1154 |

**E11_naive_withaug** (10305 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 46 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 128 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 36 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 20 | true WSSV scored higher for Black Gill |
| missed_co_infection | 92 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 4 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 110 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2070 | 0.0816 | 44 | 125 |
| TigerShrimpBD | 8235 | 0.0006 | 2 | 3 |

**E1_disease_single_sdibd** (2067 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 158 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 187 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 104 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 102 | true WSSV scored higher for Black Gill |
| missed_co_infection | 108 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 0 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 120 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2067 | 0.1669 | 158 | 187 |

**E1_disease_single_tiger** (1803 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 2 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 2 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 46 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 5 | true WSSV scored higher for Black Gill |
| missed_co_infection | 0 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 33 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 3 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| TigerShrimpBD | 1803 | 0.0022 | 2 | 2 |

**E2_pooled_asl_ordinary** (3873 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 138 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 184 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 229 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 151 | true WSSV scored higher for Black Gill |
| missed_co_infection | 15 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 172 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 93 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2067 | 0.1466 | 122 | 181 |
| TigerShrimpBD | 1806 | 0.0105 | 16 | 3 |

**E2_pooled_bce_balanced** (3873 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 170 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 210 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 114 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 66 | true WSSV scored higher for Black Gill |
| missed_co_infection | 198 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 34 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 111 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2067 | 0.1567 | 153 | 171 |
| TigerShrimpBD | 1806 | 0.0310 | 17 | 39 |

**E2_pooled_bce_ordinary** (3873 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 119 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 177 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 115 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 46 | true WSSV scored higher for Black Gill |
| missed_co_infection | 116 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 30 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 123 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2067 | 0.1355 | 112 | 168 |
| TigerShrimpBD | 1806 | 0.0089 | 7 | 9 |

**E2_pooled_bce_strongcolor** (3873 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 141 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 175 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 162 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 15 | true WSSV scored higher for Black Gill |
| missed_co_infection | 194 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 49 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 85 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2067 | 0.1456 | 134 | 167 |
| TigerShrimpBD | 1806 | 0.0083 | 7 | 8 |

**E2_pooled_bce_withaug** (3873 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 112 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 126 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 65 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 42 | true WSSV scored higher for Black Gill |
| missed_co_infection | 172 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 11 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 127 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2067 | 0.1147 | 111 | 126 |
| TigerShrimpBD | 1806 | 0.0006 | 1 | 0 |

**E2_pooled_focal_ordinary** (3873 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 109 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 228 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 307 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 20 | true WSSV scored higher for Black Gill |
| missed_co_infection | 362 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 85 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 115 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2067 | 0.1563 | 105 | 218 |
| TigerShrimpBD | 1806 | 0.0078 | 4 | 10 |

**E3_cross_sd2tg** (2853 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 212 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 1114 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 47 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 375 | true WSSV scored higher for Black Gill |
| missed_co_infection | 0 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 1283 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 995 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| TigerShrimpBD | 2853 | 0.4648 | 212 | 1114 |

**E3_cross_tg2sd** (2787 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 845 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 851 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 160 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 620 | true WSSV scored higher for Black Gill |
| missed_co_infection | 0 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 0 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 1515 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2787 | 0.6085 | 845 | 851 |

**E6_multitask_fixed** (3873 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 104 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 318 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 213 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 36 | true WSSV scored higher for Black Gill |
| missed_co_infection | 197 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 70 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 72 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2067 | 0.1858 | 95 | 289 |
| TigerShrimpBD | 1806 | 0.0210 | 9 | 29 |

**E6_multitask_uncertainty** (3873 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 167 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 174 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 194 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 25 | true WSSV scored higher for Black Gill |
| missed_co_infection | 75 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 63 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 106 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2067 | 0.1379 | 113 | 172 |
| TigerShrimpBD | 1806 | 0.0310 | 54 | 2 |

**E7_cross_mixstyle_sd2tg** (2853 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 274 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 939 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 62 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 341 | true WSSV scored higher for Black Gill |
| missed_co_infection | 0 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 1971 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 348 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| TigerShrimpBD | 2853 | 0.4252 | 274 | 939 |

**E7_cross_mixstyle_tg2sd** (2787 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 807 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 726 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 44 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 819 | true WSSV scored higher for Black Gill |
| missed_co_infection | 0 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 0 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 1406 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2787 | 0.5501 | 807 | 726 |

**E7_mtl_coral** (3873 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 209 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 145 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 178 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 61 | true WSSV scored higher for Black Gill |
| missed_co_infection | 149 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 42 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 97 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2067 | 0.1635 | 196 | 142 |
| TigerShrimpBD | 1806 | 0.0089 | 13 | 3 |

**E7_mtl_mixstyle** (3873 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 189 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 384 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 292 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 23 | true WSSV scored higher for Black Gill |
| missed_co_infection | 202 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 73 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 104 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2067 | 0.2409 | 175 | 323 |
| TigerShrimpBD | 1806 | 0.0415 | 14 | 61 |

**E8_lightweight_disease** (3873 test predictions)

| Category | N | Description |
|---|---|---|
| false_healthy | 106 | diseased shrimp predicted healthy (missed disease; the operationally costly error) |
| false_diseased | 78 | healthy shrimp predicted diseased (false alarm) |
| bg_predicted_as_wssv | 33 | true Black Gill scored higher for WSSV |
| wssv_predicted_as_bg | 33 | true WSSV scored higher for Black Gill |
| missed_co_infection | 125 | BG+WSSV co-infection where at least one attribute was missed |
| yellow_head_errors | 18 | Yellow Head errors among rows where Yellow Head is supervised (only TigerShrimpBD annota |
| confidently_wrong | 87 | health prediction wrong with confidence > 0.9 |

Per-source failure rates:

| Source | N | Health error rate | False healthy | False diseased |
|---|---|---|---|---|
| ShrimpDiseaseImageBD | 2067 | 0.0851 | 99 | 77 |
| TigerShrimpBD | 1806 | 0.0044 | 7 | 1 |

**E12_biomass_loso** (16521 images / 170 individuals)

| Held-out DB | N individuals | MAE (g) | MedAE (g) | Bias (g) |
|---|---|---|---|---|
| DB1 | 22 | 3.842 | 3.176 | -0.477 |
| DB2 | 58 | 3.714 | 2.221 | 0.241 |
| DB3 | 50 | 9.963 | 10.181 | -9.538 |
| DB4 | 20 | 21.219 | 21.295 | 21.219 |
| DB5 | 20 | 20.645 | 20.473 | 20.645 |

Size extremes: smallest 10% MAE 21.584 g (n=30), largest 10% MAE 10.734 g (n=19), median individual error 7.056 g.

**E13_biomass_within_db** (16521 images / 170 individuals)

| Held-out DB | N individuals | MAE (g) | MedAE (g) | Bias (g) |
|---|---|---|---|---|
| DB1 | 22 | 3.560 | 2.641 | -0.690 |
| DB2 | 58 | 3.539 | 2.665 | -0.362 |
| DB3 | 50 | 3.813 | 3.406 | -0.014 |
| DB4 | 20 | 1.333 | 1.344 | -0.142 |
| DB5 | 20 | 1.240 | 1.275 | -0.031 |

Size extremes: smallest 10% MAE 2.378 g (n=30), largest 10% MAE 6.597 g (n=19), median individual error 1.959 g.

**E4_biomass_single_image** (16521 images / 170 individuals)

| Held-out DB | N individuals | MAE (g) | MedAE (g) | Bias (g) |
|---|---|---|---|---|
| DB1 | 22 | 3.938 | 3.245 | -0.679 |
| DB2 | 58 | 3.712 | 2.120 | 0.531 |
| DB3 | 50 | 4.073 | 3.176 | 1.985 |
| DB4 | 20 | 21.123 | 21.252 | 21.123 |
| DB5 | 20 | 6.562 | 6.117 | 6.250 |

Size extremes: smallest 10% MAE 15.229 g (n=30), largest 10% MAE 6.018 g (n=19), median individual error 3.679 g.

**E5_biomass_multiview_attention** (510 images / 170 individuals)

| Held-out DB | N individuals | MAE (g) | MedAE (g) | Bias (g) |
|---|---|---|---|---|
| DB1 | 22 | 5.258 | 4.772 | 3.187 |
| DB2 | 58 | 4.141 | 2.500 | -1.096 |
| DB3 | 50 | 3.906 | 3.022 | 1.266 |
| DB4 | 20 | 22.010 | 22.505 | 22.010 |
| DB5 | 20 | 11.962 | 11.775 | 11.962 |

Size extremes: smallest 10% MAE 18.672 g (n=30), largest 10% MAE 6.604 g (n=19), median individual error 4.333 g.

**E5_biomass_multiview_mean** (510 images / 170 individuals)

| Held-out DB | N individuals | MAE (g) | MedAE (g) | Bias (g) |
|---|---|---|---|---|
| DB1 | 22 | 5.178 | 4.598 | 3.324 |
| DB2 | 58 | 4.717 | 3.299 | -1.140 |
| DB3 | 50 | 12.041 | 12.313 | -11.848 |
| DB4 | 20 | 1.606 | 1.000 | 0.849 |
| DB5 | 20 | 7.297 | 6.853 | 7.297 |

Size extremes: smallest 10% MAE 6.119 g (n=30), largest 10% MAE 11.836 g (n=19), median individual error 5.420 g.

**E6_multitask_fixed** (12633 images / 130 individuals)

| Held-out DB | N individuals | MAE (g) | MedAE (g) | Bias (g) |
|---|---|---|---|---|
| DB1 | 22 | 5.779 | 4.522 | -5.351 |
| DB2 | 58 | 4.268 | 3.121 | 2.009 |
| DB3 | 50 | 3.508 | 2.698 | 1.268 |

Size extremes: smallest 10% MAE 10.083 g (n=14), largest 10% MAE 5.389 g (n=19), median individual error 2.992 g.

**E6_multitask_uncertainty** (12633 images / 130 individuals)

| Held-out DB | N individuals | MAE (g) | MedAE (g) | Bias (g) |
|---|---|---|---|---|
| DB1 | 22 | 4.069 | 2.494 | -3.385 |
| DB2 | 58 | 3.999 | 3.408 | 2.650 |
| DB3 | 50 | 3.958 | 3.091 | -0.878 |

Size extremes: smallest 10% MAE 8.857 g (n=14), largest 10% MAE 6.059 g (n=19), median individual error 3.013 g.

**E7_mtl_coral** (12633 images / 130 individuals)

| Held-out DB | N individuals | MAE (g) | MedAE (g) | Bias (g) |
|---|---|---|---|---|
| DB1 | 22 | 3.710 | 2.462 | -2.875 |
| DB2 | 58 | 3.770 | 2.634 | 2.370 |
| DB3 | 50 | 3.595 | 2.884 | 0.336 |

Size extremes: smallest 10% MAE 10.157 g (n=14), largest 10% MAE 5.193 g (n=19), median individual error 2.651 g.

**E7_mtl_mixstyle** (12633 images / 130 individuals)

| Held-out DB | N individuals | MAE (g) | MedAE (g) | Bias (g) |
|---|---|---|---|---|
| DB1 | 22 | 3.557 | 2.437 | -0.880 |
| DB2 | 58 | 3.417 | 2.270 | 1.226 |
| DB3 | 50 | 5.148 | 5.004 | -4.179 |

Size extremes: smallest 10% MAE 7.792 g (n=14), largest 10% MAE 7.355 g (n=19), median individual error 3.002 g.

**E8_lightweight_biomass** (16521 images / 170 individuals)

| Held-out DB | N individuals | MAE (g) | MedAE (g) | Bias (g) |
|---|---|---|---|---|
| DB1 | 22 | 6.608 | 5.808 | -6.608 |
| DB2 | 58 | 3.200 | 2.375 | 1.514 |
| DB3 | 50 | 4.254 | 3.382 | -3.122 |
| DB4 | 20 | 1.436 | 1.253 | 1.362 |
| DB5 | 20 | 2.664 | 2.548 | -0.989 |

Size extremes: smallest 10% MAE 3.066 g (n=30), largest 10% MAE 7.301 g (n=19), median individual error 2.830 g.

Montages and the full breakdown are in `outputs/error_analysis/`.

## 12. Failed runs

No run failed. Every configuration that was launched completed and wrote predictions and metrics.

Failures encountered during the pipeline (resolved, recorded for transparency):

- `data.mendeley.com` returns HTTP 403 to every `requests`/urllib3 call regardless of User-Agent, while identical URLs succeed under stdlib `urllib`. Isolated by testing 4 User-Agent strings across both stacks; the downloader uses `urllib`.
- The CDN throttles sustained parallel range requests and stalls mid-transfer. Non-resumable ranges therefore never completed the 1.46 GB biomass archive (each retry discarded up to 91 MB). Ranges were made resumable (reconnect with `Range: bytes=<received>-`), after which the archive completed and matched its publisher-declared SHA-256.

## 13. Known limitations

0. **TigerShrimpBD's health label is perfectly confounded with acquisition session** (all healthy images June-July 2024, all diseased April 2024, disjoint days). Within-dataset healthy-versus-diseased accuracy on that dataset therefore measures session separability, not disease recognition, and a perfect score there is expected rather than impressive. See section 3.1. The same test could not be run on ShrimpDiseaseImageBD because its filenames carry no timestamp, so the confound is neither confirmed nor excluded there. This is the single most important caveat in this report.
1. **Class support is very small for two labels.** Using original images only, TigerShrimpBD contributes 39 Black Gill and 50 Yellow Head originals. Yellow Head is supervised in no other dataset. Per-label estimates for Yellow Head rest on ~10 test images per fold and should be read as indicative only.
2. **Yellow Head cannot be evaluated cross-dataset at all**, because ShrimpDiseaseImageBD has no such category; likewise BG_WSSV co-infection is inexpressible in TigerShrimpBD. The cross-dataset protocol is therefore scored on Healthy / Black Gill / WSSV only.
3. **Biomass generalization is measured across only five ponds/collections** and 170 animals. Leave-one-DB-out with 5 folds gives 5 domain-shift observations; the between-DB variance is large and the intervals reflect that.
4. **The biomass dataset version differs from the one cited in the task.** Version 1 is not retrievable through the Mendeley API (empty file listing), so version 2 was used. Recorded in `LICENSE_NOTES.md` and the audit.
5. **Compute profile is `standard`, not `full`.** The whole matrix ran on a single 8 GB laptop GPU. Measured cost at 5 folds / 14 epochs was 12-13 hours for E1-E8 alone, which would have completed E1-E2 thoroughly and left E4-E8 unrun; covering every mandatory experiment at 3 folds (2 for ablations) was judged more valuable than covering two of them at 5. E9 spends its budget on seeds rather than folds: 3 seeds x 2 folds per selected model. Headline numbers are therefore means over 3 folds (2 for ablations), not 5, and the intervals reflect that. Exact values are in section 5 and in every run's saved config.
6. **Grad-CAM / self-supervised pretraining / detection (section 17 optional extensions) were not run.** They are optional and were not reached.
7. **Perceptual near-duplicate fusion uses a Hamming threshold of 6 on a 64-bit pHash, blocked within dataset+class.** A different threshold would change group sizes and hence fold composition. The chosen value, the resulting group-size histogram and the 31 cross-dataset near-duplicate pairs are all recorded in the audit so the choice is auditable.

## 14. Exact reproduction commands

```bash
# 0. environment (Python 3.11+; this run used 3.13.5)
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r requirements-lock.txt

# 1. acquire data (verifies every SHA-256 against the publisher's value)
python scripts/download_data.py

# 2. manifest, cache, audit, splits  (audit exits non-zero if leakage-critical)
python scripts/build_manifests.py --workers 12
python scripts/build_image_cache.py --max-side 448 --workers 12
python scripts/audit_data.py
python scripts/build_splits.py --seed 42

# 3. preflight (tests, loaders, 20-sample overfit, 1 epoch, leakage)
python scripts/preflight.py

# 4. baselines and the mandatory matrix
python scripts/run_biomass_baselines.py
python scripts/train_disease.py   --profile standard --only E1 E2 E3
python scripts/train_biomass.py   --profile standard --only E4 E5
python scripts/train_multitask.py --profile standard --only E6 E7
python scripts/run_ablations.py   --profile standard --only E8
python scripts/run_e9_seeds.py    --profile standard

# 5. evaluation, figures, error analysis, export, report
python scripts/evaluate.py --n-boot 2000
python scripts/make_figures.py
python scripts/error_analysis.py
python scripts/export_best_lightweight.py
python scripts/write_report.py
```

Or simply `make all`.


## 15. Model export verification


**outputs/checkpoints/E8_lightweight_disease_f0_s2026_best.pt** (mobilenetv3_large)

| Format | Size | Max |diff| | Mean |diff| | allclose(1e-4) |
|---|---|---|---|---|
| torchscript | 20.753 MB | 5.150e-05 | 3.874e-06 | yes |
| onnx | 0.421 MB | 3.433e-05 | 3.057e-06 | yes |

| Runtime | ms/image | images/s |
|---|---|---|
| pytorch_cpu | 23.079 | 43.3 |
| onnxruntime_cpu | 5.155 | 194.0 |
| pytorch_gpu | 0.739 | 1353.5 |

Verified on 64 real images from the manifest.

**outputs/checkpoints/E8_lightweight_disease_f0_s42_best.pt** (mobilenetv3_large)

| Format | Size | Max |diff| | Mean |diff| | allclose(1e-4) |
|---|---|---|---|---|
| torchscript | 20.753 MB | 1.621e-05 | 2.174e-06 | yes |
| onnx | 0.421 MB | 2.003e-05 | 2.033e-06 | yes |

| Runtime | ms/image | images/s |
|---|---|---|
| pytorch_cpu | 20.026 | 49.9 |
| onnxruntime_cpu | 3.859 | 259.1 |
| pytorch_gpu | 0.742 | 1347.5 |

Verified on 64 real images from the manifest.