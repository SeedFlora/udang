"""Experiment registry: the mandatory matrix E1-E9 as explicit configurations.

Three compute profiles (specification section 19):

  smoke     - a handful of steps on a tiny subset, one fold. Proves the pipeline
              runs end to end. Produces NO reportable claim.
  standard  - the full mandatory matrix E0-E8 at one seed with reduced epochs and,
              for ablations only, reduced fold counts.
  full      - standard plus the three-seed E9 runs and complete fold coverage.

Two leveled round-1 revision profiles: `standard3` (all disease folds/ablations at
3) and `standard5` (all biomass folds/ablations at 5, so DB4/DB5 are held out).

Every reduction relative to `full` is recorded in the config so the report can
state exactly what was shortened rather than implying full coverage.
"""
from __future__ import annotations

from dataclasses import replace

from src.models.multitask import ModelConfig
from src.training.experiment import ExperimentConfig
from src.training.trainer import TrainConfig

SEEDS = (42, 2026, 3407)

PROFILES = {
    "smoke":    {"epochs": 1, "folds": 1, "ablation_folds": 1, "image_size": 160,
                 "patience": 1, "limit": 6},
    # `standard` is sized so that the ENTIRE mandatory matrix E1-E9 completes on one
    # 8 GB laptop GPU. Measured cost at 5 folds / 14 epochs was ~12-13 h for E1-E8
    # alone, which would have finished E1-E2 thoroughly and left E4-E8 unrun.
    # Covering every mandatory experiment at 3 folds is worth more than covering two
    # of them at 5. The reduction is stated explicitly in the report rather than
    # presented as full coverage.
    "standard": {"epochs": 12, "folds": 3, "ablation_folds": 2, "image_size": 320,
                 "patience": 4, "limit": None},
    "full":     {"epochs": 30, "folds": 5, "ablation_folds": 5, "image_size": 320,
                 "patience": 8, "limit": None},
    # Round-1 revision (reviewer A5 "consistent conditions"): the same recipe as
    # `standard` with every fold count leveled. `standard3` levels the disease
    # ablations to 3 folds; `standard5` runs the five biomass folds so DB4 and DB5
    # are actually held out (reviewer A9). Epochs and patience are unchanged so
    # new and kept runs are comparable.
    "standard3": {"epochs": 12, "folds": 3, "ablation_folds": 3, "image_size": 320,
                  "patience": 4, "limit": None},
    "standard5": {"epochs": 12, "folds": 5, "ablation_folds": 5, "image_size": 320,
                  "patience": 4, "limit": None},
}


def _train(p: dict, **kw) -> TrainConfig:
    base = TrainConfig(
        epochs=p["epochs"], early_stop_patience=p["patience"],
        # guarantee at least half the epoch budget before early stopping may fire
        min_epochs=max(1, p["epochs"] // 2),
        lr=3e-4, weight_decay=1e-4, warmup_epochs=1 if p["epochs"] <= 3 else 2,
        amp=True, monitor="val_primary", monitor_mode="max",
    )
    return replace(base, **kw)


def _folds(n: int) -> tuple[int, ...]:
    return tuple(range(n))


def build_matrix(profile: str = "standard", seed: int = 42,
                 only: list[str] | None = None) -> list[ExperimentConfig]:
    p = PROFILES[profile]
    F, AF, IS = p["folds"], p["ablation_folds"], p["image_size"]
    bs = 12 if IS >= 320 else 24
    cfgs: list[ExperimentConfig] = []

    def add(c: ExperimentConfig) -> None:
        if only is None or c.experiment in only or c.name in only:
            cfgs.append(c)

    # num_workers=0 (main-process loading). This Windows + Python 3.13 setup
    # intermittently fails DataLoader worker spawn with
    # "_pickle.UnpicklingError: pickle data was truncated" -- the parent's pickle
    # pipe to a spawning worker breaks mid-send -- which hard-kills the whole
    # training process (it is not catchable per-fold). It recurred on multiple runs
    # once persistent_workers was disabled (the RAM-leak fix), because workers then
    # respawn every epoch, multiplying spawn events. Loading in the main process
    # removes the spawn path entirely. Worker COUNT does not affect batch ordering
    # or results -- only load parallelism -- so this is safe to mix with the runs
    # already completed at num_workers>0. The image cache (448 px, ~2 ms/decode)
    # keeps the single-process loader fast enough.
    common = dict(image_size=IS, batch_size=bs, biomass_batch_size=bs, seed=seed,
                  num_workers=0)

    # ---------------- E1: single-dataset disease baselines ----------------
    for ds, proto in (("sdibd", "disease_cv_sdibd"), ("tiger", "disease_cv_tiger")):
        add(ExperimentConfig(
            name=f"E1_disease_single_{ds}", experiment="E1", tasks=("disease",),
            protocol=proto, folds=_folds(F),
            model=ModelConfig(backbone="convnext_tiny", pretrained=True,
                              enable_biomass=False),
            train=_train(p), notes=f"single-dataset disease baseline on {ds}", **common))

    # ---------------- E2: pooled hierarchical disease ----------------
    # headline configuration
    add(ExperimentConfig(
        name="E2_pooled_bce_ordinary", experiment="E2", tasks=("disease",),
        protocol="disease_cv", folds=_folds(F),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True, enable_biomass=False),
        train=_train(p, attr_loss="bce"), source_balanced=False,
        notes="pooled disease, BCE, ordinary sampling", **common))
    # loss ablation
    add(ExperimentConfig(
        name="E2_pooled_asl_ordinary", experiment="E2", tasks=("disease",),
        protocol="disease_cv", folds=_folds(AF),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True, enable_biomass=False),
        train=_train(p, attr_loss="asl"), source_balanced=False,
        notes="ABLATION: masked asymmetric loss vs BCE", **common))
    add(ExperimentConfig(
        name="E2_pooled_focal_ordinary", experiment="E2", tasks=("disease",),
        protocol="disease_cv", folds=_folds(AF),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True, enable_biomass=False),
        train=_train(p, attr_loss="focal"), source_balanced=False,
        notes="ABLATION: masked focal loss vs BCE", **common))
    # sampling ablation
    add(ExperimentConfig(
        name="E2_pooled_bce_balanced", experiment="E2", tasks=("disease",),
        protocol="disease_cv", folds=_folds(AF),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True, enable_biomass=False),
        train=_train(p, attr_loss="bce"), source_balanced=True,
        notes="ABLATION: source-balanced sampling vs ordinary", **common))
    # published-augmentation ablation
    add(ExperimentConfig(
        name="E2_pooled_bce_withaug", experiment="E2", tasks=("disease",),
        protocol="disease_cv_withaug", folds=_folds(AF),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True, enable_biomass=False),
        train=_train(p, attr_loss="bce"), include_published_aug=True,
        notes="ABLATION: with published TigerShrimpBD augmented copies "
              "(leakage-safe grouping retained)", **common))
    # colour-augmentation ablation
    add(ExperimentConfig(
        name="E2_pooled_bce_strongcolor", experiment="E2", tasks=("disease",),
        protocol="disease_cv", folds=_folds(AF),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True, enable_biomass=False),
        train=_train(p, attr_loss="bce"), strong_color_aug=True,
        notes="ABLATION: aggressive colour transforms (disease signs are colour-dependent)",
        **common))

    # ---------------- E3: cross-dataset ----------------
    for proto in ("cross_sd2tg", "cross_tg2sd"):
        add(ExperimentConfig(
            name=f"E3_{proto}", experiment="E3", tasks=("disease",),
            protocol=proto, folds=(0,),
            model=ModelConfig(backbone="convnext_tiny", pretrained=True,
                              enable_biomass=False),
            train=_train(p, attr_loss="bce"),
            notes="cross-dataset transfer; thresholds tuned on SOURCE validation only",
            **common))

    # ---------------- E4: biomass single-image ----------------
    add(ExperimentConfig(
        name="E4_biomass_single_image", experiment="E4", tasks=("biomass",),
        protocol="biomass_lodo", folds=_folds(F),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True,
                          enable_health=False, enable_attrs=False),
        train=_train(p, monitor="val_primary"), multiview=False,
        notes="single-image biomass regression, leave-one-DB-out", **common))

    # ---------------- E5: biomass multi-view ----------------
    for pool in ("mean", "attention"):
        add(ExperimentConfig(
            name=f"E5_biomass_multiview_{pool}", experiment="E5", tasks=("biomass",),
            protocol="biomass_lodo", folds=_folds(AF),
            model=ModelConfig(backbone="convnext_tiny", pretrained=True,
                              enable_health=False, enable_attrs=False, pooling=pool),
            train=_train(p), multiview=True, max_views=4,
            notes=f"multi-view individual-level regression, {pool} pooling", **common))

    # ---------------- E6: shared-encoder multi-task ----------------
    add(ExperimentConfig(
        name="E6_multitask_fixed", experiment="E6", tasks=("disease", "biomass"),
        protocol="disease_cv", folds=_folds(AF),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True),
        train=_train(p, task_weighting="fixed"),
        notes="shared encoder, disease + biomass, fixed task weights", **common))
    add(ExperimentConfig(
        name="E6_multitask_uncertainty", experiment="E6", tasks=("disease", "biomass"),
        protocol="disease_cv", folds=_folds(AF),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True),
        train=_train(p, task_weighting="uncertainty"),
        notes="shared encoder, learned homoscedastic uncertainty weighting", **common))

    # ---------------- E7: domain generalization ----------------
    add(ExperimentConfig(
        name="E7_mtl_mixstyle", experiment="E7", tasks=("disease", "biomass"),
        protocol="disease_cv", folds=_folds(AF),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True, mixstyle=True,
                          mixstyle_p=0.5, mixstyle_alpha=0.1),
        train=_train(p), source_balanced=True,
        notes="ERM + source balancing + MixStyle in early stages", **common))
    add(ExperimentConfig(
        name="E7_mtl_coral", experiment="E7", tasks=("disease", "biomass"),
        protocol="disease_cv", folds=_folds(AF),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True),
        train=_train(p, coral_lambda=0.5), source_balanced=True,
        notes="ERM + source balancing + CORAL feature alignment", **common))
    add(ExperimentConfig(
        name="E7_cross_mixstyle_sd2tg", experiment="E7", tasks=("disease",),
        protocol="cross_sd2tg", folds=(0,),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True, mixstyle=True,
                          enable_biomass=False),
        train=_train(p), source_balanced=True,
        notes="MixStyle effect on cross-dataset transfer", **common))
    add(ExperimentConfig(
        name="E7_cross_mixstyle_tg2sd", experiment="E7", tasks=("disease",),
        protocol="cross_tg2sd", folds=(0,),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True, mixstyle=True,
                          enable_biomass=False),
        train=_train(p), source_balanced=True,
        notes="MixStyle effect on cross-dataset transfer", **common))

    # ---------------- E8: lightweight ----------------
    add(ExperimentConfig(
        name="E8_lightweight_disease", experiment="E8", tasks=("disease",),
        protocol="disease_cv", folds=_folds(AF),
        model=ModelConfig(backbone="mobilenetv3_large", pretrained=True,
                          enable_biomass=False),
        train=_train(p, attr_loss="bce"),
        notes="MobileNetV3-Large efficiency comparison, pooled disease", **common))
    add(ExperimentConfig(
        name="E8_lightweight_biomass", experiment="E8", tasks=("biomass",),
        protocol="biomass_lodo", folds=_folds(AF),
        model=ModelConfig(backbone="mobilenetv3_large", pretrained=True,
                          enable_health=False, enable_attrs=False),
        train=_train(p), multiview=False,
        notes="MobileNetV3-Large efficiency comparison, biomass", **common))

    # ---------------- E10: cross-dataset backbone sweep (revision, A7) ----------
    # Identical recipe to E3; only the encoder changes. Swin is constructed at the
    # training resolution (fixed-size backbone); SWIN_KW is empty because Task 8's
    # probe confirmed the plain img_size=320 call loads pretrained weights and
    # forwards correctly with no window_size override needed.
    SWIN_KW = {}   # e.g. {"window_size": 10} if the plain img_size=320 probe failed
    for bb in ("resnet50", "efficientnet_b0", "swin_t"):
        for proto, tag in (("cross_sd2tg", "sd2tg"), ("cross_tg2sd", "tg2sd")):
            add(ExperimentConfig(
                name=f"E10_{bb}_{tag}", experiment="E10", tasks=("disease",),
                protocol=proto, folds=(0,),
                model=ModelConfig(backbone=bb, pretrained=True, enable_biomass=False,
                                  img_size=IS if bb == "swin_t" else None,
                                  backbone_kwargs=dict(SWIN_KW) if bb == "swin_t" else {}),
                train=_train(p, attr_loss="bce"),
                notes=f"REVISION A7: cross-dataset backbone sweep ({bb}), same recipe as E3",
                **common))

    # ---------------- E11: naive image-level split, augmented pool (revision, A4) --
    add(ExperimentConfig(
        name="E11_naive_withaug", experiment="E11", tasks=("disease",),
        protocol="disease_cv_naive_withaug", folds=_folds(AF),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True, enable_biomass=False),
        train=_train(p, attr_loss="bce"), include_published_aug=True,
        notes="REVISION A4 CONTROL ARM: image-level random split over the augmented "
              "pool, no grouping; DELIBERATELY leaky, reported as protocol-ladder rung 1",
        **common))

    # ---------------- E12 / E13: biomass protocol controls (revision, A8/A9) -----
    # Both protocol fields are set on purpose: the loader reads biomass_protocol,
    # the registry logs protocol; setting one would label a LOSO run that trained LODO.
    add(ExperimentConfig(
        name="E12_biomass_loso", experiment="E12", tasks=("biomass",),
        protocol="biomass_loso", biomass_protocol="biomass_loso", folds=(0, 1, 2),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True,
                          enable_health=False, enable_attrs=False),
        train=_train(p), multiview=False,
        notes="REVISION A9: leave-one-SITE-out (DB1 | DB2+DB3 | DB4+DB5)", **common))
    add(ExperimentConfig(
        name="E13_biomass_within_db", experiment="E13", tasks=("biomass",),
        protocol="biomass_within_db", biomass_protocol="biomass_within_db",
        folds=(0, 1, 2, 3, 4),
        model=ModelConfig(backbone="convnext_tiny", pretrained=True,
                          enable_health=False, enable_attrs=False),
        train=_train(p), multiview=False,
        notes="REVISION A4/A8: within-database grouped CV, the protocol control for LODO",
        **common))

    return cfgs


def build_e9(profile: str = "full", best: dict | None = None) -> list[ExperimentConfig]:
    """Final repeated runs at three seeds for the selected best models.

    `best` maps a role to a config name chosen on VALIDATION performance only.
    """
    p = PROFILES[profile]
    best = best or {
        "disease_single": "E1_disease_single_tiger",
        "disease_pooled": "E2_pooled_bce_ordinary",
        "biomass_single_image": "E4_biomass_single_image",
        "multitask": "E6_multitask_fixed",
        "lightweight": "E8_lightweight_disease",
    }
    # Seeds are the axis E9 exists to measure, so spend the budget there rather
    # than on folds: 3 seeds x 2 folds gives 6 independent estimates per selected
    # model, enough for a mean, an SD and a bootstrap interval, at 40% of the cost
    # of 3 seeds x 3 folds.
    e9_folds = _folds(min(2, p["folds"]))
    out: list[ExperimentConfig] = []
    for seed in SEEDS:
        for c in build_matrix(profile, seed=seed, only=list(best.values())):
            out.append(replace(c, name=f"E9_{c.name}", experiment="E9",
                               folds=e9_folds,
                               notes=f"E9 repeated run (seed {seed}) of {c.name}"))
    return out
