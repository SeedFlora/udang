"""Label mapping and partial-label mask semantics (specification section 7)."""
from __future__ import annotations

import pytest

from src.data.labels import (
    DISEASE_ATTRS, CROSS_DATASET_COMMON_CLASSES, harmonize,
    is_cross_dataset_eligible, normalize_raw_class,
)


class TestHarmonization:
    @pytest.mark.parametrize("raw,diseased,bg,wssv", [
        ("Healthy", 0, 0, 0), ("BG", 1, 1, 0), ("WSSV", 1, 0, 1), ("BG_WSSV", 1, 1, 1),
    ])
    def test_shrimp_disease_image_bd_mapping(self, raw, diseased, bg, wssv):
        h = harmonize("ShrimpDiseaseImageBD", raw)
        assert h.diseased == diseased
        assert h.attrs["black_gill"] == bg
        assert h.attrs["wssv"] == wssv
        # yellow_head is NOT assessed by this dataset -> must be unknown
        assert h.masks["yellow_head"] == 0, "yellow_head must be masked as unknown"
        assert h.masks["black_gill"] == 1 and h.masks["wssv"] == 1

    @pytest.mark.parametrize("raw,diseased,bg,wssv,yh", [
        ("Healthy", 0, 0, 0, 0), ("Black Gill", 1, 1, 0, 0),
        ("WSSV", 1, 0, 1, 0), ("Yellow Head", 1, 0, 0, 1),
    ])
    def test_tiger_mapping(self, raw, diseased, bg, wssv, yh):
        h = harmonize("TigerShrimpBD", raw)
        assert (h.diseased, h.attrs["black_gill"], h.attrs["wssv"],
                h.attrs["yellow_head"]) == (diseased, bg, wssv, yh)
        # every attribute is supervised in TigerShrimpBD
        assert all(h.masks[a] == 1 for a in DISEASE_ATTRS)

    def test_unknown_is_never_a_negative(self):
        """The whole point of the mask: an unknown must be distinguishable from 0."""
        sd = harmonize("ShrimpDiseaseImageBD", "Healthy")
        tg = harmonize("TigerShrimpBD", "Healthy")
        # both store yellow_head == 0, but only Tiger asserts it
        assert sd.attrs["yellow_head"] == tg.attrs["yellow_head"] == 0
        assert sd.masks["yellow_head"] == 0
        assert tg.masks["yellow_head"] == 1

    def test_co_infection_is_representable(self):
        h = harmonize("ShrimpDiseaseImageBD", "BG_WSSV")
        assert h.attrs["black_gill"] == 1 and h.attrs["wssv"] == 1

    @pytest.mark.parametrize("variant", ["bg_wssv", "BG WSSV", "bg-wssv", "  BG_WSSV  "])
    def test_spelling_variants_normalise(self, variant):
        assert normalize_raw_class("ShrimpDiseaseImageBD", variant) == "BG_WSSV"

    @pytest.mark.parametrize("variant", ["white spot syndrome virus", "WSSV", "wssv"])
    def test_tiger_spelling_variants(self, variant):
        assert normalize_raw_class("TigerShrimpBD", variant) == "WSSV"

    def test_unknown_class_refuses_to_guess(self):
        with pytest.raises(KeyError):
            harmonize("TigerShrimpBD", "Totally Made Up Disease")

    def test_as_row_emits_all_mask_columns(self):
        row = harmonize("TigerShrimpBD", "WSSV").as_row()
        for a in DISEASE_ATTRS:
            assert a in row and f"label_mask_{a}" in row


class TestCrossDatasetEligibility:
    def test_common_space_only(self):
        assert is_cross_dataset_eligible("ShrimpDiseaseImageBD", "Healthy")
        assert is_cross_dataset_eligible("ShrimpDiseaseImageBD", "BG")
        assert is_cross_dataset_eligible("ShrimpDiseaseImageBD", "WSSV")
        # co-infection cannot be expressed by TigerShrimpBD
        assert not is_cross_dataset_eligible("ShrimpDiseaseImageBD", "BG_WSSV")
        # Yellow Head cannot be expressed by ShrimpDiseaseImageBD
        assert not is_cross_dataset_eligible("TigerShrimpBD", "Yellow Head")
        assert is_cross_dataset_eligible("TigerShrimpBD", "Black Gill")

    def test_common_classes_constant(self):
        assert CROSS_DATASET_COMMON_CLASSES == ("Healthy", "Black Gill", "WSSV")


def test_attribute_order_is_stable():
    """Head indices, thresholds and metric vectors all assume this exact order."""
    assert DISEASE_ATTRS == ("black_gill", "wssv", "yellow_head")
