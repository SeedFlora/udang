"""Dataset parsing, spreadsheet parsing, and ONNX/TorchScript export consistency."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.audit.dedup import normalize_original_stem
from src.data.biomass_parse import (
    individual_id, parse_image_name, plausibility_flags, resolve_columns,
)
from src.data.manifest import (
    is_junk_path, parse_sdibd_name, scan_shrimp_disease_image_bd, sdibd_individual_id,
)
from src.models.multitask import ModelConfig, ShrimpMultiTaskModel


class TestBiomassImageNames:
    @pytest.mark.parametrize("name,ind,view", [
        ("s1_1.png", 1, 1), ("s22_15.png", 22, 15), ("s10_10.png", 10, 10),
        ("S3_7.PNG", 3, 7), (" s4_2.png ", 4, 2), ("s12-3.png", 12, 3),
    ])
    def test_parses_individual_and_view(self, name, ind, view):
        assert parse_image_name(name) == (ind, view)

    @pytest.mark.parametrize("name", ["IMG_1234.jpg", "shrimp.png", "s_1.png", "1_2.png"])
    def test_rejects_non_matching(self, name):
        assert parse_image_name(name) == (None, None)

    def test_individual_id_is_db_scoped(self):
        """Sample numbering restarts at 1 in every DB; a bare number would merge
        different animals across ponds."""
        assert individual_id("DB1", 1) != individual_id("DB2", 1)
        assert individual_id("DB1", 1) == individual_id("DB1", 1)


class TestSpreadsheetColumnResolution:
    @pytest.mark.parametrize("headers,expected", [
        (["SAMPLE", "LENGTH (cm)", "WEIGHT (g)", "COMPLETE SHRIMP IMAGES"],
         {"sample", "length_cm", "weight_g", "complete_body"}),
        (["SAMPLE", "CEPHALOTHORAX (cm)", "LENGTH (cm)", "WEIGHT (g)"],
         {"sample", "cephalothorax_cm", "length_cm", "weight_g"}),
        (["sample", "length", "weight"], {"sample", "length_cm", "weight_g"}),
        (["Sample ", " Weight (g) "], {"sample", "weight_g"}),
    ])
    def test_header_variants_map_to_canonical(self, headers, expected):
        df = pd.DataFrame(columns=headers)
        assert set(resolve_columns(df).values()) == expected

    def test_unknown_headers_are_ignored_not_guessed(self):
        df = pd.DataFrame(columns=["SAMPLE", "SOME UNRELATED COLUMN"])
        m = resolve_columns(df)
        assert set(m.values()) == {"sample"}


class TestPlausibility:
    def test_flags_impossible_values(self):
        assert "weight_out_of_range" in plausibility_flags(
            pd.Series({"weight_g": -1.0, "length_cm": 10.0}))
        assert "length_out_of_range" in plausibility_flags(
            pd.Series({"weight_g": 10.0, "length_cm": 99.0}))
        assert "cephalothorax_ge_length" in plausibility_flags(
            pd.Series({"weight_g": 10.0, "length_cm": 5.0, "cephalothorax_cm": 6.0}))

    def test_accepts_realistic_values(self):
        assert plausibility_flags(pd.Series(
            {"weight_g": 30.0, "length_cm": 11.3, "cephalothorax_cm": 3.5})) == []

    def test_missing_values_do_not_flag(self):
        assert plausibility_flags(pd.Series(
            {"weight_g": 20.0, "length_cm": 10.0, "cephalothorax_cm": np.nan})) == []


class TestJunkPathFiltering:
    """The ShrimpDiseaseImageBD zips carry a __MACOSX tree that mirrors every image
    path with a `._`-prefixed AppleDouble stub of the same extension."""

    @pytest.mark.parametrize("rel", [
        "__MACOSX/Raw Images/1. Healthy/._IMG.jpg",
        "Raw Images/1. Healthy/._IMG.jpg",
        "Raw Images/.DS_Store",
        "Raw Images/Thumbs.db",
    ])
    def test_rejects_junk(self, tmp_path, rel):
        assert is_junk_path(tmp_path / rel, tmp_path)

    @pytest.mark.parametrize("rel", [
        "Raw Images/1. Healthy/IMG.jpg",
        "Annotated Diseased Shrimp Images/1. BG/images/BG-28-img-2.jpg",
    ])
    def test_accepts_real_images(self, tmp_path, rel):
        assert not is_junk_path(tmp_path / rel, tmp_path)


class TestAugmentationStemNormalisation:
    def test_copy_suffix_with_spaced_separator(self):
        """Real filename: 'aug_0_IMG-20240421-WA0079 - Copy.jpg'. A single optional
        separator would leave a trailing ' -' and split the family."""
        a, _ = normalize_original_stem("aug_0_IMG-20240421-WA0079 - Copy.jpg")
        b, _ = normalize_original_stem("aug_0_IMG-20240421-WA0079.jpg")
        assert a == b == "IMG-20240421-WA0079"

    def test_stacked_markers(self):
        s, applied = normalize_original_stem("aug_3_IMG-1234_resized.jpg")
        assert s == "IMG-1234"
        assert len(applied) >= 2


class TestExportConsistency:
    """ONNX / TorchScript parity. Uses a small untrained model so the test is fast
    and offline, but exercises the exact wrapper used by scripts/export_model.py."""

    @staticmethod
    def _wrapper():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "export_model",
            Path(__file__).resolve().parents[1] / "scripts" / "export_model.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = ShrimpMultiTaskModel(ModelConfig(backbone="mobilenetv3_large",
                                                 pretrained=False))
        return mod.ExportWrapper(model).eval()

    def test_wrapper_output_layout(self):
        w = self._wrapper()
        with torch.no_grad():
            out = w(torch.randn(2, 3, 128, 128))
        # [health(1) | attrs(3) | biomass(3)]
        assert out.shape == (2, 7)

    def test_torchscript_matches_pytorch(self, tmp_path):
        w = self._wrapper()
        x = torch.randn(4, 3, 128, 128)
        with torch.no_grad():
            ref = w(x)
        ts = torch.jit.trace(w, x[:2])
        p = tmp_path / "m.pt"
        torch.jit.save(ts, p)
        with torch.no_grad():
            got = torch.jit.load(p)(x)
        assert torch.allclose(got, ref, atol=1e-4), \
            f"max diff {float((got - ref).abs().max()):.3e}"

    def test_onnx_matches_pytorch(self, tmp_path):
        ort = pytest.importorskip("onnxruntime")
        w = self._wrapper()
        x = torch.randn(4, 3, 128, 128)
        with torch.no_grad():
            ref = w(x)
        p = tmp_path / "m.onnx"
        torch.onnx.export(w, x[:2], p, input_names=["image"],
                          output_names=["predictions"],
                          dynamic_axes={"image": {0: "batch"},
                                        "predictions": {0: "batch"}},
                          opset_version=17)
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        got = torch.from_numpy(sess.run(None, {"image": x.numpy()})[0])
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-4), \
            f"max diff {float((got - ref).abs().max()):.3e}"


class TestSdibdImageNames:
    """SDIBD names are <CLASS>-<individual>-img-<view>.jpg: 416 animals over
    1,149 images. 118 of the 142 individual numbers recur across classes, so the
    animal key must be class-scoped or Healthy-14 and BG-14 would merge."""

    @pytest.mark.parametrize("name,cls,ind,view", [
        ("BG-34-img-3.jpg", "BG", 34, 3),
        ("Healthy-14-img-1.jpg", "Healthy", 14, 1),
        ("WSSV-118-img-2.JPG", "WSSV", 118, 2),
        ("WSSV_BG-33-img-2.jpg", "WSSV_BG", 33, 2),
        ("  bg-1-img-1.jpg ", "bg", 1, 1),
    ])
    def test_parses_class_individual_view(self, name, cls, ind, view):
        assert parse_sdibd_name(name) == (cls, ind, view)

    @pytest.mark.parametrize("name", [
        "IMG-20240414-WA0014.jpg", "s12_3.png", "BG-34.jpg", "BG-34-img.jpg",
        "._BG-34-img-3.jpg", "aug_0_BG-34-img-3.jpg",
    ])
    def test_rejects_non_matching(self, name):
        assert parse_sdibd_name(name) == (None, None, None)

    def test_individual_id_is_class_scoped(self):
        assert sdibd_individual_id("Healthy", 14) != sdibd_individual_id("BG", 14)
        assert sdibd_individual_id("BG", 14) == sdibd_individual_id("BG", 14)
        assert sdibd_individual_id("BG_WSSV", 7) == "sdibd::BG_WSSV::007"


def _jpg(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (120, 30, 30)).save(p, "JPEG")


class TestSdibdScannerSetsAnimalIds:
    def test_views_of_one_animal_share_an_id_and_classes_do_not_merge(self, tmp_path):
        root = tmp_path
        _jpg(root / "Raw Images" / "1. Healthy" / "Healthy-14-img-1.jpg")
        _jpg(root / "Raw Images" / "1. Healthy" / "Healthy-14-img-2.jpg")
        _jpg(root / "Raw Images" / "2. BG" / "BG-14-img-1.jpg")
        _jpg(root / "Annotated Diseased Shrimp Images" / "1. BG" / "images" / "BG-14-img-1.jpg")
        (root / "Annotated Diseased Shrimp Images" / "1. BG" / "labels").mkdir(parents=True)
        (root / "Annotated Diseased Shrimp Images" / "1. BG" / "labels" / "BG-14-img-1.txt").write_text("0 0.5 0.5 0.2 0.2")

        recs, diag = scan_shrimp_disease_image_bd(root, "3")
        by = {(r["path"].name, r["record_role"]): r for r in recs}

        h1 = by[("Healthy-14-img-1.jpg", "raw")]
        h2 = by[("Healthy-14-img-2.jpg", "raw")]
        bg_raw = by[("BG-14-img-1.jpg", "raw")]
        bg_ann = by[("BG-14-img-1.jpg", "annotated")]

        assert h1["individual_id"] == h2["individual_id"] == "sdibd::Healthy::014"
        assert (h1["view_index"], h2["view_index"]) == (1, 2)
        assert bg_raw["individual_id"] == bg_ann["individual_id"] == "sdibd::BG::014"
        assert h1["individual_id"] != bg_raw["individual_id"]
        assert diag["unparsed_names"] == []
        assert diag["class_token_mismatch"] == []

    def test_unparsable_and_mismatched_names_are_reported_not_guessed(self, tmp_path):
        root = tmp_path
        _jpg(root / "Raw Images" / "1. Healthy" / "IMG_0001.jpg")        # no pattern
        _jpg(root / "Raw Images" / "1. Healthy" / "BG-3-img-1.jpg")      # token != folder
        recs, diag = scan_shrimp_disease_image_bd(root, "3")
        by = {r["path"].name: r for r in recs}
        assert by["IMG_0001.jpg"]["individual_id"] is None
        assert diag["unparsed_names"] == ["Raw Images/1. Healthy/IMG_0001.jpg"]
        assert diag["class_token_mismatch"] == ["Raw Images/1. Healthy/BG-3-img-1.jpg"]
        # scoped by the FOLDER class, never by the filename token
        assert by["BG-3-img-1.jpg"]["individual_id"] == "sdibd::Healthy::003"

    def test_co_infection_class_token_matches_its_folder(self, tmp_path):
        """'WSSV_BG' normalises to the same canonical class as its folder token
        regardless of word order, so it must NOT be reported as a mismatch."""
        root = tmp_path
        _jpg(root / "Raw Images" / "4. WSSV_BG" / "WSSV_BG-33-img-2.jpg")
        recs, diag = scan_shrimp_disease_image_bd(root, "3")
        by = {r["path"].name: r for r in recs}
        assert diag["class_token_mismatch"] == []
        assert by["WSSV_BG-33-img-2.jpg"]["individual_id"] == "sdibd::BG_WSSV::033"
