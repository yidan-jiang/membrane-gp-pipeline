"""Tests on synthetic images; no microscope data or Cellpose install required."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from pm_pipeline import stages
from pm_pipeline.config import Config, load_config
from pm_pipeline.io_utils import Paths, average_projection
from pm_pipeline.run import run


# ----------------------------------------------------------------- helpers

def _two_touching_cells(shape=(40, 60)) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    labels[10:30, 10:30] = 1
    labels[10:30, 30:50] = 2  # shares an edge with cell 1
    return labels


# ----------------------------------------------------------------- erosion / rings

def test_footprints():
    np.testing.assert_array_equal(stages._footprint(2, "chebyshev"), np.ones((5, 5), bool))
    assert stages._footprint(3, "euclidean").shape == (7, 7)


@pytest.mark.parametrize("metric", ["euclidean", "chebyshev"])
@pytest.mark.parametrize("radius", [1, 3, 5, 7, 8, 9])
def test_erosion_matches_skimage_binary_erosion(metric, radius):
    """The scipy call is pixel-identical to the skimage wrapper it replaces."""
    import warnings

    from skimage.morphology import binary_erosion, disk

    rng = np.random.default_rng(radius)
    blob = np.zeros((120, 120), bool)
    for _ in range(6):
        r, c = rng.integers(20, 100, size=2)
        blob[max(r - 15, 0):r + 15, max(c - 10, 0):c + 20] = True
    footprint = disk(radius) if metric == "euclidean" else np.ones((2 * radius + 1,) * 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = binary_erosion(np.pad(blob, radius), footprint)[radius:-radius, radius:-radius]
    labels = blob.astype(np.int32)
    got = stages.erode_labels(labels, radius, metric) > 0
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("metric", ["euclidean", "chebyshev"])
def test_erosion_preserves_labels_and_never_merges(metric):
    labels = _two_touching_cells()
    interior = stages.erode_labels(labels, radius=3, metric=metric)
    assert set(np.unique(interior)) == {0, 1, 2}
    # Every interior pixel lies inside its own parent cell.
    assert np.all(labels[interior > 0] == interior[interior > 0])
    # Erosion is per label: the shared edge is eroded from both sides.
    assert not np.any(interior[:, 27:33])


def test_chebyshev_removes_exact_square_rim():
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[5:25, 5:25] = 7  # 20 x 20
    interior = stages.erode_labels(labels, radius=2, metric="chebyshev")
    rows, cols = np.nonzero(interior)
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (7, 22, 7, 22)  # 16 x 16


def test_rings_drop_cells_that_vanish(tmp_path):
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[5:25, 5:25] = 1  # survives r=3
    labels[30:33, 30:33] = 2  # 3 x 3, vanishes under r=3
    cfg = Config(experiment_dir=tmp_path, erosion_radii=[3])
    paths = Paths(cfg, "img")
    paths.qc_labels.parent.mkdir(parents=True)
    tifffile.imwrite(paths.qc_labels, labels.astype(np.uint16))

    stages.stage_rings(cfg, paths)
    interior = tifffile.imread(paths.interior(3))
    periphery = tifffile.imread(paths.periphery(3))

    assert 2 not in interior and 2 not in periphery
    # For surviving cells, interior and periphery partition the cell exactly.
    np.testing.assert_array_equal((interior > 0) | (periphery > 0), labels == 1)
    assert not np.any((interior > 0) & (periphery > 0))


def test_curated_labels_take_precedence(tmp_path):
    cfg = Config(experiment_dir=tmp_path)
    paths = Paths(cfg, "img")
    paths.qc_labels.parent.mkdir(parents=True)
    tifffile.imwrite(paths.qc_labels, np.zeros((4, 4), np.uint16))
    assert paths.working_labels() == paths.qc_labels
    tifffile.imwrite(paths.curated_labels, np.zeros((4, 4), np.uint16))
    assert paths.working_labels() == paths.curated_labels


# ----------------------------------------------------------------- statistics

def test_nan_safe_stats_all_nan_is_nan_not_zero():
    out = stages._nan_safe_stats(np.full(10, np.nan, dtype=np.float32))
    assert np.isnan(out["gp_mean"]) and out["n_valid_px"] == 0


def test_nan_safe_stats_ignores_nan_pixels():
    out = stages._nan_safe_stats(np.array([0.1, 0.3, np.nan, np.nan], dtype=np.float32))
    assert out["n_valid_px"] == 2
    assert out["gp_mean"] == pytest.approx(0.2)
    assert out["gp_median"] == pytest.approx(0.2)


# ----------------------------------------------------------------- projection

def test_projection_rounds_half_up_like_imagej():
    stack = np.array([[[0, 1, 2]], [[1, 2, 3]]], dtype=np.uint16)  # means 0.5, 1.5, 2.5
    projection = np.floor(average_projection(stack) + 0.5).astype(np.uint16)
    np.testing.assert_array_equal(projection, [[1, 2, 3]])


# ----------------------------------------------------------------- QC

def test_qc_uses_physical_area_and_records_reasons(tmp_path):
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[20:40, 20:40] = 1  # 400 px  -> 100 µm² at 0.5 µm/px: kept
    labels[60:63, 60:63] = 2  # 9 px    -> 2.25 µm²: too small
    labels[0:20, 70:90] = 3   # touches the top border
    cfg = Config(experiment_dir=tmp_path, min_area_um2=50, max_area_um2=3000)
    paths = Paths(cfg, "img")
    paths.cellpose_mask.parent.mkdir(parents=True)
    tifffile.imwrite(paths.cellpose_mask, labels.astype(np.uint16))

    stages.stage_qc(cfg, paths, pixel_size_um=0.5)
    table = pd.read_csv(paths.qc_table).set_index("label")
    kept = tifffile.imread(paths.qc_labels)

    assert table.loc[1, "kept"] and table.loc[1, "area_um2"] == pytest.approx(100.0)
    assert "too_small" in table.loc[2, "reason_dropped"]
    assert "touches_border" in table.loc[3, "reason_dropped"]
    assert set(np.unique(kept)) == {0, 1}


def test_qc_refuses_to_guess_without_pixel_size(tmp_path):
    with pytest.raises(ValueError, match="pixel size"):
        Config(experiment_dir=tmp_path).area_bounds_px(None)


# ----------------------------------------------------------------- paths / config

def test_glob_pattern_does_not_confuse_stem_1_with_10(tmp_path):
    (tmp_path / "GP_s_1_20250101.tif").touch()
    (tmp_path / "GP_s_10_20250101.tif").touch()
    cfg = Config(experiment_dir=tmp_path, gp_pattern="GP_{stem}_*.tif")
    assert Paths(cfg, "s_1").gp_image.name == "GP_s_1_20250101.tif"


def test_ambiguous_glob_raises(tmp_path):
    (tmp_path / "GP_s_1_a.tif").touch()
    (tmp_path / "GP_s_1_b.tif").touch()
    cfg = Config(experiment_dir=tmp_path, gp_pattern="GP_{stem}_*.tif")
    with pytest.raises(ValueError, match="matched 2 files"):
        _ = Paths(cfg, "s_1").gp_image


def test_config_roundtrip_and_unknown_keys(tmp_path):
    cfg = Config(experiment_dir=tmp_path, erosion_radii=[9, 5, 5], erosion_metric="chebyshev")
    assert cfg.erosion_radii == [5, 9]
    cfg.to_yaml(tmp_path / "pipeline.yaml")
    assert load_config(tmp_path).erosion_metric == "chebyshev"

    (tmp_path / "pipeline.yaml").write_text("not_a_key: 1\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(tmp_path)


# ----------------------------------------------------------------- end to end

def test_end_to_end_without_cellpose(tmp_path):
    """Stages A, C, E, F on a synthetic experiment whose masks already exist."""
    exp = tmp_path / "exp1"
    exp.mkdir()
    shape = (80, 80)

    stack = np.random.default_rng(0).integers(0, 100, size=(3, *shape), dtype=np.uint16)
    tifffile.imwrite(exp / "img1.tif", stack, photometric="minisblack")

    labels = np.zeros(shape, dtype=np.int32)
    labels[10:40, 10:40] = 1
    labels[45:75, 45:75] = 2

    gp = np.full(shape, np.nan, dtype=np.float32)
    gp[labels == 1] = 0.30
    gp[labels == 2] = -0.10
    gp[12, 12] = np.nan  # a NaN pixel inside the periphery of cell 1

    (exp / "ave_projected_images").mkdir()
    (exp / "gp_images").mkdir()
    tifffile.imwrite(exp / "ave_projected_images" / "img1_avgProj_cp_masks.tif",
                     labels.astype(np.uint16))
    tifffile.imwrite(exp / "gp_images" / "img1_gp_images.tif", gp)

    Config(experiment_dir=exp, raw_patterns=["*.tif"], erosion_radii=[3],
           pixel_size_um=0.5, min_area_um2=50).to_yaml(exp / "pipeline.yaml")

    assert run(exp, "ACEF", force=False) == 0

    assert (exp / "ave_projected_images" / "img1_avgProj.tif").exists()
    summary = pd.read_csv(exp / "results" / "exp1_summary.csv")
    assert len(summary) == 6  # 2 cells x {periphery, interior, whole}

    by = summary.set_index(["label", "region"])
    assert by.loc[(1, "periphery"), "gp_mean"] == pytest.approx(0.30)
    assert by.loc[(1, "periphery"), "n_nan_px"] == 1
    assert by.loc[(2, "whole"), "gp_mean"] == pytest.approx(-0.10)
    assert by.loc[(2, "whole"), "area_um2"] == pytest.approx(900 * 0.25)

    # Re-running is a no-op that still succeeds (outputs are skipped, not recomputed).
    ring = exp / "mask_PM" / "img1_periphery_r3.tif"
    before = ring.stat().st_mtime_ns
    assert run(exp, "ACEF", force=False) == 0
    assert ring.stat().st_mtime_ns == before


def test_curation_after_a_run_is_picked_up_by_stages_EF(tmp_path):
    """Regression: curated labels written after E/F must invalidate rings and measurements."""
    import os

    exp = tmp_path / "exp2"
    exp.mkdir()
    shape = (80, 80)
    tifffile.imwrite(exp / "img1.tif", np.zeros((2, *shape), np.uint16), photometric="minisblack")
    labels = np.zeros(shape, dtype=np.int32)
    labels[10:40, 10:40] = 1
    labels[45:75, 45:75] = 2
    gp = np.where(labels > 0, 0.25, np.nan).astype(np.float32)
    (exp / "ave_projected_images").mkdir()
    (exp / "gp_images").mkdir()
    tifffile.imwrite(exp / "ave_projected_images" / "img1_avgProj_cp_masks.tif",
                     labels.astype(np.uint16))
    tifffile.imwrite(exp / "gp_images" / "img1_gp_images.tif", gp)
    Config(experiment_dir=exp, raw_patterns=["*.tif"], erosion_radii=[3],
           pixel_size_um=0.5).to_yaml(exp / "pipeline.yaml")

    assert run(exp, "ACEF", force=False) == 0
    summary = exp / "results" / "exp2_summary.csv"
    assert set(pd.read_csv(summary)["label"]) == {1, 2}

    # Curate: remove cell 2, with a timestamp clearly after the first run.
    curated = labels.copy()
    curated[curated == 2] = 0
    cpath = exp / "qc" / "img1_labels_curated.tif"
    tifffile.imwrite(cpath, curated.astype(np.uint16))
    later = (exp / "results" / "img1_gp_r3.csv").stat().st_mtime + 10
    os.utime(cpath, (later, later))

    assert run(exp, "EF", force=False) == 0
    assert set(pd.read_csv(summary)["label"]) == {1}
    assert 2 not in tifffile.imread(exp / "mask_PM" / "img1_periphery_r3.tif")
