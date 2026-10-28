"""Pipeline stages A-F. Each stage is independently callable and skips existing output.

Stages E and F additionally recompute when an input (curated labels, rings, GP image) is
newer than their output, so re-running after curation picks up the edits.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import disk

from .config import Config
from .io_utils import (
    Paths,
    assert_same_shape,
    average_projection,
    load_raw_stack,
    read_projection,
    read_labels,
    read_pixel_size_um,
    write_labels,
)

logger = logging.getLogger(__name__)


def _up_to_date(output: Path, *inputs: Path) -> bool:
    """True if `output` exists and is at least as new as every existing input.

    Stages E and F depend on files a person can edit between runs (curated labels written
    by stage D, regenerated GP images). Skipping on existence alone would silently keep
    measurements made from the old inputs.
    """
    if not output.exists():
        return False
    newest_input = max((p.stat().st_mtime for p in inputs if p.exists()), default=0.0)
    return output.stat().st_mtime >= newest_input


# --------------------------------------------------------------------------- A

def stage_projection(cfg: Config, raw_path: Path, paths: Paths, force: bool = False) -> Path:
    """Average over non-spatial axes to get the segmentation reference image."""
    if paths.projection.exists() and not force:
        logger.debug("projection exists, skipping: %s", paths.projection.name)
        return paths.projection

    stack = load_raw_stack(raw_path)
    projection = average_projection(stack)
    # floor(x + 0.5) is round-half-up, matching ImageJ's Z Project. np.round would use
    # banker's rounding and differ by 1 grey level on ~1.6% of pixels.
    projection = np.floor(projection + 0.5).astype(np.uint16)

    paths.projection.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(paths.projection, projection)
    logger.info("projection -> %s", paths.projection.name)
    return paths.projection


# --------------------------------------------------------------------------- B

def _build_cellpose_model(cfg: Config):
    from cellpose import models

    try:
        return models.Cellpose(gpu=cfg.cellpose_gpu, model_type=cfg.cellpose_model)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not build Cellpose model {cfg.cellpose_model!r}: {exc}. "
            "Cellpose 4.x dropped the cyto/cyto2 model_type API; install cellpose<4 "
            "(pip install 'cellpose>=2,<4'), or set cellpose_model to one your installed "
            "version supports."
        ) from exc


def stage_segment(cfg: Config, paths: Paths, model=None, force: bool = False) -> Path:
    """Cellpose segmentation of the average projection."""
    if paths.cellpose_mask.exists() and not force:
        logger.debug("mask exists, skipping: %s", paths.cellpose_mask.name)
        return paths.cellpose_mask

    if model is None:
        model = _build_cellpose_model(cfg)

    projection = read_projection(paths.projection)
    result = model.eval(projection, diameter=cfg.cellpose_diameter, channels=[0, 0])
    masks = result[0]

    write_labels(paths.cellpose_mask, masks)
    logger.info("segmented %s -> %d objects", paths.cellpose_mask.name, int(masks.max()))
    return paths.cellpose_mask


# --------------------------------------------------------------------------- C

def stage_qc(cfg: Config, paths: Paths, pixel_size_um: float | None, force: bool = False) -> Path:
    """Drop debris and mis-segmented cells, recording why each label went."""
    if paths.qc_labels.exists() and not force:
        logger.debug("qc labels exist, skipping: %s", paths.qc_labels.name)
        return paths.qc_labels

    labels = read_labels(paths.cellpose_mask)
    min_area, max_area = cfg.area_bounds_px(pixel_size_um)
    height, width = labels.shape
    area_scale = pixel_size_um**2 if pixel_size_um else None

    rows = []
    keep = np.zeros_like(labels)
    for prop in regionprops(labels):
        reasons = []
        if prop.area < min_area:
            reasons.append("too_small")
        if prop.area > max_area:
            reasons.append("too_large")
        if prop.solidity < cfg.min_solidity:
            reasons.append("low_solidity")
        if prop.eccentricity > cfg.max_eccentricity:
            reasons.append("high_eccentricity")
        if cfg.exclude_border:
            min_row, min_col, max_row, max_col = prop.bbox
            if min_row == 0 or min_col == 0 or max_row == height or max_col == width:
                reasons.append("touches_border")

        if not reasons:
            keep[labels == prop.label] = prop.label

        rows.append(
            {
                "stem": paths.stem,
                "label": prop.label,
                "area_px": prop.area,
                "area_um2": prop.area * area_scale if area_scale else np.nan,
                "solidity": round(float(prop.solidity), 4),
                "eccentricity": round(float(prop.eccentricity), 4),
                "kept": not reasons,
                "reason_dropped": ";".join(reasons),
            }
        )

    paths.qc_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths.qc_table, index=False)
    write_labels(paths.qc_labels, keep)

    n_kept = sum(r["kept"] for r in rows)
    if rows and not n_kept:
        logger.warning(
            "qc %s: kept 0/%d objects -- nothing to measure. Check the bounds in %s "
            "against the measurements in %s.",
            paths.stem, len(rows), "pipeline.yaml", paths.qc_table.name,
        )
    else:
        logger.info("qc %s: kept %d/%d objects", paths.stem, n_kept, len(rows))
    return paths.qc_labels


# --------------------------------------------------------------------------- E

def _footprint(radius: int, metric: str) -> np.ndarray:
    """Disk (euclidean) or full square (chebyshev) of the given radius."""
    if metric == "euclidean":
        return disk(radius).astype(bool)
    return np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)


def erode_labels(labels: np.ndarray, radius: int, metric: str = "euclidean") -> np.ndarray:
    """Erode every label independently so neighbouring cells can never merge.

    This is what CellProfiler's ExpandOrShrinkObjects achieved; doing it on the whole
    label image at once would let adjacent cells bleed into each other.
    """
    footprint = _footprint(radius, metric)
    interior = np.zeros_like(labels)

    for prop in regionprops(labels):
        min_row, min_col, max_row, max_col = prop.bbox
        # Pad so erosion sees background outside the bounding box rather than the crop edge.
        padded = np.pad(prop.image, radius, mode="constant", constant_values=False)
        # Same call skimage.morphology.binary_erosion makes internally (border_value=True);
        # used directly because that wrapper is deprecated from scikit-image 0.26.
        eroded = ndi.binary_erosion(padded, structure=footprint, border_value=True)[
            radius : radius + (max_row - min_row),
            radius : radius + (max_col - min_col),
        ]
        if not eroded.any():
            continue
        window = interior[min_row:max_row, min_col:max_col]
        window[eroded] = prop.label

    return interior


def stage_rings(cfg: Config, paths: Paths, force: bool = False) -> dict[int, tuple[Path, Path]]:
    """Interior = eroded cell; periphery = cell minus interior. Label IDs preserved."""
    outputs: dict[int, tuple[Path, Path]] = {}
    whole = None

    for radius in cfg.erosion_radii:
        interior_path = paths.interior(radius)
        periphery_path = paths.periphery(radius)
        outputs[radius] = (interior_path, periphery_path)

        labels_path = paths.working_labels()
        if (
            not force
            and _up_to_date(interior_path, labels_path)
            and _up_to_date(periphery_path, labels_path)
        ):
            logger.debug("rings r%d up to date, skipping: %s", radius, paths.stem)
            continue

        if whole is None:
            whole = read_labels(paths.working_labels())

        interior = erode_labels(whole, radius, cfg.erosion_metric)

        # A cell that vanishes entirely under erosion has no meaningful interior, so its
        # "periphery" would just be the whole cell. Drop it from both rather than report it.
        survived = np.isin(whole, np.unique(interior[interior > 0]))
        periphery = np.where(survived & (interior == 0), whole, 0)

        lost = len(np.unique(whole)) - len(np.unique(interior))
        if lost:
            logger.info("r%d: %d object(s) vanished under erosion in %s", radius, lost, paths.stem)

        write_labels(interior_path, interior)
        write_labels(periphery_path, periphery)
        logger.info("rings r%d -> %s", radius, periphery_path.name)

    return outputs


# --------------------------------------------------------------------------- F

def _nan_safe_stats(values: np.ndarray) -> dict[str, float]:
    """GP images carry NaN background, so an all-NaN region must yield NaN, never 0."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "gp_mean": np.nan,
            "gp_median": np.nan,
            "gp_std": np.nan,
            "gp_min": np.nan,
            "gp_max": np.nan,
            "n_valid_px": 0,
        }
    return {
        "gp_mean": float(finite.mean()),
        "gp_median": float(np.median(finite)),
        "gp_std": float(finite.std()),
        "gp_min": float(finite.min()),
        "gp_max": float(finite.max()),
        "n_valid_px": int(finite.size),
    }


MEASUREMENT_COLUMNS = [
    "stem", "label", "region", "erosion_radius_px", "erosion_metric",
    "area_px", "area_um2", "n_nan_px",
    "gp_mean", "gp_median", "gp_std", "gp_min", "gp_max", "n_valid_px",
    "cell_area_px", "cell_feret_diameter_um", "cell_eccentricity", "cell_solidity",
    "pixel_size_um",
]


def _measure_region(
    label_image: np.ndarray,
    gp: np.ndarray,
    region_name: str,
    radius: int,
    metric: str,
    stem: str,
    geometry: dict[int, dict],
    pixel_size_um: float | None,
) -> list[dict]:
    area_scale = pixel_size_um**2 if pixel_size_um else None
    rows = []
    for prop in regionprops(label_image):
        window = gp[prop.slice][prop.image]
        row = {
            "stem": stem,
            "label": prop.label,
            "region": region_name,
            "erosion_radius_px": radius,
            "erosion_metric": metric,
            "area_px": prop.area,
            "area_um2": prop.area * area_scale if area_scale else np.nan,
            "n_nan_px": int(prop.area) - int(np.isfinite(window).sum()),
            **_nan_safe_stats(window),
            **geometry.get(prop.label, {}),
            "pixel_size_um": pixel_size_um if pixel_size_um else np.nan,
        }
        rows.append(row)
    return rows


def stage_measure(
    cfg: Config, paths: Paths, pixel_size_um: float | None, force: bool = False
) -> list[Path]:
    """Per-cell GP statistics over periphery, interior and whole-cell masks."""
    written: list[Path] = []
    gp = None
    whole = None
    geometry: dict[int, dict] = {}

    for radius in cfg.erosion_radii:
        out_path = paths.measurements(radius)
        written.append(out_path)
        inputs = (
            paths.working_labels(), paths.interior(radius), paths.periphery(radius),
            paths.gp_image,
        )
        if not force and _up_to_date(out_path, *inputs):
            logger.debug("measurements r%d up to date, skipping: %s", radius, paths.stem)
            continue

        if gp is None:
            if not paths.gp_image.exists():
                raise FileNotFoundError(
                    f"No GP image for {paths.stem} at {paths.gp_image}. "
                    "GP images are generated separately and must exist before measuring."
                )
            gp = tifffile.imread(paths.gp_image).astype(np.float32)
            whole = read_labels(paths.working_labels())
            assert_same_shape(gp, whole, paths.gp_image.name, paths.working_labels().name)

            # Cell-level shape descriptors, attached to every row for downstream filtering.
            length_scale = pixel_size_um if pixel_size_um else np.nan
            for prop in regionprops(whole):
                geometry[prop.label] = {
                    "cell_area_px": prop.area,
                    "cell_feret_diameter_um": prop.feret_diameter_max * length_scale,
                    "cell_eccentricity": round(float(prop.eccentricity), 4),
                    "cell_solidity": round(float(prop.solidity), 4),
                }

        interior = read_labels(paths.interior(radius))
        periphery = read_labels(paths.periphery(radius))

        rows: list[dict] = []
        for name, image in (
            ("periphery", periphery),
            ("interior", interior),
            ("whole", whole),
        ):
            rows.extend(
                _measure_region(
                    image, gp, name, radius, cfg.erosion_metric,
                    paths.stem, geometry, pixel_size_um,
                )
            )

        # An image can legitimately have no cells (a lipid control, or everything filtered
        # out by QC); write a headed but empty CSV rather than failing.
        frame = pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS)
        if not rows:
            logger.warning("no cells to measure in %s (r%d)", paths.stem, radius)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False, na_rep="NaN")
        logger.info("measured r%d -> %s (%d rows)", radius, out_path.name, len(rows))

    return written


def build_summary(cfg: Config) -> Path | None:
    """Concatenate every per-image measurement CSV into one experiment-level table."""
    results_dir = cfg.path(cfg.results_dir)
    parts = sorted(results_dir.glob("*_gp_*.csv"))
    if not parts:
        logger.warning("no measurement CSVs to summarise in %s", results_dir)
        return None

    # Images with no cells contribute a headed but empty CSV; concatenating those trips a
    # pandas FutureWarning and adds nothing, so drop them.
    frames = [f for f in (pd.read_csv(p) for p in parts) if not f.empty]
    if not frames:
        logger.warning("every measurement CSV in %s is empty", results_dir)
        return None
    summary = pd.concat(frames, ignore_index=True)
    out_path = results_dir / f"{cfg.experiment_dir.name}_summary.csv"
    summary.to_csv(out_path, index=False, na_rep="NaN")
    logger.info("summary -> %s (%d rows from %d files)", out_path.name, len(summary), len(parts))
    return out_path


def measure_pixel_size(cfg: Config, raw_path: Path) -> float | None:
    return read_pixel_size_um(raw_path, cfg)
