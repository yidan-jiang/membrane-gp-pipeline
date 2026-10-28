"""Path resolution, raw-image loading and metadata extraction."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tifffile

from .config import Config

logger = logging.getLogger(__name__)

# aicsimageio logs a multi-line warning for every reader it tries before succeeding (or
# failing), which buries the pipeline's own output. We report the outcome ourselves.
logging.getLogger("aicsimageio").setLevel(logging.ERROR)


def find_raw_files(cfg: Config) -> list[Path]:
    """Raw acquisitions at the top level of the experiment dir, sorted by name."""
    found: list[Path] = []
    for pattern in cfg.raw_patterns:
        found.extend(cfg.experiment_dir.glob(pattern))
    return sorted(f for f in found if f.is_file())


def stem_of(raw_path: Path) -> str:
    return raw_path.stem


class Paths:
    """Every artifact path for one image, derived from its stem."""

    def __init__(self, cfg: Config, stem: str):
        self.cfg = cfg
        self.stem = stem

    def _resolve(self, pattern: str) -> Path:
        """Expand {stem}; glob patterns must match exactly one existing file.

        Globs matter because some naming conventions embed a per-file timestamp that
        cannot be reconstructed from the stem alone.
        """
        relative = pattern.format(stem=self.stem)
        if "*" not in relative:
            return self.cfg.experiment_dir / relative

        matches = sorted(self.cfg.experiment_dir.glob(relative))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            # Return the unglobbed path so callers report a useful missing-file error.
            return self.cfg.experiment_dir / relative
        raise ValueError(
            f"Pattern {relative!r} matched {len(matches)} files for stem {self.stem!r}: "
            f"{[m.name for m in matches]}. Make the pattern more specific."
        )

    @property
    def projection(self) -> Path:
        return self._resolve(self.cfg.proj_pattern)

    @property
    def cellpose_mask(self) -> Path:
        return self._resolve(self.cfg.mask_pattern)

    @property
    def gp_image(self) -> Path:
        return self._resolve(self.cfg.gp_pattern)

    @property
    def qc_labels(self) -> Path:
        return self.cfg.path(self.cfg.qc_dir) / f"{self.stem}_labels_qc.tif"

    @property
    def qc_table(self) -> Path:
        return self.cfg.path(self.cfg.qc_dir) / f"{self.stem}_qc.csv"

    @property
    def curated_labels(self) -> Path:
        return self.cfg.path(self.cfg.qc_dir) / f"{self.stem}_labels_curated.tif"

    def interior(self, radius: int) -> Path:
        tag = self.cfg.radius_tag(radius)
        return self.cfg.path(self.cfg.interior_dir) / f"{self.stem}_interior_{tag}.tif"

    def periphery(self, radius: int) -> Path:
        tag = self.cfg.radius_tag(radius)
        return self.cfg.path(self.cfg.periphery_dir) / f"{self.stem}_periphery_{tag}.tif"

    def measurements(self, radius: int) -> Path:
        tag = self.cfg.radius_tag(radius)
        return self.cfg.path(self.cfg.results_dir) / f"{self.stem}_gp_{tag}.csv"

    def working_labels(self) -> Path:
        """Curated labels win over auto-QC labels when a curation pass has happened."""
        return self.curated_labels if self.curated_labels.exists() else self.qc_labels


def load_raw_stack(path: Path) -> np.ndarray:
    """Load a raw acquisition as an array whose last two axes are Y, X."""
    if path.suffix.lower() == ".czi":
        from aicsimageio import AICSImage

        return AICSImage(path).get_image_data("CZYX", T=0)
    return tifffile.imread(path)


def average_projection(stack: np.ndarray) -> np.ndarray:
    """Average over every non-spatial axis.

    For the 32-channel lambda LSMs this averages the spectral channels, which is what
    ImageJ's Z Project (Average Intensity) did to those stacks; for Z-stacks it averages Z.
    """
    if stack.ndim < 2:
        raise ValueError(f"Expected an image with at least 2 dimensions, got shape {stack.shape}")
    if stack.ndim == 2:
        return stack
    reduce_axes = tuple(range(stack.ndim - 2))
    return stack.mean(axis=reduce_axes)


def read_pixel_size_um(path: Path, cfg: Config) -> float | None:
    """Physical pixel size in microns, from file metadata, falling back to config."""
    try:
        if path.suffix.lower() == ".czi":
            from aicsimageio import AICSImage

            x = AICSImage(path).physical_pixel_sizes.X
            if x:
                return float(x)
        else:
            with tifffile.TiffFile(path) as tif:
                meta = tif.lsm_metadata
                if meta and meta.get("VoxelSizeX"):
                    # LSM records voxel size in metres.
                    return float(meta["VoxelSizeX"]) * 1e6
    except Exception as exc:
        # Keep this to one line; the underlying errors are very verbose.
        logger.warning(
            "Could not read pixel size from %s (%s). Falling back to configured "
            "pixel_size_um=%s.",
            path.name, type(exc).__name__, cfg.pixel_size_um,
        )

    return cfg.pixel_size_um


def assert_same_shape(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {name_a} is {a.shape} but {name_b} is {b.shape}. "
            "These must come from the same acquisition."
        )


def read_projection(path: Path) -> np.ndarray:
    """Read a 2D reference image, dropping singleton axes.

    Projections written by the older batch_cellpose.py route are saved as (1, Y, X).
    """
    image = tifffile.imread(path)
    return np.squeeze(image)


def read_labels(path: Path) -> np.ndarray:
    return tifffile.imread(path).astype(np.int32)


def write_labels(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = int(labels.max()) if labels.size else 0
    dtype = np.uint16 if peak <= np.iinfo(np.uint16).max else np.int32
    tifffile.imwrite(path, labels.astype(dtype))
