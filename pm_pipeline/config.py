"""Configuration for the plasma-membrane GP pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

CONFIG_FILENAME = "pipeline.yaml"


@dataclass
class Config:
    experiment_dir: Path

    raw_patterns: list[str] = field(default_factory=lambda: ["*.lsm", "*.czi"])

    # Input locations, relative to experiment_dir, with {stem} substituted per image.
    # A '*' makes the pattern a glob, which must resolve to exactly one file -- needed
    # where names carry a per-file timestamp (GP_{stem}_20250122_174653.tif).
    proj_pattern: str = "ave_projected_images/{stem}_avgProj.tif"
    mask_pattern: str = "ave_projected_images/{stem}_avgProj_cp_masks.tif"
    gp_pattern: str = "gp_images/{stem}_gp_images.tif"

    # Output locations (written by this pipeline, so plain directories).
    qc_dir: str = "qc"
    interior_dir: str = "masks_interior"
    periphery_dir: str = "mask_PM"
    results_dir: str = "results"

    cellpose_model: str = "cyto2"
    cellpose_diameter: float = 100.0
    cellpose_gpu: bool = True

    erosion_radii: list[int] = field(default_factory=lambda: [5, 7, 9])

    # "euclidean" erodes with a disk, so radius R removes a true R-pixel rim.
    # "chebyshev" erodes with a square, reproducing CellProfiler's ExpandOrShrinkObjects.
    # CellProfiler "shrink by N" followed by IdentifyTertiaryObjects with
    # "shrink smaller object prior to subtraction" equals chebyshev radius N+1.
    erosion_metric: str = "euclidean"

    # Physical bounds, so they stay comparable across objectives and zoom levels.
    # Defaults bracket plausible keratinocyte sizes (~8-60 um across).
    min_area_um2: float = 50.0
    max_area_um2: float = 3000.0
    min_solidity: float = 0.8
    max_eccentricity: float = 0.99
    exclude_border: bool = True

    # Fallback only; the real value is read from the raw file's metadata.
    pixel_size_um: float | None = None

    def __post_init__(self):
        self.experiment_dir = Path(self.experiment_dir)
        self.erosion_radii = sorted({int(r) for r in self.erosion_radii})
        if self.erosion_metric not in ("euclidean", "chebyshev"):
            raise ValueError(
                f"erosion_metric must be 'euclidean' or 'chebyshev', got {self.erosion_metric!r}"
            )

    def radius_tag(self, radius: int) -> str:
        """Filename tag; the metric is encoded so the two never overwrite each other."""
        return f"r{radius}" if self.erosion_metric == "euclidean" else f"cheb{radius}"

    def area_bounds_px(self, pixel_size_um: float | None) -> tuple[float, float]:
        """QC area bounds converted to pixels for the image being filtered.

        Pixel area is not comparable between acquisitions -- zoom varies within a single
        experiment folder -- so the bounds are physical and converted per image.
        """
        if not pixel_size_um:
            raise ValueError(
                "Cannot apply the µm² QC bounds without a pixel size. The pixel size is "
                "normally read from the raw file, but that failed (for .czi this usually "
                "means aicspylibczi is not installed). Set `pixel_size_um` in pipeline.yaml "
                "to the correct value for this dataset and re-run. Guessing a threshold "
                "here would silently filter cells on a fabricated size."
            )
        scale = pixel_size_um**2
        return self.min_area_um2 / scale, self.max_area_um2 / scale

    def path(self, subdir: str) -> Path:
        return self.experiment_dir / subdir

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        data = asdict(self)
        data["experiment_dir"] = str(self.experiment_dir)
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


def load_config(experiment_dir: Path) -> Config:
    """Load pipeline.yaml from the experiment dir, or build defaults."""
    experiment_dir = Path(experiment_dir)
    candidate = experiment_dir / CONFIG_FILENAME
    if candidate.exists():
        cfg = Config.from_yaml(candidate)
        cfg.experiment_dir = experiment_dir
        return cfg
    return Config(experiment_dir=experiment_dir)
