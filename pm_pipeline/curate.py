"""Stage D: manual label curation in napari (requires the `curate` extra).

    python -m pm_pipeline.curate <experiment_dir> [--stem NAME]

Opens the average projection with the auto-QC labels on top. Paint out debris, restore a
wrongly dropped cell, then press S (or close the window) to save. The result is written to
qc/<stem>_labels_curated.tif, which every later stage prefers over the auto-QC labels, so
re-running `pm_pipeline.run --stages EF` picks up the edits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .io_utils import (
    Paths, find_raw_files, read_labels, read_projection, stem_of, write_labels,
)

logger = logging.getLogger("pm_pipeline.curate")


def curate_one(cfg, stem: str) -> None:
    import napari

    paths = Paths(cfg, stem)
    if not paths.projection.exists():
        raise FileNotFoundError(f"No projection for {stem}: {paths.projection}")

    source = paths.working_labels()
    if not source.exists():
        raise FileNotFoundError(f"No QC labels for {stem}: run stages ABC first ({source})")

    projection = read_projection(paths.projection)
    labels = read_labels(source)

    viewer = napari.Viewer(title=f"curate: {stem}")
    # napari defaults contrast limits to the dtype's full range (e.g. uint8 -> [0, 255]),
    # not the data's actual range, so a dim reference image renders as solid black
    # underneath the label overlay unless we stretch it to what's actually there.
    lo, hi = float(projection.min()), float(projection.max())
    viewer.add_image(
        projection, name="avgProj", colormap="gray",
        contrast_limits=(lo, hi if hi > lo else lo + 1),
    )
    labels_layer = viewer.add_labels(labels, name="cells")

    @viewer.bind_key("s")
    def _save(_viewer):
        write_labels(paths.curated_labels, labels_layer.data)
        logger.info("saved -> %s", paths.curated_labels)

    logger.info("editing %s (loaded from %s)", stem, source.name)
    napari.run()

    write_labels(paths.curated_labels, labels_layer.data)
    logger.info("saved -> %s", paths.curated_labels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--stem", help="Curate one image; omit to step through all of them")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    cfg = load_config(args.experiment_dir)
    stems = [args.stem] if args.stem else [stem_of(p) for p in find_raw_files(cfg)]
    if not stems:
        logger.error("Nothing to curate in %s", cfg.experiment_dir)
        return 1

    for stem in stems:
        try:
            curate_one(cfg, stem)
        except FileNotFoundError as exc:
            logger.warning("skipping %s: %s", stem, exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
