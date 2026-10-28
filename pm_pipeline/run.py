"""CLI: run the plasma-membrane GP pipeline over one experiment folder.

    python -m pm_pipeline.run <experiment_dir> [--stages ABCEF] [--force]

Stage D (manual curation) is interactive and lives in curate.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import stages
from .config import CONFIG_FILENAME, load_config
from .io_utils import Paths, find_raw_files, stem_of

ALL_STAGES = "ABCEF"
logger = logging.getLogger("pm_pipeline")


def run(experiment_dir: Path, selected: str, force: bool, limit: int | None = None) -> int:
    cfg = load_config(experiment_dir)
    if not cfg.experiment_dir.is_dir():
        logger.error("No such experiment directory: %s", cfg.experiment_dir)
        return 1

    raw_files = find_raw_files(cfg)
    if limit:
        raw_files = raw_files[:limit]
    if not raw_files:
        logger.error(
            "No raw files matching %s in %s", cfg.raw_patterns, cfg.experiment_dir
        )
        return 1

    logger.info("%d raw file(s); stages %s; radii %s", len(raw_files), selected, cfg.erosion_radii)
    cfg.to_yaml(cfg.experiment_dir / CONFIG_FILENAME)

    model = None
    failures = 0

    for index, raw_path in enumerate(raw_files, start=1):
        stem = stem_of(raw_path)
        paths = Paths(cfg, stem)
        logger.info("[%d/%d] %s", index, len(raw_files), stem)

        try:
            pixel_size = stages.measure_pixel_size(cfg, raw_path)

            if "A" in selected:
                stages.stage_projection(cfg, raw_path, paths, force=force)

            if "B" in selected:
                if not paths.cellpose_mask.exists() or force:
                    if model is None:
                        model = stages._build_cellpose_model(cfg)
                stages.stage_segment(cfg, paths, model=model, force=force)

            if "C" in selected:
                stages.stage_qc(cfg, paths, pixel_size, force=force)

            if "E" in selected:
                stages.stage_rings(cfg, paths, force=force)

            if "F" in selected:
                stages.stage_measure(cfg, paths, pixel_size, force=force)

        except Exception as exc:
            failures += 1
            logger.error("failed on %s: %s", stem, exc)
            logger.debug("traceback", exc_info=True)

    if "F" in selected:
        stages.build_summary(cfg)

    if failures:
        logger.warning("finished with %d failure(s) out of %d", failures, len(raw_files))
        return 1

    logger.info("finished cleanly")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument(
        "--stages",
        default=ALL_STAGES,
        help=f"Subset of {ALL_STAGES} to run (A=project B=segment C=qc E=rings F=measure)",
    )
    parser.add_argument("--force", action="store_true", help="Recompute existing outputs")
    parser.add_argument("--limit", type=int, help="Only process the first N raw files")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    selected = args.stages.upper()
    unknown = set(selected) - set(ALL_STAGES)
    if unknown:
        parser.error(f"Unknown stage(s): {sorted(unknown)}. Valid stages are {ALL_STAGES}.")

    return run(args.experiment_dir, selected, args.force, args.limit)


if __name__ == "__main__":
    sys.exit(main())
