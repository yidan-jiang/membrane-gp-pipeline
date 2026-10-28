"""Automated plasma-membrane GP measurement pipeline.

Consumes GP images produced separately (ImageJ macros), and automates everything
downstream: average projection, Cellpose segmentation, QC filtering, periphery-ring
generation and per-cell GP measurement.
"""

from .config import Config, load_config

__version__ = "1.0.0"

__all__ = ["Config", "load_config", "__version__"]
