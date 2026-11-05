# membrane-gp-pipeline

Automated per-cell measurement of plasma-membrane (PM) lipid order from Laurdan (or other
solvatochromic probe) generalized-polarization (GP) images.

Laurdan labels both the plasma membrane and intracellular membranes. This pipeline
separates a PM-enriched peripheral ring from the cell interior for every segmented cell,
then reports NaN-aware GP statistics for each region. It reproduces a manual
ImageJ → Cellpose → CellProfiler → ImageJ workflow as one resumable command.

**GP images are an input, not an output.** Compute pixel-wise GP images first (e.g. with an
ImageJ/Fiji GP macro or plugin, with background set to NaN); the pipeline consumes them and
never recomputes GP.

## Stages

| Stage | What it does | Output |
|---|---|---|
| **A** projection | Average-intensity projection over all non-spatial axes (Z or spectral channels) of the raw `.lsm`/`.czi`, rounded half-up to match ImageJ *Z Project (Average Intensity)* | reference image for segmentation |
| **B** segmentation | Cellpose on the projection | label image |
| **C** QC | Drops objects by physical area (µm²), solidity, eccentricity and border contact; records the reason for every dropped object | `qc/<stem>_labels_qc.tif`, `qc/<stem>_qc.csv` |
| **D** curation *(optional, interactive)* | Edit labels in napari (remove debris, restore cells) | `qc/<stem>_labels_curated.tif` |
| **E** rings | Erodes each cell **independently** (neighbours can never merge); interior = eroded cell, periphery = cell − interior; label IDs are preserved; cells that vanish under erosion are dropped from both | `masks_interior/`, `mask_PM/` |
| **F** measurement | GP statistics per cell for `periphery`, `interior` and `whole`, for every erosion radius | `results/<stem>_gp_<tag>.csv`, `results/<experiment>_summary.csv` |

Stages skip outputs that already exist, so re-running after adding images only does the new
work. Stages E and F also check timestamps: rings are rebuilt when the labels they came from
are newer, and measurements are redone when the labels, rings or GP image are newer. Curating
an image (stage D) and then running `--stages EF` therefore updates that image's results.
Curated labels, when present, take precedence over the automatic QC labels in stages E and F.
`--force` recomputes everything regardless.

## Installation

Python ≥ 3.9.

```bash
git clone https://github.com/yidan-jiang/membrane-gp-pipeline.git
cd membrane-gp-pipeline
pip install -e .                 # stages A, C, E, F (LSM input)
pip install -e ".[segment]"      # + stage B (Cellpose 2.x/3.x)
pip install -e ".[czi]"          # + Zeiss .czi input
pip install -e ".[curate]"       # + stage D (napari)
```

Stage B uses the Cellpose `models.Cellpose(model_type=...)` API, which Cellpose 4 removed;
the `segment` extra therefore pins `cellpose>=2,<4`. Stage B is practical only on a GPU; if
masks already exist it is skipped.

## Usage

```bash
# full run over one experiment folder
python -m pm_pipeline.run /path/to/experiment

# a subset of stages, a few images, or forced recomputation
python -m pm_pipeline.run /path/to/experiment --stages ABC --limit 3
python -m pm_pipeline.run /path/to/experiment --stages EF --force

# optional manual curation, then re-measure
python -m pm_pipeline.curate /path/to/experiment --stem image_01   # press `s` to save
python -m pm_pipeline.run /path/to/experiment --stages EF
```

`pip install` also provides `pm-pipeline` and `pm-curate` as console commands. Curation
opens a Qt window and needs a display (local session or X11 forwarding).

## Input layout and configuration

Raw acquisitions (`*.lsm`, `*.czi`) at the top level of the experiment folder define the image
*stems*. All parameters live in `<experiment>/pipeline.yaml`; if it does not exist, defaults are
written there on the first run, so each experiment records the parameters it was processed
with. Edit it and re-run with `--force` to apply changes. See
[`examples/pipeline.yaml`](examples/pipeline.yaml) for an annotated template.

The projection, mask and GP image locations are patterns relative to the experiment folder,
with `{stem}` substituted per image:

```yaml
proj_pattern: "ave_projected_images/{stem}_avgProj.tif"
mask_pattern: "ave_projected_images/{stem}_avgProj_cp_masks.tif"
gp_pattern:   "gp_images/{stem}_gp_images.tif"
```

A `*` makes a pattern a glob that must resolve to **exactly one** file (useful when GP file names
carry a timestamp, e.g. `GP_{stem}_*.tif`; the `_` before `*` keeps stem `x_1` from matching
`x_10`). An ambiguous glob raises an error rather than guessing. If your folder does not use the
default layout, write `pipeline.yaml` **before** the first run.

### Key parameters

| Parameter | Default | Notes |
|---|---|---|
| `cellpose_model` | `cyto2` | any model name your Cellpose version accepts |
| `cellpose_diameter` | `100` | **pixels** — scale with magnification (below) |
| `min_area_um2`, `max_area_um2` | `50`, `3000` | **µm²**, converted per image from its own pixel size |
| `min_solidity` | `0.8` | |
| `max_eccentricity` | `0.99` | |
| `exclude_border` | `true` | drop objects touching the image edge |
| `erosion_radii` | `[5, 7, 9]` | **pixels** — scale with magnification (below) |
| `erosion_metric` | `euclidean` | `euclidean` (disk) or `chebyshev` (square; CellProfiler-equivalent) |
| `pixel_size_um` | `null` | fallback only; normally read from the raw file |

**Pixel size** is read from the raw file (LSM `VoxelSizeX`; CZI `physical_pixel_sizes`, which
needs the `czi` extra). If it cannot be read and `pixel_size_um` is unset, QC stops with an error
instead of filtering on a guessed size.

**Area bounds are physical** because zoom can vary between images in one folder.
`cellpose_diameter` and `erosion_radii`, in contrast, are pixel counts and are *not*
auto-scaled. When changing objective or zoom, scale both by `reference_pixel_size / pixel_size`
and round. For example, the defaults were tuned at 0.2076 µm/px (40×); at 0.415 µm/px (20×),
`cellpose_diameter: 50` and `erosion_radii: [3, 4, 5]` keep the same physical cell size and a
~1.0–1.9 µm ring. Check the diameter empirically on a few images.

**Erosion metric.** `euclidean` removes a true R-pixel rim. `chebyshev` erodes with a square and
reproduces CellProfiler `ExpandOrShrinkObjects`: CellProfiler *shrink by N* followed by
`IdentifyTertiaryObjects` with *"shrink smaller object prior to subtraction"* equals
`chebyshev` radius N+1. The two are not interchangeable (a disk-7 ring is about a third
thinner than CellProfiler's "7-pixel" ring), so output names encode the metric (`_r7` vs
`_cheb7`) and can never overwrite each other.

If you change `erosion_radii` after a run, delete the stale `masks_interior/*_<old tag>.tif`,
`mask_PM/*_<old tag>.tif` and `results/*_gp_<old tag>.csv` first: the summary concatenates
every `results/*_gp_*.csv` it finds.

**Other sample types.** Defaults are tuned for keratinocytes. Giant unilamellar vesicle (GUV)
standards are much smaller (~20–50 µm²); process them separately with, e.g.,
`min_area_um2: 10`, `max_area_um2: 500`, `erosion_radii: [3]`.

## Output

`results/<experiment>_summary.csv` has one row per (image, cell, region, erosion radius), with
`region` ∈ {`periphery`, `interior`, `whole`}:

| Column | Meaning |
|---|---|
| `stem`, `label` | image and Cellpose label ID (identical across regions and radii, so measurements pair) |
| `region`, `erosion_radius_px`, `erosion_metric` | which mask |
| `gp_mean`, `gp_median`, `gp_std`, `gp_min`, `gp_max` | GP statistics over the finite pixels of the region |
| `n_valid_px`, `n_nan_px` | finite and NaN pixel counts in the region |
| `area_px`, `area_um2` | region area |
| `cell_area_px`, `cell_feret_diameter_um`, `cell_eccentricity`, `cell_solidity` | whole-cell shape descriptors |
| `pixel_size_um` | pixel size actually used |

`qc/<stem>_qc.csv` lists **every** segmented object with `kept` and `reason_dropped`, so nothing
disappears silently. An image in which QC keeps nothing produces a warning and a headed, empty
CSV, not an error.

### NaN handling

GP images carry NaN background. All statistics are NaN-aware: a region whose pixels are all
NaN yields NaN (never 0), and a partly-NaN region is summarised over its valid pixels, with both
counts reported. NaN-propagating tools report NaN for every region containing a single
background pixel. CellProfiler is one example: in one of our datasets 45% of periphery rings
contain at least one NaN pixel.

## Validation

- **Projection** is bit-identical to ImageJ *Z Project (Average Intensity)* on all images
  tested. The pipeline rounds half-up as ImageJ does; `np.round` uses banker's rounding and
  differs by one grey level on ~1.6% of pixels.
- **Ring geometry**: `chebyshev 8` reproduces CellProfiler's 7-pixel periphery masks at
  IoU 0.99998 (1–7 disagreeing pixels per image out of 54k–100k), with exact per-label
  correspondence (32/32 cells). On an independent 63× CZI dataset it reproduces
  previously generated periphery masks at IoU 1.0000.
- **Measurements** match an independent NumPy computation over CellProfiler's own periphery
  masks exactly (r = 1.000000, max difference 6 × 10⁻⁵, i.e. float32 rounding).
- **Cross-route**: on a 40× spectral (lambda-mode) LSM dataset with 85% NaN pixels, 100% of
  per-cell whole-cell GP medians match those produced by an earlier independent Python
  implementation.
- **Biological control**: on GUV standards the pipeline recovers the expected separation in
  membrane order. DOPC (liquid-disordered) GP = −0.344 ± 0.082 (n = 74); POPC + 50%
  cholesterol (liquid-ordered) GP = +0.229 ± 0.032 (n = 57); difference +0.573 GP (Welch
  t = 55, p = 2 × 10⁻⁷⁶).

Unit tests on synthetic data (no microscope data or Cellpose needed):

```bash
pip install -e ".[test]"
pytest
```

## Citation

If you use this software, please cite it using the metadata in [`CITATION.cff`](CITATION.cff)
(GitHub's *Cite this repository* button). A manuscript describing the method is in preparation.

## Acknowledgements

This pipeline was developed with [Claude](https://claude.ai) (Anthropic) as co-author: Claude
helped write the code, tests and documentation. All methods and results were reviewed and validated
by the author.

## License

MIT — see [`LICENSE`](LICENSE).
