# Migration status

This page tracks the migration from legacy Sentinel-2 monthly-composite scripts to the `crome` package using AlphaEarth Foundations embeddings.

## Implemented

### Package infrastructure

- `src/crome` package scaffold with `pyproject.toml`, entry points, and `src/` layout
- 18 CLI subcommands under the `crome` entry point with dict-based dispatch
- Shared argument group builders (`cli_args.py`) to eliminate duplication across parsers
- Shared manifest resolution helpers (`manifest.py`)
- Centralized year validation in `constants.py` (supports per-year calls; multi-year workflows validate one year at a time)
- Frozen dataclass configuration objects with `to_dict()` serialization
- `CROME_DATA_ROOT` environment variable for user-scoped output roots
- GitHub Actions CI: Python 3.10--3.12, ruff linting, pytest, package build, Nextflow smoke test
- MkDocs documentation site with Material theme

### Data acquisition

- AlphaEarth download via edown with canonical A00--A63 band validation
- CROME reference discovery from DEFRA DSP search results and landing-page file manifests
- National `.gpkg.zip` download and extraction, with `- Complete` fallback for legacy years
- FlatGeobuf normalization and AOI-specific subset materialization
- Annual footprint dissolution for driving AlphaEarth downloads by CROME coverage area

### Discovery and labeling

- Native AlphaEarth raster discovery from edown manifests, directories, or single files
- Year filtering from manifest metadata, sidecar JSON, or image properties
- Centroid-to-pixel and polygon-to-pixel label transfer modes with configurable overlap policies
- One global label mapping per batch for consistent label IDs across tiles
- AOI-specific reference subsetting to avoid unstable national normalized FGB reads

### Training and prediction

- Block-wise feature/label extraction with per-tile lineage tracking
- Content-based sample caching with SHA-256 keys for idempotent reuse
- Random Forest training with feature-level holdout, pixel-level stratified split, or full-table fallback
- Proportional per-tile subsampling for `--max-train-rows` cap
- Pooled model training from cached per-tile samples via pipeline manifests
- Block-wise 10 m crop prediction with model serialization via joblib
- QC manifests with AOI bounds, raster metadata, label coverage stats, and overlay PNGs

### Orchestration

- `prepare-tile-batch` / `run-tile-plan` / `train-pooled-from-tile-results` orchestration boundary
- `prepare-footprint-tile-batch` for CROME-footprint-driven AlphaEarth acquisition + batch prep
- Nextflow DSL2 wrapper with `jasmin` and `jasmin_special` Slurm profiles
- Slurm job array support, lenient cache mode, and per-process CPU threading control
- Idempotent pipeline reruns via `skip_completed` (checks for existing QC manifests)

### Code quality

- Python `logging` across all pipeline modules (discovery, labeling, training, prediction, orchestration)
- Specific exception handling (CRSError, ValueError, OSError) instead of bare `except Exception`
- Extracted QC manifest writing into shared helper to reduce pipeline function complexity
- 80 tests with synthetic raster fixtures covering config, bands, rasterization, training, prediction, CLI, orchestration, and Nextflow

## Intentionally deferred

- Sampling refactor (stratified spatial sampling beyond the current centroid/polygon modes)
- Live Earth Engine integration tests (requires authenticated EE session in CI)
- GPU-backed CI paths
- ONNX model export (currently joblib-serialized Random Forest)

## Remaining gaps

- Validate edown against real AlphaEarth imagery for one UK AOI/year and confirm output dtype, CRS, manifest layout, and native-image identity in Python 3.13
- Test the batch workflow on real CROME references and real AlphaEarth rasters beyond synthetic fixtures
- Production validation of the Nextflow wrapper on JASMIN at scale (100+ tiles)
- Accuracy benchmarking against legacy Sentinel-2 pipeline outputs

## Legacy scripts

The original Sentinel-2 scripts remain at the repository root for reference:

| Script | Purpose |
|--------|---------|
| `get_monthly_composite.py` | Generate Sentinel-2 monthly composites via Earth Engine |
| `get_satellite_embeddings.py` | Legacy AlphaEarth download wrapper |
| `sample_spectra.py` | Extract training samples from composites |
| `merge_samples.py` | Merge per-tile training samples |
| `train_xgboost.py` | Train XGBoost classifier on merged samples |
| `reclassify_image.py` | Apply trained model to produce crop maps |
| `export_to_asset.py` | Export classification results to Earth Engine assets |

These are not imported by the `crome` package and will be removed once the migration is validated on real data.
