# CROME

Crop classification workflows for UK crop mapping, currently centered on Sentinel-2 monthly composite scripts and an active migration toward AlphaEarth Foundations embeddings.

## Current state

- legacy workflow scripts still live at the repository root
- the new package scaffold now lives under `src/crome`
- the migration/package/bootstrap plan is tracked in `MIGRATION_PLAN.md`
- the package now includes a working baseline pipeline from native AlphaEarth raster discovery to CROME-aligned labels, training table aggregation, random-forest model, and predicted crop rasters
- AlphaEarth is treated as native image/AOI input data, while CROME remains a vector hexagon reference source for later 10 m label transfer
- CROME references can now be discovered from DEFRA DSP search pages, resolved to the correct national `.gpkg.zip` asset, downloaded locally, and extracted for the pipeline

## Key files

- `MIGRATION_PLAN.md`
- `pyproject.toml`
- `src/crome/acquisition/alphaearth.py`
- `src/crome/acquisition/crome.py`
- `tests/`
- `get_monthly_composite.py`
- `sample_spectra.py`
- `merge_samples.py`
- `train_xgboost.py`
- `reclassify_image.py`

## Next planned direction

- validate `edown` against `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` for one real UK AOI/year
- run the unified download-to-baseline workflow in an Earth Engine-authenticated environment
- scale the implemented baseline workflow from synthetic tests to real AOI/year runs
- expand GitHub Actions beyond the initial package/test/docs scaffolding
- publish and refine documentation with GitHub Pages

## Implemented package workflow

1. Download AlphaEarth imagery, run the baseline in one command, or point the package at an existing manifest or raw-output directory.
2. Download the national CROME GeoPackage reference from DEFRA DSP, including legacy `- Complete` years when plain-year national datasets do not exist.
3. Discover native AlphaEarth feature rasters and isolate per-feature artifacts.
4. Rasterize CROME vector references onto each feature raster grid using one global crop label mapping for the whole batch.
5. Aggregate the feature/label training table across usable rasters, preserving `feature_id` and `source_image_id` lineage.
6. Train a random-forest baseline model and prefer feature-level holdout when multiple native rasters are available.
7. Predict 10 m crop maps per native raster.

## Reference acquisition

Use the standalone CROME downloader when you want a local GeoPackage copy before running the model:

```bash
crome download-crome --year 2017 --output-root ./outputs --dry-run
crome download-crome --year 2017 --output-root ./outputs
```

`download-run-baseline` can also auto-download the CROME reference if you omit `--reference-path`:

```bash
crome download-run-baseline --year 2017 --aoi-label east-anglia --bbox -1 51 0 52 --output-root ./outputs
```

The downloader resolves DEFRA search results on `environment.data.gov.uk`, follows the dataset landing page, inspects the server-rendered file list, prefers the national `.gpkg.zip` asset, and falls back to `- Complete` variants for older nationwide releases.

You can set a user-specific default artifact root with:

```bash
export CROME_DATA_ROOT=/gws/ssde/j25a/nceo_isp/public/CROME
```

When `CROME_DATA_ROOT` is set, CLI commands use it as the default `--output-root`. This is opt-in and does not change the repo default for other users. An explicit `--output-root` still wins.

The current 2024 national CROME GeoPackage is layered by county and can expose positive-area overlaps at county seams. For that package, use `--overlap-policy first` unless you want the safer default behavior to stop on overlaps.
