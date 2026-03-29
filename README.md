# CROME

Crop classification workflows for UK crop mapping, currently centered on Sentinel-2 monthly composite scripts and an active migration toward AlphaEarth Foundations embeddings.

## Current state

- legacy workflow scripts still live at the repository root
- the new package scaffold now lives under `src/crome`
- the migration/package/bootstrap plan is tracked in `MIGRATION_PLAN.md`
- the package now includes a working baseline pipeline from AlphaEarth feature raster to CROME-aligned labels, training table, random-forest model, and predicted crop raster
- AlphaEarth is treated as native image/AOI input data, while CROME remains a vector hexagon reference source for later 10 m label transfer

## Key files

- `MIGRATION_PLAN.md`
- `pyproject.toml`
- `src/crome/acquisition/alphaearth.py`
- `tests/`
- `get_monthly_composite.py`
- `sample_spectra.py`
- `merge_samples.py`
- `train_xgboost.py`
- `reclassify_image.py`

## Next planned direction

- validate `edown` against `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` for one real UK AOI/year
- extend the live AlphaEarth acquisition path to a supported Python 3.10-3.12 runtime with Earth Engine auth
- scale the implemented baseline workflow from synthetic tests to real AOI/year runs
- expand GitHub Actions beyond the initial package/test/docs scaffolding
- publish and refine documentation with GitHub Pages

## Implemented package workflow

1. Download or otherwise provide one AlphaEarth feature raster.
2. Rasterize CROME vector references onto that raster grid.
3. Build a feature/label training table.
4. Train a random-forest baseline model.
5. Predict a 10 m crop map on an AlphaEarth raster.
