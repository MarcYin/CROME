# CROME

Crop classification workflows for UK crop mapping, currently centered on Sentinel-2 monthly composite scripts and an active migration toward AlphaEarth Foundations embeddings.

## Current state

- legacy workflow scripts still live at the repository root
- the new package scaffold now lives under `src/crome`
- the migration/package/bootstrap plan is tracked in `MIGRATION_PLAN.md`
- the first AlphaEarth acquisition entry point is implemented without touching sampling, training, or inference
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
- implement the vector-to-raster reference path from CROME hexagons onto the AlphaEarth grid
- start moving the legacy workflow into importable `crome` modules
- expand GitHub Actions beyond the initial package/test/docs scaffolding
- publish and refine documentation with GitHub Pages
