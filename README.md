# CROME

Crop classification workflows for UK crop mapping, currently centered on Sentinel-2 monthly composite scripts and an active migration plan toward AlphaEarth Foundations embeddings.

## Current state

- legacy workflow scripts live at the repository root
- the migration/package/bootstrap plan is tracked in `MIGRATION_PLAN.md`
- GitHub/package/CI/docs scaffolding is planned but not implemented yet

## Key files

- `MIGRATION_PLAN.md`
- `get_monthly_composite.py`
- `sample_spectra.py`
- `merge_samples.py`
- `train_xgboost.py`
- `reclassify_image.py`

## Next planned direction

- validate `edown` against `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- move the workflow into an installable `crome` package
- add GitHub Actions for tests and docs
- publish documentation with GitHub Pages
