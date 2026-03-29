# Migration Status

## Implemented in the first slice

- package scaffold in `src/crome`
- canonical AlphaEarth band validation for `A00` to `A63`
- pure request/config objects for one AlphaEarth AOI/year run
- lazy `edown` integration that avoids import-time Earth Engine side effects
- native AlphaEarth image lineage preserved when the download summary exposes image ids
- CLI and legacy wrapper for the new AlphaEarth entry point
- pure reference-label contracts for CROME hexagon inputs
- rasterization of CROME vectors onto the AlphaEarth 10 m grid
- training-table creation from feature rasters plus label rasters
- baseline random-forest training and prediction on the new package boundary
- CPU-safe tests for config, band ordering, rasterization, training, prediction, and CLI behavior

## Intentionally deferred

- sampling refactor
- live Earth Engine integration tests
- GPU-backed CI paths

## Remaining gap

- validate `edown` against one real UK AOI/year and confirm output dtype, CRS, manifest layout, and native-image identity in a Python 3.10-3.12 runtime
- test the full workflow on real CROME references and real AlphaEarth rasters beyond the synthetic fixture path
