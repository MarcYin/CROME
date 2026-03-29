# Migration Status

## Implemented in the first slice

- package scaffold in `src/crome`
- canonical AlphaEarth band validation for `A00` to `A63`
- pure request/config objects for one AlphaEarth AOI/year run
- lazy `edown` integration that avoids import-time Earth Engine side effects
- native AlphaEarth image lineage preserved when the download summary exposes image ids
- CLI and legacy wrapper for the new AlphaEarth entry point
- pure reference-label contracts for CROME hexagon inputs
- CPU-safe tests for config, band ordering, request assembly, and CLI behavior

## Intentionally deferred

- sampling refactor
- training refactor
- inference refactor
- live Earth Engine integration tests
- GPU-backed CI paths

## Next step

- validate `edown` against one real UK AOI/year and confirm output dtype, CRS, manifest layout, and native-image identity
- implement vector-to-raster label transfer from CROME hexagons onto the AlphaEarth 10 m grid
