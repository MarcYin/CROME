# Migration Status

## Implemented in the first slice

- package scaffold in `src/crome`
- canonical AlphaEarth band validation for `A00` to `A63`
- pure request/config objects for one AlphaEarth tile-year run
- lazy `edown` integration that avoids import-time Earth Engine side effects
- CLI and legacy wrapper for the new AlphaEarth entry point
- CPU-safe tests for config, band ordering, request assembly, and CLI behavior

## Intentionally deferred

- sampling refactor
- training refactor
- inference refactor
- live Earth Engine integration tests
- GPU-backed CI paths

## Next step

- validate `edown` against one real UK AOI/year and confirm output dtype, CRS, and manifest layout
