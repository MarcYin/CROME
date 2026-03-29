# CROME

`CROME` is the package boundary for the UK crop mapping workflow migration.

## Current scope

- preserve the legacy Sentinel-2 scripts while moving new logic into `src/crome`
- add an AlphaEarth download entry point without importing the legacy scripts
- treat AlphaEarth imagery as native image/AOI inputs rather than legacy Sentinel-2 tile IDs
- treat CROME as vector hexagon references that will later be rasterized onto the AlphaEarth 10 m grid
- keep the first package slice CPU-safe and testable without Earth Engine or GPU access

## First implemented command

The first migration command is:

```bash
crome download-alphaearth --year 2024 --aoi-label east-anglia --bbox -1 51 0 52 --dry-run
```

The legacy wrapper is also available as:

```bash
python get_satellite_embeddings.py --year 2024 --aoi-label east-anglia --bbox -1 51 0 52 --dry-run
```
