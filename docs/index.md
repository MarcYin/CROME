# CROME

`CROME` is the package boundary for the UK crop mapping workflow migration.

## Current scope

- preserve the legacy Sentinel-2 scripts while moving new logic into `src/crome`
- add an AlphaEarth download entry point without importing the legacy scripts
- treat AlphaEarth imagery as native image/AOI inputs rather than legacy Sentinel-2 tile IDs
- treat CROME as vector hexagon references that are rasterized onto the AlphaEarth 10 m grid for model training
- keep the first package slice CPU-safe and testable without Earth Engine or GPU access

## Implemented commands

The current migration commands are:

```bash
crome download-alphaearth --year 2024 --aoi-label east-anglia --bbox -1 51 0 52 --dry-run
crome rasterize-reference --feature-raster alphaearth.tif --reference-path crome.geojson --year 2024 --aoi-label east-anglia --output-root ./outputs
crome build-training-table --feature-raster alphaearth.tif --label-raster ./outputs/reference/crome_hex/REF_crome_hex_east-anglia_2024/labels.tif --output-dir ./outputs/training
crome train-model --training-table ./outputs/training/training_table.pkl --output-dir ./outputs/model --label-mapping ./outputs/reference/crome_hex/REF_crome_hex_east-anglia_2024/labels.json
crome predict-map --feature-raster alphaearth.tif --model-path ./outputs/model/model.pkl --output-raster ./outputs/prediction.tif
crome run-baseline-pipeline --feature-input ./download-output --reference-path crome.geojson --year 2024 --aoi-label east-anglia --output-root ./outputs
```

`run-baseline-pipeline` accepts either a single feature raster, a directory tree of native AlphaEarth GeoTIFFs, or an `edown` manifest via `--manifest-path`.
When multiple native rasters are present, the batch pipeline keeps one global CROME label mapping across the run and prefers feature-level holdout over pixel-level holdout.

The legacy wrapper is also available as:

```bash
python get_satellite_embeddings.py --year 2024 --aoi-label east-anglia --bbox -1 51 0 52 --dry-run
```
