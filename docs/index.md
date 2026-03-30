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
pip install .[ee]
crome download-alphaearth --year 2024 --aoi-label east-anglia --bbox -1 51 0 52 --dry-run
crome download-crome --year 2017 --output-root ./outputs --dry-run
crome download-crome --year 2017 --output-root ./outputs
crome download-run-baseline --year 2024 --aoi-label east-anglia --bbox -1 51 0 52 --reference-path crome.geojson --output-root ./outputs
crome download-run-baseline --year 2017 --aoi-label east-anglia --bbox -1 51 0 52 --output-root ./outputs
crome rasterize-reference --feature-raster alphaearth.tif --reference-path crome.geojson --year 2024 --aoi-label east-anglia --output-root ./outputs
crome build-training-table --feature-raster alphaearth.tif --label-raster ./outputs/reference/crome_hex/REF_crome_hex_east-anglia_2024/labels.tif --output-dir ./outputs/training
crome train-model --training-table ./outputs/training/training_table.pkl --output-dir ./outputs/model --label-mapping ./outputs/reference/crome_hex/REF_crome_hex_east-anglia_2024/labels.json
crome predict-map --feature-raster alphaearth.tif --model-path ./outputs/model/model.pkl --output-raster ./outputs/prediction.tif
crome run-baseline-pipeline --feature-input ./download-output --reference-path crome.geojson --year 2024 --aoi-label east-anglia --output-root ./outputs
```

`download-run-baseline` is the shortest operator path when you want the package to call `edown`, discover native AlphaEarth rasters, rasterize CROME labels, train the baseline model, and emit prediction rasters in one pass.
If you do not pass `--reference-path`, the workflow now auto-downloads the national CROME GeoPackage from DEFRA DSP and uses the extracted `.gpkg` as the reference source.
`run-baseline-pipeline` accepts either a single feature raster, a directory tree of native AlphaEarth GeoTIFFs, or an `edown` manifest via `--manifest-path`.
When multiple native rasters are present, the batch pipeline keeps one global CROME label mapping across the run and prefers feature-level holdout over pixel-level holdout.
`download-crome` resolves the DEFRA search results and landing-page `files` list, prefers the national `.gpkg.zip` asset for the requested year, and automatically falls back to `- Complete` nationwide releases for older years such as 2016 and 2017.

To keep a user-specific shared data root without hardcoding it into the project, set:

```bash
export CROME_DATA_ROOT=/gws/ssde/j25a/nceo_isp/public/CROME
```

All CLI commands that accept `--output-root` use that environment variable only when the flag is omitted. Other users still fall back to `data/alphaearth`.

For the current 2024 national CROME GeoPackage, county-layer seams can produce positive-area overlaps. Use `--overlap-policy first` for live runs with that package unless you intentionally want overlap errors to stop the workflow.

The legacy wrapper is also available as:

```bash
python get_satellite_embeddings.py --year 2024 --aoi-label east-anglia --bbox -1 51 0 52 --dry-run
```
