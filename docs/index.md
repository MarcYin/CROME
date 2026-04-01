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
crome build-training-table --feature-raster alphaearth.tif --label-raster ./outputs/reference/crome_hex/REF_crome_hex_east-anglia_2024/labels.tif --label-mapping ./outputs/reference/crome_hex/REF_crome_hex_east-anglia_2024/labels.json --output-dir ./outputs/training
crome list-feature-rasters --manifest-path ./outputs/raw/alphaearth/<run>/manifests/run.json --format tsv
crome prepare-tile-batch --manifest-path ./outputs/raw/alphaearth/<run>/manifests/run.json --reference-path ./outputs/raw/crome/CROME_2024_national/extracted/Crop_Map_of_England_CROME_2024.gpkg --year 2024 --output-root ./outputs --aoi-label east-anglia
crome run-tile-plan --tile-plan ./outputs/workflow/<tile-batch-namespace>/BATCH_east-anglia_2024/tiles/<tile-id>.json
crome train-pooled-from-tile-results --batch-manifest ./outputs/workflow/<tile-batch-namespace>/BATCH_east-anglia_2024/batch_manifest.json --tile-result ./work/<tile-a>.tile-result.json --tile-result ./work/<tile-b>.tile-result.json
crome build-training-table-from-cache --cache-manifest ./outputs/training/tiles/<model-namespace>/TRAIN_IMAGE_FULL_2024/dataset/sample_cache_manifest.json --output-dir ./outputs/training/global-2024
crome train-model --training-table ./outputs/training/global-2024/training_table.pkl --output-dir ./outputs/training/global-2024/model --max-train-rows 50000
crome train-pooled-model --pipeline-manifest ./outputs/training/<model-namespace>/TRAIN_RUN_A_2024/pipeline.json --pipeline-manifest ./outputs/training/<model-namespace>/TRAIN_RUN_B_2024/pipeline.json --output-dir ./outputs/training/global-2024 --max-train-rows 50000
crome train-model --training-table ./outputs/training/training_table.pkl --output-dir ./outputs/model --label-mapping ./outputs/reference/crome_hex/REF_crome_hex_east-anglia_2024/labels.json
crome predict-map --feature-raster alphaearth.tif --model-path ./outputs/model/model.pkl --output-raster ./outputs/prediction.tif
crome run-baseline-pipeline --feature-input ./download-output --reference-path crome.geojson --year 2024 --aoi-label east-anglia --output-root ./outputs
crome prepare-tile-batch --manifest-path ./outputs/raw/alphaearth/.../manifests/run.json --reference-path ./outputs/raw/crome/.../extracted/Crop_Map_of_England_CROME_2024.gpkg --year 2024 --output-root ./outputs
crome run-tile-plan --tile-plan ./outputs/workflow/.../tiles/<tile>.json
crome train-pooled-from-tile-results --batch-manifest ./outputs/workflow/.../batch_manifest.json --tile-result ./outputs/workflow/.../<tile>.tile-result.json
```

`download-run-baseline` is the shortest operator path when you want the package to call `edown`, discover native AlphaEarth rasters, rasterize CROME labels, train one baseline model per AlphaEarth image tile, and emit prediction rasters in one pass.
If you do not pass `--reference-path`, the workflow now auto-downloads the national CROME GeoPackage from DEFRA DSP, and the pipeline then clips or reuses a batch-specific subset before using it as the runtime reference source.
`run-baseline-pipeline` accepts either a single feature raster, a directory tree of native AlphaEarth GeoTIFFs, or an `edown` manifest via `--manifest-path`.
`list-feature-rasters` is the discovery boundary for workflow schedulers: it emits stable `feature_id`, `tile_id`, `source_image_id`, and raster paths for one manifest or feature root.
`prepare-tile-batch`, `run-tile-plan`, and `train-pooled-from-tile-results` are the cluster-scheduler boundary: they prepare one batch manifest plus per-tile manifests, run one tile per job, and then train one pooled model from the emitted tile results.
When multiple native rasters are present, the batch pipeline keeps one global CROME label mapping across the run, but it now writes labels, training tables, models, and predictions under tile-specific roots keyed by the AlphaEarth `feature_id` or source image id.
Those tile roots are also namespaced by the label-transfer and model configuration so centroid and polygon runs on the same AlphaEarth tiles do not overwrite each other.
Each batch run still writes a summary `pipeline.json` and `qc.json`, while each tile also gets its own training/model artifacts, reusable `sample_cache_manifest.json`, and `metrics.json` with `evaluation_mode`, `accuracy`, `macro_f1`, and `weighted_f1` when a holdout split is available.
For large pooled or global tables, `train-model` can cap only the training side of the split via `--max-train-rows`, which keeps held-out tile evaluation intact while making very dense polygon-fill tables tractable.
`train-pooled-model` is the operator shortcut for that workflow: it reads one or more prior `pipeline.json` files, resolves their `sample_cache_manifest.json` inputs, builds the pooled dataset, and trains the pooled model in one step.
That cache layout keeps later global model training efficient because `build-training-table-from-cache` can combine per-tile cached samples without rereading the original feature rasters.
For cluster execution, `prepare-tile-batch`, `run-tile-plan`, and `train-pooled-from-tile-results` expose the same work as one shared subset-prep stage, one job per AlphaEarth tile, and one pooled gather/training stage. The intended operational boundary is to run `prepare-tile-batch` once outside Nextflow, then let Nextflow schedule the per-tile and pooled steps.
`download-crome` resolves the DEFRA search results and landing-page `files` list, prefers the national `.gpkg.zip` asset for the requested year, and automatically falls back to `- Complete` nationwide releases for older years such as 2016 and 2017.
The default label mode is `centroid_to_pixel`, so each CROME hexagon contributes supervision at the single AlphaEarth pixel containing its centroid. Pass `--label-mode polygon_to_pixel` if you intentionally want polygon-fill training labels.

## Nextflow on JASMIN

The repo includes [main.nf](/home/users/marcyin/UK_crop_map/nextflow/main.nf) and [nextflow.config](/home/users/marcyin/UK_crop_map/nextflow/nextflow.config). The recommended JASMIN profile is `jasmin`, which maps the multi-core tile and pooled stages to the `standard` partition with QoS `high`. `jasmin_special` is reserved for users who have access to the high-memory `special` partition. GitHub Actions now installs Nextflow and runs a tiny local smoke test against this wrapper so the scheduler entrypoint is validated continuously. The shared-filesystem profile uses Nextflow's `lenient` cache mode, and the tile stage can be grouped into Slurm job arrays with `--tile_array_size`.

```bash
nextflow run nextflow/main.nf \
  -c nextflow/nextflow.config \
  -profile jasmin \
  --batch_manifest /gws/ssde/j25a/nceo_isp/public/CROME/workflow/<tile-batch-namespace>/BATCH_cambridge-norfolk_2024/batch_manifest.json \
  --output_root /gws/ssde/j25a/nceo_isp/public/CROME \
  --slurm_account nceo_isp
```

The wrapper expects a prepared `batch_manifest.json`, then fans out one `run-tile-plan` task per AlphaEarth tile and gathers the emitted tile results into one optional pooled-training step. It reads rasters in place from the shared filesystem instead of staging full GeoTIFFs into each task directory.
The repo also includes a Nextflow+Slurm wrapper for JASMIN in [nextflow.md](/home/users/marcyin/UK_crop_map/docs/nextflow.md).

To keep a user-specific shared data root without hardcoding it into the project, set:

```bash
export CROME_DATA_ROOT=/gws/ssde/j25a/nceo_isp/public/CROME
```

All CLI commands that accept `--output-root` use that environment variable only when the flag is omitted. Other users still fall back to `data/alphaearth`.

For the current 2024 national CROME GeoPackage, county-layer seams can produce positive-area overlaps. The live baseline path now avoids depending on the unstable national normalized FGB for AOI bbox reads by clipping or reusing an AOI-specific subset from the national source first. If you point the pipeline at a raw overlapping vector source, use `--overlap-policy first` for live runs unless you intentionally want overlap errors to stop the workflow.

The legacy wrapper is also available as:

```bash
python get_satellite_embeddings.py --year 2024 --aoi-label east-anglia --bbox -1 51 0 52 --dry-run
```
