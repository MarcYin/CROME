# CROME

Crop classification workflows for UK crop mapping, currently centered on Sentinel-2 monthly composite scripts and an active migration toward AlphaEarth Foundations embeddings.

## Current state

- legacy workflow scripts still live at the repository root
- the new package scaffold now lives under `src/crome`
- the migration/package/bootstrap plan is tracked in `MIGRATION_PLAN.md`
- the package now includes a working baseline pipeline from native AlphaEarth raster discovery to tile-level CROME-aligned labels, tile-level training/model artifacts, and predicted crop rasters
- the pipeline now writes run-level QC/provenance manifests and reusable sampled-row caches so later global model training can reuse AOI samples efficiently
- AlphaEarth is treated as native image/AOI input data, while CROME remains a vector hexagon reference source for later 10 m label transfer
- CROME references can now be discovered from DEFRA DSP search pages, resolved to the correct national `.gpkg.zip` asset, downloaded locally, and normalized into a FlatGeobuf reference for the pipeline

## Key files

- `MIGRATION_PLAN.md`
- `pyproject.toml`
- `src/crome/acquisition/alphaearth.py`
- `src/crome/acquisition/crome.py`
- `tests/`
- `get_monthly_composite.py`
- `sample_spectra.py`
- `merge_samples.py`
- `train_xgboost.py`
- `reclassify_image.py`

## Next planned direction

- validate `edown` against `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` for one real UK AOI/year
- run the unified download-to-baseline workflow in an Earth Engine-authenticated environment
- scale the implemented baseline workflow from synthetic tests to real AOI/year runs
- expand GitHub Actions beyond the initial package/test/docs scaffolding
- publish and refine documentation with GitHub Pages

## Implemented package workflow

1. Download AlphaEarth imagery, run the baseline in one command, or point the package at an existing manifest or raw-output directory.
2. Download the national CROME GeoPackage reference from DEFRA DSP, including legacy `- Complete` years when plain-year national datasets do not exist.
3. Discover native AlphaEarth feature rasters and isolate per-feature artifacts.
4. Rasterize CROME vector references onto each AlphaEarth image tile using one global crop label mapping for the whole batch.
5. By default, transfer each CROME hexagon to the single AlphaEarth pixel containing its centroid, instead of filling every covered pixel inside the polygon.
6. Cache immutable per-feature sampled rows so repeated AOI runs and later global training can reuse extracted training data instead of rescanning rasters.
7. Train one tile-local training table and baseline model per AlphaEarth image tile, preserving `feature_id` and `source_image_id` lineage in cached samples for later global training.
8. Write run-level QC/provenance with requested AOI bounds, actual raster bounds, AOI window, label coverage stats, and reference metadata.
9. Predict a 10 m crop map per AlphaEarth image tile using that tile's local model.
10. Rebuild larger regional or global training tables later by combining the cached per-tile sample manifests instead of rescanning rasters.

## Reference acquisition

Use the standalone CROME downloader when you want a local reference copy before running the model:

```bash
crome download-crome --year 2017 --output-root ./outputs --dry-run
crome download-crome --year 2017 --output-root ./outputs
```

`download-run-baseline` can also auto-download the CROME reference if you omit `--reference-path`:

```bash
crome download-run-baseline --year 2017 --aoi-label east-anglia --bbox -1 51 0 52 --output-root ./outputs
```

The downloader resolves DEFRA search results on `environment.data.gov.uk`, follows the dataset landing page, inspects the server-rendered file list, prefers the national `.gpkg.zip` asset, and falls back to `- Complete` variants for older nationwide releases. During live baseline runs, the package now treats the extracted national GeoPackage as the source of truth and automatically materializes or reuses a batch-specific subset under `raw/crome/.../subsets/` before rasterization.

Each baseline run now writes:
- `pipeline.json` for the high-level batch summary
- `qc.json` for AOI-vs-raster coverage, label density, and reference provenance
- per-tile label artifacts under `reference/.../tiles/<label-namespace>/`
- per-tile training/model artifacts under `training/tiles/<model-namespace>/`
- per-tile predictions under `prediction/tiles/<model-namespace>/`
- per-tile `sample_cache_manifest.json` files for reusable sampled-row shards
- `metrics.json` files that now include `evaluation_mode`, `accuracy`, `macro_f1`, and `weighted_f1` when a holdout split is available

Those cache manifests can be combined later for efficient global model training without resampling the original rasters:

```bash
crome build-training-table-from-cache \
  --cache-manifest ./outputs/training/tiles/<model-namespace>/TRAIN_IMAGE_FULL_2024/dataset/sample_cache_manifest.json \
  --cache-manifest ./outputs/training/tiles/<model-namespace>/TRAIN_IMAGE_LEFT_2024/dataset/sample_cache_manifest.json \
  --output-dir ./outputs/training/global-2024
crome list-feature-rasters --manifest-path ./outputs/raw/alphaearth/<run>/manifests/run.json --format tsv
crome train-pooled-model \
  --pipeline-manifest ./outputs/training/<model-namespace>/TRAIN_RUN_A_2024/pipeline.json \
  --pipeline-manifest ./outputs/training/<model-namespace>/TRAIN_RUN_B_2024/pipeline.json \
  --output-dir ./outputs/training/global-2024 \
  --max-train-rows 50000
```

The tile namespaces are derived from the label-transfer mode, reference settings, and model/training configuration so repeated runs against the same AlphaEarth image tiles do not overwrite each other.
For very large pooled tables, `crome train-model` also accepts `--max-train-rows` so global fits can cap the training subset after the holdout split while still evaluating on the full held-out tiles.
`crome train-pooled-model` wraps the pooled path end to end by reading prior `pipeline.json` files, gathering their cached sample manifests, building the combined table, and training the pooled model in one command.
`crome list-feature-rasters` is the stable discovery boundary for cluster schedulers such as Nextflow: it emits `feature_id`, `tile_id`, `source_image_id`, and raster paths for one manifest or feature root.
`crome prepare-tile-batch`, `crome run-tile-plan`, and `crome train-pooled-from-tile-results` are the stable orchestration boundary for Slurm schedulers: they prepare one tile batch from a manifest or raster root, run one tile per job without output collisions, and optionally assemble one pooled model after the tile jobs finish.

## Nextflow on JASMIN

The repo now includes a Slurm-oriented Nextflow wrapper under [main.nf](/home/users/marcyin/UK_crop_map/workflows/nextflow/main.nf) with queue profiles in [nextflow.config](/home/users/marcyin/UK_crop_map/workflows/nextflow/nextflow.config). The workflow assumes you prepare a batch once, then let Nextflow fan out one `crome run-tile-plan` task per AlphaEarth image tile and optionally run one pooled `crome train-pooled-from-tile-results` step after the tile jobs finish.

Typical JASMIN usage:

```bash
crome prepare-tile-batch \
  --manifest-path /gws/ssde/j25a/nceo_isp/public/CROME/raw/alphaearth/AEF_cambridge-fringe-smoke_annual_embedding_2024/manifests/run-20260330T213729Z.json \
  --reference-path /gws/ssde/j25a/nceo_isp/public/CROME/raw/crome/CROME_2024_national/extracted/Crop_Map_of_England_CROME_2024.gpkg \
  --output-root /gws/ssde/j25a/nceo_isp/public/CROME \
  --year 2024 \
  --aoi-label cambridge-norfolk-2024

nextflow run workflows/nextflow/main.nf \
  -c workflows/nextflow/nextflow.config \
  -profile jasmin \
  --batch_manifest /gws/ssde/j25a/nceo_isp/public/CROME/workflow/<tile-batch-namespace>/BATCH_cambridge-norfolk-2024_2024/batch_manifest.json \
  --tile_cpus 16 \
  --tile_memory '128 GB' \
  --pooled_cpus 48 \
  --pooled_memory '512 GB'
```

The current JASMIN queue guidance supports this split well: `debug` is the right smoke-test profile, `standard` plus QoS `high` is the default for multi-core per-tile jobs because it allows up to `96` CPUs per job and `1000 GB` memory, and the access-controlled `special` partition exposes `6 TB` nodes with QoS `special` up to `96` CPUs and `3000 GB` memory for uncapped pooled fits. Source: https://help.jasmin.ac.uk/docs/batch-computing/slurm-queues/

Use `-profile jasmin_special` only if your account has access to the JASMIN `special` partition. The workflow keeps the scientific contract unchanged:

- one task per prepared AlphaEarth image tile
- tile-local labels, models, and predictions written by the existing `crome` CLI without batch-manifest collisions
- pooled training assembled only from emitted per-tile `pipeline.json` manifests
- shared sample-cache reuse preserved under the existing cache namespace layout

You can set a user-specific default artifact root with:

```bash
export CROME_DATA_ROOT=/gws/ssde/j25a/nceo_isp/public/CROME
```

When `CROME_DATA_ROOT` is set, CLI commands use it as the default `--output-root`. This is opt-in and does not change the repo default for other users. An explicit `--output-root` still wins.

The default label mode is `centroid_to_pixel`, which treats each CROME hexagon as one supervision point at the pixel containing its centroid. If you want the older polygon-fill behavior, pass `--label-mode polygon_to_pixel`.

The current 2024 national CROME GeoPackage is layered by county and can expose positive-area overlaps at county seams. The live baseline path now clips or reuses an AOI-specific subset from the national source before label rasterization, which avoids relying on the unstable national normalized FGB for AOI bbox reads. If you point the pipeline at a raw overlapping vector source, `--overlap-policy first` is still the pragmatic live-run override.
