# CLI reference

All commands are available through the `crome` entry point. Run `crome <command> --help` for full argument details.

## Data acquisition

### download-alphaearth

Download AlphaEarth annual embeddings from Earth Engine via edown.

```bash
# Dry-run: inspect the download configuration
crome download-alphaearth \
  --year 2024 --aoi-label east-anglia \
  --bbox -1 51 0 52 --dry-run

# Download to a specific output root
crome download-alphaearth \
  --year 2024 --aoi-label east-anglia \
  --bbox -1 51 0 52 --output-root ./outputs
```

Requires the `ee` extra (`pip install .[ee]`) and an authenticated Earth Engine session.

### download-crome

Download the national CROME reference for one year from DEFRA's Data Services Platform.

```bash
crome download-crome --year 2024 --output-root ./outputs --dry-run
crome download-crome --year 2024 --output-root ./outputs
```

The downloader searches DEFRA DSP, resolves the `.gpkg.zip` archive, extracts the GeoPackage, and normalizes it to FlatGeobuf. For older years (e.g. 2016, 2017), it falls back to `- Complete` archive variants.

### prepare-crome-subset

Materialize or reuse one AOI-specific CROME subset for a set of AlphaEarth tiles.

```bash
crome prepare-crome-subset \
  --reference-path ./crome_2024.gpkg \
  --feature-input ./alphaearth-rasters/ \
  --year 2024 --output-root ./outputs
```

### export-crome-footprint

Export the dissolved spatial footprint of a CROME reference as GeoJSON.

```bash
crome export-crome-footprint \
  --reference-path ./crome_2024.gpkg --year 2024 \
  --output-root ./outputs
```

Useful for driving AlphaEarth downloads to cover the exact CROME extent.

## Discovery

### list-feature-rasters

Discover AlphaEarth rasters from an edown manifest or a directory of GeoTIFFs.

```bash
# From a manifest
crome list-feature-rasters \
  --manifest-path ./outputs/raw/alphaearth/.../manifests/run.json \
  --format tsv

# From a directory
crome list-feature-rasters --feature-input ./alphaearth-rasters/ --format json
```

Output formats: `json` (default), `jsonl`, `tsv`. Each record includes `feature_id`, `tile_id`, `source_image_id`, and `raster_path`.

`discover-feature-rasters` is an alias for this command.

## Labeling

### rasterize-reference

Rasterize CROME vector labels onto one AlphaEarth raster's grid.

```bash
crome rasterize-reference \
  --feature-raster ./alphaearth_tile.tif \
  --reference-path ./crome_2024.gpkg \
  --year 2024 --output-dir ./labels
```

Key options:

- `--label-mode centroid_to_pixel` (default) -- one hexagon = one pixel at centroid
- `--label-mode polygon_to_pixel` -- fill all pixels covered by the polygon
- `--overlap-policy first|last|error` -- how to handle overlapping polygons
- `--all-touched` -- rasterize all touched pixels (polygon mode only)

## Training

### build-training-table

Extract a training table from one aligned feature/label raster pair.

```bash
crome build-training-table \
  --feature-raster ./alphaearth_tile.tif \
  --label-raster ./labels/labels.tif \
  --label-mapping ./labels/labels.json \
  --output-dir ./training/dataset \
  --sample-cache-root ./cache/samples
```

### build-training-table-from-cache

Combine cached per-tile samples into one training table without re-reading rasters.

```bash
crome build-training-table-from-cache \
  --cache-manifest ./tile_a/dataset/sample_cache_manifest.json \
  --cache-manifest ./tile_b/dataset/sample_cache_manifest.json \
  --output-dir ./training/global-2024
```

### train-model

Train a Random Forest classifier from a prepared training table.

```bash
crome train-model \
  --training-table ./training/dataset/training_table.pkl \
  --output-dir ./training/model \
  --n-estimators 200 --n-jobs -1 \
  --max-train-rows 50000
```

### train-pooled-model

Build and train one pooled model from prior batch pipeline manifests.

```bash
crome train-pooled-model \
  --pipeline-manifest ./run_a/pipeline.json \
  --pipeline-manifest ./run_b/pipeline.json \
  --output-dir ./training/pooled \
  --max-train-rows 100000
```

## Prediction

### predict-map

Predict a crop raster from one AlphaEarth feature raster and a trained model.

```bash
crome predict-map \
  --feature-raster ./alphaearth_tile.tif \
  --model-path ./training/model/model.pkl \
  --output-raster ./prediction/crop_map.tif
```

## Pipelines

### run-baseline-pipeline

End-to-end pipeline: discover rasters, rasterize labels, train, and predict.

```bash
# From a directory of rasters
crome run-baseline-pipeline \
  --feature-input ./alphaearth-rasters/ \
  --reference-path ./crome_2024.gpkg \
  --year 2024 --output-root ./outputs

# From an edown manifest
crome run-baseline-pipeline \
  --manifest-path ./manifests/run.json \
  --reference-path ./crome_2024.gpkg \
  --year 2024 --output-root ./outputs
```

Each tile gets its own labels, training table, model, prediction, and QC manifest. Tiles with no CROME coverage are skipped by default (use `--fail-on-empty-labels` to error instead).

### download-run-baseline

Download AlphaEarth imagery and run the full pipeline in one command.

```bash
# Auto-downloads CROME reference when --reference-path is omitted
crome download-run-baseline \
  --year 2024 --aoi-label east-anglia \
  --bbox -1 51 0 52 --output-root ./outputs

# Or provide an existing reference
crome download-run-baseline \
  --year 2024 --aoi-label east-anglia \
  --bbox -1 51 0 52 \
  --reference-path ./crome_2024.gpkg \
  --output-root ./outputs
```

### prepare-footprint-tile-batch

Resolve the full CROME footprint for one year, download intersecting AlphaEarth tiles, and prepare a tile batch for cluster execution.

```bash
crome prepare-footprint-tile-batch \
  --year 2024 --output-root ./outputs --dry-run
```

## Cluster orchestration

### prepare-tile-batch

Prepare per-tile JSON plans and a batch manifest for Nextflow/Slurm execution.

```bash
crome prepare-tile-batch \
  --manifest-path ./manifests/run.json \
  --reference-path ./crome_2024.gpkg \
  --year 2024 --output-root ./outputs \
  --aoi-label cambridge-norfolk
```

### run-tile-plan

Execute one prepared per-tile plan. Typically called by Nextflow, not directly.

```bash
crome run-tile-plan \
  --tile-plan ./workflow/.../tiles/TILE_ID.json \
  --n-jobs 16
```

### train-pooled-from-tile-results

Train a pooled model from tile-result JSONs emitted by `run-tile-plan`.

```bash
crome train-pooled-from-tile-results \
  --batch-manifest ./workflow/.../batch_manifest.json \
  --tile-result ./work/tile_a.tile-result.json \
  --tile-result ./work/tile_b.tile-result.json
```

## Common options

These options appear across most pipeline and orchestration commands:

| Flag | Default | Description |
|------|---------|-------------|
| `--label-column` | `lucode` | CROME column containing crop class labels |
| `--geometry-column` | `geometry` | CROME column containing geometries |
| `--label-mode` | `centroid_to_pixel` | Label transfer method |
| `--overlap-policy` | `first` | How to handle overlapping polygons |
| `--nodata-label` | `-1` | Integer label for unlabeled pixels |
| `--test-size` | `0.2` | Holdout fraction for evaluation |
| `--random-state` | `42` | Random seed |
| `--n-estimators` | `200` | Number of Random Forest trees |
| `--n-jobs` | `-1` | CPU parallelism (-1 = all cores) |
| `--max-train-rows` | None | Cap on training rows after holdout split |
| `--no-predict` | off | Skip prediction, stop after training |
| `--fail-on-empty-labels` | off | Error on tiles with no CROME coverage |
