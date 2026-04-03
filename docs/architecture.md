# Architecture

## Data flow

```
                          ┌──────────────────┐
                          │  AlphaEarth (EE)  │
                          │  64-band annual   │
                          │  embeddings 10 m  │
                          └────────┬─────────┘
                                   │ download
                          ┌────────▼─────────┐      ┌───────────────┐
                          │  Discovery        │      │  CROME (DEFRA)│
                          │  Manifest or dir  │      │  Vector hex   │
                          │  scan for .tif    │      │  references   │
                          └────────┬─────────┘      └──────┬────────┘
                                   │                        │
                          ┌────────▼────────────────────────▼───────┐
                          │  Labeling                               │
                          │  Rasterize CROME labels onto each tile  │
                          │  (centroid_to_pixel or polygon_to_pixel)│
                          └────────┬───────────────────────────────┘
                                   │
                          ┌────────▼─────────┐
                          │  Training         │
                          │  Extract table    │──> cached samples
                          │  Train RF model   │
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │  Prediction       │
                          │  Block-wise 10 m  │
                          │  crop map         │
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │  Pooling          │
                          │  Combine cached   │
                          │  tile samples for │
                          │  regional model   │
                          └──────────────────┘
```

## Module map

### Acquisition

| Module | Purpose |
|--------|---------|
| `acquisition/alphaearth.py` | Download AlphaEarth embeddings via edown (Earth Engine) |
| `acquisition/crome.py` | Discover, download, extract, and normalize CROME references from DEFRA DSP |

### Core pipeline

| Module | Purpose |
|--------|---------|
| `discovery.py` | Find AlphaEarth rasters from edown manifests, directories, or single files |
| `labeling.py` | Rasterize CROME vector labels onto the AlphaEarth grid with centroid or polygon-fill modes |
| `training.py` | Build training tables from aligned feature/label rasters, train Random Forest models, manage sample caches |
| `predict.py` | Block-wise crop prediction on feature rasters using trained models |
| `features.py` | Read and validate AlphaEarth raster metadata (bands, CRS, transform) |

### Orchestration

| Module | Purpose |
|--------|---------|
| `pipeline.py` | End-to-end baseline: discover, label, train, predict across all tiles |
| `orchestration.py` | Cluster-parallel tile batch preparation, execution, and pooled model training |
| `workflow.py` | Operator-facing wrappers that combine download + pipeline in one call |

### Shared utilities

| Module | Purpose |
|--------|---------|
| `bands.py` | Canonical A00--A63 band ordering and validation |
| `cli.py` | Top-level CLI router with dict-based dispatch to 18 subcommands |
| `cli_args.py` | Shared argument group builders to avoid duplication across parsers |
| `config.py` | Frozen dataclass request and specification objects |
| `constants.py` | Shared constants (collection ID, year range, resolution) and year validation |
| `manifest.py` | Shared manifest path resolution helpers |
| `paths.py` | Output directory construction and filesystem-safe label sanitization |
| `qc.py` | QC overlay PNG generation and manifest helpers |
| `runtime.py` | PROJ_DATA environment detection for GDAL/OGR |
| `schema.py` | Feature column validation and reference schema contracts |

## Design decisions

### Frozen dataclasses for configuration

All request and result objects (`AlphaEarthDownloadRequest`, `CromeReferenceConfig`, `AlphaEarthTrainingSpec`, `PipelineFeatureResult`, etc.) are frozen dataclasses with `slots=True`. This makes them hashable, immutable after construction, and safe to pass between pipeline stages.

### One global label mapping per batch

When processing multiple AlphaEarth tiles in one batch, a single label-to-integer mapping is built from the full CROME reference before any tile is processed. This ensures label IDs are consistent across tiles and makes pooled training straightforward.

### Content-based sample caching

Each tile's extracted training samples are cached under a SHA-256 key derived from the feature raster, label raster, and label mapping signatures. Repeated runs against the same inputs reuse cached samples. This makes pooled training efficient: `build-training-table-from-cache` combines per-tile caches without re-reading rasters.

### Block-wise raster I/O

Training extraction and prediction both iterate over rasterio block windows rather than loading full rasters into memory. This keeps the working set small even for large AlphaEarth tiles.

### Model serialization with joblib

Trained model bundles (model object, feature names, label mapping) are serialized with `joblib.dump(compress=3)` rather than raw pickle. This provides better compression and is the scikit-learn recommended approach.

### Idempotent reruns

The pipeline's `skip_completed` parameter checks for existing per-tile QC manifests before processing. This allows safe reruns of partially completed batches on HPC without re-processing finished tiles.

### Logging

All pipeline modules use Python's standard `logging` library. Progress messages, skipped tiles, and debug details are logged at `INFO`, `WARNING`, and `DEBUG` levels respectively. Configure logging in your application or set `PYTHONPATH` and use `logging.basicConfig(level=logging.INFO)`.

## Output directory layout

```
$CROME_DATA_ROOT/
  raw/
    alphaearth/         Downloaded AlphaEarth rasters and edown manifests
    crome/              Downloaded CROME GeoPackages, extracts, subsets
  cache/
    samples/            Reusable per-tile training sample caches
  reference/
    crome_hex/tiles/    Per-tile rasterized label rasters and mappings
  training/
    tiles/              Per-tile training tables, models, metrics
    pooled/             Pooled regional/national models
  prediction/
    tiles/              Per-tile predicted crop rasters
  workflow/
    tile_batch_*/       Nextflow batch manifests and per-tile plans
```

Tile and batch directories are further namespaced by label-transfer mode, reference settings, and model configuration, so different runs on the same AlphaEarth tiles do not overwrite each other.
