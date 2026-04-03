# Configuration

## Environment variables

### CROME_DATA_ROOT

Sets the default `--output-root` for all CLI commands.

```bash
export CROME_DATA_ROOT=/gws/ssde/j25a/nceo_isp/public/CROME
```

When unset, the default output root is `data/alphaearth` relative to the working directory. An explicit `--output-root` flag always takes precedence.

## Label transfer modes

The `--label-mode` flag controls how CROME vector hexagons are transferred onto the AlphaEarth 10 m raster grid.

### centroid_to_pixel (default)

Each CROME hexagon contributes **one labeled pixel** at the AlphaEarth pixel containing its centroid. If the centroid falls outside the polygon (e.g. for concave shapes), the representative point is used instead.

- Produces sparse supervision: one label per hexagon
- Fast, no edge effects
- Recommended for most workflows

### polygon_to_pixel

All pixels whose centers fall inside (or touch, if `--all-touched` is set) a CROME polygon are labeled.

- Produces dense supervision: many labels per hexagon
- Can create very large training tables; consider `--max-train-rows` for pooled training
- Edge pixels may have mixed land cover

## Overlap policies

When multiple CROME polygons map to the same pixel, `--overlap-policy` controls the behavior:

| Policy | Behavior |
|--------|----------|
| `first` (default) | Keep the first label assigned to the pixel |
| `last` | Overwrite with the last label |
| `error` | Raise an error if any overlap is detected |

The 2024 national CROME GeoPackage is layered by county and has positive-area overlaps at county seams. Use `first` (the default) for production runs. Use `error` only with clean, non-overlapping reference sources.

## Model training options

### Holdout evaluation

The `--test-size` flag (default `0.2`) controls the evaluation holdout fraction. When multiple tiles are present, the pipeline attempts **feature-level holdout** (holding out entire tiles) before falling back to pixel-level stratified splitting.

Evaluation modes reported in `metrics.json`:

| Mode | Meaning |
|------|---------|
| `feature_holdout` | Entire tiles held out for evaluation |
| `pixel_holdout` | Pixel-level stratified split within the training table |
| `train_only_single_class` | Only one class present; dummy classifier fitted |
| `train_only_no_holdout` | Not enough samples for a valid split |

### Training row cap

`--max-train-rows` caps the number of training rows used to fit the model **after** the holdout split. The held-out evaluation set is unaffected. When multiple tiles are present, the cap is distributed proportionally across tiles to preserve geographic diversity.

### Random Forest parameters

| Flag | Default | Notes |
|------|---------|-------|
| `--n-estimators` | `200` | Number of trees. Use `400` for pooled national models. |
| `--n-jobs` | `-1` | CPU parallelism. On JASMIN, match to allocated Slurm CPUs. |
| `--random-state` | `42` | Reproducibility seed. |

## Output directory layout

All outputs are organized under `$CROME_DATA_ROOT` (or `--output-root`):

```
<output-root>/
  raw/
    alphaearth/
      AEF_<aoi>_annual_embedding_<year>/      edown outputs per AOI/year
    crome/
      CROME_<year>_<variant>/
        archive/          Downloaded .gpkg.zip
        extracted/        Extracted GeoPackage
        normalized/       FlatGeobuf conversion
        footprints/       Dissolved annual footprints
        subsets/          AOI-specific reference subsets
  cache/
    samples/
      crome_hex/year=<year>/label_mode=<mode>/  Per-tile cached training rows
  reference/
    crome_hex/tiles/<namespace>/                Per-tile rasterized labels
  training/
    tiles/<namespace>/TRAIN_<tile>_<year>/
      dataset/            Training table + sample cache manifest
      model/              model.pkl + metrics.json
      qc/                 Per-tile QC manifest + overlay PNG
    pooled/<namespace>/                         Pooled regional models
  prediction/
    tiles/<namespace>/PRED_<tile>_<year>/        Predicted crop rasters
  workflow/
    <namespace>/BATCH_<label>_<year>/
      batch_manifest.json                       Batch-level plan
      tiles/                                    Per-tile JSON plans
```

### Namespacing

Directory paths include content-derived namespace hashes so that different configurations (label mode, overlap policy, model parameters) on the same AlphaEarth tiles do not overwrite each other. For example, a `centroid_to_pixel` run and a `polygon_to_pixel` run on the same tiles produce separate output trees.

## Sample caching

Each tile's extracted training samples are cached under a SHA-256 key derived from the feature raster path, label raster content hash, and label mapping. This means:

- **Repeated runs** against the same inputs reuse cached samples (cache hit)
- **Changed references** or re-rasterized labels produce new cache entries
- **Pooled training** reads cached samples directly without re-opening rasters

Cache manifests (`sample_cache_manifest.json`) track which cache entries contributed to a training table. Use `build-training-table-from-cache` to combine manifests from multiple pipeline runs.

## Logging

All pipeline modules log through Python's standard `logging` library at the `crome.*` namespace. To see progress during pipeline runs:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Or from the command line:

```bash
PYTHONPATH=src python -c "
import logging; logging.basicConfig(level=logging.INFO)
from crome.pipeline import run_baseline_pipeline
# ...
"
```

Log levels used:

| Level | Content |
|-------|---------|
| `INFO` | Discovery counts, per-tile progress, training row counts, completion |
| `WARNING` | Skipped tiles, conditional year notices, reference retry |
| `DEBUG` | CRS transform failures, invalid raster details, cache hits |
