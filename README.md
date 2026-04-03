# CROME

UK crop classification at 10 m resolution using [AlphaEarth Foundations](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) satellite embeddings and [CROME](https://www.data.gov.uk/dataset/be5d88c9-acfb-4052-bf6b-ee9a416cfe60/crop-map-of-england-crome) vector references from DEFRA.

The package replaces the legacy Sentinel-2 monthly-composite pipeline with a streamlined workflow: download AlphaEarth 64-band annual embeddings, rasterize CROME hexagon labels onto the 10 m grid, train Random Forest classifiers per tile, and predict crop maps — all from a single CLI.

## Quickstart

```bash
pip install .            # core package
pip install .[ee]        # adds edown for Earth Engine downloads
pip install .[dev]       # adds pytest, ruff, build

# dry-run to inspect what would be downloaded
crome download-run-baseline \
  --year 2024 --aoi-label east-anglia \
  --bbox -1 51 0 52 --output-root ./outputs --dry-run

# full pipeline: download, label, train, predict
crome download-run-baseline \
  --year 2024 --aoi-label east-anglia \
  --bbox -1 51 0 52 --output-root ./outputs
```

If you already have AlphaEarth rasters and a CROME reference on disk:

```bash
crome run-baseline-pipeline \
  --feature-input ./alphaearth-rasters/ \
  --reference-path ./crome_2024.gpkg \
  --year 2024 --output-root ./outputs
```

## Pipeline overview

```
AlphaEarth download ──> Discover rasters ──> Rasterize CROME labels
                                                      │
                              Predict crop map <── Train model
```

Each tile produces: labels, a training table, a Random Forest model, a prediction raster, cached training samples, and QC/provenance manifests. Cached samples can be pooled later for regional or national models without re-reading the original rasters.

## Cluster execution (JASMIN)

For large-area runs, a Nextflow wrapper parallelizes tiles across Slurm:

```bash
# 1. Prepare batch on login node
crome prepare-tile-batch \
  --manifest-path ./outputs/raw/alphaearth/.../manifests/run.json \
  --reference-path ./crome_2024.gpkg \
  --year 2024 --output-root ./outputs

# 2. Fan out with Nextflow
nextflow run nextflow/main.nf \
  -c nextflow/nextflow.config -profile jasmin \
  --batch_manifest ./outputs/workflow/.../batch_manifest.json \
  --output_root ./outputs --slurm_account my-gws
```

See [docs/nextflow.md](docs/nextflow.md) for queue profiles, resource tuning, and special-partition usage.

## Key commands

| Stage | Command | Purpose |
|-------|---------|---------|
| Acquire | `download-alphaearth` | Download AlphaEarth embeddings via edown |
| Acquire | `download-crome` | Download national CROME reference from DEFRA |
| Discover | `list-feature-rasters` | Enumerate AlphaEarth rasters from manifests or directories |
| Label | `rasterize-reference` | Rasterize CROME vectors onto the AlphaEarth grid |
| Train | `build-training-table` | Extract feature/label training tables |
| Train | `train-model` | Train a Random Forest classifier |
| Predict | `predict-map` | Predict a 10 m crop raster |
| Pipeline | `run-baseline-pipeline` | End-to-end: discover, label, train, predict |
| Pipeline | `download-run-baseline` | Download + full pipeline in one command |
| Cluster | `prepare-tile-batch` | Materialize per-tile plans for Nextflow/Slurm |
| Cluster | `run-tile-plan` | Execute one tile plan |
| Pooling | `train-pooled-from-tile-results` | Train a pooled model from tile outputs |

Run `crome <command> --help` for full usage. See the [CLI reference](docs/cli.md) for detailed examples.

## Configuration

Set a shared data root to avoid passing `--output-root` on every call:

```bash
export CROME_DATA_ROOT=/gws/ssde/j25a/nceo_isp/public/CROME
```

The default label transfer mode is `centroid_to_pixel` (one CROME hexagon = one pixel at its centroid). Use `--label-mode polygon_to_pixel` for polygon-fill labels. See [docs/configuration.md](docs/configuration.md) for all options.

## Documentation

Full documentation is available at the [project site](https://marcyin.github.io/CROME/) or locally under `docs/`:

- [Installation](docs/installation.md) -- setup, dependencies, optional extras
- [Architecture](docs/architecture.md) -- module map, data flow, design decisions
- [CLI reference](docs/cli.md) -- all commands with grouped examples
- [Configuration](docs/configuration.md) -- environment variables, label modes, output layout
- [Nextflow on JASMIN](docs/nextflow.md) -- cluster execution with Slurm profiles
- [Migration status](docs/migration.md) -- what is implemented, what is deferred

## Development

```bash
pip install -e .[dev]
pytest                   # 80 tests, ~15 s
ruff check src/ tests/   # linting
```

## Project structure

```
src/crome/
  acquisition/       AlphaEarth and CROME download helpers
  bands.py           Canonical A00-A63 band ordering
  cli.py             Top-level CLI with 18 subcommands
  config.py          Frozen dataclass request/config objects
  constants.py       Shared constants and year validation
  discovery.py       Native raster discovery from manifests or directories
  features.py        Feature-raster metadata reading
  labeling.py        CROME vector rasterization onto the AlphaEarth grid
  orchestration.py   Cluster-parallel tile batch management
  pipeline.py        End-to-end baseline pipeline
  predict.py         Block-wise 10 m crop prediction
  training.py        Training table construction and Random Forest fitting
  workflow.py        Operator-facing combined download + pipeline wrappers
nextflow/            Nextflow wrapper for JASMIN Slurm execution
tests/               80 tests with synthetic raster fixtures
```

## License

MIT
