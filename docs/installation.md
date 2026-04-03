# Installation

## Requirements

- Python 3.10 or later
- GDAL/PROJ libraries (typically bundled with `rasterio` wheels)

## Install from source

```bash
git clone https://github.com/MarcYin/UK_crop_map.git
cd UK_crop_map

# Core package (training, prediction, CLI)
pip install .

# With Earth Engine download support
pip install .[ee]

# Development (tests, linting, build)
pip install -e .[dev]

# Documentation (mkdocs)
pip install .[docs]
```

## Dependencies

### Core

| Package | Version | Purpose |
|---------|---------|---------|
| geopandas | >= 1.0 | Vector geometry operations |
| numpy | >= 1.26 | Numerical computing |
| pandas | >= 2.2 | Tabular data handling |
| rasterio | >= 1.3 | Geospatial raster I/O |
| scikit-learn | >= 1.5 | Random Forest classifiers |
| shapely | >= 2.0 | Geometric operations |

### Optional extras

| Extra | Packages | When needed |
|-------|----------|-------------|
| `ee` | edown >= 0.2.0 | Downloading AlphaEarth imagery from Earth Engine |
| `dev` | pytest >= 9, ruff >= 0.11, build >= 1.2 | Running tests and linting |
| `docs` | mkdocs >= 1.6, mkdocs-material >= 9.6 | Building documentation site |

## Environment setup

### Output root

By default, artifacts are written to `data/alphaearth` relative to your working directory. Set `CROME_DATA_ROOT` to use a shared location:

```bash
export CROME_DATA_ROOT=/gws/ssde/j25a/nceo_isp/public/CROME
```

All CLI commands that accept `--output-root` will use this as the default. An explicit `--output-root` flag always takes precedence.

### Earth Engine authentication

When using `crome download-alphaearth` or `crome download-run-baseline`, you need an authenticated Earth Engine session. Follow the [edown setup instructions](https://github.com/MarcYin/edown) or run:

```bash
earthengine authenticate
```

### PROJ data

The package automatically detects and configures `PROJ_DATA` from `pyproj` on import. If you see CRS-related warnings, ensure `pyproj` is installed and its data directory is accessible.

## Verify installation

```bash
crome --help              # list all subcommands
crome download-alphaearth --help  # check edown integration
python -m pytest tests/   # run test suite (~15 s)
```

## Nextflow (for cluster execution)

Nextflow is **not** bundled with the Python package. Install it separately for JASMIN Slurm execution:

```bash
# On JASMIN, load from the software environment or install manually
curl -s https://get.nextflow.io | bash
./nextflow -version
```

See [Nextflow on JASMIN](nextflow.md) for queue profiles and resource configuration.
