# CROME

`crome` is a Python package for 10 m crop classification across England using Google's [AlphaEarth Foundations](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) satellite embeddings and DEFRA's [Crop Map of England (CROME)](https://www.data.gov.uk/dataset/be5d88c9-acfb-4052-bf6b-ee9a416cfe60/crop-map-of-england-crome) vector references.

## What it does

1. **Downloads** AlphaEarth 64-band annual embeddings at 10 m resolution (2017--2025) via [edown](https://github.com/MarcYin/edown)
2. **Downloads** the national CROME GeoPackage reference from DEFRA's Data Services Platform
3. **Rasterizes** CROME hexagon labels onto each AlphaEarth tile's native grid
4. **Trains** a tile-local Random Forest classifier with cached, reusable training samples
5. **Predicts** a 10 m crop map per tile
6. **Pools** tile-level samples into regional or national models without re-reading rasters

All steps are accessible as individual CLI commands or as combined pipelines. For cluster execution, a Nextflow wrapper parallelizes tiles across Slurm on JASMIN.

## Getting started

```bash
pip install .[ee]

crome download-run-baseline \
  --year 2024 --aoi-label east-anglia \
  --bbox -1 51 0 52 --output-root ./outputs
```

See the [Installation guide](installation.md) for setup details and optional dependencies.

## Documentation

| Page | Contents |
|------|----------|
| [Installation](installation.md) | Setup, dependencies, optional extras, environment variables |
| [Architecture](architecture.md) | Module map, data flow, design decisions |
| [CLI reference](cli.md) | All 18 commands grouped by workflow stage with examples |
| [Configuration](configuration.md) | Label modes, overlap policies, output directory layout |
| [Nextflow on JASMIN](nextflow.md) | Cluster execution with Slurm queue profiles |
| [Migration status](migration.md) | What is implemented, what is deferred, remaining gaps |

## Supported years

AlphaEarth annual embeddings cover **2017--2024** (stable) and **2025** (rolling by UTM zone). CROME references are available from DEFRA for 2016 onwards. Each download or pipeline run targets one year; multi-year workflows run one year at a time and pool the results.
