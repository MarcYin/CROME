# Migration Plan: Replace S2 Monthly Composite Inputs with Google Satellite Embeddings

## Goal

Replace the current classification inputs based on Sentinel-2 monthly composites with annual AlphaEarth Foundations embeddings from `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, while preserving the existing sample extraction, model training, and raster reclassification workflow as much as possible.

Reference-label correction:

- AlphaEarth should be used as the feature source on its native image grid.
- CROME should be treated as a vector hexagon reference dataset rather than per-pixel truth.
- The downstream training target is therefore a 10 m raster label surface derived by transferring or rasterizing those CROME polygons onto the AlphaEarth grid.

Repository delivery goal:

- turn this directory into a proper Python package and GitHub repository at `git@github.com:MarcYin/CROME.git`
- add automated testing/build checks in GitHub Actions
- publish project documentation to GitHub Pages

## Scope Clarification

- The official Earth Engine catalog currently lists `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` availability as `2017-01-01T00:00:00Z` through `2025-01-01T00:00:00Z`, which corresponds to annual layers for **2017 through 2024**.
- The same catalog page also states that, as of **2026-01-29**, `2025` embeddings are being added on a rolling basis by UTM zone.
- The migration should therefore target **2017-2024** as the documented stable baseline and treat `2025` as conditional per-tile coverage that must be checked at runtime.
- The requested `A00` to `A63` band set is consistent with the official dataset description.

## Current Pipeline Scan

### 1. Data creation

- `get_monthly_composite.py`
  - `generate_monthly_composites(s2_tile, year, parent_dir='./')`
  - Builds a 12-month Sentinel-2 cloud-masked median composite per MGRS tile.
  - Source collections:
    - `COPERNICUS/S2_SR_HARMONIZED`
    - `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`
  - Writes one zarr per tile/year:
    - `S2_{tile}_monthly_comosite_{year}.zarr`
  - Output structure is effectively:
    - dimensions: `time`, `lat`, `lon`
    - variables: `B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B11`, `B12`

- `export_to_asset.py`
  - Alternate export path that writes monthly composites to Earth Engine assets instead of zarr.
  - Not part of the main local training path, but it duplicates the same monthly-composite assumption.

### 2. Sample extraction

- `sample_spectra.py`
  - `extract_crop_samples(parent_dir, zarr_path, crop_map_file)`
  - Opens each tile zarr, computes pixel indices for crop-map features, and extracts values for the 10 Sentinel-2 bands across 12 months.
  - Saves:
    - `{tile}_crop_samples.nc`
    - `{tile}_crop_samples.fgb`
  - Output structure:
    - `band_values` dims: `band`, `month`, `sample`
    - `lucode` dims: `sample`

### 3. Sample merge

- `merge_samples.py`
  - Merges per-tile sample files into a year-level training file.
  - Removes samples where all extracted values are zero.
  - Writes:
    - `S2_{year}_crop_samples.nc`
    - `S2_{year}_crop_samples.fgb`

### 4. Training

- `train_ml.py`
  - `load_crop_samples(year)`
  - Multi-class training path.
  - Flattens `10 bands x 12 months = 120` features into columns named `B2_1 ... B12_12`.

- `train_xgboost.py`
  - `load_crop_samples(year)`
  - Binary training path for winter wheat (`AC66` vs other).
  - Also assumes the `10 x 12` feature layout and hard-coded Sentinel-2 band names.

- `train_ml_per_tile.py`
  - `load_data(tile, year, parent_dir)`
  - Per-tile training path.
  - Flattens all features but still assumes the existing sampled file structure came from monthly composites.

### 5. Inference / reclassification

- `reclassify_image.py`
  - `predict_zarr(zarr_path, model, bands, chunk_size)`
  - Opens a tile zarr, loads the selected bands, reshapes them into `band-month` feature columns, and predicts class labels per pixel.

- `reclassify_image_tile.py`
  - Same pattern as above, but using tile-specific models and a slightly different flattened column naming convention.

### 6. Supporting utility

- `gee_downloader.py`
  - Contains a generic chunked Earth Engine image download path via `download_google_earth_engine_images(...)`.
  - This is the closest existing utility to an embedding downloader, but it is not currently integrated into the crop-map pipeline.
  - Fastest read suggests this file may need validation before reuse because it references symbols such as `logging`, `os`, `retry`, and `get_logger` that are not visible in the inspected section headers/imports.

## Concrete Data Flow Today

1. `get_monthly_composite.py` creates per-tile, per-year zarrs from Sentinel-2 monthly composites.
2. `sample_spectra.py` samples those zarrs using the crop map for the matching year.
3. `merge_samples.py` combines tile samples into one annual training dataset.
4. `train_ml.py` or `train_xgboost.py` train models from the merged dataset.
5. `reclassify_image.py` or `reclassify_image_tile.py` apply the trained model back to each tile zarr.

## Target Package and Repository Layout

Recommended repository shape after the first package refactor:

```text
CROME/
  pyproject.toml
  README.md
  LICENSE
  .gitignore
  src/
    crome/
      __init__.py
      cli.py
      config.py
      download.py
      sampling.py
      training.py
      inference.py
  scripts/
    get_monthly_composite.py
    train_xgboost.py
    ...
  tests/
    test_schema.py
    test_sampling.py
    test_cli.py
  docs/
    index.md
    migration.md
    api.md
  .github/
    workflows/
      ci.yml
      docs.yml
      release.yml
```

Packaging direction:

- use a `src/` layout so imports are isolated from the repository root
- keep current ad hoc scripts temporarily under `scripts/` while logic is moved into package modules
- expose the main workflows through a package CLI such as `crome download`, `crome sample`, `crome train`, and `crome predict`
- store configuration in package-level config helpers or YAML/TOML config files rather than hard-coded years and paths in scripts

## Migration Strategy

### Phase 0: Reframe the repo as a Python package before the data-source swap

Recommended approach:

- Convert the current script collection into an installable package so that acquisition, sampling, training, and inference code can be imported, tested, documented, and versioned consistently.
- Keep thin script entry points during the migration, but move real logic into package modules first.

Recommended package rules:

- use a `src/` layout to avoid import-path ambiguity during tests
- make Earth Engine and GPU-heavy dependencies optional extras instead of mandatory base dependencies
- split install targets into:
  - runtime
  - `dev`
  - `docs`
  - `ee`
  - `gpu`

Recommended temporary compatibility wrappers:

- `get_satellite_embeddings.py`
- `sample_spectra.py`
- `merge_samples.py`
- `train_xgboost.py`
- `reclassify_image.py`

Each wrapper should:

- parse arguments
- call package code
- remain backward-compatible only until the migration is stable

### Phase 1: Introduce a new embedding acquisition path

Recommended approach:

- Add a new acquisition script instead of mutating `get_monthly_composite.py` in place.
- Keep the existing Sentinel-2 path available until the embedding path has been validated end-to-end.

Planned work:

1. Add a new script, for example `get_satellite_embeddings.py`.
2. Read annual images from:
   - `ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")`
3. Filter per year using:
   - `filterDate(f"{year}-01-01", f"{year+1}-01-01")`
4. Select all 64 embedding bands:
   - `A00` through `A63`
5. Export/download one local raster or zarr per spatial unit and year.

Open design choice:

- Do not drive AlphaEarth acquisition from legacy Sentinel-2 MGRS tile IDs.
- Use the native AlphaEarth images discovered for an AOI/year and keep their original image/tile identity in the download manifest.
- Local run labels may still exist for naming outputs, but they are not dataset-native tile identifiers.

Recommendation:

- Treat AlphaEarth acquisition as `AOI + year -> one or more native AlphaEarth images`, not `S2 tile + year -> one raster`.

### Phase 2: Change the sampled feature schema

Current schema:

- `band_values(band, month, sample)`

Target schema:

- Prefer a simpler embedding schema such as `band_values(feature, sample)` where `feature` is `A00` to `A63`.
- Alternatively keep `band` as the dimension name, but remove the `month` dimension entirely.
- Keep the reference-label path separate from the feature path:
  - AlphaEarth provides per-pixel features
  - CROME provides polygon or hexagon reference labels that must be transferred to the 10 m feature grid

Planned edits:

- `sample_spectra.py`
  - Remove hard-coded Sentinel-2 band list.
  - Read all embedding variables or an explicit `A00` to `A63` list.
  - Save sampled features without a month dimension.

- `merge_samples.py`
  - Update concatenation logic to handle `feature x sample` instead of `band x month x sample`.
  - Keep the all-zero or all-missing sample filter, but adjust the axis logic.

### Phase 3: Generalize training loaders

Current assumption:

- Training scripts hard-code 120 feature columns from `10 bands x 12 months`.

Target assumption:

- Training scripts should derive feature names from the sampled dataset itself.

Planned edits:

- `train_ml.py`
  - Rewrite `load_crop_samples(year)` to inspect the sampled file and build feature names dynamically.

- `train_xgboost.py`
  - Same change as above for the binary winter wheat path.

- `train_ml_per_tile.py`
  - Change `load_data(...)` so it loads the new feature layout without assuming monthly composites.

Recommended feature naming:

- `A00 ... A63`

This keeps model inputs stable and makes train/inference feature alignment easier to verify.

### Phase 4: Generalize inference

Current assumption:

- `reclassify_image.py` and `reclassify_image_tile.py` build feature frames from hard-coded band names and 12 monthly steps.

Target assumption:

- Inference should read a known ordered feature list and flatten only those features.

Planned edits:

- `reclassify_image.py`
  - Update `predict_zarr(...)` to work with `A00 ... A63`.
  - Remove the month-based column construction.

- `reclassify_image_tile.py`
  - Make the same change.

Recommended guardrail:

- Save the exact feature order used for training alongside each model, then require inference to load and apply that same order.

## Exact Files and Functions Most Likely to Change

Highest priority:

- `pyproject.toml`
  - new package metadata, optional extras, and console-script entry points
- `.gitignore`
  - repo hygiene for data outputs, caches, docs builds, and local environments
- `.github/workflows/ci.yml`
  - lint, tests, and package build
- `.github/workflows/docs.yml`
  - docs build and GitHub Pages deployment
- `docs/`
  - package, CLI, and migration documentation
- `tests/`
  - CPU-safe baseline tests plus skipped EE/GPU integration paths
- `src/crome/`
  - new importable package modules extracted from the current scripts
- `get_monthly_composite.py`
  - likely replaced or complemented by a new embedding acquisition script
- `sample_spectra.py`
  - `extract_crop_samples(...)`
- `merge_samples.py`
  - top-level merge logic
- `train_ml.py`
  - `load_crop_samples(year)`
- `train_xgboost.py`
  - `load_crop_samples(year)`
- `train_ml_per_tile.py`
  - `load_data(tile, year, parent_dir)`
- `reclassify_image.py`
  - `predict_zarr(zarr_path, model, bands, chunk_size)`
- `reclassify_image_tile.py`
  - `predict_zarr(zarr_path, model, bands, chunk_size)`

Secondary:

- `export_to_asset.py`
  - only if the Earth Engine asset export path still matters
- `get_monthly_composite.sh`
  - likely superseded by a new batch script for embeddings
- `reclassify_image.sh`
  - may need year/model-path updates only

Optional reuse candidate:

- `gee_downloader.py`
  - reuse only after validating that it runs cleanly and supports the needed Earth Engine image/bounds workflow

## Risks and Unknowns

### 1. 2025 data availability

- Official docs currently expose annual layers through 2024, not 2025.
- This is the main scope mismatch with the requested year range.

Fastest next check:

- Query the Earth Engine collection directly for the distinct years actually present before implementation begins.

### 2. Spatial indexing mismatch

- Current code assumes Sentinel-2 MGRS tiles, but the AlphaEarth path should not.
- The embedding dataset should be handled through native AlphaEarth images discovered for the target AOI/year.

Fastest next check:

- Inspect one embedding image over a known UK AOI and confirm the manifest preserves the native AlphaEarth image identities.

### 3. Feature datatype mismatch

- Current pipeline frequently coerces imagery to `uint16` and treats zeros as invalid.
- Embeddings are semantic float vectors, so `uint16` casting is likely wrong.

Fastest next check:

- Inspect one downloaded embedding image locally and confirm dtype, nodata behavior, and whether zero is a valid feature value.

### 4. Downloader choice is now partially validated

- `edown` is published on PyPI as version `0.1.0`, released on **March 28, 2026**.
- The verified PyPI description and project docs describe it as a generic Google Earth Engine downloader that can:
  - search an `ImageCollection` by date range and AOI
  - download native-grid GeoTIFFs
  - optionally build alignment-aware Zarr stacks
- The documented CLI and Python API accept a generic `collection_id` plus an explicit band list, so `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` with `A00` to `A63` appears compatible in principle.
- The repo already contains `gee_downloader.py`, but it does not appear production-ready without verification and is no longer the only plausible download path.

Fastest next check:

- Validate `edown` against one AlphaEarth AOI/year by:
  - searching the collection for a known AOI/year
  - downloading one image with bands `A00` to `A63`
  - confirming the output dtype, CRS, and band naming match downstream expectations

### 5. Sample geometry assumptions

- `sample_spectra.py` extracts `geometry.x` and `geometry.y`, which implies point-like geometries.
- If the crop map contains polygons in some years, this logic may depend on prior preprocessing.
- The corrected CROME target is polygon or hexagon vector data, not native per-pixel labels.
- The migration therefore needs an explicit vector-to-raster label-transfer step before model training can be trusted.

Fastest next check:

- Inspect one crop map file schema and geometry type before changing the sampling logic.
- Define how polygon overlaps, partial pixels, and outside-polygon nodata should be handled on the 10 m AlphaEarth grid.

### 6. Repository hygiene risk

- This directory is not currently a git repository and has no ignore rules yet.
- Without an explicit bootstrap step, the first push could accidentally include generated data, notebook outputs, or credentials.

Fastest next check:

- add `.gitignore` before the first commit
- inspect the tree for local data outputs and secret material before pushing

### 7. CI environment mismatch

- Current training code depends on environment-specific packages such as Earth Engine auth, `xgboost`, and `cudf`.
- A naive GitHub Actions workflow will fail unless CPU-safe and authenticated test paths are separated.

Fastest next check:

- identify which modules can be tested without Earth Engine credentials or GPU support and make that the required CI baseline

## Implementation Sequence

1. Bootstrap the GitHub repository at `git@github.com:MarcYin/CROME.git` with a clean first commit:
   - initialize git
   - add `.gitignore`
   - commit the current scripts plus this migration plan
   - push a `main` branch
2. Add package scaffolding:
   - `pyproject.toml`
   - `src/crome/`
   - `tests/`
   - `docs/`
3. Add CI and documentation automation:
   - `.github/workflows/ci.yml`
   - `.github/workflows/docs.yml`
   - `mkdocs` configuration
4. Confirm actual embedding years available in Earth Engine and reduce the initial target range to 2017-2024 unless 2025 appears for the target AOI.
5. Validate `edown` on one AlphaEarth tile-year and lock the download approach before broader refactoring.
6. Implement a new embedding acquisition path and write local files with a stable naming convention.
7. Refactor `sample_spectra.py` to emit a generic feature-based sample dataset.
8. Refactor `merge_samples.py` for the new sample schema.
9. Refactor training loaders to derive feature names dynamically.
10. Refactor inference to use the same stored feature order.
11. Run one year on one tile end-to-end before scaling out.
12. Only after validation, retire or de-emphasize the Sentinel-2 monthly-composite path.

## GitHub Repository Bootstrap Plan

Current confirmed state before the first push:

- this directory is not currently a git repository
- no `.gitignore`, `README.md`, `LICENSE`, or `pyproject.toml` exists yet
- no `tests/`, `docs/`, or `.github/workflows/` tree exists yet

Initial repo tasks:

1. Initialize a git repository locally and set:
   - `origin = git@github.com:MarcYin/CROME.git`
2. Use `main` as the default branch.
3. Add repository hygiene before the first substantial implementation commit:
   - `.gitignore` for Python caches, local envs, package builds, docs build output, notebook checkpoints, and data/model artifacts
   - `README.md` with project purpose and current migration status
   - license file if the repository is intended to be public
4. Keep the first push small:
   - current source files
   - migration plan
   - repo hygiene files
   - no generated data, models, or local credentials

Primary repo hygiene risks to address before implementation begins:

- current scripts use hard-coded absolute paths, which will fail in CI and for other users unless moved into config or CLI options
- future Earth Engine outputs, models, and Zarr/NetCDF/GeoTIFF artifacts must stay ignored by default
- notebooks should be treated as documentation or experiments, not as the package entry point
- no license decision has been captured yet, which matters if the repository will be public

Recommended `.gitignore` coverage:

- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`
- `.venv/`, `venv/`, `env/`
- `build/`, `dist/`, `*.egg-info/`
- `site/`, `.mkdocs/`
- `.ipynb_checkpoints/`
- `data/`, `outputs/`, `models/`, `artifacts/`
- `*.zarr/`, `*.nc`, `*.tif`, `*.tiff`, `*.fgb`

## GitHub Actions and Documentation Plan

### CI workflow

Create `.github/workflows/ci.yml` with:

- triggers:
  - `push` to `main`
  - `pull_request`
- jobs:
  - install package with development extras
  - run linting
  - run tests
  - run package build

Recommended CI steps:

- Python matrix for actively supported versions
- `python -m pip install -e .[dev]`
- `ruff check .`
- `pytest`
- `python -m build`
- no Earth Engine credentials required for the default CI path
- fail fast on lint/test/build before any slower optional jobs

Minimum test coverage target for the first package pass:

- feature schema tests
- one downloader/sampling smoke test with mocks or tiny fixtures
- one CLI integration smoke test

### Documentation workflow

Create `.github/workflows/docs.yml` with:

- trigger on push to `main` and `workflow_dispatch`
- build docs from `docs/`
- upload the built site as a Pages artifact
- deploy using GitHub Pages actions
- set `permissions` to the minimum required for Pages deploy:
  - `contents: read`
  - `pages: write`
  - `id-token: write`
- set a Pages deployment `concurrency` group so only the latest `main` deployment wins

Recommended docs stack:

- `mkdocs` with `mkdocs-material` for a fast GitHub Pages path
- include:
  - installation
  - CLI usage
  - migration status
  - data schema documentation
  - API reference for package modules as they appear

### Release/build workflow

Create `.github/workflows/release.yml` after the package is installable:

- trigger on version tags
- build sdist and wheel
- attach build artifacts to GitHub releases

Optional later step:

- publish to PyPI once the package surface is stable

### Integration workflow boundary

Keep live Earth Engine and `edown` verification out of the default PR workflow.

Recommended approach:

- deterministic CI on every PR with no secrets
- a separate integration workflow triggered by `workflow_dispatch` or schedule
- Earth Engine credentials scoped only to that integration workflow
- integration failures should never block docs-only or package-structure changes

## Validation Plan

Minimum validation before full migration:

1. Download one embedding scene for one UK tile footprint and verify:
   - year selection
   - 64 bands present
   - dtype
   - spatial resolution
2. Extract samples for one year and confirm:
   - expected sample count
   - feature count = 64
   - no schema drift between tiles
3. Train one binary model and one multi-class model on the new features.
4. Reclassify one tile and verify:
   - output raster alignment
   - no feature-order mismatch
   - no chunking/dtype failures
5. Compare performance against the current Sentinel-2 monthly baseline before switching the default pipeline.
6. Verify GitHub Actions can:
   - run tests on every push
   - build the package successfully
   - deploy docs to GitHub Pages from `main`
7. Verify one failure path and one recovery path for repository automation:
   - failure path: missing integration secrets skips the integration workflow cleanly without failing unit-test CI
   - recovery path: a broken docs deploy is fixed by the next successful push to `main`, without manual branch surgery

## Recommended Deliverable Boundary for the First Implementation Pass

First pass should deliver:

- package scaffold
- GitHub repo bootstrap on `main`
- deterministic GitHub Actions CI
- GitHub Pages docs deployment
- embedding acquisition for one year
- sampled feature extraction for one year
- one training path updated
- one inference path updated
- explicit documentation of the new feature schema

First pass should not attempt:

- full removal of the old Sentinel-2 workflow
- support for 2025 unless the dataset is actually published
- simultaneous refactoring of every legacy training variant

## References

- Earth Engine Data Catalog:
  - https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- Earth Engine community tutorial:
  - https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction
- `edown` package:
  - https://pypi.org/project/edown/
  - https://marcyin.github.io/edown/
- Comparative package check:
  - https://pypi.org/project/gee-downloader/
