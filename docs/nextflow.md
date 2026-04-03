# Nextflow on JASMIN

The repository includes a Slurm-oriented Nextflow wrapper under `nextflow/main.nf` with queue profiles in `nextflow/nextflow.config`.

It does not replace the Python pipeline. It schedules the existing `crome` commands in a cluster-friendly order:

1. **Prepare** -- `crome prepare-tile-batch` runs on the login node. It discovers AlphaEarth tiles, materializes one shared CROME subset, and writes one tile-plan JSON per tile.
2. **Fan out** -- Nextflow reads the batch manifest and runs one `crome run-tile-plan` task per tile in parallel across Slurm.
3. **Gather** -- The tile-result JSONs are optionally collected into one `crome train-pooled-from-tile-results` task for a regional or national model.

Tile identity, subset reuse, cache reuse, and pooled-training provenance are preserved throughout.

## Prerequisites

Nextflow is **not** bundled with the Python package. Install it in your JASMIN environment:

```bash
curl -s https://get.nextflow.io | bash
./nextflow -version
```

Java 11+ is required (Java 17 recommended). On JASMIN, check `module avail java` or install via conda.

## Queue mapping

| Partition | QoS | Max CPUs | Max memory | Max time | Use case |
|-----------|-----|----------|------------|----------|----------|
| `debug` | `debug` | 8 | -- | 1 h | Batch planning / control |
| `standard` | `high` | 96 | 1000 GB | 48 h | Per-tile training |
| `special` | `special` | 96 | 3000 GB | 48 h | High-memory pooled training |

Source: <https://help.jasmin.ac.uk/docs/batch-computing/slurm-queues/>

## Typical run

### Step 1: Prepare the batch

Run on the login node (no Slurm allocation needed):

```bash
crome prepare-tile-batch \
  --manifest-path /gws/.../raw/alphaearth/AEF_..._2024/manifests/run.json \
  --reference-path /gws/.../raw/crome/CROME_2024_national/extracted/Crop_Map_of_England_CROME_2024.gpkg \
  --output-root /gws/ssde/j25a/nceo_isp/public/CROME \
  --year 2024 \
  --aoi-label cambridge-norfolk-2024
```

This emits a `batch_manifest.json` and per-tile plans under the workflow directory.

### Step 2: Launch Nextflow

```bash
nextflow run nextflow/main.nf \
  -c nextflow/nextflow.config \
  -profile jasmin \
  --batch_manifest /gws/.../workflow/<namespace>/BATCH_cambridge-norfolk-2024_2024/batch_manifest.json \
  --output_root /gws/ssde/j25a/nceo_isp/public/CROME \
  --tile_cpus 16 \
  --tile_memory '128 GB' \
  --pooled_cpus 64 \
  --pooled_memory '512 GB' \
  --slurm_account my-gws
```

### Step 3 (optional): Special partition

If your account has access to the high-memory `special` partition:

```bash
nextflow run nextflow/main.nf \
  -c nextflow/nextflow.config \
  -profile jasmin_special \
  --batch_manifest /path/to/batch_manifest.json \
  --output_root /gws/ssde/j25a/nceo_isp/public/CROME \
  --slurm_account my-gws
```

## Nextflow parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_manifest` | required | Path to the prepared batch manifest JSON |
| `--output_root` | from batch | Override output root for all tasks |
| `--run_pooled_model` | `true` | Run the pooled training step after tile jobs |
| `--tile_cpus` | profile default | CPUs allocated per tile task |
| `--tile_memory` | profile default | Memory per tile task |
| `--tile_n_jobs` | `tile_cpus` | RandomForest CPU parallelism per tile |
| `--pooled_cpus` | profile default | CPUs for pooled training |
| `--pooled_memory` | profile default | Memory for pooled training |
| `--pooled_n_jobs` | `pooled_cpus` | RandomForest CPU parallelism for pooled model |
| `--tile_array_size` | 128 | Group tile jobs into Slurm arrays |
| `--slurm_account` | none | Slurm account for job submission |
| `--max_train_rows` | none | Cap pooled training rows |
| `--n_estimators` | none | Override tree count for pooled model |
| `--python` | `python` | Python executable path |

## Notes

- Use `--manifest-path` with `prepare-tile-batch` to preserve edown source-image identity for each tile.
- Use `--feature-input` when you have a directory of GeoTIFFs without an edown manifest.
- The JASMIN profiles set `cache = 'lenient'` as a workaround for shared-filesystem timestamp inconsistencies.
- `--tile_array_size` groups tile jobs into Slurm job arrays to reduce scheduler overhead.
- Rasters are read in place from the shared filesystem; Nextflow does not stage full GeoTIFFs into task directories.
- Each tile plan references the same prepared CROME subset, so the reference clipping happens once.
- Per-tile outputs (labels, models, predictions, sample caches) are written by the Python pipeline, so pooled training via `crome train-pooled-model` remains compatible.
- GitHub Actions installs Nextflow and runs a local smoke test against the wrapper on every push to `main`.
