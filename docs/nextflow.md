# Nextflow on JASMIN

The repo includes a Slurm-oriented Nextflow wrapper in [main.nf](/home/users/marcyin/UK_crop_map/nextflow/main.nf) with queue profiles in [nextflow.config](/home/users/marcyin/UK_crop_map/nextflow/nextflow.config).
GitHub Actions installs Nextflow and runs a small local smoke test for this wrapper on every push to `main`.

It does not replace the Python pipeline. It schedules the existing `crome` commands in a cluster-friendly order:

1. `crome prepare-tile-batch` discovers AlphaEarth tiles, materializes one shared CROME subset for the batch, and writes one tile-plan JSON per AlphaEarth tile.
2. `nextflow/main.nf` takes the emitted `batch_manifest.json` and runs one `crome run-tile-plan` task per tile-plan JSON.
3. The resulting tile-result JSON files are optionally gathered into one `crome train-pooled-from-tile-results` task.

That keeps tile identity, subset reuse, cache reuse, and pooled-training provenance intact while letting Slurm parallelize the expensive work.

## Queue mapping

The current JASMIN Slurm queue guidance supports this split:

- `debug` partition with QoS `debug`: up to `8` CPUs for `1 hour`; use for the lightweight batch-planning/control step.
- `standard` partition with QoS `high`: up to `96` CPUs, `48 hours`, and `1000 GB` per job; use for per-tile training or pooled training when you want multi-core random-forest fits.
- `special` partition with QoS `special`: access-controlled, backed by `6 TB` / `192` core nodes, with jobs allowed up to `96` CPUs, `48 hours`, and `3000 GB`; use only if your account has access.

Source: <https://help.jasmin.ac.uk/docs/batch-computing/slurm-queues/>

## Typical run

```bash
crome prepare-tile-batch \
  --manifest-path /gws/ssde/j25a/nceo_isp/public/CROME/raw/alphaearth/AEF_cambridge-fringe-smoke_annual_embedding_2024/manifests/run-20260330T213729Z.json \
  --reference-path /gws/ssde/j25a/nceo_isp/public/CROME/raw/crome/CROME_2024_national/extracted/Crop_Map_of_England_CROME_2024.gpkg \
  --output-root /gws/ssde/j25a/nceo_isp/public/CROME \
  --year 2024 \
  --aoi-label cambridge-norfolk-2024

nextflow run nextflow/main.nf -c nextflow/nextflow.config -profile jasmin \
  --batch_manifest /gws/ssde/j25a/nceo_isp/public/CROME/workflow/<tile-batch-namespace>/BATCH_cambridge-norfolk-2024_2024/batch_manifest.json \
  --output_root /gws/ssde/j25a/nceo_isp/public/CROME \
  --tile_partition standard \
  --tile_qos high \
  --tile_cpus 16 \
  --tile_memory '128 GB' \
  --pooled_partition standard \
  --pooled_qos high \
  --pooled_cpus 64 \
  --pooled_memory '512 GB' \
  --slurm_account my-gws
```

If you have access to the high-memory `special` partition, switch to:

```bash
nextflow run nextflow/main.nf -c nextflow/nextflow.config -profile jasmin_special \
  --batch_manifest /path/to/batch_manifest.json \
  --output_root /gws/ssde/j25a/nceo_isp/public/CROME \
  --slurm_account my-gws
```

## Notes

- Use `prepare-tile-batch --manifest-path` when you want the workflow to preserve the original `edown` source-image identity for each AlphaEarth tile.
- Use `prepare-tile-batch --feature-input` when you already have a directory tree of AlphaEarth GeoTIFFs and do not need manifest metadata.
- The JASMIN profiles set `cache = 'lenient'`, which Nextflow documents as a workaround for shared-filesystem timestamp inconsistencies.
- The tile stage can be grouped into Slurm job arrays with `--tile_array_size <N>` to reduce scheduler overhead when you have many AlphaEarth tiles.
- The Nextflow wrapper reads rasters in place from the shared filesystem; it does not stage the full AlphaEarth GeoTIFFs into each task directory.
- Each tile plan points at the same prepared CROME subset for the batch, so the expensive reference clipping work happens once.
- Per-tile labels, models, predictions, and sample caches still come from the existing Python code, so pooled training remains compatible with `crome train-pooled-model`.
- Nextflow itself is not bundled with the Python package. Load or install it in your JASMIN software environment before running the workflow.
