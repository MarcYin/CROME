# Nextflow on JASMIN

The cluster wrapper lives in [main.nf](/home/users/marcyin/UK_crop_map/workflows/nextflow/main.nf) and [nextflow.config](/home/users/marcyin/UK_crop_map/workflows/nextflow/nextflow.config).

The workflow shape is:

1. Run `crome prepare-tile-batch` once on the login node or an interactive session.
2. Run one `crome run-tile-plan` task per prepared AlphaEarth image tile.
3. Optionally run one pooled `crome train-pooled-from-tile-results` task after the tile jobs finish.

Typical usage:

```bash
crome prepare-tile-batch \
  --manifest-path /gws/ssde/j25a/nceo_isp/public/CROME/raw/alphaearth/<run>/manifests/run.json \
  --reference-path /gws/ssde/j25a/nceo_isp/public/CROME/raw/crome/CROME_2024_national/extracted/Crop_Map_of_England_CROME_2024.gpkg \
  --output-root /gws/ssde/j25a/nceo_isp/public/CROME \
  --year 2024 \
  --aoi-label east-anglia-2024

nextflow run workflows/nextflow/main.nf \
  -c workflows/nextflow/nextflow.config \
  -profile jasmin \
  --batch_manifest /gws/ssde/j25a/nceo_isp/public/CROME/workflow/<tile-batch-namespace>/BATCH_east-anglia-2024_2024/batch_manifest.json \
  --tile_cpus 16 \
  --tile_memory '128 GB' \
  --pooled_cpus 48 \
  --pooled_memory '512 GB'
```

The JASMIN queue defaults in the workflow are based on the current queue guidance:

- `debug` partition with QoS `debug`: up to `8` CPUs and `1 hour`, useful for smoke tests.
- `standard` partition with QoS `high`: up to `96` CPUs, `48 hours`, and `1000 GB`, which is the default profile for multi-core tile jobs and pooled training.
- `special` partition with QoS `special`: up to `96` CPUs, `48 hours`, and `3000 GB` on the new `6 TB` nodes, for very large pooled fits if your account has access.

Source: https://help.jasmin.ac.uk/docs/batch-computing/slurm-queues/

Why the workflow is structured this way:

- The per-tile commands keep tile-local labels, models, predictions, and caches aligned with the existing `crome` CLI.
- The prepared tile manifests give each tile job a unique run label, so one-tile `pipeline.json` files do not overwrite each other.
- The pooled stage only trusts emitted per-tile `pipeline.json` paths, which preserves the same provenance chain as the local CLI path.

The `jasmin_special` profile only changes the pooled-model step to the access-controlled `special` partition. The tile jobs stay on `standard` with QoS `high`, which is the better default for many parallel tile tasks.
