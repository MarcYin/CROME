#!/usr/bin/env python
"""
Full England CROME run for years 2017-2024.

Drives the crome package to:
1. Download CROME references for each year (if not already present)
2. Export the CROME footprint bounding box
3. Download AlphaEarth tiles for the footprint per year
4. Prepare tile batches for Nextflow/Slurm execution
5. Optionally run the pipeline directly (single-machine) or emit Nextflow commands

Usage:
    # Dry-run: show what would be downloaded and how many tiles per year
    python scripts/run_england_2017_2024.py --dry-run

    # Prepare all years (download CROME refs + AlphaEarth + batch plans)
    python scripts/run_england_2017_2024.py --prepare

    # Run pipeline directly for one year (single-machine, for testing)
    python scripts/run_england_2017_2024.py --run-year 2024

    # Emit Nextflow commands for all years (for cluster execution)
    python scripts/run_england_2017_2024.py --nextflow-commands

Environment:
    CROME_DATA_ROOT     Shared output root (default: /gws/ssde/j25a/nceo_isp/public/CROME)
    Earth Engine must be authenticated: earthengine authenticate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure the package is importable from the repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("run_england")

YEARS = list(range(2017, 2025))  # 2017-2024 inclusive
DEFAULT_OUTPUT_ROOT = "/gws/ssde/j25a/nceo_isp/public/CROME"

# England CROME footprint in EPSG:4326 (union of all county layers)
ENGLAND_BBOX = (-7.06, 49.86, 2.08, 55.82)


def get_output_root() -> Path:
    return Path(os.environ.get("CROME_DATA_ROOT", DEFAULT_OUTPUT_ROOT))


def download_crome_reference(year: int, output_root: Path) -> Path | None:
    """Download CROME reference for one year if not already present."""
    from crome.config import CromeDownloadRequest
    from crome.acquisition.crome import download_crome_reference as _download

    logger.info("Downloading CROME reference for %d...", year)
    try:
        request = CromeDownloadRequest(
            year=year,
            output_root=output_root,
            prefer_complete=True,
            extract=True,
            force=False,
        )
        result = _download(request)
        ref_path = result.reference_path
        logger.info("CROME %d reference: %s", year, ref_path)
        return ref_path
    except Exception as exc:
        logger.error("Failed to download CROME %d: %s", year, exc)
        return None


def find_existing_crome_reference(year: int, output_root: Path) -> Path | None:
    """Find an already-downloaded CROME reference for one year."""
    from crome.paths import crome_download_root
    for variant in ("national", "complete"):
        root = crome_download_root(output_root, year, variant_label=variant)
        extracted = root / "extracted"
        if extracted.exists():
            gpkgs = sorted(extracted.glob("*.gpkg"))
            if gpkgs:
                return gpkgs[0]
    return None


def download_alphaearth_tiles(year: int, output_root: Path, bbox: tuple, aoi_label: str) -> dict | None:
    """Download AlphaEarth tiles for one year's CROME footprint."""
    from crome.config import AlphaEarthDownloadRequest
    from crome.acquisition.alphaearth import download_alphaearth_images

    logger.info("Downloading AlphaEarth tiles for %d (bbox=%s)...", year, bbox)
    try:
        request = AlphaEarthDownloadRequest(
            year=year,
            output_root=output_root,
            aoi_label=aoi_label,
            bbox=bbox,
        )
        result = download_alphaearth_images(request)
        logger.info(
            "AlphaEarth %d: %d images downloaded to %s",
            year, len(result.source_image_ids), result.output_root,
        )
        return {
            "manifest_path": str(result.manifest_path),
            "output_root": str(result.output_root),
            "image_count": len(result.source_image_ids),
        }
    except Exception as exc:
        logger.error("Failed to download AlphaEarth %d: %s", year, exc)
        return None


def prepare_tile_batch(
    year: int,
    output_root: Path,
    manifest_path: str,
    reference_path: Path,
    aoi_label: str,
) -> dict | None:
    """Prepare a tile batch for one year."""
    from crome.orchestration import prepare_tile_batch as _prepare

    logger.info("Preparing tile batch for %d...", year)
    try:
        result = _prepare(
            feature_input=None,
            manifest_path=manifest_path,
            reference_path=reference_path,
            year=year,
            output_root=output_root,
            aoi_label=aoi_label,
            n_estimators=400,
        )
        logger.info(
            "Batch %d: %d tiles prepared → %s",
            year, len(result.tile_manifest_paths), result.batch_manifest_path,
        )
        return {
            "batch_manifest_path": str(result.batch_manifest_path),
            "tile_count": len(result.tile_manifest_paths),
            "workflow_output_root": str(result.workflow_output_root),
        }
    except Exception as exc:
        logger.error("Failed to prepare batch for %d: %s", year, exc)
        return None


def dry_run(output_root: Path) -> None:
    """Show what would be done without downloading anything."""
    logger.info("=== DRY RUN ===")
    logger.info("Output root: %s", output_root)
    logger.info("England bbox (EPSG:4326): %s", ENGLAND_BBOX)
    logger.info("Years: %s", YEARS)
    print()

    total_tiles = 0
    for year in YEARS:
        existing_ref = find_existing_crome_reference(year, output_root)
        ref_status = f"exists: {existing_ref}" if existing_ref else "will download"
        aoi_label = f"england-crome-{year}"

        # Estimate tile count from bbox
        lon_range = ENGLAND_BBOX[2] - ENGLAND_BBOX[0]
        lat_range = ENGLAND_BBOX[3] - ENGLAND_BBOX[1]
        import math
        est_tiles = int(math.ceil(lon_range / 0.7) * math.ceil(lat_range / 0.7))
        total_tiles += est_tiles

        print(f"  {year}:")
        print(f"    CROME reference: {ref_status}")
        print(f"    AOI label: {aoi_label}")
        print(f"    Estimated AlphaEarth tiles: ~{est_tiles}")
        print()

    print(f"  Total estimated tiles across all years: ~{total_tiles}")
    print(f"  Estimated disk for AlphaEarth rasters: ~{total_tiles * 100 / 1024:.0f} GB")
    print()


def prepare_all(output_root: Path) -> dict:
    """Download CROME refs, AlphaEarth tiles, and prepare batches for all years."""
    results = {}

    for year in YEARS:
        logger.info("=" * 60)
        logger.info("Processing year %d", year)
        logger.info("=" * 60)

        aoi_label = f"england-crome-{year}"
        year_result = {"year": year, "aoi_label": aoi_label}

        # Step 1: CROME reference
        ref_path = find_existing_crome_reference(year, output_root)
        if ref_path is None:
            ref_path = download_crome_reference(year, output_root)
        if ref_path is None:
            logger.error("Skipping year %d: no CROME reference available", year)
            year_result["status"] = "skipped_no_reference"
            results[year] = year_result
            continue
        year_result["reference_path"] = str(ref_path)

        # Step 2: Download AlphaEarth tiles
        ae_result = download_alphaearth_tiles(year, output_root, ENGLAND_BBOX, aoi_label)
        if ae_result is None:
            logger.error("Skipping year %d: AlphaEarth download failed", year)
            year_result["status"] = "skipped_download_failed"
            results[year] = year_result
            continue
        year_result["alphaearth"] = ae_result

        # Step 3: Prepare tile batch
        batch_result = prepare_tile_batch(
            year, output_root, ae_result["manifest_path"], ref_path, aoi_label,
        )
        if batch_result is None:
            logger.error("Skipping year %d: batch preparation failed", year)
            year_result["status"] = "skipped_batch_failed"
            results[year] = year_result
            continue
        year_result["batch"] = batch_result
        year_result["status"] = "prepared"
        results[year] = year_result

    # Write summary manifest
    summary_path = output_root / "england_2017_2024_run_summary.json"
    summary = {
        "bbox": list(ENGLAND_BBOX),
        "output_root": str(output_root),
        "years": {str(k): v for k, v in results.items()},
        "total_tiles": sum(
            r.get("batch", {}).get("tile_count", 0) for r in results.values()
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Summary written to %s", summary_path)

    # Print summary
    print("\n" + "=" * 60)
    print("PREPARATION SUMMARY")
    print("=" * 60)
    for year, result in sorted(results.items()):
        status = result.get("status", "unknown")
        tiles = result.get("batch", {}).get("tile_count", 0)
        print(f"  {year}: {status} ({tiles} tiles)")
    total = summary["total_tiles"]
    print(f"\n  Total tiles: {total}")
    print(f"  Summary: {summary_path}")

    return results


def emit_nextflow_commands(output_root: Path) -> None:
    """Print Nextflow commands for all prepared batches."""
    summary_path = output_root / "england_2017_2024_run_summary.json"
    if not summary_path.exists():
        logger.error("No summary found. Run --prepare first.")
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("# Nextflow commands for England 2017-2024")
    print(f"# Total tiles: {summary['total_tiles']}")
    print(f"# Output root: {summary['output_root']}")
    print()

    for year_str, result in sorted(summary["years"].items()):
        if result.get("status") != "prepared":
            print(f"# Year {year_str}: {result.get('status', 'unknown')} — skipping")
            continue
        batch_manifest = result["batch"]["batch_manifest_path"]
        tile_count = result["batch"]["tile_count"]
        print(f"# Year {year_str}: {tile_count} tiles")
        print(f"nextflow run nextflow/main.nf \\")
        print(f"  -c nextflow/nextflow.config \\")
        print(f"  -profile jasmin \\")
        print(f"  --batch_manifest {batch_manifest} \\")
        print(f"  --output_root {summary['output_root']} \\")
        print(f"  --tile_cpus 16 \\")
        print(f"  --tile_memory '128 GB' \\")
        print(f"  --pooled_cpus 64 \\")
        print(f"  --pooled_memory '512 GB' \\")
        print(f"  --slurm_account nceo_isp")
        print()


def run_single_year(year: int, output_root: Path) -> None:
    """Run the full pipeline for one year on a single machine."""
    from crome.workflow import prepare_footprint_tile_batch
    from crome.orchestration import run_tile_plan, train_pooled_from_tile_results

    aoi_label = f"england-crome-{year}"

    # Check for existing CROME reference
    ref_path = find_existing_crome_reference(year, output_root)
    if ref_path is None:
        ref_path = download_crome_reference(year, output_root)
    if ref_path is None:
        logger.error("Cannot run year %d: no CROME reference", year)
        return

    # Download AlphaEarth + prepare batch
    ae_result = download_alphaearth_tiles(year, output_root, ENGLAND_BBOX, aoi_label)
    if ae_result is None:
        logger.error("Cannot run year %d: AlphaEarth download failed", year)
        return

    batch_result = prepare_tile_batch(
        year, output_root, ae_result["manifest_path"], ref_path, aoi_label,
    )
    if batch_result is None:
        logger.error("Cannot run year %d: batch preparation failed", year)
        return

    batch_manifest = batch_result["batch_manifest_path"]
    tile_count = batch_result["tile_count"]
    logger.info("Running %d tiles for year %d on single machine...", tile_count, year)

    # Load tile plans
    batch_payload = json.loads(Path(batch_manifest).read_text(encoding="utf-8"))
    tile_manifest_paths = batch_payload.get("tile_manifest_paths", [])

    # Run each tile
    tile_result_paths = []
    for i, tile_path in enumerate(tile_manifest_paths, 1):
        logger.info("Tile %d/%d: %s", i, tile_count, Path(tile_path).stem)
        try:
            result = run_tile_plan(tile_path)
            # Write tile result
            tile_result_path = Path(batch_result["workflow_output_root"]) / f"{Path(tile_path).stem}.tile-result.json"
            tile_result_path.parent.mkdir(parents=True, exist_ok=True)
            payload = result.pipeline.to_dict()
            payload["pipeline_manifest_path"] = str(result.pipeline.pipeline_manifest_path)
            tile_result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tile_result_paths.append(str(tile_result_path))
            logger.info("Tile %d/%d: complete (%d features)", i, tile_count, len(result.pipeline.feature_results))
        except Exception as exc:
            logger.error("Tile %d/%d failed: %s", i, tile_count, exc)

    # Pooled training
    if len(tile_result_paths) >= 2:
        logger.info("Training pooled model from %d tile results...", len(tile_result_paths))
        try:
            pooled = train_pooled_from_tile_results(
                batch_manifest_path=batch_manifest,
                tile_result_paths=tile_result_paths,
            )
            logger.info(
                "Pooled model trained: %d rows, metrics at %s",
                pooled.training_table.row_count,
                pooled.trained_model.metrics_path,
            )
        except Exception as exc:
            logger.error("Pooled training failed: %s", exc)
    else:
        logger.warning("Only %d tile result(s); skipping pooled training", len(tile_result_paths))


def check_results(output_root: Path) -> None:
    """Check and summarize results from completed runs."""
    summary_path = output_root / "england_2017_2024_run_summary.json"
    if not summary_path.exists():
        logger.error("No summary found at %s", summary_path)
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("=" * 70)
    print("RESULTS CHECK: England 2017-2024")
    print("=" * 70)
    print(f"Output root: {summary['output_root']}")
    print()

    grand_total_tiles = 0
    grand_completed = 0
    grand_models = 0
    grand_predictions = 0

    for year_str, year_info in sorted(summary.get("years", {}).items()):
        year = int(year_str)
        status = year_info.get("status", "unknown")
        batch = year_info.get("batch", {})
        batch_manifest = batch.get("batch_manifest_path")
        tile_count = batch.get("tile_count", 0)
        grand_total_tiles += tile_count

        if status != "prepared" or not batch_manifest or not Path(batch_manifest).exists():
            print(f"  {year}: {status} (no batch manifest)")
            continue

        # Check tile results
        batch_payload = json.loads(Path(batch_manifest).read_text(encoding="utf-8"))
        tile_manifests = batch_payload.get("tile_manifest_paths", [])

        completed_tiles = 0
        models_found = 0
        predictions_found = 0
        metrics_data = []

        for tile_path in tile_manifests:
            tile_payload = json.loads(Path(tile_path).read_text(encoding="utf-8"))
            tile_id = tile_payload.get("tile_id", "unknown")
            # Check if pipeline.json exists for this tile
            # Look for training outputs
            workflow_root = batch.get("workflow_output_root", "")
            tile_result = Path(workflow_root) / f"{tile_id}.tile-result.json"
            if tile_result.exists():
                completed_tiles += 1
                result_payload = json.loads(tile_result.read_text(encoding="utf-8"))
                features = result_payload.get("features", [])
                for feat in features:
                    if feat.get("model_path") and Path(feat["model_path"]).exists():
                        models_found += 1
                    if feat.get("prediction_raster_path") and Path(feat["prediction_raster_path"]).exists():
                        predictions_found += 1
                    metrics_path = feat.get("metrics_path")
                    if metrics_path and Path(metrics_path).exists():
                        m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
                        metrics_data.append({
                            "tile_id": tile_id,
                            "accuracy": m.get("accuracy"),
                            "macro_f1": m.get("macro_f1"),
                            "weighted_f1": m.get("weighted_f1"),
                            "eval_mode": m.get("evaluation_mode"),
                            "train_rows": m.get("fit_row_count"),
                        })

        grand_completed += completed_tiles
        grand_models += models_found
        grand_predictions += predictions_found

        print(f"  {year}: {completed_tiles}/{tile_count} tiles completed, "
              f"{models_found} models, {predictions_found} predictions")

        if metrics_data:
            accs = [m["accuracy"] for m in metrics_data if m["accuracy"] is not None]
            f1s = [m["macro_f1"] for m in metrics_data if m["macro_f1"] is not None]
            wf1s = [m["weighted_f1"] for m in metrics_data if m["weighted_f1"] is not None]
            rows = [m["train_rows"] for m in metrics_data if m["train_rows"] is not None]
            if accs:
                print(f"    Accuracy: min={min(accs):.3f}, median={sorted(accs)[len(accs)//2]:.3f}, "
                      f"max={max(accs):.3f}")
            if f1s:
                print(f"    Macro F1: min={min(f1s):.3f}, median={sorted(f1s)[len(f1s)//2]:.3f}, "
                      f"max={max(f1s):.3f}")
            if wf1s:
                print(f"    Weighted F1: min={min(wf1s):.3f}, median={sorted(wf1s)[len(wf1s)//2]:.3f}, "
                      f"max={max(wf1s):.3f}")
            if rows:
                print(f"    Training rows: min={min(rows)}, median={sorted(rows)[len(rows)//2]}, "
                      f"max={max(rows)}, total={sum(rows)}")

    print()
    print(f"  TOTAL: {grand_completed}/{grand_total_tiles} tiles, "
          f"{grand_models} models, {grand_predictions} predictions")

    # Check pooled models
    pooled_dir = output_root / "training" / "pooled"
    if pooled_dir.exists():
        pooled_models = list(pooled_dir.rglob("model.pkl"))
        pooled_metrics = list(pooled_dir.rglob("metrics.json"))
        print(f"\n  Pooled models: {len(pooled_models)}")
        for mp in pooled_metrics:
            m = json.loads(mp.read_text(encoding="utf-8"))
            print(f"    {mp.parent.parent.name}: acc={m.get('accuracy')}, "
                  f"macro_f1={m.get('macro_f1')}, weighted_f1={m.get('weighted_f1')}, "
                  f"rows={m.get('fit_row_count')}")


def main():
    parser = argparse.ArgumentParser(description="Full England CROME run for 2017-2024")
    parser.add_argument("--output-root", default=None, help="Override CROME_DATA_ROOT")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Show plan without downloading")
    group.add_argument("--prepare", action="store_true", help="Download data and prepare all batches")
    group.add_argument("--run-year", type=int, help="Run full pipeline for one year (single-machine)")
    group.add_argument("--nextflow-commands", action="store_true", help="Emit Nextflow commands")
    group.add_argument("--check-results", action="store_true", help="Check and summarize completed results")
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else get_output_root()
    os.environ["CROME_DATA_ROOT"] = str(output_root)

    if args.dry_run:
        dry_run(output_root)
    elif args.prepare:
        prepare_all(output_root)
    elif args.run_year:
        if args.run_year not in YEARS:
            parser.error(f"Year must be in {YEARS}")
        run_single_year(args.run_year, output_root)
    elif args.nextflow_commands:
        emit_nextflow_commands(output_root)
    elif args.check_results:
        check_results(output_root)


if __name__ == "__main__":
    main()
