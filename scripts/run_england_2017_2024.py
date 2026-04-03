#!/usr/bin/env python
"""
Full England CROME run for years 2017-2024.

Downloads all years in parallel (4 workers per year) and starts tile-level
pipeline processing as soon as each tile finishes downloading, without
waiting for the full year to complete.

Usage:
    # Dry-run: show what would be downloaded and how many tiles per year
    python scripts/run_england_2017_2024.py --dry-run

    # Prepare all years in parallel (download CROME refs + AlphaEarth + batch plans)
    python scripts/run_england_2017_2024.py --prepare

    # Run pipeline on all downloaded tiles (after --prepare or incrementally)
    python scripts/run_england_2017_2024.py --run-tiles

    # Emit Nextflow commands for all prepared batches
    python scripts/run_england_2017_2024.py --nextflow-commands

    # Check and summarize results
    python scripts/run_england_2017_2024.py --check-results

Environment:
    CROME_DATA_ROOT     Shared output root (default: /gws/ssde/j25a/nceo_isp/public/CROME)
    Earth Engine must be authenticated: earthengine authenticate
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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

DOWNLOAD_WORKERS_PER_YEAR = 4
PREPARE_WORKERS_PER_YEAR = 4
MAX_CONCURRENT_YEARS = 8


def get_output_root() -> Path:
    return Path(os.environ.get("CROME_DATA_ROOT", DEFAULT_OUTPUT_ROOT))


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


def download_crome_reference(year: int, output_root: Path) -> Path | None:
    """Download CROME reference for one year if not already present."""
    from crome.config import CromeDownloadRequest
    from crome.acquisition.crome import download_crome_reference as _download

    logger.info("[%d] Downloading CROME reference...", year)
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
        logger.info("[%d] CROME reference: %s", year, ref_path)
        return ref_path
    except Exception as exc:
        logger.error("[%d] Failed to download CROME: %s", year, exc)
        return None


def prepare_one_year(year: int, output_root: Path) -> dict:
    """Download AlphaEarth tiles for one year and prepare the tile batch.

    Runs with DOWNLOAD_WORKERS_PER_YEAR concurrent download threads so
    multiple years can run in parallel without saturating EE.
    """
    from crome.config import AlphaEarthDownloadRequest
    from crome.acquisition.alphaearth import download_alphaearth_images
    from crome.orchestration import prepare_tile_batch

    aoi_label = f"england-crome-{year}"
    result = {"year": year, "aoi_label": aoi_label}

    # Step 1: CROME reference
    ref_path = find_existing_crome_reference(year, output_root)
    if ref_path is None:
        ref_path = download_crome_reference(year, output_root)
    if ref_path is None:
        result["status"] = "skipped_no_reference"
        return result
    result["reference_path"] = str(ref_path)

    # Step 2: Download AlphaEarth tiles (4 workers per year)
    logger.info("[%d] Downloading AlphaEarth tiles (workers=%d)...", year, DOWNLOAD_WORKERS_PER_YEAR)
    try:
        request = AlphaEarthDownloadRequest(
            year=year,
            output_root=output_root,
            aoi_label=aoi_label,
            bbox=ENGLAND_BBOX,
        )
        ae_result = download_alphaearth_images(
            request,
            download_workers=DOWNLOAD_WORKERS_PER_YEAR,
            prepare_workers=PREPARE_WORKERS_PER_YEAR,
        )
        logger.info(
            "[%d] AlphaEarth: %d images downloaded to %s",
            year, len(ae_result.source_image_ids), ae_result.output_root,
        )
        result["alphaearth"] = {
            "manifest_path": str(ae_result.manifest_path),
            "output_root": str(ae_result.output_root),
            "image_count": len(ae_result.source_image_ids),
        }
    except Exception as exc:
        logger.error("[%d] AlphaEarth download failed: %s", year, exc)
        result["status"] = "skipped_download_failed"
        return result

    # Step 3: Prepare tile batch
    logger.info("[%d] Preparing tile batch...", year)
    try:
        batch = prepare_tile_batch(
            feature_input=None,
            manifest_path=str(ae_result.manifest_path),
            reference_path=ref_path,
            year=year,
            output_root=output_root,
            aoi_label=aoi_label,
            n_estimators=400,
        )
        logger.info("[%d] Batch prepared: %d tiles -> %s", year, len(batch.tile_manifest_paths), batch.batch_manifest_path)
        result["batch"] = {
            "batch_manifest_path": str(batch.batch_manifest_path),
            "tile_count": len(batch.tile_manifest_paths),
            "workflow_output_root": str(batch.workflow_output_root),
        }
        result["status"] = "prepared"
    except Exception as exc:
        logger.error("[%d] Batch preparation failed: %s", year, exc)
        result["status"] = "skipped_batch_failed"

    return result


def run_tile(tile_plan_path: str, year: int) -> dict:
    """Run a single tile's pipeline. Called from the tile runner pool."""
    from crome.orchestration import run_tile_plan
    tile_id = Path(tile_plan_path).stem
    try:
        result = run_tile_plan(tile_plan_path)
        pipeline_manifest = str(result.pipeline.pipeline_manifest_path)
        feature_count = len(result.pipeline.feature_results)
        logger.info("[%d] Tile %s: complete (%d features)", year, tile_id, feature_count)
        return {
            "tile_id": tile_id,
            "status": "completed",
            "pipeline_manifest_path": pipeline_manifest,
            "feature_count": feature_count,
        }
    except Exception as exc:
        logger.error("[%d] Tile %s failed: %s", year, tile_id, exc)
        return {"tile_id": tile_id, "status": "failed", "error": str(exc)}


def dry_run(output_root: Path) -> None:
    """Show what would be done without downloading anything."""
    logger.info("=== DRY RUN ===")
    logger.info("Output root: %s", output_root)
    logger.info("England bbox (EPSG:4326): %s", ENGLAND_BBOX)
    logger.info("Years: %s (all downloaded in parallel)", YEARS)
    logger.info("Download workers per year: %d", DOWNLOAD_WORKERS_PER_YEAR)
    logger.info("Total concurrent EE connections: ~%d", DOWNLOAD_WORKERS_PER_YEAR * len(YEARS))
    print()

    total_tiles = 0
    for year in YEARS:
        existing_ref = find_existing_crome_reference(year, output_root)
        ref_status = f"exists: {existing_ref}" if existing_ref else "will download"
        aoi_label = f"england-crome-{year}"
        est_tiles = 66  # measured from 2017 discovery
        total_tiles += est_tiles

        print(f"  {year}:")
        print(f"    CROME reference: {ref_status}")
        print(f"    AOI label: {aoi_label}")
        print(f"    Estimated AlphaEarth tiles: ~{est_tiles}")
        print()

    print(f"  Total estimated tiles across all years: ~{total_tiles}")
    print(f"  Estimated disk for AlphaEarth rasters: ~{total_tiles * 600 / 1024:.0f} GB")
    print(f"  Strategy: {len(YEARS)} years in parallel, {DOWNLOAD_WORKERS_PER_YEAR} workers each")
    print()


def prepare_all(output_root: Path) -> dict:
    """Download all years in parallel and prepare batches."""
    logger.info(
        "Starting parallel preparation: %d years, %d download workers each",
        len(YEARS), DOWNLOAD_WORKERS_PER_YEAR,
    )

    results = {}
    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_YEARS) as executor:
        futures = {
            executor.submit(prepare_one_year, year, output_root): year
            for year in YEARS
        }
        for future in as_completed(futures):
            year = futures[future]
            try:
                result = future.result()
                results[year] = result
                status = result.get("status", "unknown")
                tiles = result.get("batch", {}).get("tile_count", 0)
                logger.info("[%d] Finished: %s (%d tiles)", year, status, tiles)
            except Exception as exc:
                logger.error("[%d] Crashed: %s", year, exc)
                results[year] = {"year": year, "status": "crashed", "error": str(exc)}

    # Write summary manifest
    summary_path = output_root / "england_2017_2024_run_summary.json"
    summary = {
        "bbox": list(ENGLAND_BBOX),
        "output_root": str(output_root),
        "years": {str(k): v for k, v in sorted(results.items())},
        "total_tiles": sum(
            r.get("batch", {}).get("tile_count", 0) for r in results.values()
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Summary written to %s", summary_path)

    print("\n" + "=" * 60)
    print("PREPARATION SUMMARY")
    print("=" * 60)
    for year in YEARS:
        r = results.get(year, {})
        status = r.get("status", "unknown")
        tiles = r.get("batch", {}).get("tile_count", 0)
        print(f"  {year}: {status} ({tiles} tiles)")
    total = summary["total_tiles"]
    print(f"\n  Total tiles: {total}")
    print(f"  Summary: {summary_path}")

    return results


def run_tiles(output_root: Path, n_tile_workers: int = 4) -> None:
    """Run pipeline on all prepared tiles, processing each as soon as ready."""
    summary_path = output_root / "england_2017_2024_run_summary.json"
    if not summary_path.exists():
        logger.error("No summary found. Run --prepare first.")
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    all_tile_jobs = []

    for year_str, year_info in sorted(summary.get("years", {}).items()):
        year = int(year_str)
        if year_info.get("status") != "prepared":
            continue
        batch_manifest = year_info["batch"]["batch_manifest_path"]
        if not Path(batch_manifest).exists():
            continue
        batch_payload = json.loads(Path(batch_manifest).read_text(encoding="utf-8"))
        for tile_path in batch_payload.get("tile_manifest_paths", []):
            all_tile_jobs.append((tile_path, year))

    logger.info("Running %d tiles with %d parallel workers", len(all_tile_jobs), n_tile_workers)

    completed = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=n_tile_workers) as executor:
        futures = {
            executor.submit(run_tile, tile_path, year): (tile_path, year)
            for tile_path, year in all_tile_jobs
        }
        for future in as_completed(futures):
            tile_path, year = futures[future]
            try:
                result = future.result()
                if result["status"] == "completed":
                    completed += 1
                else:
                    failed += 1
            except Exception as exc:
                logger.error("Tile %s crashed: %s", Path(tile_path).stem, exc)
                failed += 1

            if (completed + failed) % 20 == 0:
                logger.info("Progress: %d/%d completed, %d failed", completed, len(all_tile_jobs), failed)

    logger.info("DONE: %d completed, %d failed out of %d tiles", completed, failed, len(all_tile_jobs))


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
            print(f"# Year {year_str}: {result.get('status', 'unknown')} -- skipping")
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
        print(f"  --slurm_account nceo_isp &")
        print()
    print("wait  # wait for all years to finish")


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

        batch_payload = json.loads(Path(batch_manifest).read_text(encoding="utf-8"))
        tile_manifests = batch_payload.get("tile_manifest_paths", [])

        completed_tiles = 0
        models_found = 0
        predictions_found = 0
        metrics_data = []

        for tile_path in tile_manifests:
            tile_payload = json.loads(Path(tile_path).read_text(encoding="utf-8"))
            tile_id = tile_payload.get("tile_id", "unknown")
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

    pooled_dir = output_root / "training" / "pooled"
    if pooled_dir.exists():
        pooled_metrics = list(pooled_dir.rglob("metrics.json"))
        print(f"\n  Pooled models: {len(pooled_metrics)}")
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
    group.add_argument("--prepare", action="store_true", help="Download all years in parallel and prepare batches")
    group.add_argument("--run-tiles", action="store_true", help="Run pipeline on all prepared tiles")
    group.add_argument("--nextflow-commands", action="store_true", help="Emit Nextflow commands")
    group.add_argument("--check-results", action="store_true", help="Check and summarize completed results")
    parser.add_argument("--tile-workers", type=int, default=4, help="Parallel tile pipeline workers for --run-tiles")
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else get_output_root()
    os.environ["CROME_DATA_ROOT"] = str(output_root)

    if args.dry_run:
        dry_run(output_root)
    elif args.prepare:
        prepare_all(output_root)
    elif args.run_tiles:
        run_tiles(output_root, n_tile_workers=args.tile_workers)
    elif args.nextflow_commands:
        emit_nextflow_commands(output_root)
    elif args.check_results:
        check_results(output_root)


if __name__ == "__main__":
    main()
