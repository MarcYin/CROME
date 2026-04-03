"""Operator-facing workflow wrappers that combine download and baseline execution."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from .acquisition.alphaearth import (
    AlphaEarthDownloadResult,
    download_alphaearth_images,
    download_result_to_dict,
)
from .acquisition.crome import (
    CromeDownloadResult,
    CromeReferenceFootprint,
    download_crome_reference,
    reference_footprint,
)
from .cli_args import (
    add_crome_download_args,
    add_pipeline_behavior_args,
    add_reference_args,
    add_training_args,
)
from .config import AlphaEarthDownloadRequest, CromeDownloadRequest
from .orchestration import PreparedTileBatchResult, prepare_tile_batch
from .paths import OUTPUT_ROOT_ENV_VAR, default_output_root
from .pipeline import BaselinePipelineResult, run_baseline_pipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DownloadBaselineResult:
    """Combined result for a download plus baseline-pipeline run."""

    download: AlphaEarthDownloadResult
    pipeline: BaselinePipelineResult
    reference_download: CromeDownloadResult | None = None


@dataclass(frozen=True, slots=True)
class PrepareFootprintTileBatchResult:
    """Combined result for a footprint-driven AlphaEarth download and tile-batch prep."""

    batch: PreparedTileBatchResult
    download: AlphaEarthDownloadResult
    footprint: CromeReferenceFootprint
    reference_download: CromeDownloadResult | None = None


def _warn_if_conditional_year(request: AlphaEarthDownloadRequest) -> None:
    if request.conditional_year:
        logger.warning(
            "2025 AlphaEarth coverage is documented as rolling by UTM zone as of January 29, 2026."
        )
        print(
            "Warning: 2025 AlphaEarth coverage is documented as rolling by UTM zone as of "
            "January 29, 2026.",
            file=sys.stderr,
        )


def _reference_result_to_dict(result: CromeDownloadResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "archive_path": str(result.archive_path),
        "archive_url": result.archive_url,
        "dataset_id": result.dataset_id,
        "extracted_path": str(result.extracted_path) if result.extracted_path is not None else None,
        "landing_page_url": result.landing_page_url,
        "manifest_path": str(result.manifest_path),
        "normalized_path": str(result.normalized_path) if result.normalized_path is not None else None,
        "output_root": str(result.output_root),
        "reference_path": str(result.reference_path) if result.reference_path is not None else None,
        "source_layer": result.source_layer,
        "title": result.title,
        "variant": result.variant,
        "year": result.year,
    }


def _reference_footprint_to_dict(result: CromeReferenceFootprint) -> dict[str, object]:
    return {
        "bounds": [float(value) for value in result.bounds],
        "bounds_lonlat": [float(value) for value in result.bounds_lonlat],
        "crs": result.crs,
        "reference_path": str(result.reference_path),
        "source_layer": result.source_layer,
        "year": result.year,
    }


def _pipeline_result_to_dict_fallback(result: object) -> dict[str, object]:
    """Fallback serializer for pipeline results lacking to_dict() (e.g. test mocks)."""
    features = getattr(result, "feature_results", ())
    return {
        "feature_count": len(features),
        "features": [
            f.to_dict() if hasattr(f, "to_dict") else {
                "feature_id": getattr(f, "feature_id", None),
                "feature_raster_path": str(getattr(f, "feature_raster_path", "")),
                "label_mapping_path": str(getattr(f, "label_mapping_path", "")),
                "label_raster_path": str(getattr(f, "label_raster_path", "")),
                "metrics_path": str(getattr(f, "metrics_path", "")),
                "model_path": str(getattr(f, "model_path", "")),
                "prediction_metadata_path": (
                    str(f.prediction_metadata_path) if getattr(f, "prediction_metadata_path", None) else None
                ),
                "prediction_output_root": (
                    str(f.prediction_output_root) if getattr(f, "prediction_output_root", None) else None
                ),
                "prediction_raster_path": (
                    str(f.prediction_raster_path) if getattr(f, "prediction_raster_path", None) else None
                ),
                "qc_manifest_path": str(getattr(f, "qc_manifest_path", "")),
                "sample_cache_manifest_path": (
                    str(f.sample_cache_manifest_path) if getattr(f, "sample_cache_manifest_path", None) else None
                ),
                "sample_cache_root": (
                    str(f.sample_cache_root) if getattr(f, "sample_cache_root", None) else None
                ),
                "source_image_id": getattr(f, "source_image_id", None),
                "tile_id": getattr(f, "tile_id", None),
                "training_metadata_path": str(getattr(f, "training_metadata_path", "")),
                "training_output_root": str(getattr(f, "training_output_root", "")),
                "training_table_path": str(getattr(f, "training_table_path", "")),
            }
            for f in features
        ],
        "manifest_path": (
            str(result.manifest_path) if getattr(result, "manifest_path", None) else None
        ),
        "pipeline_manifest_path": str(getattr(result, "pipeline_manifest_path", "")),
        "qc_manifest_path": str(getattr(result, "qc_manifest_path", "")),
        "reference_input_path": (
            str(result.reference_input_path) if getattr(result, "reference_input_path", None) else None
        ),
        "reference_manifest_path": (
            str(result.reference_manifest_path) if getattr(result, "reference_manifest_path", None) else None
        ),
        "reference_path": (
            str(result.reference_path) if getattr(result, "reference_path", None) else None
        ),
        "sample_cache_root": (
            str(result.sample_cache_root) if getattr(result, "sample_cache_root", None) else None
        ),
        "skipped_feature_count": len(getattr(result, "skipped_features", ())),
    }


def _should_retry_with_extracted_reference(exc: Exception) -> bool:
    message = str(exc)
    return any(
        token in message
        for token in (
            "Reference source does not expose any non-null labels.",
            "not recognized as being in a supported file format",
            "No vector layers were found",
        )
    )


def _resolve_crome_reference(
    *,
    reference_path: Path | str | None,
    download_reference: bool,
    year: int,
    output_root: Path | str,
    prefer_complete: bool,
    force_download: bool,
) -> tuple[Path, CromeDownloadResult | None]:
    """Resolve a CROME reference path, downloading if needed."""
    resolved = Path(reference_path) if reference_path is not None else None
    download_result = None
    if download_reference or resolved is None:
        download_result = download_crome_reference(
            CromeDownloadRequest(
                year=year,
                output_root=output_root,
                prefer_complete=prefer_complete,
                extract=True,
                force=force_download,
            )
        )
        resolved = download_result.reference_path
    if resolved is None:
        raise ValueError("A CROME reference path could not be resolved.")
    return resolved, download_result


def download_and_run_baseline(
    *,
    year: int,
    output_root: Path | str,
    reference_path: Path | str | None,
    aoi_label: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    geojson: Path | str | None = None,
    download_reference: bool = False,
    prefer_complete_reference: bool = True,
    force_reference_download: bool = False,
    label_column: str = "lucode",
    geometry_column: str = "geometry",
    label_mode: str = "centroid_to_pixel",
    overlap_policy: str = "first",
    all_touched: bool = False,
    nodata_label: int = -1,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    n_jobs: int = -1,
    max_train_rows: int | None = None,
    predict: bool = True,
    skip_empty_labels: bool = True,
) -> DownloadBaselineResult:
    """Download AlphaEarth features and immediately run the baseline pipeline."""

    request = AlphaEarthDownloadRequest(
        year=year,
        output_root=output_root,
        aoi_label=aoi_label,
        bbox=bbox,
        geojson=geojson,
    )
    logger.info("Downloading AlphaEarth imagery for year %d", year)
    download = download_alphaearth_images(request)

    resolved_reference_path, reference_download = _resolve_crome_reference(
        reference_path=reference_path,
        download_reference=download_reference,
        year=year,
        output_root=output_root,
        prefer_complete=prefer_complete_reference,
        force_download=force_reference_download,
    )

    pipeline_kwargs = {
        "feature_input": download.output_root,
        "manifest_path": download.manifest_path,
        "year": request.year,
        "output_root": request.output_root,
        "aoi_label": request.aoi_label,
        "label_column": label_column,
        "geometry_column": geometry_column,
        "label_mode": label_mode,
        "overlap_policy": overlap_policy,
        "all_touched": all_touched,
        "nodata_label": nodata_label,
        "test_size": test_size,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "n_jobs": n_jobs,
        "max_train_rows": max_train_rows,
        "predict": predict,
        "skip_empty_labels": skip_empty_labels,
    }
    try:
        pipeline = run_baseline_pipeline(
            reference_path=resolved_reference_path,
            **pipeline_kwargs,
        )
    except Exception as exc:
        extracted_reference_path = (
            reference_download.extracted_path if reference_download is not None else None
        )
        should_retry = (
            extracted_reference_path is not None
            and resolved_reference_path != extracted_reference_path
            and _should_retry_with_extracted_reference(exc)
        )
        if not should_retry:
            raise
        logger.warning("Retrying pipeline with extracted reference path: %s", extracted_reference_path)
        pipeline = run_baseline_pipeline(
            reference_path=extracted_reference_path,
            **pipeline_kwargs,
        )
    return DownloadBaselineResult(
        download=download,
        pipeline=pipeline,
        reference_download=reference_download,
    )


def prepare_footprint_tile_batch(
    *,
    year: int,
    output_root: Path | str,
    reference_path: Path | str | None,
    aoi_label: str | None = None,
    download_reference: bool = False,
    prefer_complete_reference: bool = True,
    force_reference_download: bool = False,
    reference_download_result: CromeDownloadResult | None = None,
    label_column: str = "lucode",
    geometry_column: str = "geometry",
    label_mode: str = "centroid_to_pixel",
    overlap_policy: str = "first",
    all_touched: bool = False,
    nodata_label: int = -1,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 400,
    n_jobs: int = -1,
    max_train_rows: int | None = None,
    predict: bool = True,
    skip_empty_labels: bool = True,
) -> PrepareFootprintTileBatchResult:
    """Resolve one full CROME footprint, download intersecting AlphaEarth tiles, and prepare a tile batch."""

    resolved_output_root = Path(output_root)

    if reference_download_result is not None:
        resolved_reference_path = reference_download_result.reference_path
        reference_download = reference_download_result
    else:
        resolved_reference_path, reference_download = _resolve_crome_reference(
            reference_path=reference_path,
            download_reference=download_reference,
            year=year,
            output_root=resolved_output_root,
            prefer_complete=prefer_complete_reference,
            force_download=force_reference_download,
        )

    if resolved_reference_path is None:
        raise ValueError("A CROME reference path could not be resolved.")

    footprint = reference_footprint(resolved_reference_path, year=year)
    footprint_label = aoi_label or "england-crome-footprint"
    request = AlphaEarthDownloadRequest(
        year=year,
        output_root=resolved_output_root,
        aoi_label=footprint_label,
        bbox=footprint.bounds_lonlat,
    )
    _warn_if_conditional_year(request)
    logger.info("Downloading AlphaEarth tiles for CROME footprint (year %d)", year)
    download = download_alphaearth_images(request)
    batch = prepare_tile_batch(
        feature_input=download.output_root,
        manifest_path=download.manifest_path,
        reference_path=resolved_reference_path,
        year=year,
        output_root=resolved_output_root,
        aoi_label=request.aoi_label,
        label_column=label_column,
        geometry_column=geometry_column,
        label_mode=label_mode,
        overlap_policy=overlap_policy,
        all_touched=all_touched,
        nodata_label=nodata_label,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        max_train_rows=max_train_rows,
        predict=predict,
        skip_empty_labels=skip_empty_labels,
    )
    return PrepareFootprintTileBatchResult(
        batch=batch,
        download=download,
        footprint=footprint,
        reference_download=reference_download,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download AlphaEarth annual embeddings and run the baseline pipeline."
    )
    parser.add_argument("--year", required=True, type=int, help="Target annual layer year.")
    parser.add_argument(
        "--aoi-label",
        default=None,
        help=(
            "Optional download or batch-summary label. Downstream labels, training artifacts, models, "
            "and predictions are keyed by discovered AlphaEarth feature tiles."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=default_output_root(),
        help=(
            "Base directory for AlphaEarth outputs and downstream artifacts. "
            f"Defaults to ${OUTPUT_ROOT_ENV_VAR} when set, otherwise data/alphaearth."
        ),
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="AOI bbox in lon/lat coordinates.",
    )
    parser.add_argument("--geojson", default=None, help="Path to an AOI GeoJSON file.")
    add_crome_download_args(parser)
    add_reference_args(parser)
    add_training_args(parser)
    add_pipeline_behavior_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved download and baseline request without importing edown or running the model.",
    )
    return parser


def build_prepare_footprint_tile_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the full CROME classification footprint for one year, download the "
            "intersecting AlphaEarth tiles, and prepare a per-tile batch."
        )
    )
    parser.add_argument("--year", required=True, type=int, help="Target annual layer year.")
    parser.add_argument(
        "--aoi-label",
        default="england-crome-footprint",
        help=(
            "Optional batch label. The runtime still operates per discovered AlphaEarth tile, "
            "but this label namespaces the per-year batch."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=default_output_root(),
        help=(
            "Base directory for AlphaEarth outputs and downstream artifacts. "
            f"Defaults to ${OUTPUT_ROOT_ENV_VAR} when set, otherwise data/alphaearth."
        ),
    )
    add_crome_download_args(parser)
    add_reference_args(parser)
    add_training_args(parser, n_estimators_default=400)
    add_pipeline_behavior_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved CROME footprint, AlphaEarth download request, and batch config without downloading.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    request = AlphaEarthDownloadRequest(
        year=args.year,
        output_root=args.output_root,
        aoi_label=args.aoi_label,
        bbox=tuple(args.bbox) if args.bbox is not None else None,
        geojson=args.geojson,
    )
    _warn_if_conditional_year(request)

    if args.dry_run:
        should_download_reference = args.download_crome_reference or args.reference_path is None
        payload = {
            "download": request.to_dict(),
            "reference_download": (
                CromeDownloadRequest(
                    year=request.year,
                    output_root=request.output_root,
                    prefer_complete=not args.prefer_plain_crome,
                    extract=True,
                    force=args.force_crome_download,
                ).to_dict()
                if should_download_reference
                else None
            ),
            "pipeline": {
                "all_touched": args.all_touched,
                "aoi_label": request.aoi_label,
                "artifact_unit": "alphaearth_feature_tile",
                "fail_on_empty_labels": args.fail_on_empty_labels,
                "geometry_column": args.geometry_column,
                "label_mode": args.label_mode,
                "label_column": args.label_column,
                "n_estimators": args.n_estimators,
                "n_jobs": args.n_jobs,
                "max_train_rows": args.max_train_rows,
                "no_predict": args.no_predict,
                "nodata_label": args.nodata_label,
                "output_root": str(request.output_root),
                "overlap_policy": args.overlap_policy,
                "random_state": args.random_state,
                "reference_path": str(Path(args.reference_path)) if args.reference_path else None,
                "test_size": args.test_size,
                "year": request.year,
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    result = download_and_run_baseline(
        year=request.year,
        output_root=request.output_root,
        reference_path=args.reference_path,
        aoi_label=request.aoi_label,
        bbox=request.bbox,
        geojson=request.geojson,
        download_reference=args.download_crome_reference or args.reference_path is None,
        prefer_complete_reference=not args.prefer_plain_crome,
        force_reference_download=args.force_crome_download,
        label_column=args.label_column,
        geometry_column=args.geometry_column,
        label_mode=args.label_mode,
        overlap_policy=args.overlap_policy,
        all_touched=args.all_touched,
        nodata_label=args.nodata_label,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        max_train_rows=args.max_train_rows,
        predict=not args.no_predict,
        skip_empty_labels=not args.fail_on_empty_labels,
    )
    pipeline_dict = (
        result.pipeline.to_dict()
        if hasattr(result.pipeline, "to_dict")
        else _pipeline_result_to_dict_fallback(result.pipeline)
    )
    payload = {
        "download": download_result_to_dict(result.download),
        "reference_download": _reference_result_to_dict(result.reference_download),
        "pipeline": pipeline_dict,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main_prepare_footprint_tile_batch(argv: list[str] | None = None) -> int:
    parser = build_prepare_footprint_tile_batch_parser()
    args = parser.parse_args(argv)
    resolved_output_root = Path(args.output_root)

    resolved_reference_path, reference_download = _resolve_crome_reference(
        reference_path=args.reference_path,
        download_reference=args.download_crome_reference or args.reference_path is None,
        year=args.year,
        output_root=resolved_output_root,
        prefer_complete=not args.prefer_plain_crome,
        force_download=args.force_crome_download,
    )

    footprint = reference_footprint(resolved_reference_path, year=args.year)
    request = AlphaEarthDownloadRequest(
        year=args.year,
        output_root=resolved_output_root,
        aoi_label=args.aoi_label,
        bbox=footprint.bounds_lonlat,
    )
    _warn_if_conditional_year(request)

    if args.dry_run:
        payload = {
            "download": request.to_dict(),
            "footprint": _reference_footprint_to_dict(footprint),
            "reference_download": _reference_result_to_dict(reference_download),
            "batch": {
                "all_touched": args.all_touched,
                "aoi_label": request.aoi_label,
                "artifact_unit": "alphaearth_feature_tile",
                "fail_on_empty_labels": args.fail_on_empty_labels,
                "geometry_column": args.geometry_column,
                "label_column": args.label_column,
                "label_mode": args.label_mode,
                "max_train_rows": args.max_train_rows,
                "n_estimators": args.n_estimators,
                "n_jobs": args.n_jobs,
                "no_predict": args.no_predict,
                "nodata_label": args.nodata_label,
                "output_root": str(request.output_root),
                "overlap_policy": args.overlap_policy,
                "random_state": args.random_state,
                "reference_path": str(resolved_reference_path),
                "test_size": args.test_size,
                "year": args.year,
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    result = prepare_footprint_tile_batch(
        year=args.year,
        output_root=resolved_output_root,
        reference_path=resolved_reference_path,
        aoi_label=request.aoi_label,
        download_reference=False,
        prefer_complete_reference=not args.prefer_plain_crome,
        force_reference_download=args.force_crome_download,
        reference_download_result=reference_download,
        label_column=args.label_column,
        geometry_column=args.geometry_column,
        label_mode=args.label_mode,
        overlap_policy=args.overlap_policy,
        all_touched=args.all_touched,
        nodata_label=args.nodata_label,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        max_train_rows=args.max_train_rows,
        predict=not args.no_predict,
        skip_empty_labels=not args.fail_on_empty_labels,
    )
    payload = {
        "batch": {
            "batch_label": result.batch.batch_label,
            "batch_manifest_path": str(result.batch.batch_manifest_path),
            "output_root": str(result.batch.output_root),
            "pooled_output_dir": str(result.batch.pooled_output_dir),
            "reference_input_path": str(result.batch.reference_input_path),
            "reference_manifest_path": (
                str(result.batch.reference_manifest_path)
                if result.batch.reference_manifest_path is not None
                else None
            ),
            "reference_path": str(result.batch.reference_path),
            "tile_count": len(result.batch.tile_manifest_paths),
            "tile_manifest_paths": [str(path) for path in result.batch.tile_manifest_paths],
            "workflow_output_root": str(result.batch.workflow_output_root),
            "year": result.batch.year,
        },
        "download": download_result_to_dict(result.download),
        "footprint": _reference_footprint_to_dict(result.footprint),
        "reference_download": _reference_result_to_dict(result.reference_download),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
