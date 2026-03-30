"""Operator-facing workflow wrappers that combine download and baseline execution."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .acquisition.alphaearth import (
    AlphaEarthDownloadResult,
    download_alphaearth_images,
    download_result_to_dict,
)
from .acquisition.crome import download_crome_reference
from .config import AlphaEarthDownloadRequest, CromeDownloadRequest
from .paths import OUTPUT_ROOT_ENV_VAR, default_output_root
from .pipeline import BaselinePipelineResult, run_baseline_pipeline


@dataclass(frozen=True, slots=True)
class DownloadBaselineResult:
    """Combined result for a download plus baseline-pipeline run."""

    download: AlphaEarthDownloadResult
    pipeline: BaselinePipelineResult
    reference_download: Any | None = None


def _warn_if_conditional_year(request: AlphaEarthDownloadRequest) -> None:
    if request.conditional_year:
        print(
            "Warning: 2025 AlphaEarth coverage is documented as rolling by UTM zone as of "
            "January 29, 2026.",
            file=sys.stderr,
        )


def _pipeline_result_to_dict(result: BaselinePipelineResult) -> dict[str, Any]:
    return {
        "feature_count": len(result.feature_results),
        "manifest_path": str(result.manifest_path) if result.manifest_path is not None else None,
        "metrics_path": str(result.metrics_path),
        "model_path": str(result.model_path),
        "pipeline_manifest_path": str(result.pipeline_manifest_path),
        "skipped_feature_count": len(result.skipped_features),
        "training_metadata_path": str(result.training_metadata_path),
        "training_table_path": str(result.training_table_path),
    }


def _reference_result_to_dict(result: Any | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "archive_path": str(result.archive_path),
        "archive_url": result.archive_url,
        "dataset_id": result.dataset_id,
        "extracted_path": str(result.extracted_path) if result.extracted_path is not None else None,
        "landing_page_url": result.landing_page_url,
        "manifest_path": str(result.manifest_path),
        "output_root": str(result.output_root),
        "title": result.title,
        "variant": result.variant,
        "year": result.year,
    }


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
    overlap_policy: str = "error",
    all_touched: bool = False,
    nodata_label: int = -1,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
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
    download = download_alphaearth_images(request)
    resolved_reference_path = Path(reference_path) if reference_path is not None else None
    reference_download = None
    if download_reference or resolved_reference_path is None:
        reference_download = download_crome_reference(
            CromeDownloadRequest(
                year=year,
                output_root=output_root,
                prefer_complete=prefer_complete_reference,
                extract=True,
                force=force_reference_download,
            )
        )
        resolved_reference_path = reference_download.extracted_path
    if resolved_reference_path is None:
        raise ValueError("A CROME reference path could not be resolved.")
    pipeline = run_baseline_pipeline(
        feature_input=download.output_root,
        manifest_path=download.manifest_path,
        reference_path=resolved_reference_path,
        year=request.year,
        output_root=request.output_root,
        aoi_label=request.aoi_label,
        label_column=label_column,
        geometry_column=geometry_column,
        overlap_policy=overlap_policy,
        all_touched=all_touched,
        nodata_label=nodata_label,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        predict=predict,
        skip_empty_labels=skip_empty_labels,
    )
    return DownloadBaselineResult(
        download=download,
        pipeline=pipeline,
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
        help="Optional run label used only in local naming. This is not an AlphaEarth tile identifier.",
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
    parser.add_argument(
        "--reference-path",
        default=None,
        help="Path to an existing CROME vector reference file. If omitted, the workflow downloads CROME.",
    )
    parser.add_argument(
        "--download-crome-reference",
        action="store_true",
        help="Download the CROME reference even when --reference-path is supplied.",
    )
    parser.add_argument(
        "--prefer-plain-crome",
        action="store_true",
        help="Prefer plain-year CROME titles over '- Complete' titles when auto-downloading references.",
    )
    parser.add_argument(
        "--force-crome-download",
        action="store_true",
        help="Re-download the CROME archive when auto-downloading references.",
    )
    parser.add_argument("--label-column", default="lucode", help="Reference class column.")
    parser.add_argument("--geometry-column", default="geometry", help="Reference geometry column.")
    parser.add_argument(
        "--overlap-policy",
        choices=("error", "first", "last"),
        default="error",
        help="Policy for overlapping reference polygons.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Rasterize with all_touched=True instead of pixel-center semantics.",
    )
    parser.add_argument("--nodata-label", type=int, default=-1, help="Output nodata label id.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Evaluation holdout fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Random forest tree count.")
    parser.add_argument(
        "--fail-on-empty-labels",
        action="store_true",
        help="Fail instead of skipping feature rasters that have no usable CROME coverage.",
    )
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="Stop after training without emitting prediction rasters.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved download and baseline request without importing edown or running the model.",
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
                "fail_on_empty_labels": args.fail_on_empty_labels,
                "geometry_column": args.geometry_column,
                "label_column": args.label_column,
                "n_estimators": args.n_estimators,
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
        overlap_policy=args.overlap_policy,
        all_touched=args.all_touched,
        nodata_label=args.nodata_label,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        predict=not args.no_predict,
        skip_empty_labels=not args.fail_on_empty_labels,
    )
    payload = {
        "download": download_result_to_dict(result.download),
        "reference_download": _reference_result_to_dict(result.reference_download),
        "pipeline": _pipeline_result_to_dict(result.pipeline),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
