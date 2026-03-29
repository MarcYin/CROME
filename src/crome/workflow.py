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
from .config import AlphaEarthDownloadRequest
from .pipeline import BaselinePipelineResult, run_baseline_pipeline


@dataclass(frozen=True, slots=True)
class DownloadBaselineResult:
    """Combined result for a download plus baseline-pipeline run."""

    download: AlphaEarthDownloadResult
    pipeline: BaselinePipelineResult


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


def download_and_run_baseline(
    *,
    year: int,
    output_root: Path | str,
    reference_path: Path | str,
    aoi_label: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    geojson: Path | str | None = None,
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
    pipeline = run_baseline_pipeline(
        feature_input=download.output_root,
        manifest_path=download.manifest_path,
        reference_path=reference_path,
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
    return DownloadBaselineResult(download=download, pipeline=pipeline)


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
        default="data/alphaearth",
        help="Base directory for AlphaEarth outputs and downstream artifacts.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="AOI bbox in lon/lat coordinates.",
    )
    parser.add_argument("--geojson", default=None, help="Path to an AOI GeoJSON file.")
    parser.add_argument("--reference-path", required=True, help="Path to the CROME vector reference file.")
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
        payload = {
            "download": request.to_dict(),
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
                "reference_path": str(Path(args.reference_path)),
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
        "pipeline": _pipeline_result_to_dict(result.pipeline),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
