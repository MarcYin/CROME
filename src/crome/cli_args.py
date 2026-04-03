"""Shared CLI argument group builders to avoid duplication across subcommands."""

from __future__ import annotations

import argparse


def add_reference_args(parser: argparse.ArgumentParser) -> None:
    """Add reference/label-transfer arguments shared by pipeline, orchestration, and workflow."""
    parser.add_argument("--label-column", default="lucode", help="Reference class column.")
    parser.add_argument("--geometry-column", default="geometry", help="Reference geometry column.")
    parser.add_argument(
        "--label-mode",
        choices=("centroid_to_pixel", "polygon_to_pixel"),
        default="centroid_to_pixel",
        help="How CROME vector labels are transferred onto the AlphaEarth grid.",
    )
    parser.add_argument(
        "--overlap-policy",
        choices=("error", "first", "last"),
        default="first",
        help="Policy for overlapping reference polygons.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Rasterize with all_touched=True instead of pixel-center semantics.",
    )
    parser.add_argument("--nodata-label", type=int, default=-1, help="Output nodata label id.")


def add_training_args(
    parser: argparse.ArgumentParser,
    *,
    n_estimators_default: int = 200,
) -> None:
    """Add model-training arguments shared by pipeline, orchestration, and workflow."""
    parser.add_argument("--test-size", type=float, default=0.2, help="Evaluation holdout fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--n-estimators", type=int, default=n_estimators_default, help="Random forest tree count."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="CPU parallelism passed to RandomForestClassifier.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap on training rows after holdout splitting.",
    )


def add_pipeline_behavior_args(parser: argparse.ArgumentParser) -> None:
    """Add pipeline-behaviour flags shared by pipeline, orchestration, and workflow."""
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


def add_crome_download_args(parser: argparse.ArgumentParser) -> None:
    """Add CROME reference download arguments shared by workflow subcommands."""
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
