"""Cluster-friendly orchestration helpers for per-tile batch execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .acquisition.crome import materialize_crome_reference_subset
from .cli_args import add_pipeline_behavior_args, add_reference_args, add_training_args
from .discovery import discover_feature_rasters
from .manifest import find_reference_manifest
from .paths import (
    feature_tile_name,
    pooled_training_output_root,
    sanitize_label,
    workflow_batch_output_root,
)
from .pipeline import BaselinePipelineResult, run_baseline_pipeline
from .training import PooledTrainingResult, train_pooled_model_from_pipeline_manifests

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedTileBatchResult:
    """Materialized plan for one cluster-parallel tile batch."""

    batch_label: str
    batch_manifest_path: Path
    output_root: Path
    pooled_output_dir: Path
    reference_input_path: Path
    reference_manifest_path: Path | None
    reference_path: Path
    tile_manifest_paths: tuple[Path, ...]
    workflow_output_root: Path
    year: int


@dataclass(frozen=True, slots=True)
class TilePlanRunResult:
    """Outputs from executing one prepared tile plan."""

    batch_manifest_path: Path
    pipeline: BaselinePipelineResult
    tile_plan_path: Path


def _default_batch_label(
    *,
    feature_input: Path | str | None,
    manifest_path: Path | str | None,
    aoi_label: str | None,
) -> str:
    if aoi_label is not None and aoi_label.strip():
        return sanitize_label(aoi_label, default="batch")
    if feature_input is not None:
        path = Path(feature_input)
        label = path.stem if path.is_file() else path.name
        return sanitize_label(label, default="batch")
    if manifest_path is not None:
        return sanitize_label(Path(manifest_path).parent.name, default="batch")
    return "batch"


def _workflow_namespace(payload: dict[str, object]) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"tile_batch_{digest}"


def _load_json_payload(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def prepare_tile_batch(
    *,
    feature_input: Path | str | None,
    manifest_path: Path | str | None,
    reference_path: Path | str,
    year: int,
    output_root: Path | str,
    aoi_label: str | None = None,
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
) -> PreparedTileBatchResult:
    """Prepare one batch manifest plus per-tile manifests for cluster execution."""

    if feature_input is None and manifest_path is None:
        raise ValueError("Provide at least one of feature_input or manifest_path.")

    resolved_feature_input = Path(feature_input) if feature_input is not None else None
    resolved_manifest_path = Path(manifest_path) if manifest_path is not None else None
    resolved_reference_input = Path(reference_path)
    resolved_output_root = Path(output_root)
    batch_label = _default_batch_label(
        feature_input=resolved_feature_input,
        manifest_path=resolved_manifest_path,
        aoi_label=aoi_label,
    )

    discovered = discover_feature_rasters(
        feature_input=resolved_feature_input,
        manifest_path=resolved_manifest_path,
        requested_year=year,
    )
    logger.info("Discovered %d feature raster(s) for batch '%s'", len(discovered), batch_label)

    resolved_reference_path = materialize_crome_reference_subset(
        resolved_reference_input,
        feature_raster_paths=[feature.raster_path for feature in discovered],
        subset_label=batch_label,
        year=year,
    )
    reference_manifest_path = find_reference_manifest(resolved_reference_path)

    namespace = _workflow_namespace(
        {
            "all_touched": all_touched,
            "feature_input": str(resolved_feature_input) if resolved_feature_input is not None else None,
            "feature_rasters": [str(feature.raster_path.resolve()) for feature in discovered],
            "geometry_column": geometry_column,
            "label_column": label_column,
            "label_mode": label_mode,
            "manifest_path": str(resolved_manifest_path) if resolved_manifest_path is not None else None,
            "max_train_rows": max_train_rows,
            "n_estimators": n_estimators,
            "n_jobs": n_jobs,
            "nodata_label": nodata_label,
            "output_root": str(resolved_output_root.resolve()),
            "overlap_policy": overlap_policy,
            "predict": predict,
            "random_state": random_state,
            "reference_input_path": str(resolved_reference_input.resolve()),
            "reference_path": str(Path(resolved_reference_path).resolve()),
            "skip_empty_labels": skip_empty_labels,
            "test_size": test_size,
            "year": year,
        }
    )
    workflow_output_root = workflow_batch_output_root(
        resolved_output_root,
        batch_label,
        year,
        namespace=namespace,
    )
    tile_manifest_root = workflow_output_root / "tiles"
    tile_manifest_root.mkdir(parents=True, exist_ok=True)

    pooled_output_dir = pooled_training_output_root(
        resolved_output_root,
        batch_label,
        year,
        namespace=namespace,
    )

    tile_manifest_paths: list[Path] = []
    tile_payloads: list[dict[str, object]] = []
    for feature in discovered:
        tile_id = feature_tile_name(
            feature_id=feature.feature_id,
            source_image_id=feature.source_image_id,
            feature_raster_path=feature.raster_path,
        )
        tile_run_label = f"{batch_label}_{tile_id}"
        payload = {
            "all_touched": all_touched,
            "batch_label": batch_label,
            "batch_manifest_path": str(workflow_output_root / "batch_manifest.json"),
            "feature_id": feature.feature_id,
            "feature_raster_path": str(feature.raster_path),
            "geometry_column": geometry_column,
            "label_column": label_column,
            "label_mode": label_mode,
            "manifest_path": str(resolved_manifest_path) if resolved_manifest_path is not None else None,
            "n_estimators": n_estimators,
            "n_jobs": n_jobs,
            "nodata_label": nodata_label,
            "output_root": str(resolved_output_root),
            "overlap_policy": overlap_policy,
            "predict": predict,
            "random_state": random_state,
            "reference_input_path": str(resolved_reference_input),
            "reference_manifest_path": (
                str(reference_manifest_path) if reference_manifest_path is not None else None
            ),
            "reference_path": str(resolved_reference_path),
            "skip_empty_labels": skip_empty_labels,
            "source_image_id": feature.source_image_id,
            "test_size": test_size,
            "tile_id": tile_id,
            "tile_run_label": tile_run_label,
            "workflow_namespace": namespace,
            "year": year,
        }
        tile_manifest_path = tile_manifest_root / f"{tile_id}.json"
        tile_manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tile_manifest_paths.append(tile_manifest_path)
        tile_payloads.append(
            {
                "feature_id": feature.feature_id,
                "feature_raster_path": str(feature.raster_path),
                "source_image_id": feature.source_image_id,
                "tile_id": tile_id,
                "tile_manifest_path": str(tile_manifest_path),
                "tile_run_label": tile_run_label,
            }
        )

    batch_manifest_path = workflow_output_root / "batch_manifest.json"
    batch_manifest_path.write_text(
        json.dumps(
            {
                "aoi_label": aoi_label,
                "batch_label": batch_label,
                "feature_input": (
                    str(resolved_feature_input) if resolved_feature_input is not None else None
                ),
                "manifest_path": (
                    str(resolved_manifest_path) if resolved_manifest_path is not None else None
                ),
                "pooled_model": {
                    "max_train_rows": max_train_rows,
                    "n_estimators": n_estimators,
                    "n_jobs": n_jobs,
                    "output_dir": str(pooled_output_dir),
                    "random_state": random_state,
                    "test_size": test_size,
                },
                "reference_input_path": str(resolved_reference_input),
                "reference_manifest_path": (
                    str(reference_manifest_path) if reference_manifest_path is not None else None
                ),
                "reference_path": str(resolved_reference_path),
                "tile_count": len(tile_payloads),
                "tile_manifest_paths": [str(path) for path in tile_manifest_paths],
                "tiles": tile_payloads,
                "workflow_namespace": namespace,
                "workflow_output_root": str(workflow_output_root),
                "year": year,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    logger.info("Prepared %d tile plan(s) for batch '%s'", len(tile_manifest_paths), batch_label)
    return PreparedTileBatchResult(
        batch_label=batch_label,
        batch_manifest_path=batch_manifest_path,
        output_root=resolved_output_root,
        pooled_output_dir=pooled_output_dir,
        reference_input_path=resolved_reference_input,
        reference_manifest_path=reference_manifest_path,
        reference_path=Path(resolved_reference_path),
        tile_manifest_paths=tuple(tile_manifest_paths),
        workflow_output_root=workflow_output_root,
        year=year,
    )


def run_tile_plan(tile_plan_path: Path | str, *, n_jobs_override: int | None = None) -> TilePlanRunResult:
    """Execute one prepared per-tile plan."""

    resolved_tile_plan_path = Path(tile_plan_path)
    payload = _load_json_payload(resolved_tile_plan_path)
    resolved_n_jobs = (
        int(n_jobs_override)
        if n_jobs_override is not None
        else int(payload.get("n_jobs", -1))
    )
    batch_manifest_path = Path(str(payload["batch_manifest_path"]))
    logger.info("Running tile plan %s", resolved_tile_plan_path.name)
    result = run_baseline_pipeline(
        feature_input=payload["feature_raster_path"],
        manifest_path=None,
        reference_path=payload["reference_path"],
        year=int(payload["year"]),
        output_root=payload["output_root"],
        aoi_label=payload["tile_run_label"],
        label_column=str(payload["label_column"]),
        geometry_column=str(payload["geometry_column"]),
        label_mode=str(payload["label_mode"]),
        overlap_policy=str(payload["overlap_policy"]),
        all_touched=bool(payload["all_touched"]),
        nodata_label=int(payload["nodata_label"]),
        test_size=float(payload["test_size"]),
        random_state=int(payload["random_state"]),
        n_estimators=int(payload["n_estimators"]),
        n_jobs=resolved_n_jobs,
        max_train_rows=(
            int(payload["max_train_rows"])
            if payload.get("max_train_rows") is not None
            else None
        ),
        predict=bool(payload["predict"]),
        skip_empty_labels=bool(payload["skip_empty_labels"]),
    )
    return TilePlanRunResult(
        batch_manifest_path=batch_manifest_path,
        pipeline=result,
        tile_plan_path=resolved_tile_plan_path,
    )


def train_pooled_from_tile_results(
    *,
    batch_manifest_path: Path | str,
    tile_result_paths: list[Path | str],
    output_dir: Path | str | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
    n_estimators: int | None = None,
    n_jobs: int | None = None,
    max_train_rows: int | None = None,
) -> PooledTrainingResult:
    """Train one pooled model from prepared tile-result JSON payloads."""

    batch_payload = _load_json_payload(batch_manifest_path)
    pooled_payload = batch_payload.get("pooled_model")
    if not isinstance(pooled_payload, dict):
        raise ValueError("Batch manifest does not contain pooled_model settings.")
    resolved_tile_result_paths = [Path(path) for path in tile_result_paths]
    if not resolved_tile_result_paths:
        raise ValueError("At least one tile result path is required.")
    pipeline_manifest_paths: list[Path] = []
    for tile_result_path in resolved_tile_result_paths:
        payload = _load_json_payload(tile_result_path)
        pipeline_manifest_path = payload.get("pipeline_manifest_path")
        if not isinstance(pipeline_manifest_path, str) or not pipeline_manifest_path:
            raise ValueError(f"Tile result does not contain pipeline_manifest_path: {tile_result_path}")
        pipeline_manifest_paths.append(Path(pipeline_manifest_path))

    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(str(pooled_payload["output_dir"]))
    )
    resolved_test_size = float(test_size) if test_size is not None else float(pooled_payload["test_size"])
    resolved_random_state = (
        int(random_state) if random_state is not None else int(pooled_payload["random_state"])
    )
    resolved_n_estimators = (
        int(n_estimators) if n_estimators is not None else int(pooled_payload["n_estimators"])
    )
    resolved_n_jobs = int(n_jobs) if n_jobs is not None else int(pooled_payload.get("n_jobs", -1))
    resolved_max_train_rows = max_train_rows
    if resolved_max_train_rows is None and pooled_payload.get("max_train_rows") is not None:
        resolved_max_train_rows = int(pooled_payload["max_train_rows"])

    logger.info(
        "Training pooled model from %d tile result(s)", len(resolved_tile_result_paths)
    )
    return train_pooled_model_from_pipeline_manifests(
        pipeline_manifest_paths,
        resolved_output_dir,
        test_size=resolved_test_size,
        random_state=resolved_random_state,
        n_estimators=resolved_n_estimators,
        n_jobs=resolved_n_jobs,
        max_train_rows=resolved_max_train_rows,
    )


def build_prepare_tile_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare one batch manifest plus per-tile manifests for cluster execution."
    )
    parser.add_argument("--feature-input", default=None, help="Feature raster or directory.")
    parser.add_argument("--manifest-path", default=None, help="AlphaEarth download manifest.")
    parser.add_argument("--reference-path", required=True, help="Path to the CROME reference source.")
    parser.add_argument("--year", required=True, type=int, help="Reference year.")
    parser.add_argument("--output-root", required=True, help="Base output directory.")
    parser.add_argument("--aoi-label", default=None, help="Optional human-readable batch label.")
    add_reference_args(parser)
    add_training_args(parser)
    add_pipeline_behavior_args(parser)
    return parser


def build_run_tile_plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one prepared per-tile batch plan.")
    parser.add_argument("--tile-plan", required=True, help="Path to one prepared tile JSON manifest.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Optional override for tile-local RandomForest CPU parallelism.",
    )
    return parser


def build_train_pooled_from_tile_results_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one pooled model from prepared tile result JSON payloads."
    )
    parser.add_argument("--batch-manifest", required=True, help="Prepared batch manifest JSON path.")
    parser.add_argument(
        "--tile-result",
        action="append",
        required=True,
        default=None,
        help="Tile result JSON emitted by crome run-tile-plan. Repeat for each tile.",
    )
    parser.add_argument("--output-dir", default=None, help="Override pooled model output directory.")
    parser.add_argument("--test-size", type=float, default=None, help="Override holdout fraction.")
    parser.add_argument("--random-state", type=int, default=None, help="Override random seed.")
    parser.add_argument("--n-estimators", type=int, default=None, help="Override tree count.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Override RandomForest CPU parallelism.")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap applied after the holdout split during pooled training.",
    )
    return parser


def main_prepare_tile_batch(argv: list[str] | None = None) -> int:
    parser = build_prepare_tile_batch_parser()
    args = parser.parse_args(argv)
    if args.feature_input is None and args.manifest_path is None:
        parser.error("Provide at least one of --feature-input or --manifest-path.")
    result = prepare_tile_batch(
        feature_input=args.feature_input,
        manifest_path=args.manifest_path,
        reference_path=args.reference_path,
        year=args.year,
        output_root=args.output_root,
        aoi_label=args.aoi_label,
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
    print(
        json.dumps(
            {
                "batch_label": result.batch_label,
                "batch_manifest_path": str(result.batch_manifest_path),
                "output_root": str(result.output_root),
                "pooled_output_dir": str(result.pooled_output_dir),
                "reference_input_path": str(result.reference_input_path),
                "reference_manifest_path": (
                    str(result.reference_manifest_path)
                    if result.reference_manifest_path is not None
                    else None
                ),
                "reference_path": str(result.reference_path),
                "tile_count": len(result.tile_manifest_paths),
                "tile_manifest_paths": [str(path) for path in result.tile_manifest_paths],
                "workflow_output_root": str(result.workflow_output_root),
                "year": result.year,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main_run_tile_plan(argv: list[str] | None = None) -> int:
    parser = build_run_tile_plan_parser()
    args = parser.parse_args(argv)
    result = run_tile_plan(args.tile_plan, n_jobs_override=args.n_jobs)
    payload = result.pipeline.to_dict()
    payload["batch_manifest_path"] = str(result.batch_manifest_path)
    payload["tile_plan_path"] = str(result.tile_plan_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main_train_pooled_from_tile_results(argv: list[str] | None = None) -> int:
    parser = build_train_pooled_from_tile_results_parser()
    args = parser.parse_args(argv)
    result = train_pooled_from_tile_results(
        batch_manifest_path=args.batch_manifest,
        tile_result_paths=args.tile_result,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        max_train_rows=args.max_train_rows,
    )
    print(
        json.dumps(
            {
                "dataset_label_mapping_path": str(result.dataset_label_mapping_path),
                "metrics_path": str(result.trained_model.metrics_path),
                "model_path": str(result.trained_model.model_path),
                "pipeline_manifest_paths": [str(path) for path in result.pipeline_manifest_paths],
                "pooled_manifest_path": str(result.pooled_manifest_path),
                "row_count": result.training_table.row_count,
                "sample_cache_manifest_paths": [str(path) for path in result.sample_cache_manifest_paths],
                "training_metadata_path": str(result.training_table.metadata_path),
                "training_table_path": str(result.training_table.table_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
