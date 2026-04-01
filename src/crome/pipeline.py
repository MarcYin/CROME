"""High-level orchestration for batch AlphaEarth-to-CROME baseline runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .acquisition.crome import materialize_crome_reference_subset
from .config import AlphaEarthDownloadRequest, AlphaEarthTrainingSpec, CromeReferenceConfig
from .discovery import discover_feature_rasters
from .features import read_feature_raster_spec
from .labeling import (
    NoReferenceCoverageError,
    load_reference_label_mapping,
    rasterize_crome_reference,
    reference_source_bbox_for_feature_rasters,
)
from .paths import (
    OUTPUT_ROOT_ENV_VAR,
    default_output_root,
    feature_tile_name,
    prediction_tile_output_root,
    reference_tile_output_root,
    sample_cache_root,
    training_output_root,
    training_tile_output_root,
)
from .predict import predict_crop_map
from .qc import (
    load_manifest_payload,
    reference_summary,
    requested_aoi_from_manifest,
    requested_aoi_window,
    write_qc_overlay_png,
)
from .training import TrainingRasterPair, build_training_table_from_pairs, train_random_forest


@dataclass(frozen=True, slots=True)
class PipelineFeatureResult:
    """Outputs for one processed native AlphaEarth raster."""

    tile_id: str
    feature_id: str
    feature_raster_path: Path
    label_mapping_path: Path
    label_raster_path: Path
    metrics_path: Path
    model_path: Path
    prediction_output_root: Path | None
    prediction_metadata_path: Path | None
    prediction_raster_path: Path | None
    qc_manifest_path: Path
    sample_cache_manifest_path: Path | None
    sample_cache_root: Path | None
    source_image_id: str | None
    training_metadata_path: Path
    training_output_root: Path
    training_table_path: Path


@dataclass(frozen=True, slots=True)
class SkippedFeatureResult:
    """A discovered feature raster that was skipped by policy."""

    feature_id: str
    feature_raster_path: Path
    reason: str
    source_image_id: str | None


@dataclass(frozen=True, slots=True)
class BaselinePipelineResult:
    """Summary outputs for one batch AlphaEarth baseline run."""

    feature_results: tuple[PipelineFeatureResult, ...]
    manifest_path: Path | None
    pipeline_manifest_path: Path
    qc_manifest_path: Path
    reference_input_path: Path
    reference_manifest_path: Path | None
    reference_path: Path
    skipped_features: tuple[SkippedFeatureResult, ...]
    sample_cache_root: Path | None


def _default_aoi_label(
    feature_input: Path | str | None,
    manifest_path: Path | str | None,
) -> str | None:
    if feature_input is not None:
        path = Path(feature_input)
        return path.stem if path.is_file() else path.name
    if manifest_path is not None:
        return Path(manifest_path).parent.name
    return None


def _build_training_spec(
    *,
    feature_input: Path | str | None,
    manifest_path: Path | str | None,
    reference_path: Path | str,
    year: int,
    output_root: Path | str,
    aoi_label: str | None,
    label_column: str,
    geometry_column: str,
    label_mode: str,
    overlap_policy: str,
    all_touched: bool,
    nodata_label: int,
) -> AlphaEarthTrainingSpec:
    alphaearth = AlphaEarthDownloadRequest(
        year=year,
        output_root=output_root,
        aoi_label=aoi_label or _default_aoi_label(feature_input, manifest_path),
        bbox=(0.0, 0.0, 1.0, 1.0),
    )
    reference = CromeReferenceConfig(
        source_path=reference_path,
        year=year,
        aoi_label=alphaearth.aoi_label,
        label_column=label_column,
        geometry_column=geometry_column,
        all_touched=all_touched,
    )
    return AlphaEarthTrainingSpec(
        alphaearth=alphaearth,
        reference=reference,
        label_mode=label_mode,
        overlap_policy=overlap_policy,
        nodata_label=nodata_label,
    )


def _reference_manifest_path(reference_path: Path | str) -> Path | None:
    resolved = Path(reference_path)
    candidates = (
        resolved.with_suffix(".json"),
        resolved.parent / "manifest.json",
        resolved.parent.parent / "manifest.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def _download_summary_by_image_id(
    manifest_payload: dict[str, object] | None,
) -> dict[str, dict[str, object]]:
    if not isinstance(manifest_payload, dict):
        return {}
    download_payload = manifest_payload.get("download")
    if not isinstance(download_payload, dict):
        return {}
    results = download_payload.get("results")
    if not isinstance(results, list):
        return {}
    summary: dict[str, dict[str, object]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        image_id = result.get("image_id")
        if not isinstance(image_id, str):
            continue
        summary[image_id] = {
            key: result.get(key)
            for key in ("chunk_count", "metadata_path", "status", "tiff_path")
        }
    return summary


def _sample_cache_namespace(spec: AlphaEarthTrainingSpec) -> str:
    payload = {
        "all_touched": spec.reference.all_touched,
        "geometry_column": spec.reference.geometry_column,
        "label_column": spec.reference.label_column,
        "label_mode": spec.label_mode,
        "overlap_policy": spec.overlap_policy,
        "reference_name": spec.reference.reference_name,
        "year": spec.reference.year,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return (
        f"{spec.reference.reference_name}_{spec.label_mode}_{spec.reference.label_column}_"
        f"{spec.overlap_policy}_{digest}"
    )


def _tile_label_output_namespace(spec: AlphaEarthTrainingSpec) -> str:
    return _sample_cache_namespace(spec)


def _tile_model_output_namespace(
    spec: AlphaEarthTrainingSpec,
    *,
    test_size: float,
    random_state: int,
    n_estimators: int,
) -> str:
    payload = {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "sample_cache_namespace": _sample_cache_namespace(spec),
        "test_size": test_size,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{_sample_cache_namespace(spec)}_rf_{digest}"


def _feature_tile_id(feature: object) -> str:
    return feature_tile_name(
        feature_id=getattr(feature, "feature_id", None),
        source_image_id=getattr(feature, "source_image_id", None),
        feature_raster_path=getattr(feature, "raster_path", None),
    )


def _feature_result_payload(feature: PipelineFeatureResult) -> dict[str, object]:
    return {
        "feature_id": feature.feature_id,
        "feature_raster_path": str(feature.feature_raster_path),
        "label_mapping_path": str(feature.label_mapping_path),
        "label_raster_path": str(feature.label_raster_path),
        "metrics_path": str(feature.metrics_path),
        "model_path": str(feature.model_path),
        "prediction_metadata_path": (
            str(feature.prediction_metadata_path) if feature.prediction_metadata_path else None
        ),
        "prediction_output_root": (
            str(feature.prediction_output_root) if feature.prediction_output_root is not None else None
        ),
        "prediction_raster_path": (
            str(feature.prediction_raster_path) if feature.prediction_raster_path else None
        ),
        "qc_manifest_path": str(feature.qc_manifest_path),
        "sample_cache_manifest_path": (
            str(feature.sample_cache_manifest_path) if feature.sample_cache_manifest_path is not None else None
        ),
        "sample_cache_root": (
            str(feature.sample_cache_root) if feature.sample_cache_root is not None else None
        ),
        "source_image_id": feature.source_image_id,
        "tile_id": feature.tile_id,
        "training_metadata_path": str(feature.training_metadata_path),
        "training_output_root": str(feature.training_output_root),
        "training_table_path": str(feature.training_table_path),
    }


def _qc_feature_payload(
    feature: PipelineFeatureResult,
    *,
    download_summary: dict[str, dict[str, object]],
    qc_output_dir: Path,
    requested_aoi: dict[str, object] | None,
) -> dict[str, object]:
    feature_spec = read_feature_raster_spec(feature.feature_raster_path)
    label_payload = json.loads(feature.label_mapping_path.read_text(encoding="utf-8"))
    requested_window = requested_aoi_window(
        feature.feature_raster_path,
        requested_aoi,
    )
    qc_png_path = qc_output_dir / f"{feature.feature_id}.png"
    write_qc_overlay_png(
        feature.feature_raster_path,
        feature.label_raster_path,
        qc_png_path,
        requested_window=requested_window,
    )
    return {
        "download": download_summary.get(feature.source_image_id or "", {}),
        "feature_id": feature.feature_id,
        "feature_raster_path": str(feature.feature_raster_path),
        "label_mapping_path": str(feature.label_mapping_path),
        "label_qc": {
            "feature_bounds": label_payload.get("feature_bounds"),
            "feature_crs": label_payload.get("feature_crs"),
            "feature_shape": label_payload.get("feature_shape"),
            "label_stats": label_payload.get("label_stats"),
            "reference_bounds_in_feature_crs": label_payload.get("reference_bounds_in_feature_crs"),
            "reference_centroid_bounds_in_feature_crs": label_payload.get(
                "reference_centroid_bounds_in_feature_crs"
            ),
            "reference_feature_count": label_payload.get("reference_feature_count"),
        },
        "label_raster_path": str(feature.label_raster_path),
        "label_qc_png_path": str(qc_png_path),
        "metrics_path": str(feature.metrics_path),
        "model_path": str(feature.model_path),
        "prediction_raster_path": (
            str(feature.prediction_raster_path) if feature.prediction_raster_path is not None else None
        ),
        "prediction_output_root": (
            str(feature.prediction_output_root) if feature.prediction_output_root is not None else None
        ),
        "sample_cache_manifest_path": (
            str(feature.sample_cache_manifest_path) if feature.sample_cache_manifest_path is not None else None
        ),
        "sample_cache_root": (
            str(feature.sample_cache_root) if feature.sample_cache_root is not None else None
        ),
        "raster": {
            "band_count": feature_spec.count,
            "bounds": [float(value) for value in feature_spec.bounds],
            "crs": feature_spec.crs,
            "dtype": feature_spec.dtype,
            "height": feature_spec.height,
            "transform": [float(value) for value in feature_spec.transform[:6]],
            "width": feature_spec.width,
        },
        "requested_aoi_window": requested_window,
        "source_image_id": feature.source_image_id,
        "tile_id": feature.tile_id,
        "training_metadata_path": str(feature.training_metadata_path),
        "training_output_root": str(feature.training_output_root),
        "training_table_path": str(feature.training_table_path),
    }


def run_baseline_pipeline(
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
    overlap_policy: str = "error",
    all_touched: bool = False,
    nodata_label: int = -1,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    predict: bool = True,
    skip_empty_labels: bool = True,
) -> BaselinePipelineResult:
    """Run the baseline pipeline across one or more native AlphaEarth rasters."""

    discovered = discover_feature_rasters(feature_input=feature_input, manifest_path=manifest_path)
    spec = _build_training_spec(
        feature_input=feature_input,
        manifest_path=manifest_path,
        reference_path=reference_path,
        year=year,
        output_root=output_root,
        aoi_label=aoi_label,
        label_column=label_column,
        geometry_column=geometry_column,
        label_mode=label_mode,
        overlap_policy=overlap_policy,
        all_touched=all_touched,
        nodata_label=nodata_label,
    )

    processed_features: list[PipelineFeatureResult] = []
    skipped_features: list[SkippedFeatureResult] = []
    label_output_namespace = _tile_label_output_namespace(spec)
    model_output_namespace = _tile_model_output_namespace(
        spec,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    sample_cache_dir = sample_cache_root(
        output_root,
        year,
        cache_label=_sample_cache_namespace(spec),
        label_mode=spec.label_mode,
        reference_name=spec.reference.reference_name,
    )
    batch_output_dir = training_output_root(
        output_root,
        spec.alphaearth.aoi_label,
        year,
        namespace=model_output_namespace,
    )
    resolved_reference_path = materialize_crome_reference_subset(
        reference_path,
        feature_raster_paths=[feature.raster_path for feature in discovered],
        subset_label=spec.alphaearth.aoi_label,
        year=year,
    )
    if Path(resolved_reference_path) != Path(reference_path):
        spec = _build_training_spec(
            feature_input=feature_input,
            manifest_path=manifest_path,
            reference_path=resolved_reference_path,
            year=year,
            output_root=output_root,
            aoi_label=aoi_label,
            label_column=label_column,
            geometry_column=geometry_column,
            label_mode=label_mode,
            overlap_policy=overlap_policy,
            all_touched=all_touched,
            nodata_label=nodata_label,
        )
    reference_bbox = reference_source_bbox_for_feature_rasters(
        resolved_reference_path,
        [feature.raster_path for feature in discovered],
    )
    global_label_to_id, _ = load_reference_label_mapping(
        resolved_reference_path,
        label_column,
        bbox=reference_bbox,
    )

    manifest_payload = load_manifest_payload(manifest_path)
    requested_aoi = requested_aoi_from_manifest(manifest_payload)
    download_summary = _download_summary_by_image_id(manifest_payload)
    reference_manifest_path = _reference_manifest_path(resolved_reference_path)

    for feature in discovered:
        tile_id = _feature_tile_id(feature)
        reference_output_dir = reference_tile_output_root(
            output_root,
            tile_id,
            year,
            reference_name=spec.reference.reference_name,
            namespace=label_output_namespace,
        )
        try:
            rasterized = rasterize_crome_reference(
                feature.raster_path,
                spec,
                label_to_id=global_label_to_id,
                output_dir=reference_output_dir,
            )
        except NoReferenceCoverageError as exc:
            if not skip_empty_labels:
                raise
            skipped_features.append(
                SkippedFeatureResult(
                    feature_id=feature.feature_id,
                    feature_raster_path=feature.raster_path,
                    reason=str(exc),
                    source_image_id=feature.source_image_id,
                )
            )
            continue

        training_pair = TrainingRasterPair(
            feature.raster_path,
            rasterized.label_raster_path,
            feature_id=feature.feature_id,
            label_mapping_path=rasterized.label_mapping_path,
            source_image_id=feature.source_image_id,
        )
        training_output_dir = training_tile_output_root(
            output_root,
            tile_id,
            year,
            namespace=model_output_namespace,
        )
        training_table = build_training_table_from_pairs(
            [training_pair],
            training_output_dir / "dataset",
            sample_cache_root=sample_cache_dir,
            sample_cache_metadata={
                "aoi_label": spec.alphaearth.aoi_label,
                "feature_id": feature.feature_id,
                "label_column": spec.reference.label_column,
                "label_mode": spec.label_mode,
                "overlap_policy": spec.overlap_policy,
                "reference_manifest_path": (
                    str(reference_manifest_path) if reference_manifest_path is not None else None
                ),
                "reference_name": spec.reference.reference_name,
                "reference_path": str(resolved_reference_path),
                "source_image_id": feature.source_image_id,
                "tile_id": tile_id,
                "year": year,
            },
        )
        trained = train_random_forest(
            training_table.table_path,
            training_output_dir / "model",
            test_size=test_size,
            random_state=random_state,
            n_estimators=n_estimators,
            label_mapping_path=rasterized.label_mapping_path,
        )
        prediction_output_dir = (
            prediction_tile_output_root(
                output_root,
                tile_id,
                year,
                namespace=model_output_namespace,
            )
            if predict
            else None
        )
        prediction_metadata_path = None
        prediction_raster_path = None
        if predict:
            prediction_output_dir.mkdir(parents=True, exist_ok=True)
            prediction = predict_crop_map(
                feature.raster_path,
                trained.model_path,
                prediction_output_dir / "prediction.tif",
                nodata_label=spec.nodata_label,
            )
            prediction_metadata_path = prediction.metadata_path
            prediction_raster_path = prediction.prediction_raster_path

        feature_result = PipelineFeatureResult(
            tile_id=tile_id,
            feature_id=feature.feature_id,
            feature_raster_path=feature.raster_path,
            label_mapping_path=rasterized.label_mapping_path,
            label_raster_path=rasterized.label_raster_path,
            metrics_path=trained.metrics_path,
            model_path=trained.model_path,
            prediction_output_root=prediction_output_dir,
            prediction_metadata_path=prediction_metadata_path,
            prediction_raster_path=prediction_raster_path,
            qc_manifest_path=training_output_dir / "qc" / "run_qc.json",
            sample_cache_manifest_path=training_table.sample_cache_manifest_path,
            sample_cache_root=training_table.sample_cache_root,
            source_image_id=feature.source_image_id,
            training_metadata_path=training_table.metadata_path,
            training_output_root=training_output_dir,
            training_table_path=training_table.table_path,
        )
        feature_result.qc_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        feature_result.qc_manifest_path.write_text(
            json.dumps(
                {
                    "aoi_label": spec.alphaearth.aoi_label,
                    "feature_count": 1,
                    "features": [
                        _qc_feature_payload(
                            feature_result,
                            download_summary=download_summary,
                            qc_output_dir=feature_result.qc_manifest_path.parent,
                            requested_aoi=requested_aoi,
                        )
                    ],
                    "manifest_path": str(manifest_path) if manifest_path is not None else None,
                    "reference_input_path": str(Path(reference_path)),
                    "reference_manifest_path": (
                        str(reference_manifest_path) if reference_manifest_path is not None else None
                    ),
                    "reference_path": str(resolved_reference_path),
                    "reference_summary": reference_summary(resolved_reference_path),
                    "requested_aoi": (
                        {
                            "bounds": list(requested_aoi["bounds"]),
                            "crs": requested_aoi["crs"],
                            "source": requested_aoi["source"],
                        }
                        if requested_aoi is not None
                        else None
                    ),
                    "sample_cache_manifest_path": (
                        str(feature_result.sample_cache_manifest_path)
                        if feature_result.sample_cache_manifest_path is not None
                        else None
                    ),
                    "sample_cache_root": (
                        str(feature_result.sample_cache_root)
                        if feature_result.sample_cache_root is not None
                        else None
                    ),
                    "skipped_feature_count": 0,
                    "skipped_features": [],
                    "tile_id": feature_result.tile_id,
                    "year": year,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        processed_features.append(feature_result)

    if not processed_features:
        raise ValueError("No discovered feature rasters produced usable CROME labels.")
    qc_output_dir = batch_output_dir / "qc"
    qc_output_dir.mkdir(parents=True, exist_ok=True)
    qc_manifest_path = qc_output_dir / "run_qc.json"
    qc_manifest_path.write_text(
        json.dumps(
            {
                "aoi_label": spec.alphaearth.aoi_label,
                "feature_count": len(processed_features),
                "features": [
                    _qc_feature_payload(
                        feature,
                        download_summary=download_summary,
                        qc_output_dir=qc_output_dir,
                        requested_aoi=requested_aoi,
                    )
                    for feature in processed_features
                ],
                "manifest_path": str(manifest_path) if manifest_path is not None else None,
                "reference_input_path": str(Path(reference_path)),
                "reference_manifest_path": (
                    str(reference_manifest_path) if reference_manifest_path is not None else None
                ),
                "reference_path": str(resolved_reference_path),
                "reference_summary": reference_summary(resolved_reference_path),
                "requested_aoi": (
                    {
                        "bounds": list(requested_aoi["bounds"]),
                        "crs": requested_aoi["crs"],
                        "source": requested_aoi["source"],
                    }
                    if requested_aoi is not None
                    else None
                ),
                "sample_cache_manifest_paths": [
                    str(feature.sample_cache_manifest_path)
                    for feature in processed_features
                    if feature.sample_cache_manifest_path is not None
                ],
                "sample_cache_root": str(sample_cache_dir),
                "skipped_feature_count": len(skipped_features),
                "skipped_features": [
                    {
                        "feature_id": feature.feature_id,
                        "feature_raster_path": str(feature.feature_raster_path),
                        "reason": feature.reason,
                        "source_image_id": feature.source_image_id,
                    }
                    for feature in skipped_features
                ],
                "year": year,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    pipeline_manifest_path = batch_output_dir / "pipeline.json"
    pipeline_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_payload = {
        "aoi_label": spec.alphaearth.aoi_label,
        "feature_count": len(processed_features),
        "features": [_feature_result_payload(feature) for feature in processed_features],
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "label_mode": spec.label_mode,
        "qc_manifest_path": str(qc_manifest_path),
        "reference_input_path": str(Path(reference_path)),
        "reference_manifest_path": (
            str(reference_manifest_path) if reference_manifest_path is not None else None
        ),
        "reference_path": str(resolved_reference_path),
        "reference_summary": reference_summary(resolved_reference_path),
        "sample_cache_manifest_paths": [
            str(feature.sample_cache_manifest_path)
            for feature in processed_features
            if feature.sample_cache_manifest_path is not None
        ],
        "sample_cache_root": str(sample_cache_dir),
        "skipped_feature_count": len(skipped_features),
        "skipped_features": [
            {
                "feature_id": feature.feature_id,
                "feature_raster_path": str(feature.feature_raster_path),
                "reason": feature.reason,
                "source_image_id": feature.source_image_id,
            }
            for feature in skipped_features
        ],
        "year": year,
    }
    pipeline_manifest_path.write_text(
        json.dumps(pipeline_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return BaselinePipelineResult(
        feature_results=tuple(processed_features),
        manifest_path=Path(manifest_path) if manifest_path is not None else None,
        pipeline_manifest_path=pipeline_manifest_path,
        qc_manifest_path=qc_manifest_path,
        reference_input_path=Path(reference_path),
        reference_manifest_path=reference_manifest_path,
        reference_path=Path(resolved_reference_path),
        skipped_features=tuple(skipped_features),
        sample_cache_root=sample_cache_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the baseline AlphaEarth-to-CROME pipeline over one or more native rasters."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--feature-input",
        default=None,
        help=(
            "Path to one AlphaEarth feature raster or a directory tree containing native AlphaEarth "
            "GeoTIFFs."
        ),
    )
    input_group.add_argument(
        "--manifest-path",
        default=None,
        help="Path to an edown manifest; raster discovery falls back to the manifest directory.",
    )
    parser.add_argument("--reference-path", required=True, help="Path to the CROME vector reference file.")
    parser.add_argument("--year", required=True, type=int, help="Reference year.")
    parser.add_argument(
        "--aoi-label",
        default=None,
        help=(
            "Optional batch-summary label. Tile labels, training artifacts, models, and predictions "
            "are keyed by discovered AlphaEarth feature tiles."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=default_output_root(),
        help=(
            f"Base output directory. Defaults to ${OUTPUT_ROOT_ENV_VAR} when set, "
            "otherwise data/alphaearth."
        ),
    )
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_baseline_pipeline(
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
        predict=not args.no_predict,
        skip_empty_labels=not args.fail_on_empty_labels,
    )
    payload = {
        "feature_count": len(result.feature_results),
        "features": [_feature_result_payload(feature) for feature in result.feature_results],
        "manifest_path": str(result.manifest_path) if result.manifest_path is not None else None,
        "pipeline_manifest_path": str(result.pipeline_manifest_path),
        "qc_manifest_path": str(result.qc_manifest_path),
        "reference_input_path": str(result.reference_input_path),
        "reference_manifest_path": (
            str(result.reference_manifest_path) if result.reference_manifest_path is not None else None
        ),
        "reference_path": str(result.reference_path),
        "skipped_feature_count": len(result.skipped_features),
        "sample_cache_root": str(result.sample_cache_root) if result.sample_cache_root is not None else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
