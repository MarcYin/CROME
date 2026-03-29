"""High-level orchestration for batch AlphaEarth-to-CROME baseline runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .config import AlphaEarthDownloadRequest, AlphaEarthTrainingSpec, CromeReferenceConfig
from .discovery import discover_feature_rasters
from .labeling import NoReferenceCoverageError, load_reference_label_mapping, rasterize_crome_reference
from .predict import predict_crop_map
from .training import TrainingRasterPair, build_training_table_from_pairs, train_random_forest


@dataclass(frozen=True, slots=True)
class PipelineFeatureResult:
    """Outputs for one processed native AlphaEarth raster."""

    feature_id: str
    feature_raster_path: Path
    label_mapping_path: Path
    label_raster_path: Path
    prediction_metadata_path: Path | None
    prediction_raster_path: Path | None
    source_image_id: str | None


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
    metrics_path: Path
    model_path: Path
    pipeline_manifest_path: Path
    skipped_features: tuple[SkippedFeatureResult, ...]
    training_metadata_path: Path
    training_table_path: Path


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
        overlap_policy=overlap_policy,
        nodata_label=nodata_label,
    )


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
        overlap_policy=overlap_policy,
        all_touched=all_touched,
        nodata_label=nodata_label,
    )

    processed_features: list[PipelineFeatureResult] = []
    skipped_features: list[SkippedFeatureResult] = []
    training_pairs: list[TrainingRasterPair] = []
    global_label_to_id, _ = load_reference_label_mapping(reference_path, label_column)

    for feature in discovered:
        output_dir = spec.reference_output_root / "features" / feature.feature_id
        try:
            rasterized = rasterize_crome_reference(
                feature.raster_path,
                spec,
                label_to_id=global_label_to_id,
                output_dir=output_dir,
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

        training_pairs.append(
            TrainingRasterPair(
                feature.raster_path,
                rasterized.label_raster_path,
                feature_id=feature.feature_id,
                source_image_id=feature.source_image_id,
            )
        )
        processed_features.append(
            PipelineFeatureResult(
                feature_id=feature.feature_id,
                feature_raster_path=feature.raster_path,
                label_mapping_path=rasterized.label_mapping_path,
                label_raster_path=rasterized.label_raster_path,
                prediction_metadata_path=None,
                prediction_raster_path=None,
                source_image_id=feature.source_image_id,
            )
        )

    if not training_pairs:
        raise ValueError("No discovered feature rasters produced usable CROME labels.")

    training_output_dir = spec.training_output_root / "dataset"
    training_table = build_training_table_from_pairs(training_pairs, training_output_dir)
    trained = train_random_forest(
        training_table.table_path,
        spec.training_output_root / "model",
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        label_mapping_path=processed_features[0].label_mapping_path,
    )
    metrics_payload = json.loads(trained.metrics_path.read_text(encoding="utf-8"))

    finalized_features: list[PipelineFeatureResult] = []
    for feature in processed_features:
        if predict:
            prediction = predict_crop_map(
                feature.feature_raster_path,
                trained.model_path,
                spec.prediction_output_root / f"{feature.feature_id}.tif",
                nodata_label=spec.nodata_label,
            )
            finalized_features.append(
                PipelineFeatureResult(
                    feature_id=feature.feature_id,
                    feature_raster_path=feature.feature_raster_path,
                    label_mapping_path=feature.label_mapping_path,
                    label_raster_path=feature.label_raster_path,
                    prediction_metadata_path=prediction.metadata_path,
                    prediction_raster_path=prediction.prediction_raster_path,
                    source_image_id=feature.source_image_id,
                )
            )
        else:
            finalized_features.append(feature)

    pipeline_manifest_path = spec.training_output_root / "pipeline.json"
    pipeline_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_payload = {
        "aoi_label": spec.alphaearth.aoi_label,
        "feature_count": len(finalized_features),
        "features": [
            {
                "feature_id": feature.feature_id,
                "feature_raster_path": str(feature.feature_raster_path),
                "label_mapping_path": str(feature.label_mapping_path),
                "label_raster_path": str(feature.label_raster_path),
                "prediction_metadata_path": (
                    str(feature.prediction_metadata_path) if feature.prediction_metadata_path else None
                ),
                "prediction_raster_path": (
                    str(feature.prediction_raster_path) if feature.prediction_raster_path else None
                ),
                "source_image_id": feature.source_image_id,
            }
            for feature in finalized_features
        ],
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "metrics": metrics_payload,
        "metrics_path": str(trained.metrics_path),
        "model_path": str(trained.model_path),
        "prediction_output_root": str(spec.prediction_output_root) if predict else None,
        "reference_path": str(reference_path),
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
        "training_metadata_path": str(training_table.metadata_path),
        "training_table_path": str(training_table.table_path),
        "year": year,
    }
    pipeline_manifest_path.write_text(
        json.dumps(pipeline_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return BaselinePipelineResult(
        feature_results=tuple(finalized_features),
        manifest_path=Path(manifest_path) if manifest_path is not None else None,
        metrics_path=trained.metrics_path,
        model_path=trained.model_path,
        pipeline_manifest_path=pipeline_manifest_path,
        skipped_features=tuple(skipped_features),
        training_metadata_path=training_table.metadata_path,
        training_table_path=training_table.table_path,
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
    parser.add_argument("--aoi-label", default=None, help="Optional run label used for output naming.")
    parser.add_argument("--output-root", default="data/alphaearth", help="Base output directory.")
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
        "manifest_path": str(result.manifest_path) if result.manifest_path is not None else None,
        "metrics_path": str(result.metrics_path),
        "model_path": str(result.model_path),
        "pipeline_manifest_path": str(result.pipeline_manifest_path),
        "skipped_feature_count": len(result.skipped_features),
        "training_metadata_path": str(result.training_metadata_path),
        "training_table_path": str(result.training_table_path),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
