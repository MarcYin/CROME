"""Predict 10 m crop maps from AlphaEarth feature rasters."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .features import read_feature_raster_spec


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Outputs from prediction on one feature raster."""

    metadata_path: Path
    prediction_raster_path: Path


def predict_crop_map(
    feature_raster_path: Path | str,
    model_path: Path | str,
    output_raster_path: Path | str,
    *,
    nodata_label: int = -1,
) -> PredictionResult:
    """Predict a classified crop raster from one AlphaEarth feature raster."""

    feature_spec = read_feature_raster_spec(feature_raster_path)
    model_path = Path(model_path)
    output_raster_path = Path(output_raster_path)
    output_raster_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("rb") as file:
        bundle = pickle.load(file)

    feature_names = tuple(bundle["feature_names"])
    if feature_names != feature_spec.band_names:
        raise ValueError("Feature raster bands do not match the trained model feature order.")

    model = bundle["model"]
    metadata_path = output_raster_path.with_suffix(".json")

    with rasterio.open(feature_raster_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="int32", nodata=nodata_label, compress="deflate")
        with rasterio.open(output_raster_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                features = src.read(window=window, out_dtype="float32")
                flat_features = features.reshape(features.shape[0], -1).T
                valid = np.isfinite(flat_features).all(axis=1)
                predicted = np.full(flat_features.shape[0], nodata_label, dtype="int32")
                if valid.any():
                    valid_frame = pd.DataFrame(flat_features[valid], columns=feature_names)
                    predicted[valid] = model.predict(valid_frame).astype("int32")
                dst.write(predicted.reshape(features.shape[1], features.shape[2]), 1, window=window)

    metadata = {
        "feature_names": list(feature_names),
        "feature_raster_path": str(feature_raster_path),
        "model_path": str(model_path),
        "nodata_label": nodata_label,
        "prediction_raster_path": str(output_raster_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return PredictionResult(metadata_path=metadata_path, prediction_raster_path=output_raster_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict a crop map from an AlphaEarth feature raster.")
    parser.add_argument("--feature-raster", required=True, help="Path to one AlphaEarth feature raster.")
    parser.add_argument("--model-path", required=True, help="Path to a trained model pickle.")
    parser.add_argument("--output-raster", required=True, help="Output GeoTIFF for predicted labels.")
    parser.add_argument("--nodata-label", type=int, default=-1, help="Prediction nodata label.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = predict_crop_map(
        args.feature_raster,
        args.model_path,
        args.output_raster,
        nodata_label=args.nodata_label,
    )
    print(
        json.dumps(
            {
                "metadata_path": str(result.metadata_path),
                "prediction_raster_path": str(result.prediction_raster_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
