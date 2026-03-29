"""Build training tables and fit crop classifiers."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .features import read_feature_raster_spec


@dataclass(frozen=True, slots=True)
class TrainingTableResult:
    """Outputs from building a training table."""

    columns: tuple[str, ...]
    label_column: str
    metadata_path: Path
    row_count: int
    table_path: Path


@dataclass(frozen=True, slots=True)
class TrainedModelResult:
    """Outputs from model fitting."""

    metrics_path: Path
    model_path: Path
    row_count: int


def _validate_alignment(
    feature_raster_path: Path | str,
    label_raster_path: Path | str,
) -> tuple[tuple[str, ...], int]:
    feature_spec = read_feature_raster_spec(feature_raster_path)
    with rasterio.open(label_raster_path) as label_src:
        if label_src.width != feature_spec.width or label_src.height != feature_spec.height:
            raise ValueError("Feature raster and label raster dimensions do not match.")
        if str(label_src.crs) != feature_spec.crs:
            raise ValueError("Feature raster and label raster CRS do not match.")
        if not label_src.transform.almost_equals(feature_spec.transform):
            raise ValueError("Feature raster and label raster transforms do not match.")
        nodata_label = int(label_src.nodata) if label_src.nodata is not None else -1
    return feature_spec.band_names, nodata_label


def build_training_table(
    feature_raster_path: Path | str,
    label_raster_path: Path | str,
    output_dir: Path | str,
) -> TrainingTableResult:
    """Extract one feature/label training table from aligned rasters."""

    feature_names, nodata_label = _validate_alignment(feature_raster_path, label_raster_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "training_table.pkl"
    metadata_path = output_dir / "training_table.json"

    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    with rasterio.open(feature_raster_path) as feature_src, rasterio.open(label_raster_path) as label_src:
        for _, window in feature_src.block_windows(1):
            features = feature_src.read(window=window, out_dtype="float32")
            labels = label_src.read(1, window=window)
            valid = (labels != nodata_label) & np.isfinite(features).all(axis=0)
            if not valid.any():
                continue
            feature_chunks.append(features[:, valid].T)
            label_chunks.append(labels[valid].astype("int32"))

    if not feature_chunks:
        raise ValueError("No labeled training pixels were found for the provided rasters.")

    feature_matrix = np.concatenate(feature_chunks, axis=0)
    label_vector = np.concatenate(label_chunks, axis=0)
    table = pd.DataFrame(feature_matrix, columns=feature_names)
    table["label_id"] = label_vector
    table.to_pickle(table_path)

    metadata = {
        "columns": list(feature_names),
        "feature_raster_path": str(feature_raster_path),
        "label_column": "label_id",
        "label_raster_path": str(label_raster_path),
        "nodata_label": nodata_label,
        "row_count": int(table.shape[0]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return TrainingTableResult(
        columns=feature_names,
        label_column="label_id",
        metadata_path=metadata_path,
        row_count=int(table.shape[0]),
        table_path=table_path,
    )


def train_random_forest(
    training_table_path: Path | str,
    output_dir: Path | str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
) -> TrainedModelResult:
    """Train a random-forest classifier from a prepared training table."""

    table_path = Path(training_table_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_pickle(table_path)
    label_column = "label_id"
    feature_names = [column for column in table.columns if column != label_column]
    X = table[feature_names]
    y = table[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    metrics = {
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "feature_names": feature_names,
        "label_column": label_column,
        "model_type": "RandomForestClassifier",
        "row_count": int(table.shape[0]),
        "test_size": test_size,
    }

    model_path = output_dir / "model.pkl"
    metrics_path = output_dir / "metrics.json"
    with model_path.open("wb") as file:
        pickle.dump(
            {
                "feature_names": feature_names,
                "label_column": label_column,
                "model": model,
            },
            file,
        )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    return TrainedModelResult(metrics_path=metrics_path, model_path=model_path, row_count=int(table.shape[0]))


def build_training_table_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a training table from aligned rasters.")
    parser.add_argument("--feature-raster", required=True, help="Path to one AlphaEarth feature raster.")
    parser.add_argument("--label-raster", required=True, help="Path to one aligned label raster.")
    parser.add_argument("--output-dir", required=True, help="Directory for the training table outputs.")
    return parser


def build_train_model_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a crop classifier from a training table.")
    parser.add_argument("--training-table", required=True, help="Path to a pickled training table.")
    parser.add_argument("--output-dir", required=True, help="Directory for model outputs.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Evaluation holdout fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Random forest tree count.")
    return parser


def main_build_training_table(argv: list[str] | None = None) -> int:
    parser = build_training_table_parser()
    args = parser.parse_args(argv)
    result = build_training_table(args.feature_raster, args.label_raster, args.output_dir)
    print(
        json.dumps(
            {
                "columns": list(result.columns),
                "label_column": result.label_column,
                "metadata_path": str(result.metadata_path),
                "row_count": result.row_count,
                "table_path": str(result.table_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main_train_model(argv: list[str] | None = None) -> int:
    parser = build_train_model_parser()
    args = parser.parse_args(argv)
    result = train_random_forest(
        args.training_table,
        args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    print(
        json.dumps(
            {
                "metrics_path": str(result.metrics_path),
                "model_path": str(result.model_path),
                "row_count": result.row_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
