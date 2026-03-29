"""Build training tables and fit crop classifiers."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .features import read_feature_raster_spec, valid_feature_mask
from .paths import feature_artifact_name
from .schema import alphaearth_feature_columns


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


@dataclass(frozen=True, slots=True)
class TrainingRasterPair:
    """One aligned feature/label raster pair plus optional lineage metadata."""

    feature_raster_path: Path
    label_raster_path: Path
    feature_id: str
    source_image_id: str | None = None

    def __init__(
        self,
        feature_raster_path: Path | str,
        label_raster_path: Path | str,
        *,
        feature_id: str | None = None,
        source_image_id: str | None = None,
    ) -> None:
        object.__setattr__(self, "feature_raster_path", Path(feature_raster_path))
        object.__setattr__(self, "label_raster_path", Path(label_raster_path))
        object.__setattr__(
            self,
            "feature_id",
            feature_artifact_name(feature_id or feature_raster_path),
        )
        object.__setattr__(self, "source_image_id", source_image_id)


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

    return build_training_table_from_pairs(
        [TrainingRasterPair(feature_raster_path, label_raster_path)],
        output_dir,
    )


def _extract_training_rows(
    pair: TrainingRasterPair,
) -> tuple[tuple[str, ...], int, np.ndarray, np.ndarray]:
    """Extract one feature/label training table from aligned rasters."""

    feature_names, nodata_label = _validate_alignment(pair.feature_raster_path, pair.label_raster_path)
    feature_spec = read_feature_raster_spec(pair.feature_raster_path)

    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    with rasterio.open(pair.feature_raster_path) as feature_src, rasterio.open(
        pair.label_raster_path
    ) as label_src:
        for _, window in feature_src.block_windows(1):
            features = feature_src.read(window=window, out_dtype="float32")
            labels = label_src.read(1, window=window)
            valid = (labels != nodata_label) & valid_feature_mask(features, nodata=feature_spec.nodata)
            if not valid.any():
                continue
            feature_chunks.append(features[:, valid].T)
            label_chunks.append(labels[valid].astype("int32"))

    if not feature_chunks:
        raise ValueError("No labeled training pixels were found for the provided rasters.")

    return (
        feature_names,
        nodata_label,
        np.concatenate(feature_chunks, axis=0),
        np.concatenate(label_chunks, axis=0),
    )


def build_training_table_from_pairs(
    feature_label_pairs: Sequence[TrainingRasterPair | tuple[Path | str, Path | str]],
    output_dir: Path | str,
) -> TrainingTableResult:
    """Extract one combined feature/label training table from aligned raster pairs."""

    pairs = [
        pair
        if isinstance(pair, TrainingRasterPair)
        else TrainingRasterPair(pair[0], pair[1])
        for pair in feature_label_pairs
    ]
    if not pairs:
        raise ValueError("At least one feature/label raster pair is required.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "training_table.pkl"
    metadata_path = output_dir / "training_table.json"

    feature_names: tuple[str, ...] | None = None
    table_chunks: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []

    for pair in pairs:
        pair_feature_names, nodata_label, feature_matrix, label_vector = _extract_training_rows(pair)
        if feature_names is None:
            feature_names = pair_feature_names
        elif pair_feature_names != feature_names:
            raise ValueError("All feature rasters must expose the same AlphaEarth band order.")

        chunk = pd.DataFrame(feature_matrix, columns=feature_names)
        chunk["feature_id"] = pair.feature_id
        chunk["source_image_id"] = pair.source_image_id
        chunk["label_id"] = label_vector
        table_chunks.append(chunk)
        sources.append(
            {
                "feature_id": pair.feature_id,
                "feature_raster_path": str(pair.feature_raster_path),
                "label_raster_path": str(pair.label_raster_path),
                "nodata_label": nodata_label,
                "row_count": int(label_vector.shape[0]),
                "source_image_id": pair.source_image_id,
            }
        )

    if feature_names is None:
        raise ValueError("No aligned feature/label rows were extracted.")

    table = pd.concat(table_chunks, ignore_index=True)
    table.to_pickle(table_path)

    metadata = {
        "columns": list(feature_names),
        "feature_count": len({pair.feature_id for pair in pairs}),
        "feature_ids": [pair.feature_id for pair in pairs],
        "feature_raster_paths": [str(pair.feature_raster_path) for pair in pairs],
        "label_column": "label_id",
        "label_raster_paths": [str(pair.label_raster_path) for pair in pairs],
        "lineage_columns": ["feature_id", "source_image_id"],
        "row_count": int(table.shape[0]),
        "source_image_ids": [pair.source_image_id for pair in pairs],
        "sources": sources,
    }
    if len(pairs) == 1:
        metadata["feature_raster_path"] = str(pairs[0].feature_raster_path)
        metadata["label_raster_path"] = str(pairs[0].label_raster_path)
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
    label_mapping_path: Path | str | None = None,
) -> TrainedModelResult:
    """Train a random-forest classifier from a prepared training table."""

    table_path = Path(training_table_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_pickle(table_path)
    label_column = "label_id"
    feature_names = [
        column for column in alphaearth_feature_columns() if column in table.columns
    ]
    if tuple(feature_names) != alphaearth_feature_columns():
        raise ValueError("Training table is missing the canonical AlphaEarth feature columns.")
    X = table[feature_names]
    y = table[label_column]

    if table.empty:
        raise ValueError("Training table is empty.")

    class_counts = y.value_counts().sort_index()
    if len(class_counts) < 2:
        raise ValueError("Need at least two classes to train a classifier.")

    evaluation_mode = "pixel_holdout"
    evaluation_note = (
        "Holdout metrics are computed from a pixel-level split within the provided training table. "
        "They are not tile-level validation metrics."
    )
    train_feature_ids: list[str] | None = None
    test_feature_ids: list[str] | None = None

    feature_groups = (
        table["feature_id"]
        if "feature_id" in table.columns and table["feature_id"].nunique() > 1
        else None
    )
    can_pixel_hold_out = (
        0.0 < test_size < 1.0
        and int(class_counts.min()) >= 2
        and int(np.ceil(len(y) * test_size)) >= len(class_counts)
    )
    if feature_groups is not None and 0.0 < test_size < 1.0:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=feature_groups))
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        if y_train.nunique() >= 2 and not X_test.empty:
            evaluation_mode = "feature_holdout"
            evaluation_note = (
                "Holdout metrics are computed by holding out whole native AlphaEarth feature rasters "
                "grouped by feature_id."
            )
            train_feature_ids = sorted(str(value) for value in table.iloc[train_idx]["feature_id"].unique())
            test_feature_ids = sorted(str(value) for value in table.iloc[test_idx]["feature_id"].unique())
        elif can_pixel_hold_out:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        else:
            X_train = X
            y_train = y
            X_test = None
            y_test = None
            evaluation_mode = "train_only_no_holdout"
            evaluation_note = (
                "Model fitted on the full training table because neither feature-level nor pixel-level "
                "holdout was valid for the available class counts."
            )
    elif can_pixel_hold_out:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    else:
        X_train = X
        y_train = y
        X_test = None
        y_test = None
        evaluation_mode = "train_only_no_holdout"
        evaluation_note = (
            "Model fitted on the full training table because the requested holdout split was not "
            "valid for the available class counts."
        )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification = classification_report(y_test, y_pred, output_dict=True)
        training_accuracy = None
    else:
        y_pred = model.predict(X_train)
        accuracy = None
        classification = None
        training_accuracy = accuracy_score(y_train, y_pred)

    label_mapping = None
    if label_mapping_path is not None:
        label_mapping = json.loads(Path(label_mapping_path).read_text(encoding="utf-8"))

    metrics = {
        "accuracy": accuracy,
        "classification_report": classification,
        "evaluation_mode": evaluation_mode,
        "evaluation_note": evaluation_note,
        "feature_names": feature_names,
        "feature_group_count": (
            int(feature_groups.nunique()) if feature_groups is not None else int(table["feature_id"].nunique())
            if "feature_id" in table.columns
            else None
        ),
        "label_column": label_column,
        "label_mapping": label_mapping,
        "model_type": "RandomForestClassifier",
        "row_count": int(table.shape[0]),
        "test_feature_ids": test_feature_ids,
        "test_size": test_size,
        "train_feature_ids": train_feature_ids,
        "training_accuracy": training_accuracy,
    }

    model_path = output_dir / "model.pkl"
    metrics_path = output_dir / "metrics.json"
    with model_path.open("wb") as file:
        pickle.dump(
            {
                "feature_names": feature_names,
                "label_column": label_column,
                "label_mapping": label_mapping,
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
    parser.add_argument(
        "--label-mapping",
        default=None,
        help="Optional labels.json sidecar to persist id_to_label metadata in the model bundle.",
    )
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
        label_mapping_path=args.label_mapping,
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
