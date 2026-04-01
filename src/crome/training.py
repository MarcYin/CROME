"""Build training tables and fit crop classifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .features import read_feature_raster_spec, valid_feature_mask
from .paths import feature_artifact_name
from .schema import alphaearth_feature_columns

_TRAINING_SAMPLE_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrainingTableResult:
    """Outputs from building a training table."""

    columns: tuple[str, ...]
    label_column: str
    metadata_path: Path
    row_count: int
    sample_cache_manifest_path: Path | None
    sample_cache_root: Path | None
    table_path: Path


@dataclass(frozen=True, slots=True)
class TrainedModelResult:
    """Outputs from model fitting."""

    metrics_path: Path
    model_path: Path
    row_count: int


@dataclass(frozen=True, slots=True)
class PooledTrainingResult:
    """Outputs from pooled model training driven by pipeline manifests."""

    dataset_label_mapping_path: Path
    pipeline_manifest_paths: tuple[Path, ...]
    pooled_manifest_path: Path
    sample_cache_manifest_paths: tuple[Path, ...]
    trained_model: TrainedModelResult
    training_table: TrainingTableResult


@dataclass(frozen=True, slots=True)
class TrainingRasterPair:
    """One aligned feature/label raster pair plus optional lineage metadata."""

    feature_raster_path: Path
    label_raster_path: Path
    feature_id: str
    label_mapping_path: Path | None = None
    source_image_id: str | None = None

    def __init__(
        self,
        feature_raster_path: Path | str,
        label_raster_path: Path | str,
        *,
        label_mapping_path: Path | str | None = None,
        feature_id: str | None = None,
        source_image_id: str | None = None,
    ) -> None:
        object.__setattr__(self, "feature_raster_path", Path(feature_raster_path))
        object.__setattr__(self, "label_raster_path", Path(label_raster_path))
        object.__setattr__(
            self,
            "label_mapping_path",
            Path(label_mapping_path) if label_mapping_path is not None else None,
        )
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
    *,
    label_mapping_path: Path | str | None = None,
    sample_cache_root: Path | str | None = None,
) -> TrainingTableResult:
    """Extract one feature/label training table from aligned rasters."""

    return build_training_table_from_pairs(
        [
            TrainingRasterPair(
                feature_raster_path,
                label_raster_path,
                label_mapping_path=label_mapping_path,
            )
        ],
        output_dir,
        sample_cache_root=sample_cache_root,
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


def _path_signature(path: Path | str) -> dict[str, object]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "mtime_ns": stat.st_mtime_ns,
        "path": str(resolved),
        "size_bytes": stat.st_size,
    }


def _label_raster_signature(path: Path | str) -> dict[str, object]:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with rasterio.open(resolved) as src:
        for _, window in src.block_windows(1):
            digest.update(src.read(1, window=window, masked=False).tobytes())
        return {
            "band_sha256": digest.hexdigest(),
            "bounds": [float(value) for value in src.bounds],
            "crs": str(src.crs) if src.crs is not None else None,
            "dtype": str(src.dtypes[0]),
            "height": int(src.height),
            "nodata": src.nodata,
            "transform": [float(value) for value in src.transform[:6]],
            "width": int(src.width),
        }


def _canonical_label_mapping_signature(
    label_mapping: dict[str, object] | None,
) -> dict[str, object] | None:
    if label_mapping is None:
        return None
    return {
        "feature_bounds": label_mapping.get("feature_bounds"),
        "feature_crs": label_mapping.get("feature_crs"),
        "feature_shape": label_mapping.get("feature_shape"),
        "geometry_column": label_mapping.get("geometry_column"),
        "id_to_label": label_mapping.get("id_to_label"),
        "label_column": label_mapping.get("label_column"),
        "label_mode": label_mapping.get("label_mode"),
        "label_to_id": label_mapping.get("label_to_id"),
        "nodata_label": label_mapping.get("nodata_label"),
        "reference_bounds_in_feature_crs": label_mapping.get("reference_bounds_in_feature_crs"),
        "reference_centroid_bounds_in_feature_crs": label_mapping.get(
            "reference_centroid_bounds_in_feature_crs"
        ),
        "reference_feature_count": label_mapping.get("reference_feature_count"),
        "year": label_mapping.get("year"),
    }


def _pair_source_signatures(pair: TrainingRasterPair) -> dict[str, object]:
    label_mapping = _load_pair_label_mapping(pair)
    return {
        "feature_raster": _path_signature(pair.feature_raster_path),
        "label_mapping": _canonical_label_mapping_signature(label_mapping),
        "label_raster": _label_raster_signature(pair.label_raster_path),
    }


def _sample_cache_key(pair: TrainingRasterPair) -> str:
    payload = {
        "feature_id": pair.feature_id,
        "feature_raster": _path_signature(pair.feature_raster_path),
        "label_signatures": _pair_source_signatures(pair),
        "schema_version": _TRAINING_SAMPLE_CACHE_SCHEMA_VERSION,
        "source_image_id": pair.source_image_id,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:24]


def _sample_cache_paths(
    pair: TrainingRasterPair,
    sample_cache_root: Path,
) -> tuple[Path, Path]:
    cache_key = _sample_cache_key(pair)
    entry_dir = sample_cache_root / cache_key[:2] / pair.feature_id / cache_key
    return entry_dir / "samples.pkl", entry_dir / "metadata.json"


def _load_pair_label_mapping(pair: TrainingRasterPair) -> dict[str, object] | None:
    if pair.label_mapping_path is None or not pair.label_mapping_path.exists():
        return None
    return json.loads(pair.label_mapping_path.read_text(encoding="utf-8"))


def _extract_training_frame(
    pair: TrainingRasterPair,
) -> tuple[tuple[str, ...], pd.DataFrame, dict[str, object]]:
    pair_feature_names, nodata_label, feature_matrix, label_vector = _extract_training_rows(pair)
    frame = pd.DataFrame(feature_matrix, columns=pair_feature_names)
    frame["feature_id"] = pair.feature_id
    frame["source_image_id"] = pair.source_image_id
    frame["label_id"] = label_vector
    label_mapping = _load_pair_label_mapping(pair)
    metadata = {
        "cache_key": _sample_cache_key(pair),
        "columns": list(pair_feature_names),
        "frame_columns": list(frame.columns),
        "feature_id": pair.feature_id,
        "feature_raster_path": str(pair.feature_raster_path),
        "label_raster_path": str(pair.label_raster_path),
        "label_column": "label_id",
        "lineage_columns": ["feature_id", "source_image_id"],
        "nodata_label": nodata_label,
        "row_count": int(label_vector.shape[0]),
        "sample_cache_schema_version": _TRAINING_SAMPLE_CACHE_SCHEMA_VERSION,
        "source_image_id": pair.source_image_id,
        "source_signatures": _pair_source_signatures(pair),
    }
    if pair.label_mapping_path is not None:
        metadata["label_mapping_path"] = str(pair.label_mapping_path)
    if label_mapping is not None:
        metadata["id_to_label"] = label_mapping.get("id_to_label")
        metadata["label_to_id"] = label_mapping.get("label_to_id")
    return pair_feature_names, frame, metadata


def _load_cached_training_frame(
    pair: TrainingRasterPair,
    sample_cache_root: Path,
) -> tuple[tuple[str, ...], pd.DataFrame, dict[str, object]] | None:
    data_path, metadata_path = _sample_cache_paths(pair, sample_cache_root)
    if not data_path.exists() or not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("sample_cache_schema_version") != _TRAINING_SAMPLE_CACHE_SCHEMA_VERSION:
        return None
    expected_signatures = _pair_source_signatures(pair)
    if metadata.get("source_signatures") != expected_signatures:
        return None

    columns = metadata.get("columns")
    if not isinstance(columns, list) or not columns:
        return None

    frame = pd.read_pickle(data_path)
    frame_columns = metadata.get("frame_columns")
    if not isinstance(frame_columns, list) or not frame_columns:
        frame_columns = list(columns) + ["feature_id", "source_image_id", "label_id"]
    expected_columns = [str(value) for value in frame_columns]
    if list(frame.columns) != expected_columns:
        return None
    return tuple(str(value) for value in columns), frame, metadata


def _write_cached_training_frame(
    pair: TrainingRasterPair,
    frame: pd.DataFrame,
    metadata: dict[str, object],
    sample_cache_root: Path,
) -> tuple[Path, Path]:
    data_path, metadata_path = _sample_cache_paths(pair, sample_cache_root)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(data_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return data_path, metadata_path


def build_training_table_from_pairs(
    feature_label_pairs: Sequence[TrainingRasterPair | tuple[Path | str, Path | str]],
    output_dir: Path | str,
    *,
    sample_cache_root: Path | str | None = None,
    sample_cache_metadata: dict[str, object] | None = None,
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
    resolved_sample_cache_root = Path(sample_cache_root) if sample_cache_root is not None else None
    sample_cache_manifest_path = (
        output_dir / "sample_cache_manifest.json" if resolved_sample_cache_root is not None else None
    )

    feature_names: tuple[str, ...] | None = None
    table_chunks: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    cache_entries: list[dict[str, object]] = []
    seen_sample_keys: set[str] = set()

    for pair in pairs:
        sample_key = _sample_cache_key(pair)
        if sample_key in seen_sample_keys:
            raise ValueError(
                "Duplicate feature/label sample inputs detected; refusing to double-count cached rows."
            )
        seen_sample_keys.add(sample_key)
        cache_status = "miss"
        cached = None
        if resolved_sample_cache_root is not None:
            cached = _load_cached_training_frame(pair, resolved_sample_cache_root)
        if cached is not None:
            pair_feature_names, chunk, source_metadata = cached
            cache_status = "hit"
        else:
            pair_feature_names, chunk, source_metadata = _extract_training_frame(pair)
            if resolved_sample_cache_root is not None:
                data_path, metadata_cache_path = _write_cached_training_frame(
                    pair,
                    chunk,
                    source_metadata,
                    resolved_sample_cache_root,
                )
                cache_entries.append(
                    {
                        "cache_data_path": str(data_path),
                        "cache_key": sample_key,
                        "cache_metadata_path": str(metadata_cache_path),
                        "cache_status": cache_status,
                        "feature_id": pair.feature_id,
                        "row_count": source_metadata["row_count"],
                    }
                )
                if feature_names is None:
                    feature_names = pair_feature_names
                elif pair_feature_names != feature_names:
                    raise ValueError("All feature rasters must expose the same AlphaEarth band order.")
                table_chunks.append(chunk)
                sources.append(source_metadata)
                continue
        if feature_names is None:
            feature_names = pair_feature_names
        elif pair_feature_names != feature_names:
            raise ValueError("All feature rasters must expose the same AlphaEarth band order.")
        table_chunks.append(chunk)
        sources.append(source_metadata)
        if resolved_sample_cache_root is not None:
            data_path, metadata_cache_path = _sample_cache_paths(pair, resolved_sample_cache_root)
            cache_entries.append(
                    {
                        "cache_data_path": str(data_path),
                        "cache_key": sample_key,
                        "cache_metadata_path": str(metadata_cache_path),
                        "cache_status": cache_status,
                        "feature_id": pair.feature_id,
                        "row_count": source_metadata["row_count"],
                    }
            )

    if feature_names is None:
        raise ValueError("No aligned feature/label rows were extracted.")

    table = pd.concat(table_chunks, ignore_index=True)
    table.to_pickle(table_path)

    if sample_cache_manifest_path is not None and resolved_sample_cache_root is not None:
        sample_cache_manifest_path.write_text(
            json.dumps(
                {
                    "cache_entry_count": len(cache_entries),
                    "cache_metadata": sample_cache_metadata,
                    "cache_root": str(resolved_sample_cache_root),
                    "cache_schema_version": _TRAINING_SAMPLE_CACHE_SCHEMA_VERSION,
                    "entries": cache_entries,
                    "feature_count": len({pair.feature_id for pair in pairs}),
                    "row_count": int(table.shape[0]),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    metadata = {
        "cache_entry_count": len(cache_entries),
        "cache_hit_count": sum(1 for entry in cache_entries if entry["cache_status"] == "hit"),
        "cache_miss_count": sum(1 for entry in cache_entries if entry["cache_status"] == "miss"),
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
    if resolved_sample_cache_root is not None:
        metadata["sample_cache_metadata"] = sample_cache_metadata
        metadata["sample_cache_manifest_path"] = str(sample_cache_manifest_path)
        metadata["sample_cache_root"] = str(resolved_sample_cache_root)
    if len(pairs) == 1:
        metadata["feature_raster_path"] = str(pairs[0].feature_raster_path)
        metadata["label_raster_path"] = str(pairs[0].label_raster_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return TrainingTableResult(
        columns=feature_names,
        label_column="label_id",
        metadata_path=metadata_path,
        row_count=int(table.shape[0]),
        sample_cache_manifest_path=sample_cache_manifest_path,
        sample_cache_root=resolved_sample_cache_root,
        table_path=table_path,
    )


def _load_cache_entry_frame(
    cache_data_path: Path | str,
    cache_metadata_path: Path | str,
) -> tuple[tuple[str, ...], pd.DataFrame, dict[str, object]]:
    data_path = Path(cache_data_path)
    metadata_path = Path(cache_metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("sample_cache_schema_version") != _TRAINING_SAMPLE_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported sample cache schema in {metadata_path}: "
            f"{metadata.get('sample_cache_schema_version')!r}"
        )

    columns = metadata.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"Cached sample metadata is missing columns in {metadata_path}.")

    frame = pd.read_pickle(data_path)
    frame_columns = metadata.get("frame_columns")
    if not isinstance(frame_columns, list) or not frame_columns:
        frame_columns = list(columns) + ["feature_id", "source_image_id", "label_id"]
    expected_columns = [str(value) for value in frame_columns]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"Cached sample columns do not match metadata in {data_path}.")

    return tuple(str(value) for value in columns), frame, metadata


def _cache_metadata_id_to_label(metadata: dict[str, object]) -> dict[int, str]:
    payload = metadata.get("id_to_label")
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            "Cached samples are missing id_to_label metadata required for global label remapping."
        )
    normalized: dict[int, str] = {}
    for key, value in payload.items():
        normalized[int(key)] = str(value)
    return normalized


def build_training_table_from_cache_manifests(
    cache_manifest_paths: Sequence[Path | str],
    output_dir: Path | str,
) -> TrainingTableResult:
    """Materialize one training table from one or more cached-sample manifests."""

    manifest_paths = [Path(path) for path in cache_manifest_paths]
    if not manifest_paths:
        raise ValueError("At least one sample cache manifest path is required.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "training_table.pkl"
    metadata_path = output_dir / "training_table.json"
    label_mapping_path = output_dir / "labels.json"
    sample_cache_manifest_path = output_dir / "sample_cache_sources.json"

    entries_by_key: dict[str, dict[str, object]] = {}
    cache_manifest_metadata: list[dict[str, object]] = []
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Sample cache manifest has no entries: {manifest_path}")
        cache_manifest_metadata.append(
            {
                "cache_entry_count": payload.get("cache_entry_count"),
                "cache_metadata": payload.get("cache_metadata"),
                "cache_root": payload.get("cache_root"),
                "manifest_path": str(manifest_path),
            }
        )
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cache_key = entry.get("cache_key")
            data_path = entry.get("cache_data_path")
            metadata_cache_path = entry.get("cache_metadata_path")
            if not isinstance(data_path, str) or not isinstance(metadata_cache_path, str):
                continue
            dedupe_key = str(cache_key) if cache_key is not None else f"{data_path}|{metadata_cache_path}"
            entries_by_key.setdefault(
                dedupe_key,
                {
                    "cache_data_path": data_path,
                    "cache_key": cache_key,
                    "cache_metadata_path": metadata_cache_path,
                },
            )

    if not entries_by_key:
        raise ValueError("No valid sample-cache entries were found in the provided manifests.")

    loaded_entries: list[tuple[pd.DataFrame, dict[str, object]]] = []
    feature_names: tuple[str, ...] | None = None
    all_labels: set[str] = set()
    for entry in entries_by_key.values():
        pair_feature_names, frame, metadata = _load_cache_entry_frame(
            entry["cache_data_path"],
            entry["cache_metadata_path"],
        )
        if feature_names is None:
            feature_names = pair_feature_names
        elif pair_feature_names != feature_names:
            raise ValueError("Cached sample manifests do not share the same AlphaEarth band order.")
        all_labels.update(_cache_metadata_id_to_label(metadata).values())
        loaded_entries.append((frame, metadata))

    if feature_names is None:
        raise ValueError("No cached sample rows were loaded.")

    global_label_to_id = {label: idx for idx, label in enumerate(sorted(all_labels))}
    global_id_to_label = {str(idx): label for label, idx in sorted(global_label_to_id.items(), key=lambda item: item[1])}
    remapped_tables: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    for frame, metadata in loaded_entries:
        id_to_label = _cache_metadata_id_to_label(metadata)
        local_to_global = {
            local_id: global_label_to_id[label_name] for local_id, label_name in id_to_label.items()
        }
        remapped = frame.copy()
        remapped["label_id"] = remapped["label_id"].map(local_to_global).astype("int32")
        remapped_tables.append(remapped)
        sources.append(
            {
                "cache_key": metadata.get("cache_key"),
                "feature_id": metadata.get("feature_id"),
                "feature_raster_path": metadata.get("feature_raster_path"),
                "global_label_mapping": {
                    str(local_id): global_id for local_id, global_id in sorted(local_to_global.items())
                },
                "label_raster_path": metadata.get("label_raster_path"),
                "row_count": metadata.get("row_count"),
                "source_image_id": metadata.get("source_image_id"),
            }
        )

    table = pd.concat(remapped_tables, ignore_index=True)
    table.to_pickle(table_path)

    label_mapping_path.write_text(
        json.dumps(
            {
                "id_to_label": global_id_to_label,
                "label_column": "label_id",
                "label_to_id": global_label_to_id,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    sample_cache_manifest_path.write_text(
        json.dumps(
            {
                "cache_entry_count": len(entries_by_key),
                "cache_manifests": cache_manifest_metadata,
                "entries": list(entries_by_key.values()),
                "row_count": int(table.shape[0]),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    metadata_path.write_text(
        json.dumps(
            {
                "cache_entry_count": len(entries_by_key),
                "cache_hit_count": len(entries_by_key),
                "cache_miss_count": 0,
                "columns": list(feature_names),
                "feature_count": int(table["feature_id"].nunique()) if "feature_id" in table.columns else None,
                "feature_ids": sorted(str(value) for value in table["feature_id"].unique()),
                "id_to_label": global_id_to_label,
                "label_column": "label_id",
                "label_mapping_path": str(label_mapping_path),
                "label_to_id": global_label_to_id,
                "lineage_columns": ["feature_id", "source_image_id"],
                "row_count": int(table.shape[0]),
                "sample_cache_manifest_path": str(sample_cache_manifest_path),
                "sample_cache_metadata": cache_manifest_metadata,
                "sample_cache_root": None,
                "sources": sources,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return TrainingTableResult(
        columns=feature_names,
        label_column="label_id",
        metadata_path=metadata_path,
        row_count=int(table.shape[0]),
        sample_cache_manifest_path=sample_cache_manifest_path,
        sample_cache_root=None,
        table_path=table_path,
    )


def _sample_cache_manifest_paths_from_pipeline_payload(payload: dict[str, object]) -> tuple[Path, ...]:
    explicit_paths = payload.get("sample_cache_manifest_paths")
    if isinstance(explicit_paths, list):
        paths = [Path(value) for value in explicit_paths if isinstance(value, str)]
        if paths:
            return tuple(paths)
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("Pipeline manifest does not contain sample_cache_manifest_paths.")
    paths = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        sample_cache_manifest_path = feature.get("sample_cache_manifest_path")
        if isinstance(sample_cache_manifest_path, str):
            paths.append(Path(sample_cache_manifest_path))
    if not paths:
        raise ValueError("Pipeline manifest does not reference any sample_cache_manifest.json files.")
    return tuple(paths)


def _load_pipeline_pooling_inputs(
    pipeline_manifest_paths: Sequence[Path | str],
) -> tuple[tuple[Path, ...], tuple[Path, ...], dict[str, object], list[dict[str, object]]]:
    resolved_pipeline_paths = tuple(dict.fromkeys(Path(path).resolve() for path in pipeline_manifest_paths))
    if not resolved_pipeline_paths:
        raise ValueError("At least one pipeline manifest path is required.")

    compatibility: dict[str, object] | None = None
    sample_cache_manifest_paths: list[Path] = []
    pipeline_sources: list[dict[str, object]] = []
    for pipeline_manifest_path in resolved_pipeline_paths:
        payload = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
        label_mode = payload.get("label_mode")
        cache_paths = _sample_cache_manifest_paths_from_pipeline_payload(payload)
        pipeline_sources.append(
            {
                "aoi_label": payload.get("aoi_label"),
                "feature_count": payload.get("feature_count"),
                "label_mode": label_mode,
                "pipeline_manifest_path": str(pipeline_manifest_path),
                "reference_input_path": payload.get("reference_input_path"),
                "reference_path": payload.get("reference_path"),
                "sample_cache_manifest_paths": [str(path) for path in cache_paths],
                "sample_cache_root": payload.get("sample_cache_root"),
                "year": payload.get("year"),
            }
        )
        for cache_path in cache_paths:
            cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cache_metadata = cache_payload.get("cache_metadata")
            if not isinstance(cache_metadata, dict):
                raise ValueError(f"Sample cache manifest is missing cache_metadata: {cache_path}")
            if label_mode is not None and cache_metadata.get("label_mode") != label_mode:
                raise ValueError(
                    f"Pipeline label_mode does not match sample cache metadata for {cache_path}: "
                    f"{label_mode!r} != {cache_metadata.get('label_mode')!r}"
                )
            current = {
                "label_column": cache_metadata.get("label_column"),
                "label_mode": cache_metadata.get("label_mode"),
                "overlap_policy": cache_metadata.get("overlap_policy"),
                "reference_name": cache_metadata.get("reference_name"),
            }
            if compatibility is None:
                compatibility = current
            elif current != compatibility:
                raise ValueError(
                    "Pipeline manifests are not compatible for pooled training. "
                    f"Expected {compatibility}, got {current} from {cache_path}."
                )
            sample_cache_manifest_paths.append(cache_path.resolve())

    resolved_sample_cache_manifest_paths = tuple(dict.fromkeys(sample_cache_manifest_paths))
    if not resolved_sample_cache_manifest_paths:
        raise ValueError("No sample cache manifests were resolved from the provided pipeline manifests.")
    return (
        resolved_pipeline_paths,
        resolved_sample_cache_manifest_paths,
        compatibility or {},
        pipeline_sources,
    )


def train_pooled_model_from_pipeline_manifests(
    pipeline_manifest_paths: Sequence[Path | str],
    output_dir: Path | str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    n_jobs: int = -1,
    max_train_rows: int | None = None,
) -> PooledTrainingResult:
    """Build and train one pooled model from prior batch pipeline manifests."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"
    model_dir = output_dir / "model"
    pooled_manifest_path = output_dir / "pooled_training.json"

    (
        resolved_pipeline_paths,
        resolved_sample_cache_manifest_paths,
        compatibility,
        pipeline_sources,
    ) = _load_pipeline_pooling_inputs(pipeline_manifest_paths)

    training_table = build_training_table_from_cache_manifests(
        resolved_sample_cache_manifest_paths,
        dataset_dir,
    )
    dataset_label_mapping_path = dataset_dir / "labels.json"
    trained_model = train_random_forest(
        training_table.table_path,
        model_dir,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        max_train_rows=max_train_rows,
        label_mapping_path=dataset_label_mapping_path if dataset_label_mapping_path.exists() else None,
    )
    training_metadata = json.loads(training_table.metadata_path.read_text(encoding="utf-8"))
    pooled_manifest_path.write_text(
        json.dumps(
            {
                "compatibility": compatibility,
                "dataset_label_mapping_path": str(dataset_label_mapping_path),
                "metrics_path": str(trained_model.metrics_path),
                "model_path": str(trained_model.model_path),
                "model_parameters": {
                    "max_train_rows": max_train_rows,
                    "n_estimators": n_estimators,
                    "n_jobs": n_jobs,
                    "random_state": random_state,
                    "test_size": test_size,
                },
                "pipeline_manifest_paths": [str(path) for path in resolved_pipeline_paths],
                "pipeline_sources": pipeline_sources,
                "sample_cache_manifest_path": (
                    str(training_table.sample_cache_manifest_path)
                    if training_table.sample_cache_manifest_path is not None
                    else None
                ),
                "sample_cache_manifest_paths": [str(path) for path in resolved_sample_cache_manifest_paths],
                "training_row_count": training_table.row_count,
                "training_table_metadata_path": str(training_table.metadata_path),
                "training_table_path": str(training_table.table_path),
                "years": sorted(
                    {
                        int(source["year"])
                        for source in pipeline_sources
                        if isinstance(source.get("year"), int)
                    }
                ),
                "feature_ids": training_metadata.get("feature_ids"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return PooledTrainingResult(
        dataset_label_mapping_path=dataset_label_mapping_path,
        pipeline_manifest_paths=resolved_pipeline_paths,
        pooled_manifest_path=pooled_manifest_path,
        sample_cache_manifest_paths=resolved_sample_cache_manifest_paths,
        trained_model=trained_model,
        training_table=training_table,
    )


def train_random_forest(
    training_table_path: Path | str,
    output_dir: Path | str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    n_jobs: int = -1,
    max_train_rows: int | None = None,
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

    if table.empty:
        raise ValueError("Training table is empty.")
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero.")

    all_indices = np.arange(len(table))
    y = table[label_column]
    class_counts = y.value_counts().sort_index()

    evaluation_mode = "pixel_holdout"
    evaluation_note = (
        "Holdout metrics are computed from a pixel-level split within the provided training table. "
        "They are not tile-level validation metrics."
    )
    train_feature_ids: list[str] | None = None
    test_feature_ids: list[str] | None = None
    training_subsample: dict[str, Any] | None = None

    feature_groups = (
        table["feature_id"]
        if "feature_id" in table.columns and table["feature_id"].nunique() > 1
        else None
    )
    if len(class_counts) < 2:
        train_idx = all_indices
        test_idx = None
        evaluation_mode = "train_only_single_class"
        evaluation_note = (
            "Model fitted on the full training table because this tile exposes only one labeled class."
        )
        model = DummyClassifier(strategy="most_frequent")
    else:
        can_pixel_hold_out = (
            0.0 < test_size < 1.0
            and int(class_counts.min()) >= 2
            and int(np.ceil(len(y) * test_size)) >= len(class_counts)
        )
        if feature_groups is not None and 0.0 < test_size < 1.0:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(all_indices, y, groups=feature_groups))
            if y.iloc[train_idx].nunique() >= 2 and len(test_idx) > 0:
                evaluation_mode = "feature_holdout"
                evaluation_note = (
                    "Holdout metrics are computed by holding out whole native AlphaEarth feature rasters "
                    "grouped by feature_id."
                )
                train_feature_ids = sorted(str(value) for value in table.iloc[train_idx]["feature_id"].unique())
                test_feature_ids = sorted(str(value) for value in table.iloc[test_idx]["feature_id"].unique())
            elif can_pixel_hold_out:
                train_idx, test_idx = train_test_split(
                    all_indices,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y,
                )
            else:
                train_idx = all_indices
                test_idx = None
                evaluation_mode = "train_only_no_holdout"
                evaluation_note = (
                    "Model fitted on the full training table because neither feature-level nor pixel-level "
                    "holdout was valid for the available class counts."
                )
        elif can_pixel_hold_out:
            train_idx, test_idx = train_test_split(
                all_indices,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        else:
            train_idx = all_indices
            test_idx = None
            evaluation_mode = "train_only_no_holdout"
            evaluation_note = (
                "Model fitted on the full training table because the requested holdout split was not "
                "valid for the available class counts."
            )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    pre_subsample_train_row_count = int(len(train_idx))
    if max_train_rows is not None and max_train_rows <= 0:
        raise ValueError("max_train_rows must be positive when provided.")
    if max_train_rows is not None and len(train_idx) > max_train_rows:
        sampling_columns = [label_column]
        if "feature_id" in table.columns:
            sampling_columns.append("feature_id")
        train_frame = table.iloc[train_idx][sampling_columns].copy()
        if "feature_id" in train_frame.columns and train_frame["feature_id"].nunique() > 1:
            feature_counts = train_frame["feature_id"].value_counts().sort_index()
            proportional = feature_counts / feature_counts.sum() * int(max_train_rows)
            quotas = np.floor(proportional).astype(int)
            quotas = quotas.clip(lower=1, upper=feature_counts)
            remaining = int(max_train_rows) - int(quotas.sum())
            if remaining > 0:
                remainders = (proportional - quotas).sort_values(ascending=False)
                for feature_id in remainders.index:
                    if remaining <= 0:
                        break
                    if quotas.loc[feature_id] >= feature_counts.loc[feature_id]:
                        continue
                    quotas.loc[feature_id] += 1
                    remaining -= 1
            elif remaining < 0:
                remainders = (proportional - quotas).sort_values()
                for feature_id in remainders.index:
                    if remaining >= 0:
                        break
                    if quotas.loc[feature_id] <= 1:
                        continue
                    quotas.loc[feature_id] -= 1
                    remaining += 1

            sampled_frames: list[pd.DataFrame] = []
            for offset, (feature_id, quota) in enumerate(quotas.items()):
                feature_frame = train_frame.loc[train_frame["feature_id"] == feature_id]
                quota = int(min(max(quota, 1), len(feature_frame)))
                if quota >= len(feature_frame):
                    sampled_frames.append(feature_frame)
                    continue
                label_counts = feature_frame[label_column].value_counts()
                can_stratify = (
                    label_counts.min() >= 2
                    and quota >= label_counts.size
                    and (len(feature_frame) - quota) >= label_counts.size
                )
                if can_stratify:
                    sampled_index, _ = train_test_split(
                        feature_frame.index,
                        train_size=quota,
                        random_state=random_state + offset,
                        stratify=feature_frame[label_column],
                    )
                    sampled_frames.append(feature_frame.loc[sampled_index])
                else:
                    sampled_frames.append(
                        feature_frame.sample(n=quota, random_state=random_state + offset)
                    )
            sampled_train = pd.concat(sampled_frames, axis=0, ignore_index=False)
            sampled_train = sampled_train.sample(frac=1.0, random_state=random_state)
            sampling_strategy = "per_feature_label_stratified"
        else:
            label_counts = train_frame[label_column].value_counts()
            can_stratify = (
                label_counts.min() >= 2
                and int(max_train_rows) >= label_counts.size
                and (len(train_frame) - int(max_train_rows)) >= label_counts.size
            )
            if can_stratify:
                sampled_index, _ = train_test_split(
                    train_frame.index,
                    train_size=int(max_train_rows),
                    random_state=random_state,
                    stratify=train_frame[label_column],
                )
                sampled_train = train_frame.loc[sampled_index]
                sampling_strategy = "label_stratified"
            else:
                sampled_train = train_frame.sample(n=int(max_train_rows), random_state=random_state)
                sampling_strategy = "uniform_random"
        train_idx = sampled_train.index.to_numpy()
        training_subsample = {
            "applied": True,
            "input_row_count": pre_subsample_train_row_count,
            "max_train_rows": int(max_train_rows),
            "output_row_count": int(len(train_idx)),
            "strategy": sampling_strategy,
        }
        evaluation_note = (
            f"{evaluation_note} Training rows were capped at {int(max_train_rows)} via {sampling_strategy} "
            "sampling after the holdout split."
        )
    train_frame = table.iloc[train_idx]
    X_train = train_frame[feature_names]
    y_train = train_frame[label_column]
    fit_row_count = int(len(X_train))
    model.fit(X_train, y_train)
    if test_idx is not None:
        del X_train
        del train_frame
        test_frame = table.iloc[test_idx]
        X_test = test_frame[feature_names]
        y_test = test_frame[label_column]
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification = classification_report(y_test, y_pred, output_dict=True)
        macro_f1 = classification.get("macro avg", {}).get("f1-score")
        weighted_f1 = classification.get("weighted avg", {}).get("f1-score")
        training_accuracy = None
    else:
        y_pred = model.predict(X_train)
        accuracy = None
        classification = None
        macro_f1 = None
        weighted_f1 = None
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
        "fit_row_count": fit_row_count,
        "holdout_row_count": int(len(test_idx)) if test_idx is not None else 0,
        "label_column": label_column,
        "label_mapping": label_mapping,
        "macro_f1": macro_f1,
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "n_jobs": n_jobs,
        "pre_subsample_train_row_count": pre_subsample_train_row_count,
        "row_count": int(table.shape[0]),
        "test_feature_ids": test_feature_ids,
        "test_size": test_size,
        "train_feature_ids": train_feature_ids,
        "training_subsample": training_subsample,
        "training_accuracy": training_accuracy,
        "weighted_f1": weighted_f1,
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
    parser.add_argument(
        "--label-mapping",
        default=None,
        help="Optional labels.json sidecar to persist id_to_label metadata in cached samples.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for the training table outputs.")
    parser.add_argument(
        "--sample-cache-root",
        default=None,
        help="Optional shared cache root for reusable extracted training samples.",
    )
    return parser


def build_train_model_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a crop classifier from a training table.")
    parser.add_argument("--training-table", required=True, help="Path to a pickled training table.")
    parser.add_argument("--output-dir", required=True, help="Directory for model outputs.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Evaluation holdout fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Random forest tree count.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="CPU parallelism passed to RandomForestClassifier. Use the allocated Slurm CPUs for cluster runs.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap on the number of training rows used to fit the model after holdout splitting.",
    )
    parser.add_argument(
        "--label-mapping",
        default=None,
        help="Optional labels.json sidecar to persist id_to_label metadata in the model bundle.",
    )
    return parser


def build_training_table_from_cache_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a training table by combining one or more cached-sample manifests."
    )
    parser.add_argument(
        "--cache-manifest",
        "--sample-cache-manifest",
        dest="sample_cache_manifest",
        action="append",
        required=True,
        help="Path to one sample_cache_manifest.json produced by a prior pipeline run.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for the combined training table outputs.")
    return parser


def build_train_pooled_model_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and train one pooled model from one or more batch pipeline manifests."
    )
    parser.add_argument(
        "--pipeline-manifest",
        action="append",
        required=True,
        help="Path to one pipeline.json produced by a prior run-baseline-pipeline or download-run-baseline run.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for pooled dataset and model outputs.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Evaluation holdout fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Random forest tree count.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="CPU parallelism passed to RandomForestClassifier for the pooled model fit.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap on the number of training rows used to fit the pooled model after holdout splitting.",
    )
    return parser


def main_build_training_table(argv: list[str] | None = None) -> int:
    parser = build_training_table_parser()
    args = parser.parse_args(argv)
    result = build_training_table(
        args.feature_raster,
        args.label_raster,
        args.output_dir,
        label_mapping_path=args.label_mapping,
        sample_cache_root=args.sample_cache_root,
    )
    print(
        json.dumps(
            {
                "columns": list(result.columns),
                "label_column": result.label_column,
                "metadata_path": str(result.metadata_path),
                "row_count": result.row_count,
                "sample_cache_manifest_path": (
                    str(result.sample_cache_manifest_path)
                    if result.sample_cache_manifest_path is not None
                    else None
                ),
                "sample_cache_root": (
                    str(result.sample_cache_root) if result.sample_cache_root is not None else None
                ),
                "table_path": str(result.table_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main_build_training_table_from_cache(argv: list[str] | None = None) -> int:
    parser = build_training_table_from_cache_parser()
    args = parser.parse_args(argv)
    result = build_training_table_from_cache_manifests(args.sample_cache_manifest, args.output_dir)
    print(
        json.dumps(
            {
                "columns": list(result.columns),
                "label_column": result.label_column,
                "label_mapping_path": str(Path(args.output_dir) / "labels.json"),
                "metadata_path": str(result.metadata_path),
                "row_count": result.row_count,
                "sample_cache_manifest_path": (
                    str(result.sample_cache_manifest_path)
                    if result.sample_cache_manifest_path is not None
                    else None
                ),
                "sample_cache_root": None,
                "table_path": str(result.table_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main_train_pooled_model(argv: list[str] | None = None) -> int:
    parser = build_train_pooled_model_parser()
    args = parser.parse_args(argv)
    result = train_pooled_model_from_pipeline_manifests(
        args.pipeline_manifest,
        args.output_dir,
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
                "row_count": result.trained_model.row_count,
                "sample_cache_manifest_paths": [str(path) for path in result.sample_cache_manifest_paths],
                "training_metadata_path": str(result.training_table.metadata_path),
                "training_table_path": str(result.training_table.table_path),
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
        n_jobs=args.n_jobs,
        max_train_rows=args.max_train_rows,
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
