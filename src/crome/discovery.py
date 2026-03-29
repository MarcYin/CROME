"""Discover native AlphaEarth feature rasters from files, directories, or manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .features import read_feature_raster_spec
from .paths import feature_artifact_name

_MANIFEST_PATH_KEYS = (
    "path",
    "raster_path",
    "file_path",
    "output_path",
    "local_path",
    "filename",
    "file",
    "href",
)
_MANIFEST_ID_KEYS = ("source_image_id", "image_id", "ee_image_id", "ee_id", "id")
_MANIFEST_FEATURE_KEYS = ("feature_id", "tile_id", "name", "stem")


@dataclass(frozen=True, slots=True)
class DiscoveredFeatureRaster:
    """One native AlphaEarth raster ready for downstream processing."""

    feature_id: str
    raster_path: Path
    source_image_id: str | None = None


def _iter_manifest_entries(node: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if isinstance(node, dict):
        entries.append(node)
        for value in node.values():
            entries.extend(_iter_manifest_entries(value))
    elif isinstance(node, list | tuple):
        for value in node:
            entries.extend(_iter_manifest_entries(value))
    return entries


def _resolve_manifest_path(candidate: Any, manifest_path: Path) -> Path | None:
    if not isinstance(candidate, str):
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def _build_feature_record(
    raster_path: Path,
    *,
    feature_id: str | None = None,
    source_image_id: str | None = None,
) -> DiscoveredFeatureRaster | None:
    if not raster_path.exists() or not raster_path.is_file():
        return None
    try:
        read_feature_raster_spec(raster_path)
    except Exception:
        return None
    return DiscoveredFeatureRaster(
        feature_id=feature_artifact_name(feature_id or raster_path),
        raster_path=raster_path,
        source_image_id=source_image_id,
    )


def _discover_from_manifest(manifest_path: Path) -> list[DiscoveredFeatureRaster]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    discovered: dict[Path, DiscoveredFeatureRaster] = {}
    for entry in _iter_manifest_entries(payload):
        resolved_path = None
        for key in _MANIFEST_PATH_KEYS:
            resolved_path = _resolve_manifest_path(entry.get(key), manifest_path)
            if resolved_path is not None:
                break
        if resolved_path is None:
            continue

        source_image_id = next(
            (value for key in _MANIFEST_ID_KEYS if isinstance((value := entry.get(key)), str)),
            None,
        )
        manifest_feature_id = next(
            (value for key in _MANIFEST_FEATURE_KEYS if isinstance((value := entry.get(key)), str)),
            None,
        )
        record = _build_feature_record(
            resolved_path,
            feature_id=manifest_feature_id,
            source_image_id=source_image_id,
        )
        if record is not None:
            discovered[record.raster_path] = record
    return sorted(discovered.values(), key=lambda item: str(item.raster_path))


def _discover_from_path(feature_input: Path) -> list[DiscoveredFeatureRaster]:
    if feature_input.is_file():
        record = _build_feature_record(feature_input)
        return [record] if record is not None else []

    if not feature_input.is_dir():
        raise FileNotFoundError(f"Feature input does not exist: {feature_input}")

    discovered: dict[Path, DiscoveredFeatureRaster] = {}
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        for candidate in sorted(feature_input.rglob(pattern)):
            record = _build_feature_record(candidate)
            if record is not None:
                discovered[record.raster_path] = record
    return sorted(discovered.values(), key=lambda item: str(item.raster_path))


def discover_feature_rasters(
    *,
    feature_input: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> tuple[DiscoveredFeatureRaster, ...]:
    """Discover native AlphaEarth feature rasters.

    If a manifest is provided, discovery first attempts to resolve feature rasters from
    the manifest contents and then falls back to filesystem scanning.
    """

    if feature_input is None and manifest_path is None:
        raise ValueError("Provide at least one of feature_input or manifest_path.")

    discovered: dict[Path, DiscoveredFeatureRaster] = {}

    resolved_manifest_path = Path(manifest_path) if manifest_path is not None else None
    if resolved_manifest_path is not None:
        if not resolved_manifest_path.exists():
            raise FileNotFoundError(f"Manifest path does not exist: {resolved_manifest_path}")
        for record in _discover_from_manifest(resolved_manifest_path):
            discovered[record.raster_path] = record

    if feature_input is not None:
        resolved_feature_input = Path(feature_input)
    elif resolved_manifest_path is not None:
        resolved_feature_input = resolved_manifest_path.parent
    else:
        raise ValueError("A feature input path could not be resolved.")

    for record in _discover_from_path(resolved_feature_input):
        discovered.setdefault(record.raster_path, record)

    if not discovered:
        raise ValueError(
            "No AlphaEarth feature rasters were found. Provide a feature raster, a directory "
            "containing native AlphaEarth GeoTIFFs, or a valid manifest path."
        )

    return tuple(sorted(discovered.values(), key=lambda item: (item.feature_id, str(item.raster_path))))
