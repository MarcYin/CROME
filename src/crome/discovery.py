"""Discover native AlphaEarth feature rasters from files, directories, or manifests."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .features import read_feature_raster_spec
from .paths import feature_artifact_name, feature_tile_name

_MANIFEST_PATH_KEYS = (
    "path",
    "raster_path",
    "file_path",
    "output_path",
    "relative_tiff_path",
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


def _parse_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if 1000 <= value <= 9999:
            return value
        if value > 10_000_000_000:
            return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).year
        if value > 100_000_000:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).year
        return None
    if isinstance(value, float):
        return _parse_year(int(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if len(text) == 4 and text.isdigit():
            return int(text)
        if text.isdigit():
            return _parse_year(int(text))
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).year
        except ValueError:
            return None
    return None


def _extract_requested_year(payload: dict[str, Any], requested_year: int | None) -> int | None:
    if requested_year is not None:
        return requested_year
    config_payload = payload.get("config")
    if isinstance(config_payload, dict):
        return _parse_year(config_payload.get("start_date"))
    search_payload = payload.get("search")
    if isinstance(search_payload, dict):
        return _parse_year(search_payload.get("start_date"))
    return None


def _sidecar_metadata_path(raster_path: Path) -> Path:
    return raster_path.with_suffix(f"{raster_path.suffix}.metadata.json")


def _image_year_from_metadata(node: dict[str, Any]) -> int | None:
    for key in ("acquisition_time_utc", "local_datetime", "start_date", "year"):
        year = _parse_year(node.get(key))
        if year is not None:
            return year

    properties = node.get("properties")
    if isinstance(properties, dict):
        for key in ("system:time_start", "time_start"):
            year = _parse_year(properties.get(key))
            if year is not None:
                return year

    raw_image_info = node.get("raw_image_info")
    if isinstance(raw_image_info, dict):
        year = _image_year_from_metadata(raw_image_info)
        if year is not None:
            return year

    return None


def _raster_year(raster_path: Path, manifest_entry: dict[str, Any] | None = None) -> int | None:
    if isinstance(manifest_entry, dict):
        year = _image_year_from_metadata(manifest_entry)
        if year is not None:
            return year

    sidecar_path = _sidecar_metadata_path(raster_path)
    if not sidecar_path.exists():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return _image_year_from_metadata(payload)
    return None


def _matches_requested_year(
    raster_path: Path,
    *,
    requested_year: int | None,
    manifest_entry: dict[str, Any] | None = None,
) -> bool:
    if requested_year is None:
        return True
    raster_year = _raster_year(raster_path, manifest_entry=manifest_entry)
    return raster_year is None or raster_year == requested_year


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


def _resolve_manifest_path(
    candidate: Any,
    manifest_path: Path,
    *,
    base_path: Path | None = None,
) -> Path | None:
    if not isinstance(candidate, str):
        return None
    path = Path(candidate)
    if not path.is_absolute():
        root = base_path if base_path is not None else manifest_path.parent
        path = root / path
    return path


def _manifest_output_root(payload: dict[str, Any], manifest_path: Path) -> Path:
    for key in ("download", "config"):
        node = payload.get(key)
        if not isinstance(node, dict):
            continue
        output_root = node.get("output_root")
        resolved = _resolve_manifest_path(output_root, manifest_path)
        if resolved is not None:
            return resolved
    return manifest_path.parent


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


def _discover_from_edown_manifest(
    payload: dict[str, Any],
    manifest_path: Path,
    *,
    manifest_root: Path,
    requested_year: int | None = None,
) -> tuple[list[DiscoveredFeatureRaster], set[Path]]:
    discovered: dict[Path, DiscoveredFeatureRaster] = {}
    blocked_paths: set[Path] = set()
    search_payload = payload.get("search")
    download_payload = payload.get("download")
    if not isinstance(search_payload, dict) and not isinstance(download_payload, dict):
        return [], blocked_paths

    effective_requested_year = _extract_requested_year(payload, requested_year)

    search_images: dict[str, dict[str, Any]] = {}
    if isinstance(search_payload, dict):
        images = search_payload.get("images")
        if isinstance(images, list):
            for image in images:
                if not isinstance(image, dict):
                    continue
                image_id = image.get("image_id")
                if isinstance(image_id, str):
                    search_images[image_id] = image

    if isinstance(download_payload, dict):
        results = download_payload.get("results")
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict):
                    continue
                image_id = result.get("image_id")
                if not isinstance(image_id, str):
                    continue
                status = result.get("status")
                if isinstance(status, str) and status == "failed":
                    continue
                resolved_path = _resolve_manifest_path(
                    result.get("tiff_path"),
                    manifest_path,
                    base_path=manifest_root,
                )
                if resolved_path is None:
                    search_entry = search_images.get(image_id, {})
                    resolved_path = _resolve_manifest_path(
                        search_entry.get("relative_tiff_path"),
                        manifest_path,
                        base_path=manifest_root,
                    )
                if resolved_path is None:
                    continue
                search_entry = search_images.get(image_id)
                if not _matches_requested_year(
                    resolved_path,
                    requested_year=effective_requested_year,
                    manifest_entry=search_entry,
                ):
                    blocked_paths.add(resolved_path)
                    continue
                record = _build_feature_record(
                    resolved_path,
                    source_image_id=image_id,
                )
                if record is not None:
                    discovered[record.raster_path] = record

    if discovered:
        return sorted(discovered.values(), key=lambda item: str(item.raster_path)), blocked_paths

    for image_id, image in search_images.items():
        resolved_path = _resolve_manifest_path(
            image.get("relative_tiff_path"),
            manifest_path,
            base_path=manifest_root,
        )
        if resolved_path is None:
            continue
        if not _matches_requested_year(
            resolved_path,
            requested_year=effective_requested_year,
            manifest_entry=image,
        ):
            blocked_paths.add(resolved_path)
            continue
        record = _build_feature_record(
            resolved_path,
            source_image_id=image_id,
        )
        if record is not None:
            discovered[record.raster_path] = record

    return sorted(discovered.values(), key=lambda item: str(item.raster_path)), blocked_paths


def _discover_from_manifest(
    manifest_path: Path,
    *,
    requested_year: int | None = None,
) -> tuple[list[DiscoveredFeatureRaster], Path, set[Path]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_root = _manifest_output_root(payload, manifest_path)
    discovered: dict[Path, DiscoveredFeatureRaster] = {}
    blocked_paths: set[Path] = set()
    manifest_records, blocked_paths = _discover_from_edown_manifest(
        payload,
        manifest_path,
        manifest_root=manifest_root,
        requested_year=requested_year,
    )
    for record in manifest_records:
        discovered[record.raster_path] = record
    for entry in _iter_manifest_entries(payload):
        resolved_path = None
        for key in _MANIFEST_PATH_KEYS:
            resolved_path = _resolve_manifest_path(
                entry.get(key),
                manifest_path,
                base_path=manifest_root,
            )
            if resolved_path is not None:
                break
        if resolved_path is None or resolved_path in blocked_paths:
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
    return sorted(discovered.values(), key=lambda item: str(item.raster_path)), manifest_root, blocked_paths


def _discover_from_path(
    feature_input: Path,
    *,
    requested_year: int | None = None,
    blocked_paths: set[Path] | None = None,
) -> list[DiscoveredFeatureRaster]:
    blocked = blocked_paths or set()
    if feature_input.is_file():
        if feature_input in blocked or not _matches_requested_year(
            feature_input,
            requested_year=requested_year,
        ):
            return []
        record = _build_feature_record(feature_input)
        return [record] if record is not None else []

    if not feature_input.is_dir():
        raise FileNotFoundError(f"Feature input does not exist: {feature_input}")

    discovered: dict[Path, DiscoveredFeatureRaster] = {}
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        for candidate in sorted(feature_input.rglob(pattern)):
            if candidate in blocked or not _matches_requested_year(
                candidate,
                requested_year=requested_year,
            ):
                continue
            record = _build_feature_record(candidate)
            if record is not None:
                discovered[record.raster_path] = record
    return sorted(discovered.values(), key=lambda item: str(item.raster_path))


def discover_feature_rasters(
    *,
    feature_input: Path | str | None = None,
    manifest_path: Path | str | None = None,
    requested_year: int | None = None,
) -> tuple[DiscoveredFeatureRaster, ...]:
    """Discover native AlphaEarth feature rasters.

    If a manifest is provided, discovery first attempts to resolve feature rasters from
    the manifest contents and then falls back to filesystem scanning.
    """

    if feature_input is None and manifest_path is None:
        raise ValueError("Provide at least one of feature_input or manifest_path.")

    discovered: dict[Path, DiscoveredFeatureRaster] = {}
    manifest_root: Path | None = None
    blocked_paths: set[Path] = set()

    resolved_manifest_path = Path(manifest_path) if manifest_path is not None else None
    if resolved_manifest_path is not None:
        if not resolved_manifest_path.exists():
            raise FileNotFoundError(f"Manifest path does not exist: {resolved_manifest_path}")
        manifest_records, manifest_root, blocked_paths = _discover_from_manifest(
            resolved_manifest_path,
            requested_year=requested_year,
        )
        for record in manifest_records:
            discovered[record.raster_path] = record

    if feature_input is not None:
        resolved_feature_input = Path(feature_input)
    elif manifest_root is not None:
        resolved_feature_input = manifest_root
    else:
        raise ValueError("A feature input path could not be resolved.")

    for record in _discover_from_path(
        resolved_feature_input,
        requested_year=requested_year,
        blocked_paths=blocked_paths,
    ):
        discovered.setdefault(record.raster_path, record)

    if not discovered:
        raise ValueError(
            "No AlphaEarth feature rasters were found. Provide a feature raster, a directory "
            "containing native AlphaEarth GeoTIFFs, or a valid manifest path."
        )

    return tuple(sorted(discovered.values(), key=lambda item: (item.feature_id, str(item.raster_path))))


def discovered_feature_payload(
    *,
    feature_input: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> list[dict[str, str | None]]:
    """Return discovered feature rasters as JSON/TSV-friendly records."""

    payload: list[dict[str, str | None]] = []
    for record in discover_feature_rasters(feature_input=feature_input, manifest_path=manifest_path):
        payload.append(
            {
                "feature_id": record.feature_id,
                "tile_id": feature_tile_name(
                    feature_id=record.feature_id,
                    source_image_id=record.source_image_id,
                    feature_raster_path=record.raster_path,
                ),
                "source_image_id": record.source_image_id,
                "raster_path": str(record.raster_path),
            }
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List discovered AlphaEarth feature rasters from a manifest or feature-input root."
    )
    parser.add_argument(
        "--feature-input",
        default=None,
        help="Path to one AlphaEarth feature raster or a directory tree containing GeoTIFFs.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Path to an edown manifest; discovery falls back to the manifest directory.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "jsonl", "tsv"),
        default="json",
        help="Output format for discovered features.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional file path for the rendered discovery payload.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.feature_input is None and args.manifest_path is None:
        parser.error("Provide at least one of --feature-input or --manifest-path.")
    payload = discovered_feature_payload(
        feature_input=args.feature_input,
        manifest_path=args.manifest_path,
    )
    rendered = ""
    if args.format == "tsv":
        rendered = "feature_id\ttile_id\tsource_image_id\traster_path\n"
        for item in payload:
            rendered += (
                "\t".join(
                    (
                        item["feature_id"] or "",
                        item["tile_id"] or "",
                        item["source_image_id"] or "",
                        item["raster_path"] or "",
                    )
                )
                + "\n"
            )
    elif args.format == "jsonl":
        rendered = "\n".join(json.dumps(item, sort_keys=True) for item in payload)
        if rendered:
            rendered += "\n"
    else:
        rendered = json.dumps({"feature_count": len(payload), "features": payload}, indent=2, sort_keys=True)
        rendered += "\n"
    if args.output_path is not None:
        Path(args.output_path).write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
