"""AlphaEarth download helpers for the first migration slice."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from crome.config import AlphaEarthDownloadRequest
from crome.discovery import discover_feature_rasters
from crome.paths import OUTPUT_ROOT_ENV_VAR, default_output_root


class NoCoverageError(LookupError):
    """Raised when a download run resolves to no intersecting imagery."""


class DownloadFailedError(RuntimeError):
    """Raised when edown resolves imagery but fails to produce usable rasters."""


@dataclass(frozen=True, slots=True)
class AlphaEarthDownloadResult:
    """Small result surface for the first migration slice."""

    aoi_label: str
    bands: tuple[str, ...]
    collection_id: str
    conditional_year: bool
    manifest_path: Path | None
    output_root: Path
    source_image_ids: tuple[str, ...]
    year: int


def _load_edown() -> Any:
    try:
        from edown import AOI, DownloadConfig, download_images
    except ImportError as exc:
        raise RuntimeError(
            "edown is required for AlphaEarth downloads. Install `crome[ee]` "
            "or `pip install edown>=0.2.0` in the active environment."
        ) from exc

    return SimpleNamespace(AOI=AOI, DownloadConfig=DownloadConfig, download_images=download_images)


def download_result_to_dict(result: AlphaEarthDownloadResult) -> dict[str, Any]:
    """Return a JSON-safe summary payload for one AlphaEarth download run."""

    return {
        "aoi_label": result.aoi_label,
        "bands": list(result.bands),
        "collection_id": result.collection_id,
        "conditional_year": result.conditional_year,
        "manifest_path": str(result.manifest_path) if result.manifest_path else None,
        "output_root": str(result.output_root),
        "source_image_ids": list(result.source_image_ids),
        "year": result.year,
    }


def _build_aoi(request: AlphaEarthDownloadRequest, edown_module: Any) -> Any:
    aoi_cls = edown_module.AOI
    if request.bbox is not None:
        return aoi_cls.from_bbox(request.bbox)

    for constructor_name in ("from_geojson_path", "from_geojson", "from_geojson_file"):
        constructor = getattr(aoi_cls, constructor_name, None)
        if constructor is not None:
            return constructor(request.geojson)

    raise RuntimeError(
        "The installed edown build does not expose a geojson AOI constructor. "
        "Use --bbox or upgrade edown."
    )


def build_download_config(
    request: AlphaEarthDownloadRequest,
    edown_module: Any | None = None,
) -> Any:
    """Build an edown DownloadConfig for one AlphaEarth request."""

    module = edown_module or _load_edown()
    aoi = _build_aoi(request, module)
    return module.DownloadConfig(
        collection_id=request.collection_id,
        start_date=request.start_date,
        end_date=request.end_date,
        aoi=aoi,
        bands=request.bands,
        output_root=request.dataset_output_root,
    )


def _extract_image_count(summary: Any) -> int | None:
    downloaded = getattr(summary, "downloaded", None)
    skipped = getattr(summary, "skipped", None)
    failed = getattr(summary, "failed", None)
    if any(isinstance(value, int) for value in (downloaded, skipped, failed)):
        return sum(value for value in (downloaded, skipped, failed) if isinstance(value, int))

    for attribute in (
        "downloaded_images",
        "downloaded_image_count",
        "discovered_images",
        "discovered_image_count",
        "image_count",
    ):
        value = getattr(summary, attribute, None)
        if isinstance(value, int):
            return value
        if isinstance(value, (list, tuple)):
            return len(value)
    return None


def _extract_image_ids(summary: Any) -> tuple[str, ...]:
    image_ids: list[str] = []

    def append_candidate(value: Any) -> None:
        if isinstance(value, str):
            image_ids.append(value)
            return
        if isinstance(value, dict):
            for key in ("image_id", "id", "ee_id"):
                candidate = value.get(key)
                if isinstance(candidate, str):
                    image_ids.append(candidate)
                    return
            return
        for attribute in ("image_id", "id", "ee_id"):
            candidate = getattr(value, attribute, None)
            if isinstance(candidate, str):
                image_ids.append(candidate)
                return

    for attribute in ("source_image_ids", "image_ids", "images", "discovered_images", "downloaded_images"):
        value = getattr(summary, attribute, None)
        if isinstance(value, (list, tuple)):
            for item in value:
                append_candidate(item)

    results = getattr(summary, "results", None)
    if isinstance(results, (list, tuple)):
        for item in results:
            append_candidate(item)

    # Preserve order while removing duplicates.
    return tuple(dict.fromkeys(image_ids))


def _extract_download_statuses(summary: Any) -> tuple[str, ...]:
    results = getattr(summary, "results", None)
    if not isinstance(results, (list, tuple)):
        return ()

    statuses: list[str] = []
    for item in results:
        status = getattr(item, "status", None)
        if isinstance(status, str):
            statuses.append(status)
        elif isinstance(item, dict):
            candidate = item.get("status")
            if isinstance(candidate, str):
                statuses.append(candidate)
    return tuple(statuses)


def _extract_download_errors(summary: Any) -> tuple[str, ...]:
    results = getattr(summary, "results", None)
    if not isinstance(results, (list, tuple)):
        return ()

    errors: list[str] = []
    for item in results:
        error = getattr(item, "error", None)
        if isinstance(error, str) and error:
            errors.append(error)
        elif isinstance(item, dict):
            candidate = item.get("error")
            if isinstance(candidate, str) and candidate:
                errors.append(candidate)
    return tuple(errors)


def _successful_download_count(summary: Any) -> int | None:
    downloaded = getattr(summary, "downloaded", None)
    if isinstance(downloaded, int):
        skipped = getattr(summary, "skipped", None)
        skipped_existing = 0
        if isinstance(skipped, int):
            # edown aggregates all skipped statuses; existing files are still usable outputs.
            statuses = _extract_download_statuses(summary)
            skipped_existing = sum(1 for status in statuses if status == "skipped_existing")
        return downloaded + skipped_existing

    statuses = _extract_download_statuses(summary)
    if statuses:
        return sum(1 for status in statuses if status in {"downloaded", "skipped_existing"})

    return None


def _raise_download_failure(request: AlphaEarthDownloadRequest, summary: Any) -> None:
    statuses = _extract_download_statuses(summary)
    errors = _extract_download_errors(summary)
    lowered = " | ".join(errors).lower()

    if statuses and all(status == "skipped_outside_aoi" for status in statuses):
        raise NoCoverageError(
            f"No AlphaEarth imagery intersected AOI '{request.aoi_label}' for year {request.year}."
        )

    hint = ""
    if "disk quota exceeded" in lowered or "no space left on device" in lowered:
        hint = (
            f" Output root '{request.dataset_output_root}' ran out of writable space. "
            "Use a smaller AOI or a filesystem with more free space."
        )

    if statuses:
        raise DownloadFailedError(
            "AlphaEarth download did not produce any usable rasters. "
            f"Observed statuses: {sorted(set(statuses))}." + hint
        )

    raise DownloadFailedError(
        "AlphaEarth download failed before usable rasters were produced." + hint
    )


def _filter_source_image_ids_for_requested_year(
    manifest_path: Path | None,
    *,
    requested_year: int,
    source_image_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if manifest_path is None or not manifest_path.exists() or not source_image_ids:
        return source_image_ids

    try:
        discovered = discover_feature_rasters(manifest_path=manifest_path, requested_year=requested_year)
    except ValueError:
        return ()
    allowed_image_ids = {
        record.source_image_id for record in discovered if isinstance(record.source_image_id, str)
    }
    if not allowed_image_ids:
        return ()
    return tuple(image_id for image_id in source_image_ids if image_id in allowed_image_ids)


def download_alphaearth_images(
    request: AlphaEarthDownloadRequest,
    edown_module: Any | None = None,
) -> AlphaEarthDownloadResult:
    """Run one AlphaEarth download through edown."""

    module = edown_module or _load_edown()
    config = build_download_config(request, module)
    summary = module.download_images(config)

    image_count = _extract_image_count(summary)
    if image_count == 0:
        raise NoCoverageError(
            f"No AlphaEarth imagery intersected AOI '{request.aoi_label}' for year {request.year}."
        )
    successful_count = _successful_download_count(summary)
    if successful_count == 0:
        _raise_download_failure(request, summary)

    manifest_value = getattr(summary, "manifest_path", None)
    manifest_path = Path(manifest_value) if manifest_value is not None else None
    source_image_ids = _filter_source_image_ids_for_requested_year(
        manifest_path,
        requested_year=request.year,
        source_image_ids=_extract_image_ids(summary),
    )
    if not source_image_ids:
        raise DownloadFailedError(
            "AlphaEarth download resolved imagery, but none of the discovered images matched "
            f"requested year {request.year}."
        )

    return AlphaEarthDownloadResult(
        aoi_label=request.aoi_label or "aoi",
        bands=request.bands,
        collection_id=request.collection_id,
        conditional_year=request.conditional_year,
        manifest_path=manifest_path,
        output_root=request.dataset_output_root,
        source_image_ids=source_image_ids,
        year=request.year,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download AlphaEarth annual embeddings.")
    parser.add_argument("--year", required=True, type=int, help="Target annual layer year.")
    parser.add_argument(
        "--aoi-label",
        dest="aoi_label",
        default=None,
        help="Optional run label used only in local naming. This is not an AlphaEarth tile identifier.",
    )
    parser.add_argument("--tile-id", dest="aoi_label", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-root",
        default=default_output_root(),
        help=(
            f"Base directory for AlphaEarth outputs. Defaults to ${OUTPUT_ROOT_ENV_VAR} "
            "when set, otherwise data/alphaearth."
        ),
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="AOI bbox in lon/lat coordinates.",
    )
    parser.add_argument(
        "--geojson",
        default=None,
        help="Path to an AOI GeoJSON file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved request as JSON without importing edown or downloading.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    request = AlphaEarthDownloadRequest(
        year=args.year,
        output_root=args.output_root,
        aoi_label=args.aoi_label,
        bbox=tuple(args.bbox) if args.bbox is not None else None,
        geojson=args.geojson,
    )

    if request.conditional_year:
        print(
            "Warning: 2025 AlphaEarth coverage is documented as rolling by UTM zone as of "
            "January 29, 2026.",
            file=sys.stderr,
        )

    if args.dry_run:
        print(json.dumps(request.to_dict(), indent=2, sort_keys=True))
        return 0

    result = download_alphaearth_images(request)
    print(json.dumps(download_result_to_dict(result), indent=2, sort_keys=True))
    return 0
