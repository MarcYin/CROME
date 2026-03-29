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


class NoCoverageError(LookupError):
    """Raised when a download run resolves to no intersecting imagery."""


@dataclass(frozen=True, slots=True)
class AlphaEarthDownloadResult:
    """Small result surface for the first migration slice."""

    bands: tuple[str, ...]
    collection_id: str
    conditional_year: bool
    manifest_path: Path | None
    output_root: Path
    tile_id: str
    year: int


def _load_edown() -> Any:
    try:
        from edown import AOI, DownloadConfig, download_images
    except ImportError as exc:
        raise RuntimeError(
            "edown is required for AlphaEarth downloads. Install `crome[ee]` "
            "with Python 3.12 or earlier."
        ) from exc

    return SimpleNamespace(AOI=AOI, DownloadConfig=DownloadConfig, download_images=download_images)


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
            f"No AlphaEarth imagery intersected tile '{request.tile_id}' for year {request.year}."
        )

    manifest_value = getattr(summary, "manifest_path", None)
    manifest_path = Path(manifest_value) if manifest_value is not None else None

    return AlphaEarthDownloadResult(
        bands=request.bands,
        collection_id=request.collection_id,
        conditional_year=request.conditional_year,
        manifest_path=manifest_path,
        output_root=request.dataset_output_root,
        tile_id=request.tile_id or "aoi",
        year=request.year,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download AlphaEarth annual embeddings.")
    parser.add_argument("--year", required=True, type=int, help="Target annual layer year.")
    parser.add_argument(
        "--tile-id",
        default=None,
        help="Optional AOI label used only in output naming for this first slice.",
    )
    parser.add_argument(
        "--output-root",
        default="data/alphaearth",
        help="Base directory for AlphaEarth outputs.",
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
        tile_id=args.tile_id,
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
    payload = {
        "bands": list(result.bands),
        "collection_id": result.collection_id,
        "conditional_year": result.conditional_year,
        "manifest_path": str(result.manifest_path) if result.manifest_path else None,
        "output_root": str(result.output_root),
        "tile_id": result.tile_id,
        "year": result.year,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
