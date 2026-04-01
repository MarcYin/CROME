"""CROME reference discovery and download helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

import pyogrio
from rasterio.warp import transform_bounds

from crome.config import CromeDownloadRequest
from crome.paths import (
    OUTPUT_ROOT_ENV_VAR,
    crome_archive_path,
    crome_download_root,
    crome_extract_root,
    crome_normalized_root,
    default_output_root,
    sanitize_label,
)
from crome.runtime import ensure_proj_data_env

_NEXT_DATA_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)
_TITLE_PATTERN = re.compile(
    r"^Crop Map of England \(CROME\) (?P<year>\d{4})(?: - (?P<variant>.+))?$",
    re.IGNORECASE,
)
_USER_AGENT = "crome/0.1.0 (+https://github.com/MarcYin/CROME)"
_BUFFER_SIZE = 1024 * 1024


class CromeDownloadError(RuntimeError):
    """Raised when a CROME landing page or archive cannot be resolved."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Small HTTP response surface for HTML/JSON page fetches."""

    body: bytes
    status_code: int
    url: str


@dataclass(frozen=True, slots=True)
class CromeLandingPage:
    """Structured search-result match for one CROME dataset landing page."""

    dataset_id: str
    title: str
    url: str
    year: int
    variant: str | None

    @property
    def complete_variant(self) -> bool:
        return self.variant is not None and self.variant.casefold() == "complete"

    @property
    def national_variant(self) -> bool:
        return self.variant is None or self.complete_variant

    @property
    def variant_label(self) -> str:
        if self.variant is None:
            return "national"
        return sanitize_label(self.variant, default="national").lower()


@dataclass(frozen=True, slots=True)
class CromeDownloadResult:
    """Local result surface for one downloaded CROME reference archive."""

    archive_path: Path
    archive_url: str
    dataset_id: str
    extracted_path: Path | None
    landing_page_url: str
    manifest_path: Path
    normalized_path: Path | None
    output_root: Path
    source_layer: str | None
    title: str
    variant: str | None
    year: int

    @property
    def reference_path(self) -> Path | None:
        return self.extracted_path or self.normalized_path


@dataclass(frozen=True, slots=True)
class CromeSubsetResult:
    """Local result surface for one AOI-specific CROME subset."""

    feature_count: int | None
    manifest_path: Path
    output_root: Path
    requested_bounds: tuple[float, float, float, float]
    requested_crs: str
    source_bounds: tuple[float, float, float, float] | None
    source_layer: str | None
    source_path: Path
    subset_label: str
    subset_path: Path
    tile_ids: tuple[str, ...]
    tile_set_id: str
    year: int


HttpGetter = Callable[[str, float], HttpResponse]
FileDownloader = Callable[[str, Path, float], None]


def download_result_to_dict(result: CromeDownloadResult) -> dict[str, Any]:
    """Return a JSON-safe summary payload for one CROME download run."""

    return {
        "archive_path": str(result.archive_path),
        "archive_url": result.archive_url,
        "dataset_id": result.dataset_id,
        "extracted_path": str(result.extracted_path) if result.extracted_path is not None else None,
        "landing_page_url": result.landing_page_url,
        "manifest_path": str(result.manifest_path),
        "normalized_path": str(result.normalized_path) if result.normalized_path is not None else None,
        "output_root": str(result.output_root),
        "reference_path": str(result.reference_path) if result.reference_path is not None else None,
        "source_layer": result.source_layer,
        "title": result.title,
        "variant": result.variant,
        "year": result.year,
    }


def subset_result_to_dict(result: CromeSubsetResult) -> dict[str, Any]:
    """Return a JSON-safe summary payload for one CROME subset run."""

    return {
        "feature_count": result.feature_count,
        "manifest_path": str(result.manifest_path),
        "output_root": str(result.output_root),
        "requested_bounds": list(result.requested_bounds),
        "requested_crs": result.requested_crs,
        "source_bounds": list(result.source_bounds) if result.source_bounds is not None else None,
        "source_layer": result.source_layer,
        "source_path": str(result.source_path),
        "subset_label": result.subset_label,
        "subset_path": str(result.subset_path),
        "tile_ids": list(result.tile_ids),
        "tile_set_id": result.tile_set_id,
        "year": result.year,
    }


def _path_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "mtime_ns": stat.st_mtime_ns,
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
    }


def _subset_output_root(
    reference_path: Path,
    output_root: Path | str,
    year: int,
) -> Path:
    if reference_path.parent.name == "subsets":
        return reference_path.parent
    if reference_path.parent.name in {"extracted", "normalized"}:
        return reference_path.parent.parent / "subsets"
    return crome_download_root(output_root, year, variant_label="subset") / "subsets"


def _is_valid_subset(subset_path: Path) -> bool:
    ensure_proj_data_env()
    if not subset_path.exists() or subset_path.stat().st_size <= 0:
        return False
    try:
        info = pyogrio.read_info(subset_path)
        feature_count = info.get("features")
        return isinstance(feature_count, int) and feature_count > 0
    except Exception:
        return False


def _subset_tile_ids(feature_raster_paths: Sequence[Path | str]) -> tuple[str, ...]:
    tile_ids = sorted(
        {
            sanitize_label(Path(path).stem, default="tile")
            for path in feature_raster_paths
        }
    )
    if not tile_ids:
        raise ValueError("At least one feature raster path is required to derive a subset tile identity.")
    return tuple(tile_ids)


def _subset_tile_set_id(tile_ids: Sequence[str], subset_label: str | None) -> str:
    if tile_ids:
        if len(tile_ids) == 1:
            return f"tile_{tile_ids[0]}"
        digest = hashlib.sha256(json.dumps(list(tile_ids), sort_keys=True).encode("utf-8")).hexdigest()[:12]
        return f"tiles_{len(tile_ids)}_{digest}"
    return sanitize_label(subset_label, default="aoi")


def materialize_crome_subset(
    reference_path: Path | str,
    *,
    output_root: Path | str,
    year: int,
    subset_bounds: tuple[float, float, float, float],
    subset_label: str | None,
    tile_ids: Sequence[str] | None = None,
    requested_crs: str | None = None,
    source_layer: str | None = None,
    force: bool = False,
) -> CromeSubsetResult:
    """Clip one AOI-specific FlatGeobuf subset from a larger CROME vector source."""

    ensure_proj_data_env()
    source_path = Path(reference_path)
    subset_root = _subset_output_root(source_path, output_root, year)
    subset_root.mkdir(parents=True, exist_ok=True)

    if source_layer is None and source_path.suffix.casefold() == ".gpkg":
        source_layer = _canonical_reference_layer(source_path, year)

    info = pyogrio.read_info(source_path, layer=source_layer)
    source_crs = info.get("crs")
    source_bounds = subset_bounds
    if requested_crs is not None and isinstance(source_crs, str) and source_crs:
        source_bounds = transform_bounds(requested_crs, source_crs, *subset_bounds, densify_pts=21)

    normalized_tile_ids = tuple(str(value) for value in (tile_ids or ()))
    tile_set_id = _subset_tile_set_id(normalized_tile_ids, subset_label)
    subset_name = (
        f"{sanitize_label(source_layer or source_path.stem, default='crome')}_{tile_set_id}.fgb"
    )
    subset_path = subset_root / subset_name
    manifest_path = subset_path.with_suffix(".json")
    expected_signature = _path_signature(source_path)
    expected_manifest = {
        "requested_bounds": [float(value) for value in subset_bounds],
        "requested_crs": requested_crs,
        "source_bounds": [float(value) for value in source_bounds],
        "source_layer": source_layer,
        "source_path": str(source_path),
        "source_signature": expected_signature,
        "tile_ids": list(normalized_tile_ids),
        "tile_set_id": tile_set_id,
        "year": year,
    }

    if subset_path.exists() and manifest_path.exists() and not force and _is_valid_subset(subset_path):
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest_payload = None
        if isinstance(manifest_payload, dict):
            comparable = {
                key: manifest_payload.get(key)
                for key in (
                    "requested_bounds",
                    "requested_crs",
                    "source_bounds",
                    "source_layer",
                    "source_path",
                    "source_signature",
                    "tile_ids",
                    "tile_set_id",
                    "year",
                )
            }
            if comparable == expected_manifest:
                subset_info = pyogrio.read_info(subset_path)
                feature_count = subset_info.get("features")
                return CromeSubsetResult(
                    feature_count=int(feature_count) if isinstance(feature_count, int) else None,
                    manifest_path=manifest_path,
                    output_root=subset_root,
                    requested_bounds=tuple(float(value) for value in subset_bounds),
                    requested_crs=requested_crs or (str(source_crs) if isinstance(source_crs, str) else ""),
                    source_bounds=tuple(float(value) for value in source_bounds),
                    source_layer=source_layer,
                    source_path=source_path,
                    subset_label=sanitize_label(subset_label, default="aoi"),
                    subset_path=subset_path,
                    tile_ids=normalized_tile_ids,
                    tile_set_id=tile_set_id,
                    year=year,
                )

    temp_subset_path = subset_path.with_suffix(".tmp.fgb")
    if temp_subset_path.exists():
        temp_subset_path.unlink()
    if subset_path.exists():
        subset_path.unlink()
    if manifest_path.exists():
        manifest_path.unlink()

    minx, miny, maxx, maxy = (float(value) for value in source_bounds)
    ogr2ogr = shutil.which("ogr2ogr")
    if ogr2ogr is not None:
        command = [
            ogr2ogr,
            "-f",
            "FlatGeobuf",
            str(temp_subset_path),
            str(source_path),
        ]
        if source_layer is not None:
            command.append(source_layer)
        command.extend(["-spat", str(minx), str(miny), str(maxx), str(maxy)])
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise CromeDownloadError(
                "ogr2ogr failed while clipping the CROME AOI subset: "
                + completed.stderr.strip()
            )
    else:
        frame = pyogrio.read_dataframe(
            source_path,
            layer=source_layer,
            bbox=(minx, miny, maxx, maxy),
        )
        if frame.empty:
            raise CromeDownloadError("No CROME reference features intersected the requested subset bounds.")
        pyogrio.write_dataframe(frame, temp_subset_path, driver="FlatGeobuf")

    if not _is_valid_subset(temp_subset_path):
        raise CromeDownloadError("No CROME reference features intersected the requested subset bounds.")

    temp_subset_path.replace(subset_path)
    subset_info = pyogrio.read_info(subset_path)
    feature_count = subset_info.get("features")
    manifest_payload = {
        **expected_manifest,
        "feature_count": int(feature_count) if isinstance(feature_count, int) else None,
        "manifest_path": str(manifest_path),
        "output_root": str(subset_root),
        "subset_label": sanitize_label(subset_label, default="aoi"),
        "subset_path": str(subset_path),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return CromeSubsetResult(
        feature_count=int(feature_count) if isinstance(feature_count, int) else None,
        manifest_path=manifest_path,
        output_root=subset_root,
        requested_bounds=tuple(float(value) for value in subset_bounds),
        requested_crs=requested_crs or (str(source_crs) if isinstance(source_crs, str) else ""),
        source_bounds=tuple(float(value) for value in source_bounds),
        source_layer=source_layer,
        source_path=source_path,
        subset_label=sanitize_label(subset_label, default="aoi"),
        subset_path=subset_path,
        tile_ids=normalized_tile_ids,
        tile_set_id=tile_set_id,
        year=year,
    )


def _default_http_get(url: str, timeout_s: float) -> HttpResponse:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=timeout_s) as response:
        body = response.read()
        status_code = getattr(response, "status", 200)
        return HttpResponse(body=body, status_code=status_code, url=response.geturl())


def _default_download_file(url: str, destination: Path, timeout_s: float) -> None:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(request, timeout=timeout_s) as response, partial_path.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=_BUFFER_SIZE)
        partial_path.replace(destination)
    finally:
        if partial_path.exists():
            partial_path.unlink()


def _extract_next_data_payload(html: str) -> dict[str, Any]:
    match = _NEXT_DATA_PATTERN.search(html)
    if match is None:
        raise CromeDownloadError("Could not locate __NEXT_DATA__ payload on the DEFRA page.")
    return json.loads(match.group(1))


def _parse_title(title: str) -> tuple[int, str | None]:
    match = _TITLE_PATTERN.match(title.strip())
    if match is None:
        raise ValueError(f"Not a recognised CROME title: {title!r}")
    year = int(match.group("year"))
    variant = match.group("variant")
    return year, variant.strip() if isinstance(variant, str) else None


def _build_search_url(request: CromeDownloadRequest, *, page: int) -> str:
    return (
        f"{request.search_base_url}?"
        + urlencode({"page": page, "pagesize": request.pagesize, "query": request.query})
    )


def _is_crome_dataset_title(title: str, year: int) -> bool:
    try:
        title_year, _variant = _parse_title(title)
    except ValueError:
        return False
    return title_year == year


def search_crome_landing_pages(
    request: CromeDownloadRequest,
    http_get: HttpGetter | None = None,
) -> tuple[CromeLandingPage, ...]:
    """Search DEFRA DSP for CROME landing pages matching one year."""

    getter = http_get or _default_http_get
    discovered: dict[str, CromeLandingPage] = {}
    page = 1
    total_count: int | None = None

    while True:
        response = getter(_build_search_url(request, page=page), request.timeout_s)
        payload = _extract_next_data_payload(response.body.decode("utf-8"))
        page_props = payload.get("props", {}).get("pageProps", {})
        datasets = page_props.get("datasets") or []
        if total_count is None:
            raw_count = page_props.get("count")
            total_count = raw_count if isinstance(raw_count, int) else len(datasets)

        for dataset in datasets:
            if not isinstance(dataset, dict):
                continue
            dataset_id = dataset.get("id")
            title = dataset.get("title")
            if not isinstance(dataset_id, str) or not isinstance(title, str):
                continue
            if not _is_crome_dataset_title(title, request.year):
                continue
            year, variant = _parse_title(title)
            discovered.setdefault(
                dataset_id,
                CromeLandingPage(
                    dataset_id=dataset_id,
                    title=title,
                    url=f"https://environment.data.gov.uk/dataset/{dataset_id}",
                    year=year,
                    variant=variant,
                ),
            )

        if discovered or not datasets:
            break
        if total_count is not None and page * request.pagesize >= total_count:
            break
        page += 1

    return tuple(discovered.values())


def resolve_crome_landing_page(
    request: CromeDownloadRequest,
    http_get: HttpGetter | None = None,
) -> CromeLandingPage:
    """Resolve the national CROME landing page for one year."""

    candidates = search_crome_landing_pages(request, http_get=http_get)
    plain_pages = tuple(page for page in candidates if page.variant is None)
    complete_pages = tuple(page for page in candidates if page.complete_variant)
    regional_pages = tuple(page for page in candidates if not page.national_variant)

    if request.prefer_complete and complete_pages:
        return complete_pages[0]
    if plain_pages:
        return plain_pages[0]
    if complete_pages:
        return complete_pages[0]

    if regional_pages:
        variants = ", ".join(sorted(page.title for page in regional_pages))
        raise CromeDownloadError(
            f"Found only regional CROME datasets for {request.year}: {variants}"
        )
    raise CromeDownloadError(
        f"Could not find a CROME landing page for {request.year} using query {request.query!r}."
    )


def _match_gpkg_file(files: tuple[dict[str, str], ...], landing_page: CromeLandingPage) -> dict[str, str]:
    gpkg_files = tuple(
        entry
        for entry in files
        if entry.get("name", "").casefold().endswith(".gpkg.zip") and entry.get("url")
    )
    if not gpkg_files:
        raise CromeDownloadError(
            f"No .gpkg.zip download was listed on {landing_page.url}."
        )

    year = landing_page.year
    preferred_suffixes = []
    if landing_page.complete_variant:
        preferred_suffixes.append(f"_{year}_complete.gpkg.zip")
    if landing_page.variant is None:
        preferred_suffixes.append(f"_{year}.gpkg.zip")
    preferred_suffixes.append(f"_{year}.gpkg.zip")
    preferred_suffixes.append(f"_{year}_complete.gpkg.zip")

    for suffix in preferred_suffixes:
        for entry in gpkg_files:
            name = entry["name"].casefold()
            if name.endswith(suffix):
                return entry

    for entry in gpkg_files:
        name = entry["name"].casefold()
        if f"crome_{year}" in name:
            return entry

    raise CromeDownloadError(
        f"No CROME GeoPackage archive matched year {year} on {landing_page.url}."
    )


def _build_public_file_url(file_dataset_id: str, file_name: str) -> str:
    encoded_name = quote(file_name)
    return (
        "https://environment.data.gov.uk/file-management-open/data-sets/"
        f"{file_dataset_id}/files/{encoded_name}"
    )


def extract_crome_gpkg_zip_url(
    landing_page: CromeLandingPage,
    http_get: HttpGetter | None = None,
    *,
    timeout_s: float = 30.0,
) -> str:
    """Parse the landing page and return the preferred .gpkg.zip download URL."""

    getter = http_get or _default_http_get
    response = getter(landing_page.url, timeout_s)
    payload = _extract_next_data_payload(response.body.decode("utf-8"))
    page_props = payload.get("props", {}).get("pageProps", {})
    dataset = page_props.get("dataset") or {}
    file_dataset = dataset.get("dataSet") if isinstance(dataset, dict) else {}
    file_dataset_id = file_dataset.get("id") if isinstance(file_dataset, dict) else None
    raw_files = page_props.get("files") or []
    files: list[dict[str, str]] = []
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        url = item.get("fileURI")
        if isinstance(name, str) and isinstance(url, str):
            files.append({"name": name, "url": url})

    matched = _match_gpkg_file(tuple(files), landing_page)
    if isinstance(file_dataset_id, str) and file_dataset_id:
        return _build_public_file_url(file_dataset_id, matched["name"])
    return matched["url"]


def _extract_gpkg_from_archive(
    archive_path: Path,
    output_root: Path,
    *,
    force: bool,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        gpkg_members = tuple(
            member
            for member in archive.namelist()
            if member.casefold().endswith(".gpkg") and not member.endswith("/")
        )
        if not gpkg_members:
            raise CromeDownloadError(f"No .gpkg member found inside {archive_path}.")
        member_name = gpkg_members[0]
        extracted_path = output_root / Path(member_name).name
        if extracted_path.exists() and not force:
            return extracted_path
        with archive.open(member_name) as source, extracted_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=_BUFFER_SIZE)
        return extracted_path


def _canonical_reference_layer(
    extracted_path: Path,
    year: int,
) -> str:
    ensure_proj_data_env()
    layers = pyogrio.list_layers(extracted_path)
    if len(layers) == 0:
        raise CromeDownloadError(f"No vector layers were found in {extracted_path}.")

    exact_name = f"Crop_Map_of_England_{year}"
    for item in layers:
        layer_name = str(item[0])
        if layer_name.casefold() == exact_name.casefold():
            return layer_name

    if len(layers) == 1:
        return str(layers[0][0])

    ranked: list[tuple[int, int, str]] = []
    for item in layers:
        layer_name = str(item[0])
        info = pyogrio.read_info(extracted_path, layer=layer_name)
        feature_count = info.get("features")
        ranked.append(
            (
                int(feature_count) if isinstance(feature_count, int) else -1,
                -len(layer_name),
                layer_name,
            )
        )
    ranked.sort(reverse=True)
    return ranked[0][2]


def _is_valid_normalized_reference(normalized_path: Path) -> bool:
    ensure_proj_data_env()
    if not normalized_path.exists() or normalized_path.stat().st_size <= 0:
        return False
    try:
        layers = pyogrio.list_layers(normalized_path)
        if len(layers) == 0:
            return False
        info = pyogrio.read_info(normalized_path)
        feature_count = info.get("features")
        if not isinstance(feature_count, int) or feature_count <= 0:
            return False
        sample = pyogrio.read_dataframe(normalized_path, max_features=1)
        if sample.empty or sample.geometry is None or sample.geometry.iloc[0] is None:
            return False
        geometry = sample.geometry.iloc[0]
        if geometry.is_empty:
            return False
        probe = pyogrio.read_dataframe(
            normalized_path,
            read_geometry=False,
            bbox=tuple(float(value) for value in geometry.bounds),
            max_features=1,
        )
        return not probe.empty
    except Exception:
        return False


def _normalize_gpkg_to_flatgeobuf(
    extracted_path: Path,
    output_root: Path,
    *,
    year: int,
    force: bool,
) -> tuple[Path, str]:
    ensure_proj_data_env()
    source_layer = _canonical_reference_layer(extracted_path, year)
    output_root.mkdir(parents=True, exist_ok=True)
    normalized_path = output_root / f"{sanitize_label(source_layer, default='crome')}.fgb"
    if normalized_path.exists() and force:
        normalized_path.unlink()
    if normalized_path.exists() and not force:
        if _is_valid_normalized_reference(normalized_path):
            return normalized_path, source_layer
        normalized_path.unlink()

    ogr2ogr = shutil.which("ogr2ogr")
    if ogr2ogr is not None:
        completed = subprocess.run(
            [
                ogr2ogr,
                "-f",
                "FlatGeobuf",
                str(normalized_path),
                str(extracted_path),
                source_layer,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise CromeDownloadError(
                "ogr2ogr failed while normalizing the CROME GeoPackage to FlatGeobuf: "
                + completed.stderr.strip()
            )
    else:
        frame = pyogrio.read_dataframe(extracted_path, layer=source_layer)
        pyogrio.write_dataframe(frame, normalized_path, driver="FlatGeobuf")

    if not _is_valid_normalized_reference(normalized_path):
        raise CromeDownloadError(
            f"Normalized CROME FlatGeobuf is invalid or unreadable after creation: {normalized_path}"
        )

    metadata_path = output_root / "manifest.json"
    metadata_path.write_text(
        json.dumps(
            {
                "normalized_path": str(normalized_path),
                "source_gpkg": str(extracted_path),
                "source_layer": source_layer,
                "year": year,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return normalized_path, source_layer


def _subset_root_for_reference(reference_path: Path) -> Path | None:
    parent_name = reference_path.parent.name.casefold()
    if parent_name in {"extracted", "normalized"}:
        return reference_path.parent.parent / "subsets"
    if parent_name == "subsets":
        return reference_path.parent
    return None


def _preferred_subset_source(reference_path: Path, *, year: int) -> tuple[Path, str | None] | None:
    parent_name = reference_path.parent.name.casefold()
    if parent_name == "subsets":
        return reference_path, None
    if parent_name == "extracted" and reference_path.suffix.casefold() == ".gpkg":
        return reference_path, _canonical_reference_layer(reference_path, year)
    if parent_name == "normalized":
        extracted_dir = reference_path.parent.parent / "extracted"
        extracted_candidates = sorted(extracted_dir.glob("*.gpkg"))
        if extracted_candidates:
            extracted_path = extracted_candidates[0]
            return extracted_path, _canonical_reference_layer(extracted_path, year)
    return None


def materialize_crome_reference_subset(
    reference_path: Path | str,
    *,
    feature_raster_paths: Sequence[Path | str],
    subset_label: str | None,
    year: int,
    force: bool = False,
) -> Path:
    """Materialize or reuse one AOI-specific CROME subset for discovered AlphaEarth rasters.

    Automatic subsetting is conservative:
    - managed national CROME sources under `extracted/` or `normalized/` are clipped
    - existing AOI subset paths under `subsets/` are reused as-is
    - arbitrary external vectors are left untouched
    """

    from crome.labeling import reference_source_bbox_for_feature_rasters

    resolved_reference_path = Path(reference_path)
    subset_root = _subset_root_for_reference(resolved_reference_path)
    if subset_root is None:
        return resolved_reference_path
    if resolved_reference_path.parent.name.casefold() == "subsets" and not force:
        return resolved_reference_path

    preferred_source = _preferred_subset_source(resolved_reference_path, year=year)
    if preferred_source is None:
        return resolved_reference_path
    source_path, source_layer = preferred_source
    tile_ids = _subset_tile_ids(feature_raster_paths)

    bbox = reference_source_bbox_for_feature_rasters(
        source_path,
        [Path(path) for path in feature_raster_paths],
    )
    if bbox is None:
        return resolved_reference_path

    subset = materialize_crome_subset(
        source_path,
        output_root=subset_root,
        year=year,
        subset_bounds=tuple(float(value) for value in bbox),
        subset_label=subset_label,
        tile_ids=tile_ids,
        source_layer=source_layer,
        force=force,
    )
    return subset.subset_path


def download_crome_reference(
    request: CromeDownloadRequest,
    http_get: HttpGetter | None = None,
    download_file: FileDownloader | None = None,
) -> CromeDownloadResult:
    """Resolve and download the preferred CROME GeoPackage archive for one year."""

    getter = http_get or _default_http_get
    downloader = download_file or _default_download_file
    landing_page = resolve_crome_landing_page(request, http_get=getter)
    archive_url = extract_crome_gpkg_zip_url(
        landing_page,
        http_get=getter,
        timeout_s=request.timeout_s,
    )
    archive_filename = Path(urlparse(archive_url).path).name
    output_root = crome_download_root(
        request.output_root,
        request.year,
        variant_label=landing_page.variant_label,
    )
    archive_path = crome_archive_path(
        request.output_root,
        request.year,
        archive_filename,
        variant_label=landing_page.variant_label,
    )
    partial_archive_path = archive_path.with_suffix(archive_path.suffix + ".part")
    if request.force and archive_path.exists():
        archive_path.unlink()
    if request.force and partial_archive_path.exists():
        partial_archive_path.unlink()
    if not archive_path.exists() and partial_archive_path.exists():
        partial_archive_path.unlink()
    if not archive_path.exists():
        downloader(archive_url, archive_path, request.timeout_s)

    extracted_path = None
    normalized_path = None
    source_layer = None
    if request.extract:
        extracted_path = _extract_gpkg_from_archive(
            archive_path,
            crome_extract_root(
                request.output_root,
                request.year,
                variant_label=landing_page.variant_label,
            ),
            force=request.force,
        )
        normalized_path, source_layer = _normalize_gpkg_to_flatgeobuf(
            extracted_path,
            crome_normalized_root(
                request.output_root,
                request.year,
                variant_label=landing_page.variant_label,
            ),
            year=request.year,
            force=request.force,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    result = CromeDownloadResult(
        archive_path=archive_path,
        archive_url=archive_url,
        dataset_id=landing_page.dataset_id,
        extracted_path=extracted_path,
        landing_page_url=landing_page.url,
        manifest_path=manifest_path,
        normalized_path=normalized_path,
        output_root=output_root,
        source_layer=source_layer,
        title=landing_page.title,
        variant=landing_page.variant,
        year=landing_page.year,
    )
    manifest_path.write_text(
        json.dumps(download_result_to_dict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the national CROME reference for one year and normalize it to FlatGeobuf."
    )
    parser.add_argument("--year", required=True, type=int, help="Target CROME year.")
    parser.add_argument(
        "--output-root",
        default=default_output_root(),
        help=(
            "Base directory for downloaded CROME archives, extracted GeoPackages, and normalized FlatGeobuf references. "
            f"Defaults to ${OUTPUT_ROOT_ENV_VAR} when set, otherwise data/alphaearth."
        ),
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional custom DEFRA search query. Defaults to the exact CROME year title.",
    )
    parser.add_argument(
        "--prefer-plain",
        action="store_true",
        help="Prefer plain-year titles over '- Complete' titles when both exist.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep only the downloaded .gpkg.zip archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even when local copies already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the landing page and archive URL without downloading.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    request = CromeDownloadRequest(
        year=args.year,
        output_root=args.output_root,
        prefer_complete=not args.prefer_plain,
        extract=not args.no_extract,
        force=args.force,
        query=args.query,
    )

    if args.dry_run:
        landing_page = resolve_crome_landing_page(request)
        archive_url = extract_crome_gpkg_zip_url(landing_page, timeout_s=request.timeout_s)
        archive_filename = Path(urlparse(archive_url).path).name
        variant_label = landing_page.variant_label
        payload = request.to_dict()
        payload.update(
            {
                "archive_path": str(
                    crome_archive_path(
                        request.output_root,
                        request.year,
                        archive_filename,
                        variant_label=variant_label,
                    )
                ),
                "archive_url": archive_url,
                "dataset_id": landing_page.dataset_id,
                "extracted_root": str(
                    crome_extract_root(
                        request.output_root,
                        request.year,
                        variant_label=variant_label,
                    )
                ),
                "normalized_root": str(
                    crome_normalized_root(
                        request.output_root,
                        request.year,
                        variant_label=variant_label,
                    )
                ),
                "landing_page_url": landing_page.url,
                "title": landing_page.title,
                "variant": landing_page.variant,
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    result = download_crome_reference(request)
    print(json.dumps(download_result_to_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
