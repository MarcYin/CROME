import json
from pathlib import Path
import zipfile

import geopandas as gpd
import numpy as np
import pyogrio
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from crome.acquisition.crome import (
    CromeDownloadError,
    HttpResponse,
    _build_search_url,
    download_crome_reference,
    export_crome_footprint,
    extract_crome_gpkg_zip_url,
    materialize_crome_reference_subset,
    reference_footprint,
    resolve_crome_landing_page,
)
from crome.config import CromeDownloadRequest


def _write_reference_gpkg(path: Path) -> None:
    national = gpd.GeoDataFrame(
        {
            "lucode": ["wheat", "barley"],
            "geometry": [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]),
            ],
        },
        crs="EPSG:27700",
    )
    county = gpd.GeoDataFrame(
        {
            "lucode": ["wheat"],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        crs="EPSG:27700",
    )
    national.to_file(path, layer="Crop_Map_of_England_2017", driver="GPKG")
    county.to_file(path, layer="Crop_Map_of_England_2017_Cambridgeshire", driver="GPKG")


def _write_feature_raster(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=4,
        count=64,
        dtype="float32",
        crs="EPSG:27700",
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
    ) as dst:
        dst.write(np.ones((64, 2, 4), dtype="float32"))


def _next_data_html(page_props: dict) -> str:
    payload = {
        "buildId": "test-build",
        "gssp": True,
        "isFallback": False,
        "page": "/test",
        "props": {"pageProps": page_props},
        "query": {},
        "runtimeConfig": {},
        "scriptLoader": [],
    }
    return (
        "<html><body><script id=\"__NEXT_DATA__\" type=\"application/json\">"
        + json.dumps(payload)
        + "</script></body></html>"
    )


def _search_html(*datasets: dict[str, object], count: int | None = None) -> str:
    return _next_data_html(
        {
            "allowEmptySearch": False,
            "count": len(datasets) if count is None else count,
            "datasets": list(datasets),
            "error": None,
            "query": {},
        }
    )


def _dataset_html(title: str, files: list[dict[str, str]], *, file_dataset_id: str | None = None) -> str:
    dataset = {"title": title}
    if file_dataset_id is not None:
        dataset["dataSet"] = {"id": file_dataset_id}
    return _next_data_html(
        {
            "dataset": dataset,
            "error": None,
            "files": files,
            "query": {},
        }
    )


def test_resolve_crome_landing_page_prefers_plain_year_page_for_modern_releases() -> None:
    request = CromeDownloadRequest(year=2024, output_root="outputs")
    search_url = _build_search_url(request, page=1)
    http_map = {
        search_url: _search_html(
            {"id": "2024", "title": "Crop Map of England (CROME) 2024"},
            {"id": "2023", "title": "Crop Map of England (CROME) 2023"},
        )
    }

    landing_page = resolve_crome_landing_page(request, http_get=lambda url, _timeout: HttpResponse(body=http_map[url].encode("utf-8"), status_code=200, url=url))

    assert landing_page.dataset_id == "2024"
    assert landing_page.title == "Crop Map of England (CROME) 2024"
    assert landing_page.complete_variant is False


def test_resolve_crome_landing_page_prefers_complete_variant_when_present() -> None:
    request = CromeDownloadRequest(year=2017, output_root="outputs")
    search_url = _build_search_url(request, page=1)
    http_map = {
        search_url: _search_html(
            {"id": "midlands", "title": "Crop Map of England (CROME) 2017 - Midlands"},
            {"id": "complete", "title": "Crop Map of England (CROME) 2017 - Complete"},
            {"id": "south-east", "title": "Crop Map of England (CROME) 2017 - South East"},
        )
    }

    landing_page = resolve_crome_landing_page(request, http_get=lambda url, _timeout: HttpResponse(body=http_map[url].encode("utf-8"), status_code=200, url=url))

    assert landing_page.dataset_id == "complete"
    assert landing_page.complete_variant is True


def test_resolve_crome_landing_page_rejects_regional_only_results() -> None:
    request = CromeDownloadRequest(year=2017, output_root="outputs")
    search_url = _build_search_url(request, page=1)
    http_map = {
        search_url: _search_html(
            {"id": "midlands", "title": "Crop Map of England (CROME) 2017 - Midlands"},
            {"id": "north", "title": "Crop Map of England (CROME) 2017 - North"},
        )
    }

    with pytest.raises(CromeDownloadError, match="regional CROME datasets"):
        resolve_crome_landing_page(
            request,
            http_get=lambda url, _timeout: HttpResponse(
                body=http_map[url].encode("utf-8"),
                status_code=200,
                url=url,
            ),
        )


def test_extract_crome_gpkg_zip_url_selects_gpkg_zip_not_gdb_or_geojson() -> None:
    request = CromeDownloadRequest(year=2024, output_root="outputs")
    search_url = _build_search_url(request, page=1)
    dataset_url = "https://environment.data.gov.uk/dataset/2024"
    file_dataset_id = "public-2024"
    http_map = {
        search_url: _search_html({"id": "2024", "title": "Crop Map of England (CROME) 2024"}),
        dataset_url: _dataset_html(
            "Crop Map of England (CROME) 2024",
            [
                {"name": "Crop_Map_of_England_CROME_2024.gdb.zip", "fileURI": "https://example.test/2024.gdb.zip"},
                {"name": "Crop_Map_of_England_CROME_2024.geojson.zip", "fileURI": "https://example.test/2024.geojson.zip"},
                {"name": "Crop_Map_of_England_CROME_2024.gpkg.zip", "fileURI": "https://example.test/2024.gpkg.zip"},
            ],
            file_dataset_id=file_dataset_id,
        ),
    }

    def http_get(url: str, _timeout: float) -> HttpResponse:
        return HttpResponse(body=http_map[url].encode("utf-8"), status_code=200, url=url)

    landing_page = resolve_crome_landing_page(request, http_get=http_get)
    download_url = extract_crome_gpkg_zip_url(landing_page, http_get=http_get)

    assert download_url == (
        "https://environment.data.gov.uk/file-management-open/data-sets/"
        "public-2024/files/Crop_Map_of_England_CROME_2024.gpkg.zip"
    )


def test_download_crome_reference_writes_expected_archive_and_gpkg(tmp_path: Path) -> None:
    request = CromeDownloadRequest(year=2017, output_root=tmp_path)
    search_url = _build_search_url(request, page=1)
    dataset_url = "https://environment.data.gov.uk/dataset/complete"
    http_map = {
        search_url: _search_html(
            {"id": "complete", "title": "Crop Map of England (CROME) 2017 - Complete"}
        ),
        dataset_url: _dataset_html(
            "Crop Map of England (CROME) 2017 - Complete",
            [
                {
                    "name": "Crop_Map_of_England_CROME_2017_Complete.gpkg.zip",
                    "fileURI": "https://example.test/Crop_Map_of_England_CROME_2017_Complete.gpkg.zip",
                }
            ],
            file_dataset_id="public-2017",
        ),
    }

    def http_get(url: str, _timeout: float) -> HttpResponse:
        return HttpResponse(body=http_map[url].encode("utf-8"), status_code=200, url=url)

    def download_file(url: str, destination: Path, _timeout: float) -> None:
        assert url == (
            "https://environment.data.gov.uk/file-management-open/data-sets/"
            "public-2017/files/Crop_Map_of_England_CROME_2017_Complete.gpkg.zip"
        )
        source_gpkg = tmp_path / "source.gpkg"
        _write_reference_gpkg(source_gpkg)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.write(
                source_gpkg,
                arcname="Crop_Map_of_England_CROME_2017_Complete.gpkg",
            )

    result = download_crome_reference(request, http_get=http_get, download_file=download_file)

    assert result.archive_path.exists()
    assert result.extracted_path == (
        tmp_path
        / "raw"
        / "crome"
        / "CROME_2017_complete"
        / "extracted"
        / "Crop_Map_of_England_CROME_2017_Complete.gpkg"
    )
    assert result.extracted_path.exists()
    assert result.normalized_path == (
        tmp_path
        / "raw"
        / "crome"
        / "CROME_2017_complete"
        / "normalized"
        / "Crop_Map_of_England_2017.fgb"
    )
    assert result.reference_path == result.extracted_path
    assert result.source_layer == "Crop_Map_of_England_2017"
    assert result.normalized_path.exists()
    info = pyogrio.read_info(result.normalized_path)
    assert info["features"] == 2
    assert json.loads(result.manifest_path.read_text(encoding="utf-8"))["dataset_id"] == "complete"


def test_reference_footprint_prefers_exact_national_layer(tmp_path: Path) -> None:
    reference_path = tmp_path / "Crop_Map_of_England_CROME_2024.gpkg"
    county = gpd.GeoDataFrame(
        {
            "lucode": ["wheat"],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        crs="EPSG:27700",
    )
    national = gpd.GeoDataFrame(
        {
            "lucode": ["wheat", "barley"],
            "geometry": [
                Polygon([(10, 10), (20, 10), (20, 20), (10, 20)]),
                Polygon([(20, 10), (40, 10), (40, 20), (20, 20)]),
            ],
        },
        crs="EPSG:27700",
    )
    county.to_file(reference_path, layer="Crop_Map_of_England_2024_Durham", driver="GPKG")
    national.to_file(reference_path, layer="Crop_Map_of_England_2024", driver="GPKG")

    result = reference_footprint(reference_path, year=2024)

    assert result.source_layer == "Crop_Map_of_England_2024"
    assert result.bounds == (10.0, 10.0, 40.0, 20.0)
    assert len(result.bounds_lonlat) == 4


def test_export_crome_footprint_prefers_exact_national_layer(tmp_path: Path) -> None:
    reference_path = tmp_path / "Crop_Map_of_England_CROME_2024.gpkg"
    county = gpd.GeoDataFrame(
        {
            "lucode": ["wheat", "barley", "oats"],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
        },
        crs="EPSG:27700",
    )
    national = gpd.GeoDataFrame(
        {
            "lucode": ["wheat", "barley"],
            "geometry": [
                Polygon([(10, 10), (20, 10), (20, 20), (10, 20)]),
                Polygon([(20, 10), (40, 10), (40, 20), (20, 20)]),
            ],
        },
        crs="EPSG:27700",
    )
    county.to_file(reference_path, layer="Crop_Map_of_England_2024_Durham", driver="GPKG")
    national.to_file(reference_path, layer="Crop_Map_of_England_2024", driver="GPKG")

    result = export_crome_footprint(
        reference_path,
        output_root=tmp_path,
        year=2024,
        footprint_label="production",
    )

    assert result.source_layer == "Crop_Map_of_England_2024"
    assert result.feature_count == 2
    assert result.footprint_bounds == (10.0, 10.0, 40.0, 20.0)
    assert result.footprint_path.exists()
    footprint = gpd.read_file(result.footprint_path)
    assert len(footprint) == 1
    assert tuple(round(value, 6) for value in footprint.total_bounds) == (10.0, 10.0, 40.0, 20.0)
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["source_layer"] == "Crop_Map_of_England_2024"
    assert payload["feature_count"] == 2


def test_download_crome_reference_rebuilds_invalid_normalized_fgb(tmp_path: Path) -> None:
    request = CromeDownloadRequest(year=2017, output_root=tmp_path)
    search_url = _build_search_url(request, page=1)
    dataset_url = "https://environment.data.gov.uk/dataset/complete"
    http_map = {
        search_url: _search_html(
            {"id": "complete", "title": "Crop Map of England (CROME) 2017 - Complete"}
        ),
        dataset_url: _dataset_html(
            "Crop Map of England (CROME) 2017 - Complete",
            [
                {
                    "name": "Crop_Map_of_England_CROME_2017_Complete.gpkg.zip",
                    "fileURI": "https://example.test/Crop_Map_of_England_CROME_2017_Complete.gpkg.zip",
                }
            ],
            file_dataset_id="public-2017",
        ),
    }

    def http_get(url: str, _timeout: float) -> HttpResponse:
        return HttpResponse(body=http_map[url].encode("utf-8"), status_code=200, url=url)

    def download_file(url: str, destination: Path, _timeout: float) -> None:
        assert url.endswith("Crop_Map_of_England_CROME_2017_Complete.gpkg.zip")
        source_gpkg = tmp_path / "source.gpkg"
        _write_reference_gpkg(source_gpkg)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.write(source_gpkg, arcname="Crop_Map_of_England_CROME_2017_Complete.gpkg")

    normalized_path = (
        tmp_path
        / "raw"
        / "crome"
        / "CROME_2017_complete"
        / "normalized"
        / "Crop_Map_of_England_2017.fgb"
    )
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.write_bytes(b"")

    result = download_crome_reference(request, http_get=http_get, download_file=download_file)

    assert result.normalized_path == normalized_path
    assert result.normalized_path.exists()
    assert result.normalized_path.stat().st_size > 0
    info = pyogrio.read_info(result.normalized_path)
    assert info["features"] == 2


def test_download_crome_reference_discards_stale_partial_archive(tmp_path: Path) -> None:
    request = CromeDownloadRequest(year=2017, output_root=tmp_path)
    search_url = _build_search_url(request, page=1)
    dataset_url = "https://environment.data.gov.uk/dataset/complete"
    http_map = {
        search_url: _search_html(
            {"id": "complete", "title": "Crop Map of England (CROME) 2017 - Complete"}
        ),
        dataset_url: _dataset_html(
            "Crop Map of England (CROME) 2017 - Complete",
            [
                {
                    "name": "Crop_Map_of_England_CROME_2017_Complete.gpkg.zip",
                    "fileURI": "https://example.test/Crop_Map_of_England_CROME_2017_Complete.gpkg.zip",
                }
            ],
            file_dataset_id="public-2017",
        ),
    }

    def http_get(url: str, _timeout: float) -> HttpResponse:
        return HttpResponse(body=http_map[url].encode("utf-8"), status_code=200, url=url)

    archive_path = (
        tmp_path
        / "raw"
        / "crome"
        / "CROME_2017_complete"
        / "archive"
        / "Crop_Map_of_England_CROME_2017_Complete.gpkg.zip"
    )
    partial_path = archive_path.with_suffix(".zip.part")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_bytes(b"stale")

    def download_file(url: str, destination: Path, _timeout: float) -> None:
        assert url.endswith("Crop_Map_of_England_CROME_2017_Complete.gpkg.zip")
        assert not partial_path.exists()
        source_gpkg = tmp_path / "source.gpkg"
        _write_reference_gpkg(source_gpkg)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.write(source_gpkg, arcname="Crop_Map_of_England_CROME_2017_Complete.gpkg")

    result = download_crome_reference(request, http_get=http_get, download_file=download_file)

    assert result.archive_path.exists()
    assert not partial_path.exists()


def test_download_crome_reference_removes_stale_partial_archive(tmp_path: Path) -> None:
    request = CromeDownloadRequest(year=2017, output_root=tmp_path)
    search_url = _build_search_url(request, page=1)
    dataset_url = "https://environment.data.gov.uk/dataset/complete"
    http_map = {
        search_url: _search_html(
            {"id": "complete", "title": "Crop Map of England (CROME) 2017 - Complete"}
        ),
        dataset_url: _dataset_html(
            "Crop Map of England (CROME) 2017 - Complete",
            [
                {
                    "name": "Crop_Map_of_England_CROME_2017_Complete.gpkg.zip",
                    "fileURI": "https://example.test/Crop_Map_of_England_CROME_2017_Complete.gpkg.zip",
                }
            ],
            file_dataset_id="public-2017",
        ),
    }

    def http_get(url: str, _timeout: float) -> HttpResponse:
        return HttpResponse(body=http_map[url].encode("utf-8"), status_code=200, url=url)

    archive_path = (
        tmp_path
        / "raw"
        / "crome"
        / "CROME_2017_complete"
        / "archive"
        / "Crop_Map_of_England_CROME_2017_Complete.gpkg.zip"
    )
    partial_path = archive_path.with_suffix(".zip.part")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_bytes(b"stale")

    def download_file(url: str, destination: Path, _timeout: float) -> None:
        assert url == (
            "https://environment.data.gov.uk/file-management-open/data-sets/"
            "public-2017/files/Crop_Map_of_England_CROME_2017_Complete.gpkg.zip"
        )
        assert destination == archive_path
        assert not partial_path.exists()
        source_gpkg = tmp_path / "source.gpkg"
        _write_reference_gpkg(source_gpkg)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.write(
                source_gpkg,
                arcname="Crop_Map_of_England_CROME_2017_Complete.gpkg",
            )

    result = download_crome_reference(request, http_get=http_get, download_file=download_file)

    assert result.archive_path.exists()
    assert not partial_path.exists()


def test_materialize_crome_reference_subset_writes_and_reuses_subset(tmp_path: Path) -> None:
    source_gpkg = (
        tmp_path
        / "raw"
        / "crome"
        / "CROME_2017_complete"
        / "extracted"
        / "source.gpkg"
    )
    feature_raster = tmp_path / "alphaearth.tif"
    source_gpkg.parent.mkdir(parents=True, exist_ok=True)
    _write_reference_gpkg(source_gpkg)
    _write_feature_raster(feature_raster)

    subset_path = materialize_crome_reference_subset(
        source_gpkg,
        feature_raster_paths=[feature_raster],
        subset_label="cambridge-fringe",
        year=2017,
    )

    assert subset_path == (
        tmp_path
        / "raw"
        / "crome"
        / "CROME_2017_complete"
        / "subsets"
        / "Crop_Map_of_England_2017_tile_alphaearth.fgb"
    )
    assert subset_path.exists()
    assert pyogrio.read_info(subset_path)["features"] == 2

    manifest_path = subset_path.with_suffix(".json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["source_path"] == str(source_gpkg)
    assert payload["source_layer"] == "Crop_Map_of_England_2017"
    assert payload["source_signature"]["path"] == str(source_gpkg.resolve())
    assert payload["subset_label"] == "cambridge-fringe"
    assert payload["tile_ids"] == ["alphaearth"]
    assert payload["tile_set_id"] == "tile_alphaearth"

    reused_path = materialize_crome_reference_subset(
        source_gpkg,
        feature_raster_paths=[feature_raster],
        subset_label="cambridge-fringe",
        year=2017,
    )
    assert reused_path == subset_path


def test_materialize_crome_reference_subset_leaves_external_vector_unchanged(tmp_path: Path) -> None:
    source_gpkg = tmp_path / "source.gpkg"
    feature_raster = tmp_path / "alphaearth.tif"
    _write_reference_gpkg(source_gpkg)
    _write_feature_raster(feature_raster)

    subset_path = materialize_crome_reference_subset(
        source_gpkg,
        feature_raster_paths=[feature_raster],
        subset_label="cambridge-fringe",
        year=2017,
    )

    assert subset_path == source_gpkg


def test_materialize_crome_reference_subset_uses_tile_set_id_for_multi_tile_batches(
    tmp_path: Path,
) -> None:
    source_gpkg = (
        tmp_path
        / "raw"
        / "crome"
        / "CROME_2017_complete"
        / "extracted"
        / "source.gpkg"
    )
    first_feature_raster = tmp_path / "tile_a.tif"
    second_feature_raster = tmp_path / "tile_b.tif"
    source_gpkg.parent.mkdir(parents=True, exist_ok=True)
    _write_reference_gpkg(source_gpkg)
    _write_feature_raster(first_feature_raster)
    _write_feature_raster(second_feature_raster)

    subset_path = materialize_crome_reference_subset(
        source_gpkg,
        feature_raster_paths=[second_feature_raster, first_feature_raster],
        subset_label="cambridge-fringe",
        year=2017,
    )

    payload = json.loads(subset_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert subset_path.name.startswith("Crop_Map_of_England_2017_tiles_2_")
    assert payload["tile_ids"] == ["tile_a", "tile_b"]
    assert payload["tile_set_id"].startswith("tiles_2_")
