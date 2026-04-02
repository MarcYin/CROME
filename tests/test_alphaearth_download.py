import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from crome.acquisition.alphaearth import (
    DownloadFailedError,
    build_download_config,
    download_alphaearth_images,
)
from crome.bands import ALPHAEARTH_BANDS
from crome.config import AlphaEarthDownloadRequest


class FakeAOI:
    @staticmethod
    def from_bbox(bbox: tuple[float, float, float, float]) -> tuple[str, tuple[float, float, float, float]]:
        return ("bbox", bbox)

    @staticmethod
    def from_geojson_path(path: Path) -> tuple[str, Path]:
        return ("geojson", path)


@dataclass
class FakeDownloadConfig:
    collection_id: str
    start_date: str
    end_date: str
    aoi: object
    bands: tuple[str, ...]
    output_root: Path


def _write_feature_raster(path: Path) -> None:
    data = np.zeros((len(ALPHAEARTH_BANDS), 4, 4), dtype="float32")
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": len(ALPHAEARTH_BANDS),
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 4, 1, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = ALPHAEARTH_BANDS


def _write_manifest(
    path: Path,
    rasters: list[tuple[str, Path, str]],
    *,
    start_date: str,
) -> None:
    output_root = path.parent.parent if path.parent.name == "manifests" else path.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"output_root": str(output_root), "start_date": start_date},
        "download": {
            "output_root": str(output_root),
            "results": [
                {
                    "image_id": image_id,
                    "status": "downloaded",
                    "tiff_path": str(raster.relative_to(output_root)),
                }
                for image_id, raster, _ in rasters
            ],
        },
        "schema_version": "0.1.1",
        "search": {
            "images": [
                {
                    "image_id": image_id,
                    "acquisition_time_utc": acquisition_time_utc,
                    "relative_tiff_path": str(raster.relative_to(output_root)),
                }
                for image_id, raster, acquisition_time_utc in rasters
            ]
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_build_download_config_uses_canonical_alphaearth_request() -> None:
    fake_module = SimpleNamespace(AOI=FakeAOI, DownloadConfig=FakeDownloadConfig)
    request = AlphaEarthDownloadRequest(
        year=2024,
        output_root="data/alphaearth",
        aoi_label="east-anglia",
        bbox=(-1.0, 51.0, 0.0, 52.0),
    )

    config = build_download_config(request, fake_module)
    assert config.collection_id == "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    assert config.start_date == "2024-01-01"
    assert config.end_date == "2025-01-01"
    assert config.aoi == ("bbox", (-1.0, 51.0, 0.0, 52.0))
    assert config.output_root == Path(
        "data/alphaearth/raw/alphaearth/AEF_east-anglia_annual_embedding_2024"
    )


def test_download_alphaearth_images_wraps_summary() -> None:
    fake_module = SimpleNamespace(
        AOI=FakeAOI,
        DownloadConfig=FakeDownloadConfig,
        download_images=lambda config: SimpleNamespace(
            downloaded_images=1,
            discovered_images=["GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/test-image-1"],
            manifest_path=Path(config.output_root) / "manifest.json",
        ),
    )
    request = AlphaEarthDownloadRequest(
        year=2024,
        output_root="data/alphaearth",
        aoi_label="east-anglia",
        bbox=(-1.0, 51.0, 0.0, 52.0),
    )

    result = download_alphaearth_images(request, fake_module)
    assert result.aoi_label == "east-anglia"
    assert result.output_root == Path(
        "data/alphaearth/raw/alphaearth/AEF_east-anglia_annual_embedding_2024"
    )
    assert result.source_image_ids == ("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/test-image-1",)
    assert result.manifest_path == Path(
        "data/alphaearth/raw/alphaearth/AEF_east-anglia_annual_embedding_2024/manifest.json"
    )


def test_download_alphaearth_images_supports_edown_results_summary_shape() -> None:
    fake_module = SimpleNamespace(
        AOI=FakeAOI,
        DownloadConfig=FakeDownloadConfig,
        download_images=lambda config: SimpleNamespace(
            downloaded=1,
            failed=0,
            skipped=0,
            manifest_path=Path(config.output_root) / "manifests" / "run.json",
            results=(
                SimpleNamespace(
                    image_id="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/test-image-2",
                    status="downloaded",
                ),
            ),
        ),
    )
    request = AlphaEarthDownloadRequest(
        year=2024,
        output_root="data/alphaearth",
        aoi_label="east-anglia",
        bbox=(-1.0, 51.0, 0.0, 52.0),
    )

    result = download_alphaearth_images(request, fake_module)
    assert result.source_image_ids == ("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/test-image-2",)
    assert result.manifest_path == Path(
        "data/alphaearth/raw/alphaearth/AEF_east-anglia_annual_embedding_2024/manifests/run.json"
    )


def test_download_alphaearth_images_filters_out_of_year_source_image_ids(tmp_path: Path) -> None:
    output_root = tmp_path / "data" / "alphaearth" / "raw" / "alphaearth" / "AEF_east-anglia_annual_embedding_2017"
    manifest_path = output_root / "manifests" / "run.json"
    first_raster = output_root / "images" / "alphaearth_2017.tif"
    second_raster = output_root / "images" / "alphaearth_2018.tif"
    _write_feature_raster(first_raster)
    _write_feature_raster(second_raster)
    _write_manifest(
        manifest_path,
        [
            (
                "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/test-image-2017",
                first_raster,
                "2017-01-01T00:00:00+00:00",
            ),
            (
                "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/test-image-2018",
                second_raster,
                "2018-01-01T00:00:00+00:00",
            ),
        ],
        start_date="2017-01-01",
    )
    fake_module = SimpleNamespace(
        AOI=FakeAOI,
        DownloadConfig=FakeDownloadConfig,
        download_images=lambda _config: SimpleNamespace(
            downloaded=2,
            failed=0,
            skipped=0,
            manifest_path=manifest_path,
            results=(
                SimpleNamespace(
                    image_id="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/test-image-2017",
                    status="downloaded",
                ),
                SimpleNamespace(
                    image_id="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/test-image-2018",
                    status="downloaded",
                ),
            ),
        ),
    )
    request = AlphaEarthDownloadRequest(
        year=2017,
        output_root=tmp_path / "data" / "alphaearth",
        aoi_label="east-anglia",
        bbox=(-1.0, 51.0, 0.0, 52.0),
    )

    result = download_alphaearth_images(request, fake_module)

    assert result.source_image_ids == (
        "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/test-image-2017",
    )


def test_download_alphaearth_images_raises_disk_quota_failure() -> None:
    fake_module = SimpleNamespace(
        AOI=FakeAOI,
        DownloadConfig=FakeDownloadConfig,
        download_images=lambda _config: SimpleNamespace(
            downloaded=0,
            failed=2,
            skipped=0,
            results=(
                SimpleNamespace(image_id="IMG_A", status="failed", error="Disk quota exceeded"),
                SimpleNamespace(image_id="IMG_B", status="failed", error="Disk quota exceeded"),
            ),
        ),
    )
    request = AlphaEarthDownloadRequest(
        year=2024,
        output_root="data/alphaearth",
        aoi_label="east-anglia",
        bbox=(-1.0, 51.0, 0.0, 52.0),
    )

    with pytest.raises(DownloadFailedError, match="ran out of writable space"):
        download_alphaearth_images(request, fake_module)

def test_download_alphaearth_images_raises_when_no_images_match_requested_year(tmp_path: Path) -> None:
    output_root = tmp_path / "raw" / "alphaearth" / "AEF_cambridge_annual_embedding_2017"
    feature_raster = output_root / "images" / "GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL" / "IMAGE_2018.tif"
    _write_feature_raster(feature_raster)
    manifest_path = output_root / "manifests" / "run.json"
    _write_manifest(
        manifest_path,
        [
            ("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/IMAGE_2018", feature_raster, "2018-01-01T00:00:00+00:00"),
        ],
        start_date="2017-01-01",
    )

    fake_module = SimpleNamespace(
        AOI=FakeAOI,
        DownloadConfig=FakeDownloadConfig,
        download_images=lambda _config: SimpleNamespace(
            downloaded=1,
            failed=0,
            skipped=0,
            manifest_path=manifest_path,
            results=(
                SimpleNamespace(
                    image_id="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/IMAGE_2018",
                    status="downloaded",
                ),
            ),
        ),
    )
    request = AlphaEarthDownloadRequest(
        year=2017,
        output_root=tmp_path / "data",
        aoi_label="cambridge",
        bbox=(0.0, 0.0, 1.0, 1.0),
    )

    with pytest.raises(DownloadFailedError, match="requested year 2017"):
        download_alphaearth_images(request, fake_module)
