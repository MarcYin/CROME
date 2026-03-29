from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from crome.acquisition.alphaearth import build_download_config, download_alphaearth_images
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
