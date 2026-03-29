from pathlib import Path

from crome.config import AlphaEarthDownloadRequest


def test_alphaearth_request_builds_stable_paths_for_bbox() -> None:
    request = AlphaEarthDownloadRequest(
        year=2024,
        output_root=Path("data") / "alphaearth",
        tile_id="30UXD",
        bbox=(-1.0, 51.0, 0.0, 52.0),
    )

    assert request.start_date == "2024-01-01"
    assert request.end_date == "2025-01-01"
    assert request.conditional_year is False
    assert request.dataset_output_root == Path("data/alphaearth/AEF_30UXD_annual_embedding_2024")


def test_alphaearth_request_flags_2025_as_conditional() -> None:
    request = AlphaEarthDownloadRequest(
        year=2025,
        output_root="data/alphaearth",
        geojson="aoi.geojson",
    )

    assert request.conditional_year is True
    assert request.tile_id == "aoi"
