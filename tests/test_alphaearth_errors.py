import pytest

from crome.config import AlphaEarthDownloadRequest


def test_alphaearth_request_rejects_out_of_range_year() -> None:
    with pytest.raises(ValueError, match="between 2017 and 2025"):
        AlphaEarthDownloadRequest(
            year=2016,
            output_root="data/alphaearth",
            bbox=(-1.0, 51.0, 0.0, 52.0),
        )


def test_alphaearth_request_requires_exactly_one_aoi_input() -> None:
    with pytest.raises(ValueError, match="exactly one of bbox or geojson"):
        AlphaEarthDownloadRequest(
            year=2024,
            output_root="data/alphaearth",
            tile_id="30UXD",
        )


def test_alphaearth_request_rejects_invalid_bbox_order() -> None:
    with pytest.raises(ValueError, match="Bounding boxes must be ordered"):
        AlphaEarthDownloadRequest(
            year=2024,
            output_root="data/alphaearth",
            bbox=(1.0, 51.0, 0.0, 52.0),
        )
