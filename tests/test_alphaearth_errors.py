import pytest

from crome.acquisition.alphaearth import _load_edown
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
            aoi_label="england-se",
        )


def test_alphaearth_request_rejects_invalid_bbox_order() -> None:
    with pytest.raises(ValueError, match="Bounding boxes must be ordered"):
        AlphaEarthDownloadRequest(
            year=2024,
            output_root="data/alphaearth",
            bbox=(1.0, 51.0, 0.0, 52.0),
        )


def test_training_spec_requires_matching_aoi_labels() -> None:
    alphaearth = AlphaEarthDownloadRequest(
        year=2024,
        output_root="data/alphaearth",
        aoi_label="east-anglia",
        bbox=(0.0, 52.0, 1.0, 53.0),
    )

    with pytest.raises(ValueError, match="AOI label"):
        from crome.config import AlphaEarthTrainingSpec, CromeReferenceConfig

        AlphaEarthTrainingSpec(
            alphaearth=alphaearth,
            reference=CromeReferenceConfig(
                source_path="crome_2024.fgb",
                year=2024,
                aoi_label="england-ne",
            ),
        )


def test_load_edown_error_mentions_supported_install_path(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "edown":
            raise ImportError("edown missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="pip install edown>=0.2.0"):
        _load_edown()
