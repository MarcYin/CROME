from pathlib import Path

from crome.config import AlphaEarthDownloadRequest, AlphaEarthTrainingSpec, CromeReferenceConfig


def test_alphaearth_request_builds_stable_paths_for_bbox() -> None:
    request = AlphaEarthDownloadRequest(
        year=2024,
        output_root=Path("data") / "alphaearth",
        aoi_label="england-se",
        bbox=(-1.0, 51.0, 0.0, 52.0),
    )

    assert request.start_date == "2024-01-01"
    assert request.end_date == "2025-01-01"
    assert request.conditional_year is False
    assert request.dataset_output_root == Path(
        "data/alphaearth/raw/alphaearth/AEF_england-se_annual_embedding_2024"
    )


def test_alphaearth_request_flags_2025_as_conditional() -> None:
    request = AlphaEarthDownloadRequest(
        year=2025,
        output_root="data/alphaearth",
        geojson="aoi.geojson",
    )

    assert request.conditional_year is True
    assert request.aoi_label == "aoi"


def test_training_spec_tracks_crome_hexagon_reference_outputs() -> None:
    alphaearth = AlphaEarthDownloadRequest(
        year=2024,
        output_root="data/alphaearth",
        aoi_label="east-anglia",
        bbox=(0.0, 52.0, 1.0, 53.0),
    )
    reference = CromeReferenceConfig(
        source_path="crome_2024.fgb",
        year=2024,
        aoi_label="east-anglia",
    )
    spec = AlphaEarthTrainingSpec(alphaearth=alphaearth, reference=reference)

    assert spec.reference_output_root == Path(
        "data/alphaearth/reference/crome_hex/REF_crome_hex_east-anglia_2024"
    )
    assert spec.training_output_root == Path("data/alphaearth/training/TRAIN_east-anglia_2024")
