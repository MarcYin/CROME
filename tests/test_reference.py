import pytest

from crome.config import AlphaEarthDownloadRequest, AlphaEarthTrainingSpec, CromeReferenceConfig
from crome.reference import validate_reference_columns
from crome.schema import alphaearth_feature_columns, validate_feature_order, validate_reference_contract


def test_reference_columns_validation_requires_label_and_geometry() -> None:
    columns = ("lucode", "geometry", "farm_id")
    assert validate_reference_columns(columns, "lucode", "geometry") == columns

    with pytest.raises(ValueError, match="missing required columns"):
        validate_reference_columns(("lucode",), "lucode", "geometry")


def test_reference_contract_distinguishes_vector_labels_from_feature_schema() -> None:
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

    assert validate_feature_order(alphaearth_feature_columns()) == alphaearth_feature_columns()
    assert validate_reference_contract(("lucode", "geometry"), spec) is spec
