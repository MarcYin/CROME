from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from crome.config import AlphaEarthDownloadRequest, AlphaEarthTrainingSpec, CromeReferenceConfig
from crome.labeling import load_reference_label_mapping
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


def test_load_reference_label_mapping_reads_distinct_labels_from_gpkg(tmp_path: Path) -> None:
    gpkg_path = tmp_path / "crome.gpkg"
    left = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    right = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    gdf = gpd.GeoDataFrame(
        {"lucode": ["wheat", "barley"], "geometry": [left, right]},
        crs="EPSG:3857",
    )
    gdf.to_file(gpkg_path, driver="GPKG")

    label_to_id, labels = load_reference_label_mapping(gpkg_path, "lucode")

    assert labels == ("barley", "wheat")
    assert label_to_id == {"barley": 0, "wheat": 1}


def test_load_reference_label_mapping_unions_multilayer_gpkg(tmp_path: Path) -> None:
    gpkg_path = tmp_path / "crome_multi.gpkg"
    county_a = gpd.GeoDataFrame(
        {
            "lucode": ["wheat"],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        crs="EPSG:3857",
    )
    county_b = gpd.GeoDataFrame(
        {
            "lucode": ["water"],
            "geometry": [Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])],
        },
        crs="EPSG:3857",
    )
    county_a.to_file(gpkg_path, layer="County_A", driver="GPKG")
    county_b.to_file(gpkg_path, layer="County_B", driver="GPKG")

    label_to_id, labels = load_reference_label_mapping(gpkg_path, "lucode")

    assert labels == ("water", "wheat")
    assert label_to_id == {"water": 0, "wheat": 1}
