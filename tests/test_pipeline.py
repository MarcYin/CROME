from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from crome.bands import ALPHAEARTH_BANDS
from crome.config import AlphaEarthDownloadRequest, AlphaEarthTrainingSpec, CromeReferenceConfig
from crome.labeling import rasterize_crome_reference
from crome.predict import predict_crop_map
from crome.training import build_training_table, train_random_forest


def _write_feature_raster(path: Path) -> None:
    data = np.zeros((len(ALPHAEARTH_BANDS), 4, 4), dtype="float32")
    data[:, :, :2] = 0.0
    data[:, :, 2:] = 100.0

    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": len(ALPHAEARTH_BANDS),
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 4, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = ALPHAEARTH_BANDS


def _write_reference_geojson(path: Path) -> None:
    left = Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])
    right = Polygon([(2, 0), (4, 0), (4, 4), (2, 4)])
    gdf = gpd.GeoDataFrame(
        {"lucode": ["wheat", "barley"], "geometry": [left, right]},
        crs="EPSG:3857",
    )
    gdf.to_file(path, driver="GeoJSON")


def test_end_to_end_pipeline(tmp_path: Path) -> None:
    feature_raster = tmp_path / "alphaearth.tif"
    reference_geojson = tmp_path / "crome.geojson"
    _write_feature_raster(feature_raster)
    _write_reference_geojson(reference_geojson)

    alphaearth = AlphaEarthDownloadRequest(
        year=2024,
        output_root=tmp_path / "outputs",
        aoi_label="east-anglia",
        bbox=(0.0, 0.0, 4.0, 4.0),
    )
    reference = CromeReferenceConfig(
        source_path=reference_geojson,
        year=2024,
        aoi_label="east-anglia",
    )
    spec = AlphaEarthTrainingSpec(alphaearth=alphaearth, reference=reference)

    rasterized = rasterize_crome_reference(feature_raster, spec)
    training_table = build_training_table(
        feature_raster,
        rasterized.label_raster_path,
        tmp_path / "outputs" / "training",
    )
    trained = train_random_forest(training_table.table_path, tmp_path / "outputs" / "model")
    prediction = predict_crop_map(
        feature_raster,
        trained.model_path,
        tmp_path / "outputs" / "prediction.tif",
    )

    assert rasterized.label_raster_path.exists()
    assert training_table.table_path.exists()
    assert trained.model_path.exists()
    assert prediction.prediction_raster_path.exists()

    with rasterio.open(rasterized.label_raster_path) as labels_src, rasterio.open(
        prediction.prediction_raster_path
    ) as pred_src:
        labels = labels_src.read(1)
        preds = pred_src.read(1)

    valid = labels != -1
    assert valid.any()
    assert np.array_equal(labels[valid], preds[valid])

    mapping = json.loads(rasterized.label_mapping_path.read_text(encoding="utf-8"))
    assert mapping["label_to_id"] == {"barley": 0, "wheat": 1}
