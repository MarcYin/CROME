from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

import crome.training as training_module
from crome.bands import ALPHAEARTH_BANDS
from crome.config import AlphaEarthDownloadRequest, AlphaEarthTrainingSpec, CromeReferenceConfig
from crome.discovery import discover_feature_rasters
from crome.labeling import rasterize_crome_reference
from crome.pipeline import run_baseline_pipeline
from crome.predict import predict_crop_map
from crome.training import TrainingRasterPair, build_training_table, train_random_forest


def _write_feature_raster(
    path: Path,
    *,
    nodata: float | None = None,
    invalid_pixels: tuple[tuple[int, int], ...] = (),
    pattern: str = "split",
    value_offset: float = 0.0,
    width: int = 4,
    height: int = 4,
    x_origin: float = 0.0,
    y_origin: float = 4.0,
) -> None:
    data = np.zeros((len(ALPHAEARTH_BANDS), height, width), dtype="float32")
    if pattern == "split":
        split = max(width // 2, 1)
        data[:, :, :split] = 0.0 + value_offset
        data[:, :, split:] = 100.0 + value_offset
    elif pattern == "low":
        data[:, :, :] = 0.0 + value_offset
    elif pattern == "high":
        data[:, :, :] = 100.0 + value_offset
    else:
        raise ValueError(f"Unsupported pattern: {pattern}")
    if nodata is not None:
        for row, col in invalid_pixels:
            data[:, row, col] = nodata

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(ALPHAEARTH_BANDS),
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": from_origin(x_origin, y_origin, 1, 1),
    }
    if nodata is not None:
        profile["nodata"] = nodata
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


def _write_reference_gpkg(path: Path) -> None:
    left = Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])
    right = Polygon([(2, 0), (4, 0), (4, 4), (2, 4)])
    gdf = gpd.GeoDataFrame(
        {"lucode": ["wheat", "barley"], "geometry": [left, right]},
        crs="EPSG:3857",
    )
    gdf.to_file(path, driver="GPKG")


def _write_reference_fgb(path: Path) -> None:
    left = Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])
    right = Polygon([(2, 0), (4, 0), (4, 4), (2, 4)])
    gdf = gpd.GeoDataFrame(
        {"lucode": ["wheat", "barley"], "geometry": [left, right]},
        crs="EPSG:3857",
    )
    gdf.to_file(path, driver="FlatGeobuf")


def _write_multilayer_reference_gpkg(path: Path) -> None:
    non_overlapping = gpd.GeoDataFrame(
        {
            "lucode": ["water"],
            "geometry": [Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])],
        },
        crs="EPSG:3857",
    )
    overlapping = gpd.GeoDataFrame(
        {
            "lucode": ["wheat", "barley"],
            "geometry": [
                Polygon([(0, 0), (2, 0), (2, 4), (0, 4)]),
                Polygon([(2, 0), (4, 0), (4, 4), (2, 4)]),
            ],
        },
        crs="EPSG:3857",
    )
    non_overlapping.to_file(path, layer="County_A", driver="GPKG")
    overlapping.to_file(path, layer="County_B", driver="GPKG")


def _hexagon(center_x: float, center_y: float, radius: float) -> Polygon:
    vertices = [
        (
            center_x + radius * math.cos(math.radians(angle)),
            center_y + radius * math.sin(math.radians(angle)),
        )
        for angle in range(0, 360, 60)
    ]
    return Polygon(vertices)


def _write_reference_hex_geojson(path: Path) -> None:
    left = _hexagon(1.5, 2.5, 1.1)
    right = _hexagon(5.5, 2.5, 1.1)
    gdf = gpd.GeoDataFrame(
        {"lucode": ["wheat", "barley"], "geometry": [left, right]},
        crs="EPSG:3857",
    )
    gdf.to_file(path, driver="GeoJSON")


def _write_manifest(
    path: Path,
    rasters: list[tuple[str, Path]],
    *,
    aoi_bounds: tuple[float, float, float, float] | None = None,
    chunk_count: int | None = None,
) -> None:
    output_root = path.parent.parent if path.parent.name == "manifests" else path.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"output_root": str(output_root)},
        "download": {
            "output_root": str(output_root),
            "results": [
                {
                    "chunk_count": chunk_count,
                    "image_id": image_id,
                    "status": "downloaded",
                    "tiff_path": str(raster.relative_to(output_root)),
                }
                for image_id, raster in rasters
            ],
        },
        "schema_version": "0.1.1",
        "search": {
            "aoi_bounds": list(aoi_bounds) if aoi_bounds is not None else None,
            "images": [
                {
                    "image_id": image_id,
                    "relative_tiff_path": str(raster.relative_to(output_root)),
                }
                for image_id, raster in rasters
            ]
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    spec = AlphaEarthTrainingSpec(
        alphaearth=alphaearth,
        reference=reference,
        label_mode="polygon_to_pixel",
    )

    rasterized = rasterize_crome_reference(feature_raster, spec)
    training_table = build_training_table(
        feature_raster,
        rasterized.label_raster_path,
        tmp_path / "outputs" / "training",
    )
    trained = train_random_forest(training_table.table_path, tmp_path / "outputs" / "model")
    metrics = json.loads(trained.metrics_path.read_text(encoding="utf-8"))
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
    assert metrics["macro_f1"] is not None
    assert metrics["weighted_f1"] is not None

    mapping = json.loads(rasterized.label_mapping_path.read_text(encoding="utf-8"))
    assert mapping["label_to_id"] == {"barley": 0, "wheat": 1}
    assert mapping["label_stats"]["labeled_pixel_count"] == 16
    assert mapping["reference_feature_count"] == 2


def test_rasterize_reference_supports_gpkg_sources(tmp_path: Path) -> None:
    feature_raster = tmp_path / "alphaearth.tif"
    reference_gpkg = tmp_path / "crome.gpkg"
    _write_feature_raster(feature_raster)
    _write_reference_gpkg(reference_gpkg)

    alphaearth = AlphaEarthDownloadRequest(
        year=2024,
        output_root=tmp_path / "outputs",
        aoi_label="east-anglia",
        bbox=(0.0, 0.0, 4.0, 4.0),
    )
    reference = CromeReferenceConfig(
        source_path=reference_gpkg,
        year=2024,
        aoi_label="east-anglia",
    )
    spec = AlphaEarthTrainingSpec(
        alphaearth=alphaearth,
        reference=reference,
        label_mode="polygon_to_pixel",
    )

    rasterized = rasterize_crome_reference(feature_raster, spec)

    assert rasterized.label_raster_path.exists()
    mapping = json.loads(rasterized.label_mapping_path.read_text(encoding="utf-8"))
    assert mapping["label_to_id"] == {"barley": 0, "wheat": 1}


def test_rasterize_reference_supports_multilayer_gpkg_sources(tmp_path: Path) -> None:
    feature_raster = tmp_path / "alphaearth.tif"
    reference_gpkg = tmp_path / "crome_multi.gpkg"
    _write_feature_raster(feature_raster)
    _write_multilayer_reference_gpkg(reference_gpkg)

    alphaearth = AlphaEarthDownloadRequest(
        year=2024,
        output_root=tmp_path / "outputs",
        aoi_label="east-anglia",
        bbox=(0.0, 0.0, 4.0, 4.0),
    )
    reference = CromeReferenceConfig(
        source_path=reference_gpkg,
        year=2024,
        aoi_label="east-anglia",
    )
    spec = AlphaEarthTrainingSpec(
        alphaearth=alphaearth,
        reference=reference,
        label_mode="polygon_to_pixel",
    )

    rasterized = rasterize_crome_reference(feature_raster, spec)

    assert rasterized.label_raster_path.exists()
    mapping = json.loads(rasterized.label_mapping_path.read_text(encoding="utf-8"))
    assert mapping["label_to_id"] == {"barley": 0, "wheat": 1}


def test_rasterize_reference_supports_flatgeobuf_sources(tmp_path: Path) -> None:
    feature_raster = tmp_path / "alphaearth.tif"
    reference_fgb = tmp_path / "crome.fgb"
    _write_feature_raster(feature_raster)
    _write_reference_fgb(reference_fgb)

    alphaearth = AlphaEarthDownloadRequest(
        year=2024,
        output_root=tmp_path / "outputs",
        aoi_label="east-anglia",
        bbox=(0.0, 0.0, 4.0, 4.0),
    )
    reference = CromeReferenceConfig(
        source_path=reference_fgb,
        year=2024,
        aoi_label="east-anglia",
    )
    spec = AlphaEarthTrainingSpec(
        alphaearth=alphaearth,
        reference=reference,
        label_mode="polygon_to_pixel",
    )

    rasterized = rasterize_crome_reference(feature_raster, spec)

    assert rasterized.label_raster_path.exists()
    mapping = json.loads(rasterized.label_mapping_path.read_text(encoding="utf-8"))
    assert mapping["label_to_id"] == {"barley": 0, "wheat": 1}


def test_centroid_to_pixel_labels_one_pixel_per_reference_hexagon(tmp_path: Path) -> None:
    feature_raster = tmp_path / "alphaearth.tif"
    reference_geojson = tmp_path / "crome_hex.geojson"
    _write_feature_raster(feature_raster, width=7, height=5, y_origin=5.0)
    _write_reference_hex_geojson(reference_geojson)

    alphaearth = AlphaEarthDownloadRequest(
        year=2024,
        output_root=tmp_path / "outputs",
        aoi_label="east-anglia",
        bbox=(0.0, 0.0, 7.0, 5.0),
    )
    reference = CromeReferenceConfig(
        source_path=reference_geojson,
        year=2024,
        aoi_label="east-anglia",
    )

    centroid_spec = AlphaEarthTrainingSpec(
        alphaearth=alphaearth,
        reference=reference,
        label_mode="centroid_to_pixel",
    )
    polygon_spec = AlphaEarthTrainingSpec(
        alphaearth=alphaearth,
        reference=reference,
        label_mode="polygon_to_pixel",
    )

    centroid_rasterized = rasterize_crome_reference(
        feature_raster,
        centroid_spec,
        output_dir=tmp_path / "outputs" / "centroid",
    )
    polygon_rasterized = rasterize_crome_reference(
        feature_raster,
        polygon_spec,
        output_dir=tmp_path / "outputs" / "polygon",
    )

    with rasterio.open(centroid_rasterized.label_raster_path) as src:
        centroid_labels = src.read(1)
    with rasterio.open(polygon_rasterized.label_raster_path) as src:
        polygon_labels = src.read(1)

    centroid_valid = centroid_labels != -1
    polygon_valid = polygon_labels != -1
    assert int(centroid_valid.sum()) == 2
    assert int(polygon_valid.sum()) > int(centroid_valid.sum())
    assert centroid_labels[2, 1] == 1
    assert centroid_labels[2, 5] == 0


def test_training_and_prediction_ignore_finite_feature_nodata(tmp_path: Path) -> None:
    feature_raster = tmp_path / "alphaearth_nodata.tif"
    reference_geojson = tmp_path / "crome.geojson"
    _write_feature_raster(
        feature_raster,
        nodata=-9999.0,
        invalid_pixels=((0, 0),),
    )
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
    spec = AlphaEarthTrainingSpec(
        alphaearth=alphaearth,
        reference=reference,
        label_mode="polygon_to_pixel",
    )

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

    assert training_table.row_count == 15
    with rasterio.open(prediction.prediction_raster_path) as pred_src:
        preds = pred_src.read(1)
    assert preds[0, 0] == -1


def test_discover_feature_rasters_reads_manifest_and_filters_feature_files(tmp_path: Path) -> None:
    feature_dir = tmp_path / "raw"
    feature_dir.mkdir()
    first_raster = feature_dir / "alphaearth_a.tif"
    second_raster = feature_dir / "alphaearth_b.tif"
    noise_raster = feature_dir / "noise.tif"
    _write_feature_raster(first_raster)
    _write_feature_raster(second_raster, value_offset=10.0)
    with rasterio.open(
        noise_raster,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 4, 1, 1),
    ) as dst:
        dst.write(np.zeros((1, 4, 4), dtype="float32"))

    manifest_path = feature_dir / "manifests" / "run.json"
    _write_manifest(
        manifest_path,
        [
            ("IMAGE_A", first_raster),
            ("IMAGE_B", second_raster),
        ],
    )

    discovered = discover_feature_rasters(manifest_path=manifest_path)
    assert len(discovered) == 2
    assert [item.feature_id for item in discovered] == ["alphaearth_a", "alphaearth_b"]
    assert [item.source_image_id for item in discovered] == ["IMAGE_A", "IMAGE_B"]


def test_run_baseline_pipeline_batches_native_feature_rasters(tmp_path: Path) -> None:
    feature_dir = tmp_path / "raw"
    feature_dir.mkdir()
    first_raster = feature_dir / "alphaearth_full.tif"
    second_raster = feature_dir / "alphaearth_left.tif"
    third_raster = feature_dir / "alphaearth_right.tif"
    reference_geojson = tmp_path / "crome.geojson"
    manifest_path = feature_dir / "manifests" / "run.json"
    _write_feature_raster(first_raster)
    _write_feature_raster(second_raster, pattern="low", width=2, x_origin=0.0)
    _write_feature_raster(third_raster, pattern="high", width=2, x_origin=2.0)
    _write_reference_geojson(reference_geojson)
    _write_manifest(
        manifest_path,
        [
            ("IMAGE_FULL", first_raster),
            ("IMAGE_LEFT", second_raster),
            ("IMAGE_RIGHT", third_raster),
        ],
        aoi_bounds=(0.0, 0.0, 4.0, 4.0),
    )

    result = run_baseline_pipeline(
        feature_input=feature_dir,
        manifest_path=manifest_path,
        reference_path=reference_geojson,
        year=2024,
        output_root=tmp_path / "outputs",
        aoi_label="east-anglia",
        label_mode="polygon_to_pixel",
        test_size=0.0,
    )

    assert len(result.feature_results) == 3
    assert not result.skipped_features
    assert result.pipeline_manifest_path.exists()
    assert result.qc_manifest_path.exists()
    assert result.sample_cache_root is not None

    feature_tables = []

    for feature in result.feature_results:
        assert feature.tile_id in {
            "IMAGE_FULL",
            "IMAGE_LEFT",
            "IMAGE_RIGHT",
        }
        assert feature.label_raster_path.exists()
        assert feature.training_table_path.exists()
        assert feature.training_metadata_path.exists()
        assert feature.model_path.exists()
        assert feature.metrics_path.exists()
        assert feature.qc_manifest_path.exists()
        assert feature.sample_cache_manifest_path is not None
        assert feature.sample_cache_manifest_path.exists()
        assert feature.prediction_raster_path is not None
        feature_tables.append(pd.read_pickle(feature.training_table_path))
        with rasterio.open(feature.label_raster_path) as label_src, rasterio.open(
            feature.prediction_raster_path
        ) as pred_src:
            labels = label_src.read(1)
            preds = pred_src.read(1)
        valid = labels != -1
        assert valid.any()
        assert np.array_equal(labels[valid], preds[valid])

    table = pd.concat(feature_tables, ignore_index=True)
    assert set(table["feature_id"]) == {"alphaearth_full", "alphaearth_left", "alphaearth_right"}
    assert set(table["source_image_id"]) == {"IMAGE_FULL", "IMAGE_LEFT", "IMAGE_RIGHT"}

    payload = json.loads(result.pipeline_manifest_path.read_text(encoding="utf-8"))
    assert payload["feature_count"] == 3
    assert len(payload["features"]) == 3
    assert all(item["tile_id"] in {"IMAGE_FULL", "IMAGE_LEFT", "IMAGE_RIGHT"} for item in payload["features"])
    assert all(Path(item["model_path"]).exists() for item in payload["features"])
    assert all(Path(item["training_table_path"]).exists() for item in payload["features"])
    assert payload["qc_manifest_path"] == str(result.qc_manifest_path)
    assert payload["sample_cache_root"] is not None
    assert len(payload["sample_cache_manifest_paths"]) == 3
    assert payload["skipped_feature_count"] == 0

    qc_payload = json.loads(result.qc_manifest_path.read_text(encoding="utf-8"))
    assert qc_payload["feature_count"] == 3
    assert len(qc_payload["sample_cache_manifest_paths"]) == 3
    assert qc_payload["reference_path"] == str(reference_geojson)
    assert qc_payload["reference_summary"]["reference_path"] == str(reference_geojson)
    assert qc_payload["requested_aoi"]["bounds"] == [0.0, 0.0, 4.0, 4.0]
    assert qc_payload["features"][0]["label_qc"]["label_stats"]["labeled_pixel_count"] > 0
    assert Path(qc_payload["features"][0]["label_qc_png_path"]).exists()


def test_run_baseline_pipeline_skips_rasters_without_reference_coverage(tmp_path: Path) -> None:
    feature_dir = tmp_path / "raw"
    feature_dir.mkdir()
    covered_raster = feature_dir / "covered.tif"
    uncovered_raster = feature_dir / "uncovered.tif"
    reference_geojson = tmp_path / "crome.geojson"
    _write_feature_raster(covered_raster)
    _write_feature_raster(uncovered_raster, x_origin=100.0, y_origin=104.0)
    _write_reference_geojson(reference_geojson)

    result = run_baseline_pipeline(
        feature_input=feature_dir,
        manifest_path=None,
        reference_path=reference_geojson,
        year=2024,
        output_root=tmp_path / "outputs",
        aoi_label="east-anglia",
        label_mode="polygon_to_pixel",
    )

    assert len(result.feature_results) == 1
    assert len(result.skipped_features) == 1
    assert result.skipped_features[0].feature_id == "uncovered"


def test_run_baseline_pipeline_materializes_crome_subset_from_extracted_gpkg(tmp_path: Path) -> None:
    feature_dir = tmp_path / "raw"
    feature_dir.mkdir()
    feature_raster = feature_dir / "alphaearth_full.tif"
    manifest_path = feature_dir / "manifests" / "run.json"
    reference_gpkg = (
        tmp_path
        / "raw"
        / "crome"
        / "CROME_2024_national"
        / "extracted"
        / "crome_2024.gpkg"
    )
    _write_feature_raster(feature_raster)
    reference_gpkg.parent.mkdir(parents=True, exist_ok=True)
    _write_reference_gpkg(reference_gpkg)
    _write_manifest(
        manifest_path,
        [("IMAGE_FULL", feature_raster)],
        aoi_bounds=(0.0, 0.0, 4.0, 4.0),
    )

    result = run_baseline_pipeline(
        feature_input=feature_dir,
        manifest_path=manifest_path,
        reference_path=reference_gpkg,
        year=2024,
        output_root=tmp_path / "outputs",
        aoi_label="east-anglia",
        label_mode="polygon_to_pixel",
        test_size=0.0,
    )

    payload = json.loads(result.pipeline_manifest_path.read_text(encoding="utf-8"))
    subset_path = Path(payload["reference_path"])
    assert subset_path.exists()
    assert subset_path.parent.name == "subsets"
    assert subset_path.suffix == ".fgb"
    assert result.reference_input_path == reference_gpkg
    assert result.reference_path == subset_path


def test_run_baseline_pipeline_namespaces_tile_outputs_by_label_mode(tmp_path: Path) -> None:
    feature_raster = tmp_path / "alphaearth.tif"
    reference_geojson = tmp_path / "crome.geojson"
    _write_feature_raster(feature_raster)
    _write_reference_geojson(reference_geojson)

    centroid_result = run_baseline_pipeline(
        feature_input=feature_raster,
        manifest_path=None,
        reference_path=reference_geojson,
        year=2024,
        output_root=tmp_path / "outputs",
        aoi_label="east-anglia",
        label_mode="centroid_to_pixel",
        test_size=0.0,
    )
    polygon_result = run_baseline_pipeline(
        feature_input=feature_raster,
        manifest_path=None,
        reference_path=reference_geojson,
        year=2024,
        output_root=tmp_path / "outputs",
        aoi_label="east-anglia",
        label_mode="polygon_to_pixel",
        test_size=0.0,
    )

    centroid_feature = centroid_result.feature_results[0]
    polygon_feature = polygon_result.feature_results[0]

    assert centroid_result.pipeline_manifest_path != polygon_result.pipeline_manifest_path
    assert centroid_feature.label_raster_path != polygon_feature.label_raster_path
    assert centroid_feature.training_output_root != polygon_feature.training_output_root
    assert centroid_feature.prediction_output_root != polygon_feature.prediction_output_root


def test_build_training_table_reuses_sample_cache(tmp_path: Path, monkeypatch) -> None:
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
    spec = AlphaEarthTrainingSpec(
        alphaearth=alphaearth,
        reference=reference,
        label_mode="polygon_to_pixel",
    )
    rasterized = rasterize_crome_reference(feature_raster, spec)
    rerasterized = rasterize_crome_reference(
        feature_raster,
        spec,
        output_dir=tmp_path / "outputs" / "rerasterized",
    )

    sample_cache_root = tmp_path / "outputs" / "cache"
    first = training_module.build_training_table_from_pairs(
        [
            TrainingRasterPair(
                feature_raster,
                rasterized.label_raster_path,
                label_mapping_path=rasterized.label_mapping_path,
            )
        ],
        tmp_path / "outputs" / "training-first",
        sample_cache_root=sample_cache_root,
    )
    assert first.sample_cache_manifest_path is not None
    assert first.sample_cache_manifest_path.exists()

    def fail_extract(_pair):
        raise AssertionError("cache miss: training rows were extracted again")

    monkeypatch.setattr(training_module, "_extract_training_frame", fail_extract)
    second = training_module.build_training_table_from_pairs(
        [
            TrainingRasterPair(
                feature_raster,
                rerasterized.label_raster_path,
                label_mapping_path=rerasterized.label_mapping_path,
            )
        ],
        tmp_path / "outputs" / "training-second",
        sample_cache_root=sample_cache_root,
    )

    metadata = json.loads(second.metadata_path.read_text(encoding="utf-8"))
    assert metadata["cache_hit_count"] == 1
    assert metadata["cache_miss_count"] == 0
    assert metadata["sample_cache_root"] == str(sample_cache_root)


def test_build_training_table_reuses_cache_when_aoi_label_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    feature_raster = tmp_path / "alphaearth.tif"
    reference_geojson = tmp_path / "crome.geojson"
    _write_feature_raster(feature_raster)
    _write_reference_geojson(reference_geojson)

    alphaearth = AlphaEarthDownloadRequest(
        year=2024,
        output_root=tmp_path / "outputs",
        aoi_label="run-one",
        bbox=(0.0, 0.0, 4.0, 4.0),
    )
    reference = CromeReferenceConfig(
        source_path=reference_geojson,
        year=2024,
        aoi_label=alphaearth.aoi_label,
    )
    spec = AlphaEarthTrainingSpec(
        alphaearth=alphaearth,
        reference=reference,
        label_mode="centroid_to_pixel",
    )

    first_rasterized = rasterize_crome_reference(
        feature_raster,
        spec,
        output_dir=tmp_path / "outputs" / "run-one",
    )
    second_rasterized = rasterize_crome_reference(
        feature_raster,
        spec,
        output_dir=tmp_path / "outputs" / "run-two",
    )
    second_mapping = json.loads(second_rasterized.label_mapping_path.read_text(encoding="utf-8"))
    second_mapping["aoi_label"] = "run-two"
    second_rasterized.label_mapping_path.write_text(
        json.dumps(second_mapping, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    sample_cache_root = tmp_path / "outputs" / "cache"
    training_module.build_training_table_from_pairs(
        [
            TrainingRasterPair(
                feature_raster,
                first_rasterized.label_raster_path,
                label_mapping_path=first_rasterized.label_mapping_path,
            )
        ],
        tmp_path / "outputs" / "training-first",
        sample_cache_root=sample_cache_root,
    )

    def fail_extract(_pair):
        raise AssertionError("cache miss: training rows were extracted again")

    monkeypatch.setattr(training_module, "_extract_training_frame", fail_extract)
    second = training_module.build_training_table_from_pairs(
        [
            TrainingRasterPair(
                feature_raster,
                second_rasterized.label_raster_path,
                label_mapping_path=second_rasterized.label_mapping_path,
            )
        ],
        tmp_path / "outputs" / "training-second",
        sample_cache_root=sample_cache_root,
    )

    metadata = json.loads(second.metadata_path.read_text(encoding="utf-8"))
    assert metadata["cache_hit_count"] == 1
    assert metadata["cache_miss_count"] == 0


def test_build_training_table_from_cache_manifests_deduplicates_cached_sources(tmp_path: Path) -> None:
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
    spec = AlphaEarthTrainingSpec(
        alphaearth=alphaearth,
        reference=reference,
        label_mode="polygon_to_pixel",
    )
    rasterized = rasterize_crome_reference(feature_raster, spec)

    sample_cache_root = tmp_path / "outputs" / "cache"
    built = training_module.build_training_table_from_pairs(
        [
            TrainingRasterPair(
                feature_raster,
                rasterized.label_raster_path,
                label_mapping_path=rasterized.label_mapping_path,
            )
        ],
        tmp_path / "outputs" / "training",
        sample_cache_root=sample_cache_root,
    )
    assert built.sample_cache_manifest_path is not None

    combined = training_module.build_training_table_from_cache_manifests(
        [built.sample_cache_manifest_path, built.sample_cache_manifest_path],
        tmp_path / "outputs" / "training-global",
    )
    assert combined.row_count == built.row_count
    combined_metadata = json.loads(combined.metadata_path.read_text(encoding="utf-8"))
    assert combined_metadata["cache_entry_count"] == 1
