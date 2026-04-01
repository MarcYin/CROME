from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from crome import cli
from crome.bands import ALPHAEARTH_BANDS


def _write_feature_raster(path: Path, *, x_origin: float) -> None:
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
        "transform": from_origin(x_origin, 4, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = ALPHAEARTH_BANDS


def _write_reference_gpkg(path: Path) -> None:
    left = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    right = Polygon([(4, 0), (8, 0), (8, 4), (4, 4)])
    gdf = gpd.GeoDataFrame(
        {"lucode": ["wheat", "barley"], "geometry": [left, right]},
        crs="EPSG:3857",
    )
    gdf.to_file(path, driver="GPKG")


def _write_manifest(path: Path, rasters: list[tuple[str, Path]]) -> None:
    output_root = path.parent.parent if path.parent.name == "manifests" else path.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"output_root": str(output_root)},
        "download": {
            "output_root": str(output_root),
            "results": [
                {
                    "image_id": image_id,
                    "status": "downloaded",
                    "tiff_path": str(raster.relative_to(output_root)),
                }
                for image_id, raster in rasters
            ],
        },
        "schema_version": "0.1.1",
        "search": {
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


def _write_managed_reference(tmp_path: Path) -> Path:
    reference_path = (
        tmp_path
        / "raw"
        / "crome"
        / "CROME_2024_national"
        / "extracted"
        / "Crop_Map_of_England_CROME_2024.gpkg"
    )
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    _write_reference_gpkg(reference_path)
    return reference_path


def test_cli_prepare_tile_batch_writes_tile_manifests(tmp_path: Path, capsys) -> None:
    output_root = tmp_path / "outputs"
    image_root = output_root / "images"
    image_root.mkdir(parents=True)
    feature_a = image_root / "tile_a.tif"
    feature_b = image_root / "tile_b.tif"
    _write_feature_raster(feature_a, x_origin=0.0)
    _write_feature_raster(feature_b, x_origin=4.0)
    manifest_path = output_root / "manifests" / "run.json"
    _write_manifest(
        manifest_path,
        [
            ("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/tile_a", feature_a),
            ("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/tile_b", feature_b),
        ],
    )
    reference_path = _write_managed_reference(tmp_path)

    exit_code = cli.main(
        [
            "prepare-tile-batch",
            "--manifest-path",
            str(manifest_path),
            "--reference-path",
            str(reference_path),
            "--year",
            "2024",
            "--output-root",
            str(output_root),
            "--aoi-label",
            "cambridge-norfolk",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tile_count"] == 2
    batch_manifest_path = Path(payload["batch_manifest_path"])
    assert batch_manifest_path.exists()
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    assert batch_manifest["reference_path"].endswith(".fgb")
    assert len(batch_manifest["tile_manifest_paths"]) == 2
    tile_manifests = [Path(path) for path in batch_manifest["tile_manifest_paths"]]
    assert all(path.exists() for path in tile_manifests)
    tile_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in tile_manifests]
    assert len({payload["tile_run_label"] for payload in tile_payloads}) == 2
    assert all(payload["reference_path"] == batch_manifest["reference_path"] for payload in tile_payloads)


def test_cli_run_tile_plan_and_pooled_training(tmp_path: Path, capsys) -> None:
    output_root = tmp_path / "outputs"
    image_root = output_root / "images"
    image_root.mkdir(parents=True)
    feature_a = image_root / "tile_a.tif"
    feature_b = image_root / "tile_b.tif"
    _write_feature_raster(feature_a, x_origin=0.0)
    _write_feature_raster(feature_b, x_origin=4.0)
    manifest_path = output_root / "manifests" / "run.json"
    _write_manifest(
        manifest_path,
        [
            ("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/tile_a", feature_a),
            ("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL/tile_b", feature_b),
        ],
    )
    reference_path = _write_managed_reference(tmp_path)

    prepare_exit = cli.main(
        [
            "prepare-tile-batch",
            "--manifest-path",
            str(manifest_path),
            "--reference-path",
            str(reference_path),
            "--year",
            "2024",
            "--output-root",
            str(output_root),
            "--aoi-label",
            "cambridge-norfolk",
            "--n-estimators",
            "10",
        ]
    )
    assert prepare_exit == 0
    prepare_payload = json.loads(capsys.readouterr().out)
    batch_manifest_path = Path(prepare_payload["batch_manifest_path"])
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))

    tile_result_paths: list[Path] = []
    for tile_manifest_path in batch_manifest["tile_manifest_paths"]:
        exit_code = cli.main(["run-tile-plan", "--tile-plan", tile_manifest_path])
        assert exit_code == 0
        tile_payload = json.loads(capsys.readouterr().out)
        assert tile_payload["feature_count"] == 1
        pipeline_manifest_path = Path(tile_payload["pipeline_manifest_path"])
        assert pipeline_manifest_path.exists()
        tile_result_path = tmp_path / f"{Path(tile_manifest_path).stem}.tile-result.json"
        tile_result_path.write_text(json.dumps(tile_payload, indent=2, sort_keys=True), encoding="utf-8")
        tile_result_paths.append(tile_result_path)

    pooled_exit = cli.main(
        [
            "train-pooled-from-tile-results",
            "--batch-manifest",
            str(batch_manifest_path),
            "--tile-result",
            str(tile_result_paths[0]),
            "--tile-result",
            str(tile_result_paths[1]),
        ]
    )
    assert pooled_exit == 0
    pooled_payload = json.loads(capsys.readouterr().out)
    metrics_path = Path(pooled_payload["metrics_path"])
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["evaluation_mode"] in {"feature_holdout", "train_only_no_holdout"}
    assert Path(pooled_payload["pooled_manifest_path"]).exists()
