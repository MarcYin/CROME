import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from crome import cli
from crome.bands import ALPHAEARTH_BANDS


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


def test_cli_download_alphaearth_dry_run(capsys) -> None:
    exit_code = cli.main(
        [
            "download-alphaearth",
            "--year",
            "2024",
            "--aoi-label",
            "east-anglia",
            "--bbox",
            "-1.0",
            "51.0",
            "0.0",
            "52.0",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["aoi_label"] == "east-anglia"
    assert payload["bands"][0] == "A00"
    assert payload["bands"][-1] == "A63"


def test_cli_download_alphaearth_forwards_request(monkeypatch, capsys) -> None:
    captured = {}

    def fake_download(request):
        captured["request"] = request
        return SimpleNamespace(
            aoi_label=request.aoi_label,
            bands=request.bands,
            collection_id=request.collection_id,
            conditional_year=request.conditional_year,
            manifest_path=None,
            output_root=request.dataset_output_root,
            source_image_ids=(),
            year=request.year,
        )

    monkeypatch.setattr(cli.alphaearth, "download_alphaearth_images", fake_download)

    exit_code = cli.main(
        [
            "download-alphaearth",
            "--year",
            "2024",
            "--aoi-label",
            "east-anglia",
            "--bbox",
            "-1.0",
            "51.0",
            "0.0",
            "52.0",
        ]
    )

    assert exit_code == 0
    assert captured["request"].aoi_label == "east-anglia"
    assert json.loads(capsys.readouterr().out)["year"] == 2024


def test_cli_requires_subcommand() -> None:
    try:
        cli.main([])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("Expected argparse to exit when no subcommand is supplied.")


def test_cli_entrypoint_forwards_sys_argv(monkeypatch) -> None:
    captured = {}

    def fake_main(argv=None):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(cli.alphaearth, "main", fake_main)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "crome",
            "download-alphaearth",
            "--year",
            "2024",
            "--aoi-label",
            "east-anglia",
            "--bbox",
            "-1.0",
            "51.0",
            "0.0",
            "52.0",
            "--dry-run",
        ],
    )

    assert cli.main() == 0
    assert captured["argv"] == [
        "--year",
        "2024",
        "--aoi-label",
        "east-anglia",
        "--bbox",
        "-1.0",
        "51.0",
        "0.0",
        "52.0",
        "--dry-run",
    ]


def test_cli_end_to_end_pipeline(tmp_path: Path) -> None:
    feature_raster = tmp_path / "alphaearth.tif"
    reference_geojson = tmp_path / "crome.geojson"
    output_root = tmp_path / "outputs"
    training_dir = output_root / "training-cli"
    model_dir = output_root / "model-cli"
    prediction_raster = output_root / "prediction-cli.tif"
    _write_feature_raster(feature_raster)
    _write_reference_geojson(reference_geojson)

    assert (
        cli.main(
            [
                "rasterize-reference",
                "--feature-raster",
                str(feature_raster),
                "--reference-path",
                str(reference_geojson),
                "--year",
                "2024",
                "--aoi-label",
                "east-anglia",
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )

    label_raster = (
        output_root
        / "reference"
        / "crome_hex"
        / "REF_crome_hex_east-anglia_2024"
        / "labels.tif"
    )
    assert label_raster.exists()

    assert (
        cli.main(
            [
                "build-training-table",
                "--feature-raster",
                str(feature_raster),
                "--label-raster",
                str(label_raster),
                "--output-dir",
                str(training_dir),
            ]
        )
        == 0
    )
    training_table = training_dir / "training_table.pkl"
    assert training_table.exists()

    assert (
        cli.main(
            [
                "train-model",
                "--training-table",
                str(training_table),
                "--output-dir",
                str(model_dir),
            ]
        )
        == 0
    )
    model_path = model_dir / "model.pkl"
    assert model_path.exists()

    assert (
        cli.main(
            [
                "predict-map",
                "--feature-raster",
                str(feature_raster),
                "--model-path",
                str(model_path),
                "--output-raster",
                str(prediction_raster),
            ]
        )
        == 0
    )
    assert prediction_raster.exists()
