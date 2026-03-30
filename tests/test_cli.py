import json
import os
from pathlib import Path
import subprocess
import sys
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


def _write_reference_gpkg(path: Path) -> None:
    left = Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])
    right = Polygon([(2, 0), (4, 0), (4, 4), (2, 4)])
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


def test_cli_download_crome_dry_run(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli.crome,
        "resolve_crome_landing_page",
        lambda request: SimpleNamespace(
            dataset_id="complete",
            title="Crop Map of England (CROME) 2017 - Complete",
            url="https://environment.data.gov.uk/dataset/complete",
            variant="Complete",
            variant_label="complete",
        ),
    )
    monkeypatch.setattr(
        cli.crome,
        "extract_crome_gpkg_zip_url",
        lambda landing_page, timeout_s=30.0: (
            "https://example.test/Crop_Map_of_England_CROME_2017_Complete.gpkg.zip"
        ),
    )

    exit_code = cli.main(["download-crome", "--year", "2017", "--dry-run"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset_id"] == "complete"
    assert payload["archive_url"].endswith(".gpkg.zip")


def test_cli_download_run_baseline_uses_env_output_root(monkeypatch, capsys) -> None:
    monkeypatch.setenv("CROME_DATA_ROOT", "/gws/ssde/j25a/nceo_isp/public/CROME")

    exit_code = cli.main(
        [
            "download-run-baseline",
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
    assert payload["download"]["output_root"] == "/gws/ssde/j25a/nceo_isp/public/CROME"
    assert payload["reference_download"]["output_root"] == "/gws/ssde/j25a/nceo_isp/public/CROME"
    assert payload["pipeline"]["label_mode"] == "centroid_to_pixel"
    assert payload["pipeline"]["output_root"] == "/gws/ssde/j25a/nceo_isp/public/CROME"


def test_cli_output_root_flag_overrides_env_default(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("CROME_DATA_ROOT", "/gws/ssde/j25a/nceo_isp/public/CROME")

    exit_code = cli.main(
        [
            "download-run-baseline",
            "--year",
            "2024",
            "--aoi-label",
            "east-anglia",
            "--bbox",
            "-1.0",
            "51.0",
            "0.0",
            "52.0",
            "--output-root",
            str(tmp_path / "override"),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["download"]["output_root"] == str(tmp_path / "override")
    assert payload["reference_download"]["output_root"] == str(tmp_path / "override")
    assert payload["pipeline"]["output_root"] == str(tmp_path / "override")


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
                "--label-mode",
                "polygon_to_pixel",
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


def test_cli_run_baseline_pipeline_from_manifest(tmp_path: Path, capsys) -> None:
    feature_dir = tmp_path / "raw"
    feature_dir.mkdir()
    first_raster = feature_dir / "alphaearth_a.tif"
    second_raster = feature_dir / "alphaearth_b.tif"
    reference_geojson = tmp_path / "crome.geojson"
    manifest_path = feature_dir / "manifests" / "run.json"
    output_root = tmp_path / "outputs"
    _write_feature_raster(first_raster)
    _write_feature_raster(second_raster)
    _write_reference_geojson(reference_geojson)
    _write_manifest(
        manifest_path,
        [
            ("IMAGE_A", first_raster),
            ("IMAGE_B", second_raster),
        ],
    )

    exit_code = cli.main(
        [
            "run-baseline-pipeline",
            "--manifest-path",
            str(manifest_path),
            "--reference-path",
            str(reference_geojson),
            "--year",
            "2024",
            "--aoi-label",
            "east-anglia",
            "--output-root",
            str(output_root),
            "--label-mode",
            "polygon_to_pixel",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["feature_count"] == 2
    assert payload["skipped_feature_count"] == 0
    assert Path(payload["pipeline_manifest_path"]).exists()


def test_cli_download_run_baseline(monkeypatch, tmp_path: Path, capsys) -> None:
    feature_dir = tmp_path / "raw"
    feature_dir.mkdir()
    feature_raster = feature_dir / "alphaearth_downloaded.tif"
    reference_geojson = tmp_path / "crome.geojson"
    manifest_path = feature_dir / "manifests" / "run.json"
    output_root = tmp_path / "outputs"
    _write_feature_raster(feature_raster)
    _write_reference_geojson(reference_geojson)
    _write_manifest(manifest_path, [("IMAGE_DOWNLOADED", feature_raster)])

    def fake_download(request):
        return SimpleNamespace(
            aoi_label=request.aoi_label,
            bands=request.bands,
            collection_id=request.collection_id,
            conditional_year=request.conditional_year,
            manifest_path=manifest_path,
            output_root=feature_dir,
            source_image_ids=("IMAGE_DOWNLOADED",),
            year=request.year,
        )

    monkeypatch.setattr(cli.workflow, "download_alphaearth_images", fake_download)

    exit_code = cli.main(
        [
            "download-run-baseline",
            "--year",
            "2024",
            "--aoi-label",
            "east-anglia",
            "--bbox",
            "-1.0",
            "51.0",
            "0.0",
            "52.0",
            "--reference-path",
            str(reference_geojson),
            "--output-root",
            str(output_root),
            "--label-mode",
            "polygon_to_pixel",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["download"]["manifest_path"] == str(manifest_path)
    assert payload["pipeline"]["feature_count"] == 1
    assert Path(payload["pipeline"]["pipeline_manifest_path"]).exists()


def test_cli_download_run_baseline_auto_downloads_crome(monkeypatch, tmp_path: Path, capsys) -> None:
    feature_dir = tmp_path / "raw"
    feature_dir.mkdir()
    feature_raster = feature_dir / "alphaearth_downloaded.tif"
    manifest_path = feature_dir / "manifests" / "run.json"
    extracted_reference = tmp_path / "raw" / "crome" / "CROME_2024_national" / "extracted" / "crome_2024.gpkg"
    output_root = tmp_path / "outputs"
    _write_feature_raster(feature_raster)
    extracted_reference.parent.mkdir(parents=True, exist_ok=True)
    _write_reference_gpkg(extracted_reference)
    _write_manifest(manifest_path, [("IMAGE_DOWNLOADED", feature_raster)])

    def fake_alphaearth_download(request):
        return SimpleNamespace(
            aoi_label=request.aoi_label,
            bands=request.bands,
            collection_id=request.collection_id,
            conditional_year=request.conditional_year,
            manifest_path=manifest_path,
            output_root=feature_dir,
            source_image_ids=("IMAGE_DOWNLOADED",),
            year=request.year,
        )

    def fake_crome_download(request):
        return SimpleNamespace(
            archive_path=tmp_path / "raw" / "crome" / "archive.zip",
            archive_url="https://example.test/Crop_Map_of_England_CROME_2024.gpkg.zip",
            dataset_id="2024",
            extracted_path=extracted_reference,
            landing_page_url="https://environment.data.gov.uk/dataset/2024",
            manifest_path=tmp_path / "raw" / "crome" / "manifest.json",
            normalized_path=None,
            output_root=extracted_reference.parent.parent,
            reference_path=extracted_reference,
            source_layer="Crop_Map_of_England_2024",
            title="Crop Map of England (CROME) 2024",
            variant=None,
            year=request.year,
        )

    monkeypatch.setattr(cli.workflow, "download_alphaearth_images", fake_alphaearth_download)
    monkeypatch.setattr(cli.workflow, "download_crome_reference", fake_crome_download)

    exit_code = cli.main(
        [
            "download-run-baseline",
            "--year",
            "2024",
            "--aoi-label",
            "east-anglia",
            "--bbox",
            "-1.0",
            "51.0",
            "0.0",
            "52.0",
            "--output-root",
            str(output_root),
            "--label-mode",
            "polygon_to_pixel",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reference_download"]["dataset_id"] == "2024"
    assert payload["reference_download"]["reference_path"] == str(extracted_reference)
    assert payload["pipeline"]["feature_count"] == 1


def test_python_module_cli_executes_main(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crome.cli",
            "download-run-baseline",
            "--year",
            "2024",
            "--aoi-label",
            "east-anglia",
            "--bbox",
            "-1.0",
            "51.0",
            "0.0",
            "52.0",
            "--reference-path",
            str(tmp_path / "crome.geojson"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)
    assert payload["download"]["aoi_label"] == "east-anglia"
    assert payload["pipeline"]["reference_path"] == str(tmp_path / "crome.geojson")
