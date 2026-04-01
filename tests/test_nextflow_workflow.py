from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
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


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow is not installed")
def test_nextflow_local_smoke_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
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
    assert (
        cli.main(
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
                "nextflow-smoke",
                "--n-estimators",
                "10",
                "--n-jobs",
                "1",
                "--label-mode",
                "polygon_to_pixel",
            ]
        )
        == 0
    )
    prepare_payload = json.loads(capsys.readouterr().out)
    batch_manifest_path = Path(prepare_payload["batch_manifest_path"])
    assert batch_manifest_path.exists()

    env = os.environ.copy()
    env["NXF_HOME"] = str(tmp_path / ".nxf")

    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(repo_root / "nextflow" / "main.nf"),
            "-c",
            str(repo_root / "nextflow" / "nextflow.config"),
            "-profile",
            "local",
            "-ansi-log",
            "false",
            "--batch_manifest",
            str(batch_manifest_path),
            "--output_root",
            str(output_root),
            "--work_dir",
            str(tmp_path / "nextflow-work"),
            "--n-estimators",
            "10",
            "--tile_cpus",
            "1",
            "--tile_memory",
            "2 GB",
            "--pooled_cpus",
            "1",
            "--pooled_memory",
            "2 GB",
            "--tile_n_jobs",
            "1",
            "--pooled_n_jobs",
            "1",
        ],
        capture_output=True,
        check=False,
        cwd=repo_root,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    report_root = output_root / "nextflow" / "reports"
    assert (report_root / "trace.txt").exists()
    assert (report_root / "timeline.html").exists()
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    assert len(batch_manifest["tile_manifest_paths"]) == 2
    assert batch_manifest["reference_path"].endswith(".fgb")

    pipeline_manifest_paths = sorted(output_root.rglob("pipeline.json"))
    assert len(pipeline_manifest_paths) == 2

    pooled_manifest_paths = list(output_root.rglob("pooled_training.json"))
    assert len(pooled_manifest_paths) == 1
    pooled_manifest = json.loads(pooled_manifest_paths[0].read_text(encoding="utf-8"))
    assert Path(pooled_manifest["training_table_path"]).exists()
    assert Path(pooled_manifest["metrics_path"]).exists()
