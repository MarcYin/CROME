"""Small QC helpers shared by pipeline manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds
from shapely.geometry import shape

from .features import read_feature_raster_spec
from .runtime import ensure_proj_data_env


def reference_summary(reference_path: Path | str) -> dict[str, object]:
    """Return a compact JSON-safe summary for one vector reference source."""

    ensure_proj_data_env()
    path = Path(reference_path)
    layers = pyogrio.list_layers(path)
    layer_names = [str(item[0]) for item in layers]
    first_layer = layer_names[0] if layer_names else None
    info = pyogrio.read_info(path, layer=first_layer)
    total_bounds = info.get("total_bounds")
    bounds = (
        [float(value) for value in total_bounds]
        if hasattr(total_bounds, "__len__") and len(total_bounds) == 4
        else None
    )
    feature_count = info.get("features")
    return {
        "bounds": bounds,
        "crs": info.get("crs"),
        "driver": info.get("driver"),
        "feature_count": int(feature_count) if isinstance(feature_count, int) else None,
        "layer_count": len(layer_names),
        "layer_names": layer_names,
        "reference_path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def load_manifest_payload(manifest_path: Path | str | None) -> dict[str, object] | None:
    """Load one manifest payload when it exists."""

    if manifest_path is None:
        return None
    path = Path(manifest_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def requested_aoi_from_manifest(
    manifest_payload: dict[str, object] | None,
) -> dict[str, object] | None:
    """Return the requested AOI bounds and CRS when the manifest exposes them."""

    if not isinstance(manifest_payload, dict):
        return None

    search_payload = manifest_payload.get("search")
    if isinstance(search_payload, dict):
        bounds = search_payload.get("aoi_bounds")
        if isinstance(bounds, list) and len(bounds) == 4:
            return {
                "bounds": tuple(float(value) for value in bounds),
                "crs": str(search_payload.get("aoi_bounds_crs") or "EPSG:4326"),
                "source": "manifest.search.aoi_bounds",
            }

    config_payload = manifest_payload.get("config")
    if isinstance(config_payload, dict):
        aoi_payload = config_payload.get("aoi")
        if isinstance(aoi_payload, dict):
            geometry = aoi_payload.get("geometry")
            if isinstance(geometry, dict):
                geom = shape(geometry)
                minx, miny, maxx, maxy = geom.bounds
                return {
                    "bounds": (float(minx), float(miny), float(maxx), float(maxy)),
                    "crs": "EPSG:4326",
                    "source": "manifest.config.aoi.geometry",
                }
    return None


def requested_aoi_window(
    feature_raster_path: Path | str,
    requested_aoi: dict[str, object] | None,
) -> dict[str, object] | None:
    """Project the requested AOI onto one feature raster and return its pixel window."""

    if not isinstance(requested_aoi, dict):
        return None
    requested_bounds = requested_aoi.get("bounds")
    requested_crs = requested_aoi.get("crs")
    if not isinstance(requested_bounds, tuple) or len(requested_bounds) != 4:
        return None
    if not isinstance(requested_crs, str) or not requested_crs:
        return None

    feature_spec = read_feature_raster_spec(feature_raster_path)
    if feature_spec.crs is None:
        return None

    try:
        minx, miny, maxx, maxy = transform_bounds(
            requested_crs,
            feature_spec.crs,
            *requested_bounds,
            densify_pts=21,
        )
    except Exception:
        return None

    rows: list[int] = []
    cols: list[int] = []
    for x in (minx, maxx):
        for y in (miny, maxy):
            row_index, col_index = rowcol(feature_spec.transform, x, y)
            rows.append(int(row_index))
            cols.append(int(col_index))

    clipped_row_min = max(0, min(rows))
    clipped_row_max = min(feature_spec.height - 1, max(rows))
    clipped_col_min = max(0, min(cols))
    clipped_col_max = min(feature_spec.width - 1, max(cols))
    return {
        "aoi_bounds_in_feature_crs": [float(minx), float(miny), float(maxx), float(maxy)],
        "clipped_window": {
            "col_max": clipped_col_max,
            "col_min": clipped_col_min,
            "row_max": clipped_row_max,
            "row_min": clipped_row_min,
        },
        "raw_window": {
            "col_max": max(cols),
            "col_min": min(cols),
            "row_max": max(rows),
            "row_min": min(rows),
        },
    }


def write_qc_overlay_png(
    feature_raster_path: Path | str,
    label_raster_path: Path | str,
    output_path: Path | str,
    *,
    requested_window: dict[str, object] | None = None,
) -> Path:
    """Write a compact preview PNG showing labels over the feature raster."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_preview_dim = 512

    with rasterio.open(feature_raster_path) as feature_src, rasterio.open(label_raster_path) as label_src:
        scale = min(1.0, max_preview_dim / max(feature_src.height, feature_src.width))
        preview_height = max(1, int(round(feature_src.height * scale)))
        preview_width = max(1, int(round(feature_src.width * scale)))

        feature_band = feature_src.read(
            1,
            out_shape=(preview_height, preview_width),
            resampling=Resampling.bilinear,
            masked=False,
        ).astype("float32")
        valid_feature = np.isfinite(feature_band)
        if feature_src.nodata is not None and np.isfinite(feature_src.nodata):
            valid_feature &= feature_band != feature_src.nodata

        grayscale = np.zeros((preview_height, preview_width), dtype="uint8")
        if valid_feature.any():
            low = float(np.nanmin(feature_band[valid_feature]))
            high = float(np.nanmax(feature_band[valid_feature]))
            if high > low:
                grayscale[valid_feature] = np.clip(
                    ((feature_band[valid_feature] - low) / (high - low)) * 255.0,
                    0.0,
                    255.0,
                ).astype("uint8")
            else:
                grayscale[valid_feature] = 127

        rgb = np.stack([grayscale, grayscale, grayscale], axis=0)
        labels = label_src.read(
            1,
            out_shape=(preview_height, preview_width),
            resampling=Resampling.nearest,
            masked=False,
        )
        nodata_label = int(label_src.nodata) if label_src.nodata is not None else -1
        valid_labels = labels != nodata_label
        rgb[0, valid_labels] = 255
        rgb[1, valid_labels] = 96
        rgb[2, valid_labels] = 96

        clipped_window = requested_window.get("clipped_window") if isinstance(requested_window, dict) else None
        if isinstance(clipped_window, dict):
            row_scale = preview_height / feature_src.height
            col_scale = preview_width / feature_src.width
            row_min = max(0, min(preview_height - 1, int(np.floor(clipped_window["row_min"] * row_scale))))
            row_max = max(
                0,
                min(
                    preview_height - 1,
                    int(np.ceil((clipped_window["row_max"] + 1) * row_scale) - 1),
                ),
            )
            col_min = max(0, min(preview_width - 1, int(np.floor(clipped_window["col_min"] * col_scale))))
            col_max = max(
                0,
                min(
                    preview_width - 1,
                    int(np.ceil((clipped_window["col_max"] + 1) * col_scale) - 1),
                ),
            )
            rgb[:, row_min, col_min : col_max + 1] = np.array([[0], [255], [0]], dtype="uint8")
            rgb[:, row_max, col_min : col_max + 1] = np.array([[0], [255], [0]], dtype="uint8")
            rgb[:, row_min : row_max + 1, col_min] = np.array([[0], [255], [0]], dtype="uint8")
            rgb[:, row_min : row_max + 1, col_max] = np.array([[0], [255], [0]], dtype="uint8")

    with rasterio.open(
        output_path,
        "w",
        driver="PNG",
        height=preview_height,
        width=preview_width,
        count=3,
        dtype="uint8",
    ) as dst:
        dst.write(rgb)

    return output_path
