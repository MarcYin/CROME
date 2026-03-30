"""Rasterize CROME vector references onto the AlphaEarth grid."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds
from shapely.geometry import box

from .config import AlphaEarthDownloadRequest, AlphaEarthTrainingSpec, CromeReferenceConfig
from .features import read_feature_raster_spec
from .paths import OUTPUT_ROOT_ENV_VAR, default_output_root
from .runtime import ensure_proj_data_env


@dataclass(frozen=True, slots=True)
class RasterizedReferenceResult:
    """Outputs from rasterizing CROME references onto the feature grid."""

    label_mapping_path: Path
    label_raster_path: Path
    label_values: tuple[str, ...]
    nodata_label: int


class NoReferenceCoverageError(ValueError):
    """Raised when a feature raster has no usable CROME coverage."""


def _reference_layer_name(reference_path: Path | str) -> str | None:
    ensure_proj_data_env()
    layers = pyogrio.list_layers(reference_path)
    if len(layers) == 0:
        return None
    first = layers[0]
    return str(first[0])


def _reference_layer_names(reference_path: Path | str) -> tuple[str, ...]:
    ensure_proj_data_env()
    return tuple(str(item[0]) for item in pyogrio.list_layers(reference_path))


def _reference_info(reference_path: Path | str, *, layer: str | None = None) -> dict[str, object]:
    ensure_proj_data_env()
    return pyogrio.read_info(reference_path, layer=layer)


def _reference_bbox_in_source_crs(
    reference_path: Path | str,
    feature_bounds: tuple[float, float, float, float],
    feature_crs: str | None,
) -> tuple[float, float, float, float] | None:
    if feature_crs is None:
        return None

    info = _reference_info(reference_path, layer=_reference_layer_name(reference_path))
    source_crs = info.get("crs")
    if not isinstance(source_crs, str) or not source_crs:
        return None

    try:
        return transform_bounds(feature_crs, source_crs, *feature_bounds, densify_pts=21)
    except Exception:
        return None


def _read_reference_geometries(
    reference_path: Path | str,
    *,
    label_column: str,
    geometry_column: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    ensure_proj_data_env()
    frames: list[gpd.GeoDataFrame] = []

    for layer in _reference_layer_names(reference_path):
        if bbox is not None:
            info = _reference_info(reference_path, layer=layer)
            total_bounds = info.get("total_bounds")
            if hasattr(total_bounds, "__len__") and len(total_bounds) == 4:
                if (
                    total_bounds[2] < bbox[0]
                    or total_bounds[0] > bbox[2]
                    or total_bounds[3] < bbox[1]
                    or total_bounds[1] > bbox[3]
                ):
                    continue

        read_kwargs: dict[str, object] = {"layer": layer, "columns": [label_column]}
        if bbox is not None:
            read_kwargs["bbox"] = bbox

        gdf = pyogrio.read_dataframe(reference_path, **read_kwargs)
        if gdf.empty:
            continue
        if label_column not in gdf.columns:
            raise ValueError(f"Reference data is missing required column: {label_column}")
        if gdf.geometry is None:
            raise ValueError("Reference data is missing geometry.")

        source_geometry_column = (
            geometry_column if geometry_column in gdf.columns else gdf.geometry.name
        )
        if source_geometry_column is None:
            raise ValueError("Reference data is missing geometry.")

        frames.append(
            gdf[[label_column, source_geometry_column]].rename(
                columns={source_geometry_column: "geometry"}
            )
        )

    if not frames:
        info = _reference_info(reference_path, layer=_reference_layer_name(reference_path))
        source_crs = info.get("crs")
        return gpd.GeoDataFrame({label_column: []}, geometry=[], crs=source_crs)

    concatenated = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(concatenated, geometry="geometry", crs=frames[0].crs)


def _quote_sqlite_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _load_gpkg_distinct_labels(
    reference_path: Path | str,
    label_column: str,
) -> tuple[str, ...]:
    layers = _reference_layer_names(reference_path)
    if not layers:
        raise ValueError(f"Could not determine a layer name for {reference_path}.")
    statements = [
        (
            f"SELECT DISTINCT {_quote_sqlite_identifier(label_column)} AS label "
            f"FROM {_quote_sqlite_identifier(layer)} "
            f"WHERE {_quote_sqlite_identifier(label_column)} IS NOT NULL"
        )
        for layer in layers
    ]
    query = " UNION ".join(statements) + " ORDER BY label"
    with sqlite3.connect(reference_path) as connection:
        rows = connection.execute(query).fetchall()
    labels = tuple(str(row[0]) for row in rows if row[0] is not None)
    if not labels:
        raise ValueError("Reference source does not expose any non-null labels.")
    return labels


def _load_distinct_labels_from_vector(
    reference_path: Path | str,
    label_column: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> tuple[str, ...]:
    ensure_proj_data_env()
    labels: set[str] = set()
    for layer in _reference_layer_names(reference_path):
        if bbox is not None:
            info = _reference_info(reference_path, layer=layer)
            total_bounds = info.get("total_bounds")
            if hasattr(total_bounds, "__len__") and len(total_bounds) == 4:
                if (
                    total_bounds[2] < bbox[0]
                    or total_bounds[0] > bbox[2]
                    or total_bounds[3] < bbox[1]
                    or total_bounds[1] > bbox[3]
                ):
                    continue
        table = pyogrio.read_dataframe(
            reference_path,
            layer=layer,
            columns=[label_column],
            read_geometry=False,
            bbox=bbox,
        )
        if label_column not in table.columns:
            raise ValueError(f"Reference data is missing required column: {label_column}")
        labels.update(str(value) for value in table[label_column] if value is not None)
    labels = tuple(sorted(labels))
    if not labels:
        raise ValueError("Reference source does not expose any non-null labels.")
    return labels


def _has_positive_area_overlaps(gdf: gpd.GeoDataFrame) -> bool:
    if gdf.empty or len(gdf) == 1:
        return False

    spatial_index = gdf.sindex
    for idx, geometry in enumerate(gdf.geometry):
        if geometry is None or geometry.is_empty:
            continue
        for candidate in spatial_index.intersection(geometry.bounds):
            if candidate <= idx:
                continue
            other = gdf.geometry.iloc[candidate]
            if other is None or other.is_empty:
                continue
            if not geometry.intersects(other):
                continue
            intersection = geometry.intersection(other)
            if not intersection.is_empty and intersection.area > 0:
                return True
    return False


def _load_reference_geometries(
    feature_raster_path: Path | str,
    spec: AlphaEarthTrainingSpec,
) -> gpd.GeoDataFrame:
    feature_spec = read_feature_raster_spec(feature_raster_path)
    read_bbox = _reference_bbox_in_source_crs(
        spec.reference.source_path,
        feature_spec.bounds,
        feature_spec.crs,
    )
    gdf = _read_reference_geometries(
        spec.reference.source_path,
        label_column=spec.reference.label_column,
        geometry_column=spec.reference.geometry_column,
        bbox=read_bbox,
    )

    if gdf.crs is None:
        if spec.reference.target_crs is None:
            raise ValueError("Reference data has no CRS and no target_crs override was provided.")
        gdf = gdf.set_crs(spec.reference.target_crs)

    if feature_spec.crs is None:
        raise ValueError("Feature raster has no CRS; cannot align vector references.")

    gdf = gdf.to_crs(feature_spec.crs)
    gdf = gdf[[spec.reference.label_column, spec.reference.geometry_column]].rename(
        columns={spec.reference.geometry_column: "geometry"}
    )
    gdf = gdf[gdf[spec.reference.label_column].notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    if gdf.empty:
        raise NoReferenceCoverageError("No valid reference geometries remained after filtering.")

    raster_bounds = box(*feature_spec.bounds)
    gdf = gdf[gdf.geometry.intersects(raster_bounds)].copy()
    if gdf.empty:
        raise NoReferenceCoverageError("Reference geometries do not intersect the feature raster bounds.")

    if spec.label_mode == "polygon_to_pixel" and spec.overlap_policy == "error" and _has_positive_area_overlaps(gdf):
        raise ValueError("Reference geometries overlap with positive area; overlap_policy='error'.")

    return gdf


def _label_mapping(gdf: gpd.GeoDataFrame, label_column: str) -> tuple[dict[str, int], tuple[str, ...]]:
    labels = tuple(sorted({str(value) for value in gdf[label_column]}))
    return {label: idx for idx, label in enumerate(labels)}, labels


def load_reference_label_mapping(
    reference_path: Path | str,
    label_column: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Build one stable label mapping from the full CROME reference source."""

    reference_path = Path(reference_path)
    if reference_path.suffix.casefold() == ".gpkg" and bbox is None:
        labels = _load_gpkg_distinct_labels(reference_path, label_column)
    else:
        labels = _load_distinct_labels_from_vector(reference_path, label_column, bbox=bbox)
    return {label: idx for idx, label in enumerate(labels)}, labels


def reference_source_bbox_for_feature_rasters(
    reference_path: Path | str,
    feature_raster_paths: list[Path | str],
) -> tuple[float, float, float, float] | None:
    """Return one union bbox in the reference CRS covering all feature rasters."""

    reference_bboxes: list[tuple[float, float, float, float]] = []
    for feature_raster_path in feature_raster_paths:
        feature_spec = read_feature_raster_spec(feature_raster_path)
        bbox = _reference_bbox_in_source_crs(
            reference_path,
            feature_spec.bounds,
            feature_spec.crs,
        )
        if bbox is not None:
            reference_bboxes.append(bbox)

    if not reference_bboxes:
        return None

    minx = min(bbox[0] for bbox in reference_bboxes)
    miny = min(bbox[1] for bbox in reference_bboxes)
    maxx = max(bbox[2] for bbox in reference_bboxes)
    maxy = max(bbox[3] for bbox in reference_bboxes)
    return (minx, miny, maxx, maxy)


def _centroid_label_array(
    gdf: gpd.GeoDataFrame,
    feature_spec: object,
    spec: AlphaEarthTrainingSpec,
    label_to_id: dict[str, int],
) -> np.ndarray:
    label_array = np.full((feature_spec.height, feature_spec.width), spec.nodata_label, dtype="int32")

    for row in gdf.itertuples(index=False):
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        point = geometry.centroid
        if not geometry.covers(point):
            point = geometry.representative_point()
        row_index, col_index = rowcol(feature_spec.transform, point.x, point.y)
        if (
            row_index < 0
            or row_index >= feature_spec.height
            or col_index < 0
            or col_index >= feature_spec.width
        ):
            continue

        label_id = label_to_id[str(getattr(row, spec.reference.label_column))]
        existing = label_array[row_index, col_index]
        if existing != spec.nodata_label:
            if spec.overlap_policy == "error":
                raise ValueError(
                    "Multiple reference centroids resolved to the same AlphaEarth pixel; "
                    "set overlap_policy to first or last to continue."
                )
            if spec.overlap_policy == "first":
                continue
        label_array[row_index, col_index] = label_id

    return label_array


def rasterize_crome_reference(
    feature_raster_path: Path | str,
    spec: AlphaEarthTrainingSpec,
    *,
    label_to_id: dict[str, int] | None = None,
    output_dir: Path | str | None = None,
) -> RasterizedReferenceResult:
    """Rasterize CROME vector labels onto the AlphaEarth raster grid."""

    feature_spec = read_feature_raster_spec(feature_raster_path)
    gdf = _load_reference_geometries(feature_raster_path, spec)
    if label_to_id is None:
        label_to_id, label_values = _label_mapping(gdf, spec.reference.label_column)
    else:
        local_labels = {str(value) for value in gdf[spec.reference.label_column]}
        missing = sorted(local_labels - set(label_to_id))
        if missing:
            raise ValueError(f"Provided label mapping is missing labels: {missing}")
        label_values = tuple(label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1]))

    output_dir = Path(output_dir) if output_dir is not None else spec.reference_output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    label_raster_path = output_dir / "labels.tif"
    label_mapping_path = output_dir / "labels.json"
    if spec.label_mode == "centroid_to_pixel":
        label_array = _centroid_label_array(gdf, feature_spec, spec, label_to_id)
    else:
        rows = gdf.itertuples(index=False)
        shapes = [
            (row.geometry, label_to_id[str(getattr(row, spec.reference.label_column))])
            for row in rows
        ]
        if spec.overlap_policy == "first":
            shapes = list(reversed(shapes))

        label_array = rasterize(
            shapes=shapes,
            out_shape=(feature_spec.height, feature_spec.width),
            transform=feature_spec.transform,
            fill=spec.nodata_label,
            all_touched=spec.reference.all_touched,
            dtype="int32",
        )
    if not np.any(label_array != spec.nodata_label):
        raise NoReferenceCoverageError("Reference geometries rasterized to no labeled pixels.")

    profile = {
        "driver": "GTiff",
        "height": feature_spec.height,
        "width": feature_spec.width,
        "count": 1,
        "dtype": "int32",
        "crs": feature_spec.crs,
        "transform": feature_spec.transform,
        "nodata": spec.nodata_label,
        "compress": "deflate",
    }

    with rasterio.open(label_raster_path, "w", **profile) as dst:
        dst.write(label_array.astype("int32"), 1)
        dst.update_tags(
            aoi_label=spec.alphaearth.aoi_label or "",
            label_column=spec.reference.label_column,
            label_mode=spec.label_mode,
            overlap_policy=spec.overlap_policy,
            reference_path=str(spec.reference.source_path),
            year=str(spec.reference.year),
        )

    mapping_payload = {
        "aoi_label": spec.alphaearth.aoi_label,
        "geometry_column": spec.reference.geometry_column,
        "id_to_label": {str(idx): label for idx, label in enumerate(label_values)},
        "label_column": spec.reference.label_column,
        "label_mode": spec.label_mode,
        "label_to_id": label_to_id,
        "nodata_label": spec.nodata_label,
        "reference_path": str(spec.reference.source_path),
        "source_feature_raster": str(feature_raster_path),
        "year": spec.reference.year,
    }
    label_mapping_path.write_text(json.dumps(mapping_payload, indent=2, sort_keys=True), encoding="utf-8")

    return RasterizedReferenceResult(
        label_mapping_path=label_mapping_path,
        label_raster_path=label_raster_path,
        label_values=label_values,
        nodata_label=spec.nodata_label,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rasterize CROME references onto an AlphaEarth grid.")
    parser.add_argument("--feature-raster", required=True, help="Path to one AlphaEarth GeoTIFF.")
    parser.add_argument("--reference-path", required=True, help="Path to the CROME vector reference file.")
    parser.add_argument("--year", required=True, type=int, help="Reference year.")
    parser.add_argument("--aoi-label", default=None, help="AOI label used for output naming.")
    parser.add_argument(
        "--output-root",
        default=default_output_root(),
        help=(
            f"Base output directory. Defaults to ${OUTPUT_ROOT_ENV_VAR} when set, "
            "otherwise data/alphaearth."
        ),
    )
    parser.add_argument("--label-column", default="lucode", help="Reference class column.")
    parser.add_argument("--geometry-column", default="geometry", help="Reference geometry column.")
    parser.add_argument(
        "--label-mode",
        choices=("centroid_to_pixel", "polygon_to_pixel"),
        default="centroid_to_pixel",
        help="How CROME vector labels are transferred onto the AlphaEarth grid.",
    )
    parser.add_argument(
        "--overlap-policy",
        choices=("error", "first", "last"),
        default="error",
        help="Policy for overlapping reference polygons.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Rasterize with all_touched=True instead of pixel-center semantics.",
    )
    parser.add_argument(
        "--nodata-label",
        type=int,
        default=-1,
        help="Output nodata label id.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    feature_spec = read_feature_raster_spec(args.feature_raster)
    aoi_label = args.aoi_label or Path(args.feature_raster).stem

    alphaearth = AlphaEarthDownloadRequest(
        year=args.year,
        output_root=args.output_root,
        aoi_label=aoi_label,
        bbox=(0.0, 0.0, 1.0, 1.0),
        bands=feature_spec.band_names,
    )

    reference = CromeReferenceConfig(
        source_path=args.reference_path,
        year=args.year,
        aoi_label=alphaearth.aoi_label,
        label_column=args.label_column,
        geometry_column=args.geometry_column,
        all_touched=args.all_touched,
    )
    spec = AlphaEarthTrainingSpec(
        alphaearth=alphaearth,
        reference=reference,
        label_mode=args.label_mode,
        overlap_policy=args.overlap_policy,
        nodata_label=args.nodata_label,
    )
    result = rasterize_crome_reference(args.feature_raster, spec)
    print(
        json.dumps(
            {
                "label_mapping_path": str(result.label_mapping_path),
                "label_raster_path": str(result.label_raster_path),
                "label_values": list(result.label_values),
                "nodata_label": result.nodata_label,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
