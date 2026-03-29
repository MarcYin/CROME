"""Rasterize CROME vector references onto the AlphaEarth grid."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

from .config import AlphaEarthDownloadRequest, AlphaEarthTrainingSpec, CromeReferenceConfig
from .features import read_feature_raster_spec
from .reference import validate_reference_columns


@dataclass(frozen=True, slots=True)
class RasterizedReferenceResult:
    """Outputs from rasterizing CROME references onto the feature grid."""

    label_mapping_path: Path
    label_raster_path: Path
    label_values: tuple[str, ...]
    nodata_label: int


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
    gdf = gpd.read_file(spec.reference.source_path)
    validate_reference_columns(gdf.columns, spec.reference.label_column, spec.reference.geometry_column)

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
        raise ValueError("No valid reference geometries remained after filtering.")

    raster_bounds = box(*feature_spec.bounds)
    gdf = gdf[gdf.geometry.intersects(raster_bounds)].copy()
    if gdf.empty:
        raise ValueError("Reference geometries do not intersect the feature raster bounds.")

    if spec.overlap_policy == "error" and _has_positive_area_overlaps(gdf):
        raise ValueError("Reference geometries overlap with positive area; overlap_policy='error'.")

    return gdf


def _label_mapping(gdf: gpd.GeoDataFrame, label_column: str) -> tuple[dict[str, int], tuple[str, ...]]:
    labels = tuple(sorted({str(value) for value in gdf[label_column]}))
    return {label: idx for idx, label in enumerate(labels)}, labels


def rasterize_crome_reference(
    feature_raster_path: Path | str,
    spec: AlphaEarthTrainingSpec,
) -> RasterizedReferenceResult:
    """Rasterize CROME vector labels onto the AlphaEarth raster grid."""

    feature_spec = read_feature_raster_spec(feature_raster_path)
    gdf = _load_reference_geometries(feature_raster_path, spec)
    label_to_id, label_values = _label_mapping(gdf, spec.reference.label_column)

    output_dir = spec.reference_output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    label_raster_path = output_dir / "labels.tif"
    label_mapping_path = output_dir / "labels.json"

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
    parser.add_argument("--output-root", default="data/alphaearth", help="Base output directory.")
    parser.add_argument("--label-column", default="lucode", help="Reference class column.")
    parser.add_argument("--geometry-column", default="geometry", help="Reference geometry column.")
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
