"""Feature-raster helpers for AlphaEarth workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rasterio
from affine import Affine
from rasterio.coords import BoundingBox

from .bands import ALPHAEARTH_BANDS, validate_alphaearth_bands


@dataclass(frozen=True, slots=True)
class FeatureRasterSpec:
    """Metadata for one AlphaEarth feature raster."""

    path: Path
    width: int
    height: int
    count: int
    crs: str | None
    transform: Affine
    dtype: str
    band_names: tuple[str, ...]
    bounds: BoundingBox
    nodata: float | int | None


def _resolve_band_names(src: rasterio.io.DatasetReader) -> tuple[str, ...]:
    descriptions = tuple(src.descriptions or ())
    cleaned = tuple(desc for desc in descriptions if desc)
    if len(cleaned) == src.count:
        return validate_alphaearth_bands(cleaned)

    if src.count == len(ALPHAEARTH_BANDS):
        return ALPHAEARTH_BANDS

    raise ValueError(
        "Feature raster must expose the canonical AlphaEarth band descriptions "
        "A00-A63 or contain exactly 64 bands."
    )


def read_feature_raster_spec(feature_raster_path: Path | str) -> FeatureRasterSpec:
    """Read and validate AlphaEarth feature-raster metadata."""

    path = Path(feature_raster_path)
    with rasterio.open(path) as src:
        band_names = _resolve_band_names(src)
        return FeatureRasterSpec(
            path=path,
            width=src.width,
            height=src.height,
            count=src.count,
            crs=str(src.crs) if src.crs is not None else None,
            transform=src.transform,
            dtype=src.dtypes[0],
            band_names=band_names,
            bounds=src.bounds,
            nodata=src.nodata,
        )
