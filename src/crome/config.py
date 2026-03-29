"""Pure configuration objects for the first AlphaEarth migration slice."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bands import ALPHAEARTH_BANDS, validate_alphaearth_bands
from .constants import (
    ALPHAEARTH_COLLECTION_ID,
    ALPHAEARTH_FIRST_YEAR,
    ALPHAEARTH_LAST_ALLOWED_YEAR,
    ALPHAEARTH_LAST_STABLE_YEAR,
)
from .paths import alphaearth_output_root, sanitize_tile_label

BBox = tuple[float, float, float, float]


def _validate_bbox(bbox: BBox) -> BBox:
    minx, miny, maxx, maxy = bbox
    if minx >= maxx or miny >= maxy:
        raise ValueError("Bounding boxes must be ordered as minx miny maxx maxy.")
    return bbox


@dataclass(frozen=True, slots=True)
class AlphaEarthDownloadRequest:
    """Pure request object for one AlphaEarth download run."""

    year: int
    output_root: Path | str
    tile_id: str | None = None
    bbox: BBox | None = None
    geojson: Path | str | None = None
    collection_id: str = ALPHAEARTH_COLLECTION_ID
    bands: tuple[str, ...] = ALPHAEARTH_BANDS

    def __post_init__(self) -> None:
        if self.year < ALPHAEARTH_FIRST_YEAR or self.year > ALPHAEARTH_LAST_ALLOWED_YEAR:
            raise ValueError(
                f"AlphaEarth year must be between {ALPHAEARTH_FIRST_YEAR} and "
                f"{ALPHAEARTH_LAST_ALLOWED_YEAR}."
            )

        if (self.bbox is None) == (self.geojson is None):
            raise ValueError("Provide exactly one of bbox or geojson.")

        object.__setattr__(self, "output_root", Path(self.output_root))
        if self.geojson is not None:
            object.__setattr__(self, "geojson", Path(self.geojson))

        if self.bbox is not None:
            object.__setattr__(self, "bbox", _validate_bbox(self.bbox))

        object.__setattr__(self, "bands", validate_alphaearth_bands(self.bands))

        default_tile = Path(self.geojson).stem if self.geojson is not None else "bbox"
        object.__setattr__(self, "tile_id", sanitize_tile_label(self.tile_id or default_tile))

    @property
    def start_date(self) -> str:
        return f"{self.year}-01-01"

    @property
    def end_date(self) -> str:
        return f"{self.year + 1}-01-01"

    @property
    def conditional_year(self) -> bool:
        return self.year > ALPHAEARTH_LAST_STABLE_YEAR

    @property
    def dataset_output_root(self) -> Path:
        return alphaearth_output_root(self.output_root, self.tile_id, self.year)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for dry-run output."""

        return {
            "bands": list(self.bands),
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "collection_id": self.collection_id,
            "conditional_year": self.conditional_year,
            "dataset_output_root": str(self.dataset_output_root),
            "end_date": self.end_date,
            "geojson": str(self.geojson) if self.geojson is not None else None,
            "output_root": str(self.output_root),
            "start_date": self.start_date,
            "tile_id": self.tile_id,
            "year": self.year,
        }
