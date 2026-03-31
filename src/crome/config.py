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
    ALPHAEARTH_TARGET_RESOLUTION_M,
    CROME_DEFAULT_GEOMETRY_COLUMN,
    CROME_DEFAULT_LABEL_COLUMN,
)
from .paths import (
    alphaearth_output_root,
    crome_download_root,
    prediction_output_root,
    reference_output_root,
    sanitize_label,
    sample_cache_root,
    training_output_root,
)

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
    aoi_label: str | None = None
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

        default_aoi = Path(self.geojson).stem if self.geojson is not None else "bbox"
        object.__setattr__(self, "aoi_label", sanitize_label(self.aoi_label or default_aoi))

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
        return alphaearth_output_root(self.output_root, self.aoi_label, self.year)

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
            "aoi_label": self.aoi_label,
            "year": self.year,
        }


@dataclass(frozen=True, slots=True)
class CromeReferenceConfig:
    """Pure configuration for vector CROME reference labels."""

    source_path: Path | str
    year: int
    aoi_label: str | None = None
    label_column: str = CROME_DEFAULT_LABEL_COLUMN
    geometry_column: str = CROME_DEFAULT_GEOMETRY_COLUMN
    target_resolution_m: float = ALPHAEARTH_TARGET_RESOLUTION_M
    target_crs: str | None = None
    reference_name: str = "crome_hex"
    all_touched: bool = False

    def __post_init__(self) -> None:
        if self.year < ALPHAEARTH_FIRST_YEAR or self.year > ALPHAEARTH_LAST_ALLOWED_YEAR:
            raise ValueError(
                f"CROME reference year must be between {ALPHAEARTH_FIRST_YEAR} and "
                f"{ALPHAEARTH_LAST_ALLOWED_YEAR}."
            )
        if not self.label_column:
            raise ValueError("label_column must be a non-empty string.")
        if not self.geometry_column:
            raise ValueError("geometry_column must be a non-empty string.")
        if self.target_resolution_m <= 0:
            raise ValueError("target_resolution_m must be positive.")

        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "aoi_label", sanitize_label(self.aoi_label or self.source_path.stem))
        object.__setattr__(self, "reference_name", sanitize_label(self.reference_name, default="crome_hex"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_touched": self.all_touched,
            "aoi_label": self.aoi_label,
            "geometry_column": self.geometry_column,
            "label_column": self.label_column,
            "reference_name": self.reference_name,
            "source_path": str(self.source_path),
            "target_crs": self.target_crs,
            "target_resolution_m": self.target_resolution_m,
            "year": self.year,
        }


@dataclass(frozen=True, slots=True)
class CromeDownloadRequest:
    """Pure request object for one CROME reference acquisition."""

    year: int
    output_root: Path | str
    prefer_complete: bool = True
    extract: bool = True
    force: bool = False
    query: str | None = None
    search_base_url: str = "https://environment.data.gov.uk/searchresults"
    timeout_s: float = 30.0
    pagesize: int = 50

    def __post_init__(self) -> None:
        if self.year < ALPHAEARTH_FIRST_YEAR or self.year > ALPHAEARTH_LAST_ALLOWED_YEAR:
            raise ValueError(
                f"CROME download year must be between {ALPHAEARTH_FIRST_YEAR} and "
                f"{ALPHAEARTH_LAST_ALLOWED_YEAR}."
            )
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive.")
        if self.pagesize <= 0:
            raise ValueError("pagesize must be positive.")

        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(
            self,
            "query",
            (self.query or f"Crop Map of England (CROME) {self.year}").strip(),
        )

    @property
    def dataset_output_root(self) -> Path:
        variant_label = "complete" if self.prefer_complete else "national"
        return crome_download_root(self.output_root, self.year, variant_label=variant_label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "extract": self.extract,
            "force": self.force,
            "output_root": str(self.output_root),
            "pagesize": self.pagesize,
            "prefer_complete": self.prefer_complete,
            "query": self.query,
            "search_base_url": self.search_base_url,
            "timeout_s": self.timeout_s,
            "year": self.year,
        }


@dataclass(frozen=True, slots=True)
class AlphaEarthTrainingSpec:
    """Pure contract for AlphaEarth features plus CROME vector references."""

    alphaearth: AlphaEarthDownloadRequest
    reference: CromeReferenceConfig
    label_mode: str = "centroid_to_pixel"
    overlap_policy: str = "error"
    nodata_label: int = -1

    def __post_init__(self) -> None:
        if self.alphaearth.year != self.reference.year:
            raise ValueError("AlphaEarth feature year and CROME reference year must match.")
        if self.alphaearth.aoi_label != self.reference.aoi_label:
            raise ValueError("AlphaEarth AOI label and CROME reference AOI label must match.")
        if self.label_mode not in {"polygon_to_pixel", "centroid_to_pixel"}:
            raise ValueError("label_mode must be one of: polygon_to_pixel, centroid_to_pixel.")
        if self.label_mode == "centroid_to_pixel" and self.reference.all_touched:
            raise ValueError("all_touched is only supported with polygon_to_pixel label mode.")
        if self.overlap_policy not in {"error", "first", "last"}:
            raise ValueError("overlap_policy must be one of: error, first, last.")

    @property
    def reference_output_root(self) -> Path:
        return reference_output_root(
            self.alphaearth.output_root,
            self.alphaearth.aoi_label,
            self.alphaearth.year,
            reference_name=self.reference.reference_name,
        )

    @property
    def training_output_root(self) -> Path:
        return training_output_root(
            self.alphaearth.output_root,
            self.alphaearth.aoi_label,
            self.alphaearth.year,
        )

    @property
    def prediction_output_root(self) -> Path:
        return prediction_output_root(
            self.alphaearth.output_root,
            self.alphaearth.aoi_label,
            self.alphaearth.year,
        )

    @property
    def sample_cache_root(self) -> Path:
        cache_label = "_".join(
            [
                self.reference.reference_name,
                self.reference.label_column,
                self.reference.geometry_column,
                self.label_mode,
                self.overlap_policy,
                "all_touched" if self.reference.all_touched else "pixel_center",
            ]
        )
        return sample_cache_root(
            self.alphaearth.output_root,
            self.alphaearth.year,
            cache_label=cache_label,
            label_mode=self.label_mode,
            reference_name=self.reference.reference_name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alphaearth": self.alphaearth.to_dict(),
            "label_mode": self.label_mode,
            "nodata_label": self.nodata_label,
            "overlap_policy": self.overlap_policy,
            "prediction_output_root": str(self.prediction_output_root),
            "reference": self.reference.to_dict(),
            "reference_output_root": str(self.reference_output_root),
            "sample_cache_root": str(self.sample_cache_root),
            "training_output_root": str(self.training_output_root),
        }
