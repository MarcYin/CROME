"""Path helpers for AlphaEarth outputs and reference targets."""

import os
from pathlib import Path
import re

_DEFAULT_OUTPUT_ROOT = "data/alphaearth"
OUTPUT_ROOT_ENV_VAR = "CROME_DATA_ROOT"

_SAFE_LABEL_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def default_output_root() -> str:
    """Return the user-scoped default artifact root.

    The package default stays ``data/alphaearth`` unless a user explicitly exports
    ``CROME_DATA_ROOT`` in their own environment.
    """

    value = os.environ.get(OUTPUT_ROOT_ENV_VAR)
    if value is None or not value.strip():
        return _DEFAULT_OUTPUT_ROOT
    return value.strip()


def sanitize_label(label: str | None, default: str = "aoi") -> str:
    """Convert a free-form label into a stable filesystem-safe token."""

    raw = (label or default).strip()
    if not raw:
        raw = default
    sanitized = _SAFE_LABEL_PATTERN.sub("-", raw).strip("-")
    return sanitized or default


def feature_artifact_name(feature_raster_path: Path | str) -> str:
    """Return a stable filesystem-safe name for one feature raster."""

    return sanitize_label(Path(feature_raster_path).stem, default="feature")


def alphaearth_run_name(aoi_label: str | None, year: int) -> str:
    """Return the stable run directory name for one AlphaEarth AOI/year run."""

    return f"AEF_{sanitize_label(aoi_label)}_annual_embedding_{year}"


def alphaearth_output_root(base_output_root: Path | str, aoi_label: str | None, year: int) -> Path:
    """Return the raw AlphaEarth output directory for one AOI/year run."""

    return Path(base_output_root) / "raw" / "alphaearth" / alphaearth_run_name(aoi_label, year)


def crome_run_name(year: int, variant_label: str | None = None) -> str:
    """Return the stable run directory name for one CROME reference acquisition."""

    suffix = sanitize_label(variant_label, default="national")
    return f"CROME_{year}_{suffix}"


def crome_download_root(
    base_output_root: Path | str,
    year: int,
    variant_label: str | None = None,
) -> Path:
    """Return the raw CROME download directory for one year/variant."""

    return Path(base_output_root) / "raw" / "crome" / crome_run_name(year, variant_label)


def crome_archive_path(
    base_output_root: Path | str,
    year: int,
    filename: str,
    variant_label: str | None = None,
) -> Path:
    """Return the archive path for one downloaded CROME file."""

    return crome_download_root(base_output_root, year, variant_label) / "archive" / filename


def crome_extract_root(
    base_output_root: Path | str,
    year: int,
    variant_label: str | None = None,
) -> Path:
    """Return the extraction directory for one downloaded CROME archive."""

    return crome_download_root(base_output_root, year, variant_label) / "extracted"


def crome_normalized_root(
    base_output_root: Path | str,
    year: int,
    variant_label: str | None = None,
) -> Path:
    """Return the normalized-vector directory for one downloaded CROME archive."""

    return crome_download_root(base_output_root, year, variant_label) / "normalized"


def cache_root(base_output_root: Path | str) -> Path:
    """Return the shared cache root for reusable pipeline artifacts."""

    return Path(base_output_root) / "cache"


def training_sample_cache_root(
    base_output_root: Path | str,
    year: int,
    *,
    cache_label: str | None = None,
) -> Path:
    """Return the reusable training-sample cache root for one year/cache namespace."""

    root = cache_root(base_output_root) / "training_samples" / str(year)
    if cache_label is None:
        return root
    return root / sanitize_label(cache_label, default="default")


def reference_run_name(
    aoi_label: str | None,
    year: int,
    reference_name: str = "crome_hex",
) -> str:
    """Return the stable reference-label run directory name."""

    return f"REF_{sanitize_label(reference_name, default='reference')}_{sanitize_label(aoi_label)}_{year}"


def reference_output_root(
    base_output_root: Path | str,
    aoi_label: str | None,
    year: int,
    reference_name: str = "crome_hex",
) -> Path:
    """Return the reference-label output directory for one AOI/year run."""

    return (
        Path(base_output_root)
        / "reference"
        / sanitize_label(reference_name, default="reference")
        / reference_run_name(aoi_label, year, reference_name=reference_name)
    )


def training_output_root(base_output_root: Path | str, aoi_label: str | None, year: int) -> Path:
    """Return the training-artifact output directory for one AOI/year run."""

    return Path(base_output_root) / "training" / f"TRAIN_{sanitize_label(aoi_label)}_{year}"


def prediction_output_root(base_output_root: Path | str, aoi_label: str | None, year: int) -> Path:
    """Return the prediction-artifact output directory for one AOI/year run."""

    return Path(base_output_root) / "prediction" / f"PRED_{sanitize_label(aoi_label)}_{year}"


def sample_cache_root(
    base_output_root: Path | str,
    year: int,
    *,
    cache_label: str | None = None,
    label_mode: str,
    reference_name: str = "crome_hex",
) -> Path:
    """Return the reusable sampled-row cache root for one year/label-mode/reference family."""

    root = (
        cache_root(base_output_root)
        / "samples"
        / sanitize_label(reference_name, default="reference")
        / f"year={year}"
        / f"label_mode={sanitize_label(label_mode, default='label-mode')}"
    )
    if cache_label is None:
        return root
    return root / sanitize_label(cache_label, default="default")
