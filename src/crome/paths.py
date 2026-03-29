"""Path helpers for AlphaEarth outputs and reference targets."""

from pathlib import Path
import re


_SAFE_LABEL_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


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
