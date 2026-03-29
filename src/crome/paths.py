"""Path helpers for AlphaEarth outputs."""

from pathlib import Path
import re


_SAFE_LABEL_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_tile_label(tile_id: str | None) -> str:
    """Convert a free-form AOI label into a stable filesystem-safe token."""

    raw = (tile_id or "aoi").strip()
    if not raw:
        raw = "aoi"
    sanitized = _SAFE_LABEL_PATTERN.sub("-", raw).strip("-")
    return sanitized or "aoi"


def alphaearth_run_name(tile_id: str | None, year: int) -> str:
    """Return the stable run directory name for one AlphaEarth tile/year."""

    return f"AEF_{sanitize_tile_label(tile_id)}_annual_embedding_{year}"


def alphaearth_output_root(base_output_root: Path | str, tile_id: str | None, year: int) -> Path:
    """Return the per-run output directory under the chosen base root."""

    return Path(base_output_root) / alphaearth_run_name(tile_id, year)
