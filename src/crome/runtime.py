"""Runtime environment helpers for geospatial dependencies."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def _is_proj_data_dir(path: Path | str | None) -> bool:
    if path is None:
        return False
    candidate = Path(path)
    return candidate.is_dir() and (candidate / "proj.db").exists()


def _proj_db_minor_version(proj_data_dir: Path | str) -> int:
    """Read DATABASE.LAYOUT.VERSION.MINOR from a proj.db file."""
    db_path = Path(proj_data_dir) / "proj.db"
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT value FROM metadata WHERE key='DATABASE.LAYOUT.VERSION.MINOR'"
        ).fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def ensure_proj_data_env() -> Path | None:
    """Ensure GDAL/OGR can resolve PROJ data files in the current environment.

    Only sets PROJ_DATA/PROJ_LIB when the pyproj database is new enough
    (MINOR >= 6) to avoid breaking rasterio's bundled PROJ on CI runners
    where pyproj ships an older database.
    """

    current_proj = os.environ.get("PROJ_DATA")
    current_proj_lib = os.environ.get("PROJ_LIB")
    if _is_proj_data_dir(current_proj) and _is_proj_data_dir(current_proj_lib):
        if _proj_db_minor_version(current_proj) >= 6:
            return Path(current_proj)

    try:
        from pyproj import datadir
    except Exception:
        return None

    candidate: Path | None = None
    for raw in (current_proj, current_proj_lib):
        if _is_proj_data_dir(raw) and _proj_db_minor_version(raw) >= 6:
            candidate = Path(raw)
            break

    if candidate is None:
        try:
            data_dir = Path(datadir.get_data_dir())
        except Exception:
            data_dir = None
        if data_dir is not None and _is_proj_data_dir(data_dir) and _proj_db_minor_version(data_dir) >= 6:
            candidate = data_dir

    if candidate is None:
        return None

    try:
        datadir.set_data_dir(str(candidate))
    except Exception:
        pass
    os.environ["PROJ_DATA"] = str(candidate)
    os.environ["PROJ_LIB"] = str(candidate)
    return candidate
