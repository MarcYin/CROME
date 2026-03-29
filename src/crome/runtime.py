"""Runtime environment helpers for geospatial dependencies."""

from __future__ import annotations

import os
from pathlib import Path


def _is_proj_data_dir(path: Path | str | None) -> bool:
    if path is None:
        return False
    candidate = Path(path)
    return candidate.is_dir() and (candidate / "proj.db").exists()


def ensure_proj_data_env() -> Path | None:
    """Ensure GDAL/OGR can resolve PROJ data files in the current environment."""

    current_proj = os.environ.get("PROJ_DATA")
    current_proj_lib = os.environ.get("PROJ_LIB")
    if _is_proj_data_dir(current_proj) and _is_proj_data_dir(current_proj_lib):
        return Path(current_proj)

    try:
        from pyproj import datadir
    except Exception:
        return None

    candidate: Path | None = None
    for raw in (current_proj, current_proj_lib):
        if _is_proj_data_dir(raw):
            candidate = Path(raw)
            break

    if candidate is None:
        try:
            data_dir = Path(datadir.get_data_dir())
        except Exception:
            data_dir = None
        if data_dir is not None and _is_proj_data_dir(data_dir):
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
