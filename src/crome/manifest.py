"""Shared manifest and path-resolution helpers."""

from __future__ import annotations

from pathlib import Path


def find_reference_manifest(reference_path: Path | str) -> Path | None:
    """Locate the JSON manifest sidecar for a vector reference source."""
    resolved = Path(reference_path)
    candidates = (
        resolved.with_suffix(".json"),
        resolved.parent / "manifest.json",
        resolved.parent.parent / "manifest.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
