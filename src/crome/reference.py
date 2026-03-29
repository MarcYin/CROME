"""Pure reference-label helpers for CROME hexagon inputs."""

from collections.abc import Iterable

from .config import CromeReferenceConfig


def build_reference_spec(**kwargs) -> CromeReferenceConfig:
    """Build a CROME reference configuration from keyword arguments."""

    return CromeReferenceConfig(**kwargs)


def validate_reference_columns(
    columns: Iterable[str],
    label_column: str,
    geometry_column: str,
) -> tuple[str, ...]:
    """Validate that the vector reference exposes the required columns."""

    available = tuple(columns)
    missing = [name for name in (label_column, geometry_column) if name not in available]
    if missing:
        raise ValueError(f"Reference data is missing required columns: {missing}")
    return available
