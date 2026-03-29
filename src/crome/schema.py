"""Pure schema helpers for the AlphaEarth migration."""

from collections.abc import Iterable, Sequence

from .bands import ALPHAEARTH_BANDS
from .config import AlphaEarthTrainingSpec


def alphaearth_feature_columns() -> tuple[str, ...]:
    """Return the canonical ordered AlphaEarth feature vector."""

    return ALPHAEARTH_BANDS


def validate_feature_order(columns: Sequence[str]) -> tuple[str, ...]:
    """Require the full canonical AlphaEarth feature order."""

    normalized = tuple(columns)
    if normalized != ALPHAEARTH_BANDS:
        raise ValueError("AlphaEarth feature columns must match the canonical A00-A63 order.")
    return normalized


def validate_reference_contract(
    columns: Iterable[str],
    spec: AlphaEarthTrainingSpec,
) -> AlphaEarthTrainingSpec:
    """Validate the training contract against known first-slice assumptions."""

    available = tuple(columns)
    for required in (spec.reference.label_column, spec.reference.geometry_column):
        if required not in available:
            raise ValueError(f"Reference data is missing required column: {required}")
    return spec
