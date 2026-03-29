"""AlphaEarth band helpers."""

from collections.abc import Sequence

from .constants import ALPHAEARTH_BAND_COUNT


def alphaearth_band_names() -> tuple[str, ...]:
    """Return the canonical AlphaEarth band order."""

    return tuple(f"A{index:02d}" for index in range(ALPHAEARTH_BAND_COUNT))


ALPHAEARTH_BANDS = alphaearth_band_names()


def validate_alphaearth_bands(bands: Sequence[str] | None = None) -> tuple[str, ...]:
    """Validate or normalize the first-slice AlphaEarth band list.

    The initial migration slice intentionally only supports the full canonical
    `A00` to `A63` embedding vector so the downstream schema stays stable.
    """

    if bands is None:
        return ALPHAEARTH_BANDS

    normalized = tuple(bands)
    if not normalized:
        raise ValueError("At least one band must be provided.")

    if normalized != ALPHAEARTH_BANDS:
        raise ValueError(
            "The first migration slice only supports the full AlphaEarth band "
            "set A00-A63 in canonical order."
        )

    return normalized
