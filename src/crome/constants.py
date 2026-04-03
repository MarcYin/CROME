"""Shared constants for the initial migration slice."""

ALPHAEARTH_COLLECTION_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
ALPHAEARTH_FIRST_YEAR = 2017
ALPHAEARTH_LAST_STABLE_YEAR = 2024
ALPHAEARTH_LAST_ALLOWED_YEAR = 2025
ALPHAEARTH_BAND_COUNT = 64
ALPHAEARTH_TARGET_RESOLUTION_M = 10.0
CROME_DEFAULT_LABEL_COLUMN = "lucode"
CROME_DEFAULT_GEOMETRY_COLUMN = "geometry"


def validate_year(year: int, *, context: str = "Year") -> int:
    """Validate that a single year falls within the AlphaEarth coverage window.

    Each download or reference request targets one year; multi-year workflows
    call this once per year.
    """
    if year < ALPHAEARTH_FIRST_YEAR or year > ALPHAEARTH_LAST_ALLOWED_YEAR:
        raise ValueError(
            f"{context} must be between {ALPHAEARTH_FIRST_YEAR} and "
            f"{ALPHAEARTH_LAST_ALLOWED_YEAR}."
        )
    return year
