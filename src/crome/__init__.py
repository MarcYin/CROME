"""CROME package scaffold for the AlphaEarth migration."""

from .bands import ALPHAEARTH_BANDS
from .constants import ALPHAEARTH_COLLECTION_ID
from .schema import alphaearth_feature_columns

__all__ = ["ALPHAEARTH_BANDS", "ALPHAEARTH_COLLECTION_ID", "alphaearth_feature_columns"]
__version__ = "0.1.0"
