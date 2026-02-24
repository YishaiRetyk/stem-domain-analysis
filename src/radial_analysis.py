"""Backward-compatibility stub — use ``src.legacy.radial_analysis`` instead."""
import warnings as _w
_w.warn("src.radial_analysis is deprecated; use src.legacy.radial_analysis",
        DeprecationWarning, stacklevel=2)
from src.legacy.radial_analysis import *  # noqa: F401,F403
