"""Backward-compatibility stub — use ``src.legacy.preprocess`` instead."""
import warnings as _w
_w.warn("src.preprocess is deprecated; use src.legacy.preprocess",
        DeprecationWarning, stacklevel=2)
from src.legacy.preprocess import *  # noqa: F401,F403
