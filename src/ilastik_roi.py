"""Backward-compatibility stub — use ``src.legacy.ilastik_roi`` instead."""
import warnings as _w
_w.warn("src.ilastik_roi is deprecated; use src.legacy.ilastik_roi",
        DeprecationWarning, stacklevel=2)
from src.legacy.ilastik_roi import *  # noqa: F401,F403
