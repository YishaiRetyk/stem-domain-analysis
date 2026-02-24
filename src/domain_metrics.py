"""Backward-compatibility stub — use ``src.legacy.domain_metrics`` instead."""
import warnings as _w
_w.warn("src.domain_metrics is deprecated; use src.legacy.domain_metrics",
        DeprecationWarning, stacklevel=2)
from src.legacy.domain_metrics import *  # noqa: F401,F403
