"""Backward-compatibility stub — use ``src.legacy.cluster_domains`` or
``src.domain_clustering`` instead."""
import warnings as _w
_w.warn("src.cluster_domains is deprecated; use src.legacy.cluster_domains "
        "or src.domain_clustering", DeprecationWarning, stacklevel=2)
from src.legacy.cluster_domains import *  # noqa: F401,F403
