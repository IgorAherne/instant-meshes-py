"""Instant Meshes with browser-based brush guidance.

The C++ core lives in :mod:`instant_meshes_brush._core`; everything re-exported
here is part of the stable Python surface.
"""

from ._core import (  # noqa: F401
    Config,
    Curve,
    ExtractedMesh,
    Session,
    SolveStatus,
    StrokeKind,
    __version__,
    get_verbose,
    set_thread_count,
    set_verbose,
)

__all__ = [
    "Config",
    "Curve",
    "ExtractedMesh",
    "Session",
    "SolveStatus",
    "StrokeKind",
    "get_verbose",
    "set_thread_count",
    "set_verbose",
    "__version__",
]
