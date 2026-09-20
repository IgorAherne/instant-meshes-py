"""Asynchronous ownership of the C++ solver sessions.

``instant_meshes_brush._core.Session`` owns a background solver thread and is
not re-entrant: every call has to come from a single thread, and the blocking
ones must never run on the asyncio loop -- uvicorn drops a WebSocket whose
event loop stalls for more than its ping timeout.  :class:`BrushSession`
therefore funnels the whole core API through one dedicated single-worker
executor per session and exposes it as coroutines.

The solve itself is *not* awaited on that worker.  ``solve_orientations`` and
``solve_positions`` only flip flags and wake the C++ thread, so the executor
stays free and a concurrent ``stop()`` or ``status()`` can still get through;
blocking it in ``wait_solve()`` for the duration of a solve would deadlock
cancellation.  Completion is detected by polling the status instead, exactly as
the native viewer polls its version counters.

:class:`SessionRegistry` owns the sessions and reaps the ones whose browser
walked away without closing its socket.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np

from . import _core, uv
from .assets import SourceMesh
from .uv import UnwrapError, UvLayout

LOG = logging.getLogger(__name__)

_T = TypeVar("_T")

DEFAULT_TTL_SECONDS = 30 * 60
DEFAULT_MAX_SESSIONS = 8
DEFAULT_SWEEP_SECONDS = 60.0

#: How often a running solve is checked for completion.  The C++ side only
#: publishes an intermediate result every ``set_preview_interval`` ms anyway,
#: so a tighter poll would buy nothing but executor round-trips.
_SOLVE_POLL_SECONDS = 0.05

#: Field names of :class:`instant_meshes_brush.Config`, in the order the C++
#: constructor takes them.  Used to translate the JSON config a browser sends.
CONFIG_FIELDS: Tuple[str, ...] = (
    "rosy",
    "posy",
    "scale",
    "face_count",
    "vertex_count",
    "crease_angle",
    "extrinsic",
    "align_to_boundaries",
    "deterministic",
    "smooth_iter",
    "pure_quad",
)

#: Accepted ``solve(field=...)`` values.  "both" runs orientations first and
#: then positions, which is what the field hierarchy requires: positions are
#: solved in the frame the orientation field defines.
SOLVE_FIELDS: Tuple[str, ...] = ("orientations", "positions", "both")

EXPORT_FORMATS: Tuple[str, ...] = ("obj", "ply")


class SessionError(RuntimeError):
    """Base class for every failure a client can provoke."""


class SessionClosedError(SessionError):
    """Raised when a session is used after :meth:`BrushSession.close`."""


class SessionLimitError(SessionError):
    """Raised when the registry is at its configured capacity."""


class NotReadyError(SessionError):
    """Raised when an operation needs a mesh that has not been loaded yet."""


class StrokeMissedError(SessionError):
    """Raised when a brush stroke did not land on the surface."""


# ---------------------------------------------------------------------------
#  Value types crossing the async boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolveState:
    """Immutable copy of ``Session.status``, safe to hold on the event loop."""

    active: bool
    progress: float
    level: int
    iterations_q: int
    iterations_o: int
    #: Message from the last solver-thread failure, or empty.  The core clears
    #: its slot as it is read, so whoever polls the status owns it from there.
    error: str = ""

    @property
    def version(self) -> Tuple[int, int]:
        """The pair a streamer compares to decide whether to resend a field."""
        return (self.iterations_q, self.iterations_o)

    @property
    def has_field(self) -> bool:
        """False before ``preprocess``, when both counters read back as -1.

        Either field can be the first one published -- a client may solve
        positions on their own -- so one published counter is enough.
        """
        return self.iterations_q >= 0 or self.iterations_o >= 0

    def as_dict(self) -> Dict[str, Any]:
        """The wire form. ``error`` is left out: it travels as an ERROR frame."""
        return {
            "active": self.active,
            "progress": self.progress,
            "level": self.level,
            "iterations_q": self.iterations_q,
            "iterations_o": self.iterations_o,
        }


@dataclass(frozen=True)
class Geometry:
    """The working mesh, after subdivision, as the browser needs to draw it."""

    name: str
    vertices: np.ndarray  # (nV, 3) float32
    faces: np.ndarray  # (nF, 3) uint32
    normals: np.ndarray  # (nV, 3) float32
    scale: float
    config: Dict[str, Any]


@dataclass(frozen=True)
class FieldSnapshot:
    """Orientation and position fields plus the version stamp they carry."""

    orientation: np.ndarray  # (nV, 3) float32
    position: np.ndarray  # (nV, 3) float32
    state: SolveState


@dataclass(frozen=True)
class StrokeResult:
    """A stroke after projection and geodesic smoothing."""

    stroke_id: int  # 0 for attractors, which leave no persistent constraint
    kind: str
    positions: np.ndarray  # (N, 3) float32
    normals: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (N,) uint32


@dataclass(frozen=True)
class Singularities:
    """Face-centre markers for both singularity kinds, ready to instance."""

    orientation_positions: np.ndarray  # (n, 3) float32
    orientation_colors: np.ndarray  # (n, 3) float32
    position_positions: np.ndarray  # (m, 3) float32
    position_colors: np.ndarray  # (m, 3) float32


@dataclass(frozen=True)
class Extraction:
    """The extracted quad mesh and its wireframe overlay."""

    vertices: np.ndarray  # (nV, 3) float32
    faces: np.ndarray  # (nF, posy) uint32
    face_normals: np.ndarray  # (nF, 3) float32
    wireframe: np.ndarray  # (2 * nF * posy, 3) float32
    wireframe_color: np.ndarray  # (2 * nF * posy, 3) float32


# ---------------------------------------------------------------------------
#  Config translation
# ---------------------------------------------------------------------------


def config_as_dict(config: "_core.Config") -> Dict[str, Any]:
    """Flatten a core ``Config`` into something JSON-serialisable."""
    return {name: getattr(config, name) for name in CONFIG_FIELDS}


def config_from_mapping(
    values: Mapping[str, Any], base: Optional["_core.Config"] = None
) -> "_core.Config":
    """Build a ``Config``, starting from ``base`` and overriding named keys.

    Unknown keys are rejected rather than ignored: a typo in a browser-sent
    config would otherwise silently leave the resolution at its default.
    """
    unknown = sorted(set(values) - set(CONFIG_FIELDS))
    if unknown:
        raise SessionError(
            f"unknown config keys {unknown}; expected any of {list(CONFIG_FIELDS)}"
        )

    config = _core.Config()
    for name in CONFIG_FIELDS:
        if name in values:
            setattr(config, name, values[name])
        elif base is not None:
            setattr(config, name, getattr(base, name))
    return config


# ---------------------------------------------------------------------------
#  Array helpers
# ---------------------------------------------------------------------------


def _as_rows(array: Any, width: int, what: str, dtype: Any) -> np.ndarray:
    """Validate and normalise an ``(N, width)`` array for the C++ boundary."""
    out = np.ascontiguousarray(array, dtype=dtype)
    if out.ndim != 2 or out.shape[1] != width:
        raise SessionError(f"{what}: expected an (N, {width}) array, got shape {out.shape}")
    if out.shape[0] == 0:
        raise SessionError(f"{what}: array is empty")
    return out


def _as_vec3(value: Any, what: str) -> np.ndarray:
    out = np.ascontiguousarray(value, dtype=np.float32).reshape(-1)
    if out.size != 3:
        raise SessionError(f"{what}: expected 3 values, got {out.size}")
    return out


#: Colours the browser draws singularity markers with.  The native viewer uses
#: a different palette (blue/red/green); this one is the UI's own convention.
_SINGULARITY_COLORS: Dict[int, Tuple[float, float, float]] = {
    1: (1.0, 0.0, 0.0),
    3: (0.0, 0.0, 1.0),
}
_SINGULARITY_FALLBACK = (1.0, 1.0, 0.0)


def _singularity_color(index: int) -> Tuple[float, float, float]:
    return _SINGULARITY_COLORS.get(index, _SINGULARITY_FALLBACK)


def _face_centres(
    vertices: np.ndarray, faces: np.ndarray, indices: Sequence[int]
) -> np.ndarray:
    """Mean of each listed triangle's three corners."""
    if not indices:
        return np.zeros((0, 3), dtype=np.float32)
    corners = faces[np.asarray(indices, dtype=np.int64)]
    return vertices[corners].mean(axis=1).astype(np.float32, copy=False)


def _markers(
    vertices: np.ndarray,
    faces: np.ndarray,
    singularities: Mapping[int, Any],
    index_of: Callable[[Any], int],
) -> Tuple[np.ndarray, np.ndarray]:
    face_indices = list(singularities.keys())
    centres = _face_centres(vertices, faces, face_indices)
    colors = np.array(
        [_singularity_color(index_of(singularities[f])) for f in face_indices],
        dtype=np.float32,
    ).reshape(-1, 3)
    return centres, colors


# ---------------------------------------------------------------------------
#  Calls that run on the session's worker thread
# ---------------------------------------------------------------------------


def _read_status(core: "_core.Session") -> SolveState:
    status = core.status
    return SolveState(
        active=status.active,
        # session.h documents progress as [0,1], but the optimizer overshoots
        # while an attractor stroke walks a singularity; clamp so a progress
        # bar bound to this never runs off its track.
        progress=min(1.0, max(0.0, float(status.progress))),
        level=status.level,
        iterations_q=status.iterations_q,
        iterations_o=status.iterations_o,
        error=status.error,
    )


def _read_snapshot(core: "_core.Session") -> FieldSnapshot:
    """Read Q, O and the version stamp in one hop across the thread boundary.

    The stamp is read last so it can only ever be *newer* than the fields it
    describes; a client that redraws a slightly stale field simply gets one
    more update when the next version lands.
    """
    orientation = core.orientation_field
    position = core.position_field
    return FieldSnapshot(orientation, position, _read_status(core))


def _starter(core: "_core.Session", step: str) -> Callable[[int], None]:
    return core.solve_orientations if step == "orientations" else core.solve_positions


def _read_geometry(core: "_core.Session", name: str) -> Geometry:
    return Geometry(
        name=name,
        vertices=core.vertices,
        faces=core.faces,
        normals=core.normals,
        scale=float(core.scale),
        config=config_as_dict(core.config),
    )


def _read_extraction(mesh: "_core.ExtractedMesh") -> Extraction:
    return Extraction(
        vertices=mesh.vertices,
        faces=mesh.faces,
        face_normals=mesh.face_normals,
        wireframe=mesh.wireframe,
        wireframe_color=mesh.wireframe_color,
    )


# ---------------------------------------------------------------------------
#  BrushSession
# ---------------------------------------------------------------------------


class BrushSession:
    """One browser connection's mesh, solver and editing state.

    Every method that touches the core is a coroutine; none of them blocks the
    event loop.  Mutating operations are serialised against each other by an
    :class:`asyncio.Lock`, while the read-only ``status`` and ``snapshot_field``
    deliberately skip it so a streamer keeps working during a long solve.
    """

    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self.mesh_name = ""
        self.last_used = time.monotonic()
        self.export_path: Optional[Path] = None

        self._core: Optional["_core.Session"] = None
        self._config = _core.Config()
        self._geometry: Optional[Geometry] = None
        #: The file as it was authored -- UVs, materials, texture maps. Kept
        #: beside the solver's copy rather than fed into it: the remesher wants
        #: one welded triangle soup and has no use for any of this, but the
        #: viewport can only show the model as its author left it from here.
        self._source: Optional[SourceMesh] = None
        self._geometry_version = 0
        self._extracted: Optional["_core.ExtractedMesh"] = None
        #: The atlas of the extraction on hand, and the slider position it was
        #: cut at. Both are dropped with the extraction they describe.
        self._uv: Optional[UvLayout] = None
        self._uv_leniency = uv.DEFAULT_LENIENCY
        #: How far the unwrapper has got, or None while it is not running.
        #: Written from the unwrapping thread and read from the event loop,
        #: which is safe for a single attribute and is why it is not a queue:
        #: the streamer samples it, so a report it misses costs nothing.
        self._uv_progress: Optional[float] = None
        #: Which unwrap the progress above belongs to.  A thread whose future
        #: was abandoned keeps running and keeps reporting; without this its
        #: next report would resurrect a progress value that nothing is left to
        #: clear, and a session that looks permanently busy is one whose UI
        #: never comes back.
        self._uv_run = 0
        self._export_dir: Optional[Path] = None

        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"imb-{session_id[:8]}"
        )
        self._lock = asyncio.Lock()
        self._solve_task: Optional[asyncio.Task] = None
        self._last_error: Optional[str] = None
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    async def open(self) -> "BrushSession":
        """Construct the C++ session on the worker thread that will own it."""
        self._core = await self._call(_core.Session)
        return self

    async def close(self) -> None:
        """Stop the solver, destroy the core and join the worker. Idempotent."""
        async with self._lock:
            if self._closed:
                return
            await self._stop_locked()
            self._drop_extraction()
            # ~Session() joins the solver thread with the GIL held, so it is
            # only cheap because the solve above was stopped and waited for.
            # Clearing the attribute from the worker keeps that destructor on
            # the one thread that is allowed to touch the object.
            if self._core is not None:
                await self._call(self._release_core)
            self._closed = True

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._executor.shutdown, True)
        if self._export_dir is not None:
            await loop.run_in_executor(None, shutil.rmtree, self._export_dir, True)
            self._export_dir = None
            self.export_path = None

    def _release_core(self) -> None:
        self._core = None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def ready(self) -> bool:
        """True once a mesh has been loaded and preprocessed."""
        return self._geometry is not None

    @property
    def solving(self) -> bool:
        """True while phases remain to be run.

        ``SolveState.active`` reports the C++ optimizer, which goes briefly
        idle between the orientation and position phases of one ``solve`` call;
        a UI that drives its spinner from this does not flicker there.
        """
        return self._solve_task is not None and not self._solve_task.done()

    @property
    def geometry(self) -> Optional[Geometry]:
        return self._geometry

    @property
    def has_extraction(self) -> bool:
        """True while an extracted mesh is on hand and still current.

        Cleared by anything that invalidates it: a rebuild, a new mesh, or a
        change to the extraction options.
        """
        return self._extracted is not None

    @property
    def geometry_version(self) -> int:
        """Increments whenever the working mesh is replaced.

        Watchers compare this the way they compare the solver's field counters,
        so a mesh loaded from anywhere reaches every attached viewport.
        """
        return self._geometry_version

    @property
    def config(self) -> Dict[str, Any]:
        """The config in force, with the values ``preprocess`` derived filled in.

        The requested config keeps ``scale``/``face_count``/``vertex_count`` at
        -1 for whatever the client left to the algorithm; reporting that back
        would show a UI a target edge length of -1.  The requested form is
        still what a later :meth:`set_config` builds on, so both are kept.
        """
        if self._geometry is None:
            return config_as_dict(self._config)

        values = dict(self._geometry.config)
        # That snapshot was taken by preprocess. The two extraction options can
        # be retargeted afterwards without one, so they are read live instead --
        # otherwise a client would be told pure quads are off while the next
        # extraction produces them.
        values["smooth_iter"] = self._config.smooth_iter
        values["pure_quad"] = self._config.pure_quad
        return values

    def touch(self) -> None:
        """Mark the session as in use, postponing the registry's TTL sweep."""
        self.last_used = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_used

    # -- executor plumbing -------------------------------------------------

    async def _call(self, fn: Callable[..., _T], *args: Any) -> _T:
        if self._closed:
            raise SessionClosedError(f"session {self.id} is closed")
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, fn, *args)
        except RuntimeError as exc:
            # close() can win the race between the check above and the submit;
            # a dead executor is a closed session, not an internal failure.
            if self._closed and "shutdown" in str(exc):
                raise SessionClosedError(f"session {self.id} is closed") from exc
            raise

    def _keep_error(self, state: SolveState) -> None:
        """Hold on to a solver-thread failure so a client can still be told.

        ``Session.status`` hands the message over and clears its slot, so the
        first poll -- usually the streamer's, not the caller who asked for the
        solve -- is the only one that ever sees it.
        """
        if state.error:
            LOG.error("solver failed in session %s: %s", self.id, state.error)
            self._last_error = state.error

    def take_error(self) -> Optional[str]:
        """Consume the last solver failure, or None if it was already shown."""
        error, self._last_error = self._last_error, None
        return error

    def _require_core(self) -> "_core.Session":
        if self._closed or self._core is None:
            raise SessionClosedError(f"session {self.id} is closed")
        return self._core

    def _require_ready(self) -> "_core.Session":
        core = self._require_core()
        self._require_geometry()
        return core

    def _require_geometry(self) -> Geometry:
        if self._geometry is None:
            raise NotReadyError("no mesh loaded; send LOAD_MESH first")
        return self._geometry

    # -- input -------------------------------------------------------------

    @property
    def source(self) -> Optional[SourceMesh]:
        """The imported file with its materials, or None if none was kept."""
        return self._source

    async def load_mesh(
        self,
        vertices: Any,
        faces: Any,
        name: str = "mesh",
        config: Optional[Mapping[str, Any]] = None,
        source: Optional[SourceMesh] = None,
    ) -> Geometry:
        """Replace the input mesh and preprocess it, returning the result."""
        v = _as_rows(vertices, 3, "vertices", np.float32)
        f = _as_rows(faces, 3, "faces", np.uint32)
        if int(f.max()) >= v.shape[0]:
            raise SessionError(
                f"faces reference vertex {int(f.max())} but only {v.shape[0]} were given"
            )

        async with self._lock:
            core = self._require_core()
            await self._stop_locked()
            self._invalidate_locked()
            if config is not None:
                self._config = config_from_mapping(config, self._config)
            await self._call(core.set_mesh, v, f)
            self.mesh_name = name
            # After the invalidate above, which drops the previous file's.
            self._source = source
            return await self._preprocess_locked()

    async def load_file(
        self, path: Any, config: Optional[Mapping[str, Any]] = None
    ) -> Geometry:
        """Load an OBJ/PLY straight through the C++ reader."""
        source = Path(path)
        async with self._lock:
            core = self._require_core()
            await self._stop_locked()
            self._invalidate_locked()
            if config is not None:
                self._config = config_from_mapping(config, self._config)
            await self._call(core.load_file, str(source))
            self.mesh_name = source.name
            return await self._preprocess_locked()

    async def set_config(self, values: Mapping[str, Any]) -> Geometry:
        """Re-target the resolution, carrying the brush strokes across.

        The fields do not come with them -- the hierarchy they were solved on
        is gone -- so a caller has to solve again, which also means an
        attractor's effect is lost where a stroke's is not.
        """
        async with self._lock:
            self._require_ready()
            await self._stop_locked()
            self._config = config_from_mapping(values, self._config)
            return await self._preprocess_locked()

    async def _preprocess_locked(self) -> Geometry:
        core = self._require_core()
        await self._call(core.preprocess, self._config)
        geometry = await self._call(_read_geometry, core, self.mesh_name)
        self._set_geometry_locked(geometry)
        return geometry

    def _set_geometry_locked(self, geometry: Optional[Geometry]) -> None:
        """Replace the geometry and bump the version every watcher polls.

        A session is shared: the Gradio panel may load a mesh or re-target the
        resolution while a browser viewport is already attached over its own
        WebSocket. Without a version the viewport would keep showing the mesh it
        first received, so the counter is what tells the streamer to resend.
        """
        self._geometry = geometry
        self._geometry_version += 1
        self._drop_extraction()

    def _invalidate_locked(self) -> None:
        # The file is gone, so are its materials; a caller loading a new mesh
        # puts the new ones back, and everyone else is simply clearing up.
        self._source = None
        self._set_geometry_locked(None)

    def _drop_extraction(self) -> None:
        """Forget the extracted mesh and everything derived from it.

        The atlas is cut from the extraction, so it cannot outlive it: keeping
        one would texture a mesh whose faces are no longer the ones it was
        flattened from.
        """
        self._extracted = None
        self._uv = None

    # -- strokes -----------------------------------------------------------

    async def project_and_add_stroke(
        self, ray_origins: Any, ray_directions: Any, kind: int
    ) -> StrokeResult:
        """Project a screen-space drag onto the mesh and keep it as a constraint.

        ``kind`` is a :class:`instant_meshes_brush.StrokeKind` value: 0 combs
        the orientation field along the stroke, 1 additionally pins an output
        edge onto it.
        """
        if int(kind) not in (int(_core.StrokeKind.ORIENTATION), int(_core.StrokeKind.EDGE)):
            raise SessionError(
                f"stroke kind must be 0 (orientation) or 1 (edge), got {kind!r}"
            )
        name = "orientation" if int(kind) == int(_core.StrokeKind.ORIENTATION) else "edge"

        async with self._lock:
            core = self._require_ready()
            await self._stop_locked()
            curve = await self._project_locked(ray_origins, ray_directions, attractor=False)
            stroke_id = await self._call(core.add_stroke, int(kind), curve)
            return _stroke_result(int(stroke_id), name, curve)

    async def apply_attractor(
        self, ray_origins: Any, ray_directions: Any, orientation: bool
    ) -> StrokeResult:
        """Queue a singularity drag along the stroke and watch it run.

        ``Session::applyAttractor`` starts the level-0 solve the move needs
        itself, with the integer variables left frozen, and the solve ends by
        itself once the singularity has arrived.  Starting another solve on top
        would thaw them -- session.h is explicit that only the attractor wants
        the frozen sweep -- and a thawed level-0 solve never terminates, so
        this only follows the one the core started.
        """
        name = "attractor_orientation" if orientation else "attractor_position"
        async with self._lock:
            core = self._require_ready()
            await self._stop_locked()
            curve = await self._project_locked(ray_origins, ray_directions, attractor=True)
            await self._call(core.apply_attractor, curve, bool(orientation))
            self._solve_task = asyncio.create_task(
                self._drive_solve(core, (), 0), name=f"imb-attractor-{self.id}"
            )
            return _stroke_result(0, name, curve)

    async def _project_locked(
        self, ray_origins: Any, ray_directions: Any, attractor: bool
    ) -> "_core.Curve":
        core = self._require_ready()
        origins = _as_rows(ray_origins, 3, "ray_origins", np.float32)
        directions = _as_rows(ray_directions, 3, "ray_directions", np.float32)
        if origins.shape[0] != directions.shape[0]:
            raise SessionError("ray_origins and ray_directions must have the same length")
        if origins.shape[0] < 2:
            raise SessionError("a stroke needs at least two samples")

        curve = await self._call(core.project_stroke, origins, directions, attractor)
        if curve is None or len(curve) < 2:
            raise StrokeMissedError("the stroke did not land on the surface")
        return curve

    async def erase_stroke(self, stroke_id: int) -> bool:
        async with self._lock:
            core = self._require_ready()
            await self._stop_locked()
            return bool(await self._call(core.erase_stroke, int(stroke_id)))

    async def erase_stroke_near(self, point: Any, eye: Any, radius: float) -> int:
        """Erase the stroke handle nearest ``point`` and visible from ``eye``."""
        p = _as_vec3(point, "point")
        e = _as_vec3(eye, "eye")
        async with self._lock:
            core = self._require_ready()
            await self._stop_locked()
            return int(await self._call(core.erase_stroke_near, p, e, float(radius)))

    async def clear_strokes(self) -> int:
        """Remove every stroke, returning how many there were.

        The count is what lets a caller decide whether a re-solve is owed; both
        halves happen in one worker call so the answer cannot be stale.
        """
        async with self._lock:
            core = self._require_ready()
            await self._stop_locked()
            return int(await self._call(_clear_strokes, core))

    async def strokes(self) -> List[Dict[str, Any]]:
        """The persistent strokes, as ``{'id', 'kind', 'n_points'}`` records."""
        core = self._require_ready()
        return await self._call(_read_strokes, core)

    async def stroke_curves(self) -> List[StrokeResult]:
        """Every surviving stroke, curve included.

        A rebuild re-projects the strokes onto the new mesh, so the curves a
        client was drawing are no longer the curves the solver is using; this
        is how it gets the replacements.
        """
        core = self._require_ready()
        return await self._call(_read_stroke_curves, core)

    # -- solving -----------------------------------------------------------

    async def solve(self, field: str = "both", level: int = -1) -> None:
        """Start a solve, replacing whatever was already running.

        ``level=-1`` runs the hierarchical schedule and finishes on its own;
        ``level=0`` refines at full resolution and runs until :meth:`stop`, so
        a ``"both"`` solve at level 0 never reaches its second phase.  The one
        level-0 solve that ends by itself belongs to :meth:`apply_attractor`,
        which starts it in the core rather than here.
        """
        if field not in SOLVE_FIELDS:
            raise SessionError(
                f"solve field must be one of {list(SOLVE_FIELDS)}, got {field!r}"
            )
        steps = ("orientations", "positions") if field == "both" else (field,)
        async with self._lock:
            core = self._require_ready()
            await self._stop_locked()
            # The extraction on hand was read out of the field this is about to
            # change. There is no Extract button in the viewport any more, so
            # nothing else would notice; holding on to it means an export after
            # a brush stroke silently writes the mesh from before it.
            self._drop_extraction()
            # Start the first phase here rather than in the driver, so that a
            # caller which reports its status right after solve() returns
            # already sees an active solver -- and so a refused start raises.
            await self._call(_starter(core, steps[0]), int(level))
            self._solve_task = asyncio.create_task(
                self._drive_solve(core, steps[1:], int(level)), name=f"imb-solve-{self.id}"
            )

    async def _drive_solve(
        self, core: "_core.Session", remaining: Sequence[str], level: int
    ) -> None:
        """Wait out the running phase, then start and wait out ``remaining``.

        The first phase is always already running -- ``solve`` starts it, and
        an attractor's is started by the core -- so this only ever follows.
        Starting a solve returns immediately (the C++ thread does the work), so
        the executor stays free for ``stop()`` and for the status polls that
        detect completion.  Blocking it in ``wait_solve`` instead would leave
        no way to interrupt.
        """
        try:
            if not await self._await_phase(core):
                return
            for step in remaining:
                await self._call(_starter(core, step), level)
                if not await self._await_phase(core):
                    return
        except SessionClosedError:
            pass  # the session was torn down mid-solve; nothing left to drive
        except Exception:
            # Nobody awaits this task except _stop_locked(), which swallows
            # results, so an unlogged failure here would be invisible.
            LOG.exception("solve failed in session %s", self.id)
            raise

    async def _await_phase(self, core: "_core.Session") -> bool:
        """Poll until the running phase ends. False if the solver failed."""
        while True:
            state = await self._call(_read_status, core)
            self._keep_error(state)
            if state.error:
                # The optimizer aborted and cleared its own flags; a further
                # phase would only walk into the same failure.
                return False
            if not state.active:
                return True
            await asyncio.sleep(_SOLVE_POLL_SECONDS)

    async def wait_solve(self) -> None:
        """Wait out the running solve, if there is one."""
        task = self._solve_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def ensure_solved(self) -> bool:
        """Wait for, or run, the solve an extraction has to read. True if it waited.

        Solving is a consequence of importing, retargeting or brushing rather
        than something a user asks for, so there is no longer a button to press
        when it has not happened.  Extraction reads the fields directly, and on
        an unsolved hierarchy that means extracting the solver's random initial
        state, so it holds the guarantee up itself.

        A solve already running is waited out rather than cut short.  Extracting
        stops the solver, and stopping one that a brush stroke had just started
        left the field half solved with nothing to restart it -- so asking to
        see the result while the field was still moving quietly threw the rest
        of that solve away.
        """
        if self._closed or self._core is None or self._geometry is None:
            return False
        if self.solving:
            await self.wait_solve()
            return True
        if (await self.status()).has_field:
            return False
        await self.solve("both", -1)
        await self.wait_solve()
        return True

    async def stop(self) -> None:
        """Ask the solver to finish its sweep and wait until it has."""
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        task, self._solve_task = self._solve_task, None
        if task is not None:
            # Cancel the driver first: once it is gone it cannot start the next
            # phase behind our back.  Its in-flight executor job still runs to
            # completion, but the single worker is FIFO, so the stop below is
            # guaranteed to be seen by whatever that job started.
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._geometry is None or self._core is None or self._closed:
            return
        await self._call(self._core.stop_solve)
        await self._call(self._core.wait_solve)

    async def status(self) -> SolveState:
        core = self._core
        if self._closed or core is None or self._geometry is None:
            return SolveState(False, 0.0, -1, -1, -1)
        state = await self._call(_read_status, core)
        self._keep_error(state)
        return state

    async def snapshot_field(self) -> Optional[FieldSnapshot]:
        """Copy Q and O out of the hierarchy, or None before a mesh exists."""
        core = self._core
        if self._closed or core is None or self._geometry is None:
            return None
        snapshot = await self._call(_read_snapshot, core)
        self._keep_error(snapshot.state)
        return snapshot

    async def set_preview_interval(self, milliseconds: int) -> None:
        """How often a hierarchical solve publishes an intermediate result."""
        core = self._require_ready()
        await self._call(core.set_preview_interval, int(milliseconds))

    # -- results -----------------------------------------------------------

    async def singularities(self) -> Singularities:
        """Face-centre markers and colours for both singularity kinds."""
        core = self._require_ready()
        return await self._call(_read_singularities, core, self._require_geometry())

    async def set_extraction_options(
        self,
        smooth_iter: Optional[int] = None,
        pure_quad: Optional[bool] = None,
    ) -> bool:
        """Retarget the two options ``extract`` reads. True if they changed.

        Everything else in the config is baked in by ``preprocess``, so changing
        it means a rebuild and the loss of every stroke.  These two are read at
        extraction time, so they are simply set -- and the extraction on hand,
        which was built with the previous values, is dropped.
        """
        async with self._lock:
            return await self._set_extraction_options_locked(smooth_iter, pure_quad)

    async def _set_extraction_options_locked(
        self, smooth_iter: Optional[int], pure_quad: Optional[bool]
    ) -> bool:
        core = self._require_ready()
        smooth = (
            self._config.smooth_iter if smooth_iter is None else max(0, int(smooth_iter))
        )
        pure = self._config.pure_quad if pure_quad is None else bool(pure_quad)
        if (smooth, pure) == (self._config.smooth_iter, self._config.pure_quad):
            return False

        # Mirrored into the stored config as well as pushed into the core: a
        # later set_config builds on it, and would otherwise revert them.
        self._config.smooth_iter = smooth
        self._config.pure_quad = pure
        await self._call(core.set_extraction_options, smooth, pure)
        self._drop_extraction()
        return True

    async def extract(
        self,
        smooth_iter: Optional[int] = None,
        pure_quad: Optional[bool] = None,
    ) -> Extraction:
        """Extract the output mesh, stopping the solve so it is self-consistent."""
        # Before the lock: ensure_solved starts a solve, which takes it too.
        await self.ensure_solved()
        async with self._lock:
            core = self._require_ready()
            await self._set_extraction_options_locked(smooth_iter, pure_quad)
            await self._stop_locked()
            mesh = await self._call(core.extract)
            self._extracted = mesh
            return await self._call(_read_extraction, mesh)

    # -- texture space -----------------------------------------------------

    @property
    def uv_layout(self) -> Optional[UvLayout]:
        """The atlas of the extraction on hand, if one has been cut."""
        return self._uv

    @property
    def uv_leniency(self) -> float:
        """Where the chart-size control stands, in [0, 1]."""
        return self._uv_leniency

    @property
    def uv_progress(self) -> Optional[float]:
        """How far an unwrap has got, or None when none is running."""
        return self._uv_progress

    async def unwrap(self, leniency: Optional[float] = None) -> UvLayout:
        """Cut a UV atlas for the extracted mesh, extracting first if need be.

        The layout is cached beside the extraction it describes and re-cut only
        when the slider moves, so hovering the control costs nothing after the
        first look.
        """
        if not uv.available():
            raise SessionError(
                "UV unwrapping needs the xatlas package: pip install xatlas"
            )
        target = uv.DEFAULT_LENIENCY if leniency is None else float(leniency)
        target = min(1.0, max(0.0, target))

        # Set before the extraction rather than after it: on a mesh that has
        # never been extracted that is most of the wait, and a control that
        # says nothing for the first two seconds looks broken.
        self._uv_run += 1
        run = self._uv_run
        report = self._progress_reporter(run)
        self._uv_progress = 0.0
        try:
            # Outside the lock: extract() takes it, and takes it again through
            # ensure_solved if nothing has ever been solved.
            if self._extracted is None:
                await self.extract()

            async with self._lock:
                self._require_ready()
                self._uv_leniency = target
                cached = self._uv
                if cached is not None and cached.leniency == target:
                    return cached
                mesh = self._extracted
                if mesh is None:
                    raise SessionError("nothing to unwrap")
                # The arrays are read on the worker, because only that thread
                # may touch the core -- but xatlas is then run off it. It needs
                # nothing from the core and takes tens of seconds, and while it
                # held the worker no status could be read: the streamer went
                # silent for the whole unwrap, which left the viewport unable
                # to tell that the solve it was waiting for had ended.
                vertices, faces = await self._call(_extraction_arrays, mesh)
                try:
                    layout = await asyncio.to_thread(
                        uv.unwrap, vertices, faces, target, report
                    )
                except UnwrapError as exc:
                    raise SessionError(str(exc)) from exc
                self._uv = layout
                return layout
        finally:
            # Bumped as well as cleared: an abandoned thread that reports again
            # after this must not set it back to a number.
            self._uv_run += 1
            self._uv_progress = None

    def _progress_reporter(self, run: int) -> Callable[[float], None]:
        """A progress callback bound to one unwrap. The streamer samples it."""

        def report(fraction: float) -> None:
            if run == self._uv_run:
                self._uv_progress = min(1.0, max(0.0, float(fraction)))

        return report

    async def extraction(self) -> Optional[Extraction]:
        """Read the extracted mesh on hand, without building a new one.

        An extraction can now happen as a side effect -- unwrapping needs one
        -- so a caller that wants to show the client what it is about to
        export cannot tell from :meth:`extract` alone whether it has been sent.
        """
        if self._extracted is None:
            return None
        return await self._call(_read_extraction, self._extracted)

    async def export_mesh(self, fmt: str = "obj") -> Path:
        """Write the last extraction to a session-private file and return it.

        An OBJ carries the atlas when one has been cut, which is how the
        texture space a caller chose with :meth:`unwrap` leaves the session;
        PLY does not, having no per-corner form for it.
        """
        suffix = fmt.lower().lstrip(".")
        if suffix not in EXPORT_FORMATS:
            raise SessionError(f"export format must be one of {list(EXPORT_FORMATS)}")
        if self._extracted is None:
            await self.extract()

        async with self._lock:
            core = self._require_ready()
            mesh = self._extracted
            if mesh is None:
                raise SessionError("nothing to export")
            directory = await self._ensure_export_dir()
            stem = Path(self.mesh_name or "mesh").stem or "mesh"
            target = directory / f"{stem}_remeshed.{suffix}"
            layout = self._uv if suffix == "obj" else None
            if layout is not None:
                await self._call(_write_textured_obj, str(target), mesh, layout)
            else:
                # Per-face normals are written as `f v//n` with a different n
                # for every face, so a loader that keys vertices on (position,
                # normal) -- trimesh does -- splits the mesh into disconnected
                # faces on re-import. The normals are recoverable from the
                # geometry, the connectivity is not, so they are left out.
                await self._call(core.write_mesh, str(target), mesh, False)
            self.export_path = target
            return target

    async def _ensure_export_dir(self) -> Path:
        if self._export_dir is None:
            loop = asyncio.get_running_loop()
            path = await loop.run_in_executor(
                None, lambda: tempfile.mkdtemp(prefix=f"imb-{self.id[:8]}-")
            )
            self._export_dir = Path(path)
        return self._export_dir


def _stroke_result(stroke_id: int, kind: str, curve: "_core.Curve") -> StrokeResult:
    return StrokeResult(
        stroke_id=stroke_id,
        kind=kind,
        positions=curve.positions,
        normals=curve.normals,
        faces=curve.faces,
    )


def _extraction_arrays(mesh: "_core.ExtractedMesh") -> Tuple[np.ndarray, np.ndarray]:
    """Copy the geometry out of the core, for use on any thread.

    Copied rather than handed over: the arrays go to a thread that keeps
    running after this session may have dropped the mesh they came from.
    """
    return np.array(mesh.vertices, copy=True), np.array(mesh.faces, copy=True)


def _write_textured_obj(path: str, mesh: "_core.ExtractedMesh", layout: UvLayout) -> None:
    return uv.write_obj(path, mesh.vertices, mesh.faces, layout)


def _clear_strokes(core: "_core.Session") -> int:
    count = len(core.strokes)
    core.clear_strokes()
    return count


#: StrokeKind values as the wire names them.
_STROKE_KIND_NAMES = {
    int(_core.StrokeKind.ORIENTATION): "orientation",
    int(_core.StrokeKind.EDGE): "edge",
}


def _read_stroke_curves(core: "_core.Session") -> List[StrokeResult]:
    return [
        _stroke_result(
            int(stroke["id"]),
            _STROKE_KIND_NAMES.get(int(stroke["kind"]), "orientation"),
            stroke["curve"],
        )
        for stroke in core.strokes
    ]


def _read_strokes(core: "_core.Session") -> List[Dict[str, Any]]:
    return [
        {"id": int(s["id"]), "kind": int(s["kind"]), "n_points": len(s["curve"])}
        for s in core.strokes
    ]


def _read_singularities(core: "_core.Session", geometry: Geometry) -> Singularities:
    """Markers for whichever fields have actually been solved.

    A field whose version counter is still negative has never been published,
    and running the singularity detector over its random initial state reports
    tens of thousands of meaningless defects -- which is why the native viewer
    guards on the same counters before refreshing its markers.
    """
    state = _read_status(core)
    empty = (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))

    if state.iterations_q >= 0:
        o_pos, o_col = _markers(
            geometry.vertices, geometry.faces, core.orientation_singularities, int
        )
    else:
        o_pos, o_col = empty

    if state.iterations_o >= 0:
        # A position singularity is an integer shift (i, j); its dominant
        # component plays the part the index plays for an orientation one.
        p_pos, p_col = _markers(
            geometry.vertices,
            geometry.faces,
            core.position_singularities,
            lambda shift: max(abs(int(shift[0])), abs(int(shift[1]))),
        )
    else:
        p_pos, p_col = empty

    return Singularities(o_pos, o_col, p_pos, p_col)


# ---------------------------------------------------------------------------
#  SessionRegistry
# ---------------------------------------------------------------------------


class SessionRegistry:
    """Session ids to sessions, with a capacity limit and an idle sweeper.

    A WebSocket close is the authoritative teardown; the sweeper only covers
    browsers that vanished without one, which would otherwise leave a solver
    thread and its hierarchy resident forever.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        sweep_seconds: float = DEFAULT_SWEEP_SECONDS,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        self.ttl_seconds = float(ttl_seconds)
        self.max_sessions = int(max_sessions)
        self.sweep_seconds = float(sweep_seconds)
        self._sessions: Dict[str, BrushSession] = {}
        self._sweeper: Optional[asyncio.Task] = None
        self._closed = False

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._sessions

    def ids(self) -> Iterator[str]:
        return iter(tuple(self._sessions))

    async def start(self) -> None:
        """Start the idle sweeper. Idempotent, and safe to skip entirely."""
        if self._sweeper is None and not self._closed:
            self._sweeper = asyncio.create_task(self._sweep_loop(), name="imb-session-sweeper")

    async def create(self) -> BrushSession:
        if self._closed:
            raise SessionClosedError("the session registry is shut down")
        if len(self._sessions) >= self.max_sessions:
            await self.sweep()
        if len(self._sessions) >= self.max_sessions:
            raise SessionLimitError(
                f"all {self.max_sessions} session slots are in use; close a tab and retry"
            )

        session_id = secrets.token_urlsafe(16)
        session = BrushSession(session_id)
        try:
            await session.open()
        except BaseException:
            # The worker thread exists from the first submit onwards, so a
            # session that never opened still has to be shut down.
            await session.close()
            raise
        self._sessions[session_id] = session
        await self.start()
        LOG.info("created session %s (%d live)", session_id, len(self._sessions))
        return session

    def get(self, session_id: str) -> Optional[BrushSession]:
        """Look a session up and mark it as in use.

        A session closed directly rather than through :meth:`close_session` is
        dropped here instead of being handed out, because every call on it
        would raise.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.closed:
            self._sessions.pop(session_id, None)
            return None
        session.touch()
        return session

    async def close_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        await session.close()
        LOG.info("closed session %s (%d live)", session_id, len(self._sessions))
        return True

    async def sweep(self) -> int:
        """Close every session idle for longer than the TTL. Returns the count."""
        stale = [
            sid for sid, s in self._sessions.items() if s.idle_seconds() > self.ttl_seconds
        ]
        for session_id in stale:
            LOG.info("reaping idle session %s", session_id)
            await self.close_session(session_id)
        return len(stale)

    async def _sweep_loop(self) -> None:
        interval = max(1.0, min(self.sweep_seconds, self.ttl_seconds))
        while True:
            await asyncio.sleep(interval)
            try:
                await self.sweep()
            except Exception:  # a reaper that dies leaks every later session
                LOG.exception("session sweep failed")

    async def aclose(self) -> None:
        """Cancel the sweeper and tear every session down."""
        self._closed = True
        sweeper, self._sweeper = self._sweeper, None
        if sweeper is not None:
            sweeper.cancel()
            await asyncio.gather(sweeper, return_exceptions=True)

        sessions = list(self._sessions.values())
        self._sessions.clear()
        results = await asyncio.gather(*(s.close() for s in sessions), return_exceptions=True)
        for session, result in zip(sessions, results):
            if isinstance(result, BaseException):
                LOG.error("session %s failed to close: %r", session.id, result)


_default_registry: Optional[SessionRegistry] = None


def default_registry() -> SessionRegistry:
    """The process-wide registry the Gradio and WebSocket layers both use."""
    global _default_registry
    if _default_registry is None:
        _default_registry = SessionRegistry()
    return _default_registry


def set_default_registry(registry: Optional[SessionRegistry]) -> None:
    """Install (or clear, with None) the registry :func:`default_registry` returns."""
    global _default_registry
    _default_registry = registry
