"""HTTP and WebSocket front end for the brushing UI.

``build_app`` returns a plain :class:`fastapi.FastAPI` that serves the viewer,
mints sessions and speaks the binary protocol in :mod:`.protocol`.  It is
deliberately mount-agnostic: a Gradio layer can call
``gradio.mount_gradio_app(build_app(), blocks, path="/")`` afterwards, and
because Starlette dispatches the first matching route in registration order,
everything registered here -- the WebSocket included -- keeps winning over the
Gradio mount.

Each connection runs two coroutines against one :class:`.session_manager.
BrushSession`: a reader that dispatches client frames, and a streamer that
polls the solver's version counters and pushes a field update only when they
move.  Neither ever calls into C++ directly; the session's own worker thread
does that, so the event loop keeps answering pings during a multi-minute solve.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    BinaryIO,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from urllib.parse import quote, urlencode, urlsplit

import numpy as np
import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import _core, assets, protocol, uv
from .assets import AssetError, SourceMesh
from .protocol import MessageType
from .uv import UvLayout
from .session_manager import (
    BrushSession,
    Extraction,
    Geometry,
    SessionError,
    SessionLimitError,
    SessionRegistry,
    Singularities,
    SolveState,
    StrokeResult,
    default_registry,
)

LOG = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

#: Where the viewer's own assets are served from. Deliberately not "/static":
#: Gradio serves its fonts and helper scripts from there, and because routes
#: registered here are matched before the Gradio mount, taking that prefix
#: would 404 them for every host application.  index.html hard-codes this.
STATIC_URL = "/imb-assets"

#: The page a host embeds, one per session: ``/viewer?session=<id>``.
VIEWER_PATH = "/viewer"

#: The viewer's own options, read by static/options.js from its URL; see
#: :func:`viewer_url`.  The first of each is the default.
VIEWER_PANEL_SIDES: Tuple[str, ...] = ("left", "right")
VIEWER_EXPORT_ACTIONS: Tuple[str, ...] = ("download", "host")
DEFAULT_EXPORT_LABEL = "Export"

DEFAULT_FPS = 15
MIN_FPS = 1
MAX_FPS = 60

#: A mesh upload is one WebSocket frame, and uvicorn's 16 MiB default would
#: close the socket with 1009 rather than report an error.
WS_MAX_SIZE = 256 * 1024 * 1024

#: Formats load_mesh_file can read. Kept here so the upload route can reject an
#: unusable file before it reaches a parser.
MESH_SUFFIXES = assets.SUFFIXES

#: Which file of an upload is the model when none of the files picked is one
#: (a zip): the formats that carry the most of a model first.
MODEL_PREFERENCE: Tuple[str, ...] = (
    ".fbx", ".glb", ".gltf", ".obj", ".dae", ".ply", ".stl", ".off"
)

#: Ceiling on what one upload puts on disk, a zip's unpacked contents included.
MAX_UPLOAD_BYTES = 2 * 1024**3

#: Ceiling on the files one zip may unpack to.
MAX_ZIP_MEMBERS = 10_000

#: How long a browser may keep a map it fetched with the import's generation
#: in the URL: for good, because that URL can never mean other bytes.
TEXTURE_CACHE = "private, max-age=31536000, immutable"

#: Stroke flavours a client may name, mapped to the session call they select.
_STROKE_KINDS: Dict[Any, str] = {
    int(_core.StrokeKind.ORIENTATION): "orientation",
    int(_core.StrokeKind.EDGE): "edge",
    "orientation": "orientation",
    "edge": "edge",
    "attractor_orientation": "attractor_orientation",
    "attractor_position": "attractor_position",
}

#: What any change to the stroke set re-solves.
#:
#: Every stroke constrains the orientation field (session.cpp fills CQ/CQw for
#: both kinds), so both brushes have to re-solve it; the edge brush also pins
#: CO and therefore continues into the positions, which is what the viewer's
#: mContinueWithPositions does.  Level -1 is the hierarchical schedule, which
#: terminates on its own -- the viewport has no Stop button.
#:
#: Erasing takes the same plan as adding: a field that keeps the shape a stroke
#: gave it after the stroke is deleted is simply wrong, and there is no longer a
#: Solve button with which to notice and correct it.
_STROKE_SOLVE: Tuple[str, int] = ("both", -1)

#: Stroke kinds that re-solve when drawn.  Attractors are absent on purpose:
#: the core starts the frozen level-0 solve their move needs, and the session
#: follows it.
_SOLVE_PLANS: Dict[str, Tuple[str, int]] = {
    "orientation": _STROKE_SOLVE,
    "edge": _STROKE_SOLVE,
}


# ---------------------------------------------------------------------------
#  The viewer page's URL
# ---------------------------------------------------------------------------


def _host_origin(raw: str) -> str:
    """``raw`` as a bare http(s) origin, or ValueError.

    The same rule the viewer applies (options.js hostOriginOf): a scheme, a
    host and maybe a port, and nothing else -- a path or a query means the
    caller has not said which page it means, and host mode must not guess.
    """
    parts = urlsplit(raw.strip())
    try:
        usable = (
            parts.scheme in ("http", "https")
            and bool(parts.hostname)
            and parts.username is None
            and parts.password is None
            and parts.path in ("", "/")
            and not parts.query
            and not parts.fragment
            # Reading the port validates it: "http://host:abc" raises here.
            and (parts.port is None or parts.port > 0)
        )
    except ValueError:
        usable = False
    if not usable:
        raise ValueError(
            f"host_origin must be a page's origin such as http://127.0.0.1:7770, not {raw!r}"
        )
    return f"{parts.scheme}://{parts.netloc}"


def viewer_url(
    session_id: str,
    *,
    panel: str = "left",
    export_label: Optional[str] = None,
    export_action: str = "download",
    host_origin: Optional[str] = None,
) -> str:
    """The path of the viewer page for ``session_id``, with its options.

    ``panel`` docks the control panel on the ``"left"`` (default) or the
    ``"right"`` of the viewport; on the right, the readout -- the brush's name
    and the status line -- moves to the viewport's bottom-right corner too.
    ``export_label`` renames the Export button (``"Accept"``).
    ``export_action="host"`` makes that button write and download nothing:
    it posts an ``export-request`` message to the page embedding the viewer,
    which answers with ``export-state`` messages (README_PYTHON.md has the
    protocol). It needs ``host_origin``, the embedding page's origin, which is
    the only origin the viewer then talks to and listens to; it is ignored in
    the default ``"download"`` mode.

    Options left at their defaults are left out of the URL, so
    ``viewer_url(session_id)`` is the plain ``/viewer?session=<id>`` every
    embedder has always used. Everything is URL-encoded.

    Raises ValueError for a side or an action the viewer does not know, and
    for host mode without a usable ``host_origin``: the viewer itself would
    only fall back to its defaults and say so in the browser's console, which
    is a mistake far easier to catch here.
    """
    if not session_id:
        raise ValueError("viewer_url needs a session id")
    if panel not in VIEWER_PANEL_SIDES:
        raise ValueError(f"panel must be one of {VIEWER_PANEL_SIDES}, not {panel!r}")
    if export_action not in VIEWER_EXPORT_ACTIONS:
        raise ValueError(
            f"export_action must be one of {VIEWER_EXPORT_ACTIONS}, not {export_action!r}"
        )

    query: List[Tuple[str, str]] = [("session", session_id)]
    if panel != VIEWER_PANEL_SIDES[0]:
        query.append(("panel", panel))
    label = " ".join((export_label or "").split())
    if label and label != DEFAULT_EXPORT_LABEL:
        query.append(("export_label", label))
    if export_action == "host":
        if not host_origin:
            raise ValueError("export_action='host' needs host_origin, the embedding page's origin")
        query.append(("export_action", export_action))
        query.append(("host_origin", _host_origin(host_origin)))
    return f"{VIEWER_PATH}?{urlencode(query, quote_via=quote)}"


# ---------------------------------------------------------------------------
#  Mesh input
# ---------------------------------------------------------------------------


def _rows(array: Any, width: int, what: str) -> np.ndarray:
    """Accept either an ``(N, width)`` array or the flat form of one."""
    if array is None:
        raise SessionError(f"missing array {what!r}")
    values = np.asarray(array)
    if values.ndim == 1:
        if values.size % width:
            raise SessionError(
                f"{what}: {values.size} values do not divide into rows of {width}"
            )
        return values.reshape(-1, width)
    if values.ndim != 2 or values.shape[1] != width:
        raise SessionError(f"{what}: expected an (N, {width}) array, got shape {values.shape}")
    return values


def read_source(path: Path) -> SourceMesh:
    """Read a model file with its materials and maps, as SessionError on failure.

    The previews are encoded straight away, which frees the images decoded to
    tell what each map is; a session would otherwise hold on to them until
    somebody asked to see a map.
    """
    try:
        source = assets.load_source(path)
    except AssetError as exc:
        raise SessionError(str(exc)) from exc
    source.encode_previews()
    return source


def load_mesh_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read any supported file down to the triangles the remesher takes."""
    try:
        return assets.solver_mesh(assets.load_source(path, materials=False))
    except AssetError as exc:
        raise SessionError(str(exc)) from exc


def _require_source(
    sessions: SessionRegistry, session_id: str
) -> Tuple[BrushSession, SourceMesh]:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    source = session.source
    if source is None:
        raise HTTPException(status_code=404, detail="this session has no imported model")
    return session, source


# ---------------------------------------------------------------------------
#  Uploads
# ---------------------------------------------------------------------------


class UploadError(Exception):
    """An upload that cannot be imported, with the HTTP status that says why."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class _Budget:
    """The bytes an upload may still write before it is refused."""

    def __init__(self, limit: int) -> None:
        self.left = limit

    def take(self, count: int) -> None:
        self.left -= count
        if self.left < 0:
            raise UploadError(
                413, f"the upload unpacks to more than {MAX_UPLOAD_BYTES // 1024**3} GB"
            )


#: Characters no file name here may hold: path separators, and what Windows
#: refuses in a name.
_UNSAFE_CHARACTERS = re.compile(r'[\x00-\x1f<>:"/\\|?*]')
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"{port}{n}" for port in ("com", "lpt") for n in range(1, 10)}
)


def _safe_name(raw: str) -> str:
    """A name that is safe to create a file under: a base name, nothing refused.

    A browser sends a base name, but the name is the client's to choose, and
    one that holds ``../`` or ``C:`` must still land inside the upload folder.
    """
    base = raw.replace("\\", "/").rsplit("/", 1)[-1]
    base = _UNSAFE_CHARACTERS.sub("_", base).strip().rstrip(". ")
    if not base:
        return "file"
    if base.split(".", 1)[0].lower() in _RESERVED_NAMES:
        return f"_{base}"
    return base


def _copy(source: BinaryIO, target: Path, budget: _Budget) -> None:
    with target.open("wb") as out:
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                return
            budget.take(len(chunk))
            out.write(chunk)


def _member_parts(name: str) -> Optional[List[str]]:
    """The safe path parts a zip member unpacks to, or None to leave it out.

    A member that climbs out of the archive ("zip slip") refuses the whole
    zip rather than being skipped: an archive built to write outside the
    folder it is unpacked into is not one to take anything else from either.
    """
    text = name.replace("\\", "/")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if text.startswith("/") or re.match(r"^[A-Za-z]:", text) or ".." in parts:
        raise UploadError(400, f"the zip holds a path that leads out of it: {name}")
    if not parts or parts[0] == "__MACOSX" or parts[-1].startswith("._"):
        return None  # macOS resource forks, which only look like the files
    return [_safe_name(part) for part in parts]


#: What reading a zip member raises when it cannot be unpacked: encrypted
#: (RuntimeError), an unsupported compression, or damaged data.
_UNZIP_ERRORS = (RuntimeError, NotImplementedError, EOFError, zipfile.BadZipFile, zlib.error)


def _unzip(handle: BinaryIO, folder: Path, budget: _Budget) -> None:
    try:
        archive = zipfile.ZipFile(handle)
    except zipfile.BadZipFile as exc:
        raise UploadError(
            400, "the zip could not be opened: it is damaged or not a zip"
        ) from exc
    root = folder.resolve()
    with archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) > MAX_ZIP_MEMBERS:
            raise UploadError(413, f"the zip holds more than {MAX_ZIP_MEMBERS} files")
        for info in members:
            parts = _member_parts(info.filename)
            if parts is None:
                continue
            target = folder.joinpath(*parts)
            if not target.resolve().is_relative_to(root):
                raise UploadError(
                    400, f"the zip holds a path that leads out of it: {info.filename}"
                )
            if target.exists():
                continue  # two names that sanitise alike: the first one wins
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(info) as member:
                    _copy(member, target, budget)
            except _UNZIP_ERRORS as exc:
                raise UploadError(400, f"{info.filename} could not be unpacked: {exc}") from exc


def _pick_model(folder: Path) -> Optional[Path]:
    """The model among unpacked files: by format, then the shallowest, then by name."""
    rank = {suffix: position for position, suffix in enumerate(MODEL_PREFERENCE)}
    found = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in MESH_SUFFIXES
    ]
    if not found:
        return None
    return min(
        found,
        key=lambda path: (
            rank.get(path.suffix.lower(), len(rank)),
            len(path.relative_to(folder).parts),
            str(path).lower(),
        ),
    )


def _gather_maps(folder: Path, model_folder: Path) -> None:
    """Link every file of the upload that sits outside the model's folder into it.

    Maps are only ever looked for inside the model's own folder, and an asset
    pack keeps them in a sibling (``Meshes/model.fbx`` beside ``Textures/``).
    Linked by base name; a name the model's folder already has is left alone.
    """
    if model_folder == folder:
        return
    for path in list(folder.rglob("*")):
        if not path.is_file() or path.is_relative_to(model_folder):
            continue
        target = model_folder / path.name
        if target.exists():
            continue
        try:
            os.link(path, target)
        except OSError:
            shutil.copyfile(path, target)


def store_upload(files: Sequence[Tuple[str, BinaryIO]], folder: Path) -> Path:
    """Write an upload into ``folder`` and return the model file among it.

    ``files`` are ``(file name, readable)`` pairs: the model with its material
    library, buffers and maps, or a zip holding them, which keeps its folder
    layout.  Names are reduced to safe base names.  The model is the first
    file picked that is one, else the best one a zip unpacked.
    """
    budget = _Budget(MAX_UPLOAD_BYTES)
    picked: Optional[Path] = None
    for name, handle in files:
        safe = _safe_name(name)
        if Path(safe).suffix.lower() == ".zip":
            _unzip(handle, folder, budget)
            continue
        target = folder / safe
        if target.exists():
            continue  # the first of two files with one name wins
        _copy(handle, target, budget)
        if picked is None and target.suffix.lower() in MESH_SUFFIXES:
            picked = target

    model = picked or _pick_model(folder)
    if model is None:
        supported = ", ".join(sorted(MESH_SUFFIXES))
        suffix = Path(_safe_name(files[0][0])).suffix.lower() if files else ""
        if len(files) == 1 and suffix != ".zip":
            raise UploadError(
                415, f"unsupported mesh format '{suffix}'; use one of {supported}"
            )
        raise UploadError(
            415, f"none of the uploaded files is a model; use one of {supported}"
        )
    if model.stat().st_size == 0:
        raise UploadError(400, f"the uploaded file {model.name} is empty")
    _gather_maps(folder, model.parent)
    return model


def _as_config(values: Any, what: str) -> Optional[Mapping[str, Any]]:
    if values is None or isinstance(values, Mapping):
        return values
    raise SessionError(
        f"{what} expects a mapping of config values, got {type(values).__name__}"
    )


def _extraction_options(
    message: protocol.Message,
) -> Tuple[Optional[int], Optional[bool]]:
    """``(smooth_iter, pure_quad)`` from a frame, None where it said nothing.

    These two ride with EXTRACT and EXPORT rather than with SET_CONFIG, because
    they are the only settings that do not need the hierarchy rebuilt -- and a
    rebuild would take every brush stroke with it.
    """
    smooth = message.get("smooth_iter")
    pure = message.get("pure_quad")
    return (None if smooth is None else int(smooth), None if pure is None else bool(pure))


# ---------------------------------------------------------------------------
#  Frame builders
# ---------------------------------------------------------------------------


def _geometry_frame(geometry: Geometry, version: int = 0) -> bytes:
    lo = geometry.vertices.min(axis=0)
    hi = geometry.vertices.max(axis=0)
    header = {
        "version": int(version),
        "name": geometry.name,
        "n_vertices": int(geometry.vertices.shape[0]),
        "n_faces": int(geometry.faces.shape[0]),
        "scale": float(geometry.scale),
        "bbox": [float(v) for v in (*lo, *hi)],
        # Lifted out of the config because the renderer needs the symmetries to
        # pick its field shader before it has looked at anything else.
        "rosy": int(geometry.config["rosy"]),
        "posy": int(geometry.config["posy"]),
        "config": geometry.config,
    }
    arrays = {
        "vertices": geometry.vertices,
        "faces": geometry.faces,
        "normals": geometry.normals,
    }
    return protocol.encode(MessageType.GEOMETRY, header, arrays)


def _field_frame(
    orientation: np.ndarray, position: np.ndarray, state: SolveState, solving: bool
) -> bytes:
    header = dict(state.as_dict())
    header["n_vertices"] = int(orientation.shape[0])
    header["solving"] = solving
    return protocol.encode(
        MessageType.FIELD, header, {"orientation": orientation, "position": position}
    )


def _singularity_frame(singularities: Singularities) -> bytes:
    header = {
        "n_orientation": int(singularities.orientation_positions.shape[0]),
        "n_position": int(singularities.position_positions.shape[0]),
    }
    arrays = {
        "orientation_positions": singularities.orientation_positions,
        "orientation_colors": singularities.orientation_colors,
        "position_positions": singularities.position_positions,
        "position_colors": singularities.position_colors,
    }
    return protocol.encode(MessageType.SINGULARITIES, header, arrays)


def _stroke_frame(result: StrokeResult, resent: bool = False) -> bytes:
    header = {
        "stroke_id": int(result.stroke_id),
        "kind": result.kind,
        "n_points": int(result.positions.shape[0]),
        # A rebuild re-projects every stroke and sends the results, which look
        # exactly like a stroke that was just drawn. The viewport shows the
        # input surface when one is drawn -- a brush needs something to draw on
        # -- and without this an echo of an old stroke took the view away from
        # whoever was looking at the result.
        "resent": bool(resent),
    }
    arrays = {
        "positions": result.positions,
        "normals": result.normals,
        "faces": result.faces,
    }
    return protocol.encode(MessageType.STROKE_RESULT, header, arrays)


def _uv_frame(layout: UvLayout, version: int = 0) -> bytes:
    """The flattened mesh, as the viewer draws it.

    Sent as an index space of its own rather than as ready-made line segments:
    the layout is the extracted mesh with its seams opened, so it is exactly as
    compact as the mesh is, and the client already knows how to fan a
    quad-dominant face array.
    """
    header = {
        "version": int(version),
        "n_charts": int(layout.chart_count),
        "n_faces": int(layout.faces.shape[0]),
        "posy": int(layout.faces.shape[1]),
        "leniency": float(layout.leniency),
        "unmapped": int(layout.unmapped),
        "cut": layout.cut_count,
    }
    arrays = {
        "uv": layout.uv,
        # int32, not uint32: a face a chart boundary went through is -1 here
        # and appears in "tris" instead, as the two triangles it became.
        "faces": layout.faces.astype(np.int32, copy=False),
        "chart": layout.chart.astype(np.int32, copy=False),
        "tris": layout.cut.astype(np.int32, copy=False).reshape(-1, 3),
        "tri_chart": layout.cut_chart.astype(np.int32, copy=False),
    }
    return protocol.encode(MessageType.UV_LAYOUT, header, arrays)


def _extracted_frame(extraction: Extraction, version: int = 0) -> bytes:
    header = {
        "version": int(version),
        "n_vertices": int(extraction.vertices.shape[0]),
        "n_faces": int(extraction.faces.shape[0]),
        "posy": int(extraction.faces.shape[1]) if extraction.faces.size else 0,
    }
    arrays = {
        "vertices": extraction.vertices,
        "faces": extraction.faces,
        "face_normals": extraction.face_normals,
        "wireframe": extraction.wireframe,
        "wireframe_color": extraction.wireframe_color,
    }
    return protocol.encode(MessageType.EXTRACTED, header, arrays)


# ---------------------------------------------------------------------------
#  One WebSocket connection
# ---------------------------------------------------------------------------


class _Connection:
    """Reader plus field streamer for a single browser socket."""

    def __init__(self, websocket: WebSocket, session: BrushSession) -> None:
        self.ws = websocket
        self.session = session
        self.fps = DEFAULT_FPS
        self._send_lock = asyncio.Lock()
        self._streamer: Optional[asyncio.Task] = None
        #: (active, solving) as last *sent*, wherever it was sent from. The
        #: streamer's change detector reads it, so a reply that a handler
        #: pushed out of band still counts as having been reported.
        self._last_summary: Optional[Tuple[bool, bool]] = None
        #: Geometry version as last *sent*, so a mesh loaded from the Gradio
        #: panel -- or any other holder of this session -- reaches this socket.
        self._sent_geometry_version = -1
        #: Unwrap progress as last sent, in whole percent; -1 for "not running".
        self._sent_uv_percent = -1

    # -- transport ---------------------------------------------------------

    async def _send(self, frame: bytes) -> None:
        # The streamer and the reader both write here; uvicorn's WebSocket
        # implementation is not safe against interleaved sends.
        async with self._send_lock:
            try:
                await self.ws.send_bytes(frame)
            except RuntimeError as exc:
                # Starlette reports a send on a socket it has already closed as
                # a RuntimeError. To every caller here that is simply a client
                # that left, and one exception type means one way to handle it.
                raise WebSocketDisconnect(code=1006) from exc

    async def _send_error(self, message: str, *, fatal: bool = False) -> None:
        await self._send(protocol.error(message, fatal=fatal))

    async def run(self) -> None:
        self._streamer = asyncio.create_task(
            self._stream_fields(), name=f"imb-stream-{self.session.id}"
        )
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                raw = message.get("bytes")
                if raw is None:
                    await self._send_error("this endpoint only accepts binary frames")
                    continue
                self.session.touch()
                await self._dispatch(raw)
        except WebSocketDisconnect:
            pass
        finally:
            self._streamer.cancel()
            await asyncio.gather(self._streamer, return_exceptions=True)
            # The C++ solver thread outlives the socket unless it is told not
            # to; the session itself stays for a reconnect until the TTL sweep.
            with contextlib.suppress(Exception):
                await self.session.stop()

    async def _dispatch(self, raw: bytes) -> None:
        try:
            message = protocol.decode(raw)
        except protocol.ProtocolError as exc:
            await self._send_error(str(exc), fatal=True)
            return

        handler = self._HANDLERS.get(message.type)
        if handler is None:
            await self._send_error(f"unsupported message type {message.type}")
            return

        try:
            await handler(self, message)
        except WebSocketDisconnect:
            # The client left mid-reply. That is not a handler failure, and
            # answering it with an error frame would only raise again.
            raise
        except SessionError as exc:
            await self._send_error(str(exc))
        except Exception as exc:  # a bad frame must never drop the socket
            LOG.exception("handler for message type %d failed", message.type)
            await self._send_error(f"{type(exc).__name__}: {exc}")

    # -- field streaming ---------------------------------------------------

    async def _stream_fields(self) -> None:
        """Push Q/O whenever the solver's version counters move.

        Each iteration awaits its own send, so a slow client throttles the
        stream instead of accumulating a backlog of already-stale frames.
        """
        last_version: Optional[Tuple[int, int]] = None
        was_active = False
        #: Set the moment anything starts solving, paid off once everything has
        #: stopped. See the comment at the send below for why it is not simply
        #: "whenever the optimizer is idle".
        markers_owed = False
        try:
            while True:
                await asyncio.sleep(1.0 / self.fps)
                if self.session.closed:
                    break
                # Keep the registry's idle sweeper away from a live socket that
                # is only watching a long solve.
                self.session.touch()

                # First, because it is the one report that costs nothing: it
                # reads a plain attribute rather than the session's worker.
                # The rest of the tick still runs -- an unwrap no longer holds
                # that worker, and muting the stream for its duration is what
                # left the viewport unable to see that the solve it was
                # waiting for had finished.
                await self._send_uv_progress()

                try:
                    # A mesh can arrive from the Gradio panel rather than from
                    # this socket, so the geometry gets the same change
                    # detection the fields do.
                    if await self._send_geometry_if_new():
                        last_version = None
                        was_active = False
                        markers_owed = False

                    state = await self.session.status()
                    solving = self.session.solving
                    await self._flush_solver_error()
                    # A solve that ends without touching the counters again --
                    # its last sweep may not have published -- still owes the
                    # client a final, exact frame: hence the idle transition.
                    settled = was_active and not state.active
                    was_active = state.active
                    markers_owed = markers_owed or state.active or solving

                    if state.has_field and (state.version != last_version or settled):
                        last_version = await self._send_field()
                        # A field that moved may have moved its defects with
                        # it, and catching the solver mid-flight is not
                        # something to rely on: an attractor's single sweep
                        # can begin and end between two ticks, which used to
                        # leave the markers on screen describing the field as
                        # it was before the drag.
                        markers_owed = True

                    # Singularities only once EVERYTHING has stopped, not merely
                    # when the optimizer is idle. It goes idle in the gap between
                    # the orientation and position phases of one solve, and the
                    # markers computed there come from a freshly solved
                    # orientation field crossed with a position field that has
                    # not caught up with it -- hundreds of defects that do not
                    # exist, which flash red across the model and then vanish.
                    if markers_owed and state.has_field and not state.active and not solving:
                        markers_owed = False
                        await self._send(_singularity_frame(await self.session.singularities()))

                    # The solver can go idle without any counter moving, so the
                    # run state gets its own change detector; without it a
                    # client is left believing a finished solve still runs.
                    if (state.active, solving) != self._last_summary:
                        await self._send_status()
                except SessionError:
                    # A mesh being swapped in, or a session reaped mid-frame:
                    # the next tick re-reads whatever the session became.
                    LOG.debug("streamer skipped a tick", exc_info=True)
            # Only the idle sweeper closes a session out from under its socket,
            # and a client that is told can offer a reload instead of hanging.
            await self._send_error("this session expired; reload the page", fatal=True)
        except asyncio.CancelledError:
            raise
        except WebSocketDisconnect:
            pass  # the socket went away; run()'s finally does the tidying up
        except Exception:
            LOG.info("field streamer for session %s stopped", self.session.id, exc_info=True)

    async def _send_uv_progress(self) -> bool:
        """Publish how far the unwrapper has got. True while one is running.

        Quantised to the percent the panel prints, so a fraction that moves
        smoothly does not cost a frame per report.
        """
        progress = self.session.uv_progress
        percent = -1 if progress is None else int(progress * 100)
        if percent != self._sent_uv_percent:
            self._sent_uv_percent = percent
            await self._send(protocol.encode(MessageType.PROGRESS, {"uv": progress}))
        return progress is not None

    def _stamp(self) -> int:
        """The geometry version every result frame is labelled with.

        A result takes seconds to build and the mesh can be replaced while it
        is in flight, so a client needs to be able to tell that the extraction
        it has just been handed describes the mesh before the last rebuild.
        """
        return self.session.geometry_version

    async def _send_geometry(self, geometry: Geometry) -> None:
        """Answer a mesh change this socket asked for.

        Marking the version as sent keeps the streamer from immediately
        repeating the same frame on its next tick.
        """
        self._sent_geometry_version = self.session.geometry_version
        await self._send(_geometry_frame(geometry, self._stamp()))
        await self._send_status()

    async def _send_geometry_if_new(self) -> bool:
        """Push the working mesh if it changed since this socket last saw it.

        Returns True when a frame went out, which tells the streamer to forget
        the field versions it was tracking: they belong to the previous mesh.
        """
        version = self.session.geometry_version
        if version == self._sent_geometry_version:
            return False

        geometry = self.session.geometry
        # Record the version either way: an invalidated session has nothing to
        # send, and re-checking it every tick would be pointless.
        self._sent_geometry_version = version
        if geometry is None:
            return False

        await self._send(_geometry_frame(geometry, self._stamp()))
        await self._send_status()
        return True

    async def _send_field(self) -> Optional[Tuple[int, int]]:
        """Send one field update. Markers are the streamer's business."""
        snapshot = await self.session.snapshot_field()
        if snapshot is None:
            return None
        await self._send(
            _field_frame(
                snapshot.orientation,
                snapshot.position,
                snapshot.state,
                self.session.solving,
            )
        )
        return snapshot.state.version

    # -- replies -----------------------------------------------------------

    async def _send_status(self) -> None:
        state = await self.session.status()
        solving = self.session.solving
        header = dict(state.as_dict())
        header.update(
            session_id=self.session.id,
            protocol=protocol.PROTOCOL_VERSION,
            ready=self.session.ready,
            solving=solving,
            mesh=self.session.mesh_name,
            fps=self.fps,
            config=self.session.config,
            # Whether xatlas is installed at all. The viewer hides its UV
            # control when it is not, rather than offering one that can only
            # answer with an error.
            uv=uv.available(),
            uv_leniency=self.session.uv_leniency,
            # Repeated here as well as in its own frame so the control's label
            # is self-correcting: a client that missed the report saying an
            # unwrap had ended would otherwise print a percentage forever.
            uv_progress=self.session.uv_progress,
            version=self.session.geometry_version,
            # How many map views the imported file has. The viewport offers a
            # button for each and none at all where there are none, which is
            # every mesh format that carries no materials.
            textures=len(self.session.source.buttons) if self.session.source else 0,
        )
        # Record what actually went out, not what a caller intended to report:
        # a handler that answers SOLVE with active=True must leave the streamer
        # knowing that the matching "it finished" frame is still owed.
        self._last_summary = (state.active, solving)
        await self._flush_solver_error()
        await self._send(protocol.encode(MessageType.STATUS, header))

    async def _flush_solver_error(self) -> None:
        """Report a solver-thread failure once, to whoever asks first.

        The core clears the message as the status is read, so the session keeps
        it; without this the optimizer would abort a solve in silence and the
        client would keep waiting for a field that is never coming.
        """
        error = self.session.take_error()
        if error:
            await self._send_error(f"the solver stopped: {error}")

    async def _send_stroke_list(self) -> None:
        strokes = await self.session.strokes()
        await self._send(
            protocol.encode(
                MessageType.STROKE_LIST, {"strokes": strokes, "count": len(strokes)}
            )
        )

    # -- handlers ----------------------------------------------------------

    async def _on_hello(self, message: protocol.Message) -> None:
        await self._send_status()

    async def _on_ping(self, message: protocol.Message) -> None:
        await self._send(protocol.encode(MessageType.PONG, dict(message.header)))

    async def _on_load_mesh(self, message: protocol.Message) -> None:
        config = _as_config(message.get("config"), "LOAD_MESH")
        # The name ends up in a file name on export, so it has to be text.
        name = message.get("name")
        name = str(name) if name is not None else None
        source: Optional[SourceMesh] = None
        if "vertices" in message.arrays and "faces" in message.arrays:
            vertices = _rows(message.arrays["vertices"], 3, "vertices")
            faces = _rows(message.arrays["faces"], 3, "faces")
        elif message.get("path"):
            path = Path(str(message.get("path")))
            source = await asyncio.to_thread(read_source, path)
            vertices, faces = await asyncio.to_thread(assets.solver_mesh, source)
            name = name or path.name
        else:
            raise SessionError(
                "LOAD_MESH needs either vertices/faces arrays or a 'path' header"
            )

        geometry = await self.session.load_mesh(
            vertices, faces, name or "mesh", config, source=source
        )
        await self._send_geometry(geometry)

    async def _on_set_config(self, message: protocol.Message) -> None:
        # The config may be nested under "config" or simply be the header. An
        # empty nested config is still a config -- it means "re-preprocess with
        # what I already set" -- so this tests for absence, not emptiness.
        values = _as_config(message.get("config"), "SET_CONFIG")
        if values is None:
            values = message.header
        geometry = await self.session.set_config(values)
        await self._send_geometry(geometry)
        # The rebuild re-projected the strokes, so the curves the client is
        # drawing are not the ones now constraining the field. Send the
        # replacements, or the viewport shows a mesh with no visible reason
        # for the shape its flow has taken.
        for stroke in await self.session.stroke_curves():
            await self._send(_stroke_frame(stroke, resent=True))
        await self._send_stroke_list()

    async def _on_stroke(self, message: protocol.Message) -> None:
        raw_kind = message.get("kind", int(_core.StrokeKind.ORIENTATION))
        kind = _STROKE_KINDS.get(raw_kind)
        if kind is None:
            known = sorted(str(k) for k in _STROKE_KINDS)
            raise SessionError(f"unknown stroke kind {raw_kind!r}; expected one of {known}")
        origins = _rows(message.arrays.get("ray_origins"), 3, "ray_origins")
        directions = _rows(message.arrays.get("ray_directions"), 3, "ray_directions")

        if kind.startswith("attractor_"):
            # The drag is the solve here, so "solve" has nothing left to say:
            # the session is already following the one the core started.
            result = await self.session.apply_attractor(
                origins, directions, orientation=kind == "attractor_orientation"
            )
            await self._send(_stroke_frame(result))
        else:
            stroke_kind = (
                _core.StrokeKind.ORIENTATION
                if kind == "orientation"
                else _core.StrokeKind.EDGE
            )
            result = await self.session.project_and_add_stroke(
                origins, directions, int(stroke_kind)
            )
            await self._send(_stroke_frame(result))
            if message.get("solve", False):
                field, level = _SOLVE_PLANS[kind]
                await self.session.solve(field, level)
        await self._send_status()

    async def _on_erase_stroke(self, message: protocol.Message) -> None:
        stroke_id = message.get("stroke_id")
        if stroke_id is not None:
            erased = await self.session.erase_stroke(int(stroke_id))
        elif message.get("point") is not None and message.get("eye") is not None:
            erased = await self.session.erase_stroke_near(
                message.get("point"), message.get("eye"), float(message.get("radius", 0.0))
            )
        else:
            raise SessionError("ERASE_STROKE needs a 'stroke_id', or a 'point' plus an 'eye'")
        await self._send_stroke_list()
        await self._resolve_after_stroke_change(bool(erased))

    async def _on_clear_strokes(self, message: protocol.Message) -> None:
        cleared = await self.session.clear_strokes()
        await self._send_stroke_list()
        await self._resolve_after_stroke_change(cleared > 0)

    async def _resolve_after_stroke_change(self, changed: bool) -> None:
        """Re-solve after a stroke was removed, exactly as adding one does.

        Skipped when nothing was actually removed -- a click that lands on no
        handle still arrives here, and restarting a solve for it would throw
        away the field the user is looking at.
        """
        if not changed:
            return
        await self.session.solve(*_STROKE_SOLVE)
        await self._send_status()

    async def _on_solve(self, message: protocol.Message) -> None:
        field = str(message.get("field", "both"))
        await self.session.solve(field, int(message.get("level", -1)))
        await self._send_status()

    async def _on_stop(self, message: protocol.Message) -> None:
        await self.session.stop()
        await self._send_status()

    async def _on_extract(self, message: protocol.Message) -> None:
        extraction = await self.session.extract(*_extraction_options(message))
        await self._send(_extracted_frame(extraction, self._stamp()))
        await self._send_status()

    async def _on_unwrap(self, message: protocol.Message) -> None:
        leniency = message.get("leniency")
        layout = await self.session.unwrap(
            None if leniency is None else float(leniency)
        )
        await self._send(_uv_frame(layout, self._stamp()))
        await self._send_status()

    async def _on_export(self, message: protocol.Message) -> None:
        fmt = str(message.get("format", "obj"))
        # Exporting is allowed without extracting first, so the options travel
        # with this frame too; setting them drops an extraction built with the
        # old ones. Whatever has to be extracted here is also sent, so the file
        # the client downloads is the mesh it is looking at.
        await self.session.set_extraction_options(*_extraction_options(message))
        if not self.session.has_extraction:
            await self.session.extract()
        # An OBJ carries its texture space, so the atlas is cut here if the
        # user never looked at it. A failure costs the file its UVs and is
        # reported, rather than costing them the export they asked for.
        if fmt.lower().lstrip(".") == "obj" and uv.available():
            leniency = message.get("leniency")
            try:
                # Sent back as well as written: the client is then holding the
                # layout its file carries, so the count beside the slider is
                # the one that left, and the next hover needs no round trip.
                layout = await self.session.unwrap(
                    None if leniency is None else float(leniency)
                )
                await self._send(_uv_frame(layout, self._stamp()))
            except SessionError as exc:
                await self._send_error(f"exported without a UV layout: {exc}")
        path = await self.session.export_mesh(fmt)
        # Sent unconditionally rather than only when this handler built it: an
        # extraction can now appear as a side effect of unwrapping, and the
        # size under the button has to be the size of the file that just left.
        extraction = await self.session.extraction()
        if extraction is not None:
            await self._send(_extracted_frame(extraction, self._stamp()))
        header = {
            "url": f"/api/session/{self.session.id}/export",
            "filename": path.name,
            # The written suffix, not the spelling asked for: "OBJ" and ".obj"
            # are both accepted, and the client labels its download with this.
            "format": path.suffix.lstrip("."),
            "bytes": int(path.stat().st_size),
        }
        await self._send(protocol.encode(MessageType.EXPORT_READY, header))

    async def _on_subscribe(self, message: protocol.Message) -> None:
        self.fps = max(MIN_FPS, min(MAX_FPS, int(message.get("fps", DEFAULT_FPS))))
        preview_ms = message.get("preview_ms")
        if preview_ms is not None and self.session.ready:
            await self.session.set_preview_interval(int(preview_ms))
        await self._send_status()

    _HANDLERS: Dict[int, Callable[["_Connection", protocol.Message], Awaitable[None]]] = {
        MessageType.HELLO: _on_hello,
        MessageType.LOAD_MESH: _on_load_mesh,
        MessageType.SET_CONFIG: _on_set_config,
        MessageType.STROKE: _on_stroke,
        MessageType.ERASE_STROKE: _on_erase_stroke,
        MessageType.SOLVE: _on_solve,
        MessageType.STOP: _on_stop,
        MessageType.EXTRACT: _on_extract,
        MessageType.EXPORT: _on_export,
        MessageType.CLEAR_STROKES: _on_clear_strokes,
        MessageType.SUBSCRIBE: _on_subscribe,
        MessageType.PING: _on_ping,
        MessageType.UNWRAP: _on_unwrap,
    }


# ---------------------------------------------------------------------------
#  Application
# ---------------------------------------------------------------------------


def build_app(registry: Optional[SessionRegistry] = None) -> FastAPI:
    """Assemble the viewer app around ``registry`` (the shared one by default)."""
    sessions = registry if registry is not None else default_registry()

    # The algorithm narrates its progress on stdout, which is noise in a server.
    _core.set_verbose(False)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await sessions.start()
        try:
            yield
        finally:
            # Ctrl-C must not leave solver threads behind: every C++ session
            # joins its thread here, while there is still a loop to await on.
            await sessions.aclose()

    app = FastAPI(title="instant-meshes-brush", lifespan=lifespan)
    app.state.registry = sessions
    app.mount(STATIC_URL, StaticFiles(directory=STATIC_DIR), name="imb-assets")

    @app.middleware("http")
    async def revalidate_assets(request, call_next):
        """Let the assets be cached, but never used without asking first.

        The viewer is an ES module graph: main.js imports the rest by relative
        URL, so versioning the page's own script tag would not reach them. A
        browser holding one stale module while the server has a new one gives a
        viewer that half works -- which is what an upgrade of this package
        would do. `no-cache` keeps the cache and only requires the ETag to be
        checked, so the usual answer is a 304 and no bytes move.
        """
        response = await call_next(request)
        if request.url.path.startswith(STATIC_URL + "/"):
            response.headers["cache-control"] = "no-cache"
        return response

    # The page is the same file whatever the query says: the viewer reads its
    # options (panel side, Export button) from its own URL, so the server
    # neither parses nor rewrites them. See viewer_url().
    @app.get(VIEWER_PATH, include_in_schema=False)
    async def viewer() -> FileResponse:
        if not INDEX_HTML.is_file():
            raise HTTPException(status_code=404, detail=f"{INDEX_HTML.name} is not installed")
        return FileResponse(INDEX_HTML, media_type="text/html")

    @app.post("/api/session")
    async def create_session() -> Dict[str, str]:
        try:
            session = await sessions.create()
        except SessionLimitError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"session_id": session.id, "ws_url": f"/ws/{session.id}"}

    @app.post("/api/session/{session_id}/mesh")
    async def upload_mesh(
        session_id: str,
        files: Optional[List[UploadFile]] = File(default=None),
        file: Optional[UploadFile] = File(default=None),
        config: Optional[str] = Form(default=None),
    ) -> Dict[str, Any]:
        """Load a model straight from the viewport's own file picker.

        ``files`` is the model with everything it refers to -- material
        library, buffers, maps -- or one zip of them; ``file`` is the single
        model older clients send.  The reply is deliberately just a summary:
        the geometry itself reaches every socket attached to this session
        through the streamer, so a second viewport watching it updates too.
        """
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        uploads = [*(files or []), *([file] if file is not None else [])]
        if not uploads:
            raise HTTPException(status_code=400, detail="no file was uploaded")

        settings: Optional[Mapping[str, Any]] = None
        if config:
            try:
                settings = _as_config(json.loads(config), "upload config")
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"bad config: {exc}") from exc

        # The readers work from disk, and a parser should never see the raw
        # upload path, so everything lands in a scratch folder removed either
        # way.  Nothing needs it afterwards: the maps' bytes are read in.
        with tempfile.TemporaryDirectory(prefix="imb-upload-") as scratch:
            named = [(upload.filename or "", upload.file) for upload in uploads]
            try:
                model = await asyncio.to_thread(store_upload, named, Path(scratch))
            except UploadError as exc:
                raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
            try:
                source = await asyncio.to_thread(read_source, model)
                vertices, faces = await asyncio.to_thread(assets.solver_mesh, source)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            try:
                geometry = await session.load_mesh(
                    vertices, faces, model.name, settings, source=source
                )
            except SessionError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        return {
            "name": geometry.name,
            "n_vertices": int(geometry.vertices.shape[0]),
            "n_faces": int(geometry.faces.shape[0]),
            "scale": float(geometry.scale),
            "config": session.config,
            "textures": len(source.buttons),
            "generation": session.source_generation,
            "warnings": list(source.warnings),
        }

    @app.get("/api/session/{session_id}/source")
    async def source_mesh(session_id: str) -> Response:
        """The imported model as authored, for the textured views.

        Served over HTTP rather than pushed down the socket: it is big, and
        only a viewport that shows the model textured needs it.  The header
        names the import's ``generation``, which every texture URL carries.
        """
        session, source = _require_source(sessions, session_id)
        header = {
            "name": source.name,
            "generation": session.source_generation,
            "n_vertices": int(source.vertices.shape[0]),
            "n_faces": int(source.faces.shape[0]),
            # (material, first face, face count) -- one draw call each.
            "groups": [list(group) for group in source.groups],
            **source.describe(),
        }
        frame = protocol.encode(
            MessageType.GEOMETRY,
            header,
            {
                "vertices": source.vertices,
                "normals": source.normals,
                "uv": source.uv,
                "faces": source.faces,
            },
        )
        # Never cached: the same URL describes whatever was imported last.
        return Response(
            content=frame,
            media_type="application/octet-stream",
            headers={"cache-control": "no-store"},
        )

    @app.get("/api/session/{session_id}/texture/{slot}/{material}")
    async def texture(
        session_id: str, slot: int, material: int, g: Optional[int] = None
    ) -> Response:
        """One material's map for one slot, as the bytes a browser decodes.

        ``g`` is the generation from /source.  With it the answer is cached
        for good; a generation that has since been replaced is a 404, never
        the new model's map under the old model's URL.
        """
        session, source = _require_source(sessions, session_id)
        if g is not None and g != session.source_generation:
            raise HTTPException(
                status_code=404, detail="that model has been replaced; read /source again"
            )
        image = await asyncio.to_thread(source.texture, int(slot), int(material))
        if image is None:
            raise HTTPException(
                status_code=404, detail="that material has no map in that slot"
            )
        return Response(
            content=image.data,
            media_type=image.mime,
            headers={"cache-control": TEXTURE_CACHE if g is not None else "no-store"},
        )

    @app.get("/api/session/{session_id}/export")
    async def download_export(session_id: str) -> FileResponse:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        path = session.export_path
        if path is None or not path.is_file():
            raise HTTPException(
                status_code=404, detail="this session has not exported a mesh yet"
            )
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.websocket("/ws/{session_id}")
    async def viewer_socket(websocket: WebSocket, session_id: str) -> None:
        session = sessions.get(session_id)
        if session is None:
            await websocket.close(code=4404, reason="unknown session")
            return
        await websocket.accept()
        await _Connection(websocket, session).run()

    return app


def run(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    app: Optional[FastAPI] = None,
) -> None:
    """Serve the app with uvicorn, optionally behind a public Gradio tunnel."""
    config = uvicorn.Config(
        app if app is not None else build_app(),
        host=host,
        port=port,
        ws="auto",
        ws_max_size=WS_MAX_SIZE,
        # float32 geometry barely compresses, so deflate would only burn
        # event-loop CPU and add latency per megabyte.
        ws_per_message_deflate=False,
    )
    asyncio.run(_serve(config, share))


async def _serve(config: uvicorn.Config, share: bool) -> None:
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve(), name="imb-uvicorn")
    if share:
        await _announce_share_url(server, config, serving)
    await serving


async def _announce_share_url(
    server: uvicorn.Server, config: uvicorn.Config, serving: asyncio.Task
) -> None:
    """Publish the local port through Gradio's tunnel once uvicorn is up."""
    from gradio import networking  # only needed for share=True

    while not server.started:
        if serving.done():
            return
        await asyncio.sleep(0.1)

    url = await asyncio.to_thread(
        networking.setup_tunnel,
        local_host=config.host,
        local_port=config.port,
        share_token=secrets.token_urlsafe(32),
        share_server_address=None,
        share_server_tls_certificate=None,
    )
    print(f"Public viewer URL: {url}/viewer")
