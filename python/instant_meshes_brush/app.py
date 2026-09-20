"""Gradio side panel for the browser-based Instant Meshes brushing UI.

The panel mirrors the native viewer's control column -- open a mesh, choose what
to remesh it as, solve the two fields, export -- while the 3D viewport itself is
an ``<iframe>`` pointing at the viewer document served by :mod:`.server`.  That
iframe owns its own binary WebSocket, so nothing here ever pushes geometry or
fields to the browser: every control below mutates the shared
:class:`~.session_manager.BrushSession` and the viewer picks the change up on
its next frame.  The session id travels to the viewer in the iframe URL, which
is the only piece of state the two halves share.

The panel therefore keeps no solver state of its own.  Every readout is sampled
from the session at the moment it is rendered; if a number is not in the C++
session, it is not shown.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Tuple
from urllib.parse import quote

import gradio as gr
import numpy as np

from .server import load_mesh_file
from .session_manager import (
    BrushSession,
    Extraction,
    Geometry,
    SessionRegistry,
    default_registry,
)

__all__ = ["ControlPanel", "PanelSettings", "REMESH_MODES", "VIEWER_ROUTE", "build_blocks"]

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

#: Route served by :func:`.server.build_app` that hosts the viewer document.
VIEWER_ROUTE = "/viewer"

_TITLE = "Instant Meshes - brush guided retopology"

#: How often a running solve is polled for its progress readout.
_SOLVE_POLL_SECONDS = 0.4

_MIN_VERTEX_COUNT = 50

#: ``Config`` sentinel that makes the C++ derive a value instead of using one.
_AUTO = -1

#: ``solve(level=...)`` value for the coarse-to-fine schedule that self-retires.
_HIERARCHICAL = -1

_ORIENTATIONS = "orientations"
_POSITIONS = "positions"

_UPLOAD_SUFFIXES = [".obj", ".ply", ".stl", ".off"]

#: Suffixes the C++ reader handles itself; the rest go through trimesh.
_NATIVE_SUFFIXES = frozenset({".obj", ".ply"})

_UNSOLVED = "not solved yet"

_NO_MESH = "_No mesh loaded._"

_NO_STROKES = "no strokes"

_VIEWPORT_BOX = (
    "width:100%;height:calc(100vh - 140px);min-height:520px;border:0;"
    "border-radius:var(--radius-lg);background:#1a1a1a"
)

_VIEWPORT_PLACEHOLDER = (
    f'<div style="{_VIEWPORT_BOX};display:flex;align-items:center;'
    'justify-content:center;color:#888">Starting session&hellip;</div>'
)


@dataclass(frozen=True)
class RemeshMode:
    """One entry of the native viewer's "Remesh as" combo box."""

    label: str
    rosy: int
    posy: int


REMESH_MODES: Tuple[RemeshMode, ...] = (
    RemeshMode("Triangles (6-RoSy, 3-PoSy)", 6, 3),
    RemeshMode("Quads (4-RoSy, 4-PoSy)", 4, 4),
    RemeshMode("Quads (2-RoSy, 4-PoSy)", 2, 4),
)

_DEFAULT_MODE_INDEX = 1


# ---------------------------------------------------------------------------
#  Panel state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelSettings:
    """The configuration controls, coerced out of their raw component values."""

    mode_index: int
    extrinsic: bool
    align_to_boundaries: bool
    sharp_creases: bool
    crease_angle: float
    deterministic: bool
    pure_quad: bool
    smooth_iter: int

    @property
    def mode(self) -> RemeshMode:
        return REMESH_MODES[self.mode_index]

    def to_values(self, vertex_count: int) -> Dict[str, Any]:
        """The config keys the panel owns; unknown ones would be rejected.

        ``scale`` and ``face_count`` are pinned to the auto sentinel so the
        target vertex count is what the resolution is derived from -- the C++
        prefers a positive scale over everything else, so a scale left behind by
        the viewer would otherwise silently win.
        """
        return {
            "rosy": self.mode.rosy,
            "posy": self.mode.posy,
            "scale": float(_AUTO),
            "face_count": _AUTO,
            "vertex_count": vertex_count,
            # A negative angle is how session.h spells "no crease detection".
            "crease_angle": self.crease_angle if self.sharp_creases else float(_AUTO),
            "extrinsic": self.extrinsic,
            "align_to_boundaries": self.align_to_boundaries,
            "deterministic": self.deterministic,
            "smooth_iter": self.smooth_iter,
            "pure_quad": self.pure_quad,
        }


@dataclass(frozen=True)
class PanelReadouts:
    """Everything the panel displays, sampled from the session in one visit."""

    mesh: str
    orientation: str
    position: str
    strokes: str
    target_vertex_count: int
    working_vertex_count: int


def _settings(
    mode_index: int,
    extrinsic: bool,
    align_to_boundaries: bool,
    sharp_creases: bool,
    crease_angle: float,
    deterministic: bool,
    pure_quad: bool,
    smooth_iter: float,
) -> PanelSettings:
    """Coerce the raw component values, which arrive as floats from sliders."""
    return PanelSettings(
        mode_index=int(mode_index),
        extrinsic=bool(extrinsic),
        align_to_boundaries=bool(align_to_boundaries),
        sharp_creases=bool(sharp_creases),
        crease_angle=float(crease_angle),
        deterministic=bool(deterministic),
        pure_quad=bool(pure_quad),
        smooth_iter=int(smooth_iter),
    )


# ---------------------------------------------------------------------------
#  Readouts
# ---------------------------------------------------------------------------


def _mesh_line(geometry: Geometry) -> str:
    return (
        f"**Working mesh** {geometry.vertices.shape[0]:,} vertices / "
        f"{geometry.faces.shape[0]:,} triangles  \n"
        f"**Target** {int(geometry.config['vertex_count']):,} vertices, "
        f"edge length {geometry.scale:.4g}"
    )


def _singularity_line(count: int, version: int) -> str:
    """A negative version counter is how session.h marks an unpublished field."""
    return _UNSOLVED if version < 0 else f"{count:,} singularities"


async def _singularity_lines(session: BrushSession) -> Tuple[str, str]:
    """Both singularity readouts from a single visit to the solver.

    One ``Singularities`` carries the markers for both fields, so asking per
    field would run the detector twice over and let the two readouts straddle a
    solver update.
    """
    state = await session.status()
    markers = await session.singularities()
    return (
        _singularity_line(markers.orientation_positions.shape[0], state.iterations_q),
        _singularity_line(markers.position_positions.shape[0], state.iterations_o),
    )


def _stroke_line(count: int) -> str:
    return _NO_STROKES if count == 0 else f"{count:,} stroke(s)"


def _export_line(extraction: Extraction, path: Path) -> str:
    faces = extraction.faces
    summary = (
        f"Exported {extraction.vertices.shape[0]:,} vertices / {faces.shape[0]:,} faces"
    )
    if faces.shape[1] == 4:
        # A quad whose last two indices coincide is really a triangle.
        degenerate = int(np.count_nonzero(faces[:, 2] == faces[:, 3]))
        if degenerate:
            summary += f" ({degenerate:,} of them triangles)"
    return f"{summary} to `{path.name}`."


async def _sample(session: BrushSession, geometry: Geometry) -> PanelReadouts:
    """Take one consistent sample of everything the panel shows."""
    orientation, position = await _singularity_lines(session)
    return PanelReadouts(
        mesh=_mesh_line(geometry),
        orientation=orientation,
        position=position,
        strokes=_stroke_line(len(await session.strokes())),
        target_vertex_count=int(geometry.config["vertex_count"]),
        working_vertex_count=int(geometry.vertices.shape[0]),
    )


def _vertex_count_slider(readouts: PanelReadouts) -> gr.Slider:
    """Re-range the target slider around the resolution the C++ just chose.

    Gradio rejects a value outside a slider's bounds on its way back in, so a
    mesh small enough that the native default (input vertices / 16) falls under
    the usual floor has to move the floor rather than clamp the value -- every
    configuration control feeds this slider back and would otherwise fail.
    """
    target = max(1, readouts.target_vertex_count)
    return gr.Slider(
        minimum=min(_MIN_VERTEX_COUNT, target),
        maximum=max(readouts.working_vertex_count, _MIN_VERTEX_COUNT * 2),
        value=target,
    )


def _viewer_iframe(session_id: str) -> str:
    """The viewport, bound to a session through its URL.

    The id is percent-encoded rather than trusted: it is the one value that
    travels from the registry straight into markup.
    """
    source = f"{VIEWER_ROUTE}?session={quote(session_id, safe='')}"
    return (
        f'<iframe src="{source}" title="Instant Meshes viewport" '
        f'style="{_VIEWPORT_BOX};display:block" allow="fullscreen"></iframe>'
    )


@contextlib.contextmanager
def _reported() -> Iterator[None]:
    """Turn a session or solver failure into the toast Gradio shows the user.

    ``SessionError`` derives from ``RuntimeError``, and so does everything
    pybind11 raises for the core's ``std::runtime_error`` -- an unreadable OBJ
    or a mesh the preprocessor rejects reaches the panel as a plain
    ``RuntimeError`` and would otherwise escape as a bare traceback.
    ``gr.Error`` is a ``ValueError``, so re-raising here cannot wrap itself.
    """
    try:
        yield
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc


# ---------------------------------------------------------------------------
#  Handlers
# ---------------------------------------------------------------------------


class ControlPanel:
    """Event handlers for one Blocks graph.

    An instance owns nothing but the registry.  The mesh, the fields, the
    strokes and the solver's progress all live in the session, so handlers read
    them back out rather than caching them.
    """

    def __init__(self, registry: Optional[SessionRegistry] = None) -> None:
        self._registry = registry

    @property
    def registry(self) -> SessionRegistry:
        """The shared registry, resolved late so a test can install its own."""
        if self._registry is None:
            self._registry = default_registry()
        return self._registry

    def _session(self, session_id: Optional[str]) -> BrushSession:
        if not session_id:
            raise gr.Error("The page has no session yet; reload it.")
        session = self.registry.get(session_id)
        if session is None:
            raise gr.Error("This session has expired; reload the page.")
        return session

    def _ready_session(self, session_id: Optional[str]) -> BrushSession:
        session = self._session(session_id)
        if not session.ready:
            raise gr.Error("Open a mesh first.")
        return session

    # -- session lifecycle ---------------------------------------------------

    async def open_session(self) -> Tuple[str, str]:
        """Mint a session on page load and point the viewport iframe at it."""
        with _reported():
            session = await self.registry.create()
        return session.id, _viewer_iframe(session.id)

    # -- input ---------------------------------------------------------------

    async def load_mesh(
        self,
        session_id: str,
        mesh_path: str,
        mode_index: int,
        extrinsic: bool,
        align_to_boundaries: bool,
        sharp_creases: bool,
        crease_angle: float,
        deterministic: bool,
        pure_quad: bool,
        smooth_iter: float,
    ) -> Tuple[str, str, str, str, gr.Slider]:
        """Load an upload and preprocess it at the resolution the C++ picks."""
        session = self._session(session_id)
        path = Path(mesh_path)
        values = _settings(
            mode_index,
            extrinsic,
            align_to_boundaries,
            sharp_creases,
            crease_angle,
            deterministic,
            pure_quad,
            smooth_iter,
        ).to_values(_AUTO)

        with _reported():
            if path.suffix.lower() in _NATIVE_SUFFIXES:
                geometry = await session.load_file(path, values)
            else:
                # STL and OFF are triangle soups the C++ reader does not know;
                # trimesh also merges their duplicate vertices, without which
                # the mesh has no connectivity at all.
                vertices, faces = await asyncio.to_thread(load_mesh_file, path)
                geometry = await session.load_mesh(vertices, faces, path.name, values)
            readouts = await _sample(session, geometry)

        return (
            readouts.mesh,
            readouts.orientation,
            readouts.position,
            readouts.strokes,
            _vertex_count_slider(readouts),
        )

    async def apply_config(
        self,
        session_id: str,
        mode_index: int,
        vertex_count: float,
        extrinsic: bool,
        align_to_boundaries: bool,
        sharp_creases: bool,
        crease_angle: float,
        deterministic: bool,
        pure_quad: bool,
        smooth_iter: float,
    ) -> Tuple[str, str, str, str]:
        """Re-target the resolution or the field options of the loaded mesh.

        Re-running preprocess drops the strokes and resets both fields, because
        the vertex numbering changes underneath them -- so every configuration
        change sends the user back to the Solve buttons.
        """
        session = self._session(session_id)
        if not session.ready:
            return _NO_MESH, _UNSOLVED, _UNSOLVED, _NO_STROKES

        values = _settings(
            mode_index,
            extrinsic,
            align_to_boundaries,
            sharp_creases,
            crease_angle,
            deterministic,
            pure_quad,
            smooth_iter,
        ).to_values(int(vertex_count))

        with _reported():
            geometry = await session.set_config(values)
            readouts = await _sample(session, geometry)
        return readouts.mesh, readouts.orientation, readouts.position, readouts.strokes

    # -- solving -------------------------------------------------------------

    async def solve_orientation_field(self, session_id: str) -> AsyncIterator[str]:
        """Run the hierarchical orientation schedule, streaming its progress."""
        async for line in self._solve(session_id, _ORIENTATIONS):
            yield line

    async def solve_position_field(self, session_id: str) -> AsyncIterator[str]:
        """Run the hierarchical position schedule, streaming its progress."""
        async for line in self._solve(session_id, _POSITIONS):
            yield line

    async def _solve(self, session_id: str, field: str) -> AsyncIterator[str]:
        """Start a solve, report it until it retires, then count the defects.

        ``solving`` rather than ``SolveState.active`` drives the loop: the C++
        optimizer goes briefly idle between sweeps, and a readout bound to that
        would flicker back to a stale count mid-solve.
        """
        session = self._ready_session(session_id)
        with _reported():
            await session.solve(field, level=_HIERARCHICAL)
            while session.solving:
                state = await session.status()
                yield f"solving - level {state.level}, {state.progress * 100:.0f}%"
                await asyncio.sleep(_SOLVE_POLL_SECONDS)
            orientation, position = await _singularity_lines(session)
            yield orientation if field == _ORIENTATIONS else position

    # -- strokes -------------------------------------------------------------

    async def clear_strokes(self, session_id: str) -> str:
        """Drop every brush constraint; the fields keep whatever they hold now.

        The session only re-applies the constraints, so the freed field does not
        relax back until the next solve.
        """
        session = self._ready_session(session_id)
        with _reported():
            await session.clear_strokes()
            return _stroke_line(len(await session.strokes()))

    # -- output --------------------------------------------------------------

    async def export_mesh(self, session_id: str, export_format: str) -> Tuple[str, str]:
        """Extract the quad/triangle mesh and offer it as a download.

        Extraction is explicit rather than left to ``export_mesh``, which would
        happily reuse a result from before the last solve.
        """
        session = self._ready_session(session_id)
        state = await session.status()
        if state.iterations_q < 0 or state.iterations_o < 0:
            raise gr.Error("Solve the orientation and position fields before exporting.")

        with _reported():
            extraction = await session.extract()
            path = await session.export_mesh(export_format)
        return str(path), _export_line(extraction, path)


# ---------------------------------------------------------------------------
#  Layout
# ---------------------------------------------------------------------------


def build_blocks(registry: Optional[SessionRegistry] = None) -> gr.Blocks:
    """Assemble the control column and the viewport iframe.

    ``registry`` is an injection point for tests; at runtime the process-wide
    registry is resolved the first time a handler needs it, so that this shares
    sessions with the WebSocket app built by :func:`.server.build_app`.
    """
    panel = ControlPanel(registry)

    # Gradio's telemetry starts two non-daemon threads that each make an
    # outbound request; this tool is meant to run offline next to its vendored
    # three.js, where they only delay interpreter shutdown.
    with gr.Blocks(title=_TITLE, fill_height=True, analytics_enabled=False) as blocks:
        session_state = gr.State()

        with gr.Row():
            with gr.Column(scale=1, min_width=330):
                mesh_file = gr.File(
                    label="Open mesh",
                    file_types=_UPLOAD_SUFFIXES,
                    type="filepath",
                    height=100,
                )
                mesh_info = gr.Markdown(_NO_MESH)

                remesh_as = gr.Dropdown(
                    choices=[mode.label for mode in REMESH_MODES],
                    value=REMESH_MODES[_DEFAULT_MODE_INDEX].label,
                    type="index",
                    label="Remesh as",
                    filterable=False,
                )

                with gr.Accordion("Configuration details", open=False):
                    extrinsic = gr.Checkbox(True, label="Extrinsic smoothing")
                    align_to_boundaries = gr.Checkbox(False, label="Align to boundaries")
                    sharp_creases = gr.Checkbox(False, label="Sharp creases")
                    crease_angle = gr.Slider(
                        0,
                        90,
                        value=30,
                        step=1,
                        label="Crease angle (degrees)",
                        info="Only used when sharp creases are enabled.",
                    )
                    deterministic = gr.Checkbox(
                        False, label="Deterministic (reproducible, slower)"
                    )

                vertex_count = gr.Slider(
                    _MIN_VERTEX_COUNT,
                    _MIN_VERTEX_COUNT * 2,
                    value=_MIN_VERTEX_COUNT,
                    step=1,
                    precision=0,
                    label="Target vertex count",
                )

                with gr.Group():
                    gr.Markdown("#### Orientation field")
                    orientation_solve = gr.Button("Solve", variant="primary")
                    orientation_info = gr.Markdown(_UNSOLVED)

                with gr.Group():
                    gr.Markdown("#### Position field")
                    position_solve = gr.Button("Solve", variant="primary")
                    position_info = gr.Markdown(_UNSOLVED)

                with gr.Group():
                    gr.Markdown("#### Brush strokes\n\nDrawn in the viewport.")
                    clear_strokes_button = gr.Button("Clear strokes")
                    stroke_info = gr.Markdown(_NO_STROKES)

                with gr.Accordion("Export mesh", open=True):
                    export_format = gr.Dropdown(
                        choices=["obj", "ply"],
                        value="obj",
                        label="Format",
                        filterable=False,
                    )
                    pure_quad = gr.Checkbox(False, label="Pure quad mesh")
                    smooth_iter = gr.Slider(
                        0, 10, value=2, step=1, precision=0, label="Smoothing iterations"
                    )
                    export_button = gr.Button("Export mesh", variant="primary")
                    export_file = gr.File(label="Download", interactive=False)
                    export_info = gr.Markdown("")

            with gr.Column(scale=3, min_width=480):
                viewport = gr.HTML(
                    _VIEWPORT_PLACEHOLDER,
                    js_on_load=None,
                    container=False,
                    padding=False,
                )

        # `pure_quad` and `smooth_iter` only reach the extractor through Config,
        # which the C++ freezes at preprocess time -- so they rebuild like the
        # rest rather than being read at export.  Both lists below are
        # positional and must stay in step with the handler signatures.
        field_options = [
            extrinsic,
            align_to_boundaries,
            sharp_creases,
            crease_angle,
            deterministic,
            pure_quad,
            smooth_iter,
        ]
        config_inputs = [session_state, remesh_as, vertex_count, *field_options]
        config_outputs = [mesh_info, orientation_info, position_info, stroke_info]

        # `concurrency_limit=None` throughout: Gradio otherwise runs one copy of
        # an event at a time *across every browser*, so a multi-minute solve in
        # one tab would freeze the controls of all the others.  Nothing here
        # blocks the event loop -- each session owns the worker thread its C++
        # calls run on -- and `trigger_mode="once"` still keeps a single tab
        # from stacking up runs of the same button.
        blocks.load(
            panel.open_session,
            outputs=[session_state, viewport],
            concurrency_limit=None,
        )

        mesh_file.upload(
            panel.load_mesh,
            inputs=[session_state, mesh_file, remesh_as, *field_options],
            outputs=[*config_outputs, vertex_count],
            concurrency_limit=None,
        )

        for toggle in (
            remesh_as,
            extrinsic,
            align_to_boundaries,
            sharp_creases,
            deterministic,
            pure_quad,
        ):
            toggle.change(
                panel.apply_config,
                inputs=config_inputs,
                outputs=config_outputs,
                concurrency_limit=None,
            )

        # Sliders fire `change` on every pixel of a drag; `release` rebuilds once.
        for slider in (vertex_count, crease_angle, smooth_iter):
            slider.release(
                panel.apply_config,
                inputs=config_inputs,
                outputs=config_outputs,
                concurrency_limit=None,
            )

        orientation_solve.click(
            panel.solve_orientation_field,
            inputs=session_state,
            outputs=orientation_info,
            concurrency_limit=None,
        )
        position_solve.click(
            panel.solve_position_field,
            inputs=session_state,
            outputs=position_info,
            concurrency_limit=None,
        )
        clear_strokes_button.click(
            panel.clear_strokes,
            inputs=session_state,
            outputs=stroke_info,
            concurrency_limit=None,
        )

        export_button.click(
            panel.export_mesh,
            inputs=[session_state, export_format],
            outputs=[export_file, export_info],
            concurrency_limit=None,
        )

    return blocks
