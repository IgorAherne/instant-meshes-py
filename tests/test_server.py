"""WebSocket transport tests, driven through Starlette's in-process test client.

The headline case is the one that is easy to miss: a session is shared between
the Gradio control panel and the browser viewport, and the two reach it by
different routes. A mesh loaded from the panel has to arrive at a viewport that
never asked for it, which only works because the streamer watches the session's
geometry version the way it watches the solver's field counters.

Starlette's test websocket has no receive timeout, so nothing here ever reads
speculatively: every read is bounded by a PING whose PONG marks the end of the
frames the server had to send. A regression therefore fails the test instead of
hanging the suite.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List

import numpy as np
import pytest

from instant_meshes_brush import protocol, uv
from instant_meshes_brush.protocol import MessageType
from instant_meshes_brush.server import STATIC_URL, build_app
from instant_meshes_brush import session_manager
from instant_meshes_brush.session_manager import SessionRegistry

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


@pytest.fixture
def registry() -> SessionRegistry:
    # These tests finish in well under a second each, so only the capacity is
    # worth pinning; the default TTL is nowhere near short enough to interfere.
    return SessionRegistry(max_sessions=4)


@pytest.fixture
def client(registry: SessionRegistry):
    with TestClient(build_app(registry)) as test_client:
        yield test_client


class Socket:
    """A test websocket that reads in bounded rounds."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self._token = 0

    def send(self, msg_type: int, header=None, arrays=None) -> None:
        self.ws.send_bytes(protocol.encode(msg_type, header or {}, arrays))

    def drain(self) -> List[protocol.Message]:
        """Every frame the server owes us right now, and no more.

        The reader is serial, so a PONG cannot overtake replies to frames sent
        before its PING -- which is what makes this terminate.
        """
        self._token += 1
        token = self._token
        self.send(MessageType.PING, {"t": token})

        frames: List[protocol.Message] = []
        for _ in range(500):
            message = protocol.decode(self.ws.receive_bytes())
            if message.type == MessageType.PONG and message.get("t") == token:
                return frames
            frames.append(message)
        raise AssertionError("the server never answered PING")

    def expect(self, wanted: int, rounds: int = 60, pause: float = 0.05) -> protocol.Message:
        """Drain repeatedly until a frame of type `wanted` shows up."""
        errors = []
        for attempt in range(rounds):
            for message in self.drain():
                if message.type == wanted:
                    return message
                if message.type == MessageType.ERROR:
                    errors.append(message.get("message"))
            if attempt:
                time.sleep(pause)
        raise AssertionError(
            f"never saw type {wanted}" + (f"; errors: {errors}" if errors else "")
        )

    def expect_all(
        self, *wanted: int, rounds: int = 60, pause: float = 0.05
    ) -> List[protocol.Message]:
        """Every listed frame type, collected across as many rounds as it takes.

        ``expect`` returns the moment it matches and drops the rest of that
        round with it, so two replies to one request can never both be caught
        with it -- whichever is second is thrown away by the first call.
        """
        seen: Dict[int, protocol.Message] = {}
        errors: List[str] = []
        for attempt in range(rounds):
            for message in self.drain():
                seen.setdefault(message.type, message)
                if message.type == MessageType.ERROR:
                    errors.append(message.get("message"))
            if all(kind in seen for kind in wanted):
                return [seen[kind] for kind in wanted]
            if attempt:
                time.sleep(pause)
        missing = [kind for kind in wanted if kind not in seen]
        raise AssertionError(
            f"never saw type(s) {missing}" + (f"; errors: {errors}" if errors else "")
        )

    def count(self, wanted: int) -> int:
        return sum(1 for message in self.drain() if message.type == wanted)


def open_session(client) -> str:
    response = client.post("/api/session")
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def load_mesh(socket: Socket, torus, vertex_count: int = 150) -> protocol.Message:
    socket.send(
        MessageType.LOAD_MESH,
        {"name": "torus.obj",
         "config": {"vertex_count": vertex_count, "deterministic": True}},
        {"vertices": np.asarray(torus.vertices), "faces": np.asarray(torus.faces)},
    )
    return socket.expect(MessageType.GEOMETRY)


def equator_rays(torus, samples: int = 16, span: float = 0.6):
    """Rays aimed straight at the torus's outer equator."""
    angles = np.linspace(-span / 2, span / 2, samples)
    points, normals = torus.surface(angles, np.zeros_like(angles))
    origins = (points + normals * 3.0).astype(np.float32)
    return origins, (points - origins).astype(np.float32)


# ---------------------------------------------------------------------------
#  HTTP surface
# ---------------------------------------------------------------------------


def test_viewer_route_and_static_assets_are_served(client) -> None:
    page = client.get("/viewer")
    assert page.status_code == 200
    assert "<canvas" in page.text

    for asset in ("main.js", "protocol.js", "renderer.js", "field_material.js",
                  "tools.js", "net.js", "panel.js", "style.css",
                  "vendor/three.module.js", "vendor/OrbitControls.js"):
        response = client.get(f"{STATIC_URL}/{asset}")
        assert response.status_code == 200, asset
        assert response.content, asset

    # index.html hard-codes the mount, so the two have to agree. Most assets
    # are reached through ES imports rather than named here; these are the ones
    # the document asks for itself, and a mismatch is a blank viewer.
    for url in re.findall(r'(?:src|href)="(/[^"]+)"', page.text):
        assert client.get(url).status_code == 200, url


def test_the_asset_mount_leaves_gradios_alone(client) -> None:
    """/static belongs to Gradio, which serves its fonts from there.

    These routes are matched before the Gradio mount, so taking that prefix
    would 404 the host application's own assets -- which is exactly what it did
    until the viewer's moved to its own.
    """
    assert STATIC_URL != "/static"
    assert client.get("/static/style.css").status_code == 404


def test_the_file_pickers_filter_matches_what_the_route_accepts() -> None:
    """The picker's list is a copy, and a copy can drift.

    A format missing from it cannot be chosen at all; one that is there but
    not accepted is a file chosen and then refused, which is worse.
    """
    from instant_meshes_brush.server import MESH_SUFFIXES, STATIC_DIR

    panel = (STATIC_DIR / "panel.js").read_text(encoding="utf-8")
    accept = re.search(r"MESH_ACCEPT = '([^']+)'", panel)
    assert accept, "panel.js no longer declares MESH_ACCEPT"
    assert set(accept.group(1).split(",")) == set(MESH_SUFFIXES)


def test_unknown_session_is_refused(client) -> None:
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/not-a-real-session"):
            pass


# ---------------------------------------------------------------------------
#  Geometry propagation
# ---------------------------------------------------------------------------


def test_load_mesh_answers_with_geometry(client, torus) -> None:
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        geometry = load_mesh(Socket(ws), torus)

    assert geometry.get("name") == "torus.obj"
    assert geometry.get("scale") > 0
    vertices = geometry.arrays["vertices"]
    faces = geometry.arrays["faces"]
    assert vertices.shape[1] == 3 and faces.shape[1] == 3
    assert geometry.arrays["normals"].shape == vertices.shape
    assert faces.max() < len(vertices)
    assert geometry.get("n_vertices") == len(vertices)


def test_a_mesh_loaded_elsewhere_reaches_an_attached_viewport(
    client, registry: SessionRegistry, torus
) -> None:
    """The regression for the control panel and the viewport being out of sync.

    This socket never sends LOAD_MESH. The mesh is pushed into the session the
    way the Gradio panel does it, and the streamer has to notice on its own.
    """
    session_id = open_session(client)
    with client.websocket_connect(f"/ws/{session_id}") as ws:
        socket = Socket(ws)
        socket.send(MessageType.SUBSCRIBE, {"fps": 60})
        socket.expect(MessageType.STATUS)

        session = registry.get(session_id)
        assert session is not None
        client.portal.call(
            session.load_mesh,
            np.asarray(torus.vertices),
            np.asarray(torus.faces),
            "from-the-panel.obj",
            {"vertex_count": 150, "deterministic": True},
        )

        geometry = socket.expect(MessageType.GEOMETRY)

    assert geometry.get("name") == "from-the-panel.obj"
    assert len(geometry.arrays["vertices"]) == geometry.get("n_vertices")


def test_uploaded_file_reaches_an_attached_viewport(client, torus, tmp_path) -> None:
    """The viewport's own Open button posts a file; the socket must see it."""
    source = tmp_path / "torus.obj"
    with source.open("w") as handle:
        for v in torus.vertices:
            handle.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in torus.faces:
            handle.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")

    session_id = open_session(client)
    with client.websocket_connect(f"/ws/{session_id}") as ws:
        socket = Socket(ws)
        socket.send(MessageType.SUBSCRIBE, {"fps": 60})
        socket.expect(MessageType.STATUS)

        response = client.post(
            f"/api/session/{session_id}/mesh",
            files={"file": ("torus.obj", source.read_bytes(), "text/plain")},
            data={"config": '{"vertex_count": 150, "deterministic": true}'},
        )
        assert response.status_code == 200, response.text
        assert response.json()["name"] == "torus.obj"

        geometry = socket.expect(MessageType.GEOMETRY)

    assert geometry.get("name") == "torus.obj"
    assert geometry.arrays["vertices"].shape[1] == 3


def test_upload_rejects_an_unsupported_format(client) -> None:
    session_id = open_session(client)
    response = client.post(
        f"/api/session/{session_id}/mesh",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415
    assert "unsupported mesh format" in response.json()["detail"]


def test_upload_to_an_unknown_session_is_404(client) -> None:
    response = client.post(
        "/api/session/nope/mesh",
        files={"file": ("a.obj", b"v 0 0 0\n", "text/plain")},
    )
    assert response.status_code == 404


def test_geometry_is_not_resent_once_the_socket_has_it(client, torus) -> None:
    """The streamer must not repeat a frame the handler already answered with."""
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        socket.send(MessageType.SUBSCRIBE, {"fps": 60})
        socket.expect(MessageType.STATUS)
        load_mesh(socket, torus)

        # Give the streamer plenty of ticks in which to misbehave.
        time.sleep(0.5)
        repeats = socket.count(MessageType.GEOMETRY)

    assert repeats == 0, "the streamer resent geometry the client already had"


# ---------------------------------------------------------------------------
#  Editing
# ---------------------------------------------------------------------------


def test_solve_streams_a_field(client, torus) -> None:
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)
        socket.send(MessageType.SOLVE, {"field": "both"})
        field = socket.expect(MessageType.FIELD)

    orientation = field.arrays["orientation"]
    assert orientation.shape == field.arrays["position"].shape
    assert orientation.shape[1] == 3
    lengths = np.linalg.norm(orientation, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-3)


def test_stroke_round_trip(client, torus) -> None:
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)
        socket.send(MessageType.SOLVE, {"field": "both"})
        socket.expect(MessageType.FIELD)

        origins, directions = equator_rays(torus)
        socket.send(
            MessageType.STROKE,
            {"kind": 0, "solve": False},
            {"ray_origins": origins, "ray_directions": directions},
        )
        stroke = socket.expect(MessageType.STROKE_RESULT)

    assert stroke.get("n_points") >= 2
    assert stroke.arrays["positions"].shape[1] == 3
    assert stroke.arrays["normals"].shape == stroke.arrays["positions"].shape
    assert len(stroke.arrays["faces"]) == stroke.get("n_points")


def test_a_stroke_that_misses_is_reported_and_the_socket_survives(client, torus) -> None:
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)

        away = np.tile(np.array([0.0, 0.0, 50.0], np.float32), (4, 1))
        up = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (4, 1))
        socket.send(
            MessageType.STROKE,
            {"kind": 0, "solve": False},
            {"ray_origins": away, "ray_directions": up},
        )
        replies = socket.drain()
        assert replies, "a stroke that misses must be answered, not ignored"

        # Whatever it answered, the socket has to stay usable: drain() only
        # returns when its own PING comes back.
        assert socket.drain() == []


def test_clearing_strokes_re_solves_without_them(client, torus) -> None:
    """The regression for a field that kept the shape a deleted stroke gave it.

    There is no Solve button left, so nothing can correct this afterwards: if
    removing a constraint does not re-solve, the stroke's influence is simply
    permanent.
    """
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)
        socket.send(MessageType.SOLVE, {"field": "both"})
        socket.expect(MessageType.FIELD)

        origins, directions = equator_rays(torus)
        socket.send(
            MessageType.STROKE,
            {"kind": 0, "solve": True},
            {"ray_origins": origins, "ray_directions": directions},
        )
        socket.expect(MessageType.STROKE_RESULT)
        while (combed := socket.expect(MessageType.STATUS)).get("active"):
            time.sleep(0.05)
        version = (combed.get("iterations_q"), combed.get("iterations_o"))

        socket.send(MessageType.CLEAR_STROKES, {})
        strokes = socket.expect(MessageType.STROKE_LIST)
        assert strokes.get("count") == 0

        # The solve started by the clear has to move the counters; the field
        # sent afterwards is the one computed without the stroke.
        for attempt in range(80):
            state = socket.expect(MessageType.STATUS)
            if (state.get("iterations_q"), state.get("iterations_o")) != version:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("clearing the strokes did not re-solve the field")


def test_a_click_that_erases_nothing_does_not_re_solve(client, torus) -> None:
    """A miss must not restart the solve the user is watching."""
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)
        socket.send(MessageType.SOLVE, {"field": "both"})
        socket.expect(MessageType.FIELD)
        while socket.expect(MessageType.STATUS).get("active"):
            time.sleep(0.05)

        socket.send(
            MessageType.ERASE_STROKE,
            {"point": [50.0, 50.0, 50.0], "eye": [0.0, 0.0, 9.0], "radius": 0.01},
        )
        socket.expect(MessageType.STROKE_LIST)
        time.sleep(0.3)

        assert not any(
            message.type == MessageType.STATUS and message.get("active")
            for message in socket.drain()
        ), "erasing nothing restarted the solver"


@pytest.fixture
def wide_phase_gap(monkeypatch):
    """Stretch the pause the solve driver leaves between its two phases.

    It polls every 50 ms in production while the stream ticks every 16, so
    whether a given seam is sampled is close to a coin toss. Widening it turns
    a race that a test would catch half the time into one it catches always.
    """
    monkeypatch.setattr(session_manager, "_SOLVE_POLL_SECONDS", 0.4)


def test_singularities_are_published_once_a_solve_has_finished(
    client, torus, wide_phase_gap
) -> None:
    """The regression for a red flash of defects that do not exist.

    A "both" solve goes idle in the gap between its two phases -- the driver
    polls every 50 ms, so the window is real and repeatable. Publishing markers
    there crosses a freshly solved orientation field with a position field that
    has not caught up, which reports hundreds of singularities that vanish a
    moment later: a patch of red flickering on the model after every stroke.

    The stream is subscribed at its maximum rate so that window is sampled;
    at the default 15 Hz the whole solve can pass between two ticks.
    """
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        socket.send(MessageType.SUBSCRIBE, {"fps": 60})
        load_mesh(socket, torus, vertex_count=2000)
        socket.send(MessageType.SOLVE, {"field": "both"})
        # Markers are the "it has all stopped" signal, so waiting for one is
        # waiting for the solve -- and needs no assumption about frame order.
        socket.expect(MessageType.SINGULARITIES, rounds=400, pause=0.02)
        socket.drain()

        origins, directions = equator_rays(torus)
        socket.send(
            MessageType.STROKE,
            {"kind": 0, "solve": True},
            {"ray_origins": origins, "ray_directions": directions},
        )

        settled = False
        marker_sets = []
        for attempt in range(400):
            for message in socket.drain():
                if message.type == MessageType.SINGULARITIES:
                    marker_sets.append(
                        (message.get("n_orientation"), message.get("n_position"))
                    )
                elif message.type == MessageType.STATUS:
                    if not message.get("solving") and not message.get("active"):
                        settled = True
            if settled and marker_sets:
                break
            time.sleep(0.02)

    assert marker_sets, "the finished solve never published its singularities"
    # Every set published for one solve has to be the settled one; a set from
    # the seam differs, and wildly. That is the whole symptom.
    assert len(set(marker_sets)) == 1, f"markers changed mid-solve: {marker_sets}"


def test_a_solve_invalidates_the_extraction(client, registry, torus) -> None:
    """Brushing moves the field, so the mesh extracted from it is stale.

    Nothing in the viewport re-extracts on its own any more -- the Extract
    button is gone -- so holding on to it would mean an export after a stroke
    silently writing the mesh from before it.
    """
    session_id = open_session(client)
    with client.websocket_connect(f"/ws/{session_id}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)
        socket.send(MessageType.SOLVE, {"field": "both"})
        socket.expect(MessageType.FIELD)

        socket.send(MessageType.EXTRACT, {})
        socket.expect(MessageType.EXTRACTED)

        session = registry.get(session_id)
        assert session is not None and session.has_extraction

        socket.send(MessageType.SOLVE, {"field": "both"})
        socket.expect(MessageType.STATUS)

    assert not session.has_extraction, "the solve kept a result built from the old field"


def test_extract_solves_a_field_that_never_was(client, torus) -> None:
    """Without a Solve button, extraction has to guarantee its own input.

    Extracting an unsolved hierarchy reads the solver's random initial state,
    which produces a mesh that looks plausible and means nothing.
    """
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)  # deliberately no SOLVE

        socket.send(MessageType.EXTRACT, {})
        extracted = socket.expect(MessageType.EXTRACTED)
        field = socket.expect(MessageType.FIELD)

    assert extracted.get("n_faces") > 0
    assert field.get("iterations_q") >= 0, "the field was never solved"


def test_retargeting_keeps_the_strokes_and_resends_their_curves(client, torus) -> None:
    """A rebuild carries the strokes, and the viewport has to be told.

    The curves are re-projected onto the new mesh, so the ones the client is
    drawing describe a surface that no longer exists. Without the replacements
    the strokes would vanish from the viewport while still steering the field
    -- a flow with no visible reason for its shape.
    """
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)
        socket.send(MessageType.SOLVE, {"field": "both"})
        socket.expect(MessageType.FIELD)

        origins, directions = equator_rays(torus)
        socket.send(
            MessageType.STROKE,
            {"kind": 0, "solve": False},
            {"ray_origins": origins, "ray_directions": directions},
        )
        drawn = socket.expect(MessageType.STROKE_RESULT)

        # One frame provokes all three replies, so they are collected together:
        # expect() discards the rest of the round it found its match in.
        socket.send(MessageType.SET_CONFIG, {"config": {"vertex_count": 600}})
        seen = {}
        for attempt in range(60):
            for reply in socket.drain():
                seen.setdefault(reply.type, reply)
            if MessageType.STROKE_LIST in seen:
                break
            time.sleep(0.05)

    assert MessageType.GEOMETRY in seen, "the rebuild never answered with a mesh"
    carried = seen.get(MessageType.STROKE_RESULT)
    listed = seen[MessageType.STROKE_LIST]
    assert listed.get("count") == 1, "the rebuild dropped the stroke"
    assert carried is not None, "the carried stroke's new curve was never sent"
    assert carried.get("stroke_id") == drawn.get("stroke_id")
    assert carried.arrays["positions"].shape[1] == 3
    # Same path, re-projected: every carried point sits on the drawn curve.
    before = drawn.arrays["positions"].astype(np.float64)
    after = carried.arrays["positions"].astype(np.float64)
    strayed = np.linalg.norm(after[:, None] - before[None], axis=2).min(1).max()
    assert strayed < 0.05, f"the carried stroke moved by {strayed:.4f}"


def test_extract_and_export(client, torus) -> None:
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)
        socket.send(MessageType.SOLVE, {"field": "both"})
        socket.expect(MessageType.FIELD)

        socket.send(MessageType.EXTRACT, {})
        extracted = socket.expect(MessageType.EXTRACTED)
        assert extracted.arrays["faces"].shape[1] == extracted.get("posy")
        assert extracted.arrays["wireframe"].shape[1] == 3
        assert (extracted.arrays["wireframe"].shape
                == extracted.arrays["wireframe_color"].shape)

        socket.send(MessageType.EXPORT, {"format": "obj"})
        ready = socket.expect(MessageType.EXPORT_READY)

    assert ready.get("bytes") > 0
    download = client.get(ready.get("url"))
    assert download.status_code == 200
    assert download.content.startswith(b"v ")


def test_export_without_extract_shows_what_it_wrote(client, torus) -> None:
    """Export extracts on demand, and must publish what it extracted.

    Otherwise the panel keeps reading "not extracted yet" beside a download
    link for a file the viewport is not showing.
    """
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)
        socket.send(MessageType.SOLVE, {"field": "both"})
        socket.expect(MessageType.FIELD)

        # Both replies to one frame, so they have to be collected together:
        # expect() discards the rest of the round it found its match in.
        socket.send(MessageType.EXPORT, {"format": "obj"})
        seen = {}
        for attempt in range(60):
            for reply in socket.drain():
                seen.setdefault(reply.type, reply)
            if MessageType.EXPORT_READY in seen:
                break
            time.sleep(0.05)

    assert MessageType.EXTRACTED in seen, "export wrote a mesh it never showed"
    assert seen[MessageType.EXTRACTED].get("n_faces") > 0
    assert seen[MessageType.EXPORT_READY].get("bytes") > 0


def test_extraction_options_do_not_rebuild_the_mesh(client, registry, torus) -> None:
    """Pure quad and smoothing ride with EXTRACT, not with SET_CONFIG.

    The panel has no Apply button any more, so anything routed through
    SET_CONFIG re-runs preprocess -- which drops every stroke. These two are
    read at extraction time, so they must not go that way: the check is that
    the geometry version, which preprocess bumps, does not move.
    """
    session_id = open_session(client)
    with client.websocket_connect(f"/ws/{session_id}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)
        socket.send(MessageType.SOLVE, {"field": "both"})
        socket.expect(MessageType.FIELD)

        session = registry.get(session_id)
        assert session is not None
        version = session.geometry_version

        socket.send(MessageType.EXTRACT, {"pure_quad": False, "smooth_iter": 0})
        mixed = socket.expect(MessageType.EXTRACTED)

        socket.send(MessageType.EXTRACT, {"pure_quad": True, "smooth_iter": 3})
        pure = socket.expect(MessageType.EXTRACTED)

    assert session.geometry_version == version, "extraction options rebuilt the hierarchy"
    assert session.config["pure_quad"] is True
    assert session.config["smooth_iter"] == 3
    # Subdividing away the triangles can only add faces, and on a quad-dominant
    # result it multiplies them.
    assert pure.get("n_faces") > mixed.get("n_faces")


def test_the_status_says_whether_unwrapping_is_possible(client, torus) -> None:
    """The viewer hides its UV control rather than offering a dead one.

    xatlas is optional, and a slider that can only answer "not installed" is
    worse than no slider: it reads as a broken feature rather than an absent
    one.
    """
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)
        # HELLO answers with a status of its own, which is how this asks for
        # one rather than waiting for the streamer to notice a change.
        socket.send(MessageType.HELLO, {})
        status = socket.expect(MessageType.STATUS)

    assert status.get("uv") is uv.available()
    assert 0.0 <= status.get("uv_leniency") <= 1.0


@pytest.mark.skipif(not uv.available(), reason="xatlas is not installed")
def test_unwrapping_extracts_a_mesh_to_flatten(client, torus) -> None:
    """Hovering the UV control is the only request the viewport makes.

    There is no Extract button behind it, so UNWRAP has to reach a solved
    field and an extracted mesh by itself -- exactly as asking to see the
    output mesh does.
    """
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)  # deliberately no SOLVE and no EXTRACT

        socket.send(MessageType.UNWRAP, {"leniency": 0.5})
        layout = socket.expect(MessageType.UV_LAYOUT)
        # Hovering is not a request to look at the output mesh, so the
        # extraction it needed is not pushed at the viewport.
        assert MessageType.EXTRACTED not in {m.type for m in socket.drain()}

    assert layout.get("n_charts") >= 1
    assert layout.get("n_faces") > 0
    assert layout.get("unmapped") == 0
    assert layout.arrays["faces"].shape == (layout.get("n_faces"), layout.get("posy"))
    assert layout.arrays["chart"].shape == (layout.get("n_faces"),)
    coordinates = layout.arrays["uv"]
    assert coordinates.shape[1] == 2
    assert 0.0 <= coordinates.min() <= coordinates.max() <= 1.0


@pytest.fixture
def slow_unwrap(monkeypatch):
    """An unwrapper that reports progress and dawdles, so it can be watched.

    The real one finishes a test torus in milliseconds, which the streamer's
    15 Hz tick would step straight over; and it needs xatlas, which this has
    no opinion about. What is under test is the route from a worker thread to
    the socket, so the layout it returns is a stub.
    """

    def fake(vertices, faces, leniency=uv.DEFAULT_LENIENCY, progress=None):
        for fraction in (0.25, 0.5, 0.75):
            if progress:
                progress(fraction)
            time.sleep(0.15)
        count = len(faces)
        return uv.UvLayout(
            uv=np.zeros((1, 2), dtype=np.float32),
            faces=np.zeros((count, 4), dtype=np.int32),
            chart=np.zeros(count, dtype=np.int32),
            cut=np.zeros((0, 3), dtype=np.int32),
            cut_face=np.zeros(0, dtype=np.int32),
            cut_slot=np.zeros((0, 3), dtype=np.int32),
            cut_chart=np.zeros(0, dtype=np.int32),
            chart_count=1,
            leniency=leniency,
        )

    monkeypatch.setattr(uv, "unwrap", fake)
    monkeypatch.setattr(uv, "available", lambda: True)


def test_unwrapping_reports_its_progress_while_it_runs(
    client, torus, slow_unwrap
) -> None:
    """The fraction has to arrive during the wait, not after it.

    It is read straight off the session rather than through the worker thread,
    so it is the one report that cannot queue behind the job it describes.
    """
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)

        socket.send(MessageType.UNWRAP, {"leniency": 0.5})
        # The reader is serial, so the first of these returns once the unwrap
        # is done, carrying everything the streamer sent in the meantime; the
        # rest wait for the tick that reports it finished.
        reported: List[object] = []
        for _ in range(20):
            reported += [
                m.get("uv") for m in socket.drain() if m.type == MessageType.PROGRESS
            ]
            if reported and reported[-1] is None:
                break
            time.sleep(0.05)
    assert len(reported) >= 3, f"only saw {reported}"
    assert reported[-1] is None, "the label is never cleared"
    moving = [value for value in reported if value is not None]
    assert moving == sorted(moving), f"progress went backwards: {moving}"
    assert max(moving) >= 0.5


def test_the_session_keeps_answering_while_an_unwrap_runs(
    client, torus, slow_unwrap
) -> None:
    """The stream must not go quiet for the length of an unwrap.

    xatlas needs nothing from the core, so it runs off the session's single
    worker thread; while it held that thread no status could be read, and the
    viewport sat through a multi-second flatten unable to learn that the solve
    it was waiting on had finished -- which is how a burst of view keys left it
    saying "solving" with nothing on screen and no way back.
    """
    import asyncio

    async def exercise() -> None:
        session = await registry_of(client).create()
        await session.load_mesh(
            np.asarray(torus.vertices),
            np.asarray(torus.faces),
            "torus.obj",
            {"vertex_count": 150, "deterministic": True},
        )
        unwrapping = asyncio.create_task(session.unwrap(0.5))
        # The stub sleeps 0.45s in total; every status read here would have
        # queued behind it, and the last one would have landed after the wait.
        answered = 0
        while not unwrapping.done():
            await session.status()
            answered += 1
            await asyncio.sleep(0.02)
        await unwrapping
        assert answered >= 5, f"only {answered} status reads got through"
        assert session.uv_progress is None
        await session.close()

    asyncio.run(exercise())


def test_an_abandoned_unwrap_cannot_resurrect_its_progress(torus) -> None:
    """A thread whose future was dropped keeps running, and keeps reporting.

    Its next report used to set a progress value that nothing was left to
    clear, and a session that reads as permanently busy is one whose viewport
    never gets another status frame.
    """
    import asyncio

    async def exercise() -> None:
        session = session_manager.BrushSession("test-abandoned")
        await session.open()
        try:
            stale = session._progress_reporter(session._uv_run)
            session._uv_run += 1  # as the end of an unwrap does
            session._uv_progress = None
            stale(0.5)
            assert session.uv_progress is None
        finally:
            await session.close()

    asyncio.run(exercise())


def test_extracting_waits_for_a_running_solve_rather_than_cutting_it_short(
    client, torus
) -> None:
    """Asking to see the result must not throw away the solve in progress.

    Extraction stops the solver so that what it reads is self-consistent, and
    stopping one that a brush stroke had just started left the field half
    solved with nothing to restart it: pressing 2 during a solve quietly
    abandoned the rest of it and extracted the state it happened to be in.
    """
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus, vertex_count=400)

        socket.send(MessageType.SOLVE, {"field": "both", "level": -1})
        socket.expect(MessageType.STATUS)
        socket.send(MessageType.EXTRACT, {})

        # The reader is serial, so the round that carries the extraction also
        # carries the status the handler sent straight after it.
        frames = []
        for _ in range(60):
            frames += socket.drain()
            if any(m.type == MessageType.EXTRACTED for m in frames):
                break
            time.sleep(0.05)
        assert any(m.type == MessageType.EXTRACTED for m in frames), "never extracted"

        # The extraction only comes back once the solve it waited for is done,
        # so by then both counters have to have been published.
        status = [m for m in frames if m.type == MessageType.STATUS][-1]
        assert status.get("solving") is False
        assert status.get("iterations_q") >= 0
        assert status.get("iterations_o") >= 0


def test_a_rebuilt_stroke_is_marked_as_resent(client, torus) -> None:
    """A rebuild re-projects its strokes; that is not somebody drawing.

    The viewport shows the input surface whenever a stroke arrives, because a
    brush needs something to draw on. An echo of an old stroke that is not
    marked takes the view away from whoever was looking at the result.
    """
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)

        origins, directions = equator_rays(torus)
        socket.send(
            MessageType.STROKE,
            {"kind": 0, "solve": False},
            {"ray_origins": origins, "ray_directions": directions},
        )
        drawn = socket.expect(MessageType.STROKE_RESULT)
        assert drawn.get("resent") is False

        socket.send(MessageType.SET_CONFIG, {"config": {"vertex_count": 220}})
        # Both arrive in the same round, and expect() drops whatever shares a
        # round with its match.
        _, echoed = socket.expect_all(MessageType.GEOMETRY, MessageType.STROKE_RESULT)
        assert echoed.get("resent") is True


def registry_of(client) -> SessionRegistry:
    return client.app.state.registry


# ---------------------------------------------------------------------------
#  The imported file's own materials
# ---------------------------------------------------------------------------


def _textured_glb(torus) -> bytes:
    """A glB of the torus with one material and one base colour map."""
    trimesh = pytest.importorskip("trimesh")
    image = pytest.importorskip("PIL.Image")

    mesh = trimesh.Trimesh(
        vertices=np.asarray(torus.vertices, dtype=np.float64),
        faces=np.asarray(torus.faces),
        process=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            name="shell", baseColorTexture=image.new("RGB", (16, 16), (12, 200, 90))
        ),
    )
    return trimesh.Scene(mesh).export(file_type="glb")


def test_an_uploaded_model_keeps_its_maps_for_the_viewport(client, torus) -> None:
    """The file as authored, alongside the welded copy the solver works on.

    They are different meshes on purpose: the maps are pinned to UVs that only
    exist on the unwelded original, and the remesher cannot take that one.
    """
    session_id = open_session(client)
    upload = client.post(
        f"/api/session/{session_id}/mesh",
        files={"file": ("torus.glb", _textured_glb(torus), "model/gltf-binary")},
        data={"config": '{"vertex_count": 150}'},
    )
    assert upload.status_code == 200, upload.text
    assert upload.json()["textures"] == 1, "the base colour map was not found"

    source = client.get(f"/api/session/{session_id}/source")
    assert source.status_code == 200
    frame = protocol.decode(source.content)
    assert frame.header["slots"] == ["base colour"]
    assert frame.header["materials"] == ["shell"]
    # Grouped by material, which is what lets the viewer draw one run each.
    assert sum(count for _, _, count in frame.header["groups"]) == frame.header["n_faces"]
    assert frame.arrays["uv"].shape == (frame.header["n_vertices"], 2)

    image = client.get(f"/api/session/{session_id}/texture/0/0")
    assert image.status_code == 200
    assert image.headers["content-type"] in ("image/jpeg", "image/png")
    assert len(image.content) > 0

    # A slot that material has nothing in is a 404, which is what tells the
    # viewer to draw it flat rather than leave the previous map on it.
    assert client.get(f"/api/session/{session_id}/texture/3/0").status_code == 404


def test_a_model_with_no_materials_offers_no_texture_buttons(client, torus) -> None:
    """Which is every format the viewer took before this, and the usual case."""
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        # LOAD_MESH answers with the geometry and a status in one round, and
        # expect() drops whatever shares a round with its match.
        socket.send(
            MessageType.LOAD_MESH,
            {"name": "torus.obj", "config": {"vertex_count": 150}},
            {"vertices": np.asarray(torus.vertices), "faces": np.asarray(torus.faces)},
        )
        _, status = socket.expect_all(MessageType.GEOMETRY, MessageType.STATUS)
    assert status.get("textures") == 0


def test_the_source_of_a_session_with_no_import_is_a_404(client, torus) -> None:
    """A mesh pushed over the socket as arrays was never a file."""
    session_id = open_session(client)
    with client.websocket_connect(f"/ws/{session_id}") as ws:
        load_mesh(Socket(ws), torus)
    assert client.get(f"/api/session/{session_id}/source").status_code == 404
    assert client.get(f"/api/session/{session_id}/texture/0/0").status_code == 404


@pytest.mark.skipif(not uv.available(), reason="xatlas is not installed")
def test_an_exported_obj_carries_its_uv_layout(client, torus, tmp_path) -> None:
    """The atlas is only worth cutting if it leaves with the file.

    Export never asks for one, so this is also the check that it cuts one on
    its own rather than writing whatever the last hover happened to leave.
    """
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        load_mesh(socket, torus)
        # The chart size rides with the frame, so pressing Export after moving
        # the slider writes what the slider says rather than what was last
        # previewed -- which may be nothing at all.
        # A hover already built an extraction behind the viewport's back, so
        # this is also the check that Export still reports the size it wrote.
        socket.send(MessageType.UNWRAP, {"leniency": 0.5})
        socket.expect(MessageType.UV_LAYOUT)

        socket.send(MessageType.EXPORT, {"format": "obj", "leniency": 0.0})
        ready, layout, extracted = socket.expect_all(
            MessageType.EXPORT_READY, MessageType.UV_LAYOUT, MessageType.EXTRACTED
        )

        written = client.get(ready.get("url"))

    assert layout.get("leniency") == pytest.approx(0.0)
    assert extracted.get("n_faces") > 0
    assert written.status_code == 200, written.text
    text = written.text
    assert "\nvt " in text
    faces = [line for line in text.splitlines() if line.startswith("f ")]
    assert faces and all("/" in line for line in faces)


def test_a_bad_frame_does_not_drop_the_socket(client) -> None:
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        ws.send_bytes(b"not a frame at all")
        errors = [m for m in socket.drain() if m.type == MessageType.ERROR]
        assert errors and errors[0].get("message")

        assert socket.drain() == []
