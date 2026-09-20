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

import time
from typing import List

import numpy as np
import pytest

from instant_meshes_brush import protocol
from instant_meshes_brush.protocol import MessageType
from instant_meshes_brush.server import build_app
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
                  "tools.js", "net.js", "style.css", "vendor/three.module.js"):
        response = client.get(f"/static/{asset}")
        assert response.status_code == 200, asset
        assert response.content, asset


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


def test_a_bad_frame_does_not_drop_the_socket(client) -> None:
    with client.websocket_connect(f"/ws/{open_session(client)}") as ws:
        socket = Socket(ws)
        ws.send_bytes(b"not a frame at all")
        errors = [m for m in socket.drain() if m.type == MessageType.ERROR]
        assert errors and errors[0].get("message")

        assert socket.drain() == []
