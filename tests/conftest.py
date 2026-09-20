"""Shared fixtures for the Instant Meshes brush test suite.

Every fixture here is sized so that a full round of the pipeline -- preprocess,
both solves and an extraction -- costs well under a tenth of a second.  The
algorithm is happy to spend minutes on a real asset; a test suite that did the
same would never be run.

A torus is the default subject because it is closed, manifold and genus 1: it
has no boundary to special-case, its Euler characteristic pins the working mesh
to ``nF == 2 * nV`` no matter how the preprocessor subdivides it, and its
principal curvature directions give the cross field something non-trivial to
align to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Tuple

import numpy as np
import pytest

import instant_meshes_brush as imb

#: Output vertex budget used by the shared session fixtures.  Small enough to
#: keep a solve at a few tens of milliseconds, large enough that the extracted
#: mesh still has interior quads and singularities to look at.
TARGET_VERTEX_COUNT = 250


@dataclass(frozen=True)
class TorusMesh:
    """A triangulated torus plus the analytic surface it was sampled from.

    Tests aim brush rays with :meth:`surface`, which keeps the expected hit
    point independent of how finely the mesh happens to be tessellated.
    """

    vertices: np.ndarray  #: (nV, 3) float32
    faces: np.ndarray  #: (nF, 3) uint32
    major_radius: float
    minor_radius: float

    def __iter__(self) -> Iterator[np.ndarray]:
        """Unpack as ``V, F``, the pair every Session entry point wants."""
        yield self.vertices
        yield self.faces

    def surface(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Exact positions and outward unit normals at the given angles.

        ``u`` runs around the ring, ``v`` around the tube; ``v == 0`` is the
        outer equator.
        """
        return _surface(u, v, self.major_radius, self.minor_radius)


def _surface(
    u: np.ndarray, v: np.ndarray, major_radius: float, minor_radius: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Torus parametrisation and its outward unit normal field."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    radial = major_radius + minor_radius * np.cos(v)
    points = np.stack(
        [radial * np.cos(u), radial * np.sin(u), minor_radius * np.sin(v)], -1
    )
    normals = np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)], -1)
    return points.astype(np.float32), normals.astype(np.float32)


def _build_torus(
    segments_u: int = 48,
    segments_v: int = 24,
    major_radius: float = 1.0,
    minor_radius: float = 0.35,
) -> TorusMesh:
    """Tessellate a torus into two triangles per quad of the (u, v) grid."""
    u = np.linspace(0.0, 2.0 * np.pi, segments_u, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, segments_v, endpoint=False)
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
    vertices = _surface(grid_u, grid_v, major_radius, minor_radius)[0].reshape(-1, 3)

    index = np.arange(segments_u * segments_v).reshape(segments_u, segments_v)
    next_u = np.roll(index, -1, axis=0)
    next_v = np.roll(index, -1, axis=1)
    next_uv = np.roll(next_u, -1, axis=1)
    lower = np.stack([index, next_u, next_uv], -1).reshape(-1, 3)
    upper = np.stack([index, next_uv, next_v], -1).reshape(-1, 3)
    faces = np.concatenate([lower, upper]).astype(np.uint32)

    # Session-scoped fixtures hand the same arrays to every test; freezing them
    # turns an accidental in-place edit into an error instead of a mystery.
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return TorusMesh(vertices, faces, major_radius, minor_radius)


@pytest.fixture(scope="session", autouse=True)
def _quiet_core() -> None:
    """The solver narrates every stage on stdout, which buries pytest's own."""
    imb.set_verbose(False)


@pytest.fixture(scope="session")
def target_vertex_count() -> int:
    """The output vertex budget the shared session fixtures ask for."""
    return TARGET_VERTEX_COUNT


@pytest.fixture(scope="session")
def make_torus() -> Callable[..., TorusMesh]:
    """Factory: ``make_torus(segments_u=.., segments_v=.., ...) -> TorusMesh``."""
    return _build_torus


@pytest.fixture(scope="session")
def torus(make_torus: Callable[..., TorusMesh]) -> TorusMesh:
    """The default 48 x 24 torus, shared read-only by the whole suite."""
    return make_torus()


@pytest.fixture
def make_session(torus: TorusMesh) -> Iterator[Callable[..., imb.Session]]:
    """Factory for preprocessed sessions.

    Keyword arguments override the :class:`instant_meshes_brush.Config` defaults
    used here, which are deterministic so that a failure is reproducible.  Every
    session handed out is dropped at teardown, because each one owns a C++
    solver thread that only stops when the object is destroyed.

    Teardown does not call ``stop_solve()`` first, and neither should callers
    who are only finished with a session: ``~Session`` (src/session.cpp:73)
    already runs ``stop()`` and then ``shutdown()`` itself, so an extra stop
    only widens the lost-wakeup window in ``Optimizer::shutdown()``
    (src/field.h:170), which notifies without holding ``mRes.mutex()``.
    """
    live = []

    def build(mesh: TorusMesh = torus, **config: object) -> imb.Session:
        settings = dict(vertex_count=TARGET_VERTEX_COUNT, deterministic=True)
        settings.update(config)
        session = imb.Session()
        session.set_mesh(mesh.vertices, mesh.faces)
        session.preprocess(imb.Config(**settings))
        live.append(session)
        return session

    yield build
    live.clear()


@pytest.fixture
def mesh_session(make_session: Callable[..., imb.Session]) -> imb.Session:
    """A preprocessed session whose fields have not been solved yet."""
    return make_session()


@pytest.fixture
def solved_session(mesh_session: imb.Session) -> imb.Session:
    """A preprocessed session with both fields run to convergence."""
    mesh_session.solve_all()
    return mesh_session
