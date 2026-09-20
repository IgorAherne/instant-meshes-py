"""Brush tools, singularity attractors, and the surface families the torus misses.

These are the regressions for three defects that were live in this tree:

* a level-0 solve silently ignored brush strokes, because the frozen sweep in
  ``field.cpp`` never reads the constraint arrays;
* an attractor path whose faces were not edge-adjacent threw from the solver
  thread, which called ``std::terminate`` and killed the interpreter;
* the TBB shim's serial fast path ran only the first chunk of each parallel
  region, so anything solved with ``set_thread_count(1)`` was quietly wrong.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pytest

import instant_meshes_brush as imb

#: Attractors need a field that actually has singularities to drag, which a
#: 250-vertex target torus does not reliably produce.
ATTRACTOR_VERTEX_COUNT = 1200


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _torus_uv(point: np.ndarray, major_radius: float) -> Tuple[float, float]:
    """Inverse of the conftest torus parametrisation."""
    u = float(np.arctan2(point[1], point[0]))
    v = float(np.arctan2(point[2], float(np.hypot(point[0], point[1])) - major_radius))
    return u, v


def _rays_at(points: np.ndarray, normals: np.ndarray, distance: float = 3.0):
    """Rays that hit `points` head-on from outside the surface."""
    origins = (points + normals * distance).astype(np.float32)
    directions = (points - origins).astype(np.float32)
    return origins, directions


def _arc_on_torus(torus, u0: float, v0: float, span: float = 0.45, samples: int = 20):
    """A stroke that follows the major circle starting at (u0, v0)."""
    u = np.linspace(u0, u0 + span, samples)
    v = np.full_like(u, v0)
    points, normals = torus.surface(u, v)
    return _rays_at(points, normals)


def _tangent_alignment(session: imb.Session, curve: imb.Curve, field: np.ndarray) -> float:
    """Mean |cos| between the 4-RoSy field and the stroke tangent near the stroke.

    A 4-RoSy direction is only defined up to a quarter turn, so the best of the
    tangent and its in-plane perpendicular is what counts as aligned.
    """
    positions = curve.positions
    tangents = np.gradient(positions, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9

    faces, normals = session.faces, session.normals
    scores = []
    for sample, face in enumerate(curve.faces):
        for vertex in faces[face]:
            q = field[vertex]
            perpendicular = np.cross(normals[vertex], q)
            scores.append(max(abs(q @ tangents[sample]), abs(perpendicular @ tangents[sample])))
    return float(np.mean(scores))


@pytest.fixture
def equator_stroke(torus) -> Callable[[imb.Session, bool], imb.Curve]:
    """Project a stroke along the torus's outer equator onto a session."""

    def project(session: imb.Session, attractor: bool = False) -> imb.Curve:
        origins, directions = _arc_on_torus(torus, u0=-0.3, v0=0.0, span=0.6, samples=24)
        curve = session.project_stroke(origins, directions, attractor)
        assert curve is not None, "the equator stroke should hit the torus"
        return curve

    return project


# ---------------------------------------------------------------------------
#  Strokes
# ---------------------------------------------------------------------------


def test_level_zero_solve_still_honours_a_stroke_after_freezing(
    solved_session: imb.Session, equator_stroke
) -> None:
    """solve_all() freezes the integer variables; brushing must still work.

    The frozen sweep ignores CQ/CQw entirely, so Session has to thaw before a
    level-0 solve or the stroke would have no effect whatsoever.
    """
    curve = equator_stroke(solved_session)
    before = solved_session.orientation_field.copy()

    solved_session.add_stroke(imb.StrokeKind.ORIENTATION, curve)
    solved_session.solve_orientations(0)
    for _ in range(40):
        if solved_session.status.iterations_q > 0:
            break
    solved_session.stop_solve()
    solved_session.wait_solve()

    after = solved_session.orientation_field
    assert not np.array_equal(before, after), "a level-0 solve ignored the stroke"
    assert _tangent_alignment(solved_session, curve, after) >= _tangent_alignment(
        solved_session, curve, before
    )


def test_edge_brush_pins_the_position_field_to_the_stroke(
    solved_session: imb.Session, equator_stroke
) -> None:
    """An edge brush constrains O as well as Q, so O should sit on the stroke."""
    curve = equator_stroke(solved_session)
    faces = solved_session.faces
    touched = np.unique(faces[curve.faces].ravel())

    def mean_distance(field: np.ndarray) -> float:
        deltas = field[touched][:, None, :] - curve.positions[None, :, :]
        return float(np.linalg.norm(deltas, axis=2).min(axis=1).mean())

    before = mean_distance(solved_session.position_field)

    solved_session.add_stroke(imb.StrokeKind.EDGE, curve)
    solved_session.solve_orientations(-1)
    solved_session.wait_solve()
    solved_session.solve_positions(-1)
    solved_session.wait_solve()

    assert mean_distance(solved_session.position_field) <= before + 1e-6


# ---------------------------------------------------------------------------
#  Singularity attractors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orientation", [True, False], ids=["orientation", "position"])
def test_attractor_moves_a_singularity(
    make_session, torus, orientation: bool
) -> None:
    """Dragging from a singular face relocates that singularity."""
    session = make_session(vertex_count=ATTRACTOR_VERTEX_COUNT)
    session.solve_all()

    def read():
        return set(
            session.orientation_singularities
            if orientation
            else session.position_singularities
        )

    before = read()
    if not before:
        pytest.skip("this field has no singularities to drag")

    # The solver moves whatever sits on the stroke's FIRST face, so the stroke
    # has to start on one that is genuinely singular.
    start_face = sorted(before)[0]
    centre = session.vertices[session.faces[start_face]].mean(axis=0)
    u0, v0 = _torus_uv(centre, torus.major_radius)

    origins, directions = _arc_on_torus(torus, u0, v0)
    curve = session.project_stroke(origins, directions, True)
    assert curve is not None
    assert curve.faces[0] == start_face, "the stroke must begin on the singular face"

    session.apply_attractor(curve, orientation=orientation)
    for _ in range(60):
        if not session.status.active:
            break
    session.stop_solve()
    session.wait_solve()

    assert session.status.error == ""
    assert read() != before, "the singularity did not move"


def test_attractor_snaps_to_the_singularity_it_was_aimed_at(
    make_session, torus
) -> None:
    """A drag that starts beside the marker still moves that singularity.

    A singularity occupies one triangle of the working mesh, and the dot drawn
    over it is many times that size, so aiming by eye lands next to the face
    rather than on it. Requiring the exact face makes the tool look broken:
    nothing happens and nothing says why.
    """
    session = make_session(vertex_count=ATTRACTOR_VERTEX_COUNT)
    session.solve_all()

    before = set(session.orientation_singularities)
    if not before:
        pytest.skip("this field has no singularities to drag")

    start_face = sorted(before)[0]
    centre = session.vertices[session.faces[start_face]].mean(axis=0)
    u0, v0 = _torus_uv(centre, torus.major_radius)

    # Start most of an edge length around the tube from the marker's own face.
    origins, directions = _arc_on_torus(torus, u0, v0 + session.scale)
    curve = session.project_stroke(origins, directions, True)
    assert curve is not None
    assert curve.faces[0] != start_face, "this test needs a stroke that misses"

    session.apply_attractor(curve, orientation=True)
    session.wait_solve()

    assert session.status.error == ""
    assert start_face not in set(session.orientation_singularities), (
        "the singularity under the marker was not the one that moved"
    )


def test_attractor_rejects_a_path_that_is_not_edge_adjacent(
    solved_session: imb.Session,
) -> None:
    """Regression: this used to terminate the process from the solver thread."""
    face_count = len(solved_session.faces)
    scattered = np.array([0, face_count // 2, face_count - 1], np.uint32)
    curve = imb.Curve.from_arrays(
        np.zeros((3, 3), np.float32),
        np.tile(np.array([0.0, 0.0, 1.0], np.float32), (3, 1)),
        scattered,
    )

    with pytest.raises(RuntimeError, match="edge-adjacent"):
        solved_session.apply_attractor(curve, orientation=True)

    # The session must still be usable, which is the whole point.
    solved_session.solve_orientations(-1)
    solved_session.wait_solve()
    assert not solved_session.status.active


# ---------------------------------------------------------------------------
#  Parallelism
# ---------------------------------------------------------------------------


def test_deterministic_solve_matches_across_thread_counts(make_session) -> None:
    """Regression: the shim's serial path once ran only the first chunk.

    The deterministic partitioning depends on grain size alone, never on the
    pool size, so one thread and many must produce identical fields.
    """
    try:
        single = make_session(deterministic=True)
        imb.set_thread_count(1)
        single.solve_all()
        q_single = single.orientation_field.copy()
        o_single = single.position_field.copy()
    finally:
        imb.set_thread_count(-1)

    parallel = make_session(deterministic=True)
    parallel.solve_all()

    assert np.array_equal(parallel.orientation_field, q_single)
    assert np.array_equal(parallel.position_field, o_single)


# ---------------------------------------------------------------------------
#  Surface families the torus does not cover
# ---------------------------------------------------------------------------


def _grid(n: int = 24, warp: float = 0.35):
    """An open patch: boundaries, which a torus has none of."""
    axis = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    z = warp * np.sin(3.0 * x) * np.cos(3.0 * y)
    vertices = np.stack([x, y, z], -1).reshape(-1, 3).astype(np.float32)

    index = np.arange(n * n).reshape(n, n)
    a, b = index[:-1, :-1], index[1:, :-1]
    c, d = index[1:, 1:], index[:-1, 1:]
    faces = np.concatenate(
        [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]
    ).astype(np.uint32)
    return vertices, faces


def _cube(n: int = 10):
    """A closed box whose 90-degree edges exercise crease handling."""
    ticks = np.linspace(-1.0, 1.0, n)
    vertices, faces = [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            base = len(vertices)
            for u in ticks:
                for v in ticks:
                    point = [0.0, 0.0, 0.0]
                    point[axis] = sign
                    point[(axis + 1) % 3] = float(u)
                    point[(axis + 2) % 3] = float(v)
                    vertices.append(point)
            for i in range(n - 1):
                for j in range(n - 1):
                    p = base + i * n + j
                    q, r, s = p + n, p + n + 1, p + 1
                    # Wind outward so the normals agree across the seams.
                    faces += [[p, q, r], [p, r, s]] if sign > 0 else [[p, r, q], [p, s, r]]

    vertices = np.array(vertices, np.float32)
    faces = np.array(faces, np.uint32)
    # Weld the duplicated seam vertices so neighbouring sides share edges.
    keys = np.round(vertices * 1e5).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return (
        np.ascontiguousarray(vertices[first]),
        np.ascontiguousarray(inverse[faces].astype(np.uint32)),
    )


@pytest.mark.parametrize(
    "name, config",
    [
        ("plain", {}),
        ("aligned", {"align_to_boundaries": True}),
        ("triangles", {"rosy": 6, "posy": 3}),
    ],
)
def test_open_patch_remeshes(name: str, config: dict) -> None:
    vertices, faces = _grid()
    session = imb.Session()
    session.set_mesh(vertices, faces)
    session.preprocess(imb.Config(vertex_count=400, deterministic=True, **config))
    session.solve_all()

    mesh = session.extract()
    assert mesh.faces.size, f"{name}: extraction produced nothing"
    assert mesh.faces.max() < mesh.vertices.shape[0]
    assert np.isfinite(mesh.vertices).all()


@pytest.mark.parametrize("crease_angle", [-1.0, 30.0], ids=["smooth", "creased"])
def test_cube_has_eight_singularities(crease_angle: float) -> None:
    """A genus-0 surface must carry total 4-RoSy index 4 * chi = 8."""
    vertices, faces = _cube()
    session = imb.Session()
    session.set_mesh(vertices, faces)
    session.preprocess(
        imb.Config(vertex_count=500, deterministic=True, crease_angle=crease_angle)
    )
    session.solve_all()

    singularities = session.orientation_singularities
    assert sum(singularities.values()) == 8, singularities
    assert session.extract().faces.size


def test_stats_describe_the_working_mesh(solved_session: imb.Session, torus) -> None:
    stats = solved_session.stats
    assert stats.surface_area > 0.0
    assert 0.0 < stats.average_edge_length <= stats.maximum_edge_length

    lower = np.array(stats.aabb_min)
    upper = np.array(stats.aabb_max)
    assert (lower < upper).all()

    reach = torus.major_radius + torus.minor_radius
    assert np.allclose(upper[:2], reach, atol=0.05)
    assert np.allclose(lower[:2], -reach, atol=0.05)
