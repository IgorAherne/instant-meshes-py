"""Tests for the headless Instant Meshes core through its Python bindings.

These exercise ``src/session.cpp`` and the shims it now stands on -- the TBB
replacement in ``ext/tbb_shim`` in particular, whose whole reason to exist is
that ``deterministic=True`` must still mean bit-identical output.  That claim is
checked strictly (``np.array_equal``, not ``allclose``): a reduction that folds
its partials in thread-completion order would drift only in the last bits, which
is exactly the failure a tolerance would hide.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np
import pytest

import instant_meshes_brush as imb

if TYPE_CHECKING:  # the fixtures supply these at run time
    from conftest import TorusMesh


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


#: How far outside the surface the test rays start, in mesh units.
_RAY_STANDOFF = 2.0


def _rays_onto_surface(
    mesh: TorusMesh, u: np.ndarray, v: np.ndarray, standoff: float = _RAY_STANDOFF
) -> Tuple[np.ndarray, np.ndarray]:
    """Rays that strike the torus at the given (u, v) angles.

    Each ray starts well outside the mesh on the surface normal and shoots
    straight back down it, so the hit point is known before the BVH is asked.
    Keep ``v`` on the outer half of the tube: further round, the normal points
    through the hole and the ray would hit the far wall first.
    """
    points, normals = mesh.surface(u, v)
    return points + standoff * normals, -normals


def _diagonal_stroke_rays(
    mesh: TorusMesh, samples: int = 14
) -> Tuple[np.ndarray, np.ndarray]:
    """Rays tracing a path that follows neither principal curvature direction.

    A stroke along the ring or around the tube would ask the field for what it
    already does on a torus, and prove nothing.
    """
    t = np.linspace(0.0, 1.0, samples)
    return _rays_onto_surface(mesh, 0.2 + 1.1 * t, -0.9 + 1.8 * t)


def _stroke_targets(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """The surface points ``_rays_onto_surface`` aimed each ray at.

    Its rays start one standoff along the outward normal and shoot straight
    back down it, so the target is the origin advanced by that standoff.
    """
    return origins + _RAY_STANDOFF * directions


def _covers(points: np.ndarray, targets: np.ndarray) -> float:
    """How far the furthest of ``points`` strays from the ``targets`` path.

    ``smooth_curve`` walks the mesh between the projected samples and inserts
    its own, so a curve cannot be compared sample for sample with the rays that
    produced it -- only asked whether it stayed where they pointed.
    """
    return float(np.linalg.norm(points[:, None] - targets[None], axis=2).min(1).max())


def _polyline_tangents(points: np.ndarray) -> np.ndarray:
    """Unit direction of travel at every sample of a polyline."""
    tangents = np.empty_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    return tangents / np.linalg.norm(tangents, axis=1, keepdims=True)


def _cross_field_deviation(
    field: np.ndarray, normals: np.ndarray, tangents: np.ndarray
) -> np.ndarray:
    """Angle in degrees from each tangent to the nearest arm of the cross.

    A 4-RoSy field is only defined up to a quarter turn about the surface
    normal, so the honest error is the smaller of the angles to ``q`` and to
    ``n x q``, measured after projecting the tangent into the tangent plane.
    Range is 0 (aligned) to 45 (as wrong as a cross field can be).
    """
    planar = tangents - (tangents * normals).sum(1, keepdims=True) * normals
    planar /= np.linalg.norm(planar, axis=1, keepdims=True)
    along = np.abs((planar * field).sum(1))
    across = np.abs((planar * np.cross(normals, field)).sum(1))
    cosine = np.maximum(along, across) / np.hypot(along, across)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _vertices_near(
    session: imb.Session, curve_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of working-mesh vertices within one edge length of a curve.

    Also returns, per vertex, the curve sample it is closest to, so a caller can
    ask what direction the stroke was heading there.
    """
    squared = ((session.vertices[:, None, :] - curve_points[None, :, :]) ** 2).sum(-1)
    nearest = squared.argmin(1)
    selected = np.sqrt(squared.min(1)) < session.scale
    return np.flatnonzero(selected), nearest


def _edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Length of every (doubly counted) triangle edge."""
    corners = vertices[faces]
    return np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2).ravel()


def _os_thread_count() -> Optional[int]:
    """Threads the operating system sees in this process, or None if unknown.

    ``threading.active_count()`` only knows about threads Python created; the
    solver and the worker pool are C++ ``std::thread``s, so a leak there is
    invisible to it and has to be counted the hard way.
    """
    try:
        with open("/proc/self/status", "r", encoding="ascii") as status:
            for line in status:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except OSError:
        pass
    if sys.platform != "win32":
        return None
    return _windows_thread_count()


class _THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_ulong),
        ("cntUsage", ctypes.c_ulong),
        ("th32ThreadID", ctypes.c_ulong),
        ("th32OwnerProcessID", ctypes.c_ulong),
        ("tpBasePri", ctypes.c_long),
        ("tpDeltaPri", ctypes.c_long),
        ("dwFlags", ctypes.c_ulong),
    ]


def _windows_thread_count() -> Optional[int]:
    """Walk the system thread list, counting the entries owned by us."""
    snapshot_threads = 0x00000004
    invalid_handle = ctypes.c_void_p(-1).value

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    entry_ptr = ctypes.POINTER(_THREADENTRY32)
    kernel32.Thread32First.argtypes = [ctypes.c_void_p, entry_ptr]
    kernel32.Thread32Next.argtypes = [ctypes.c_void_p, entry_ptr]

    snapshot = kernel32.CreateToolhelp32Snapshot(snapshot_threads, 0)
    if snapshot in (None, invalid_handle):
        return None
    entry = _THREADENTRY32()
    entry.dwSize = ctypes.sizeof(_THREADENTRY32)
    pid = os.getpid()
    count = 0
    try:
        if not kernel32.Thread32First(snapshot, ctypes.byref(entry)):
            return None
        while True:
            if entry.th32OwnerProcessID == pid:
                count += 1
            if not kernel32.Thread32Next(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)
    return count


def _ply_face_records(path: str) -> Tuple[int, List[int]]:
    """Declared face count and the vertex count of every face actually written.

    ``write_ply`` stitches degenerate quads back into n-gons and has to predict
    how many records that leaves before it writes the header; if the prediction
    and the loop ever disagree the file is silently corrupt, so walk it.
    """
    with open(path, "rb") as handle:
        blob = handle.read()
    body = blob.index(b"end_header\n") + len(b"end_header\n")
    header = blob[:body].decode("ascii").splitlines()

    counts = [int(line.split()[2]) for line in header if line.startswith("element ")]
    scalars = [0, 0]
    element = -1
    for line in header:
        if line.startswith("element "):
            element += 1
        elif line.startswith("property ") and not line.startswith("property list"):
            scalars[element] += 1
    vertex_count, face_count = counts

    offset = body + vertex_count * scalars[0] * 4
    sizes = []
    while offset < len(blob):
        corners = blob[offset]
        offset += 1 + corners * 4 + scalars[1] * 4
        sizes.append(corners)
    assert offset == len(blob), "a face record ran past the end of the file"
    return face_count, sizes


# ---------------------------------------------------------------------------
#  Preprocessing
# ---------------------------------------------------------------------------


def test_preprocess_builds_a_working_mesh(mesh_session: imb.Session) -> None:
    vertices, faces, normals = (
        mesh_session.vertices,
        mesh_session.faces,
        mesh_session.normals,
    )

    assert mesh_session.ready
    assert vertices.dtype == np.float32 and vertices.shape[1] == 3
    assert faces.dtype == np.uint32 and faces.shape[1] == 3
    assert normals.shape == vertices.shape
    assert faces.max() < len(vertices)
    # Euler's formula on a closed genus-1 surface: subdividing must not have
    # torn or welded anything.
    assert len(faces) == 2 * len(vertices)
    assert np.abs(np.linalg.norm(normals, axis=1) - 1.0).max() < 1e-5

    assert mesh_session.scale > 0.0
    assert mesh_session.scale < np.ptp(vertices, axis=0).max()
    # The preprocessor subdivides until the input resolves the target edge
    # length, so no triangle may be longer than one.
    assert _edge_lengths(vertices, faces).max() < mesh_session.scale


def test_preprocess_subdivides_a_coarse_input(
    make_torus: Callable[..., TorusMesh], make_session: Callable[..., imb.Session]
) -> None:
    """A 12 x 6 torus cannot carry a 400-vertex field until it is refined."""
    coarse = make_torus(segments_u=12, segments_v=6)
    session = make_session(coarse, vertex_count=400)

    assert len(session.vertices) > 10 * len(coarse.vertices)
    assert len(session.faces) == 2 * len(session.vertices)
    coarse_edges = _edge_lengths(coarse.vertices, coarse.faces)
    assert coarse_edges.max() > 2.0 * session.scale  # the input really was coarse
    assert _edge_lengths(session.vertices, session.faces).max() < session.scale


def test_config_survives_preprocess(
    mesh_session: imb.Session, target_vertex_count: int
) -> None:
    config = mesh_session.config

    assert config.vertex_count == target_vertex_count
    assert config.deterministic
    assert config.rosy == 4 and config.posy == 4
    # A derived scale is written back, so the client can show what it got.
    assert config.scale == pytest.approx(mesh_session.scale)


def test_extraction_options_change_the_output_without_a_rebuild(
    solved_session: imb.Session,
) -> None:
    """The only two config fields extract() reads rather than preprocess().

    The setter exists so a client can change them without rebuilding. It is
    the escape hatch, not the ordinary route for ``pure_quad``: preprocess
    aims the resolution target at a quarter of itself when subdivision is on,
    and setting the flag here leaves that aim where it was -- which is exactly
    the fourfold overshoot the test below pins down.
    """
    vertices = solved_session.vertices.copy()
    mixed = solved_session.extract()

    solved_session.set_extraction_options(smooth_iter=3, pure_quad=True)
    pure = solved_session.extract()

    assert solved_session.config.pure_quad is True
    assert solved_session.config.smooth_iter == 3
    # Subdividing away the triangles can only add faces.
    assert pure.faces.shape[0] > mixed.faces.shape[0]
    # The working mesh is the thing that must not have moved.
    np.testing.assert_array_equal(solved_session.vertices, vertices)


def test_a_pure_quad_target_counts_the_mesh_that_comes_out(torus) -> None:
    """The resolution target names the output, subdivision included.

    A pure quad mesh is produced by splitting every extracted quad into four,
    so a field aimed straight at the target overshoots it roughly fourfold --
    a request for 2,500 vertices used to return ten thousand. preprocess aims
    at a quarter instead, and hands the request back untouched so a UI's own
    box is never rewritten.
    """
    target = 800
    counts = {}
    for pure_quad in (False, True):
        session = imb.Session()
        session.set_mesh(*torus)
        session.preprocess(
            imb.Config(vertex_count=target, pure_quad=pure_quad, deterministic=True)
        )
        session.solve_all()
        counts[pure_quad] = session.extract().vertices.shape[0]
        assert session.config.vertex_count == target

    assert counts[True] == pytest.approx(target, rel=0.25)
    # Both land near the target, so neither is a multiple of the other.
    assert counts[True] == pytest.approx(counts[False], rel=0.25)


def test_session_rejects_use_before_preprocess() -> None:
    session = imb.Session()

    with pytest.raises(RuntimeError, match="preprocess"):
        session.vertices


# ---------------------------------------------------------------------------
#  Solving
# ---------------------------------------------------------------------------


def test_solve_all_terminates_and_leaves_a_usable_field(
    solved_session: imb.Session,
) -> None:
    status = solved_session.status

    assert not status.active
    assert status.progress == pytest.approx(1.0)
    # The solver thread swallows its exceptions and parks the message here, so
    # without this a failed sweep would only show up as a strange-looking field.
    # status is read once: the C++ side clears the slot as it hands it over.
    assert status.error == ""

    orientations = solved_session.orientation_field
    assert orientations.shape == solved_session.vertices.shape
    assert np.abs(np.linalg.norm(orientations, axis=1) - 1.0).max() < 1e-5

    # Every O sample is a point on the surface near its own vertex: it stays in
    # that vertex's tangent plane and within one target edge length of it.
    offsets = solved_session.position_field - solved_session.vertices
    off_plane = np.abs((offsets * solved_session.normals).sum(1))
    assert off_plane.max() < 1e-5 * solved_session.scale
    assert np.linalg.norm(offsets, axis=1).max() < solved_session.scale


def test_orientation_field_is_tangent_to_the_surface(
    solved_session: imb.Session,
) -> None:
    normals = solved_session.normals
    tangency = np.abs((solved_session.orientation_field * normals).sum(1))

    assert tangency.max() < 1e-5


def test_interactive_solve_bumps_the_version_counters(
    solved_session: imb.Session,
) -> None:
    """A streamer resends Q only when iterations_q moves, so it has to move."""
    solved_session.set_preview_interval(1)
    before = solved_session.status.iterations_q

    solved_session.solve_orientations(0)
    try:
        deadline = time.monotonic() + 5.0
        while True:
            # Each read clears any parked solver error, so the poll loop is the
            # only place one can be caught: a sweep that threw would otherwise
            # look exactly like a sweep that published nothing.
            status = solved_session.status
            assert status.error == "", status.error
            assert status.level == 0
            if status.iterations_q != before:
                break
            assert time.monotonic() < deadline, "level-0 solve published nothing"
            time.sleep(0.005)
    finally:
        solved_session.stop_solve()
        solved_session.wait_solve()

    assert not solved_session.status.active
    assert solved_session.status.iterations_q > before


def test_deterministic_solve_is_bit_identical_across_sessions(
    make_session: Callable[..., imb.Session],
) -> None:
    """Regression test for the TBB shim's deterministic reduction.

    Two sessions built from the same input must agree to the last bit; a
    reduction that folded its partials in completion order would not.
    """
    first = make_session(deterministic=True)
    second = make_session(deterministic=True)
    first.solve_all()
    second.solve_all()

    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.faces, second.faces)
    assert first.scale == second.scale
    assert np.array_equal(first.orientation_field, second.orientation_field)
    assert np.array_equal(first.position_field, second.position_field)
    assert np.array_equal(first.extract().faces, second.extract().faces)


@pytest.mark.parametrize("threads", [1, -1])
def test_thread_count_resize_still_solves_correctly(
    make_session: Callable[..., imb.Session], threads: int
) -> None:
    """The pool size must change the schedule, never the answer."""
    reference = make_session()
    reference.solve_all()
    expected_q = reference.orientation_field.copy()
    expected_o = reference.position_field.copy()

    imb.set_thread_count(threads)
    try:
        resized = make_session()
        resized.solve_all()
        orientations = resized.orientation_field
        positions = resized.position_field
        extracted = resized.extract()
    finally:
        imb.set_thread_count(-1)

    assert np.abs(np.linalg.norm(orientations, axis=1) - 1.0).max() < 1e-5
    assert extracted.faces.max() < len(extracted.vertices)
    # deterministic=True promises the same bits on one core as on all of them.
    assert np.array_equal(orientations, expected_q)
    assert np.array_equal(positions, expected_o)


# ---------------------------------------------------------------------------
#  Brush strokes
# ---------------------------------------------------------------------------


def test_project_stroke_returns_none_when_the_rays_miss(
    solved_session: imb.Session, torus: TorusMesh
) -> None:
    origins, directions = _diagonal_stroke_rays(torus)

    assert solved_session.project_stroke(origins, -directions) is None


def test_project_stroke_keeps_going_when_a_ray_misses(
    solved_session: imb.Session, torus: TorusMesh
) -> None:
    """A drag that starts beside the model still says where the flow should go.

    The rays that miss are dropped rather than rejecting the whole stroke,
    because in a browser -- unlike the desktop GUI -- a sweep across the
    silhouette is the natural way to comb an edge of the model.
    """
    origins, directions = _diagonal_stroke_rays(torus, samples=30)
    targets = _stroke_targets(origins, directions)
    spacing = np.linalg.norm(np.diff(targets, axis=0), axis=1).max()

    # Aim the first third into empty space, as a drag begun off the model does.
    lead = len(origins) // 3
    aimed_away = directions.copy()
    aimed_away[:lead] = -aimed_away[:lead]

    curve = solved_session.project_stroke(origins, aimed_away)

    assert curve is not None
    assert curve.faces.max() < len(solved_session.faces)
    # What survived is the part the rays actually hit, and only that part.
    assert _covers(curve.positions, targets[lead:]) < spacing
    missed = np.linalg.norm(curve.positions - targets[0], axis=1).min()
    assert missed > spacing, "the stroke covered ground no ray reached"


def test_project_stroke_takes_the_longer_side_of_a_gap(
    solved_session: imb.Session, torus: TorusMesh
) -> None:
    """Two runs of hits are two sweeps, not one with a shortcut between them."""
    origins, directions = _diagonal_stroke_rays(torus, samples=30)
    targets = _stroke_targets(origins, directions)
    spacing = np.linalg.norm(np.diff(targets, axis=0), axis=1).max()

    # A short run, a gap, then a long one: joining the two would draw a path
    # across ground the rays found nothing on.
    gapped = directions.copy()
    gapped[4:12] = -gapped[4:12]

    curve = solved_session.project_stroke(origins, gapped)

    assert curve is not None
    assert _covers(curve.positions, targets[12:]) < spacing, "the curve jumped the gap"


def test_project_stroke_steps_inside_the_silhouette(
    solved_session: imb.Session, torus: TorusMesh
) -> None:
    """Endpoints land on surface facing the viewer, not on the outline itself.

    A stroke that ends exactly on the silhouette combs whichever sliver of
    side-facing surface happens to be under the cursor -- a direction the user
    cannot see, which shows up as a kink in the flow along the outline.
    """
    # Straight down the z axis, sweeping inward from beyond the outer radius.
    # The first rays miss; the next graze the outer equator, whose normal lies
    # in the xy plane and so is perpendicular to the view; by the tube's crown
    # the surface faces the camera squarely.
    view = np.array([0.0, 0.0, -1.0], np.float32)
    outside = torus.major_radius + torus.minor_radius
    x = np.linspace(outside * 1.2, torus.major_radius, 40, dtype=np.float32)
    origins = np.stack([x, np.zeros_like(x), np.full_like(x, 9.0)], 1)
    directions = np.tile(view, (len(x), 1))

    curve = solved_session.project_stroke(origins, directions)

    assert curve is not None
    assert 2 <= len(curve) < len(x), "the rays that found nothing must be dropped"

    facing = np.abs(curve.normals @ view)
    assert facing[0] > 0.15, f"the stroke still begins on the silhouette ({facing[0]:.3f})"
    assert facing[-1] > 0.15


def test_project_stroke_lands_on_the_faces_it_reports(
    solved_session: imb.Session, torus: TorusMesh
) -> None:
    curve = solved_session.project_stroke(*_diagonal_stroke_rays(torus))

    assert curve is not None
    positions, faces = curve.positions, curve.faces
    assert len(curve) == len(positions) == len(faces)
    assert positions.dtype == np.float32 and faces.dtype == np.uint32
    assert faces.max() < len(solved_session.faces)
    assert np.abs(np.linalg.norm(curve.normals, axis=1) - 1.0).max() < 1e-5

    corners = solved_session.vertices[solved_session.faces[faces]]
    longest = np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2).max(1)
    plane = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    plane /= np.linalg.norm(plane, axis=1, keepdims=True)

    off_plane = np.abs(((positions - corners[:, 0]) * plane).sum(1))
    to_corner = np.linalg.norm(corners - positions[:, None, :], axis=2).min(1)
    assert (off_plane < 1e-4 * longest).all()
    assert (to_corner < longest).all()


def test_orientation_stroke_turns_the_field_toward_it(
    solved_session: imb.Session, torus: TorusMesh
) -> None:
    curve = solved_session.project_stroke(*_diagonal_stroke_rays(torus))
    assert curve is not None
    points = curve.positions
    tangents = _polyline_tangents(points)

    near, nearest_sample = _vertices_near(solved_session, points)
    assert len(near) > 50, "the stroke should touch a decent patch of the mesh"
    wanted = tangents[nearest_sample[near]]
    normals = solved_session.normals[near]

    before = _cross_field_deviation(
        solved_session.orientation_field[near], normals, wanted
    )
    solved_session.add_stroke(imb.StrokeKind.ORIENTATION, curve)
    solved_session.solve_orientations()
    solved_session.wait_solve()
    assert solved_session.status.error == ""
    after = _cross_field_deviation(
        solved_session.orientation_field[near], normals, wanted
    )

    assert after.mean() < 0.5 * before.mean()
    assert after.mean() < 6.0


def test_strokes_are_listed_and_erased(
    solved_session: imb.Session, torus: TorusMesh
) -> None:
    t = np.linspace(0.0, 1.0, 12)
    comb_rays = _rays_onto_surface(torus, 0.3 + 0.9 * t, -0.5 * t)
    edge_rays = _rays_onto_surface(torus, 3.4 + 0.9 * t, 0.5 * t)
    comb = solved_session.project_stroke(*comb_rays)
    edge = solved_session.project_stroke(*edge_rays)
    assert comb is not None and edge is not None

    comb_id = solved_session.add_stroke(imb.StrokeKind.ORIENTATION, comb)
    edge_id = solved_session.add_stroke(imb.StrokeKind.EDGE, edge)

    listed = solved_session.strokes
    assert [stroke["id"] for stroke in listed] == [comb_id, edge_id]
    assert [stroke["kind"] for stroke in listed] == [
        imb.StrokeKind.ORIENTATION,
        imb.StrokeKind.EDGE,
    ]
    assert len(listed[0]["curve"]) == len(comb)
    assert np.array_equal(listed[1]["curve"].positions, edge.positions)

    assert not solved_session.erase_stroke(comb_id + edge_id + 1)
    assert solved_session.erase_stroke(comb_id)
    assert [stroke["id"] for stroke in solved_session.strokes] == [edge_id]

    solved_session.clear_strokes()
    assert solved_session.strokes == []


def test_erase_stroke_near_picks_the_visible_handle(
    solved_session: imb.Session, torus: TorusMesh
) -> None:
    """The headless stand-in for clicking a stroke's delete marker.

    The marker floats a tenth of an edge length off the surface above the
    stroke's first point, and the C++ side ignores handles the eye cannot see,
    so a click from the far side of a closed mesh must not delete anything.
    """
    t = np.linspace(0.0, 1.0, 12)
    near_rays = _rays_onto_surface(torus, 0.3 + 0.9 * t, -0.5 * t)
    far_rays = _rays_onto_surface(torus, 3.4 + 0.9 * t, 0.5 * t)
    near = solved_session.project_stroke(*near_rays)
    far = solved_session.project_stroke(*far_rays)
    assert near is not None and far is not None

    near_id = solved_session.add_stroke(imb.StrokeKind.ORIENTATION, near)
    far_id = solved_session.add_stroke(imb.StrokeKind.EDGE, far)

    start, normal = near.positions[0], near.normals[0]
    handle = start + (solved_session.stats.average_edge_length / 10.0) * normal
    eye = start + 8.0 * normal
    radius = 0.5 * solved_session.scale

    assert solved_session.erase_stroke_near(handle + 5.0 * normal, eye, radius) == 0
    # Same handle seen from under the surface, which has to occlude it.
    assert solved_session.erase_stroke_near(handle, start - 8.0 * normal, radius) == 0
    assert [stroke["id"] for stroke in solved_session.strokes] == [near_id, far_id]

    assert solved_session.erase_stroke_near(handle, eye, radius) == near_id
    assert [stroke["id"] for stroke in solved_session.strokes] == [far_id]
    # Nothing left within reach of that handle, so a second click is a no-op.
    assert solved_session.erase_stroke_near(handle, eye, radius) == 0


def test_preprocess_carries_strokes_onto_the_rebuilt_mesh(
    mesh_session: imb.Session, torus: TorusMesh
) -> None:
    """Re-targeting the resolution must not throw away the user's brushwork.

    Only the face indices inside a stroke go stale: the curve is a path in
    space, and the preprocessor refines the surface it was drawn on rather than
    moving it. The re-projected curve therefore has to land on the same path,
    on faces of the mesh that now exists.
    """
    curve = mesh_session.project_stroke(*_diagonal_stroke_rays(torus))
    assert curve is not None
    stroke_id = mesh_session.add_stroke(imb.StrokeKind.ORIENTATION, curve)
    before = np.asarray(curve.positions, dtype=np.float64)

    mesh_session.preprocess(imb.Config(vertex_count=4000))

    carried = mesh_session.strokes
    assert [s["id"] for s in carried] == [stroke_id]
    assert carried[0]["kind"] == int(imb.StrokeKind.ORIENTATION)

    after = np.asarray(carried[0]["curve"].positions, dtype=np.float64)
    assert len(after) >= 2
    assert carried[0]["curve"].faces.max() < len(mesh_session.faces)
    # Every point of the carried curve lies on the path that was drawn.
    strayed = np.linalg.norm(after[:, None] - before[None], axis=2).min(1).max()
    assert strayed < mesh_session.scale, f"the stroke moved by {strayed:.4f}"


def test_a_new_mesh_does_not_inherit_the_previous_strokes(
    mesh_session: imb.Session, torus: TorusMesh, make_torus
) -> None:
    """Carrying strokes across a rebuild must not carry them across a model.

    They would be re-projected onto whatever occupied the same space, which for
    two meshes of similar size means silently keeping constraints drawn on a
    different object.
    """
    curve = mesh_session.project_stroke(*_diagonal_stroke_rays(torus))
    assert curve is not None
    mesh_session.add_stroke(imb.StrokeKind.ORIENTATION, curve)

    other = make_torus(major_radius=1.0, minor_radius=0.34)
    mesh_session.set_mesh(other.vertices, other.faces)
    mesh_session.preprocess(imb.Config(vertex_count=150, deterministic=True))

    assert mesh_session.strokes == []


# ---------------------------------------------------------------------------
#  Extraction and export
# ---------------------------------------------------------------------------


def test_extract_returns_quads_of_the_requested_density(
    solved_session: imb.Session, target_vertex_count: int
) -> None:
    mesh = solved_session.extract()

    assert mesh.vertices.dtype == np.float32 and mesh.vertices.shape[1] == 3
    assert mesh.faces.dtype == np.uint32 and mesh.faces.shape[1] == 4
    assert mesh.faces.max() < len(mesh.vertices)
    # Every extracted vertex has to be reachable, or the client uploads a buffer
    # with holes in it that nothing draws.
    assert len(np.unique(mesh.faces)) == len(mesh.vertices)
    assert len(mesh.face_normals) == len(mesh.faces)
    assert np.abs(np.linalg.norm(mesh.face_normals, axis=1) - 1.0).max() < 1e-5

    # One GL_LINES segment per face side, two endpoints each.
    assert mesh.wireframe.shape == (2 * len(mesh.faces) * 4, 3)
    assert mesh.wireframe_color.shape == mesh.wireframe.shape

    assert 0.5 * target_vertex_count < len(mesh.vertices) < 2.0 * target_vertex_count
    # A quad mesh on a closed surface has about as many faces as vertices.
    assert 0.5 * len(mesh.vertices) < len(mesh.faces) < 2.0 * len(mesh.vertices)


def test_extracted_triangles_repeat_their_last_index(
    solved_session: imb.Session,
) -> None:
    """The (nF, 4) layout encodes a triangle by duplicating a corner."""
    faces = solved_session.extract().faces
    triangles = faces[:, 2] == faces[:, 3]

    # Both bounds matter: with none the test would pass on a mesh that never
    # exercises the encoding, and with all of them the extractor found no quads.
    assert 0 < triangles.sum() < len(faces)
    # A repeated corner is only ever the last one, and never more than one.
    assert not (faces[:, 0] == faces[:, 1]).any()
    assert not (faces[:, 1] == faces[:, 2]).any()
    assert not (faces[~triangles, 0] == faces[~triangles, 3]).any()


def test_pure_quad_extraction_has_no_triangles(
    make_session: Callable[..., imb.Session],
) -> None:
    session = make_session(pure_quad=True)
    session.solve_all()

    faces = session.extract().faces

    assert not (faces[:, 2] == faces[:, 3]).any()


def test_singularities_index_real_faces(solved_session: imb.Session) -> None:
    face_count = len(solved_session.faces)
    orientation = solved_session.orientation_singularities
    position = solved_session.position_singularities

    # A genus-1 surface has total index zero, so a 4-RoSy field on it can be
    # singularity-free -- but the ones that exist must sit on real faces.
    assert all(0 <= face < face_count for face in orientation)
    assert all(index in (1, 3) for index in orientation.values())
    assert all(0 <= face < face_count for face in position)
    assert all(len(shift) == 2 for shift in position.values())


def test_write_mesh_round_trips_through_trimesh(
    make_session: Callable[..., imb.Session], tmp_path
) -> None:
    # trimesh only ships with the "app" extra; the core tests must not need it.
    trimesh = pytest.importorskip("trimesh")
    session = make_session(pure_quad=True)
    session.solve_all()
    mesh = session.extract()

    bounds = np.stack([mesh.vertices.min(0), mesh.vertices.max(0)])
    for suffix in (".obj", ".ply"):
        path = tmp_path / ("quads" + suffix)
        session.write_mesh(str(path), mesh)

        assert path.stat().st_size > 0
        loaded = trimesh.load(str(path), process=False)
        # trimesh triangulates on load, so each quad arrives as two triangles;
        # how many vertices that leaves depends on the loader, the geometry it
        # covers does not.
        assert len(loaded.faces) == 2 * len(mesh.faces)
        assert len(loaded.vertices) >= len(mesh.vertices)
        assert np.allclose(loaded.bounds, bounds, atol=1e-5)


def test_written_ply_declares_the_faces_it_writes(
    solved_session: imb.Session, tmp_path
) -> None:
    """A mixed triangle/quad extraction is where the PLY writer's bookkeeping
    can drift: it collapses degenerate quads into n-gons and must count them."""
    mesh = solved_session.extract()
    assert (mesh.faces[:, 2] == mesh.faces[:, 3]).any(), "expected some triangles"
    path = tmp_path / "mixed.ply"

    solved_session.write_mesh(str(path), mesh)

    declared, sizes = _ply_face_records(str(path))
    assert declared == len(sizes)
    assert min(sizes) >= 3
    assert max(sizes) > 3


def test_write_mesh_rejects_an_unknown_extension(
    solved_session: imb.Session, tmp_path
) -> None:
    with pytest.raises(RuntimeError):
        solved_session.write_mesh(str(tmp_path / "mesh.stl"), solved_session.extract())


# ---------------------------------------------------------------------------
#  Lifetime
# ---------------------------------------------------------------------------


def test_sessions_do_not_leak_solver_threads(
    torus: TorusMesh, target_vertex_count: int
) -> None:
    """Each Session owns a C++ solver thread that only its destructor stops."""

    def one_session() -> None:
        session = imb.Session()
        session.set_mesh(torus.vertices, torus.faces)
        session.preprocess(imb.Config(vertex_count=target_vertex_count))
        session.solve_all()

    # Warm up first: the worker pool spawns on demand and then lives forever, so
    # counting before it exists would measure the pool, not a leak.
    one_session()
    baseline_python = threading.active_count()
    baseline_os = _os_thread_count()

    for _ in range(10):
        one_session()

    assert threading.active_count() == baseline_python
    # A failed snapshot reports None rather than a count; comparing that to the
    # baseline would fail the test for the tooling rather than for a leak.
    final_os = _os_thread_count()
    if baseline_os is not None and final_os is not None:
        assert final_os == baseline_os
