"""The UV atlas: chart sizing, the quad-preserving mapping, and the OBJ form.

xatlas decides where the seams go; everything else in :mod:`instant_meshes_brush.
uv` decides what happens to the quads while it does.  That part is what these
tests pin down, and it is testable without the library: a stand-in atlas lets
each answer xatlas is entitled to give -- a clean unwrap, a seam that splits a
vertex, a face it refused -- be aimed at the mapping deliberately rather than
waited for.

One test at the end runs the real thing, and is skipped where it is not
installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pytest

from instant_meshes_brush import uv

xatlas_installed = pytest.mark.skipif(
    not uv.available(), reason="xatlas is not installed"
)


# ---------------------------------------------------------------------------
#  A stand-in for xatlas
# ---------------------------------------------------------------------------


Plan = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]


class _FakeChartOptions:
    def __init__(self) -> None:
        self.max_cost = 2.0


class _FakeAtlas:
    """Answers with whatever ``plan`` computes from the mesh it was given."""

    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self._result: Tuple[np.ndarray, np.ndarray, np.ndarray] = ()  # type: ignore
        # A resolution the plans below do *not* divide by, as the real Python
        # binding does not: normalising against this would be the bug.
        self.width = 512
        self.height = 512
        self.chart_options: object = None

    def add_mesh(self, positions: np.ndarray, indices: np.ndarray) -> None:
        self._positions = np.asarray(positions)
        self._indices = np.asarray(indices)

    def generate(self, chart_options: object = None, **_: object) -> None:
        self.chart_options = chart_options
        self._result = self._plan(self._positions, self._indices)

    def __getitem__(self, index: int):
        return self._result


class _FakeXatlas:
    ChartOptions = _FakeChartOptions

    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self.atlas: _FakeAtlas | None = None

    def Atlas(self) -> _FakeAtlas:  # noqa: N802 -- mirrors xatlas's own name
        self.atlas = _FakeAtlas(self._plan)
        return self.atlas


@pytest.fixture
def fake_xatlas(monkeypatch) -> Callable[[Plan], _FakeXatlas]:
    """Install a stand-in xatlas and hand back the module that was installed."""

    def install(plan: Plan) -> _FakeXatlas:
        module = _FakeXatlas(plan)
        monkeypatch.setattr(uv, "_MODULE", module)
        return module

    return install


def _planar_uv(positions: np.ndarray) -> np.ndarray:
    """The x/y of each vertex, squeezed into the unit square."""
    flat = np.asarray(positions, dtype=np.float32)[:, :2]
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    return ((flat - lo) / np.maximum(hi - lo, 1e-6)).astype(np.float32)


def identity_plan(positions: np.ndarray, indices: np.ndarray):
    """One chart, no vertex split: the simplest answer xatlas can give."""
    return np.arange(positions.shape[0], dtype=np.uint32), indices, _planar_uv(positions)


#: Where ``split_along_a_grid_line`` cuts.  A grid line rather than the middle
#: of a quad, so both of a face's fan triangles stay on the same side of it.
SEAM_X = 2.0


def split_along_a_grid_line(positions: np.ndarray, indices: np.ndarray):
    """Two charts meeting on a row of mesh edges -- an ordinary seam.

    Every vertex on the boundary ends up in the atlas twice, once per chart,
    which is the case per-corner indices exist for.
    """
    count = positions.shape[0]
    far = np.flatnonzero(positions[indices, 0].mean(axis=1) > SEAM_X)
    used = np.unique(indices[far])
    copy_of = {int(v): count + i for i, v in enumerate(used)}

    out = indices.astype(np.uint32).copy()
    for t in far:
        out[t] = [copy_of[int(v)] for v in indices[t]]

    flat = _planar_uv(positions)
    return (
        np.concatenate([np.arange(count, dtype=np.uint32), used.astype(np.uint32)]),
        out,
        np.concatenate([flat, flat[used]]).astype(np.float32),
    )


def split_first_triangle(positions: np.ndarray, indices: np.ndarray):
    """A chart boundary down one quad's diagonal, which is not a mesh edge.

    xatlas charts triangles, so it may put the two halves of a quad in
    different charts; the quad then has no single place in the atlas.
    """
    count = positions.shape[0]
    vmapping = np.concatenate(
        [np.arange(count, dtype=np.uint32), indices[0].astype(np.uint32)]
    )
    out = indices.astype(np.uint32).copy()
    out[0] = [count, count + 1, count + 2]
    uvs = np.concatenate([_planar_uv(positions), _planar_uv(positions)[indices[0]]])
    return vmapping, out, uvs.astype(np.float32)


def drop_first_triangle(positions: np.ndarray, indices: np.ndarray):
    """xatlas refusing a face, which it is entitled to do for a degenerate one."""
    return np.arange(positions.shape[0], dtype=np.uint32), indices[1:], _planar_uv(positions)


# ---------------------------------------------------------------------------
#  Meshes
# ---------------------------------------------------------------------------


@pytest.fixture
def quad_grid() -> Tuple[np.ndarray, np.ndarray]:
    """A 3x3 grid of quads in the z = 0 plane, as the extractor would store it."""
    side = 4
    xs, ys = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    vertices = np.stack(
        [xs.ravel(), ys.ravel(), np.zeros(side * side)], axis=1
    ).astype(np.float32)

    faces = []
    for i in range(side - 1):
        for j in range(side - 1):
            a = i * side + j
            faces.append([a, a + side, a + side + 1, a + 1])
    return vertices, np.asarray(faces, dtype=np.uint32)


@pytest.fixture
def mixed_faces(quad_grid) -> Tuple[np.ndarray, np.ndarray]:
    """The same grid with one face stored as a triangle -- a degenerate quad."""
    vertices, faces = quad_grid
    faces = faces.copy()
    faces[0, 3] = faces[0, 2]
    return vertices, faces


# ---------------------------------------------------------------------------
#  Chart sizing
# ---------------------------------------------------------------------------


def test_the_slider_spans_xatlas_cost_geometrically() -> None:
    """Every step should change the answer by the same proportion.

    The useful range runs from "cut at the first sign of stretch" to "cut only
    where the surface forces it", and those are a factor apart, not a
    difference: linear spacing would spend most of the track on settings that
    produce the same layout.
    """
    assert uv.chart_cost(0.0) == pytest.approx(uv.MIN_CHART_COST)
    assert uv.chart_cost(1.0) == pytest.approx(uv.MAX_CHART_COST)

    steps = [uv.chart_cost(t / 8) for t in range(9)]
    ratios = [b / a for a, b in zip(steps, steps[1:])]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_the_lenient_end_reaches_what_xatlas_would_have_chosen() -> None:
    """2.0 is the library's own default, and its answer is the largest chunks
    it is willing to cut; raising the ceiling past that changes nothing, so the
    slider has to be able to get there and has no reason to go much further."""
    assert uv.chart_cost(1.0) >= 2.0


def test_a_leniency_outside_the_slider_is_clamped() -> None:
    assert uv.chart_cost(-3.0) == uv.chart_cost(0.0)
    assert uv.chart_cost(11.0) == uv.chart_cost(1.0)


def test_the_leniency_reaches_xatlas(fake_xatlas, quad_grid) -> None:
    module = fake_xatlas(identity_plan)
    uv.unwrap(*quad_grid, leniency=1.0)
    assert module.atlas.chart_options.max_cost == pytest.approx(uv.MAX_CHART_COST)


# ---------------------------------------------------------------------------
#  Keeping the quads
# ---------------------------------------------------------------------------


def test_unwrapping_keeps_every_quad(fake_xatlas, quad_grid) -> None:
    """The remesher's whole output is its quads; unwrapping must not cost them.

    xatlas triangulates, and exporting what it returns would hand back a
    triangle soup with a texture space -- the wrong half of the two things the
    user asked for.
    """
    vertices, faces = quad_grid
    fake_xatlas(identity_plan)
    layout = uv.unwrap(vertices, faces)

    assert layout.faces.shape == faces.shape
    assert layout.unmapped == 0
    # The stand-in splits nothing, so each corner keeps its own vertex's uv.
    assert np.array_equal(layout.faces, faces)


def test_a_corner_on_a_seam_keeps_its_own_coordinate(fake_xatlas, quad_grid) -> None:
    """The point of a per-corner index: one vertex, two places in the atlas."""
    vertices, faces = quad_grid
    fake_xatlas(split_along_a_grid_line)
    layout = uv.unwrap(vertices, faces)

    assert layout.chart_count == 2
    assert layout.unmapped == 0

    islands = [
        set(int(c) for c in layout.faces[layout.chart == chart].ravel())
        for chart in (0, 1)
    ]
    assert islands[0].isdisjoint(islands[1]), "the two charts share a coordinate"

    # A vertex on the seam is used by faces in both charts, under two indices.
    shared = set(int(v) for v in faces[layout.chart == 0].ravel()) & set(
        int(v) for v in faces[layout.chart == 1].ravel()
    )
    assert shared, "the fixture did not actually put a seam through the mesh"


def test_a_quad_cut_down_its_diagonal_leaves_as_two_triangles(
    fake_xatlas, quad_grid
) -> None:
    """Half a quad in each chunk is not a quad the atlas has a place for.

    Its two halves disagree about the corners they share, and keeping either
    answer would run a seam through the middle of a face. Dropping the face
    instead would leave a hole in the atlas, which is worse for a texture bake
    than the triangle pair it really is.
    """
    vertices, faces = quad_grid
    fake_xatlas(split_first_triangle)
    layout = uv.unwrap(vertices, faces)

    assert layout.faces[0].tolist() == [-1] * 4, "the quad was kept regardless"
    assert layout.chart[0] == -1
    assert layout.cut_count == 1
    assert layout.unmapped == 0, "the face still has texture coordinates"

    # Two triangles, in different chunks, covering the quad's four corners.
    assert layout.cut.shape == (2, 3)
    assert sorted(layout.cut_chart.tolist()) == sorted(set(layout.cut_chart.tolist()))
    assert sorted(np.unique(layout.cut_slot).tolist()) == [0, 1, 2, 3]
    assert np.array_equal(layout.faces[1:], faces[1:]), "one face took the rest with it"


def test_a_face_xatlas_refused_is_reported_rather_than_faked(
    fake_xatlas, quad_grid
) -> None:
    """A dropped face must not silently take somebody else's coordinates.

    xatlas answers in the order it was asked, so a missing triangle shifts
    every later one -- and a mapping that assumed the lists lined up would
    texture the whole mesh one face out of step.
    """
    vertices, faces = quad_grid
    fake_xatlas(drop_first_triangle)
    layout = uv.unwrap(vertices, faces)

    assert layout.unmapped == 1
    assert layout.chart[0] == -1
    assert layout.cut_count == 0, "half a quad is not a face to write"
    # Everything else still landed on its own vertices, not on face 0's.
    assert np.array_equal(layout.faces[1:], faces[1:])


def test_a_triangle_stored_as_a_quad_stays_a_triangle(fake_xatlas, mixed_faces) -> None:
    """The extractor's quad-dominant output carries triangles as c2 == c3.

    Fanning that into two triangles would hand xatlas a zero-area face, which
    it drops -- so the face would come back unmapped for no reason at all.
    """
    vertices, faces = mixed_faces
    fake_xatlas(identity_plan)
    layout = uv.unwrap(vertices, faces)

    assert layout.unmapped == 0
    assert layout.faces[0, 3] == layout.faces[0, 2]


def test_the_layout_fills_the_unit_square(fake_xatlas, quad_grid) -> None:
    """Coordinates that already run 0..1 must not be divided by the atlas size.

    The Python binding normalises for you and the C library does not, and the
    difference is invisible to a "within [0, 1]" check: scaling an already
    normalised atlas by its own 512-texel width leaves every chunk in a corner
    two thousandths of the square wide, which is still within [0, 1].
    """
    vertices, faces = quad_grid
    fake_xatlas(identity_plan)
    layout = uv.unwrap(vertices, faces)

    assert layout.uv.min() == pytest.approx(0.0, abs=1e-5)
    assert layout.uv.max() == pytest.approx(1.0, abs=1e-5)


def test_coordinates_in_texels_are_scaled_by_the_atlas(fake_xatlas, quad_grid) -> None:
    """The other convention, which the C library uses, still has to work."""
    vertices, faces = quad_grid

    def in_texels(positions: np.ndarray, indices: np.ndarray):
        vmapping, out, coordinates = identity_plan(positions, indices)
        return vmapping, out, (coordinates * 512).astype(np.float32)

    fake_xatlas(in_texels)
    layout = uv.unwrap(vertices, faces)

    assert layout.uv.min() == pytest.approx(0.0, abs=1e-5)
    assert layout.uv.max() == pytest.approx(1.0, abs=1e-5)


def test_the_layout_points_v_upwards(fake_xatlas, quad_grid) -> None:
    """xatlas counts rows downwards; OBJ and every viewer of one do not."""
    vertices, faces = quad_grid
    fake_xatlas(identity_plan)
    layout = uv.unwrap(vertices, faces)

    corner = int(np.argmax(vertices[:, 1]))
    assert layout.uv[corner, 1] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
#  Writing
# ---------------------------------------------------------------------------


def test_the_written_obj_references_the_coordinates_it_writes(
    fake_xatlas, quad_grid, tmp_path: Path
) -> None:
    """The C++ writer emits `vt` lines that no face mentions, which no reader
    can use. Per-corner references are the whole reason this writer exists."""
    vertices, faces = quad_grid
    fake_xatlas(identity_plan)
    layout = uv.unwrap(vertices, faces)

    path = tmp_path / "grid.obj"
    uv.write_obj(path, vertices, faces, layout)
    text = path.read_text()

    assert text.count("\nvt ") + text.startswith("vt ") == layout.uv.shape[0]
    face_lines = [line for line in text.splitlines() if line.startswith("f ")]
    assert len(face_lines) == faces.shape[0]
    assert all("/" in line for line in face_lines)


def test_an_unmapped_face_is_written_without_texture_indices(
    fake_xatlas, quad_grid, tmp_path: Path
) -> None:
    """Better a face with no UVs than a face pointing at the origin, which
    reads as a real coordinate and smears whatever texture is applied."""
    vertices, faces = quad_grid
    fake_xatlas(drop_first_triangle)
    layout = uv.unwrap(vertices, faces)

    path = tmp_path / "grid.obj"
    uv.write_obj(path, vertices, faces, layout)
    face_lines = [line for line in path.read_text().splitlines() if line.startswith("f ")]

    # Face 0 lost one of its two triangles. Writing only the half that
    # survived would take a hole out of the *mesh*, so the quad is written
    # whole and goes without texture coordinates instead.
    assert len(face_lines) == faces.shape[0]
    assert "/" not in face_lines[0]
    assert len(face_lines[0].split()) == 5, "the quad is still a quad"
    assert all("/" in line for line in face_lines[1:])


def test_a_cut_quad_is_written_as_its_two_textured_triangles(
    fake_xatlas, quad_grid, tmp_path: Path
) -> None:
    vertices, faces = quad_grid
    fake_xatlas(split_first_triangle)
    layout = uv.unwrap(vertices, faces)

    path = tmp_path / "grid.obj"
    uv.write_obj(path, vertices, faces, layout)
    face_lines = [line for line in path.read_text().splitlines() if line.startswith("f ")]

    assert len(face_lines) == faces.shape[0] + 1, "the cut quad became a pair"
    assert all("/" in line for line in face_lines), "every face carries its UVs"
    assert [len(line.split()) for line in face_lines[:2]] == [4, 4], "two triangles"
    # Between them they cover the quad's four corners, once each.
    corners = [line.split()[1:] for line in face_lines[:2]]
    written = [int(c.split("/")[0]) for pair in corners for c in pair]
    assert sorted(set(written)) == sorted(int(c) + 1 for c in faces[0])


def test_the_written_obj_reads_back_with_its_uvs(
    fake_xatlas, quad_grid, tmp_path: Path
) -> None:
    """A file only carries a texture space if something else can find it."""
    trimesh = pytest.importorskip("trimesh")
    vertices, faces = quad_grid
    fake_xatlas(identity_plan)
    layout = uv.unwrap(vertices, faces)

    path = tmp_path / "grid.obj"
    uv.write_obj(path, vertices, faces, layout)
    loaded = trimesh.load(path, force="mesh", process=False)

    assert loaded.visual.uv is not None
    assert len(loaded.visual.uv) == len(loaded.vertices)
    assert loaded.faces.shape[0] == 2 * faces.shape[0], "quads triangulate into pairs"


def test_a_triangle_face_is_written_with_three_corners(
    fake_xatlas, mixed_faces, tmp_path: Path
) -> None:
    vertices, faces = mixed_faces
    fake_xatlas(identity_plan)
    layout = uv.unwrap(vertices, faces)

    path = tmp_path / "mixed.obj"
    uv.write_obj(path, vertices, faces, layout)
    face_lines = [line for line in path.read_text().splitlines() if line.startswith("f ")]

    assert len(face_lines[0].split()) == 4, "'f' plus three corners"
    assert len(face_lines[1].split()) == 5


# ---------------------------------------------------------------------------
#  The real library
# ---------------------------------------------------------------------------


@pytest.fixture
def remeshed_torus(solved_session):
    """A real extracted quad mesh, which is what the atlas is ever cut from."""
    mesh = solved_session.extract()
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces)


@xatlas_installed
def test_a_lenient_setting_makes_fewer_chunks(remeshed_torus) -> None:
    """The one thing the slider promises: right means fewer, larger chunks.

    The range this spans was picked by measurement, not from the library's
    documentation: xatlas's chart cost only bites below its own default, and a
    slider centred on that default moved the count by two.
    """
    vertices, faces = remeshed_torus
    strict = uv.unwrap(vertices, faces, leniency=0.0)
    lenient = uv.unwrap(vertices, faces, leniency=1.0)

    assert strict.chart_count > lenient.chart_count
    assert strict.chart_count >= 2 * lenient.chart_count, "the slider barely moves"


@xatlas_installed
def test_a_real_atlas_covers_every_face(remeshed_torus) -> None:
    """Whole quad or triangle pair, but never a hole: an untextured face in
    the middle of a bake is the failure this mapping exists to avoid."""
    vertices, faces = remeshed_torus
    layout = uv.unwrap(vertices, faces, leniency=uv.DEFAULT_LENIENCY)

    assert layout.unmapped == 0
    assert layout.faces.shape == faces.shape
    # Filling the square, not merely fitting inside it.
    assert layout.uv.min() < 0.02
    assert layout.uv.max() > 0.98
    # Quads really do get cut, which is why the pairs are there at all.
    assert layout.cut_count > 0
    assert layout.cut_count < 0.2 * faces.shape[0]


@xatlas_installed
def test_the_chunks_found_here_are_the_ones_xatlas_made(remeshed_torus) -> None:
    """The chunk count is recovered from the split vertices, not read off.

    xatlas's binding publishes a chart count but not a per-face assignment,
    so the components of its output vertex list stand in for one -- and the
    count it does publish is the check that they are the same thing.
    """
    import xatlas

    vertices, faces = remeshed_torus
    layout = uv.unwrap(vertices, faces, leniency=0.25)

    triangles, _, _ = uv._triangulate(faces)
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, triangles)
    options = xatlas.ChartOptions()
    options.max_cost = uv.chart_cost(0.25)
    atlas.generate(chart_options=options)

    assert layout.chart_count == atlas.chart_count
