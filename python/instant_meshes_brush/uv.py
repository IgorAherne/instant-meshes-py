"""UV unwrapping of the extracted mesh, through xatlas.

Instant Meshes produces topology, not a texture space, and the two are solved
by different algorithms: the field decides where the edge loops run, xatlas
decides where to cut the result open so it lies flat.  Running the second on
the first is the last step of a retopology pass, and it costs one call --
provided the quads survive it.

They mostly do here.  xatlas is given the triangulated output and hands back a
vertex list split along its seams; this module walks that answer back onto the
original quads, so a face keeps its four corners and simply gains a texture
coordinate per corner.  The alternative -- exporting xatlas's own triangles --
would throw away the only thing the remesher was run for.

The exception is the face a chart boundary runs *through*: xatlas charts
triangles, and a boundary can follow the diagonal of one of our quads, which is
not an edge of the output mesh at all.  Two or three per cent land that way,
and they leave as their two triangles -- a hole in the atlas would be worse for
a texture bake than a triangle is for the topology.

xatlas is optional.  :func:`available` is false when it is not installed, and
nothing here is reached in that case; the viewer hides its UV control instead
of offering one that cannot work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

#: Chart cost the leniency slider spans, as xatlas's ``ChartOptions.max_cost``.
#:
#: That number is the total weighted distortion a chart may accumulate before
#: xatlas stops growing it and cuts a seam, so it *is* the "how large may a
#: chunk get" control and needs no other knob beside it.
#:
#: The range is below xatlas's default of 2.0 rather than around it, because
#: the cost only bites on the way down: raising it past the default changes
#: nothing -- on a 2,300-quad torus the chart count sat at 14 from 2.0 all the
#: way to 1000, while 0.02 gave 69.  The top of the slider is therefore "what
#: xatlas would have done", the fewest chunks it is willing to cut, and the
#: rest of the travel buys smaller ones.
MIN_CHART_COST = 0.02
MAX_CHART_COST = 4.0

#: Where the slider sits until somebody moves it.
DEFAULT_LENIENCY = 0.5


class UnwrapError(RuntimeError):
    """Raised when a layout cannot be produced from the mesh as given."""


#: The imported module, False once an import has been tried and failed, and
#: None before either.  A failed import is not cached by Python -- every
#: attempt walks sys.path again -- and this question is asked on every status
#: frame, which is several times a second per open viewport.
_MODULE: Any = None


def available() -> bool:
    """True when xatlas can be imported, and the UV controls are worth showing."""
    return _import_xatlas(required=False) is not None


def _import_xatlas(required: bool = True) -> Any:
    global _MODULE
    if _MODULE is None:
        try:
            import xatlas  # noqa: PLC0415 -- optional, and only paid for when used
        except ImportError:
            _MODULE = False
        else:
            _MODULE = xatlas
    if _MODULE is False:
        if required:
            raise UnwrapError(
                "UV unwrapping needs the xatlas package: pip install xatlas"
            )
        return None
    return _MODULE


def chart_cost(leniency: float) -> float:
    """Map a slider position in [0, 1] onto xatlas's chart cost ceiling.

    Geometric rather than linear: the interesting differences are between 0.5
    and 2, not between 7 and 8, and a linear track would spend most of its
    length on settings that all look the same.
    """
    t = min(1.0, max(0.0, float(leniency)))
    return float(MIN_CHART_COST * (MAX_CHART_COST / MIN_CHART_COST) ** t)


@dataclass(frozen=True)
class UvLayout:
    """A texture space for one extracted mesh.

    ``faces`` has the shape of the mesh's own face array and indexes ``uv``
    rather than the vertex positions: a corner on a seam belongs to two charts
    and therefore has two texture coordinates, which is the whole reason the
    two index spaces are separate.

    A row of -1 means the quad did not survive whole.  xatlas charts
    *triangles*, so a chart boundary is free to run down a quad's diagonal --
    an edge that does not exist in the output mesh at all -- and about three
    per cent of them land that way.  Such a face has half of itself in each
    chunk and no single place in the atlas, so it is carried in ``cut``
    instead: its two triangles, each whole in its own chart.  Dropping it
    would leave a hole in the atlas, and a hole is worse for a texture bake
    than a triangle.
    """

    uv: np.ndarray  # (nUV, 2) float32, in [0, 1] with v pointing up
    faces: np.ndarray  # (nF, posy) int32 into uv, -1 for a face cut in two
    chart: np.ndarray  # (nF,) int32, -1 likewise
    cut: np.ndarray  # (nC, 3) int32 into uv: the triangles of the cut faces
    cut_face: np.ndarray  # (nC,) int32: which face each came from
    cut_slot: np.ndarray  # (nC, 3) int32: which corners of that face it covers
    cut_chart: np.ndarray  # (nC,) int32
    chart_count: int
    leniency: float

    @property
    def unmapped(self) -> int:
        """Faces with no texture coordinates at all, in whole or in part."""
        textured = self.chart >= 0
        if self.cut_face.size:
            textured[self.cut_face] = True
        return int(np.count_nonzero(~textured))

    @property
    def cut_count(self) -> int:
        """Quads a chart boundary went through, and that are written as pairs."""
        return int(np.unique(self.cut_face).size)


# ---------------------------------------------------------------------------
#  Triangulation
# ---------------------------------------------------------------------------


def _triangulate(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fan each face into triangles, remembering where each corner came from.

    Returns the triangles xatlas is given, plus the face and the three corner
    slots within it that every triangle occupies -- which is what lets the
    answer be written back onto the quads.

    A triangle stored as a degenerate quad (``c2 == c3``, which is how the
    extractor represents one) contributes a single triangle, not two: the
    second would be a zero-area face, and xatlas drops those.
    """
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise UnwrapError(f"expected an (nF, 3+) face array, got shape {faces.shape}")

    count, width = faces.shape
    degenerate = (
        faces[:, width - 1] == faces[:, width - 2] if width > 3 else np.zeros(count, bool)
    )

    tris: List[np.ndarray] = []
    tri_face: List[np.ndarray] = []
    tri_corner: List[np.ndarray] = []
    for i in range(2, width):
        # The last fan triangle of a degenerate quad is the one to skip.
        keep = ~degenerate if i == width - 1 and width > 3 else np.ones(count, bool)
        rows = np.flatnonzero(keep)
        if rows.size == 0:
            continue
        tris.append(faces[np.ix_(rows, [0, i - 1, i])])
        tri_face.append(rows)
        tri_corner.append(np.broadcast_to(np.array([0, i - 1, i]), (rows.size, 3)))

    if not tris:
        raise UnwrapError("the mesh has no faces to unwrap")
    return (
        np.ascontiguousarray(np.concatenate(tris), dtype=np.uint32),
        np.concatenate(tri_face).astype(np.int64, copy=False),
        np.concatenate(tri_corner).astype(np.int64, copy=False),
    )


def _corner_indices(
    triangles: np.ndarray,
    tri_face: np.ndarray,
    tri_corner: np.ndarray,
    shape: Tuple[int, int],
    vmapping: np.ndarray,
    out_triangles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scatter xatlas's split vertices back onto the faces they came from.

    Returns the per-face corner indices, the per-triangle ones, and two masks
    of faces the quad view could not hold: those a chart boundary was cut
    through, and those xatlas returned nothing for.  The two are kept apart
    because only the first can be rescued -- a face missing a triangle has to
    be written whole and untextured, or the mesh itself would lose half a
    quad.
    """
    matched = _align(triangles, vmapping, out_triangles)
    corners = np.full(shape, -1, dtype=np.int32)

    placed = matched[:, 0] >= 0
    cut = np.zeros(shape[0], dtype=bool)
    incomplete = np.zeros(shape[0], dtype=bool)
    if not placed.any():
        incomplete[:] = True
        return corners, matched, cut, incomplete

    corners[tri_face[placed][:, None], tri_corner[placed]] = matched[placed]

    # A quad is two triangles to xatlas, and a chart boundary is free to run
    # between them -- down the diagonal, which is not an edge of the output
    # mesh at all.  The two halves then disagree about the corners they share,
    # and taking whichever was written last would put a seam through the
    # middle of a face.  Such a quad has no single place in the atlas and
    # leaves as its two triangles instead.
    written = corners[tri_face[:, None], tri_corner]
    cut[tri_face[placed & (written != matched).any(axis=1)]] = True
    incomplete[tri_face[~placed]] = True
    cut &= ~incomplete

    corners[cut | incomplete] = -1
    return corners, matched, cut, incomplete


def _components(matched: np.ndarray, placed: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label each triangle with the chart it landed in.

    xatlas's Python binding publishes the chart *count* but not which triangle
    went where -- and it does not have to, because the answer is already in the
    geometry: a seam is exactly where a vertex was split in two, so the charts
    are the connected components of the split vertex list.
    """
    labels = np.full(matched.shape[0], -1, dtype=np.int32)
    rows = np.flatnonzero(placed)
    if rows.size == 0:
        return labels, 0

    parent = list(range(int(matched[rows].max()) + 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for row in rows:
        root = find(int(matched[row, 0]))
        for corner in (1, 2):
            other = find(int(matched[row, corner]))
            if other != root:
                parent[other] = root

    seen: dict = {}
    for row in rows:
        root = find(int(matched[row, 0]))
        labels[row] = seen.setdefault(root, len(seen))
    return labels, len(seen)


def _align(
    triangles: np.ndarray, vmapping: np.ndarray, out_triangles: np.ndarray
) -> np.ndarray:
    """Line xatlas's triangles back up with the ones it was given.

    Row ``i`` is the split-vertex triangle for input triangle ``i``, or -1 if
    xatlas returned none for it.
    """
    matched = np.full((triangles.shape[0], 3), -1, dtype=np.int64)
    if out_triangles.size == 0:
        return matched

    restored = vmapping[out_triangles]
    if restored.shape == triangles.shape and np.array_equal(restored, triangles):
        return out_triangles.astype(np.int64, copy=False)

    cursor = 0
    for out_row, mapped in zip(out_triangles, restored):
        while cursor < triangles.shape[0] and not np.array_equal(
            triangles[cursor], mapped
        ):
            cursor += 1
        if cursor >= triangles.shape[0]:
            break
        matched[cursor] = out_row
        cursor += 1
    return matched


def _fill_degenerate_corner(faces: np.ndarray, corners: np.ndarray) -> None:
    """Give a degenerate quad's repeated corner the coordinate of the one it repeats.

    The face keeps four slots so the array stays rectangular; leaving the
    fourth unmapped would make the whole face look unmapped to every reader.
    """
    width = faces.shape[1]
    if width <= 3:
        return
    repeated = faces[:, width - 1] == faces[:, width - 2]
    corners[repeated, width - 1] = corners[repeated, width - 2]


# ---------------------------------------------------------------------------
#  Charts
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  Unwrapping
# ---------------------------------------------------------------------------


def unwrap(
    vertices: np.ndarray, faces: np.ndarray, leniency: float = DEFAULT_LENIENCY
) -> UvLayout:
    """Flatten ``faces`` into a texture space, keeping the faces intact.

    ``vertices`` is (nV, 3) and ``faces`` is (nF, 3) or (nF, 4) as the
    extractor produces it, quad-dominant with triangles stored as degenerate
    quads.  ``leniency`` runs from 0 (many small chunks, little distortion) to
    1 (few large ones).
    """
    xatlas = _import_xatlas()

    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3:
        raise UnwrapError(f"expected an (nV, 3) vertex array, got shape {v.shape}")
    if f.size == 0:
        raise UnwrapError("the mesh has no faces to unwrap")

    triangles, tri_face, tri_corner = _triangulate(f)

    atlas = xatlas.Atlas()
    atlas.add_mesh(v, triangles)
    options = xatlas.ChartOptions()
    options.max_cost = chart_cost(leniency)
    try:
        atlas.generate(chart_options=options)
    except Exception as exc:  # xatlas raises bare RuntimeErrors
        raise UnwrapError(f"xatlas could not unwrap this mesh: {exc}") from exc

    vmapping, out_triangles, coordinates = atlas[0]
    corners, matched, was_cut, incomplete = _corner_indices(
        triangles,
        tri_face,
        tri_corner,
        tuple(f.shape),
        np.asarray(vmapping),
        np.asarray(out_triangles),
    )
    _fill_degenerate_corner(f, corners)

    placed = matched[:, 0] >= 0
    labels, chart_count = _components(matched, placed)

    # A whole face takes the chart of any of its triangles -- they agree, or it
    # would not be whole. The rest travel as triangles, each with its own.
    chart = np.full(f.shape[0], -1, dtype=np.int32)
    intact = placed & ~(was_cut | incomplete)[tri_face]
    chart[tri_face[intact]] = labels[intact]

    cut = placed & was_cut[tri_face]
    return UvLayout(
        uv=_normalise(np.asarray(coordinates, dtype=np.float32), atlas),
        faces=corners,
        chart=chart,
        cut=matched[cut].astype(np.int32, copy=False),
        cut_face=tri_face[cut].astype(np.int32, copy=False),
        cut_slot=tri_corner[cut].astype(np.int32, copy=False),
        cut_chart=labels[cut],
        chart_count=chart_count,
        leniency=float(min(1.0, max(0.0, leniency))),
    )


def _normalise(uv: np.ndarray, atlas: Any) -> np.ndarray:
    """Put the coordinates in [0, 1] with v pointing up.

    xatlas counts its rows downwards; OBJ, and every viewer that reads one,
    expects the unit square the other way up.  Flipping here rather than at
    each consumer keeps the preview and the exported file showing the same
    layout.

    The Python binding hands back coordinates already divided by the atlas
    resolution, while the C library measures them in texels.  Which one arrives
    is decided by looking rather than assumed: dividing an already-normalised
    atlas by its own width again packs the whole layout into a three-thousandth
    of the unit square, which still passes a "within [0, 1]" check.
    """
    out = np.array(uv, dtype=np.float32, copy=True).reshape(-1, 2)
    if out.size == 0:
        return out

    peak = out.max(axis=0)
    if float(peak.max()) > 1.0 + 1e-4:
        extent = np.array(
            [
                float(getattr(atlas, "width", 0) or 0),
                float(getattr(atlas, "height", 0) or 0),
            ],
            dtype=np.float32,
        )
        # An atlas with no resolution of its own still has a bounding box, and
        # scaling by that is the same answer up to the padding xatlas reserved.
        extent = np.where(extent > 0, extent, np.where(peak > 0, peak, 1.0))
        out /= extent

    out[:, 1] = 1.0 - out[:, 1]
    return np.clip(out, 0.0, 1.0, out=out)


# ---------------------------------------------------------------------------
#  Writing
# ---------------------------------------------------------------------------


def write_obj(
    path: Any, vertices: np.ndarray, faces: np.ndarray, layout: Optional[UvLayout] = None
) -> None:
    """Write a quad-dominant OBJ, with per-corner texture coordinates if given.

    The C++ writer in ``meshio.cpp`` emits ``vt`` lines but references only
    ``v//vn`` from its faces, so the coordinates it writes reach no reader.
    Per-corner indices are the whole point of an atlas -- a seam vertex has two
    of them -- so the UV form of the file is written here instead.
    """
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces)

    lines: List[str] = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    if layout is None:
        for face in f:
            lines.append(_face_line(_face_ring(face), None))
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(f"vt {u:.6f} {w:.6f}" for u, w in layout.uv.astype(np.float64))
    pieces = _cut_pieces(layout)

    for index, face in enumerate(f):
        slots = _face_slots(face)
        texture = [int(layout.faces[index, slot]) for slot in slots]
        if all(t >= 0 for t in texture):
            lines.append(_face_line([int(face[s]) for s in slots], texture))
        elif index in pieces:
            # Half of this quad is in one chunk and half in another, so it
            # leaves as the two triangles it was cut into rather than as a
            # face with no texture coordinates at all.
            for corner_slots, corner_uvs in pieces[index]:
                lines.append(
                    _face_line([int(face[s]) for s in corner_slots], list(corner_uvs))
                )
        else:
            # OBJ allows a face with no texture coordinates beside one that
            # has them, which beats a bogus (0, 0) that reads as a real
            # coordinate and smears whatever is painted on it.
            lines.append(_face_line([int(face[s]) for s in slots], None))

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cut_pieces(layout: UvLayout) -> dict:
    """Face index -> its triangles, as (corner slots, uv indices) pairs."""
    pieces: dict = {}
    for row in range(layout.cut_face.shape[0]):
        pieces.setdefault(int(layout.cut_face[row]), []).append(
            (layout.cut_slot[row], layout.cut[row])
        )
    return pieces


def _face_line(ring: Sequence[int], texture: Optional[Sequence[int]]) -> str:
    if texture is None:
        return "f " + " ".join(str(int(c) + 1) for c in ring)
    return "f " + " ".join(
        f"{int(c) + 1}/{int(t) + 1}" for c, t in zip(ring, texture)
    )


def _face_slots(face: Sequence[int]) -> List[int]:
    """The corner slots a face actually uses, dropping a quad's repeated one."""
    width = len(face)
    if width > 3 and face[width - 1] == face[width - 2]:
        return list(range(width - 1))
    return list(range(width))


def _face_ring(face: Sequence[int]) -> List[int]:
    return [int(face[slot]) for slot in _face_slots(face)]
