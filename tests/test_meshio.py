"""Mesh file I/O, and the n-gon round trip in particular.

An extracted quad mesh is written as polygons -- the triangle fans around
irregular vertices are reassembled into n-gons, so a file routinely contains
pentagons and larger. Reading one back has to recover the same surface, which
the loader could not do while it read a fixed four corners per face: every
polygon with five or more sides lost a wedge, leaving holes scattered over the
model.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pytest

import instant_meshes_brush as imb


def boundary_edges(polygons: Iterable[Sequence[int]]) -> int:
    """Edges used by exactly one polygon, i.e. the size of the mesh's holes."""
    counts: collections.Counter = collections.Counter()
    for poly in polygons:
        for i in range(len(poly)):
            counts[tuple(sorted((poly[i], poly[(i + 1) % len(poly)])))] += 1
    return sum(1 for used in counts.values() if used == 1)


def read_obj_polygons(path: Path) -> List[Tuple[int, ...]]:
    polygons = []
    for line in path.read_text().splitlines():
        if line.startswith("f "):
            polygons.append(tuple(int(p.split("/")[0]) - 1 for p in line.split()[1:]))
    return polygons


def write_obj(path: Path, vertices: np.ndarray, polygons: Sequence[Sequence[int]]) -> None:
    with path.open("w") as handle:
        for v in vertices:
            handle.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for poly in polygons:
            handle.write("f " + " ".join(str(i + 1) for i in poly) + "\n")


def hexagonal_prism_cap(radius: float = 1.0, sides: int = 6):
    """A fan-free disc: one n-gon cap plus a skirt, so the cap is the only n-gon."""
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    top = np.stack([radius * np.cos(angles), radius * np.sin(angles),
                    np.ones(sides)], axis=1)
    bottom = top * np.array([1.0, 1.0, -1.0])
    vertices = np.concatenate([top, bottom]).astype(np.float32)

    polygons: List[Tuple[int, ...]] = [tuple(range(sides))]                 # top n-gon
    polygons.append(tuple(range(2 * sides - 1, sides - 1, -1)))             # bottom n-gon
    for i in range(sides):                                                  # skirt quads
        j = (i + 1) % sides
        polygons.append((i, sides + i, sides + j, j))
    return vertices, polygons


@pytest.mark.parametrize("sides", [3, 4, 5, 6, 8, 12])
def test_polygons_of_every_degree_survive_a_round_trip(tmp_path: Path, sides: int) -> None:
    """A closed solid must still be closed after being written and read back."""
    vertices, polygons = hexagonal_prism_cap(sides=sides)
    assert boundary_edges(polygons) == 0, "the fixture itself must be closed"

    path = tmp_path / f"prism_{sides}.obj"
    write_obj(path, vertices, polygons)

    session = imb.Session()
    session.load_file(str(path))
    session.preprocess(imb.Config(vertex_count=80, deterministic=True))

    faces = session.faces
    assert boundary_edges(faces) == 0, (
        f"a {sides}-sided polygon lost corners on load, opening {boundary_edges(faces)} "
        "boundary edges in a closed solid"
    )


def test_an_ngon_face_contributes_every_triangle(tmp_path: Path) -> None:
    """A k-gon fans into k-2 triangles; reading fewer means corners were dropped."""
    for sides in (3, 4, 5, 7, 9):
        vertices, polygons = hexagonal_prism_cap(sides=sides)
        path = tmp_path / f"count_{sides}.obj"
        write_obj(path, vertices, polygons)

        session = imb.Session()
        session.load_file(str(path))
        # A large target keeps preprocess from subdividing, so the triangle
        # count is exactly what the loader produced.
        session.preprocess(imb.Config(scale=10.0, deterministic=True))

        expected = sum(len(p) - 2 for p in polygons)
        assert len(session.faces) == expected, (
            f"{sides}-gon fixture: loader produced {len(session.faces)} triangles, "
            f"expected {expected}"
        )


def test_extracted_mesh_round_trips_without_gaining_holes(tmp_path: Path, torus) -> None:
    """Export then re-import must not invent holes in a closed model."""
    session = imb.Session()
    session.set_mesh(torus.vertices, torus.faces)
    session.preprocess(imb.Config(vertex_count=300, deterministic=True))
    session.solve_all()
    mesh = session.extract()

    path = tmp_path / "torus_out.obj"
    session.write_mesh(str(path), mesh)

    written = read_obj_polygons(path)
    assert max(len(p) for p in written) > 4, "fixture should exercise real n-gons"
    assert boundary_edges(written) == 0

    reloaded = imb.Session()
    reloaded.load_file(str(path))
    reloaded.preprocess(imb.Config(scale=10.0, deterministic=True))
    assert boundary_edges(reloaded.faces) == 0


def test_faces_with_too_few_corners_are_skipped(tmp_path: Path) -> None:
    """A stray 'f a b' line must not corrupt the index stream that follows."""
    vertices, polygons = hexagonal_prism_cap(sides=5)
    path = tmp_path / "degenerate.obj"
    write_obj(path, vertices, polygons)
    with path.open("a") as handle:
        handle.write("f 1 2\n")

    session = imb.Session()
    session.load_file(str(path))
    session.preprocess(imb.Config(scale=10.0, deterministic=True))
    assert boundary_edges(session.faces) == 0
