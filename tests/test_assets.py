"""Reading a model with its materials, and keeping the solver's copy clean.

Two things are being checked here, and they pull in opposite directions.  The
viewport wants the file exactly as authored -- corners split along every UV
seam, faces grouped by material, the maps themselves -- and the remesher wants
one welded triangle soup with no attributes at all, because a hierarchy built
on split corners has a crack running down every seam.  :mod:`assets` produces
both from one read, and most of what can go wrong is one of them quietly
getting the other's mesh.

The FBX half is tested against a file written here rather than a fixture on
disk: what :mod:`fbx_media` has to survive is the shape of the format, and a
file assembled record by record says what shape that is.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import pytest

from instant_meshes_brush import assets, fbx_media

PIL = pytest.importorskip("PIL.Image")
Image = PIL


# ---------------------------------------------------------------------------
#  A binary FBX, written by hand
# ---------------------------------------------------------------------------


def _string(text: bytes) -> bytes:
    return b"S" + struct.pack("<I", len(text)) + text


def _raw(data: bytes) -> bytes:
    return b"R" + struct.pack("<I", len(data)) + data


def _long(value: int) -> bytes:
    return b"L" + struct.pack("<q", value)


def _node(name: bytes, props: bytes, count: int, children: list = ()) -> tuple:
    return (name, props, count, list(children))


def _emit(node: tuple, at: int) -> bytes:
    """Serialise one node and its children at a known absolute offset.

    A record's end offset is absolute, so the whole tree has to be laid out in
    one pass -- which is also why an FBX cannot be assembled bottom-up. A node
    with children closes with a 13-byte null record, exactly as an exporter
    writes it; one without simply stops.
    """
    name, props, count, children = node
    head = 4 + 4 + 4 + 1 + len(name) + len(props)

    body = b""
    cursor = at + head
    for child in children:
        blob = _emit(child, cursor)
        body += blob
        cursor += len(blob)
    if children:
        body += b"\0" * 13
        cursor += 13

    return (
        struct.pack("<III", cursor, count, len(props))
        + bytes([len(name)])
        + name
        + props
        + body
    )


def _png(colour) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buffer, "PNG")
    return buffer.getvalue()


def write_fbx(
    path: Path,
    *,
    embed: bytes,
    binding: bytes = b"NormalMap",
    filename: bytes = b"",
) -> None:
    """A binary FBX with one material, one texture and one image.

    ``embed`` is the image the Video record carries; pass b"" and a
    ``filename`` for the exporter that left its maps on disk instead.
    """
    video_children = [_node(b"Content", _raw(embed), 1)]
    if filename:
        video_children.append(_node(b"RelativeFilename", _string(filename), 1))
    objects = _node(
        b"Objects",
        b"",
        0,
        [
            _node(
                b"Material",
                _long(100) + _string(b"skin\x00\x01Material") + _string(b""),
                3,
            ),
            _node(
                b"Texture",
                _long(200) + _string(b"map\x00\x01Texture") + _string(b""),
                3,
            ),
            _node(
                b"Video",
                _long(300) + _string(b"map\x00\x01Video") + _string(b"Clip"),
                3,
                video_children,
            ),
        ],
    )
    connections = _node(
        b"Connections",
        b"",
        0,
        [
            _node(b"C", _string(b"OO") + _long(300) + _long(200), 3),
            _node(b"C", _string(b"OP") + _long(200) + _long(100) + _string(binding), 4),
        ],
    )

    out = bytearray(fbx_media.MAGIC + struct.pack("<I", 7400))
    for node in (objects, connections):
        out += _emit(node, len(out))
    out += b"\0" * 13
    path.write_bytes(bytes(out))


# ---------------------------------------------------------------------------
#  FBX embedded media
# ---------------------------------------------------------------------------


def test_an_embedded_map_is_found_under_its_material(tmp_path) -> None:
    """The only kind of FBX texture an upload can carry.

    Assimp reads the geometry but answers "*0" for a map the file embeds, with
    no way to ask what "*0" holds, so the bytes come from the file itself.
    """
    path = tmp_path / "model.fbx"
    image = _png((10, 200, 30))
    write_fbx(path, embed=image)

    found = fbx_media.read(path)
    assert list(found) == ["skin"]
    assert found["skin"] == [("normal", image)]


def test_the_bound_property_says_what_kind_of_map_it_is(tmp_path) -> None:
    """DiffuseColor and NormalMap are the same bytes in different channels."""
    cases = {
        b"DiffuseColor": "base colour",
        b"NormalMap": "normal",
        b"Maya|normalCamera": "normal",
        b"SpecularColor": "specular",
        b"EmissiveColor": "emissive",
        b"3dsMax|Parameters|base_color": "base colour",
    }
    for binding, channel in cases.items():
        path = tmp_path / f"{binding.decode().replace('|', '_')}.fbx"
        write_fbx(path, embed=_png((1, 2, 3)), binding=binding)
        assert fbx_media.read(path)["skin"][0][0] == channel, binding


def test_an_unreadable_file_yields_no_maps_rather_than_raising(tmp_path) -> None:
    """The geometry has already loaded by then, and is worth remeshing.

    An ASCII FBX, a truncated one and a file that is not an FBX at all all end
    up here; none of them is a reason to refuse the model.
    """
    ascii_fbx = tmp_path / "ascii.fbx"
    ascii_fbx.write_text("; FBX 7.4.0 project file\nObjects: {\n}\n")
    assert fbx_media.read(ascii_fbx) == {}

    truncated = tmp_path / "truncated.fbx"
    write_fbx(truncated, embed=_png((0, 0, 0)))
    truncated.write_bytes(truncated.read_bytes()[: len(fbx_media.MAGIC) + 8])
    assert fbx_media.read(truncated) == {}

    assert fbx_media.read(tmp_path / "missing.fbx") == {}


def test_an_external_map_is_taken_from_beside_the_file(tmp_path) -> None:
    """An exporter that did not embed still leaves them in the folder.

    That folder only exists for a file opened from disk -- an upload arrives
    alone -- but where it does, the maps are right there.
    """
    image = _png((7, 7, 200))
    (tmp_path / "skin_normal.png").write_bytes(image)
    path = tmp_path / "model.fbx"
    write_fbx(path, embed=b"", filename=b"D:\\work\\maps\\skin_normal.png")

    assert fbx_media.read(path)["skin"] == [("normal", image)]


def test_a_map_outside_the_model_s_own_folder_is_not_followed(tmp_path) -> None:
    """A path out of a file is data, not a licence to read anywhere.

    An uploaded model names its own texture paths, and "..\\..\\secrets" is a
    path like any other; only the file name is used, and only inside the
    folder the model itself came from.
    """
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.png").write_bytes(_png((0, 0, 0)))

    folder = tmp_path / "model"
    folder.mkdir()
    path = folder / "model.fbx"
    write_fbx(path, embed=b"", filename=b"..\\elsewhere\\secret.png")

    assert fbx_media.read(path) == {}


def test_embedded_maps_are_joined_to_assimp_s_materials_by_name() -> None:
    """The two readers meet here, and they only agree about names.

    Assimp numbers the materials it found; the FBX names them. Matching by
    name is what puts the skin map on the skin and not on the jacket, and the
    position is the fallback for a file whose names did not survive.
    """
    embedded = {
        "jacket": [("base colour", b"jacket")],
        "skin": [("normal", b"skin")],
    }
    assert assets._embedded_for("skin", 0, embedded) == [("normal", b"skin")]
    assert assets._embedded_for("jacket", 1, embedded) == [("base colour", b"jacket")]
    # Unnamed: fall back to the order the file listed them in.
    assert assets._embedded_for("", 1, embedded) == [("normal", b"skin")]
    assert assets._embedded_for("", 9, embedded) == []


def test_a_material_carries_the_embedded_map_it_was_matched_to() -> None:
    image = _png((3, 4, 5))
    material = assets._assimp_material(
        {"NAME": "skin"}, 0, {"skin": [("normal", image)]}
    )
    assert material.name == "skin"
    assert list(material.textures) == ["normal"]
    assert material.textures["normal"].channel == "normal"


# ---------------------------------------------------------------------------
#  Channels and slots
# ---------------------------------------------------------------------------


def test_slots_are_numbered_over_the_channels_anybody_filled() -> None:
    """A button per map that exists, not per channel that could exist.

    The viewport numbers them tex 0, tex 1 and so on, and a gap where four
    channels nobody has would sit is four buttons that do nothing.
    """
    source = assets.SourceMesh(
        name="m",
        vertices=np.zeros((3, 3), dtype=np.float32),
        uv=np.zeros((3, 2), dtype=np.float32),
        faces=np.zeros((1, 3), dtype=np.uint32),
        groups=[(0, 0, 1)],
        materials=[
            assets.Material("a", {"base colour": assets.Texture("base colour", "x", b"a")}),
            assets.Material("b", {"emissive": assets.Texture("emissive", "x", b"b")}),
        ],
    )
    assert source.slots == ["base colour", "emissive"]
    # Slot 1 is emissive for every material, and the one without it has none.
    assert source.texture(1, 1).data == b"b"
    assert source.texture(1, 0) is None
    assert source.texture(2, 0) is None


def test_a_map_is_shrunk_to_preview_size() -> None:
    """These are looked at on a model, not sampled for a render.

    A 4096 square map costs 64 MB of texture memory; the viewport is a few
    hundred pixels across and nobody can see the difference.
    """
    big = Image.new("RGB", (4096, 2048), (200, 30, 30))
    texture = assets.encode_texture(big, "base colour")
    assert texture.mime == "image/jpeg"
    with Image.open(io.BytesIO(texture.data)) as decoded:
        assert max(decoded.size) == assets.MAX_TEXTURE_PX
        assert decoded.size == (assets.MAX_TEXTURE_PX, assets.MAX_TEXTURE_PX // 2)


def test_alpha_survives_only_where_it_says_something() -> None:
    """Most exporters write an opaque map as RGBA all the same.

    A PNG of one costs several times the bytes of the JPEG it may as well be,
    and these travel over a socket to a browser.
    """
    opaque = Image.new("RGBA", (16, 16), (10, 20, 30, 255))
    assert assets.encode_texture(opaque, "base colour").mime == "image/jpeg"

    cut_out = Image.new("RGBA", (16, 16), (10, 20, 30, 255))
    cut_out.putpixel((0, 0), (0, 0, 0, 0))
    assert assets.encode_texture(cut_out, "base colour").mime == "image/png"


# ---------------------------------------------------------------------------
#  The solver's copy
# ---------------------------------------------------------------------------


def _split_quad() -> assets.SourceMesh:
    """Two triangles that share an edge, written as two separate corners each.

    This is what a UV seam or a material boundary does to a mesh: the same
    point of the same surface, duplicated so that each side can carry its own
    texture coordinate.
    """
    vertices = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0],   # first triangle
            [1, 0, 0], [1, 1, 0], [0, 1, 0],   # second, sharing two corners
        ],
        dtype=np.float32,
    )
    return assets.SourceMesh(
        name="seam.obj",
        vertices=vertices,
        uv=np.zeros((6, 2), dtype=np.float32),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32),
        groups=[(0, 0, 2)],
        materials=[assets.Material("m")],
    )


def test_the_solver_gets_the_seams_welded_shut() -> None:
    """A hierarchy built on split corners has a crack along every seam.

    The viewport needs them apart -- each side carries its own UV -- so the
    split mesh is kept and the welded one is derived, rather than the reverse.
    """
    source = _split_quad()
    vertices, faces = assets.solver_mesh(source)

    assert vertices.shape == (4, 3), "the shared edge was not welded"
    assert faces.shape == (2, 3)
    # Both triangles survive, and they now share an edge rather than a gap.
    corners = [set(face) for face in faces.tolist()]
    assert len(corners[0] & corners[1]) == 2


def test_a_triangle_that_welds_into_a_line_is_dropped() -> None:
    """The hierarchy builder divides by face area, and this one has none."""
    source = _split_quad()
    # Collapse the second triangle onto a single point.
    source.vertices[3:6] = source.vertices[0]
    _, faces = assets.solver_mesh(source)
    assert faces.shape == (1, 3)


def test_a_model_with_no_triangles_is_refused() -> None:
    empty = assets.SourceMesh(
        name="nothing.obj",
        vertices=np.zeros((0, 3), dtype=np.float32),
        uv=np.zeros((0, 2), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.uint32),
        groups=[],
        materials=[],
    )
    with pytest.raises(assets.AssetError):
        assets.solver_mesh(empty)


# ---------------------------------------------------------------------------
#  Reading a real file
# ---------------------------------------------------------------------------


def test_a_gltf_arrives_with_its_material_and_its_uvs(tmp_path, torus) -> None:
    """glB is the format where everything is in one file, and it all works."""
    trimesh = pytest.importorskip("trimesh")

    mesh = trimesh.Trimesh(
        vertices=np.asarray(torus.vertices, dtype=np.float64),
        faces=np.asarray(torus.faces),
        process=False,
    )
    uv = np.random.default_rng(0).random((len(mesh.vertices), 2))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
            name="shell", baseColorTexture=Image.new("RGB", (32, 32), (90, 140, 200))
        ),
    )
    path = tmp_path / "model.glb"
    path.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))

    source = assets.load_source(path)
    assert source.vertices.shape[0] == len(mesh.vertices)
    assert source.faces.shape == (len(mesh.faces), 3)
    assert source.uv.shape == (source.vertices.shape[0], 2)
    assert source.slots == ["base colour"]
    assert source.texture(0, 0).mime == "image/jpeg"

    # Faces are handed over grouped by material, which is what lets the viewer
    # draw one run per material instead of looking one up per face.
    assert sum(count for _, _, count in source.groups) == source.faces.shape[0]


def test_an_unsupported_suffix_says_what_is_supported(tmp_path) -> None:
    path = tmp_path / "model.blend"
    path.write_bytes(b"not a mesh")
    with pytest.raises(assets.AssetError, match="unsupported mesh format"):
        assets.load_source(path)
