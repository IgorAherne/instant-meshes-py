"""Reading a model with its materials, and keeping the solver's copy clean.

Two things are being checked here, and they pull in opposite directions.  The
viewport and the bake want the file exactly as authored -- corners split along
every UV seam, the author's normals, faces grouped by material, the maps
themselves -- and the remesher wants one welded triangle soup with no
attributes at all, because a hierarchy built on split corners has a crack
running down every seam.  :mod:`assets` produces both from one read, and most
of what can go wrong is one of them quietly getting the other's mesh.

The files are written here rather than taken from disk: a glTF, an FBX and a
Collada file assembled piece by piece say exactly what shape each format has,
and what the answer has to be.  A few checks at the end run on real models
from the author's machine and skip everywhere else.
"""

from __future__ import annotations

import io
import json
import struct
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pytest

from instant_meshes_brush import assets, fbx_media
from instant_meshes_brush.materials import MaterialSpec

Image = pytest.importorskip("PIL.Image")
trimesh = pytest.importorskip("trimesh")


def png(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(pixels)).save(buffer, "PNG")
    return buffer.getvalue()


def flat(colour: Sequence[int], size: int = 8) -> np.ndarray:
    return np.tile(np.asarray(colour, dtype=np.uint8), (size, size, 1))


def decoded(texture: assets.Texture) -> np.ndarray:
    with Image.open(io.BytesIO(texture.data)) as image:
        return np.asarray(image)


def as_json(source: assets.SourceMesh) -> Dict[str, Any]:
    """The viewer's description of a model, through JSON as strict as a browser's."""
    return json.loads(json.dumps(source.describe(), allow_nan=False))


def face_normals(source: assets.SourceMesh) -> np.ndarray:
    corners = source.vertices.astype(np.float64)[source.faces.astype(np.int64)]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
#  A glTF, written by hand
# ---------------------------------------------------------------------------


class Gltf:
    """Just enough of a glTF writer to say precisely what a file holds."""

    def __init__(self) -> None:
        self.doc: Dict[str, Any] = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "textures": [],
            "images": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [],
        }
        self.blob = bytearray()

    def _view(self, data: bytes) -> int:
        self.blob += b"\0" * (-len(self.blob) % 4)
        self.doc["bufferViews"].append(
            {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        )
        self.blob += data
        return len(self.doc["bufferViews"]) - 1

    def _accessor(self, values: np.ndarray) -> int:
        if values.dtype.kind == "f":
            data = np.ascontiguousarray(values, dtype=np.float32)
            kind = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[data.shape[1]]
            entry = {"componentType": 5126, "type": kind, "count": data.shape[0]}
            entry.update(min=data.min(0).tolist(), max=data.max(0).tolist())
        else:
            data = np.ascontiguousarray(values, dtype=np.uint32).reshape(-1)
            entry = {"componentType": 5125, "type": "SCALAR", "count": data.shape[0]}
        entry["bufferView"] = self._view(data.tobytes())
        self.doc["accessors"].append(entry)
        return len(self.doc["accessors"]) - 1

    def texture(self, image: bytes) -> int:
        self.doc["images"].append({"bufferView": self._view(image), "mimeType": "image/png"})
        self.doc["textures"].append({"source": len(self.doc["images"]) - 1})
        return len(self.doc["textures"]) - 1

    def material(self, **fields: Any) -> int:
        self.doc["materials"].append(fields)
        return len(self.doc["materials"]) - 1

    def mesh(self, *primitives: Dict[str, Any]) -> int:
        written = []
        for primitive in primitives:
            entry: Dict[str, Any] = {
                "attributes": {
                    name: self._accessor(values)
                    for name, values in primitive["attributes"].items()
                },
                "indices": self._accessor(primitive["indices"]),
            }
            if primitive.get("material") is not None:
                entry["material"] = primitive["material"]
            written.append(entry)
        self.doc["meshes"].append({"primitives": written})
        return len(self.doc["meshes"]) - 1

    def node(self, mesh: int, **transform: Any) -> None:
        self.doc["nodes"].append({"mesh": mesh, **transform})
        self.doc["scenes"][0]["nodes"].append(len(self.doc["nodes"]) - 1)

    def write(self, path: Path) -> Path:
        self.doc["buffers"] = [{"byteLength": len(self.blob)}]
        text = json.dumps(self.doc).encode("utf-8")
        text += b" " * (-len(text) % 4)
        binary = bytes(self.blob) + b"\0" * (-len(self.blob) % 4)
        chunks = struct.pack("<II", len(text), 0x4E4F534A) + text
        chunks += struct.pack("<II", len(binary), 0x004E4942) + binary
        path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(chunks)) + chunks)
        return path


def quad(offset: float = 0.0, normals: bool = True) -> Dict[str, Any]:
    """A unit square in the XY plane facing +Z, split into two triangles."""
    attributes = {
        "POSITION": np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        + [offset, 0, 0],
        "TEXCOORD_0": np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32),
    }
    if normals:
        attributes["NORMAL"] = np.tile(np.float32([0, 0, 1]), (4, 1))
    return {"attributes": attributes, "indices": np.array([0, 1, 2, 0, 2, 3])}


# ---------------------------------------------------------------------------
#  An FBX, written by hand
# ---------------------------------------------------------------------------


class I32(int):
    """An FBX 'I' property; plain ints are written as 'L'."""


class Raw(bytes):
    """An FBX 'R' property; plain bytes are written as 'S'."""


class Doubles(tuple):
    """An FBX 'd' array property."""


class Ints(tuple):
    """An FBX 'i' array property."""


def _fbx_property(value: Any) -> bytes:
    if isinstance(value, Raw):
        return b"R" + struct.pack("<I", len(value)) + value
    if isinstance(value, bytes):
        return b"S" + struct.pack("<I", len(value)) + value
    if isinstance(value, (Doubles, Ints)):
        code, form = (b"d", "d") if isinstance(value, Doubles) else (b"i", "i")
        data = struct.pack(f"<{len(value)}{form}", *value)
        return code + struct.pack("<III", len(value), 0, len(data)) + data
    if isinstance(value, I32):
        return b"I" + struct.pack("<i", value)
    if isinstance(value, int):
        return b"L" + struct.pack("<q", value)
    return b"D" + struct.pack("<d", value)


def _fbx_record(node: tuple, at: int) -> bytes:
    """One record and its children, laid out at an absolute offset (FBX 7.4)."""
    name, props, children = node
    blob = b"".join(_fbx_property(p) for p in props)
    body, cursor = b"", at + 13 + len(name) + len(blob)
    for child in children:
        emitted = _fbx_record(child, cursor)
        body += emitted
        cursor += len(emitted)
    if children:
        body += b"\0" * 13
        cursor += 13
    return (
        struct.pack("<III", cursor, len(props), len(blob))
        + bytes([len(name)])
        + name
        + blob
        + body
    )


def _p(name: bytes, kind: bytes, *values: Any) -> tuple:
    return (b"P", [name, kind, b"", b"", *values], [])


def _object(kind: bytes, identity: int, name: bytes, sub: bytes, *children: tuple) -> tuple:
    return (kind, [identity, name + b"\x00\x01" + kind, sub], list(children))


def _link(child: int, parent: int, prop: bytes = b"") -> tuple:
    return (b"C", [b"OP" if prop else b"OO", child, parent] + ([prop] if prop else []), [])


def write_fbx(
    path: Path,
    *,
    up: tuple = (2, 1),
    front: tuple = (1, -1),
    coord: tuple = (0, 1),
    unit: float = 1.0,
    albedo: Optional[bytes] = None,
) -> Path:
    """A 1 x 2 quad standing on the ground, 100 x 200 file units, one material.

    The defaults are what 3ds Max writes: Z up, -Y front, centimetres.  The
    quad lies in the file's XZ plane, so "standing" means along the file's up
    axis, whichever that is.  ``albedo`` is embedded as the diffuse map.
    """
    element = [(b"Version", [I32(101)], []), (b"Name", [b""], [])]
    geometry = _object(
        b"Geometry",
        10,
        b"quad",
        b"Mesh",
        (b"Vertices", [Doubles((0, 0, 0, 100, 0, 0, 100, 0, 200, 0, 0, 200))], []),
        (b"PolygonVertexIndex", [Ints((0, 1, 2, -4))], []),
        (b"GeometryVersion", [I32(124)], []),
        (
            b"LayerElementUV",
            [I32(0)],
            element
            + [
                (b"MappingInformationType", [b"ByPolygonVertex"], []),
                (b"ReferenceInformationType", [b"IndexToDirect"], []),
                (b"UV", [Doubles((0, 0, 1, 0, 1, 1, 0, 1))], []),
                (b"UVIndex", [Ints((0, 1, 2, 3))], []),
            ],
        ),
        (
            b"LayerElementMaterial",
            [I32(0)],
            element
            + [
                (b"MappingInformationType", [b"AllSame"], []),
                (b"ReferenceInformationType", [b"IndexToDirect"], []),
                (b"Materials", [Ints((0,))], []),
            ],
        ),
        (
            b"Layer",
            [I32(0)],
            [
                (b"Version", [I32(100)], []),
                (
                    b"LayerElement",
                    [],
                    [(b"Type", [b"LayerElementUV"], []), (b"TypedIndex", [I32(0)], [])],
                ),
                (
                    b"LayerElement",
                    [],
                    [(b"Type", [b"LayerElementMaterial"], []), (b"TypedIndex", [I32(0)], [])],
                ),
            ],
        ),
    )
    model = _object(b"Model", 20, b"quad", b"Mesh", (b"Version", [I32(232)], []))
    material = _object(
        b"Material",
        30,
        b"paint",
        b"",
        (b"Properties70", [], [_p(b"DiffuseColor", b"Color", 0.8, 0.8, 0.8)]),
    )
    objects = [geometry, model, material]
    links = [_link(10, 20), _link(20, 0), _link(30, 20)]
    if albedo is not None:
        objects += [
            _object(b"Texture", 40, b"paint_albedo", b""),
            _object(
                b"Video",
                50,
                b"paint_albedo",
                b"Clip",
                (b"Content", [Raw(albedo)], []),
                (b"RelativeFilename", [b"paint_albedo.png"], []),
            ),
        ]
        links += [_link(50, 40), _link(40, 30, b"DiffuseColor")]
    settings = [
        _p(b"UpAxis", b"int", I32(up[0])),
        _p(b"UpAxisSign", b"int", I32(up[1])),
        _p(b"FrontAxis", b"int", I32(front[0])),
        _p(b"FrontAxisSign", b"int", I32(front[1])),
        _p(b"CoordAxis", b"int", I32(coord[0])),
        _p(b"CoordAxisSign", b"int", I32(coord[1])),
        _p(b"UnitScaleFactor", b"double", unit),
    ]
    records = [
        (
            b"FBXHeaderExtension",
            [],
            [(b"FBXVersion", [I32(7400)], []), (b"Creator", [b"test"], [])],
        ),
        (
            b"GlobalSettings",
            [],
            [(b"Version", [I32(1000)], []), (b"Properties70", [], settings)],
        ),
        (b"Objects", [], objects),
        (b"Connections", [], links),
    ]
    out = bytearray(fbx_media.MAGIC + struct.pack("<I", 7400))
    for record in records:
        out += _fbx_record(record, len(out))
    out += b"\0" * 13
    path.write_bytes(bytes(out))
    return path


# ---------------------------------------------------------------------------
#  glTF: the scene graph, normals, shared materials
# ---------------------------------------------------------------------------


def test_a_gltf_arrives_with_its_material_and_its_uvs(tmp_path, torus) -> None:
    """glB is the format where everything is in one file, and it all works."""
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
    np.testing.assert_allclose(source.uv, uv, atol=1e-6)
    assert [slot.key for slot in source.slots] == ["basecolor"]
    assert source.texture(0, 0).mime == "image/jpeg"
    # Faces are handed over grouped by material, which is what lets the viewer
    # draw one run per material instead of looking one up per face.
    assert sum(count for _, _, count in source.groups) == source.faces.shape[0]


def test_primitives_that_share_a_material_are_one_material_and_one_run(tmp_path) -> None:
    """A map is encoded and fetched once however many parts wear it.

    ``Scene.dump()`` gave each primitive a copy of the material, so a model
    exported part by part cost one encode, one fetch and one draw call per
    part for the same picture.
    """
    gltf = Gltf()
    paint = gltf.material(
        name="paint",
        pbrMetallicRoughness={
            "baseColorTexture": {"index": gltf.texture(png(flat((200, 50, 20))))}
        },
    )
    gltf.node(gltf.mesh(*(dict(quad(2.0 * i), material=paint) for i in range(3))))
    source = assets.load_source(gltf.write(tmp_path / "parts.glb"))

    assert [spec.name for spec in source.materials] == ["paint"]
    assert source.groups == [(0, 0, 6)]
    assert source.vertices.shape == (12, 3)
    assert np.array_equal(source.face_materials(), np.zeros(6, np.int32))


def test_authored_normals_are_kept_and_follow_their_node(tmp_path) -> None:
    """Positions go through the node's matrix, normals through its inverse transpose.

    The normals written here are deliberately not the flat ones (they lean
    toward +X), so the check can tell kept from recomputed.  The node scales
    unevenly and mirrors in X: a mirror turns every triangle inside out, so
    the winding has to be reversed to keep facing the way the normals do.
    """
    lean = np.float32([0.6, 0.0, 0.8])
    primitive = quad()
    primitive["attributes"]["NORMAL"] = np.tile(lean, (4, 1))
    gltf = Gltf()
    gltf.node(gltf.mesh(primitive), scale=[-2.0, 1.0, 3.0], translation=[0.0, 5.0, 0.0])
    source = assets.load_source(gltf.write(tmp_path / "mirrored.glb"))

    corners = np.float32([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    np.testing.assert_allclose(source.vertices, corners * [-2, 1, 3] + [0, 5, 0])
    expected = lean / np.float32([-2.0, 1.0, 3.0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(source.normals, np.tile(expected, (4, 1)), atol=1e-6)
    assert np.allclose(np.linalg.norm(source.normals, axis=1), 1.0, atol=1e-6)
    # The winding agrees with the normals again: the surface faces +Z still.
    assert (face_normals(source)[:, 2] > 0.99).all()


def test_a_mesh_without_normals_gets_smooth_ones_with_no_seam(tmp_path) -> None:
    """Computed over the welded surface, so a UV seam does not show as a crease.

    Two quads meet at a right angle along x = 1, each with its own copy of
    the shared corners (as a UV seam leaves them).  Per split corner, each
    side would keep its own face's normal, +Z on one and +X on the other;
    welded, both copies of a corner get one normal, leaning between the two.
    """
    folded = quad(normals=False)
    wall = {
        "attributes": {
            "POSITION": np.float32([[1, 0, 0], [1, 0, -1], [1, 1, -1], [1, 1, 0]]),
            "TEXCOORD_0": np.float32([[0, 1], [1, 1], [1, 0], [0, 0]]),
        },
        "indices": np.array([0, 1, 2, 0, 2, 3]),
    }
    gltf = Gltf()
    gltf.node(gltf.mesh(folded, wall))
    source = assets.load_source(gltf.write(tmp_path / "fold.glb"))

    assert np.allclose(np.linalg.norm(source.normals, axis=1), 1.0, atol=1e-6)
    for y in (0.0, 1.0):
        copies = np.flatnonzero(np.all(source.vertices == [1.0, y, 0.0], axis=1))
        assert copies.size == 2, "the corner should arrive split, as the file wrote it"
        first, second = source.normals[copies]
        np.testing.assert_allclose(first, second, atol=1e-7)
        assert first[0] > 0.3 and first[2] > 0.3 and abs(first[1]) < 1e-6


def test_a_texture_transform_moves_the_uvs(tmp_path) -> None:
    """KHR_texture_transform is applied to the geometry: the viewer and the bake see it."""
    transform = {"offset": [0.25, 0.5], "scale": [2.0, 3.0]}
    gltf = Gltf()
    paint = gltf.material(
        name="tiled",
        pbrMetallicRoughness={
            "baseColorTexture": {
                "index": gltf.texture(png(flat((10, 20, 30)))),
                "extensions": {"KHR_texture_transform": transform},
            }
        },
    )
    gltf.node(gltf.mesh(dict(quad(), material=paint)))
    source = assets.load_source(gltf.write(tmp_path / "tiled.glb"))

    # In glTF's v-down space uv' = offset + scale * uv; the source holds v up.
    written = np.float32([[0, 1], [1, 1], [1, 0], [0, 0]])
    expected_down = written * [2.0, 3.0] + [0.25, 0.5]
    expected = np.stack([expected_down[:, 0], 1.0 - expected_down[:, 1]], 1)
    np.testing.assert_allclose(source.uv, expected, atol=1e-6)


# ---------------------------------------------------------------------------
#  Previews, slots and buttons
# ---------------------------------------------------------------------------


def test_an_opaque_material_drops_the_alpha_its_map_carries(tmp_path) -> None:
    """Pixal3D writes RGBA colour into an OPAQUE material.

    Kept, that alpha made its preview an 8.7 MB PNG of a channel nothing
    reads; a BLEND material's alpha is the point, and stays.
    """
    rgba = flat((90, 120, 150, 255), 16)
    rgba[:8, :, 3] = 40
    gltf = Gltf()
    texture = gltf.texture(png(rgba))
    opaque = gltf.material(
        name="opaque", pbrMetallicRoughness={"baseColorTexture": {"index": texture}}
    )
    blend = gltf.material(
        name="blend",
        alphaMode="BLEND",
        pbrMetallicRoughness={"baseColorTexture": {"index": texture}},
    )
    gltf.node(gltf.mesh(dict(quad(), material=opaque), dict(quad(2.0), material=blend)))
    source = assets.load_source(gltf.write(tmp_path / "alpha.glb"))

    assert source.texture(0, 0).mime == "image/jpeg"
    kept = source.texture(0, 1)
    assert kept.mime == "image/png"
    assert np.array_equal(decoded(kept)[..., 3], rgba[..., 3])
    assert [button.id for button in source.buttons] == ["basecolor", "opacity"]


def test_a_16_bit_map_is_not_saturated(tmp_path) -> None:
    """16-bit maps are scaled to 8, never clipped.

    ``convert('RGB')`` on a 16-bit height map made 99% of it white.
    """
    ramp = np.tile(np.linspace(0, 65535, 64).astype(np.uint16), (64, 1))
    Image.fromarray(ramp).save(tmp_path / "height16.png")
    (tmp_path / "rock.mtl").write_text("newmtl rock\ndisp height16.png\n")
    (tmp_path / "rock.obj").write_text(
        "mtllib rock.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\n"
        "usemtl rock\nf 1/1 2/2 3/3\n"
    )
    source = assets.load_source(tmp_path / "rock.obj")
    height = next(button for button in source.buttons if button.id == "height")
    preview = decoded(source.texture(height.slot, 0)).astype(np.int32)

    assert preview.ndim == 2 and preview.min() == 0 and preview.max() == 255
    expected = (ramp.astype(np.int64) * 255 + 32767) // 65535
    assert np.abs(preview - expected).max() <= 1


def test_a_map_is_shrunk_to_preview_size(tmp_path) -> None:
    """These are looked at on a model, not sampled for a render."""
    gltf = Gltf()
    big = np.zeros((2048, 4096, 3), np.uint8)
    big[..., 0] = 200
    paint = gltf.material(
        name="big", pbrMetallicRoughness={"baseColorTexture": {"index": gltf.texture(png(big))}}
    )
    gltf.node(gltf.mesh(dict(quad(), material=paint)))
    source = assets.load_source(gltf.write(tmp_path / "big.glb"))

    preview = decoded(source.texture(0, 0))
    assert preview.shape[:2] == (assets.MAX_TEXTURE_PX // 2, assets.MAX_TEXTURE_PX)


def test_data_maps_are_sent_lossless(tmp_path) -> None:
    """A JPEG block in a normal map is a bump that is not there."""
    rng = np.random.default_rng(3)
    tilt = rng.normal(0.0, 0.3, (32, 32, 2))
    z = np.sqrt(np.clip(1.0 - (tilt**2).sum(2, keepdims=True), 0.0, 1.0))
    normal = np.rint((np.concatenate([tilt, z], 2) * 0.5 + 0.5) * 255).astype(np.uint8)
    gltf = Gltf()
    bumpy = gltf.material(name="bumpy", normalTexture={"index": gltf.texture(png(normal))})
    gltf.node(gltf.mesh(dict(quad(), material=bumpy)))
    source = assets.load_source(gltf.write(tmp_path / "bumpy.glb"))

    slot = next(slot for slot in source.slots if slot.key == "normal")
    expected = np.array(source.materials[0].canonical(assets.MAX_TEXTURE_PX).normal)
    preview = source.texture(slot.index, 0)
    assert preview.mime == "image/png"
    assert np.array_equal(decoded(preview), expected)


def test_the_buttons_are_the_maps_a_model_really_has(tmp_path) -> None:
    """Pixal3D's metallicRoughness map has R = 0, which is not occlusion.

    One ORM slot carries roughness, metallic and AO as channels, but a button
    is only offered for a channel some map actually fills: an AO button over
    this one would render every surface black.
    """
    mr = np.zeros((16, 16, 3), np.uint8)
    mr[..., 1] = 230
    mr[..., 2] = np.tile(np.linspace(0, 255, 16).astype(np.uint8), (16, 1))
    gltf = Gltf()
    material = gltf.material(
        name="pixal",
        pbrMetallicRoughness={
            "baseColorTexture": {"index": gltf.texture(png(flat((90, 80, 60), 16)))},
            "metallicRoughnessTexture": {"index": gltf.texture(png(mr))},
        },
    )
    gltf.node(gltf.mesh(dict(quad(), material=material)))
    source = assets.load_source(gltf.write(tmp_path / "pixal.glb"))

    described = as_json(source)
    assert described["slots"] == [
        {"slot": 0, "role": "basecolor", "label": "Base colour"},
        {"slot": 1, "role": "orm", "label": "ORM"},
    ]
    assert [(b["id"], b["slot"], b["channel"]) for b in described["buttons"]] == [
        ("basecolor", 0, "rgb"),
        ("roughness", 1, "g"),
        ("metallic", 1, "b"),
    ]
    spec = described["materials"][0]
    assert spec["maps"] == {"basecolor": 0, "orm": 1}
    assert spec["orm_sources"] == ["metallic", "roughness"]
    assert spec["guesses"][1]["channels"] == {"roughness": "g", "metallic": "b"}
    # The ORM preview says the same: no AO source, so its red is 255.
    orm = decoded(source.texture(1, 0))
    assert (orm[..., 0] == 255).all() and np.array_equal(orm[..., 2], mr[..., 2])


def test_a_map_with_no_canonical_place_gets_a_button_of_its_own(tmp_path) -> None:
    """A plain specular map has no place in a metal/rough render; it is shown as it is."""
    rng = np.random.default_rng(4)
    speckle = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    (tmp_path / "albedo.png").write_bytes(png(flat((200, 60, 30), 32)))
    (tmp_path / "spec.png").write_bytes(png(speckle))
    (tmp_path / "m.mtl").write_text("newmtl paint\nmap_Kd albedo.png\nmap_Ks spec.png\n")
    (tmp_path / "m.obj").write_text(
        "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\n"
        "usemtl paint\nf 1/1 2/2 3/3\n"
    )
    source = assets.load_source(tmp_path / "m.obj")

    extra = source.buttons[-1]
    assert (extra.id, extra.label, extra.channel) == ("tex0", "Specular", "rgb")
    assert as_json(source)["materials"][0]["maps"] == {"basecolor": 0, "tex0": extra.slot}
    preview = source.texture(extra.slot, 0)
    assert preview.mime == "image/jpeg" and decoded(preview).shape == speckle.shape


def test_a_slot_a_material_does_not_fill_has_no_texture(tmp_path) -> None:
    gltf = Gltf()
    bare = gltf.material(name="bare")
    painted = gltf.material(
        name="painted",
        pbrMetallicRoughness={
            "baseColorTexture": {"index": gltf.texture(png(flat((1, 2, 3))))}
        },
    )
    gltf.node(gltf.mesh(dict(quad(), material=bare), dict(quad(2.0), material=painted)))
    source = assets.load_source(gltf.write(tmp_path / "two.glb"))

    assert [spec.name for spec in source.materials] == ["bare", "painted"]
    assert source.texture(0, 0) is None
    assert source.texture(0, 1) is not None
    assert source.texture(5, 1) is None and source.texture(0, 9) is None


def test_a_correction_reaches_the_buttons(tmp_path) -> None:
    """What the user says a map is wins over the classifier, all the way to the viewer."""
    gltf = Gltf()
    paint = gltf.material(
        name="paint",
        pbrMetallicRoughness={
            "baseColorTexture": {"index": gltf.texture(png(flat((128, 128, 128))))}
        },
    )
    gltf.node(gltf.mesh(dict(quad(), material=paint)))
    path = gltf.write(tmp_path / "gray.glb")
    key = assets.load_source(path).materials[0].refs[0].key

    corrected = assets.load_source(path, {key: {"role": "ao"}})
    assert [button.id for button in corrected.buttons] == ["ao"]
    assert as_json(corrected)["materials"][0]["guesses"][0]["evidence"][0] == "set by hand: AO"


# ---------------------------------------------------------------------------
#  FBX and Collada, through Assimp
# ---------------------------------------------------------------------------


def test_a_z_up_fbx_in_centimetres_ends_y_up_in_metres(tmp_path) -> None:
    """What 3ds Max writes, turned into what glTF and the viewer expect.

    The quad stands 200 file units tall along the file's up axis; it has to
    arrive standing 2 m tall along +Y, and facing +Z (the file's front is -Y).
    """
    for up, front, coord, name in (
        ((2, 1), (1, -1), (0, 1), "z_up.fbx"),
        ((1, 1), (2, 1), (0, 1), "y_up.fbx"),
    ):
        path = write_fbx(tmp_path / name, up=up, front=front, coord=coord)
        source = assets.load_source(path)
        extent = source.vertices.max(0) - source.vertices.min(0)
        if name == "z_up.fbx":
            np.testing.assert_allclose(extent, [1.0, 2.0, 0.0], atol=1e-6)
            np.testing.assert_allclose(source.normals, np.tile([0, 0, 1], (4, 1)), atol=1e-6)
        else:
            # Already Y up, so the file's Z stays Z: the quad lies flat, 2 m deep.
            np.testing.assert_allclose(extent, [1.0, 0.0, 2.0], atol=1e-6)


def test_an_fbx_unit_is_honoured(tmp_path) -> None:
    """UnitScaleFactor is centimetres per file unit: 100 means the file is in metres."""
    source = assets.load_source(write_fbx(tmp_path / "metres.fbx", unit=100.0))
    np.testing.assert_allclose(
        source.vertices.max(0) - source.vertices.min(0), [100, 200, 0], atol=1e-4
    )


def test_an_fbx_carries_its_embedded_map(tmp_path) -> None:
    """The only kind of FBX map an upload of the file alone can carry."""
    path = write_fbx(tmp_path / "painted.fbx", albedo=png(flat((200, 40, 40))))
    source = assets.load_source(path)

    assert [spec.name for spec in source.materials] == ["paint"]
    assert [button.id for button in source.buttons] == ["basecolor"]
    np.testing.assert_allclose(source.uv, [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-6)
    colour = decoded(source.texture(0, 0)).reshape(-1, 3).mean(0)
    assert np.abs(colour - [200, 40, 40]).max() < 3


def test_a_collada_file_is_read_through_assimp(tmp_path) -> None:
    """trimesh would need pycollada; Assimp reads it, up axis and unit included."""
    path = tmp_path / "quad.dae"
    path.write_text("""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="centimeter" meter="0.01"/><up_axis>Z_UP</up_axis></asset>
  <library_geometries><geometry id="quad"><mesh>
    <source id="p"><float_array id="pa" count="12">0 0 0 100 0 0 100 0 200 0 0 200</float_array>
      <technique_common><accessor source="#pa" count="4" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/>
        <param name="Z" type="float"/>
      </accessor></technique_common></source>
    <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
    <triangles count="2"><input semantic="VERTEX" source="#v" offset="0"/>
      <p>0 1 2 0 2 3</p></triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="s">
    <node id="n"><instance_geometry url="#quad"/></node>
  </visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#s"/></scene>
</COLLADA>
""")
    source = assets.load_source(path)
    assert source.faces.shape == (2, 3)
    np.testing.assert_allclose(
        source.vertices.max(0) - source.vertices.min(0), [1.0, 2.0, 0.0], atol=1e-6
    )
    assert np.allclose(np.linalg.norm(source.normals, axis=1), 1.0, atol=1e-6)


def test_geometry_alone_skips_the_materials(tmp_path) -> None:
    """A remesh needs no maps, and classifying a model's maps means decoding every one."""
    path = write_fbx(tmp_path / "painted.fbx", albedo=png(flat((200, 40, 40))))
    full = assets.load_source(path)
    bare = assets.load_source(path, materials=False)

    assert np.array_equal(bare.vertices, full.vertices)
    assert [spec.name for spec in bare.materials] == ["default"]
    assert bare.buttons == [] and bare.slots == []


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
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],  # first triangle
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],  # second, sharing two corners
        ],
        dtype=np.float32,
    )
    return assets.SourceMesh(
        name="seam.obj",
        vertices=vertices,
        normals=np.tile(np.float32([0, 0, 1]), (6, 1)),
        uv=np.zeros((6, 2), dtype=np.float32),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32),
        groups=[(0, 0, 2)],
        materials=[MaterialSpec(name="m")],
    )


def test_the_solver_gets_the_seams_welded_shut() -> None:
    """A hierarchy built on split corners has a crack along every seam.

    The viewport needs them apart -- each side carries its own UV -- so the
    split mesh is kept and the welded one is derived, rather than the reverse.
    """
    vertices, faces = assets.solver_mesh(_split_quad())

    assert vertices.shape == (4, 3), "the shared edge was not welded"
    assert faces.shape == (2, 3)
    # Both triangles survive, and they now share an edge rather than a gap.
    corners = [set(face) for face in faces.tolist()]
    assert len(corners[0] & corners[1]) == 2


def test_a_triangle_that_welds_into_a_line_is_dropped() -> None:
    """The hierarchy builder divides by face area, and this one has none."""
    source = _split_quad()
    source.vertices[3:6] = source.vertices[0]
    _, faces = assets.solver_mesh(source)
    assert faces.shape == (1, 3)


def test_a_model_with_no_triangles_is_refused() -> None:
    empty = assets.SourceMesh(
        name="nothing.obj",
        vertices=np.zeros((0, 3), dtype=np.float32),
        normals=np.zeros((0, 3), dtype=np.float32),
        uv=np.zeros((0, 2), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.uint32),
        groups=[],
        materials=[],
    )
    with pytest.raises(assets.AssetError):
        assets.solver_mesh(empty)


def test_an_unsupported_suffix_says_what_is_supported(tmp_path) -> None:
    path = tmp_path / "model.blend"
    path.write_bytes(b"not a mesh")
    with pytest.raises(assets.AssetError, match="unsupported mesh format"):
        assets.load_source(path)


def test_a_missing_file_is_an_asset_error(tmp_path) -> None:
    with pytest.raises(assets.AssetError, match="no such mesh file"):
        assets.load_source(tmp_path / "gone.glb")


# ---------------------------------------------------------------------------
#  Real models (the author's machine only)
# ---------------------------------------------------------------------------

PIXAL3D = Path(r"C:\_myDrive\repos\Pixal3D-stableprojectorz\output.glb")

#: Z up by its GlobalSettings, with the axis conversion in Assimp's root node.
BAMBOO = Path(
    r"C:\_myDrive\repos\warewolves-game\Werewolves\Assets\NatureParadaise\SoStylized"
    r"\Environment\Trees\Bamboo\Meshes\SM_BambooSaplingClump2.fbx"
)

#: A rigged character on which Assimp's PreTransformVertices corrupts the heap.
ARACHNYA = Path(
    r"C:\_myDrive\bsv\Assets\_Characters (Specific creatures)\Available Models (with anims)"
    r"\_DLNK (Arachnya Boss)\Arachnya Boss\Mesh\Arachnya.FBX"
)


@pytest.mark.realasset
@pytest.mark.skipif(not PIXAL3D.is_file(), reason="the Pixal3D fixture is not on this machine")
def test_a_pixal3d_model_loads_fast_with_small_previews() -> None:
    """950k triangles and two 4096 px WebP maps, the model this viewer mostly sees."""
    started = time.perf_counter()
    source = assets.load_source(PIXAL3D)
    elapsed = time.perf_counter() - started

    assert elapsed <= 2.5, f"load_source took {elapsed:.2f} s"
    assert source.faces.shape[0] == 950_209
    assert np.allclose(np.linalg.norm(source.normals, axis=1), 1.0, atol=1e-5)
    assert [button.id for button in source.buttons] == ["basecolor", "roughness", "metallic"]
    colour = source.texture(0, 0)
    assert colour.mime == "image/jpeg" and len(colour.data) <= 1.5 * 1024 * 1024


@pytest.mark.realasset
@pytest.mark.skipif(not BAMBOO.is_file(), reason="the Bamboo fixture is not on this machine")
def test_a_z_up_blender_fbx_stands_as_blender_shows_it() -> None:
    """Bounding box measured in Blender 5.1 (FBX import, then glTF's axes)."""
    source = assets.load_source(BAMBOO, materials=False)
    np.testing.assert_allclose(source.vertices.min(0), [-2.0315, -0.5698, -2.3577], atol=2e-4)
    np.testing.assert_allclose(source.vertices.max(0), [3.1020, 9.4899, 1.6281], atol=2e-4)


@pytest.mark.realasset
@pytest.mark.skipif(
    not ARACHNYA.is_file(), reason="the Arachnya fixture is not on this machine"
)
def test_a_rigged_fbx_loads_without_bringing_the_process_down() -> None:
    """Bounding box measured in Blender 5.1 (FBX import, then glTF's axes)."""
    source = assets.load_source(ARACHNYA, materials=False)
    assert source.faces.shape[0] == 7730
    np.testing.assert_allclose(
        source.vertices.min(0), [-203.6011, -4.2370, -131.5898], atol=1e-3
    )
    np.testing.assert_allclose(
        source.vertices.max(0), [203.6011, 160.1325, 115.1285], atol=1e-3
    )
