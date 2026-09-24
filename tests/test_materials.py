"""Reading a model's materials into one layout: glTF, MTL, FBX and the rules between.

Most of this runs on files written here -- a GLB with shared materials,
factors, texture transforms and a WebP map; an MTL naming every key and option
the parser knows; binary FBX files assembled record by record -- because their
right answers are known by construction.  The images come from
:mod:`texroles_synth`, whose maps the classifier is known to read correctly.

The ``realasset`` tests read model files that live on the author's machine (a
Pixal3D and a Tripo GLB, FBX files from 3ds Max and Blender, a Polycam and a
Blender MTL) and skip where those are absent.
"""

from __future__ import annotations

import base64
import io
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
from PIL import Image

import texroles_synth as synth
from instant_meshes_brush import fbx_media, materials
from instant_meshes_brush.materials import MaterialSpec, TexRef, UVTransform

SIZE = 64


def encode(pixels: np.ndarray, fmt: str = "PNG", **options: Any) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, fmt, **options)
    return buffer.getvalue()


def ref_named(spec: MaterialSpec, name: str) -> TexRef:
    return next(ref for ref in spec.refs if ref.name == name)


def normal_map(seed: int = 1, convention: str = "opengl") -> np.ndarray:
    return synth.normal_map(synth.height_field(seed), SIZE, 0.6, convention)


# ---------------------------------------------------------------------------
#  Decoding
# ---------------------------------------------------------------------------


def test_images_keep_their_channel_count_and_palettes_expand() -> None:
    rgb = synth.albedo(1, SIZE)
    assert materials.decode_image(encode(rgb)).shape == (SIZE, SIZE, 3)
    assert materials.decode_image(encode(rgb[..., 0])).shape == (SIZE, SIZE)
    rgba = synth.albedo(1, SIZE, alpha=True)
    assert materials.decode_image(encode(rgba)).shape == (SIZE, SIZE, 4)
    gray_alpha = np.dstack([rgb[..., 0], rgba[..., 3]])
    assert materials.decode_image(encode(gray_alpha)).shape == (SIZE, SIZE, 2)

    palette = Image.fromarray(rgb).convert("P")
    buffer = io.BytesIO()
    palette.save(buffer, "PNG", transparency=0)
    assert materials.decode_image(buffer.getvalue()).shape == (SIZE, SIZE, 4)
    buffer = io.BytesIO()
    Image.fromarray(rgb).convert("CMYK").save(buffer, "JPEG")
    assert materials.decode_image(buffer.getvalue()).shape == (SIZE, SIZE, 3)


def test_sixteen_bit_and_float_images_are_scaled_never_clipped() -> None:
    height = synth.grayscale_height(3, SIZE, bits=16)
    decoded = materials.decode_image(encode(height))
    assert decoded.dtype == np.uint16 and np.array_equal(decoded, height)

    metres = np.linspace(-2.0, 5.0, SIZE * SIZE, dtype=np.float32).reshape(SIZE, SIZE)
    decoded = materials.decode_image(encode(metres, "TIFF"))
    assert decoded.dtype == np.uint16
    assert (decoded.min(), decoded.max()) == (0, 65535)
    assert np.all(np.diff(decoded.reshape(-1).astype(np.int64)) >= 0)  # order kept


def test_formats_that_cannot_be_decoded_say_why() -> None:
    with pytest.raises(materials.ImageError, match="OpenEXR"):
        materials.decode_image(b"\x76\x2f\x31\x01" + bytes(64))
    with pytest.raises(materials.ImageError, match="KTX2"):
        materials.decode_image(b"\xabKTX 20\xbb" + bytes(64))
    with pytest.raises(materials.ImageError, match="format"):
        materials.decode_image(b"certainly not an image")


# ---------------------------------------------------------------------------
#  glTF
# ---------------------------------------------------------------------------

REPEAT, CLAMP, MIRROR = materials.WRAP_REPEAT, materials.WRAP_CLAMP, materials.WRAP_MIRROR


def _triangle(binary: bytearray, views: List[dict], accessors: List[dict]) -> Dict[str, int]:
    """One triangle's POSITION, TEXCOORD_0 and indices appended to a GLB's buffer."""
    arrays = (
        (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32), "VEC3", 5126, 34962),
        (np.array([[0, 0], [1, 0], [0, 1]], np.float32), "VEC2", 5126, 34962),
        (np.array([0, 1, 2], np.uint32), "SCALAR", 5125, 34963),
    )
    made = []
    for array, kind, component, target in arrays:
        views.append(
            {
                "buffer": 0,
                "byteOffset": len(binary),
                "byteLength": array.nbytes,
                "target": target,
            }
        )
        binary += array.tobytes()
        accessor = {
            "bufferView": len(views) - 1,
            "componentType": component,
            "count": len(array),
            "type": kind,
        }
        if kind == "VEC3":
            accessor.update(min=[0.0, 0.0, 0.0], max=[1.0, 1.0, 0.0])
        accessors.append(accessor)
        made.append(len(accessors) - 1)
    return {"POSITION": made[0], "TEXCOORD_0": made[1], "indices": made[2]}


def write_glb(
    path: Path,
    document: Dict[str, Any],
    images: Sequence[bytes],
    meshes: Sequence[Sequence[Optional[int]]],
) -> None:
    """A GLB with ``images`` in its binary chunk and one triangle per primitive.

    ``meshes`` lists, per mesh, the material of each primitive (None: none).
    """
    binary = bytearray()
    views: List[dict] = []
    accessors: List[dict] = []
    for image in images:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(image)})
        binary += image + b"\0" * (-len(image) % 4)
    document = dict(document)
    document["images"] = [
        dict(entry, bufferView=i) for i, entry in enumerate(document["images"])
    ]
    gltf_meshes = []
    for materials_of in meshes:
        primitives = []
        for material in materials_of:
            attributes = _triangle(binary, views, accessors)
            primitive = {
                "attributes": {k: v for k, v in attributes.items() if k != "indices"},
                "indices": attributes["indices"],
            }
            if material is not None:
                primitive["material"] = material
            primitives.append(primitive)
        gltf_meshes.append({"primitives": primitives})
    document.update(
        asset={"version": "2.0", "generator": "test writer"},
        buffers=[{"byteLength": len(binary)}],
        bufferViews=views,
        accessors=accessors,
        meshes=gltf_meshes,
        nodes=[{"mesh": i} for i in range(len(gltf_meshes))],
        scenes=[{"nodes": list(range(len(gltf_meshes)))}],
        scene=0,
    )
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    chunks = (
        struct.pack("<II", len(text), 0x4E4F534A)
        + text
        + struct.pack("<II", len(binary), 0x004E4942)
        + bytes(binary)
    )
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(chunks)) + chunks)


ALBEDO_CUTOUT = synth.albedo(1, SIZE, alpha=True)
ORM = synth.packed_orm(2, SIZE)
NORMAL = normal_map(3)
EMISSIVE = synth.emissive(4, SIZE)
PIXAL_MR = synth.pixal_metal_rough(5, SIZE)


@pytest.fixture(scope="module")
def synthetic_glb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("glb") / "model.glb"
    transform = {"offset": [0.5, 0.25], "rotation": 0.3, "scale": [2.0, 3.0]}
    document = {
        "extensionsUsed": [
            "EXT_texture_webp",
            "KHR_texture_transform",
            "KHR_materials_emissive_strength",
        ],
        "images": [
            {"mimeType": "image/png", "name": "albedo"},
            {"mimeType": "image/png"},
            {"mimeType": "image/png", "name": "rock_n"},
            {"mimeType": "image/webp", "name": "glow"},
            {"mimeType": "image/png", "name": "metal_rough"},
        ],
        "samplers": [{"wrapS": CLAMP, "wrapT": MIRROR}],
        "textures": [
            {"source": 0, "sampler": 0},
            {"source": 1},
            {"source": 2},
            {"extensions": {"EXT_texture_webp": {"source": 3}}},
            {"source": 4},
        ],
        "materials": [
            {
                "name": "shared",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {
                        "index": 0,
                        "extensions": {"KHR_texture_transform": transform},
                    },
                    "baseColorFactor": [0.5, 0.25, 1.0, 0.8],
                    "metallicFactor": 0.3,
                    "roughnessFactor": 0.7,
                    "metallicRoughnessTexture": {
                        "index": 1,
                        "extensions": {"KHR_texture_transform": transform},
                    },
                },
                "occlusionTexture": {
                    "index": 1,
                    "strength": 0.6,
                    "extensions": {"KHR_texture_transform": transform},
                },
                "normalTexture": {
                    "index": 2,
                    "scale": 0.9,
                    "extensions": {"KHR_texture_transform": transform},
                },
                "emissiveTexture": {
                    "index": 3,
                    "extensions": {"KHR_texture_transform": transform},
                },
                "emissiveFactor": [1.0, 0.5, 0.25],
                "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 4.0}},
                "alphaMode": "MASK",
                "alphaCutoff": 0.3,
                "doubleSided": True,
            },
            {
                "name": "pixal",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 4},
                },
            },
        ],
    }
    images = [
        encode(ALBEDO_CUTOUT),
        encode(ORM),
        encode(NORMAL),
        encode(EMISSIVE, "WEBP", lossless=True),
        encode(PIXAL_MR),
    ]
    write_glb(path, document, images, [[0, 0, 0], [1], [None]])
    return path


def test_gltf_primitives_sharing_a_material_share_one_spec(synthetic_glb: Path) -> None:
    read = materials.read_materials(synthetic_glb)
    assert [m.name for m in read.materials] == ["shared", "pixal"]
    assert [(p.mesh, p.primitive, p.material, p.mode) for p in read.bindings.primitives] == [
        (0, 0, 0, 4),
        (0, 1, 0, 4),
        (0, 2, 0, 4),
        (1, 0, 1, 4),
        (2, 0, None, 4),
    ]
    assert read.bindings.names == {"shared": 0, "pixal": 1}
    assert read.bindings.exporter == "test writer"
    assert read.warnings == []


def test_trimesh_adds_one_geometry_per_primitive_in_file_order(synthetic_glb: Path) -> None:
    """The contract assets.py relies on to give each trimesh geometry its MaterialSpec."""
    import trimesh

    scene = trimesh.load(synthetic_glb, process=False)
    bindings = materials.read_materials(synthetic_glb).bindings
    geometry = list(scene.geometry.values())
    assert len(geometry) == len(bindings.primitives)
    first_object: Dict[int, int] = {}
    for part, binding in zip(geometry, bindings.primitives):
        material = (
            getattr(part.visual, "material", None) if binding.material is not None else None
        )
        if binding.material is not None:
            assert first_object.setdefault(binding.material, id(material)) == id(material)
    assert len(set(first_object.values())) == 2


def test_gltf_factors_are_read_as_written(synthetic_glb: Path) -> None:
    shared, pixal = materials.read_materials(synthetic_glb).materials
    assert shared.base_color_factor == (0.5, 0.25, 1.0, 0.8)
    assert (shared.metallic, shared.roughness) == (0.3, 0.7)
    assert shared.emissive == (4.0, 2.0, 1.0)  # x KHR_materials_emissive_strength
    assert (shared.normal_scale, shared.occlusion_strength) == (0.9, 0.6)
    assert (shared.alpha_mode, shared.alpha_cutoff, shared.double_sided) == ("MASK", 0.3, True)
    assert (pixal.metallic, pixal.roughness, pixal.base_color_factor) == (1.0, 1.0, (1.0,) * 4)
    assert (pixal.alpha_mode, pixal.double_sided) == ("OPAQUE", False)


def test_one_image_in_occlusion_and_metal_rough_is_one_orm_map(synthetic_glb: Path) -> None:
    shared = materials.read_materials(synthetic_glb).materials[0]
    assert len(shared.refs) == 4
    orm = ref_named(shared, "image 1")
    assert sorted(slot.key for slot in orm.slots) == [
        "metallicRoughnessTexture",
        "occlusionTexture",
    ]
    assert (orm.role, orm.guess.channels) == (
        "orm",
        {"ao": "r", "roughness": "g", "metallic": "b"},
    )
    maps = shared.canonical()
    assert maps.orm_sources == {"ao", "roughness", "metallic"}
    assert np.array_equal(maps.orm, ORM)


def test_gltf_canonical_maps_are_the_images_themselves(synthetic_glb: Path) -> None:
    shared = materials.read_materials(synthetic_glb).materials[0]
    maps = shared.canonical()
    assert maps.basecolor.shape == (SIZE, SIZE, 4)
    assert np.array_equal(maps.basecolor, ALBEDO_CUTOUT)  # MASK: alpha kept
    assert np.array_equal(maps.normal, NORMAL) and maps.normal_space == "tangent"
    assert np.array_equal(maps.emissive, EMISSIVE)  # decoded from lossless WebP
    assert ref_named(shared, "glow").role == "emissive"
    assert maps.height is None and maps.others == ()


def test_the_r_of_a_metal_rough_map_is_never_ao_and_opaque_drops_alpha(
    synthetic_glb: Path,
) -> None:
    pixal = materials.read_materials(synthetic_glb).materials[1]
    maps = pixal.canonical()
    assert maps.orm_sources == {"roughness", "metallic"}
    assert np.all(maps.orm[..., 0] == 255)
    assert np.array_equal(maps.orm[..., 1:], PIXAL_MR[..., 1:])
    assert np.all(maps.basecolor[..., 3] == 255)
    assert np.array_equal(maps.basecolor[..., :3], ALBEDO_CUTOUT[..., :3])


def test_samplers_and_texture_transforms_are_kept(synthetic_glb: Path) -> None:
    shared = materials.read_materials(synthetic_glb).materials[0]
    albedo = ref_named(shared, "albedo")
    assert (albedo.wrap_s, albedo.wrap_t) == (CLAMP, MIRROR)
    assert (ref_named(shared, "rock_n").wrap_s, ref_named(shared, "rock_n").wrap_t) == (
        REPEAT,
        REPEAT,
    )
    expected = UVTransform(offset=(0.5, 0.25), rotation=0.3, scale=(2.0, 3.0))
    assert albedo.transform == expected and shared.uv_transform == expected


def test_uv_transform_follows_khr_texture_transform() -> None:
    transform = UVTransform(offset=(0.5, 0.25), rotation=0.3, scale=(2.0, 3.0))
    uv = np.random.default_rng(0).random((50, 2)).astype(np.float32)
    c, s = math.cos(0.3), math.sin(0.3)
    su, sv = uv[:, 0] * 2.0, uv[:, 1] * 3.0
    expected = np.stack([c * su + s * sv + 0.5, -s * su + c * sv + 0.25], 1)
    assert np.allclose(transform.apply(uv, v_up=False), expected, atol=1e-6)
    # v up: flipped into glTF's space, transformed, flipped back.
    flipped = np.stack([uv[:, 0], 1 - uv[:, 1]], 1)
    back = transform.apply(flipped, v_up=False)
    assert np.allclose(
        transform.apply(uv), np.stack([back[:, 0], 1 - back[:, 1]], 1), atol=1e-6
    )
    assert transform.apply(uv).dtype == np.float32
    quarter = UVTransform(rotation=math.pi / 2)
    assert np.allclose(quarter.apply(np.array([[1.0, 0.0]]), v_up=False), [[0.0, -1.0]])
    assert UVTransform().identity and not transform.identity


def test_a_gltf_with_external_files_and_data_uris(tmp_path: Path) -> None:
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / "rock colour.png").write_bytes(encode(synth.albedo(2, SIZE)))
    (tmp_path.parent / "secret.png").write_bytes(encode(synth.albedo(3, SIZE)))
    data_uri = "data:image/png;base64," + base64.b64encode(encode(NORMAL)).decode()
    document = {
        "asset": {"version": "2.0"},
        "images": [
            {"uri": "maps/rock%20colour.png"},
            {"uri": data_uri},
            {"uri": "../secret.png"},
            {"uri": "data:image/png;base64,not base64 at all", "name": "broken"},
        ],
        "textures": [{"source": 0}, {"source": 1}, {"source": 2}, {"source": 3}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 3},
                },
                "normalTexture": {"index": 1, "texCoord": 1},
                "emissiveTexture": {"index": 2},
            }
        ],
        "meshes": [],
    }
    path = tmp_path / "model.gltf"
    path.write_text(json.dumps(document))
    read = materials.read_materials(path)
    spec = read.materials[0]
    assert spec.name == "material 0"
    assert [(ref.name, ref.role) for ref in spec.refs] == [
        ("rock colour.png", "basecolor"),
        ("image 1", "normal"),
    ]
    assert ref_named(spec, "image 1").texcoord == 1
    assert read.warnings == [
        "material 0: broken is a data URI that cannot be decoded",
        "material 0: secret.png was not found next to the model",
        "material 0: image 1 uses UV set 1; it is shown on the first",
    ]


def test_a_specular_glossiness_gltf_is_converted_like_trimesh_does() -> None:
    from trimesh.visual.gloss import specular_to_pbr

    diffuse = synth.albedo(5, SIZE, alpha=True)
    spec_gloss = np.dstack([synth.albedo(6, SIZE) // 4, synth.gray_map(7, SIZE, 0.6)])
    factors = {
        "diffuseFactor": [0.9, 0.8, 0.7, 1.0],
        "specularFactor": [0.5, 0.6, 0.7],
        "glossinessFactor": 0.8,
    }
    expected = specular_to_pbr(
        specularFactor=factors["specularFactor"],
        glossinessFactor=0.8,
        specularGlossinessTexture=Image.fromarray(spec_gloss),
        diffuseTexture=Image.fromarray(diffuse),
        diffuseFactor=factors["diffuseFactor"],
    )
    only_factors = specular_to_pbr(
        specularFactor=factors["specularFactor"],
        glossinessFactor=0.8,
        diffuseFactor=factors["diffuseFactor"],
    )
    with_maps = dict(
        factors, diffuseTexture={"index": 0}, specularGlossinessTexture={"index": 1}
    )

    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "gloss.glb"
        document = {
            "images": [{"name": "diffuse"}, {"name": "specular_gloss"}],
            "textures": [{"source": 0}, {"source": 1}],
            "materials": [
                {
                    "alphaMode": "BLEND",
                    "extensions": {"KHR_materials_pbrSpecularGlossiness": with_maps},
                },
                {"extensions": {"KHR_materials_pbrSpecularGlossiness": factors}},
            ],
        }
        write_glb(path, document, [encode(diffuse), encode(spec_gloss)], [[0, 1]])
        textured, plain = materials.read_materials(path).materials

        assert ref_named(textured, "specular_gloss").role == "spec_gloss"
        assert textured.spec_gloss is not None
        assert (textured.base_color_factor, textured.metallic, textured.roughness) == (
            (1.0,) * 4,
            1.0,
            1.0,
        )
        maps = textured.canonical()
        assert maps.orm_sources == {"roughness", "metallic"}
        assert np.array_equal(maps.basecolor, np.asarray(expected["baseColorTexture"]))
        metal_rough = np.asarray(expected["metallicRoughnessTexture"])
        assert np.array_equal(maps.orm[..., 1:], metal_rough[..., 1:])
        assert np.all(maps.orm[..., 0] == 255)

        assert plain.spec_gloss is None
        assert plain.base_color_factor == pytest.approx(tuple(only_factors["baseColorFactor"]))
        assert plain.metallic == pytest.approx(only_factors["metallicFactor"])
        assert plain.roughness == pytest.approx(only_factors["roughnessFactor"])


# ---------------------------------------------------------------------------
#  MTL
# ---------------------------------------------------------------------------


def test_map_statement_options_are_parsed_and_the_rest_is_the_file() -> None:
    named, options = materials.parse_map_line(
        "-blendu on -blendv off -bm 0.5 -boost 1.5 -cc on -clamp on -imfchan m "
        "-mm 0.1 0.9 -texres 512 -type sphere -o 1 2 -s 3 -t 0.1 0.2 0.3 my map 1.png"
    )
    assert named == "my map 1.png"
    assert options == {
        "-blendu": ("on",),
        "-blendv": ("off",),
        "-bm": ("0.5",),
        "-boost": ("1.5",),
        "-cc": ("on",),
        "-clamp": ("on",),
        "-imfchan": ("m",),
        "-mm": ("0.1", "0.9"),
        "-texres": ("512",),
        "-type": ("sphere",),
        "-o": ("1", "2"),
        "-s": ("3",),
        "-t": ("0.1", "0.2", "0.3"),
    }
    assert materials.parse_map_line("rock.png") == ("rock.png", {})


def test_every_mtl_map_keyword_becomes_a_slot(tmp_path: Path) -> None:
    keys = [
        "map_Kd",
        "map_Ka",
        "map_Ks",
        "map_Ns",
        "map_d",
        "map_Tr",
        "map_Bump",
        "bump",
        "norm",
        "map_Kn",
        "normal",
        "map_normal",
        "map_norm",
        "disp",
        "decal",
        "refl",
        "map_refl",
        "map_Pr",
        "map_Pm",
        "map_Ps",
        "map_Ke",
        "map_ao",
        "map_RMA",
        "map_ORM",
    ]
    assert {key.lower() for key in keys} == set(materials.texroles.MTL_SLOTS)
    lines = ["newmtl all"]
    for index, key in enumerate(keys):
        (tmp_path / f"t{index}.png").write_bytes(encode(synth.gray_map(index, SIZE, 0.5)))
        lines.append(f"{key} t{index}.png")
    (tmp_path / "all.mtl").write_text("\n".join(lines))
    read = materials.read_materials(tmp_path / "all.mtl")
    assert read.warnings == []
    assert [slot.key for ref in read.materials[0].refs for slot in ref.slots] == keys


def _write_stone(folder: Path) -> Dict[str, np.ndarray]:
    """An OBJ, its MTL (in a file with a space in its name) and maps in three places."""
    (folder / "textures").mkdir()
    images = {
        "stone_colour.png": synth.albedo(8, SIZE),
        "stone_normal.png": normal_map(9),
        "stone_rough.png": synth.gray_map(10, SIZE, 0.6),
        "stone_metal.png": np.dstack(
            [
                synth.gray_map(11, SIZE, 0.2),
                synth.bimodal_mask(12, SIZE),
                synth.gray_map(13, SIZE, 0.5),
            ]
        ),
        "stone_height.png": synth.grayscale_height(14, SIZE, bits=16),
    }
    for name, pixels in images.items():
        where = folder / "textures" if name == "stone_normal.png" else folder
        (where / name).write_bytes(encode(pixels))
    (folder / "my materials.mtl").write_text(
        "# Blender 4.1.0 MTL File: 'scene.blend'\n"
        "newmtl stone\n"
        "Kd 0.2 0.4 0.6\nKs 0.5 0.5 0.5\nNs 250\nPm 0.9\n"
        "map_Kd -s 2 2 1 -o 0.25 0.5 stone_colour.png\n"
        "map_Bump -bm 0.5 stone_normal.png\n"
        "map_Ns C:\\Users\\someone\\maps\\stone_rough.png\n"
        "map_Pm -imfchan g -clamp on stone_metal.png\n"
        "disp stone_height.png\n"
        "newmtl glass\n"
        "Kd 0.5\nKe 0.1 0.2 0.3\nNs 90\nd 0.25\n"
        "newmtl rough\n"
        "Pr 0.35\nTr 0.0\n"
    )
    (folder / "model.obj").write_text(
        "# Blender 4.1.0\nmtllib my materials.mtl\no cube\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "usemtl stone\nf 1 2 3\n"
    )
    return images


def test_an_obj_finds_its_mtl_and_every_map(tmp_path: Path) -> None:
    images = _write_stone(tmp_path)
    read = materials.read_materials(tmp_path / "model.obj")
    # Only map_Kd has -s/-o: the geometry gets its transform, and a warning says so.
    assert read.warnings == [
        "stone: its maps use different texture transforms; all are shown with the base "
        "colour map's"
    ]
    assert read.bindings.names == {"stone": 0, "glass": 1, "rough": 2}
    assert read.bindings.exporter.startswith("Blender 4.1.0 MTL File")
    stone = read.materials[0]
    roles = {ref.name: ref.role for ref in stone.refs}
    assert roles == {
        "stone_colour.png": "basecolor",
        "stone_normal.png": "normal",
        "stone_rough.png": "roughness",  # Blender writes roughness to map_Ns
        "stone_metal.png": "metallic",
        "stone_height.png": "height",
    }
    metal = ref_named(stone, "stone_metal.png")
    assert metal.guess.channels == {"metallic": "g"}  # -imfchan g
    assert (metal.wrap_s, metal.wrap_t) == (CLAMP, CLAMP)
    assert stone.uv_transform == UVTransform(offset=(0.25, -1.5), scale=(2.0, 2.0))
    # ... which is MTL's own uv' = s * uv + o, with v up.
    assert np.allclose(stone.uv_transform.apply(np.array([[0.1, 0.2]])), [[0.45, 0.9]])
    assert stone.normal_scale == 0.5  # -bm

    maps = stone.canonical()
    assert np.array_equal(maps.orm[..., 2], images["stone_metal.png"][..., 1])
    assert np.array_equal(maps.orm[..., 1], images["stone_rough.png"])
    assert np.all(maps.orm[..., 0] == 255) and maps.orm_sources == {"roughness", "metallic"}
    assert maps.height.dtype == np.uint16
    assert np.array_equal(maps.height, images["stone_height.png"])  # full 16-bit precision


def test_mtl_scalars_are_factors_only_where_there_is_no_map(tmp_path: Path) -> None:
    _write_stone(tmp_path)
    stone, glass, rough = materials.read_materials(tmp_path / "model.obj").materials
    # A map replaces its scalar (Kd, Ns, Pm), the way Blender reads an MTL.
    assert stone.base_color_factor == (1.0, 1.0, 1.0, 1.0)
    assert (stone.metallic, stone.roughness) == (1.0, 1.0)
    assert (stone.alpha_mode, stone.double_sided) == ("OPAQUE", False)
    assert glass.base_color_factor == (0.5, 0.5, 0.5, 0.25)  # Kd 0.5 is a gray
    assert glass.emissive == (0.1, 0.2, 0.3)
    assert glass.roughness == pytest.approx(1 - math.sqrt(90 / 1000))
    assert (glass.metallic, glass.alpha_mode, glass.double_sided) == (0.0, "BLEND", True)
    assert (rough.roughness, rough.base_color_factor) == (0.35, (0.8, 0.8, 0.8, 1.0))


def test_the_model_s_own_folder_bounds_where_maps_are_looked_for(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.png").write_bytes(encode(synth.albedo(1, SIZE)))
    folder = tmp_path / "model"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "inside.png").write_bytes(b"x")
    (folder / "beside.png").write_bytes(b"x")

    find = materials.find_file
    assert find(folder, "..\\elsewhere\\secret.png") is None
    assert find(folder, "../elsewhere/secret.png") is None
    assert find(folder, "sub/inside.png") == (folder / "sub" / "inside.png").resolve()
    # An asset pack's own folders are searched by base name, case aside.
    assert (
        find(folder, "C:\\pack\\maps\\INSIDE.png") == (folder / "sub" / "inside.png").resolve()
    )
    assert find(folder, "D:\\work\\beside.png") == (folder / "beside.png").resolve()
    assert find(folder, "/home/someone/beside.png") == (folder / "beside.png").resolve()
    assert find(folder, "") is None

    (folder / "model.mtl").write_text("newmtl m\nmap_Kd ..\\elsewhere\\secret.png\n")
    read = materials.read_materials(folder / "model.mtl")
    assert read.materials[0].refs == []
    assert read.warnings == ["m: secret.png was not found next to the model"]


def test_an_mtl_in_a_sub_folder_may_reach_anywhere_in_the_model_s_folder(
    tmp_path: Path,
) -> None:
    (tmp_path / "materials").mkdir()
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / "wall_albedo.png").write_bytes(encode(synth.albedo(3, SIZE)))
    (tmp_path / "materials" / "wall.mtl").write_text(
        "newmtl wall\nmap_Kd ../maps/wall_albedo.png\nNs 98\n"
    )
    (tmp_path / "model.obj").write_text("mtllib materials/wall.mtl\nmtllib gone.mtl\n")
    read = materials.read_materials(tmp_path / "model.obj")
    (wall,) = read.materials
    assert [ref.name for ref in wall.refs] == ["wall_albedo.png"]
    assert read.warnings == ["model.obj: its material library gone.mtl was not found"]

    # Read on its own, the MTL's folder is the model's: ../maps is outside it.
    alone = materials.read_materials(tmp_path / "materials" / "wall.mtl")
    assert alone.materials[0].refs == []
    assert alone.warnings == ["wall: wall_albedo.png was not found next to the model"]
    # No exporter named, so Ns is a Phong exponent: roughness (2 / (n + 2)) ** 0.25.
    assert alone.materials[0].roughness == pytest.approx(0.02**0.25)


# ---------------------------------------------------------------------------
#  Canonical layout
# ---------------------------------------------------------------------------


def _mtl_model(folder: Path, maps: Dict[str, Tuple[str, np.ndarray]], header: str = "") -> Path:
    """A one-material MTL: ``{keyword: (file name, pixels)}``."""
    lines = [header, "newmtl m"] if header else ["newmtl m"]
    for keyword, (name, pixels) in maps.items():
        (folder / name).write_bytes(encode(pixels))
        lines.append(f"{keyword} {name}")
    path = folder / "model.mtl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_directx_normal_map_is_stored_green_up(tmp_path: Path) -> None:
    opengl = normal_map(4, "opengl")
    directx = normal_map(4, "directx")
    path = _mtl_model(tmp_path, {"norm": ("rock_normal_dx.png", directx)})
    spec = materials.read_materials(path).materials[0]
    assert spec.refs[0].guess.y_convention == "directx"
    assert np.array_equal(spec.canonical().normal, opengl)


def test_a_two_channel_normal_map_gets_its_blue_back(tmp_path: Path) -> None:
    full = normal_map(5)
    two = full.copy()
    two[..., 2] = 0
    spec = materials.read_materials(
        _mtl_model(tmp_path, {"norm": ("bc5_normal.png", two)})
    ).materials[0]
    rebuilt = spec.canonical().normal
    assert np.array_equal(rebuilt[..., :2], full[..., :2])
    # z follows from the 8-bit x and y, so it lands within a few steps of the original.
    error = np.abs(rebuilt[..., 2].astype(int) - full[..., 2])
    assert np.median(error) <= 1 and error.max() <= 4
    length = np.linalg.norm(rebuilt.astype(np.float64) / 127.5 - 1.0, axis=2)
    assert np.abs(length - 1.0).max() < 0.02


def test_gloss_becomes_roughness(tmp_path: Path) -> None:
    gloss = synth.gray_map(6, SIZE, 0.3)
    spec = materials.read_materials(
        _mtl_model(tmp_path, {"map_Ns": ("rock_gloss.png", gloss)}, "# 3ds Max exporter")
    ).materials[0]
    assert spec.refs[0].role == "gloss"
    assert np.array_equal(spec.canonical().orm[..., 1], 255 - gloss)


def test_unity_and_hdrp_packings_fill_the_orm_channels(tmp_path: Path) -> None:
    metal_smooth = np.dstack([synth.bimodal_mask(7, SIZE)] * 3 + [synth.gray_map(8, SIZE, 0.4)])
    mask = synth.hdrp_mask(9, SIZE)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    unity = materials.read_materials(
        _mtl_model(tmp_path / "a", {"refl": ("rock_MetallicSmoothness.png", metal_smooth)})
    ).materials[0]
    assert unity.refs[0].role == "metal_smooth"
    orm = unity.canonical().orm
    assert np.array_equal(orm[..., 2], metal_smooth[..., 0])
    assert np.array_equal(orm[..., 1], 255 - metal_smooth[..., 3])

    hdrp = materials.read_materials(
        _mtl_model(tmp_path / "b", {"map_Ks": ("rock_MaskMap.png", mask)})
    ).materials[0]
    assert hdrp.refs[0].role == "mask_hdrp"
    orm = hdrp.canonical().orm
    assert np.array_equal(orm[..., 0], mask[..., 1])
    assert np.array_equal(orm[..., 1], 255 - mask[..., 3])
    assert np.array_equal(orm[..., 2], mask[..., 0])
    assert hdrp.canonical().orm_sources == {"ao", "roughness", "metallic"}


def test_data_in_alpha_is_not_premultiplied_when_resized(tmp_path: Path) -> None:
    """Pillow premultiplies RGBA by alpha when it resizes; smoothness is not coverage."""
    metal_smooth = np.full((SIZE, SIZE, 4), 255, np.uint8)
    metal_smooth[..., 3] = synth.gray_map(10, SIZE, 0.3, 0.3)
    spec = materials.read_materials(
        _mtl_model(tmp_path, {"refl": ("rock_MetallicSmoothness.png", metal_smooth)})
    ).materials[0]
    orm = spec.canonical(max_px=SIZE // 2).orm
    assert orm.shape == (SIZE // 2, SIZE // 2, 3)
    assert np.all(orm[..., 2] == 255)


def test_maps_of_different_sizes_are_resampled_to_the_largest(tmp_path: Path) -> None:
    colour = np.full((SIZE, SIZE, 3), (200, 100, 50), np.uint8)
    cutout = synth.bimodal_mask(11, SIZE * 2)
    spec = materials.read_materials(
        _mtl_model(
            tmp_path,
            {"map_Kd": ("wall_albedo.png", colour), "map_d": ("wall_opacity.png", cutout)},
        )
    ).materials[0]
    assert (spec.alpha_mode, spec.double_sided) == ("MASK", True)
    maps = spec.canonical()
    assert maps.basecolor.shape == (SIZE * 2, SIZE * 2, 4)
    assert np.all(maps.basecolor[..., :3] == (200, 100, 50))  # Lanczos keeps a flat colour
    assert np.array_equal(maps.basecolor[..., 3], cutout)
    small = spec.canonical(max_px=SIZE // 2)
    assert small.basecolor.shape == (SIZE // 2, SIZE // 2, 4)
    assert spec.canonical() is maps  # cached per size


def test_an_eight_bit_height_is_widened_to_sixteen(tmp_path: Path) -> None:
    height = synth.grayscale_height(12, SIZE)
    spec = materials.read_materials(
        _mtl_model(tmp_path, {"disp": ("wall_height.png", height)})
    ).materials[0]
    assert np.array_equal(spec.canonical().height, height.astype(np.uint16) * 257)


def test_maps_with_no_canonical_place_are_shown_as_they_are(tmp_path: Path) -> None:
    specular = synth.albedo(13, SIZE)
    sheen = synth.gray_map(14, SIZE, 0.5)
    spec = materials.read_materials(
        _mtl_model(
            tmp_path,
            {"map_Ks": ("rock_specular.png", specular), "map_Ps": ("rock_sheen.png", sheen)},
        )
    ).materials[0]
    assert spec.sources == {}
    names = [name for name, _ in spec.canonical().others]
    assert names == ["rock_specular.png", "rock_sheen.png"]
    assert np.array_equal(spec.canonical().others[0][1], specular)


def test_unreadable_maps_are_skipped_with_a_warning(tmp_path: Path) -> None:
    (tmp_path / "sky.exr").write_bytes(b"\x76\x2f\x31\x01" + bytes(64))
    (tmp_path / "model.mtl").write_text("newmtl m\nmap_Kd sky.exr\nmap_Ks missing.png\n")
    read = materials.read_materials(tmp_path / "model.mtl")
    assert read.materials[0].refs == []
    assert read.warnings == [
        "m: missing.png was not found next to the model",
        "m: sky.exr was skipped: OpenEXR images are not supported",
    ]


def test_canonical_maps_are_read_only_and_can_be_forgotten(tmp_path: Path) -> None:
    spec = materials.read_materials(
        _mtl_model(tmp_path, {"map_Kd": ("wall_albedo.png", synth.albedo(15, SIZE))})
    ).materials[0]
    maps = spec.canonical()
    with pytest.raises(ValueError):
        maps.basecolor[0, 0, 0] = 1
    spec.clear_cache()
    again = spec.canonical()
    assert again is not maps and np.array_equal(again.basecolor, maps.basecolor)


# ---------------------------------------------------------------------------
#  Corrections
# ---------------------------------------------------------------------------


def test_a_correction_changes_the_role_and_the_canonical_maps(tmp_path: Path) -> None:
    ao = synth.bright_ao(16, SIZE)
    normal = normal_map(17)
    rough = synth.gray_map(18, SIZE, 0.6)
    path = _mtl_model(
        tmp_path,
        {
            "map_Kd": ("wall_albedo.png", ao),
            "norm": ("wall_normal.png", normal),
            "map_Pr": ("wall_roughness.png", rough),
        },
    )
    before = materials.read_materials(path).materials[0]
    assert [ref.role for ref in before.refs] == ["basecolor", "normal", "roughness"]
    assert [ref.key for ref in before.refs] == [
        "0/wall_albedo.png",
        "0/wall_normal.png",
        "0/wall_roughness.png",
    ]

    corrections = {
        "0/wall_albedo.png": {"role": "ao"},
        "0/wall_normal.png": {"flip_green": True},
        "0/wall_roughness.png": {"invert": True},
    }
    read = materials.read_materials(path, corrections)
    assert read.warnings == []
    after = read.materials[0]
    albedo = after.refs[0]
    assert (albedo.role, albedo.guess.channels, albedo.guess.confidence) == (
        "ao",
        {"ao": "r"},
        1.0,
    )
    assert albedo.guess.evidence[0] == "set by hand: AO"
    maps = after.canonical()
    assert maps.basecolor is None
    assert np.array_equal(maps.orm[..., 0], ao)
    assert np.array_equal(maps.orm[..., 1], 255 - rough)
    assert after.refs[1].guess.y_convention == "directx"
    flipped = normal.copy()
    flipped[..., 1] = 255 - flipped[..., 1]
    assert np.array_equal(maps.normal, flipped)


def test_corrections_that_do_not_apply_are_reported(tmp_path: Path) -> None:
    path = _mtl_model(tmp_path, {"map_Kd": ("wall_albedo.png", synth.albedo(19, SIZE))})
    read = materials.read_materials(
        path,
        {
            "0/wall_albedo.png": {"role": "sparkle"},
            "3/gone.png": {"role": "ao"},
        },
    )
    assert read.materials[0].refs[0].role == "basecolor"
    assert read.warnings == [
        "m: the correction for wall_albedo.png was ignored: unknown role 'sparkle'",
        "There is no map 3/gone.png to correct any more",
    ]


# ---------------------------------------------------------------------------
#  FBX
# ---------------------------------------------------------------------------


class I32(int):
    """An FBX 'I' property; plain ints are written as 'L'."""


class Raw(bytes):
    """An FBX 'R' property; plain bytes are written as 'S'."""


def _property(value: Any) -> bytes:
    if isinstance(value, Raw):
        return b"R" + struct.pack("<I", len(value)) + value
    if isinstance(value, bytes):
        return b"S" + struct.pack("<I", len(value)) + value
    if isinstance(value, I32):
        return b"I" + struct.pack("<i", value)
    if isinstance(value, int):
        return b"L" + struct.pack("<q", value)
    return b"D" + struct.pack("<d", value)


Node = Tuple[bytes, List[Any], List[Any]]


def _emit(node: Node, at: int, wide: bool) -> bytes:
    """One record and its children, laid out at an absolute offset."""
    name, props, children = node
    blob = b"".join(_property(p) for p in props)
    word = 8 if wide else 4
    body, cursor = b"", at + 3 * word + 1 + len(name) + len(blob)
    for child in children:
        emitted = _emit(child, cursor, wide)
        body += emitted
        cursor += len(emitted)
    if children:
        body += b"\0" * (3 * word + 1)
        cursor += 3 * word + 1
    head = struct.pack("<QQQ" if wide else "<III", cursor, len(props), len(blob))
    return head + bytes([len(name)]) + name + blob + body


def write_fbx(path: Path, nodes: Sequence[Node], version: int = 7400) -> None:
    wide = version >= 7500
    out = bytearray(fbx_media.MAGIC + struct.pack("<I", version))
    for node in nodes:
        out += _emit(node, len(out), wide)
    out += b"\0" * (3 * (8 if wide else 4) + 1)
    path.write_bytes(bytes(out))


def _p(name: bytes, kind: bytes, value: Any) -> Node:
    return (b"P", [name, kind, b"", b"", value], [])


def _object(kind: bytes, identity: int, name: bytes, *children: Node) -> Node:
    return (kind, [identity, name + b"\x00\x01" + kind, b""], list(children))


def _link(child: int, parent: int, prop: bytes = b"") -> Node:
    return (b"C", [b"OP" if prop else b"OO", child, parent] + ([prop] if prop else []), [])


SKIN_COLOUR = synth.albedo(20, SIZE)
SKIN_ROUGH = synth.gray_map(21, SIZE, 0.35)
SKIN_NORMAL = normal_map(22)
JACKET = synth.albedo(23, SIZE)


def write_scene(
    folder: Path, version: int = 7400, application: bytes = b"Blender (stable FBX IO)"
) -> Path:
    """Three materials (skin, plain, jacket) in a Z-up, metre-unit FBX.

    skin: an embedded colour map, a shininess map left in ``model.fbm``, and a
    normal map whose Video has no content but shares its file with one that
    does.  plain: no textures.  jacket: a colour map through a LayeredTexture,
    whose Texture names no file (its Video does), and a bump map naming a
    file outside the model's folder.
    """
    fbm = folder / "model.fbm"
    fbm.mkdir(parents=True, exist_ok=True)
    (fbm / "skin_b.png").write_bytes(encode(SKIN_ROUGH))
    (folder.parent / "secret.png").write_bytes(encode(JACKET))
    header = (
        b"FBXHeaderExtension",
        [],
        [
            (b"Creator", [b"Blender (stable FBX IO) - 4.1.0 - 5.11.4"], []),
            (
                b"SceneInfo",
                [1, b"GlobalInfo\x00\x01SceneInfo", b"UserData"],
                [
                    (
                        b"Properties70",
                        [],
                        [_p(b"Original|ApplicationName", b"KString", application)],
                    ),
                ],
            ),
        ],
    )
    settings = (
        b"GlobalSettings",
        [],
        [
            (
                b"Properties70",
                [],
                [
                    _p(b"UpAxis", b"int", I32(2)),
                    _p(b"UpAxisSign", b"int", I32(1)),
                    _p(b"FrontAxis", b"int", I32(1)),
                    _p(b"FrontAxisSign", b"int", I32(-1)),
                    _p(b"CoordAxis", b"int", I32(0)),
                    _p(b"CoordAxisSign", b"int", I32(1)),
                    _p(b"UnitScaleFactor", b"double", 100.0),
                ],
            )
        ],
    )
    objects = (
        b"Objects",
        [],
        [
            _object(b"Material", 100, b"skin"),
            _object(b"Material", 101, b"plain"),
            _object(b"Material", 102, b"jacket"),
            _object(
                b"Texture", 200, b"colour", (b"RelativeFilename", [b"D:\\work\\skin_d.png"], [])
            ),
            _object(b"Texture", 201, b"rough", (b"FileName", [b"C:\\maps\\skin_b.png"], [])),
            _object(b"Texture", 202, b"normal", (b"RelativeFilename", [b"skin_n.png"], [])),
            _object(b"Texture", 204, b"jacket"),
            _object(b"Texture", 205, b"bump", (b"RelativeFilename", [b"..\\secret.png"], [])),
            _object(b"LayeredTexture", 400, b"layers"),
            _object(b"Video", 300, b"colour", (b"Content", [Raw(encode(SKIN_COLOUR))], [])),
            _object(b"Video", 302, b"normal", (b"Filename", [b"E:\\x\\skin_n.png"], [])),
            _object(
                b"Video",
                303,
                b"normal copy",
                (b"Filename", [b"skin_n.png"], []),
                (b"Content", [Raw(encode(SKIN_NORMAL))], []),
            ),
            _object(
                b"Video",
                304,
                b"jacket",
                (b"RelativeFilename", [b"textures\\jacket_d.png"], []),
                (b"Content", [Raw(encode(JACKET))], []),
            ),
        ],
    )
    connections = (
        b"Connections",
        [],
        [
            _link(300, 200),
            _link(302, 202),
            _link(304, 204),
            _link(200, 100, b"DiffuseColor"),
            _link(201, 100, b"ShininessExponent"),
            _link(202, 100, b"NormalMap"),
            _link(204, 400),
            _link(400, 102, b"DiffuseColor"),
            _link(205, 102, b"Bump"),
        ],
    )
    path = folder / "model.fbx"
    write_fbx(path, [header, settings, objects, connections], version)
    return path


@pytest.mark.parametrize("version", [7400, 7500])
def test_fbx_media_reads_materials_textures_exporter_and_axes(
    tmp_path: Path, version: int
) -> None:
    media = fbx_media.read(write_scene(tmp_path, version))
    assert media is not None
    assert [m.name for m in media.materials] == ["skin", "plain", "jacket"]
    skin, plain, jacket = media.materials
    assert [(t.property, t.file_name) for t in skin.textures] == [
        ("DiffuseColor", "D:\\work\\skin_d.png"),
        ("ShininessExponent", "C:\\maps\\skin_b.png"),
        ("NormalMap", "skin_n.png"),
    ]
    assert skin.textures[0].data == encode(SKIN_COLOUR)
    assert skin.textures[1].data is None  # on disk, found by materials.py
    assert skin.textures[2].data == encode(SKIN_NORMAL)  # borrowed from the Video that has it
    assert plain.textures == ()
    # A texture that names no file takes its Video's; the label is the last resort.
    assert [(t.property, t.file_name, t.data is not None) for t in jacket.textures] == [
        ("DiffuseColor", "textures\\jacket_d.png", True),
        ("Bump", "..\\secret.png", False),
    ]
    assert media.creator.startswith("Blender (stable FBX IO) - 4.1.0")
    assert media.exporter == media.application == "Blender (stable FBX IO)"
    settings = media.settings
    assert settings.metres_per_unit == 1.0
    # Z up, -Y front (3ds Max): glTF x = x, y = z, z = -y.
    assert np.array_equal(settings.axes(), [[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert np.linalg.det(settings.axes()) == pytest.approx(1.0)


def test_an_fbx_that_cannot_be_parsed_reads_as_none(tmp_path: Path) -> None:
    ascii_fbx = tmp_path / "ascii.fbx"
    ascii_fbx.write_text("; FBX 7.4.0 project file\nObjects: {\n}\n")
    assert fbx_media.read(ascii_fbx) is None
    truncated = write_scene(tmp_path)
    truncated.write_bytes(truncated.read_bytes()[: len(fbx_media.MAGIC) + 40])
    assert fbx_media.read(truncated) is None
    assert fbx_media.read(tmp_path / "missing.fbx") is None


def test_default_axes_are_gltf_s_and_repeated_axes_are_refused() -> None:
    assert np.array_equal(fbx_media.GlobalSettings().axes(), np.eye(3))
    assert not fbx_media.GlobalSettings(up_axis=0, coord_axis=0).valid
    mirrored = fbx_media.GlobalSettings(coord_sign=-1)
    assert np.linalg.det(mirrored.axes()) == pytest.approx(-1.0)


ASSIMP = [
    {"NAME": "skin", "COLOR_DIFFUSE": [0.1, 0.2, 0.3], "SHININESS": 25.0, "REFLECTIVITY": 0.4},
    {
        "NAME": "renamed",
        "COLOR_DIFFUSE": [0.3, 0.3, 0.3],
        "OPACITY": 0.5,
        "COLOR_EMISSIVE": [0.5, 0.0, 0.0],
    },
    {"NAME": "jacket"},
    {"NAME": "DefaultMaterial"},
]


def test_fbx_materials_line_up_with_assimp_s(tmp_path: Path) -> None:
    read = materials.read_materials(write_scene(tmp_path), assimp_materials=ASSIMP)
    assert [m.name for m in read.materials] == ["skin", "renamed", "jacket", "DefaultMaterial"]
    skin, renamed, jacket, default = read.materials
    assert [(ref.name, ref.role) for ref in skin.refs] == [
        ("skin_d.png", "basecolor"),
        ("skin_b.png", "roughness"),  # Blender writes roughness to ShininessExponent
        ("skin_n.png", "normal"),
    ]
    assert [slot.key for ref in skin.refs for slot in ref.slots] == [
        "DiffuseColor",
        "ShininessExponent",
        "NormalMap",
    ]
    assert [ref.name for ref in jacket.refs] == ["jacket_d.png"]  # through the layered texture
    assert renamed.refs == [] and default.refs == []  # "renamed" is "plain", by position
    assert read.warnings == ["jacket: secret.png was not found next to the model"]
    assert read.bindings.exporter == "Blender (stable FBX IO)"
    assert read.bindings.fbx_settings == fbx_media.read(tmp_path / "model.fbx").settings

    maps = skin.canonical()
    assert np.array_equal(maps.basecolor[..., :3], SKIN_COLOUR)
    assert np.array_equal(maps.orm[..., 1], SKIN_ROUGH)
    assert np.array_equal(maps.normal, SKIN_NORMAL)
    # Maps replace scalars; Blender's ReflectionFactor scalar is metallic.
    assert (skin.base_color_factor, skin.roughness, skin.metallic) == ((1.0,) * 4, 1.0, 0.4)
    assert renamed.base_color_factor == (0.3, 0.3, 0.3, 0.5)
    assert (renamed.alpha_mode, renamed.emissive) == ("BLEND", (0.5, 0.0, 0.0))
    assert renamed.roughness == pytest.approx(1 - math.sqrt(20.0) / 10)  # Blender's default
    assert default.base_color_factor == (0.8, 0.8, 0.8, 1.0)


def test_3ds_max_shininess_is_gloss_and_its_reflection_is_not_metal(tmp_path: Path) -> None:
    path = write_scene(tmp_path, application=b"3ds Max")
    read = materials.read_materials(path, assimp_materials=ASSIMP)
    skin, renamed = read.materials[:2]
    assert ref_named(skin, "skin_b.png").role == "gloss"  # 3ds Max binds glossiness there
    assert np.array_equal(skin.canonical().orm[..., 1], 255 - SKIN_ROUGH)
    assert skin.metallic == 0.0  # its ReflectionFactor is reflection strength
    # Its Shininess is a Phong exponent: roughness (2 / (n + 2)) ** 0.25.
    assert renamed.roughness == pytest.approx((2 / 22) ** 0.25)


def test_ascii_fbx_falls_back_to_assimp_s_texture_types(tmp_path: Path) -> None:
    (tmp_path / "textures").mkdir()
    (tmp_path / "a_colour.png").write_bytes(encode(synth.albedo(24, SIZE)))
    (tmp_path / "textures" / "a_bump.png").write_bytes(encode(synth.grayscale_height(25, SIZE)))
    path = tmp_path / "ascii.fbx"
    path.write_text("; FBX 7.4.0 project file\n")
    raw = [
        {
            "NAME": "a",
            "TEXTURES": {1: ["C:\\elsewhere\\a_colour.png"], 6: ["*0"], 5: ["a_bump.png"]},
        }
    ]
    read = materials.read_materials(path, assimp_materials=raw)
    spec = read.materials[0]
    assert [(ref.name, ref.role, ref.slots[0].kind, ref.slots[0].key) for ref in spec.refs] == [
        ("a_colour.png", "basecolor", "assimp", "1"),
        ("a_bump.png", "height", "assimp", "5"),  # grayscale in HEIGHT is a bump map
    ]
    assert read.warnings == ["a: *0 is embedded in a way that cannot be read"]
    assert read.bindings.fbx_settings is None


def test_a_file_whose_materials_cannot_be_read_gives_a_warning(tmp_path: Path) -> None:
    broken = tmp_path / "broken.gltf"
    broken.write_text("{ not json")
    read = materials.read_materials(broken)
    assert read.materials == [] and len(read.warnings) == 1
    assert read.warnings[0].startswith("The materials of broken.gltf could not be read")
    assert materials.read_materials(tmp_path / "mesh.ply") == ([], materials.Bindings(), [])


# ---------------------------------------------------------------------------
#  Real model files
# ---------------------------------------------------------------------------

realasset = pytest.mark.realasset

PIXAL3D = Path(r"C:\_myDrive\repos\Pixal3D-stableprojectorz\output.glb")
TRIPO_GLB = Path(r"C:\_myDrive\repos\AI 3D Shooter 1\Assets\gun_Assets\selected.glb")
FLASHLIGHT = Path(r"C:\_myDrive\repos\temp\forest\Unity_Team_Project\Team_Assets") / (
    r"Flashlight\Misc\Flashlight.FBX"
)
FAE = Path(r"C:\_myDrive\repos\temp\forest\_Picturesque\Picturesque_Assets\Externals") / (
    r"Aquarius Fae Pack\FBX\Hollowed_Elevators_A1.fbx"
)
BAMBOO = Path(r"C:\_myDrive\repos\warewolves-game\Werewolves\Assets\NatureParadaise") / (
    r"SoStylized\Environment\Trees"
)
TRIPO_FBX = Path(r"C:\_myDrive\repos\warewolves-game\Werewolves\Assets\_gm\GrayboxDrafting") / (
    r"cabin_props\3d_gens\bin_bed_Assets\selected.fbx"
)
POLYCAM = Path(r"C:\_myDrive\repos\temp\buildings\beefmaster2\Beefmaster\Models\textured.mtl")
SQUIRREL = Path(r"C:\_myDrive\repos\temp\forest\_ml-agent\Project\Assets\ML-Agents\Ranger") / (
    r"05.Prefeb\Squirreltextures\uploads_files_5388307_squirrel_HP.mtl"
)


def _need(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip("not on this machine: " + ", ".join(missing))


def _roles(spec: MaterialSpec) -> Dict[str, str]:
    return {ref.name: ref.role for ref in spec.refs}


@realasset
def test_pixal3d_output_glb() -> None:
    _need(PIXAL3D)
    read = materials.read_materials(PIXAL3D)
    assert read.warnings == []
    (spec,) = read.materials
    assert [ref.role for ref in spec.refs] == ["basecolor", "orm"]
    assert spec.orm_sources == {"roughness", "metallic"} and spec.normal_space is None
    maps = spec.canonical(max_px=1024)
    assert maps.orm_sources == {"roughness", "metallic"}
    assert np.all(maps.orm[..., 0] == 255)  # R is 0 in the file: never AO
    assert maps.orm[..., 1].mean() > 250 and 180 < maps.orm[..., 2].mean() < 230
    assert np.all(maps.basecolor[..., 3] == 255)  # OPAQUE, though the image has alpha
    assert maps.normal is None and maps.others == ()


@realasset
def test_tripo_selected_glb() -> None:
    _need(TRIPO_GLB)
    (spec,) = materials.read_materials(TRIPO_GLB).materials
    by_role = {ref.role: ref for ref in spec.refs}
    assert set(by_role) == {"basecolor", "normal", "orm"}
    assert (by_role["normal"].guess.y_convention, by_role["normal"].guess.y_confident) == (
        "opengl",
        True,
    )
    assert spec.orm_sources == {"roughness", "metallic"}


@realasset
def test_flashlight_fbx_from_3ds_max() -> None:
    _need(FLASHLIGHT)
    read = materials.read_materials(FLASHLIGHT)
    assert read.bindings.exporter == "3ds Max"
    body = read.materials[read.bindings.names["q"]]
    assert _roles(body) == {
        "LP_FlashLight_AL.png": "basecolor",
        "FlashLight_NM.png": "normal",
        "LP_FlashLight_OP.png": "opacity",
        "LP_FlashLight_RFMorSP2.png": "specular",
        "FlashLight_GL.png": "gloss",
        "ReflMap_PS_sHDR_512.png": "other",
    }
    assert set(body.sources) == {"basecolor", "opacity", "normal", "roughness"}
    assert [ref.name for ref in body.others] == [
        "LP_FlashLight_RFMorSP2.png",
        "ReflMap_PS_sHDR_512.png",
    ]
    assert (body.alpha_mode, body.metallic) == ("MASK", 0.0)


@realasset
def test_hollowed_elevators_fbx_from_blender() -> None:
    _need(FAE)
    read = materials.read_materials(FAE)
    assert read.warnings == []  # maps found in FBX/Textures
    specs = {spec.name: spec for spec in read.materials}
    bark = specs["Bark_A"]
    smooth = next(ref for ref in bark.refs if "MetallicSmoothness" in ref.name)
    assert smooth.role == "metal_smooth"
    assert bark.orm_sources == {"roughness", "metallic"}
    leaves = specs["Leaves_A"]
    assert [(ref.role, ref.guess.channels) for ref in leaves.refs] == [
        ("opacity", {"opacity": "a"})
    ]
    assert leaves.alpha_mode == "MASK"


@realasset
def test_bamboo_fbx_from_blender_with_its_maps_beside_it(tmp_path: Path) -> None:
    names = ["T_BambooBark_BC.png", "T_BambooBark_R.png", "T_Leaf_Bamboo_Filled.png"]
    model = BAMBOO / "Bamboo" / "Meshes" / "SM_BambooSaplingClump2.fbx"
    _need(model, *(BAMBOO / "Textures" / name for name in names))
    shutil.copy(model, tmp_path)
    (tmp_path / "Textures").mkdir()
    for name in names:
        shutil.copy(BAMBOO / "Textures" / name, tmp_path / "Textures")
    read = materials.read_materials(tmp_path / model.name)
    bark = read.materials[read.bindings.names["M_BambooBark.010"]]
    assert _roles(bark) == {
        "T_BambooBark_BC.png": "basecolor",
        "T_BambooBark_R.png": "roughness",
    }
    rough = materials.decode_image((tmp_path / "Textures" / "T_BambooBark_R.png").read_bytes())
    orm = bark.canonical().orm
    assert np.array_equal(orm[..., 1], rough if rough.ndim == 2 else rough[..., 0])


@realasset
def test_tripo_fbx_finds_its_maps_in_the_fbm_folder() -> None:
    _need(TRIPO_FBX)
    read = materials.read_materials(TRIPO_FBX)
    assert read.warnings == []
    (spec,) = read.materials
    assert sorted(ref.role for ref in spec.refs) == ["basecolor", "normal"]


@realasset
def test_polycam_mtl() -> None:
    _need(POLYCAM)
    read = materials.read_materials(POLYCAM)
    assert read.bindings.exporter == "Created by Polycam"
    (spec,) = read.materials
    assert _roles(spec) == {
        "textured_2_uYUNJ2Tu.jpg": "basecolor",
        "textured_0_uYUNJ2Tu.jpg": "normal",
        "textured_1_uYUNJ2Tu.jpg": "ao",
    }
    assert spec.orm_sources == {"ao"}


@realasset
def test_blender_squirrel_mtl_parses_though_its_maps_are_missing() -> None:
    _need(SQUIRREL)
    lines: List[str] = []
    drafts, exporter = materials._read_mtl(SQUIRREL, "", lines.append)
    assert exporter == "Blender 3.6.1 MTL File: 'None'"
    (draft,) = drafts
    assert [(ref.slot.key, ref.slot.options) for ref in draft.refs] == [
        ("map_Kd", {}),
        ("map_Ns", {}),
        ("map_refl", {}),
        ("map_Bump", {"-bm": ("1.000000",)}),
    ]
    assert all(ref.data is None for ref in draft.refs)
    read = materials.read_materials(SQUIRREL)
    assert len(read.warnings) == 4 and read.materials[0].refs == []
