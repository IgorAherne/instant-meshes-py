"""The imported model as it was authored: materials, texture maps and all.

The remesher works on one merged triangle soup with no attributes, which is
what :func:`solver_mesh` hands it.  That throws away everything a texture needs
-- the UVs, the seams they are split along, which face belongs to which
material -- so the file is read once into a :class:`SourceMesh` that keeps all
of it, and the viewport draws the original from that when a texture is asked
for.  Nothing here reaches the solver; it exists so that what you brush on can
be looked at the way the author left it.

Two readers, because no one library covers both formats this has to take:

* trimesh handles glTF and glB, including the images they embed, and every
  format the viewer already accepted.
* FBX needs Assimp, which reads the geometry, the UVs and the material each
  mesh uses -- but hands back ``*0`` for a texture the file embeds, with no way
  to ask what ``*0`` contains.  Embedded is the only kind that can work here (a
  browser uploads one file, not the folder of maps beside it), so the images
  are lifted straight out of the FBX by :mod:`.fbx_media`.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger(__name__)


class AssetError(RuntimeError):
    """Raised when a model file cannot be read, or holds nothing usable."""


#: Texture channels, in the order the viewer numbers them.  A slot means the
#: same kind of map whichever material fills it, so "tex 1" is the normal map
#: on every material that has one rather than "whatever came second".
CHANNELS: Tuple[str, ...] = (
    "base colour",
    "normal",
    # glTF packs roughness and metalness into one image, and an FBX shininess
    # map is the same idea from the other direction; one slot, one picture.
    "metal/rough",
    "specular",
    "emissive",
    "occlusion",
)

#: Ceiling on the buttons the viewport offers.  A material with more maps than
#: this has them in channels nothing here knows how to name.
MAX_SLOTS = 6

#: Longest side an exported preview image keeps.  These are looked at on a
#: model in a viewport, not sampled for a render, and a 4096 square map costs
#: 64 MB of texture memory against 16 MB at this size.
MAX_TEXTURE_PX = 2048

#: Quality for the JPEG an opaque map is re-encoded as.  PNG is kept for maps
#: with alpha, where the channel is the point.
JPEG_QUALITY = 88

#: Suffixes :func:`load_source` can read, and which reader takes each.
TRIMESH_SUFFIXES = frozenset({".obj", ".ply", ".stl", ".off", ".glb", ".gltf", ".dae"})
ASSIMP_SUFFIXES = frozenset({".fbx"})
SUFFIXES = TRIMESH_SUFFIXES | ASSIMP_SUFFIXES


@dataclass(frozen=True)
class Texture:
    """One map, encoded as the bytes a browser can decode directly."""

    channel: str
    mime: str
    data: bytes

    @property
    def slot(self) -> int:
        return CHANNELS.index(self.channel)


@dataclass
class Material:
    """One material of the imported model, and the maps it names."""

    name: str
    textures: Dict[str, Texture] = field(default_factory=dict)

    def channels(self) -> List[str]:
        return [name for name in CHANNELS if name in self.textures]


@dataclass
class SourceMesh:
    """The imported model with its attributes, ready to be drawn as authored.

    Faces are ordered by material, so ``groups`` is all the viewer needs to
    draw each run with the right maps: one draw call per material, no per-face
    lookup, and the same layout three.js wants for a multi-material mesh.
    """

    name: str
    vertices: np.ndarray  # (nV, 3) float32
    uv: np.ndarray  # (nV, 2) float32, zeros where the file carried none
    faces: np.ndarray  # (nF, 3) uint32
    groups: List[Tuple[int, int, int]]  # (material, first face, face count)
    materials: List[Material]

    @property
    def slots(self) -> List[str]:
        """The channels any material fills, in canonical order.

        Compacted rather than fixed: a model whose only maps are colour and
        emissive gets two buttons, not one button and a gap where four
        channels nobody has would otherwise sit.
        """
        present = {name for material in self.materials for name in material.textures}
        return [name for name in CHANNELS if name in present][:MAX_SLOTS]

    def texture(self, slot: int, material: int) -> Optional[Texture]:
        slots = self.slots
        if not 0 <= slot < len(slots) or not 0 <= material < len(self.materials):
            return None
        return self.materials[material].textures.get(slots[slot])


# ---------------------------------------------------------------------------
#  Images
# ---------------------------------------------------------------------------


def encode_texture(image: Any, channel: str) -> Optional[Texture]:
    """Shrink a PIL image to preview size and encode it for a browser.

    Returns None for anything that is not an image, which is what a material
    slot holds when the file named a map it did not include.
    """
    if image is None or not hasattr(image, "size"):
        return None
    try:
        width, height = image.size
        if not (width and height):
            return None
        longest = max(width, height)
        if longest > MAX_TEXTURE_PX:
            ratio = MAX_TEXTURE_PX / float(longest)
            size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
            image = image.resize(size)

        buffer = io.BytesIO()
        # Alpha survives only where it says something. An RGBA map whose alpha
        # is solid throughout is how most exporters write an opaque one, and a
        # PNG of it costs six times the bytes of the JPEG it may as well be.
        if _has_alpha(image):
            image.convert("RGBA").save(buffer, "PNG", optimize=False)
            mime = "image/png"
        else:
            image.convert("RGB").save(buffer, "JPEG", quality=JPEG_QUALITY)
            mime = "image/jpeg"
        return Texture(channel=channel, mime=mime, data=buffer.getvalue())
    except Exception:
        LOG.debug("could not encode the %s map", channel, exc_info=True)
        return None


def _has_alpha(image: Any) -> bool:
    """Whether any pixel is actually see-through."""
    if image.mode not in ("RGBA", "LA", "PA", "P"):
        return False
    try:
        alpha = image.convert("RGBA").getchannel("A")
        return alpha.getextrema()[0] < 255
    except Exception:
        LOG.debug("could not inspect an alpha channel", exc_info=True)
        return True


def decode_texture(data: bytes, channel: str) -> Optional[Texture]:
    """Re-encode raw image bytes lifted out of a file, or drop them."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow ships with trimesh
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return encode_texture(image, channel)
    except Exception:
        LOG.debug("could not read an embedded %s map", channel, exc_info=True)
        return None


# ---------------------------------------------------------------------------
#  trimesh: glTF, glB and everything the viewer already took
# ---------------------------------------------------------------------------

#: PBRMaterial attributes, in channel order.  metallicRoughness is one image
#: with roughness in green and metalness in blue, so both slots show it.
_PBR_MAPS: Tuple[Tuple[str, str], ...] = (
    ("base colour", "baseColorTexture"),
    ("normal", "normalTexture"),
    ("metal/rough", "metallicRoughnessTexture"),
    ("emissive", "emissiveTexture"),
    ("occlusion", "occlusionTexture"),
)


def _trimesh_material(visual: Any, index: int) -> Material:
    raw = getattr(visual, "material", None)
    name = str(getattr(raw, "name", "") or f"material {index}")
    material = Material(name=name)
    if raw is None:
        return material

    for channel, attribute in _PBR_MAPS:
        texture = encode_texture(getattr(raw, attribute, None), channel)
        if texture is not None:
            material.textures[channel] = texture

    # SimpleMaterial, which is what an OBJ's .mtl becomes: one map, no channels.
    if not material.textures:
        texture = encode_texture(getattr(raw, "image", None), "base colour")
        if texture is not None:
            material.textures["base colour"] = texture
    return material


def _load_with_trimesh(path: Path) -> SourceMesh:
    import trimesh

    try:
        # process=False keeps the corners a UV seam split apart apart, which is
        # what the maps are authored against; solver_mesh welds them back for
        # the remesher, which has no use for either the seams or the UVs.
        loaded = trimesh.load(path, process=False)
    except Exception as exc:
        raise AssetError(f"could not read {path.name}: {exc}") from exc

    # dump() bakes the scene graph's transforms into the geometry, which is
    # what puts the parts of a multi-mesh glTF where its author put them.
    if isinstance(loaded, trimesh.Scene):
        parts = [part for part in loaded.dump() if isinstance(part, trimesh.Trimesh)]
    elif isinstance(loaded, trimesh.Trimesh):
        parts = [loaded]
    else:
        raise AssetError(
            f"{path.name} is a {type(loaded).__name__}, not a triangle mesh; "
            "point clouds and curves are not supported"
        )
    if not parts:
        raise AssetError(f"{path.name} does not contain any triangles")

    pieces = []
    for index, part in enumerate(parts):
        faces = np.asarray(part.faces)
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
            continue
        pieces.append(
            (
                np.asarray(part.vertices, dtype=np.float32),
                _vertex_uv(part),
                faces.astype(np.uint32, copy=False),
                _trimesh_material(part.visual, index),
            )
        )
    if not pieces:
        raise AssetError(f"{path.name} does not contain any triangles")
    return _assemble(path.name, pieces)


def _vertex_uv(part: Any) -> Optional[np.ndarray]:
    uv = getattr(getattr(part, "visual", None), "uv", None)
    if uv is None:
        return None
    values = np.asarray(uv, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] != len(part.vertices):
        return None
    return values


# ---------------------------------------------------------------------------
#  Assimp: FBX
# ---------------------------------------------------------------------------

def _assimp_module() -> Any:
    try:
        import assimp_py
    except ImportError as exc:
        raise AssetError(
            "reading FBX needs the assimp-py package: pip install assimp-py"
        ) from exc
    return assimp_py


def _load_with_assimp(path: Path) -> SourceMesh:
    assimp = _assimp_module()
    flags = (
        assimp.Process_Triangulate
        | assimp.Process_GenSmoothNormals
        | assimp.Process_PreTransformVertices
    )
    try:
        scene = assimp.import_file(str(path), flags)
    except Exception as exc:
        raise AssetError(f"could not read {path.name}: {exc}") from exc

    from . import fbx_media

    embedded = fbx_media.read(path)
    materials = [
        _assimp_material(raw, index, embedded)
        for index, raw in enumerate(scene.materials)
    ]
    if not materials:
        materials = [Material(name="material 0")]

    pieces = []
    for mesh in scene.meshes:
        indices = np.asarray(mesh.indices, dtype=np.uint32).reshape(-1, 3)
        if indices.size == 0:
            continue
        vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
        index = min(int(mesh.material_index), len(materials) - 1)
        pieces.append(
            (vertices, _assimp_uv(mesh, len(vertices)), indices, materials[index])
        )
    if not pieces:
        raise AssetError(f"{path.name} does not contain any triangles")
    return _assemble(path.name, pieces)


def _assimp_uv(mesh: Any, count: int) -> Optional[np.ndarray]:
    """The first UV set, which is the one a texture is authored against."""
    sets = getattr(mesh, "texcoords", None)
    if not sets:
        return None
    values = np.asarray(sets[0], dtype=np.float32).reshape(count, -1)
    if values.shape[1] < 2:
        return None
    return np.ascontiguousarray(values[:, :2])


def _assimp_material(
    raw: Any, index: int, embedded: "Dict[str, List[Tuple[str, bytes]]]"
) -> Material:
    name = str(raw.get("NAME") or f"material {index}") if hasattr(raw, "get") else ""
    material = Material(name=name or f"material {index}")

    # Assimp's own paths are either empty or "*0" for an embedded map, so the
    # bytes come from the file itself; the material is found by the name it
    # carries, and by its position where two share one.
    for channel, data in _embedded_for(material.name, index, embedded):
        if channel in material.textures:
            continue
        texture = decode_texture(data, channel)
        if texture is not None:
            material.textures[channel] = texture
    return material


def _embedded_for(
    name: str, index: int, embedded: "Dict[str, List[Tuple[str, bytes]]]"
) -> "List[Tuple[str, bytes]]":
    if name in embedded:
        return embedded[name]
    ordered = list(embedded.values())
    return ordered[index] if index < len(ordered) else []


# ---------------------------------------------------------------------------
#  Assembly
# ---------------------------------------------------------------------------


def _assemble(name: str, pieces: Sequence[Any]) -> SourceMesh:
    """Concatenate the parts, keeping each one's material as a face range.

    Parts sharing a material are merged into one range rather than left as
    several, because a range is a draw call and a model exported part by part
    can easily have hundreds of them.
    """
    by_material: Dict[int, Material] = {}
    order: List[int] = []
    for *_, material in pieces:
        key = id(material)
        if key not in by_material:
            by_material[key] = material
            order.append(key)
    materials = [by_material[key] for key in order]
    slot_of = {key: position for position, key in enumerate(order)}

    vertices: List[np.ndarray] = []
    uvs: List[np.ndarray] = []
    faces: List[np.ndarray] = []
    owners: List[np.ndarray] = []

    offset = 0
    for piece_vertices, piece_uv, piece_faces, material in pieces:
        count = piece_vertices.shape[0]
        vertices.append(piece_vertices)
        uvs.append(
            piece_uv
            if piece_uv is not None
            else np.zeros((count, 2), dtype=np.float32)
        )
        faces.append(piece_faces.astype(np.uint32, copy=False) + offset)
        owners.append(
            np.full(piece_faces.shape[0], slot_of[id(material)], dtype=np.int32)
        )
        offset += count

    all_faces = np.concatenate(faces)
    all_owners = np.concatenate(owners)
    # Stable, so the parts of one material keep the order they were written in.
    sort = np.argsort(all_owners, kind="stable")
    all_faces = all_faces[sort]
    all_owners = all_owners[sort]

    groups: List[Tuple[int, int, int]] = []
    if all_owners.size:
        edges = np.flatnonzero(np.diff(all_owners)) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [all_owners.size]))
        groups = [
            (int(all_owners[start]), int(start), int(end - start))
            for start, end in zip(starts, ends)
        ]

    return SourceMesh(
        name=name,
        vertices=np.concatenate(vertices).astype(np.float32, copy=False),
        uv=np.concatenate(uvs).astype(np.float32, copy=False),
        faces=all_faces,
        groups=groups,
        materials=materials,
    )


# ---------------------------------------------------------------------------
#  Public entry points
# ---------------------------------------------------------------------------


def load_source(path: Path) -> SourceMesh:
    """Read a model file with everything the viewport can show of it."""
    source = Path(path)
    if not source.is_file():
        raise AssetError(f"no such mesh file: {source}")
    suffix = source.suffix.lower()
    if suffix in ASSIMP_SUFFIXES:
        return _load_with_assimp(source)
    if suffix in TRIMESH_SUFFIXES:
        return _load_with_trimesh(source)
    raise AssetError(
        f"unsupported mesh format '{suffix}'; use one of {', '.join(sorted(SUFFIXES))}"
    )



def solver_mesh(source: SourceMesh) -> Tuple[np.ndarray, np.ndarray]:
    """The merged triangle soup the remesher works on.

    The corners a UV seam or a material boundary split apart are welded back
    together here: they are one point of one surface, and a hierarchy built on
    the split copies has a crack running along every seam.
    """
    vertices = source.vertices
    if vertices.shape[0] == 0:
        raise AssetError(f"{source.name} does not contain any vertices")

    # Round before welding so that corners written out at float32 precision
    # from two different parts still land on the same key.
    scale = float(np.abs(vertices).max()) or 1.0
    keys = np.round(vertices / (scale * 1e-6)).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)

    merged = vertices[first]
    faces = inverse.reshape(-1)[source.faces].astype(np.uint32, copy=False)
    # A triangle whose corners welded together has no area and no normal, and
    # the hierarchy builder divides by that area.
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (
        faces[:, 0] != faces[:, 2]
    )
    faces = faces[keep]
    if faces.shape[0] == 0:
        raise AssetError(f"{source.name} does not contain any triangles")
    return merged.astype(np.float32, copy=False), faces
