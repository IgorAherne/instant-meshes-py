"""The imported model as it was authored: geometry, normals, UVs, materials, maps.

The remesher works on one welded triangle soup with no attributes, which is
what :func:`solver_mesh` hands it.  Everything a textured view needs -- the UVs
and the seams they split the surface along, the normals the author shipped,
which face belongs to which material, and every material's maps -- is kept in
a :class:`SourceMesh`, read once from the file.  The viewport draws the model
as its author left it from that, and the Projection bake reads the same object
at full resolution.  Nothing here reaches the solver.

Geometry comes from one of two readers:

* trimesh, for glTF/GLB, OBJ, PLY, STL and OFF.  Its scene graph is walked
  node by node -- positions by the node's matrix, normals by the inverse
  transpose -- rather than flattened with ``Scene.dump()``, which drops the
  authored normals and copies a shared material once per primitive.
* Assimp, for FBX and Collada, with every node's transform baked in.  An FBX
  ends up in glTF's frame, Y up and in metres, from the axis system and unit
  the file itself declares.

The materials come from :func:`.materials.read_materials`, which works out
what each map is for and lays it out canonically (see that module).  Here they
become the short list of *slots* the viewport fetches maps by -- base colour,
normal, ORM, emissive, height, then every map with no place among those -- and
the *buttons* it offers: one per kind of map some material really has.
"""

from __future__ import annotations

import io
import logging
import math
import os
import threading
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from PIL import Image

from . import fbx_media, texroles
from .materials import ROLE_LABELS, WRAP_REPEAT, MaterialRead, MaterialSpec, read_materials

LOG = logging.getLogger(__name__)


class AssetError(RuntimeError):
    """Raised when a model file cannot be read, or holds nothing usable."""


#: Longest side of a preview map.  Previews are looked at on a model in a
#: viewport, not sampled for a render -- the bake reads the full-resolution
#: canonical maps -- and a 4096 square map costs 64 MB of texture memory
#: against 16 MB at this size.
MAX_TEXTURE_PX = 2048

#: Quality of the JPEG a colour map without alpha is sent as.  Its Huffman
#: tables are optimised too, which is lossless and some 3% smaller.
JPEG_QUALITY = 90

#: zlib level of the PNG previews.  Data maps have to be lossless -- a JPEG
#: block in a normal map is a bump -- and level 1 writes several times faster
#: than the default, for a slightly larger file on what is a local connection.
PNG_COMPRESS_LEVEL = 1

#: Suffixes :func:`load_source` can read, and which reader takes each.
TRIMESH_SUFFIXES = frozenset({".obj", ".ply", ".stl", ".off", ".glb", ".gltf"})
#: Collada goes through Assimp: trimesh would need pycollada, which is not
#: shipped.
ASSIMP_SUFFIXES = frozenset({".fbx", ".dae"})
SUFFIXES = TRIMESH_SUFFIXES | ASSIMP_SUFFIXES

#: The canonical maps (see :class:`.materials.CanonicalMaps`), in slot order,
#: and what the viewport calls each.
CANONICAL_MAPS: Tuple[Tuple[str, str], ...] = (
    ("basecolor", "Base colour"),
    ("normal", "Normal"),
    ("orm", "ORM"),
    ("emissive", "Emissive"),
    ("height", "Height"),
)

_SRGB_MAPS = frozenset({"basecolor", "emissive"})


class _ButtonSpec(NamedTuple):
    id: str
    label: str
    map: str
    channel: str
    colorspace: str


#: The per-map views, in the order the viewport lists them.  Roughness,
#: metallic and AO are channels of the one ORM map, opacity is the alpha of
#: the base colour.
_BUTTONS: Tuple[_ButtonSpec, ...] = (
    _ButtonSpec("basecolor", "Base colour", "basecolor", "rgb", "srgb"),
    _ButtonSpec("opacity", "Opacity", "basecolor", "a", "linear"),
    _ButtonSpec("normal", "Normal", "normal", "rgb", "linear"),
    _ButtonSpec("roughness", "Roughness", "orm", "g", "linear"),
    _ButtonSpec("metallic", "Metallic", "orm", "b", "linear"),
    _ButtonSpec("ao", "AO", "orm", "r", "linear"),
    _ButtonSpec("emissive", "Emissive", "emissive", "rgb", "srgb"),
    _ButtonSpec("height", "Height", "height", "r", "linear"),
)


# ---------------------------------------------------------------------------
#  What a read returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Texture:
    """One preview map, encoded as the bytes a browser decodes directly."""

    mime: str
    data: bytes


@dataclass(frozen=True)
class MapSlot:
    """A map the viewport fetches by number: the same kind of map on every material."""

    index: int
    #: "basecolor", "normal", "orm", "emissive" or "height" for a canonical
    #: map; "tex<N>" for one that has no place among those and is shown as
    #: it is.  Also the key of this slot in a material's ``maps``.
    key: str
    #: The canonical map's name, or what the classifier took the map for.
    role: str
    label: str
    colorspace: str


@dataclass(frozen=True)
class Button:
    """One per-map view: a channel of a slot, and how its bytes are encoded."""

    id: str
    label: str
    slot: int
    channel: str
    colorspace: str


class _Layout(NamedTuple):
    slots: List[MapSlot]
    buttons: List[Button]
    #: Per material, the slots it has: slot key -> slot index.
    maps: List[Dict[str, int]]
    #: Per material, the slot key of each of its unplaced maps, in the order
    #: of ``MaterialSpec.others`` (which is also ``CanonicalMaps.others``).
    extra: List[List[str]]


@dataclass(eq=False)
class SourceMesh:
    """The imported model with its attributes, ready to be drawn as authored.

    Faces are ordered by material, so ``groups`` is all a renderer needs to
    draw each run with the right maps: one draw call per material, no
    per-face lookup, and the layout three.js wants for a multi-material mesh.
    Treat it as read-only: the slots and buttons are worked out once.
    """

    name: str
    vertices: np.ndarray  # (nV, 3) float32
    #: (nV, 3) float32, unit length: the file's own where it has them, else
    #: area-weighted over the welded surface, so no UV seam shows as a crease.
    normals: np.ndarray
    #: (nV, 2) float32, v up (0 at the image's bottom row), with the
    #: material's texture transform applied; zeros where the file has none.
    uv: np.ndarray
    faces: np.ndarray  # (nF, 3) uint32
    groups: List[Tuple[int, int, int]]  # (material, first face, face count)
    materials: List[MaterialSpec]
    #: Artist-readable notes about maps that were missing or approximated.
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._previews: Dict[int, Dict[str, Texture]] = {}
        self._locks = [threading.Lock() for _ in self.materials]

    @cached_property
    def _layout(self) -> _Layout:
        return _layout_of(self.materials)

    @property
    def slots(self) -> List[MapSlot]:
        """The maps any material has, canonical ones first."""
        return self._layout.slots

    @property
    def buttons(self) -> List[Button]:
        """One per kind of map some material really has a source for."""
        return self._layout.buttons

    def face_materials(self) -> np.ndarray:
        """``(nF,)`` int32: the index into ``materials`` of every face."""
        owners = np.empty(self.faces.shape[0], dtype=np.int32)
        for material, first, count in self.groups:
            owners[first : first + count] = material
        return owners

    def texture(self, slot: int, material: int) -> Optional[Texture]:
        """One material's preview of one slot, or None where it has no such map.

        The first call for a material encodes all of its previews and then
        frees the decoded images behind them.
        """
        slots = self._layout.slots
        if not (0 <= slot < len(slots) and 0 <= material < len(self.materials)):
            return None
        return self._material_previews(material).get(slots[slot].key)

    def encode_previews(self) -> None:
        """Encode every preview now, and free the decoded images a read keeps.

        A server holding the model for a session calls this straight away:
        otherwise the images decoded to classify the maps stay in memory until
        somebody asks to see them.
        """
        for material in range(len(self.materials)):
            self._material_previews(material)

    def describe(self) -> Dict[str, Any]:
        """The slots, buttons and materials, as JSON-ready values for a viewer."""
        layout = self._layout
        return {
            "slots": [
                {"slot": slot.index, "role": slot.role, "label": slot.label}
                for slot in layout.slots
            ],
            "buttons": [asdict(button) for button in layout.buttons],
            "materials": [
                _describe_material(spec, maps, extra)
                for spec, maps, extra in zip(self.materials, layout.maps, layout.extra)
            ],
            "warnings": list(self.warnings),
        }

    def _material_previews(self, material: int) -> Dict[str, Texture]:
        with self._locks[material]:
            previews = self._previews.get(material)
            if previews is None:
                spec = self.materials[material]
                layout = self._layout
                lossy = {slot.key: slot.colorspace == "srgb" for slot in layout.slots}
                try:
                    previews = _encode_previews(spec, layout.extra[material], lossy)
                except Exception:
                    LOG.warning("could not make the previews of %s", spec.name, exc_info=True)
                    previews = {}
                finally:
                    spec.clear_cache()
                self._previews[material] = previews
            return previews


# ---------------------------------------------------------------------------
#  Slots and buttons
# ---------------------------------------------------------------------------


def _canonical_maps_of(spec: MaterialSpec) -> FrozenSet[str]:
    """The canonical maps ``spec.canonical()`` will have, decided without decoding."""
    sources = spec.sources
    found = {name for name in ("normal", "emissive", "height") if name in sources}
    if (
        spec.spec_gloss is not None
        or "basecolor" in sources
        or ("opacity" in sources and spec.alpha_mode != "OPAQUE")
    ):
        found.add("basecolor")
    if spec.orm_sources:
        found.add("orm")
    return frozenset(found)


def _buttons_of(spec: MaterialSpec) -> FrozenSet[str]:
    """The per-map views ``spec`` has a real texture for.

    Where a map is missing its canonical channel is 255, and the factor
    decides: a Roughness button over that would show a white nothing.
    """
    sources = spec.sources
    found = set(spec.orm_sources)
    found |= {name for name in ("normal", "emissive", "height") if name in sources}
    if "basecolor" in sources or spec.spec_gloss is not None:
        found.add("basecolor")
    if "opacity" in sources and spec.alpha_mode != "OPAQUE":
        found.add("opacity")
    return frozenset(found)


def _layout_of(materials: Sequence[MaterialSpec]) -> _Layout:
    present = [_canonical_maps_of(spec) for spec in materials]
    real = frozenset().union(*(_buttons_of(spec) for spec in materials))

    slots: List[MapSlot] = []
    for key, label in CANONICAL_MAPS:
        if any(key in maps for maps in present):
            colorspace = "srgb" if key in _SRGB_MAPS else "linear"
            slots.append(MapSlot(len(slots), key, key, label, colorspace))
    index_of = {slot.key: slot.index for slot in slots}
    buttons = [
        Button(spec.id, spec.label, index_of[spec.map], spec.channel, spec.colorspace)
        for spec in _BUTTONS
        if spec.id in real
    ]

    # Maps with no canonical place share a slot with their counterparts on
    # the other materials: the first specular map of every material is one
    # "Specular" button, not one button per file.
    extra: List[List[str]] = []
    groups: Dict[Tuple[str, int], str] = {}
    for spec in materials:
        seen: Dict[str, int] = {}
        keys: List[str] = []
        for ref in spec.others:
            rank = seen.get(ref.role, 0)
            seen[ref.role] = rank + 1
            group = (ref.role, rank)
            if group not in groups:
                groups[group] = f"tex{len(groups)}"
            keys.append(groups[group])
        extra.append(keys)

    taken = {button.label for button in buttons}
    unnamed = 0
    for (role, _), key in groups.items():
        if role == "other":
            label = f"tex {unnamed}"
            unnamed += 1
        else:
            label = _free_label(ROLE_LABELS.get(role, role), taken)
        taken.add(label)
        colorspace = "srgb" if role in texroles.SRGB_ROLES else "linear"
        slot = MapSlot(len(slots), key, role, label, colorspace)
        slots.append(slot)
        index_of[key] = slot.index
        buttons.append(Button(key, label, slot.index, "rgb", colorspace))

    maps = [
        {key: index_of[key] for key, _ in CANONICAL_MAPS if key in names}
        | {key: index_of[key] for key in keys}
        for names, keys in zip(present, extra)
    ]
    return _Layout(slots, buttons, maps, extra)


def _free_label(label: str, taken: AbstractSet[str]) -> str:
    """``label``, or "``label`` 2", "``label`` 3"... when a button already has it."""
    candidate, count = label, 1
    while candidate in taken:
        count += 1
        candidate = f"{label} {count}"
    return candidate


def _number(value: Any, default: float) -> float:
    """A float JSON can carry: a malformed file can hold NaN, which JSON cannot."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _describe_material(
    spec: MaterialSpec, maps: Dict[str, int], extra: Sequence[str]
) -> Dict[str, Any]:
    # Which button each file feeds, through which of its channels: exact,
    # where "what role did the classifier give it" would not say whether the
    # map won its quantity or lost it to a better one.
    feeds: Dict[int, Dict[str, str]] = {}
    for quantity, ref in spec.sources.items():
        channel = ref.guess.channels.get(quantity, "rgb")
        if quantity == "specular":
            # Only a specular/glossiness material counts one, and converting
            # it to metal/rough is what makes its colour and metalness.
            for button in ("basecolor", "metallic"):
                feeds.setdefault(id(ref), {}).setdefault(button, channel)
        elif quantity != "opacity" or spec.alpha_mode != "OPAQUE":
            feeds.setdefault(id(ref), {})[quantity] = channel
    for ref, key in zip(spec.others, extra):
        feeds.setdefault(id(ref), {})[key] = "rgb"

    wrapped = spec.sources.get("basecolor") or (spec.refs[0] if spec.refs else None)
    wrap = [WRAP_REPEAT, WRAP_REPEAT] if wrapped is None else [wrapped.wrap_s, wrapped.wrap_t]
    return {
        "name": spec.name,
        "maps": maps,
        "base_color_factor": [_number(v, 1.0) for v in spec.base_color_factor],
        "metallic": _number(spec.metallic, 1.0),
        "roughness": _number(spec.roughness, 1.0),
        "emissive": [_number(v, 0.0) for v in spec.emissive],
        "occlusion_strength": _number(spec.occlusion_strength, 1.0),
        "normal_scale": _number(spec.normal_scale, 1.0),
        "normal_space": spec.normal_space,
        "orm_sources": sorted(spec.orm_sources),
        "alpha_mode": spec.alpha_mode,
        "alpha_cutoff": _number(spec.alpha_cutoff, 0.5),
        "double_sided": bool(spec.double_sided),
        "wrap": wrap,
        "guesses": [
            {
                "key": ref.key,
                "file": ref.name,
                "role": ref.role,
                "label": ROLE_LABELS.get(ref.role, ref.role),
                "confidence": round(_number(ref.guess.confidence, 0.0), 3),
                "y_convention": ref.guess.y_convention,
                "y_confident": bool(ref.guess.y_confident),
                "evidence": list(ref.guess.evidence),
                "channels": feeds.get(id(ref), {}),
            }
            for ref in spec.refs
        ],
    }


# ---------------------------------------------------------------------------
#  Previews
# ---------------------------------------------------------------------------


def _encode(image: np.ndarray, *, lossy: bool) -> Texture:
    """A uint8 image as JPEG (colour, opaque, ``lossy``) or PNG (everything else).

    An alpha channel that is solid throughout is dropped: it is how most
    exporters write an opaque map, and it would cost a PNG several times the
    size of the JPEG the map may as well be.
    """
    if image.ndim == 3 and image.shape[2] in (2, 4) and int(image[..., -1].min()) == 255:
        image = image[..., :-1]
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    picture = Image.fromarray(np.ascontiguousarray(image))
    buffer = io.BytesIO()
    if lossy and picture.mode == "RGB":
        picture.save(buffer, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return Texture("image/jpeg", buffer.getvalue())
    picture.save(buffer, "PNG", compress_level=PNG_COMPRESS_LEVEL)
    return Texture("image/png", buffer.getvalue())


def _encode_previews(
    spec: MaterialSpec, extra: Sequence[str], lossy: Mapping[str, bool]
) -> Dict[str, Texture]:
    maps = spec.canonical(MAX_TEXTURE_PX)
    previews: Dict[str, Texture] = {}
    if maps.basecolor is not None:
        previews["basecolor"] = _encode(maps.basecolor, lossy=True)
    if maps.normal is not None:
        previews["normal"] = _encode(maps.normal, lossy=False)
    if maps.orm is not None:
        previews["orm"] = _encode(maps.orm, lossy=False)
    if maps.emissive is not None:
        previews["emissive"] = _encode(maps.emissive, lossy=True)
    if maps.height is not None:
        # 16 bits scaled to 8, never clipped: a browser texture holds 8.
        height = (maps.height.astype(np.uint32) * 255 + 32767) // 65535
        previews["height"] = _encode(height.astype(np.uint8), lossy=False)
    for key, (_, image) in zip(extra, maps.others):
        previews[key] = _encode(image, lossy=lossy.get(key, False))
    return previews


# ---------------------------------------------------------------------------
#  Geometry
# ---------------------------------------------------------------------------


@dataclass
class _Piece:
    """One part of the model with its own vertices and one material."""

    vertices: np.ndarray  # (n, 3) float64
    faces: np.ndarray  # (m, 3) int64
    #: (n, 3) as the file wrote them, or None where it wrote none.
    normals: Optional[np.ndarray]
    #: (n, 2), v up, or None where the file has no texture coordinates.
    uv: Optional[np.ndarray]
    #: Index into the read's materials; None for the format's default material.
    material: Optional[int]


def _transform(piece: _Piece, matrix: np.ndarray) -> _Piece:
    """``piece`` moved by a 4x4 (or 3x3 linear) matrix.

    Normals go through the inverse transpose, which keeps them perpendicular
    under non-uniform scale.  A mirroring matrix turns every triangle inside
    out, so their winding is reversed to keep them facing the way the normals
    do.
    """
    linear = matrix[:3, :3]
    offset = matrix[:3, 3] if matrix.shape == (4, 4) else np.zeros(3)
    if np.array_equal(linear, np.eye(3)) and not offset.any():
        return piece
    determinant = float(np.linalg.det(linear))
    normals = None
    if piece.normals is not None and determinant != 0.0 and math.isfinite(determinant):
        normals = piece.normals @ np.linalg.inv(linear)
    return replace(
        piece,
        vertices=piece.vertices @ linear.T + offset,
        normals=normals,
        faces=piece.faces[:, ::-1] if determinant < 0.0 else piece.faces,
    )


def _weld(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(first, inverse)``: one representative per position, and each vertex's.

    Rounded to a millionth of the model's size first, so that corners written
    out at float32 precision by two different parts still land on one key.
    """
    scale = float(np.abs(vertices).max()) if vertices.size else 1.0
    keys = np.round(vertices / ((scale or 1.0) * 1e-6)).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return first, inverse.reshape(-1)


def _smooth_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``(nV, 3)`` float32 area-weighted normals over the welded surface.

    Welded, because the corners a UV seam or a material boundary split apart
    are one point of one surface: normals averaged per split corner would show
    a crease along every seam.  A vertex no face touches points up.
    """
    first, inverse = _weld(vertices)
    welded = inverse[faces]
    points = np.asarray(vertices, dtype=np.float64)[first]
    corners = points[welded]
    # Twice each triangle's area times its normal: the weighting comes free.
    weighted = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    summed = np.zeros((first.shape[0], 3))
    for axis in range(3):
        for corner in range(3):
            summed[:, axis] += np.bincount(
                welded[:, corner], weights=weighted[:, axis], minlength=first.shape[0]
            )
    length = np.linalg.norm(summed, axis=1)
    flat = length <= 0.0
    summed[flat] = (0.0, 1.0, 0.0)
    length[flat] = 1.0
    return (summed / length[:, None]).astype(np.float32)[inverse]


def _unit_normals(vertices: np.ndarray, faces: np.ndarray, authored: np.ndarray) -> np.ndarray:
    """The authored normals made unit length, computed ones where they are missing."""
    length = np.linalg.norm(authored, axis=1)
    good = np.isfinite(length) & (length > 1e-12)
    normals = np.empty(authored.shape, dtype=np.float32)
    normals[good] = authored[good] / length[good, None]
    if not good.all():
        normals[~good] = _smooth_normals(vertices, faces)[~good]
    return normals


# --- trimesh: glTF, OBJ, PLY, STL, OFF ----------------------------------------

#: glTF primitive modes trimesh loads (points, lines, triangles, strips), and
#: those of them that are surfaces.
_TRIMESH_MODES = frozenset({0, 1, 4, 5})
_SURFACE_MODES = frozenset({4, 5})


def _library_resolver(path: Path, names: Sequence[str]) -> Any:
    """A trimesh resolver that answers every material library with bare names.

    :func:`.materials.read_materials` has read the real libraries -- all of
    them, where trimesh reads the first -- so all trimesh has to do is tag each
    part of the OBJ with the ``usemtl`` name it is drawn with.  Given only
    ``newmtl`` lines it also never opens a map a second time.
    """
    from trimesh.resolvers import FilePathResolver

    library = "".join(f"newmtl {name}\n" for name in names).encode("utf-8")

    class NamesOnly(FilePathResolver):
        def get(self, name: str) -> bytes:
            if name.strip().lower().endswith(".mtl"):
                return library
            return super().get(name)

    return NamesOnly(str(path))


def _authored_normals(geometry: Any) -> Optional[np.ndarray]:
    """The vertex normals a file wrote, or None when trimesh would compute them.

    trimesh keeps loaded normals in its cache and computes the property on
    demand otherwise; only the cache says which of the two it would hand back.
    """
    if "vertex_normals" not in geometry._cache:
        return None
    return np.asarray(geometry.vertex_normals, dtype=np.float64)


def _trimesh_uv(geometry: Any) -> Optional[np.ndarray]:
    uv = getattr(getattr(geometry, "visual", None), "uv", None)
    if uv is None:
        return None
    values = np.asarray(uv, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or values.shape[0] != len(geometry.vertices):
        return None
    return values[:, :2]


def _gltf_primitives(scene: Any, read: MaterialRead) -> Optional[Dict[str, Optional[int]]]:
    """Geometry name -> glTF material index, or None when they cannot be paired.

    trimesh adds one geometry per primitive it keeps, in file order, and gives
    primitives that share a material one material object.  Pairing is only
    trusted when the count, the kind of every primitive and that sharing all
    agree.
    """
    import trimesh

    kept = [p for p in read.bindings.primitives if p.mode in _TRIMESH_MODES]
    names = list(scene.geometry)
    if len(kept) != len(names):
        return None
    by_object: Dict[int, Optional[int]] = {}
    by_index: Dict[Optional[int], int] = {}
    for name, primitive in zip(names, kept):
        geometry = scene.geometry[name]
        if isinstance(geometry, trimesh.Trimesh) != (primitive.mode in _SURFACE_MODES):
            return None
        material = getattr(getattr(geometry, "visual", None), "material", None)
        if material is None or primitive.material is None:
            continue
        if by_object.setdefault(id(material), primitive.material) != primitive.material:
            return None
        if by_index.setdefault(primitive.material, id(material)) != id(material):
            return None
    return {name: primitive.material for name, primitive in zip(names, kept)}


def _materials_by_geometry(
    scene: Any, path: Path, read: Optional[MaterialRead]
) -> Dict[str, Optional[int]]:
    if read is None:
        return {}
    if path.suffix.lower() in (".glb", ".gltf"):
        paired = _gltf_primitives(scene, read)
        if paired is not None:
            return paired
        LOG.info("%s: primitives matched to materials by name", path.name)
    names = read.bindings.names
    found: Dict[str, Optional[int]] = {}
    for key, geometry in scene.geometry.items():
        material = getattr(getattr(geometry, "visual", None), "material", None)
        found[key] = names.get(str(getattr(material, "name", "")))
    return found


def _read_trimesh(path: Path, read: Optional[MaterialRead]) -> List[_Piece]:
    import trimesh

    options: Dict[str, Any] = {}
    if path.suffix.lower() == ".obj":
        if read is not None and read.bindings.names:
            options["resolver"] = _library_resolver(path, list(read.bindings.names))
        else:
            options["skip_materials"] = True
    try:
        # process=False keeps apart the corners a UV seam split apart, which
        # is what the maps are authored against; solver_mesh welds them back.
        scene = trimesh.load_scene(path, process=False, **options)
    except Exception as exc:
        raise AssetError(f"could not read {path.name}: {exc}") from exc

    material_of = _materials_by_geometry(scene, path, read)
    pieces: List[_Piece] = []
    for node in scene.graph.nodes_geometry:
        matrix, name = scene.graph[node]
        geometry = scene.geometry.get(name)
        if not isinstance(geometry, trimesh.Trimesh):
            continue  # points and lines: nothing to draw a map on
        faces = np.asarray(geometry.faces, dtype=np.int64).reshape(-1, 3)
        if faces.shape[0] == 0:
            continue
        piece = _Piece(
            vertices=np.asarray(geometry.vertices, dtype=np.float64),
            faces=faces,
            normals=_authored_normals(geometry),
            uv=_trimesh_uv(geometry),
            material=material_of.get(name),
        )
        pieces.append(_transform(piece, np.asarray(matrix, dtype=np.float64)))
    return pieces


# --- Assimp: FBX, Collada ---------------------------------------------------


def _assimp_module() -> Any:
    try:
        import assimp_py
    except ImportError as exc:
        raise AssetError(
            "reading FBX needs the assimp-py package: pip install assimp-py"
        ) from exc
    return assimp_py


def _assimp_uv(mesh: Any, count: int) -> Optional[np.ndarray]:
    """The first UV set, which is the one a texture is authored against."""
    sets = getattr(mesh, "texcoords", None)
    if not sets:
        return None
    values = np.asarray(sets[0], dtype=np.float64).reshape(count, -1)
    if values.shape[1] < 2:
        return None
    return values[:, :2]


def _metres_per_unit(path: Path, read: Optional[MaterialRead]) -> float:
    """How many metres one unit of an FBX's coordinates is; 1 for anything else.

    Assimp already turns an FBX's axis system to Y up, through the matrix it
    puts on the root node, but it leaves the coordinates in the file's unit:
    centimetres unless the file says otherwise.  Blender, Unity and Unreal all
    scale them by the unit the GlobalSettings declare, and so does this.
    Collada needs neither: Assimp applies its up axis and unit itself.
    """
    if path.suffix.lower() != ".fbx":
        return 1.0
    settings = read.bindings.fbx_settings if read is not None else None
    if settings is None:
        media = fbx_media.read(path)
        settings = media.settings if media is not None else None
    scale = settings.metres_per_unit if settings is not None else 1.0
    return scale if math.isfinite(scale) and scale > 0.0 else 1.0


def _node_meshes(root: Any, frame: np.ndarray) -> Iterator[Tuple[np.ndarray, List[int]]]:
    """``(world matrix, mesh indices)`` of every Assimp node that draws a mesh.

    ``frame`` is the 4x4 matrix the whole scene goes through last.  The walk
    replaces Assimp's own PreTransformVertices step, which corrupts the heap
    -- and takes the whole server down with it -- on some rigged FBX
    characters.
    """
    stack = [(root, frame)]
    while stack:
        node, parent = stack.pop()
        world = parent @ np.asarray(node.transformation, dtype=np.float64).reshape(4, 4)
        indices = list(node.mesh_indices or ())
        if indices:
            yield world, indices
        stack.extend((child, world) for child in reversed(list(node.children)))


def _assimp_piece(mesh: Any, materials: int) -> Optional[_Piece]:
    if mesh.num_faces == 0 or mesh.num_indices != 3 * mesh.num_faces:
        return None  # points or lines
    vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
    normals = mesh.normals
    material = int(mesh.material_index)
    return _Piece(
        vertices=vertices,
        faces=np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3),
        normals=(
            None if normals is None else np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        ),
        uv=_assimp_uv(mesh, vertices.shape[0]),
        material=material if 0 <= material < materials else None,
    )


def _read_assimp(
    path: Path, overrides: Optional[Mapping[str, Mapping[str, Any]]], with_materials: bool
) -> Tuple[List[_Piece], Optional[MaterialRead]]:
    assimp = _assimp_module()
    # Triangles, with points and lines sorted out of them.  No generated
    # normals: where a file has none, _smooth_normals computes them over the
    # welded surface.
    try:
        scene = assimp.import_file(
            str(path), assimp.Process_Triangulate | assimp.Process_SortByPType
        )
    except Exception as exc:
        raise AssetError(f"could not read {path.name}: {exc}") from exc

    read = (
        read_materials(path, overrides, assimp_materials=scene.materials)
        if with_materials
        else None
    )
    count = len(read.materials) if read is not None else 0
    meshes = [_assimp_piece(mesh, count) for mesh in scene.meshes]
    scale = _metres_per_unit(path, read)
    frame = np.diag([scale, scale, scale, 1.0])
    pieces: List[_Piece] = []
    for world, indices in _node_meshes(scene.root_node, frame):
        for index in indices:
            piece = meshes[index] if 0 <= index < len(meshes) else None
            if piece is not None:
                pieces.append(_transform(piece, world))
    return pieces, read


# --- Assembly ---------------------------------------------------------------


def _default_material(suffix: str) -> MaterialSpec:
    """What a part with no material is drawn with.

    glTF defines its own default (white, fully metallic and rough, which is
    also what a glTF viewer shows); every other format gets a neutral clay.
    """
    if suffix in (".glb", ".gltf"):
        return MaterialSpec(name="default")
    return MaterialSpec(
        name="default", base_color_factor=(0.8, 0.8, 0.8, 1.0), metallic=0.0, roughness=0.5
    )


def _assemble(
    name: str,
    pieces: Sequence[_Piece],
    specs: Sequence[MaterialSpec],
    default: MaterialSpec,
    warnings: List[str],
) -> SourceMesh:
    """Concatenate the parts, each material's faces one run.

    Parts sharing a material become one run rather than several, because a
    run is a draw call and a model exported part by part can easily have
    hundreds of them.
    """
    if not pieces:
        raise AssetError(f"{name} does not contain any triangles")
    # In the file's own order, the default material last.
    order = sorted(
        {piece.material for piece in pieces}, key=lambda key: (key is None, key or 0)
    )
    materials = [default if key is None else specs[key] for key in order]
    slot_of = {key: position for position, key in enumerate(order)}

    vertices: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    uvs: List[np.ndarray] = []
    faces: List[np.ndarray] = []
    owners: List[np.ndarray] = []
    offset = 0
    for piece in pieces:
        count = piece.vertices.shape[0]
        spec = materials[slot_of[piece.material]]
        uv = piece.uv
        if uv is None:
            note = f"{spec.name}: the model has no UVs there, so its maps cannot be placed"
            if spec.refs and note not in warnings:
                warnings.append(note)
            uv = np.zeros((count, 2))
        elif spec.uv_transform is not None:
            uv = spec.uv_transform.apply(uv, v_up=True)
        vertices.append(piece.vertices)
        normals.append(
            piece.normals if piece.normals is not None else np.full((count, 3), np.nan)
        )
        uvs.append(uv)
        faces.append(piece.faces + offset)
        owners.append(np.full(piece.faces.shape[0], slot_of[piece.material], dtype=np.int32))
        offset += count

    all_faces = np.concatenate(faces)
    if offset >= 2**32 or int(all_faces.min()) < 0 or int(all_faces.max()) >= offset:
        raise AssetError(f"{name} has faces that refer to vertices it does not have")
    all_owners = np.concatenate(owners)
    # Stable, so the parts of one material keep the order they were written in.
    sort = np.argsort(all_owners, kind="stable")
    all_faces = all_faces[sort].astype(np.uint32)
    all_owners = all_owners[sort]
    edges = np.flatnonzero(np.diff(all_owners)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [all_owners.size]))
    groups = [
        (int(all_owners[start]), int(start), int(end - start))
        for start, end in zip(starts, ends)
    ]

    all_vertices = np.concatenate(vertices).astype(np.float32)
    return SourceMesh(
        name=name,
        vertices=all_vertices,
        normals=_unit_normals(all_vertices, all_faces, np.concatenate(normals)),
        uv=np.concatenate(uvs).astype(np.float32),
        faces=all_faces,
        groups=groups,
        materials=materials,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
#  Public entry points
# ---------------------------------------------------------------------------


def load_source(
    path: Union[str, os.PathLike],
    overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    materials: bool = True,
) -> SourceMesh:
    """Read a model file with everything a textured view or a bake needs of it.

    ``overrides`` are the user's corrections of what a map is, keyed by
    ``TexRef.key`` (see :func:`.materials.read_materials`).  With
    ``materials=False`` only the geometry is read -- what a remesh needs -- and
    every face gets the format's default material; that skips decoding every
    map to classify it.
    """
    source = Path(path)
    if not source.is_file():
        raise AssetError(f"no such mesh file: {source}")
    suffix = source.suffix.lower()
    if suffix not in SUFFIXES:
        raise AssetError(
            f"unsupported mesh format '{suffix}'; use one of {', '.join(sorted(SUFFIXES))}"
        )
    if suffix in ASSIMP_SUFFIXES:
        pieces, read = _read_assimp(source, overrides, materials)
    else:
        read = read_materials(source, overrides) if materials else None
        pieces = _read_trimesh(source, read)
    return _assemble(
        source.name,
        pieces,
        read.materials if read is not None else [],
        _default_material(suffix),
        list(read.warnings) if read is not None else [],
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

    first, inverse = _weld(vertices)
    merged = vertices[first]
    faces = inverse[source.faces].astype(np.uint32, copy=False)
    # A triangle whose corners welded together has no area and no normal, and
    # the hierarchy builder divides by that area.
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[keep]
    if faces.shape[0] == 0:
        raise AssetError(f"{source.name} does not contain any triangles")
    return merged.astype(np.float32, copy=False), faces
