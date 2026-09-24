"""Materials of an imported model: what each texture is for, in one layout.

A model's materials arrive in three dialects.  glTF says exactly what each map
is and how its factors combine with it.  An MTL file names maps by keyword and
an FBX by the material property they are bound to, and neither is as exact as
it looks (see :mod:`.texroles`).  :func:`read_materials` reads all three --
glTF JSON directly, MTL with its own parser, FBX through :mod:`.fbx_media` with
Assimp as the fallback for ASCII FBX and Collada -- asks
:func:`.texroles.classify` what every image is, and returns one
:class:`MaterialSpec` per material.

A MaterialSpec keeps the original bytes of every map, never a re-encoded copy,
and :meth:`MaterialSpec.canonical` decodes them into the one layout everything
downstream is written against, the viewer's lit preview and the baker alike:

- **basecolor** RGBA, sRGB, alpha = opacity (255 when the material is opaque);
- **normal** RGB, tangent space with green up (OpenGL), or object space;
- **orm** RGB, linear: R ambient occlusion, G roughness, B metallic, each 255
  where the material has no map for it (its factor then decides);
- **emissive** RGB, sRGB;
- **height**, 16-bit;
- anything else, as it is.

Factors stay separate and keep glTF's meaning: what renders is factor x map.
MTL and FBX scalars are different -- a map replaces its scalar rather than
scaling it, which is how Blender's importers read them -- so there a quantity
that has a map gets a factor of 1.
"""

from __future__ import annotations

import base64
import binascii
import functools
import hashlib
import io
import json
import logging
import math
import mmap
import os
import re
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from urllib.parse import unquote

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import fbx_media, texroles, texstats
from .texroles import RoleGuess, Slot

LOG = logging.getLogger(__name__)

#: glTF sampler wrap modes; OBJ and FBX maps repeat.
WRAP_REPEAT = 10497
WRAP_CLAMP = 33071
WRAP_MIRROR = 33648

ALPHA_MODES = ("OPAQUE", "MASK", "BLEND")

#: What the interface calls each role.  The canonical quantities (basecolor,
#: opacity, normal, roughness, metallic, ao, emissive, height, specular) are
#: roles too and are named the same way.
ROLE_LABELS: Dict[str, str] = {
    "basecolor": "Base colour",
    "normal": "Normal",
    "height": "Height",
    "roughness": "Roughness",
    "gloss": "Gloss",
    "metallic": "Metallic",
    "specular": "Specular",
    "spec_gloss": "Specular/gloss",
    "ao": "AO",
    "emissive": "Emissive",
    "opacity": "Opacity",
    "orm": "ORM",
    "metal_smooth": "Metallic/smoothness",
    "mask_hdrp": "HDRP mask",
    "other": "Other",
}

#: Quantities a map can supply, and the roles that supply only that one.  A
#: map whose role is dedicated to a quantity wins it over a packed map that
#: also carries it; between equals the more confident guess wins.
_DEDICATED: Dict[str, Tuple[str, ...]] = {
    "basecolor": ("basecolor",),
    "opacity": ("opacity",),
    "normal": ("normal",),
    "roughness": ("roughness", "gloss"),
    "metallic": ("metallic",),
    "ao": ("ao",),
    "emissive": ("emissive",),
    "height": ("height",),
    "specular": ("specular", "spec_gloss"),
}
QUANTITIES: Tuple[str, ...] = tuple(_DEDICATED)

#: The canonical ORM map's channels, in order.
ORM_CHANNELS: Tuple[str, ...] = ("ao", "roughness", "metallic")

GLTF_SUFFIXES = frozenset({".glb", ".gltf"})
OBJ_SUFFIXES = frozenset({".obj", ".mtl"})
ASSIMP_SUFFIXES = frozenset({".fbx", ".dae"})

#: Ceiling on one image file, so that a corrupt length or a stray video
#: cannot be read whole.
MAX_IMAGE_BYTES = fbx_media.MAX_IMAGE_BYTES

#: Decoded images one read keeps for reuse, most recently used first.  Enough
#: for the two 4096 px maps of a Pixal3D model to be decoded once for the
#: classifier and reused for the canonical maps right after.
PIXEL_CACHE_BYTES = 512 * 1024 * 1024

#: Rows converted at a time from specular/glossiness to metal/rough, which
#: keeps the conversion's float64 temporaries to tens of megabytes.
_BAND_ROWS = 256


# ---------------------------------------------------------------------------
#  Images
# ---------------------------------------------------------------------------


class ImageError(ValueError):
    """An image that cannot be decoded; the message says why, for an artist."""


#: Formats Pillow cannot decode, recognised by their first bytes so the
#: warning can name them.
_UNSUPPORTED = (
    (b"\x76\x2f\x31\x01", "OpenEXR"),
    (b"\xabKTX 20\xbb", "KTX2 / Basis Universal"),
    (b"#?RADIANCE", "Radiance HDR"),
    (b"#?RGBE", "Radiance HDR"),
)

_EIGHT_BIT_MODES = frozenset({"L", "LA", "RGB", "RGBA"})
_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16L", "I;16B", "I;16N"})


def decode_image(data: bytes) -> np.ndarray:
    """Pixels of an encoded image, with the channel count it was stored with.

    ``(H, W)`` for gray, ``(H, W, 2)`` gray + alpha, ``(H, W, 3)`` RGB and
    ``(H, W, 4)`` RGBA.  uint8, or uint16 for 16-bit sources; 32-bit integer
    and float images become uint16 too, scaled into its range rather than
    clipped (a float height map in metres keeps its shape).  Palette,
    CMYK and other modes are expanded to RGB(A).
    """
    for magic, kind in _UNSUPPORTED:
        if data.startswith(magic):
            raise ImageError(f"{kind} images are not supported")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return _native(image)
    except UnidentifiedImageError as exc:
        raise ImageError("its format is not one that can be read") from exc
    except Image.DecompressionBombError as exc:
        raise ImageError("it is too large to load") from exc
    except Exception as exc:  # truncated or corrupt data raises a dozen types
        raise ImageError(f"it could not be decoded ({exc})") from exc


def _native(image: Image.Image) -> np.ndarray:
    mode = image.mode
    if mode in _SIXTEEN_BIT_MODES:
        return np.asarray(image).astype(np.uint16)
    if mode in ("I", "F"):
        return _widen(np.asarray(image))
    if mode not in _EIGHT_BIT_MODES:
        if mode == "1":
            target = "L"
        elif mode == "La":
            target = "LA"
        else:
            target = "RGBA" if image.has_transparency_data else "RGB"
        image = image.convert(target)
    return np.asarray(image)


def _widen(values: np.ndarray) -> np.ndarray:
    """A 32-bit integer or float image as 8 or 16 bits, scaled, never clipped."""
    if values.dtype.kind in "iu":
        low, high = int(values.min()), int(values.max())
        if low >= 0 and high <= 255:
            return values.astype(np.uint8)  # an 8-bit image Pillow opened as "I"
        if low >= 0 and high <= 65535:
            return values.astype(np.uint16)
    else:
        values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        low, high = float(values.min()), float(values.max())
        if low >= 0.0 and high <= 1.0:
            return np.rint(values * 65535.0).astype(np.uint16)
    span = float(high - low) or 1.0
    return np.rint((values.astype(np.float64) - low) * (65535.0 / span)).astype(np.uint16)


class _PixelCache:
    """Decoded images, shared by the maps of one read and kept within a byte budget."""

    def __init__(self, budget: int = PIXEL_CACHE_BYTES) -> None:
        self._budget = budget
        self._images: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()

    def pixels(self, digest: str, data: bytes) -> np.ndarray:
        with self._lock:
            image = self._images.get(digest)
            if image is not None:
                self._images.move_to_end(digest)
                return image
        image = decode_image(data)  # outside the lock: slow, and Pillow releases the GIL
        image.setflags(write=False)
        with self._lock:
            self._images[digest] = image
            total = sum(kept.nbytes for kept in self._images.values())
            while total > self._budget and len(self._images) > 1:
                _, dropped = self._images.popitem(last=False)
                total -= dropped.nbytes
        return image

    def drop(self, digest: str) -> None:
        with self._lock:
            self._images.pop(digest, None)


def _channel(pixels: np.ndarray, channel: str) -> np.ndarray:
    """One quantity out of an image: ``(H, W)`` for r/g/b/a, ``(H, W, 3)`` for rgb.

    A gray image has its one value in r, g and b alike.  Views where possible.
    """
    image = pixels if pixels.ndim == 3 else pixels[..., None]
    count = image.shape[2]
    colour = count >= 3
    if channel == "rgb":
        return image[..., :3] if colour else np.repeat(image[..., :1], 3, axis=2)
    if channel == "a":
        return image[..., count - 1]
    return image[..., "rgb".index(channel) if colour else 0]


def _to_u8(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.uint8:
        return values
    return ((values.astype(np.uint32) * 255 + 32767) // 65535).astype(np.uint8)


def _to_u16(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.uint16:
        return values
    return values.astype(np.uint16) * np.uint16(257)


def _fit(size: Tuple[int, int], max_px: Optional[int]) -> Tuple[int, int]:
    """``(width, height)`` scaled down so the longer side is at most ``max_px``."""
    width, height = size
    longest = max(width, height)
    if not max_px or longest <= max_px:
        return size
    scale = max_px / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def _resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """``image`` resampled to ``size`` (width, height) with Lanczos, one channel at a time.

    Channels are filtered independently because Pillow premultiplies an RGBA
    image by its alpha while resizing it, which corrupts every map whose
    alpha holds data (smoothness, glossiness) rather than coverage.
    """
    height, width = image.shape[:2]
    if (width, height) == tuple(size):
        return image
    if image.ndim == 2:
        return _resize_channel(image, size)
    return np.stack([_resize_channel(image[..., c], size) for c in range(image.shape[2])], 2)


def _resize_channel(channel: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    lanczos = Image.Resampling.LANCZOS
    if channel.dtype == np.uint8:
        return np.asarray(Image.fromarray(np.ascontiguousarray(channel)).resize(size, lanczos))
    resized = np.asarray(Image.fromarray(channel.astype(np.float32)).resize(size, lanczos))
    return np.clip(np.rint(resized), 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
#  What a read returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UVTransform:
    """KHR_texture_transform: ``uv' = T(offset) . R(rotation) . S(scale) . uv``.

    In glTF's texture space, where v runs down from the image's top edge.
    """

    offset: Tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    scale: Tuple[float, float] = (1.0, 1.0)

    @property
    def identity(self) -> bool:
        return self.offset == (0.0, 0.0) and self.rotation == 0.0 and self.scale == (1.0, 1.0)

    def matrix(self) -> np.ndarray:
        """The ``(3, 3)`` matrix of the transform, as the extension defines it."""
        cos, sin = math.cos(self.rotation), math.sin(self.rotation)
        offset_u, offset_v = self.offset
        translation = np.array([[1.0, 0.0, offset_u], [0.0, 1.0, offset_v], [0.0, 0.0, 1.0]])
        rotation = np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]])
        scale = np.diag([self.scale[0], self.scale[1], 1.0])
        return translation @ rotation @ scale

    def apply(self, uv: np.ndarray, *, v_up: bool = True) -> np.ndarray:
        """Transform ``(N, 2)`` UVs.

        ``v_up`` says the coordinates have v pointing up (0 at the image's
        bottom), as trimesh, OBJ and FBX hold them; they are flipped into
        glTF's convention for the transform and back.
        """
        m = self.matrix()
        u = uv[:, 0].astype(np.float64)
        v = 1.0 - uv[:, 1] if v_up else uv[:, 1].astype(np.float64)
        out_u = m[0, 0] * u + m[0, 1] * v + m[0, 2]
        out_v = m[1, 0] * u + m[1, 1] * v + m[1, 2]
        if v_up:
            out_v = 1.0 - out_v
        return np.stack([out_u, out_v], 1).astype(uv.dtype, copy=False)


@dataclass(eq=False)
class TexRef:
    """One image of one material, and what it is for.

    Every slot of the material the image is bound to is listed together:
    DiffuseColor + TransparentColor is one colour map with alpha, glTF
    occlusion + metallicRoughness on one image is one ORM map.
    """

    #: File or image name, as shown to the user and read by the classifier.
    name: str
    #: The image exactly as the model file holds it.
    data: bytes = field(repr=False)
    slots: List[Slot]
    #: What it is for (the classifier's answer, or the user's correction).
    guess: RoleGuess
    #: Identifies the map within the read: ``"<material index>/<name>"``, the
    #: key :func:`read_materials` takes corrections under.
    key: str
    #: The UV set the map is authored against (glTF texCoord).
    texcoord: int = 0
    wrap_s: int = WRAP_REPEAT
    wrap_t: int = WRAP_REPEAT
    #: KHR_texture_transform, or MTL -o/-s, when the file sets one.
    transform: Optional[UVTransform] = None
    _cache: _PixelCache = field(default_factory=_PixelCache, repr=False)

    @property
    def role(self) -> str:
        return self.guess.role

    @functools.cached_property
    def digest(self) -> str:
        return hashlib.sha1(self.data).hexdigest()

    def pixels(self) -> np.ndarray:
        """The decoded image, read-only (see :func:`decode_image`).  Raises ImageError."""
        return self._cache.pixels(self.digest, self.data)

    def forget(self) -> None:
        """Drop the decoded image; the next :meth:`pixels` decodes it again."""
        self._cache.drop(self.digest)


@dataclass(frozen=True)
class SpecGloss:
    """A specular/glossiness material's factors.

    :meth:`MaterialSpec.canonical` converts such a material to metal/rough
    texel by texel, with these factors baked into the result.
    """

    diffuse: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    specular: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    glossiness: float = 1.0


@dataclass(frozen=True)
class CanonicalMaps:
    """A material's maps in the one layout the renderer and the baker read."""

    #: ``(H, W, 4)`` uint8, sRGB colour, linear alpha = opacity.
    basecolor: Optional[np.ndarray] = None
    #: ``(H, W, 3)`` uint8; tangent space has green up (OpenGL).
    normal: Optional[np.ndarray] = None
    #: ``"tangent"`` or ``"object"`` when there is a normal map.
    normal_space: Optional[str] = None
    #: ``(H, W, 3)`` uint8, linear: R AO, G roughness, B metallic, 255 where
    #: there is no source (see ``orm_sources``).
    orm: Optional[np.ndarray] = None
    #: Which of ao / roughness / metallic came from a texture.
    orm_sources: FrozenSet[str] = frozenset()
    #: ``(H, W, 3)`` uint8, sRGB.
    emissive: Optional[np.ndarray] = None
    #: ``(H, W)`` uint16; 8-bit sources are scaled by 257.
    height: Optional[np.ndarray] = None
    #: ``(file name, uint8 image)`` of every map with no place above, with the
    #: channel count it was stored with.
    others: Tuple[Tuple[str, np.ndarray], ...] = ()


@dataclass(eq=False)
class MaterialSpec:
    """One material: its maps, its factors, and how it is drawn.

    Factors are glTF's, linear, and multiply the canonical maps; ``emissive``
    already includes KHR_materials_emissive_strength.
    """

    name: str
    refs: List[TexRef] = field(default_factory=list)
    base_color_factor: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 1.0
    roughness: float = 1.0
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    occlusion_strength: float = 1.0
    normal_scale: float = 1.0
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    #: The texture transform the geometry's UVs should get (the base colour
    #: map's when maps disagree), or None.
    uv_transform: Optional[UVTransform] = None
    #: Set for a specular/glossiness material that has maps; canonical()
    #: converts it, and the factors above are then 1.
    spec_gloss: Optional[SpecGloss] = None
    _canonical: Dict[Optional[int], CanonicalMaps] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def sources(self) -> Dict[str, TexRef]:
        """The map each quantity comes from, for the quantities any map supplies.

        Decided from the guesses alone, without decoding anything.  Specular
        counts only for a specular/glossiness material (it is converted); a
        plain specular map is not part of a metal/rough render.
        """
        converted = self.spec_gloss is not None
        best: Dict[str, Tuple[Tuple[bool, float, int], TexRef]] = {}
        for order, ref in enumerate(self.refs):
            for quantity in ref.guess.channels:
                if quantity not in _DEDICATED:
                    continue  # "invert"
                if quantity == "specular" and not converted:
                    continue  # a plain specular map is not part of a metal/rough render
                if quantity == "metallic" and converted:
                    continue  # the conversion works metallic out
                rank = (ref.role in _DEDICATED[quantity], ref.guess.confidence, -order)
                if quantity not in best or rank > best[quantity][0]:
                    best[quantity] = (rank, ref)
        return {quantity: best[quantity][1] for quantity in QUANTITIES if quantity in best}

    @property
    def orm_sources(self) -> FrozenSet[str]:
        """Which of ao / roughness / metallic come from a texture."""
        found = {quantity for quantity in ORM_CHANNELS if quantity in self.sources}
        if self.spec_gloss is not None:
            found |= {"roughness", "metallic"}
        return frozenset(found)

    @property
    def normal_space(self) -> Optional[str]:
        normal = self.sources.get("normal")
        return normal.guess.normal_space if normal is not None else None

    @property
    def others(self) -> List[TexRef]:
        """Maps that feed no canonical map: shown as they are, never rendered lit."""
        used = {id(ref) for ref in self.sources.values()}
        return [ref for ref in self.refs if id(ref) not in used]

    def canonical(self, max_px: Optional[int] = None) -> CanonicalMaps:
        """The maps in the canonical layout, at most ``max_px`` on their longer side.

        Decoded from the original bytes on first use and kept (the arrays are
        read-only); sources of different sizes that make one map are
        resampled (Lanczos) to the largest of them.  Raises ImageError when a
        map cannot be decoded, which :func:`read_materials` has already ruled
        out for every map it returns.
        """
        with self._lock:
            maps = self._canonical.get(max_px)
            if maps is None:
                maps = _canonical(self, max_px)
                self._canonical[max_px] = maps
            return maps

    def clear_cache(self) -> None:
        """Forget the canonical maps and the decoded images behind them."""
        with self._lock:
            self._canonical.clear()
        for ref in self.refs:
            ref.forget()


class PrimitiveBinding(NamedTuple):
    """The material of one glTF mesh primitive."""

    mesh: int
    primitive: int
    #: Index into the materials, or None for glTF's default material.
    material: Optional[int]
    #: glTF primitive mode (4 = triangles).
    mode: int


@dataclass(frozen=True)
class Bindings:
    """How a model's geometry refers to its materials, and what else the file says.

    - glTF: ``primitives`` lists every mesh primitive in file order.
    - OBJ: ``names`` maps each ``usemtl`` name to its material.
    - FBX and Collada: the materials are index-aligned with Assimp's
      ``scene.materials``, so a mesh's ``material_index`` is the index.
    ``names`` is filled for every format (the first material of a name wins).
    """

    primitives: Tuple[PrimitiveBinding, ...] = ()
    names: Mapping[str, int] = field(default_factory=dict)
    #: The program that wrote the file, as the file names it.
    exporter: str = ""
    #: FBX axis system and unit (binary FBX only).
    fbx_settings: Optional[fbx_media.GlobalSettings] = None


class MaterialRead(NamedTuple):
    """What :func:`read_materials` found."""

    materials: List[MaterialSpec]
    bindings: Bindings
    #: Artist-readable lines about maps that were missing, unreadable or
    #: could only be approximated.  None of them stops the import.
    warnings: List[str]


# ---------------------------------------------------------------------------
#  Canonical maps
# ---------------------------------------------------------------------------


def _canvas(refs: Iterable[TexRef], max_px: Optional[int]) -> Tuple[int, int]:
    """``(width, height)`` of the largest of ``refs``, fitted to ``max_px``."""
    shapes = [ref.pixels().shape for ref in refs]
    height, width = max(shapes, key=lambda shape: shape[0] * shape[1])[:2]
    return _fit((width, height), max_px)


def _plane(ref: TexRef, quantity: str, size: Tuple[int, int]) -> np.ndarray:
    """``quantity`` out of ``ref``, at ``size``, in the source's bit depth."""
    return _resize(_channel(ref.pixels(), ref.guess.channels[quantity]), size)


def _roughness_plane(ref: TexRef, size: Tuple[int, int]) -> np.ndarray:
    """Roughness as uint8, glossiness and smoothness inverted into it."""
    plane = _to_u8(_plane(ref, "roughness", size))
    return 255 - plane if ref.guess.channels.get("invert") == "1" else plane


def _canonical(spec: MaterialSpec, max_px: Optional[int]) -> CanonicalMaps:
    sources = spec.sources
    orm_planes: Dict[str, np.ndarray] = {}
    if spec.spec_gloss is not None:
        basecolor, orm_planes["roughness"], orm_planes["metallic"] = _converted(
            spec, sources, max_px
        )
    else:
        basecolor = _basecolor(spec, sources, max_px)

    orm = None
    feeding = [sources[q] for q in ORM_CHANNELS if q in sources]
    if feeding or orm_planes:
        sizes = [(p.shape[1], p.shape[0]) for p in orm_planes.values()]
        if feeding:
            sizes.append(_canvas(feeding, max_px))
        width, height = max(sizes, key=lambda s: s[0] * s[1])
        orm = np.full((height, width, 3), 255, np.uint8)
        for index, quantity in enumerate(ORM_CHANNELS):
            if quantity in orm_planes:
                orm[..., index] = _resize(orm_planes[quantity], (width, height))
            elif quantity == "roughness" and quantity in sources:
                orm[..., index] = _roughness_plane(sources[quantity], (width, height))
            elif quantity in sources:
                orm[..., index] = _to_u8(_plane(sources[quantity], quantity, (width, height)))

    normal = _normal(sources["normal"], max_px) if "normal" in sources else None
    emissive = None
    if "emissive" in sources:
        size = _canvas([sources["emissive"]], max_px)
        plane = _to_u8(_plane(sources["emissive"], "emissive", size))
        emissive = np.empty((size[1], size[0], 3), np.uint8)
        emissive[...] = plane if plane.ndim == 3 else plane[..., None]
    height_map = None
    if "height" in sources:
        size = _canvas([sources["height"]], max_px)
        height_map = np.array(_to_u16(_plane(sources["height"], "height", size)))

    return CanonicalMaps(
        basecolor=_frozen(basecolor),
        normal=_frozen(normal),
        normal_space=spec.normal_space,
        orm=_frozen(orm),
        orm_sources=spec.orm_sources,
        emissive=_frozen(emissive),
        height=_frozen(height_map),
        others=tuple(
            (ref.name, _frozen(np.array(_to_u8(_resize(ref.pixels(), _canvas([ref], max_px))))))
            for ref in spec.others
        ),
    )


_Image = TypeVar("_Image", np.ndarray, Optional[np.ndarray])


def _frozen(image: _Image) -> _Image:
    """``image`` made read-only: canonical maps are cached and shared between callers."""
    if image is not None:
        image.setflags(write=False)
    return image


def _basecolor(
    spec: MaterialSpec, sources: Mapping[str, TexRef], max_px: Optional[int]
) -> Optional[np.ndarray]:
    opacity = "opacity" in sources and spec.alpha_mode != "OPAQUE"
    if "basecolor" not in sources and not opacity:
        return None
    width, height = _canvas(
        [sources[q] for q in ("basecolor", "opacity") if q in sources], max_px
    )
    out = np.full((height, width, 4), 255, np.uint8)
    if "basecolor" in sources:
        out[..., :3] = _to_u8(_plane(sources["basecolor"], "basecolor", (width, height)))
    if opacity:
        out[..., 3] = _to_u8(_plane(sources["opacity"], "opacity", (width, height)))
    return out


def _normal(ref: TexRef, max_px: Optional[int]) -> np.ndarray:
    width, height = _canvas([ref], max_px)
    out = np.empty((height, width, 3), np.uint8)
    out[...] = _to_u8(_plane(ref, "normal", (width, height)))
    if texstats.is_two_channel_normal(texstats.features(ref.pixels())):
        # BC5-style: x and y only, blue constant.  z follows from them.
        xy = out[..., :2].astype(np.float32) / 127.5 - 1.0
        z = np.sqrt(np.clip(1.0 - (xy * xy).sum(axis=2), 0.0, 1.0))
        out[..., 2] = np.rint(z * 127.5 + 127.5).astype(np.uint8)
    if ref.guess.y_convention == "directx":
        np.subtract(255, out[..., 1], out=out[..., 1])
    return out


def _converted(
    spec: MaterialSpec, sources: Mapping[str, TexRef], max_px: Optional[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(basecolor RGBA, roughness, metallic)`` of a specular/glossiness material.

    The per-texel conversion is KHR's, as trimesh implements it
    (``trimesh.visual.gloss.specular_to_pbr``), run a band of rows at a time.
    """
    from trimesh.visual.gloss import specular_to_pbr

    factors = spec.spec_gloss or SpecGloss()
    inputs = ("basecolor", "opacity", "specular", "roughness")
    width, height = _canvas([sources[q] for q in inputs if q in sources], max_px)

    size = (width, height)
    diffuse = None
    if "basecolor" in sources or "opacity" in sources:
        diffuse = np.full((height, width, 4), 255, np.uint8)
        if "basecolor" in sources:
            diffuse[..., :3] = _to_u8(_plane(sources["basecolor"], "basecolor", size))
        if "opacity" in sources:
            diffuse[..., 3] = _to_u8(_plane(sources["opacity"], "opacity", size))
    specular_gloss = None
    if "specular" in sources or "roughness" in sources:
        specular_gloss = np.full((height, width, 4), 255, np.uint8)
        if "specular" in sources:
            specular_gloss[..., :3] = _to_u8(_plane(sources["specular"], "specular", size))
        if "roughness" in sources:
            specular_gloss[..., 3] = 255 - _roughness_plane(sources["roughness"], size)

    basecolor = np.empty((height, width, 4), np.uint8)
    roughness = np.empty((height, width), np.uint8)
    metallic = np.empty((height, width), np.uint8)
    for top in range(0, height, _BAND_ROWS):
        rows = slice(top, top + _BAND_ROWS)
        result = specular_to_pbr(
            specularFactor=list(factors.specular),
            glossinessFactor=factors.glossiness,
            specularGlossinessTexture=(
                None if specular_gloss is None else Image.fromarray(specular_gloss[rows])
            ),
            diffuseTexture=None if diffuse is None else Image.fromarray(diffuse[rows]),
            diffuseFactor=list(factors.diffuse),
        )
        colour = np.asarray(result["baseColorTexture"])
        basecolor[rows, :, :3] = colour[..., :3]
        basecolor[rows, :, 3] = colour[..., 3] if colour.shape[2] == 4 else 255
        metal_rough = np.asarray(result["metallicRoughnessTexture"])
        roughness[rows] = metal_rough[..., 1]
        metallic[rows] = metal_rough[..., 2]
    if spec.alpha_mode == "OPAQUE":
        basecolor[..., 3] = 255
    return basecolor, roughness, metallic


# ---------------------------------------------------------------------------
#  Readers: what each format binds, before anything is decoded
# ---------------------------------------------------------------------------


@dataclass
class _Ref:
    """One texture binding as a reader found it."""

    slot: Slot
    name: str
    data: Optional[bytes]
    #: Why ``data`` is None, for the warning.
    missing: str = "was not found next to the model"
    texcoord: int = 0
    wrap: Tuple[int, int] = (WRAP_REPEAT, WRAP_REPEAT)
    transform: Optional[UVTransform] = None


@dataclass
class _Draft:
    """A material as a reader found it: its bindings and its own numbers."""

    name: str
    refs: List[_Ref] = field(default_factory=list)
    #: glTF: the factors multiply the maps as written.  Otherwise a map
    #: replaces its scalar, and alpha mode and sidedness are inferred.
    exact: bool = False
    base: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    alpha: float = 1.0
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    occlusion_strength: float = 1.0
    normal_scale: float = 1.0
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    spec_gloss: Optional[SpecGloss] = None


def find_file(
    folder: Path, named: str, *also: str, root: Optional[Path] = None
) -> Optional[Path]:
    """The file a model names, looked for only inside the model's own folder.

    ``folder`` is where the file that names it lives, ``root`` the model's
    folder (by default the same: an MTL beside its OBJ).  Tried in order: the
    name as written when it is relative, then its base name in ``folder``, in
    its ``textures/`` and in the sub-folders ``also`` names, then the same in
    ``root``, and last anywhere below ``root`` (an unpacked asset pack keeps
    its maps in folders of its own).  Nothing outside ``root`` is read: a path
    out of a file is data from wherever the file came from, so
    ``../../secret.png`` and ``C:/Users/someone/map.png`` are only ever looked
    for by base name.
    """
    text = named.strip().strip('"').replace("\\", "/")
    if not text:
        return None
    base = PurePosixPath(text).name
    candidates = []
    if not (PureWindowsPath(named.strip()).drive or text.startswith("/")):
        candidates.append(folder / text)
    places = [folder] if root is None or root == folder else [folder, root]
    subs = ("", "textures", "Textures", *also)
    candidates += [place / sub / base for place in places for sub in subs]
    try:
        inside = (root or folder).resolve()
    except OSError:
        return None
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.is_relative_to(inside) and resolved.is_file():
                return resolved
        except (OSError, ValueError):
            continue
    return _search(inside, base)


#: Entries :func:`_search` looks at before it gives up, so that a model opened
#: from the top of a large project does not walk all of it for every map.
_SEARCH_LIMIT = 5000


def _search(root: Path, base: str) -> Optional[Path]:
    """A file named ``base`` anywhere below ``root`` (resolved), case-insensitively."""
    wanted, seen = base.lower(), 0
    for folder, folders, files in os.walk(root):
        folders.sort()  # the same answer every time
        for name in files:
            if name.lower() == wanted:
                found = (Path(folder) / name).resolve()
                return found if found.is_relative_to(root) else None
        seen += len(files) + len(folders)
        if seen > _SEARCH_LIMIT:
            break
    return None


def _read_file(path: Path) -> Optional[bytes]:
    try:
        if path.stat().st_size > MAX_IMAGE_BYTES:
            return None
        return path.read_bytes()
    except OSError:
        return None


def _base_name(named: str) -> str:
    return PurePosixPath(named.replace("\\", "/")).name or named


def _floats(values: Any, count: int, default: Sequence[float]) -> Tuple[float, ...]:
    """``count`` floats from a JSON list or MTL arguments, the rest from ``default``.

    Reading stops at the first value that is not a number.
    """
    parsed: List[float] = []
    for value in list(values or ())[:count]:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            break
    return tuple(parsed + list(default[len(parsed) : count]))


# --- glTF -------------------------------------------------------------------

_GLB_MAGIC = b"glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

#: Texture extensions whose image Pillow can decode, in order of preference
#: over the plain ``source`` (which is then the fallback copy).
_IMAGE_EXTENSIONS = ("EXT_texture_webp", "EXT_texture_avif", "MSFT_texture_dds")


def _parse_gltf(path: Path) -> Tuple[Dict[str, Any], Optional[bytes]]:
    """The JSON document and, for a GLB, its binary chunk."""
    raw = path.read_bytes()
    if raw[:4] != _GLB_MAGIC:
        return json.loads(raw.decode("utf-8-sig")), None
    (length,) = struct.unpack_from("<I", raw, 8)
    document: Optional[Dict[str, Any]] = None
    binary: Optional[bytes] = None
    at = 12
    while at + 8 <= min(length, len(raw)):
        size, kind = struct.unpack_from("<II", raw, at)
        chunk = raw[at + 8 : at + 8 + size]
        if kind == _CHUNK_JSON and document is None:
            document = json.loads(chunk.decode("utf-8"))
        elif kind == _CHUNK_BIN and binary is None:
            binary = chunk
        at += 8 + size
    if document is None:
        raise ValueError("the GLB has no JSON chunk")
    return document, binary


def _texture_infos(material: Mapping[str, Any]) -> Iterator[Tuple[str, Mapping[str, Any]]]:
    """``(slot key, textureInfo)`` for every texture a glTF material binds.

    Extension slots are prefixed with the extension's name, e.g.
    ``"KHR_materials_specular.specularTexture"``.
    """

    def walk(node: Mapping[str, Any], prefix: str) -> Iterator[Tuple[str, Mapping[str, Any]]]:
        for key, value in node.items():
            if not isinstance(value, dict):
                continue
            if key.endswith("Texture") and "index" in value:
                yield prefix + key, value
            elif key == "extensions":
                for name, extension in value.items():
                    if isinstance(extension, dict):
                        yield from walk(extension, f"{name}.")
            elif key != "extras":
                yield from walk(value, prefix)  # pbrMetallicRoughness

    return walk(material, "")


class _Gltf:
    """Lazy access to a glTF's buffers and images."""

    def __init__(self, path: Path) -> None:
        self.folder = path.parent
        self.doc, self._binary = _parse_gltf(path)
        self._buffers: Dict[int, bytes] = {}

    def _uri(self, uri: str) -> Tuple[Optional[bytes], str]:
        """``(bytes, why not)`` of a data URI or a file beside the model."""
        if uri.startswith("data:"):
            header, _, payload = uri.partition(",")
            try:
                if header.endswith(";base64"):
                    return base64.b64decode(payload), ""
            except binascii.Error:
                pass
            return None, "is a data URI that cannot be decoded"
        found = find_file(self.folder, unquote(uri))
        data = _read_file(found) if found is not None else None
        return data, "" if data is not None else "was not found next to the model"

    def buffer(self, index: int) -> bytes:
        if index not in self._buffers:
            entry = self.doc["buffers"][index]
            if "uri" in entry:
                data, why = self._uri(entry["uri"])
                if data is None:
                    raise ValueError(f"buffer {entry['uri']} {why}")
            elif self._binary is not None:
                data = self._binary
            else:
                raise ValueError(f"buffer {index} has no data")
            self._buffers[index] = data
        return self._buffers[index]

    def image(self, index: int) -> Tuple[str, Optional[bytes], str]:
        """``(name, bytes, why not)`` of image ``index``."""
        entry = self.doc["images"][index]
        uri = str(entry.get("uri", ""))
        named = "" if uri.startswith("data:") else _base_name(unquote(uri))
        name = str(entry.get("name") or named or f"image {index}")
        if "bufferView" in entry:
            view = self.doc["bufferViews"][entry["bufferView"]]
            start = view.get("byteOffset", 0)
            return name, self.buffer(view["buffer"])[start : start + view["byteLength"]], ""
        if uri:
            data, why = self._uri(uri)
            return name, data, why
        return name, None, "has no data"

    def source(self, texture: Mapping[str, Any]) -> Optional[int]:
        extensions = texture.get("extensions", {})
        for key in _IMAGE_EXTENSIONS:
            if "source" in extensions.get(key, {}):
                return int(extensions[key]["source"])
        source = texture.get("source")
        return int(source) if source is not None else None


def _read_gltf(path: Path) -> Tuple[List[_Draft], Bindings]:
    gltf = _Gltf(path)
    doc = gltf.doc
    textures = doc.get("textures", [])
    samplers = doc.get("samplers", [])

    drafts: List[_Draft] = []
    for index, material in enumerate(doc.get("materials", [])):
        pbr = material.get("pbrMetallicRoughness", {})
        extensions = material.get("extensions", {})
        strength = float(
            extensions.get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1.0)
        )
        alpha_mode = str(material.get("alphaMode", "OPAQUE")).upper()
        draft = _Draft(
            name=str(material.get("name") or f"material {index}"),
            exact=True,
            base=_floats(pbr.get("baseColorFactor"), 3, (1.0, 1.0, 1.0)),
            alpha=_floats(pbr.get("baseColorFactor"), 4, (1.0,) * 4)[3],
            metallic=float(pbr.get("metallicFactor", 1.0)),
            roughness=float(pbr.get("roughnessFactor", 1.0)),
            emissive=tuple(
                v * strength for v in _floats(material.get("emissiveFactor"), 3, (0.0,) * 3)
            ),
            occlusion_strength=float(material.get("occlusionTexture", {}).get("strength", 1.0)),
            normal_scale=float(material.get("normalTexture", {}).get("scale", 1.0)),
            alpha_mode=alpha_mode if alpha_mode in ALPHA_MODES else "OPAQUE",
            alpha_cutoff=float(material.get("alphaCutoff", 0.5)),
            double_sided=bool(material.get("doubleSided", False)),
        )
        gloss = extensions.get("KHR_materials_pbrSpecularGlossiness")
        if isinstance(gloss, dict):
            draft.spec_gloss = SpecGloss(
                diffuse=_floats(gloss.get("diffuseFactor"), 4, (1.0,) * 4),
                specular=_floats(gloss.get("specularFactor"), 3, (1.0,) * 3),
                glossiness=float(gloss.get("glossinessFactor", 1.0)),
            )

        for slot_key, info in _texture_infos(material):
            texture_index = int(info["index"])
            if not 0 <= texture_index < len(textures):
                continue
            texture = textures[texture_index]
            transform_ext = info.get("extensions", {}).get("KHR_texture_transform", {})
            texcoord = int(transform_ext.get("texCoord", info.get("texCoord", 0)))
            options = {k: info[k] for k in ("texCoord", "scale", "strength") if k in info}
            slot = Slot("gltf", slot_key, options)
            source = gltf.source(texture)
            if source is None:
                basis = "KHR_texture_basisu" in texture.get("extensions", {})
                why = "is KTX2 / Basis compressed, which is not supported"
                if not basis:
                    why = "has no image"
                draft.refs.append(_Ref(slot, f"texture {texture_index}", None, why))
                continue
            name, data, why = gltf.image(source)
            sampler = samplers[texture["sampler"]] if "sampler" in texture else {}
            wrap = tuple(int(sampler.get(key, WRAP_REPEAT)) for key in ("wrapS", "wrapT"))
            transform = UVTransform(
                offset=_floats(transform_ext.get("offset"), 2, (0.0, 0.0)),
                rotation=float(transform_ext.get("rotation", 0.0)),
                scale=_floats(transform_ext.get("scale"), 2, (1.0, 1.0)),
            )
            draft.refs.append(
                _Ref(
                    slot,
                    name,
                    data,
                    missing=why,
                    texcoord=texcoord,
                    wrap=wrap,
                    transform=None if transform.identity else transform,
                )
            )
        drafts.append(draft)

    primitives = tuple(
        PrimitiveBinding(
            mesh,
            number,
            int(primitive["material"]) if "material" in primitive else None,
            int(primitive.get("mode", 4)),
        )
        for mesh, entry in enumerate(doc.get("meshes", []))
        for number, primitive in enumerate(entry.get("primitives", []))
    )
    generator = str(doc.get("asset", {}).get("generator") or "")
    return drafts, Bindings(primitives=primitives, names=_names(drafts), exporter=generator)


def _names(drafts: Sequence[_Draft]) -> Dict[str, int]:
    names: Dict[str, int] = {}
    for index, draft in enumerate(drafts):
        names.setdefault(draft.name, index)
    return names


# --- MTL --------------------------------------------------------------------

#: MTL map options and how many arguments each takes (-o/-s/-t take 1 to 3).
_MTL_OPTIONS = {
    "-blendu": 1,
    "-blendv": 1,
    "-bm": 1,
    "-boost": 1,
    "-cc": 1,
    "-clamp": 1,
    "-imfchan": 1,
    "-mm": 2,
    "-texres": 1,
    "-type": 1,
    "-o": 3,
    "-s": 3,
    "-t": 3,
}
_MTL_VECTOR_OPTIONS = frozenset({"-o", "-s", "-t"})

#: -imfchan letters and the channel they name; "l" (luminance) and "z" (depth)
#: are what the classifier reads anyway.
_IMFCHAN = {"r": "r", "g": "g", "b": "b", "m": "a"}

_MTLLIB = re.compile(rb"^[ \t]*mtllib[ \t]+([^\r\n]+)", re.MULTILINE)
_COMMENT = re.compile(rb"^[ \t]*#+[ \t]*(\S[^\r\n]*)", re.MULTILINE)


def parse_map_line(rest: str) -> Tuple[str, Dict[str, Tuple[str, ...]]]:
    """``(file name, options)`` of the arguments of an MTL map statement.

    The file name is whatever follows the options, spaces included.
    """
    tokens = rest.split()
    options: Dict[str, Tuple[str, ...]] = {}
    i = 0
    while i < len(tokens) and tokens[i].lower() in _MTL_OPTIONS:
        option = tokens[i].lower()
        i += 1
        values: List[str] = []
        while len(values) < _MTL_OPTIONS[option] and i < len(tokens):
            if option in _MTL_VECTOR_OPTIONS and values and not _is_number(tokens[i]):
                break
            values.append(tokens[i])
            i += 1
        options[option] = tuple(values)
    return " ".join(tokens[i:]), options


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _mtl_transform(options: Mapping[str, Tuple[str, ...]]) -> Optional[UVTransform]:
    """MTL ``-o``/``-s`` as a UVTransform.

    MTL works with v up: ``uv' = s * uv + o``.  In glTF's v-down space that is
    the same scale with offset ``(o_u, 1 - s_v - o_v)``.
    """
    if "-o" not in options and "-s" not in options:
        return None
    offset_u, offset_v = _floats(options.get("-o"), 2, (0.0, 0.0))
    scale_u, scale_v = _floats(options.get("-s"), 2, (1.0, 1.0))
    transform = UVTransform((offset_u, 1.0 - scale_v - offset_v), 0.0, (scale_u, scale_v))
    return None if transform.identity else transform


def _phong_roughness(exponent: float) -> float:
    """Roughness of a Phong specular exponent ``n``: ``(2 / (n + 2)) ** 0.25``.

    Blinn-Phong ``n`` matches a microfacet ``alpha = sqrt(2 / (n + 2))``
    (Walter et al. 2007), and glTF's roughness is ``sqrt(alpha)``.
    """
    return (2.0 / (max(exponent, 0.0) + 2.0)) ** 0.25


def _blender_roughness(exponent: float, top: float) -> float:
    """What Blender reads back from the exponent its exporters write.

    ``top * (1 - roughness) ** 2``, with ``top`` 1000 in an MTL (``Ns``) and
    100 in an FBX (``Shininess``).
    """
    return 1.0 - math.sqrt(min(max(exponent, 0.0), top) / top)


def _read_mtl(
    path: Path, header: str, warn: Callable[[str], None], root: Optional[Path] = None
) -> Tuple[List[_Draft], str]:
    """The materials of one MTL file, and the program that wrote it.

    That is its first comment ("# Blender 3.6.1 MTL File", "# Created by
    Polycam"), else ``header``, the OBJ's.  Maps are looked for inside
    ``root``, the model's folder (the MTL's own by default).
    """
    text = path.read_bytes().decode("utf-8", "replace")
    exporter = ""
    found: List[Tuple[_Draft, Dict[str, Tuple[float, ...]]]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            exporter = exporter or stripped.lstrip("#").strip()
            continue
        parts = stripped.split(None, 1)
        if not parts:
            continue
        keyword, rest = parts[0], parts[1] if len(parts) > 1 else ""
        key = keyword.lower()
        if key == "newmtl":
            found.append((_Draft(name=rest.strip() or f"material {len(found)}"), {}))
        elif not found:
            continue
        elif key in texroles.MTL_SLOTS:
            named, options = parse_map_line(rest)
            if not named:
                continue
            located = find_file(path.parent, named, root=root)
            clamp = (options.get("-clamp") or ("off",))[0].lower() == "on"
            wrap = WRAP_CLAMP if clamp else WRAP_REPEAT
            found[-1][0].refs.append(
                _Ref(
                    Slot("mtl", keyword, options),
                    _base_name(named),
                    _read_file(located) if located is not None else None,
                    wrap=(wrap, wrap),
                    transform=_mtl_transform(options),
                )
            )
        elif key in ("kd", "ke", "ns", "d", "tr", "pr", "pm"):
            try:
                found[-1][1][key] = tuple(float(value) for value in rest.split()[:3])
            except ValueError:
                warn(f"{found[-1][0].name}: ignored an unreadable '{keyword}' value")

    exporter = exporter or header
    blender = "blender" in exporter.lower()
    for draft, scalars in found:
        # "Kd 0.5" is a gray: a single value stands for all three.
        if "kd" in scalars:
            draft.base = _floats(scalars["kd"] * 3, 3, (0.8,) * 3)
        if "ke" in scalars:
            draft.emissive = _floats(scalars["ke"] * 3, 3, (0.0,) * 3)
        alpha = scalars["d"][0] if "d" in scalars else 1.0 - scalars.get("tr", (0.0,))[0]
        if alpha <= 0.0:
            warn(f"{draft.name}: it is fully transparent (d/Tr), which is ignored")
            alpha = 1.0
        draft.alpha = min(alpha, 1.0)
        if "pr" in scalars:
            draft.roughness = scalars["pr"][0]
        elif "ns" in scalars:
            shininess = scalars["ns"][0]
            draft.roughness = (
                _blender_roughness(shininess, 1000.0)
                if blender
                else _phong_roughness(shininess)
            )
        draft.metallic = scalars.get("pm", (0.0,))[0]
    return [draft for draft, _ in found], exporter


def _mtllibs(obj: Path) -> Tuple[List[str], str]:
    """``(mtllib names, first comment)`` of an OBJ, scanned without parsing it."""
    with obj.open("rb") as handle:
        if obj.stat().st_size == 0:
            return [], ""
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
            names = [
                found.group(1).decode("utf-8", "replace").strip()
                for found in _MTLLIB.finditer(view)
            ]
            found = _COMMENT.search(view, 0, 4096)
            header = found.group(1).decode("utf-8", "replace").strip() if found else ""
    return names, header


def _read_obj(path: Path, warn: Callable[[str], None]) -> Tuple[List[_Draft], Bindings]:
    if path.suffix.lower() == ".mtl":
        libraries, header = [path], ""
    else:
        names, header = _mtllibs(path)
        libraries = []
        for named in names:
            # One file with spaces in its name, or several names on one line.
            parts = [named] if find_file(path.parent, named) is not None else named.split()
            for part in parts:
                located = find_file(path.parent, part)
                if located is None:
                    warn(f"{path.name}: its material library {part} was not found")
                elif located not in libraries:
                    libraries.append(located)
    drafts: List[_Draft] = []
    exporter = header
    for library in libraries:
        found, exporter = _read_mtl(library, header, warn, root=path.parent)
        drafts += found
    return drafts, Bindings(names=_names(drafts), exporter=exporter)


# --- FBX and Assimp ---------------------------------------------------------


def _assimp_materials(path: Path) -> List[Mapping[str, Any]]:
    try:
        import assimp_py
    except ImportError as exc:
        raise ValueError("reading FBX needs the assimp-py package") from exc
    return list(assimp_py.import_file(str(path), assimp_py.Process_Triangulate).materials)


def _match_fbx(
    names: Sequence[str], fbx_materials: Sequence[fbx_media.Material]
) -> List[Optional[fbx_media.Material]]:
    """The FBX material behind each Assimp material: by name, else by position.

    Position is only a fallback for a name that did not survive, and never
    for Assimp's own "DefaultMaterial", which has no FBX counterpart.
    """
    unused = list(range(len(fbx_materials)))
    matched: List[Optional[fbx_media.Material]] = [None] * len(names)
    for index, name in enumerate(names):
        for candidate in unused:
            if fbx_materials[candidate].name == name:
                matched[index] = fbx_materials[candidate]
                unused.remove(candidate)
                break
    for index, name in enumerate(names):
        if matched[index] is None and index in unused and name != "DefaultMaterial":
            matched[index] = fbx_materials[index]
            unused.remove(index)
    return matched


def _colour(
    raw: Mapping[str, Any], key: str, default: Tuple[float, float, float]
) -> Tuple[float, ...]:
    value = raw.get(key)
    return _floats(value, 3, default) if isinstance(value, (list, tuple)) else default


def _read_assimp(
    path: Path,
    assimp_materials: Optional[Sequence[Mapping[str, Any]]],
    warn: Callable[[str], None],
) -> Tuple[List[_Draft], Bindings]:
    if assimp_materials is None:
        assimp_materials = _assimp_materials(path)
    media = fbx_media.read(path) if path.suffix.lower() == ".fbx" else None
    exporter = media.exporter if media is not None else ""
    blender = "blender" in exporter.lower()
    names = [str(raw.get("NAME") or f"material {i}") for i, raw in enumerate(assimp_materials)]
    matched = _match_fbx(names, media.materials) if media is not None else []
    fbm = f"{path.stem}.fbm"

    drafts: List[_Draft] = []
    for index, raw in enumerate(assimp_materials):
        opacity = float(raw.get("OPACITY", 1.0))
        shininess = float(raw.get("SHININESS", 20.0))
        draft = _Draft(
            name=names[index],
            base=_colour(raw, "COLOR_DIFFUSE", (0.8, 0.8, 0.8)),
            alpha=opacity if 0.0 < opacity <= 1.0 else 1.0,
            # Blender writes metallic to ReflectionFactor; 3ds Max writes
            # reflection strength there, which is not metalness.
            metallic=float(raw.get("REFLECTIVITY", 0.0)) if blender else 0.0,
            roughness=(
                _blender_roughness(shininess, 100.0) if blender else _phong_roughness(shininess)
            ),
            emissive=_colour(raw, "COLOR_EMISSIVE", (0.0, 0.0, 0.0)),
        )
        if media is not None:
            material = matched[index]
            for binding in material.textures if material is not None else ():
                data = binding.data
                if data is None and binding.file_name:
                    located = find_file(path.parent, binding.file_name, fbm)
                    data = _read_file(located) if located is not None else None
                draft.refs.append(
                    _Ref(Slot("fbx", binding.property), _base_name(binding.file_name), data)
                )
        else:
            for kind, paths in dict(raw.get("TEXTURES") or {}).items():
                for named in paths:
                    if not named:
                        continue
                    if named.startswith("*"):
                        draft.refs.append(
                            _Ref(
                                Slot("assimp", str(kind)),
                                named,
                                None,
                                "is embedded in a way that cannot be read",
                            )
                        )
                        continue
                    located = find_file(path.parent, named, fbm)
                    draft.refs.append(
                        _Ref(
                            Slot("assimp", str(kind)),
                            _base_name(named),
                            _read_file(located) if located is not None else None,
                        )
                    )
        drafts.append(draft)
    return drafts, Bindings(
        names=_names(drafts),
        exporter=exporter,
        fbx_settings=media.settings if media is not None else None,
    )


# ---------------------------------------------------------------------------
#  Classification, corrections and factors
# ---------------------------------------------------------------------------


def _unique_key(prefix: str, name: str, taken: Set[str]) -> str:
    key, count = f"{prefix}/{name}", 1
    while key in taken:
        count += 1
        key = f"{prefix}/{name}#{count}"
    taken.add(key)
    return key


def _imfchan(guess: RoleGuess, slots: Sequence[Slot], pixels: np.ndarray) -> RoleGuess:
    """An MTL ``-imfchan`` names the channel a one-channel map is read from."""
    letters = [(slot.options.get("-imfchan") or ("",))[0][:1].lower() for slot in slots]
    channel = next((_IMFCHAN[x] for x in letters if x in _IMFCHAN), None)
    quantities = [q for q in guess.channels if q != "invert"]
    if channel is None or len(quantities) != 1 or guess.channels[quantities[0]] == "rgb":
        return guess
    if channel == "a" and (pixels.ndim == 2 or pixels.shape[2] not in (2, 4)):
        return guess
    channels = dict(guess.channels, **{quantities[0]: channel})
    return replace(
        guess,
        channels=channels,
        evidence=guess.evidence + (f"MTL -imfchan: read from {channel}",),
    )


def _override(guess: RoleGuess, change: Mapping[str, Any], pixels: np.ndarray) -> RoleGuess:
    """The guess after the user's correction: ``{"role", "flip_green", "invert"}``.

    ``role`` replaces the role and lays its channels out as that role's
    usually are.  ``flip_green`` swaps the green convention of a tangent-space
    normal map, and ``invert`` swaps roughness for glossiness; both are
    relative to what the classifier found.
    """
    role = str(change.get("role") or guess.role)
    if role not in texroles.ROLES:
        raise ValueError(f"unknown role {role!r}")
    notes: List[str] = []
    channels, normal_space = dict(guess.channels), guess.normal_space
    y_convention, y_confident = guess.y_convention, guess.y_confident
    if role != guess.role:
        # The classifier's own layouts, fitted to the channels the image has.
        layout = dict(texroles._LAYOUTS[role])
        channels = texroles._fit_to_image(layout, texstats.features(pixels))
        normal_space = None
        y_convention, y_confident = None, False
        if role == "normal":
            normal_space = "object" if guess.normal_space == "object" else "tangent"
            if normal_space == "tangent":
                call, _ = texstats.curl_convention(pixels)
                y_convention, y_confident = call or "opengl", call is not None
        notes.append(f"set by hand: {ROLE_LABELS[role]}")
    if change.get("flip_green") and normal_space == "tangent":
        y_convention = "directx" if y_convention == "opengl" else "opengl"
        y_confident = True
        notes.append("green flipped by hand")
    if change.get("invert") and "roughness" in channels:
        if channels.pop("invert", None) is None:
            channels["invert"] = "1"
        notes.append("roughness inverted by hand")
    if not notes:
        return guess
    return replace(
        guess,
        role=role,
        channels=channels,
        colorspace="srgb" if role in texroles.SRGB_ROLES else "linear",
        normal_space=normal_space,
        y_convention=y_convention,
        y_confident=y_confident,
        confidence=1.0,
        evidence=tuple(notes) + guess.evidence,
    )


def _texrefs(
    index: int,
    draft: _Draft,
    fmt: str,
    exporter: str,
    overrides: Mapping[str, Mapping[str, Any]],
    applied: Set[str],
    warn: Callable[[str], None],
    cache: _PixelCache,
) -> List[TexRef]:
    """Group a material's bindings by image, classify each image, apply corrections."""
    groups: "OrderedDict[str, Tuple[bytes, List[_Ref]]]" = OrderedDict()
    for ref in draft.refs:
        if ref.data is None:
            warn(f"{draft.name}: {ref.name} {ref.missing}")
            continue
        digest = hashlib.sha1(ref.data).hexdigest()
        groups.setdefault(digest, (ref.data, []))[1].append(ref)

    out: List[TexRef] = []
    taken: Set[str] = set()
    for digest, (data, bound) in groups.items():
        first = bound[0]
        slots = list(dict.fromkeys(ref.slot for ref in bound))
        try:
            pixels = cache.pixels(digest, data)
        except ImageError as exc:
            warn(f"{draft.name}: {first.name} was skipped: {exc}")
            continue
        guess = texroles.classify(pixels, first.name, slots, fmt, exporter)
        guess = _imfchan(guess, slots, pixels)
        key = _unique_key(str(index), first.name, taken)
        if key in overrides:
            applied.add(key)
            try:
                guess = _override(guess, overrides[key], pixels)
            except ValueError as exc:
                warn(f"{draft.name}: the correction for {first.name} was ignored: {exc}")
        out.append(
            TexRef(
                name=first.name,
                data=data,
                slots=slots,
                guess=guess,
                key=key,
                texcoord=first.texcoord,
                wrap_s=first.wrap[0],
                wrap_t=first.wrap[1],
                transform=first.transform,
                _cache=cache,
            )
        )
    return out


def _cutout(ref: TexRef, channel: str) -> bool:
    """Whether an opacity map is a cutout (nearly all texels clear or solid)."""
    plane = _channel(ref.pixels(), channel)[::4, ::4]
    full = 65535 if plane.dtype == np.uint16 else 255
    partial = np.count_nonzero((plane > 0.1 * full) & (plane < 0.9 * full))
    return partial < 0.1 * plane.size


def _settle(spec: MaterialSpec, draft: _Draft, warn: Callable[[str], None]) -> None:
    """Factors, alpha and UV transform, now that the maps' roles are known."""
    if draft.exact:
        _gltf_alpha(spec, draft.alpha_mode)
        spec.base_color_factor = (*draft.base, draft.alpha)
        spec.metallic, spec.roughness = draft.metallic, draft.roughness
        spec.emissive = draft.emissive
        spec.occlusion_strength = draft.occlusion_strength
        spec.normal_scale = draft.normal_scale
        spec.alpha_mode, spec.alpha_cutoff = draft.alpha_mode, draft.alpha_cutoff
        spec.double_sided = draft.double_sided
        gloss = draft.spec_gloss
    else:
        textured = spec.sources
        spec.base_color_factor = (
            *((1.0, 1.0, 1.0) if "basecolor" in textured else draft.base),
            1.0 if "opacity" in textured else draft.alpha,
        )
        spec.metallic = 1.0 if "metallic" in textured else draft.metallic
        spec.roughness = 1.0 if "roughness" in textured else draft.roughness
        spec.emissive = (1.0, 1.0, 1.0) if "emissive" in textured else draft.emissive
        if "opacity" in textured:
            opacity = textured["opacity"]
            cutout = _cutout(opacity, opacity.guess.channels["opacity"])
            spec.alpha_mode = "MASK" if cutout else "BLEND"
        elif draft.alpha < 1.0:
            spec.alpha_mode = "BLEND"
        spec.double_sided = spec.alpha_mode != "OPAQUE"
        normal = textured.get("normal")
        for slot in normal.slots if normal is not None else ():
            if "-bm" in slot.options:
                spec.normal_scale = _floats(slot.options["-bm"], 1, (1.0,))[0]
        gloss = None
        if any(ref.role == "spec_gloss" for ref in spec.refs):
            gloss = SpecGloss(diffuse=spec.base_color_factor)

    if gloss is not None:
        spec.spec_gloss = gloss
        if any(q in spec.sources for q in ("basecolor", "opacity", "specular", "roughness")):
            spec.base_color_factor, spec.metallic, spec.roughness = (1.0,) * 4, 1.0, 1.0
        else:
            spec.spec_gloss = None  # factors only: convert them once, here
            _convert_factors(spec, gloss)

    _choose_transform(spec, warn)


def _gltf_alpha(spec: MaterialSpec, alpha_mode: str) -> None:
    """In a glTF MASK or BLEND material the base colour map's alpha is opacity, always.

    The classifier drops an alpha channel it sees no variation in, which its
    random sample of texels can miss when only a few of them are clear.
    """
    base = spec.sources.get("basecolor")
    if alpha_mode == "OPAQUE" or base is None or base.role != "basecolor":
        return
    pixels = base.pixels()
    if "opacity" in base.guess.channels or pixels.ndim != 3 or pixels.shape[2] not in (2, 4):
        return
    base.guess = replace(
        base.guess,
        channels=dict(base.guess.channels, opacity="a"),
        evidence=base.guess.evidence + (f"glTF alphaMode {alpha_mode}: alpha is opacity",),
    )


def _convert_factors(spec: MaterialSpec, gloss: SpecGloss) -> None:
    from trimesh.visual.gloss import specular_to_pbr

    result = specular_to_pbr(
        specularFactor=list(gloss.specular),
        glossinessFactor=gloss.glossiness,
        diffuseFactor=list(gloss.diffuse),
    )
    colour = [float(v) for v in result["baseColorFactor"]]
    spec.base_color_factor = tuple(_floats(colour, 4, (1.0,) * 4))
    spec.metallic = float(result["metallicFactor"])
    spec.roughness = float(result["roughnessFactor"])


def _choose_transform(spec: MaterialSpec, warn: Callable[[str], None]) -> None:
    """One UV transform per material: the base colour map's, else any map's."""
    base = spec.sources.get("basecolor")
    transforms = [ref.transform for ref in spec.refs]
    spec.uv_transform = (
        base.transform
        if base is not None
        else next((t for t in transforms if t is not None), None)
    )
    if any(t != spec.uv_transform for t in transforms):
        warn(
            f"{spec.name}: its maps use different texture transforms; "
            "all are shown with the base colour map's"
        )
    for ref in spec.refs:
        if ref.texcoord:
            warn(
                f"{spec.name}: {ref.name} uses UV set {ref.texcoord}; "
                "it is shown on the first"
            )


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def read_materials(
    path: Union[str, os.PathLike],
    overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    assimp_materials: Optional[Sequence[Mapping[str, Any]]] = None,
) -> MaterialRead:
    """Every material of a model file, each map classified, factors resolved.

    Reads .glb/.gltf, .obj (through its mtllib) or .mtl, .fbx and .dae; any
    other format has no materials to read.  ``overrides`` maps a
    :attr:`TexRef.key` to the user's correction of it (see :func:`_override`).
    For FBX and Collada the materials line up with Assimp's; pass
    ``assimp_materials`` (``scene.materials``) when the scene is already
    loaded, or the file is imported again here just for them.

    Every image is decoded once, to classify it.  Nothing here raises for a
    bad map or a malformed material block: those become warnings, and a file
    whose materials cannot be read at all gives no materials and one warning.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    warnings: List[str] = []

    def warn(line: str) -> None:
        if line not in warnings:
            warnings.append(line)

    try:
        if suffix in GLTF_SUFFIXES:
            drafts, bindings = _read_gltf(source)
            fmt = "gltf"
        elif suffix in OBJ_SUFFIXES:
            drafts, bindings = _read_obj(source, warn)
            fmt = "obj"
        elif suffix in ASSIMP_SUFFIXES:
            drafts, bindings = _read_assimp(source, assimp_materials, warn)
            fmt = "fbx" if suffix == ".fbx" else "assimp"
        else:
            return MaterialRead([], Bindings(), [])
    except Exception as exc:
        LOG.warning("could not read the materials of %s", source, exc_info=True)
        why = f"The materials of {source.name} could not be read: {exc}"
        return MaterialRead([], Bindings(), [why])

    cache = _PixelCache()
    corrections = dict(overrides or {})
    applied: Set[str] = set()
    materials: List[MaterialSpec] = []
    for index, draft in enumerate(drafts):
        spec = MaterialSpec(name=draft.name)
        try:
            spec.refs = _texrefs(
                index, draft, fmt, bindings.exporter, corrections, applied, warn, cache
            )
            _settle(spec, draft, warn)
        except Exception as exc:
            LOG.warning("could not read material %s of %s", draft.name, source, exc_info=True)
            warn(f"{draft.name}: its maps could not be read ({exc})")
            spec = MaterialSpec(
                name=draft.name,
                base_color_factor=(*draft.base, draft.alpha),
                metallic=draft.metallic,
                roughness=draft.roughness,
                emissive=draft.emissive,
            )
        materials.append(spec)
    for key in corrections:
        if key not in applied:
            warn(f"There is no map {key} to correct any more")
    return MaterialRead(materials, bindings, warnings)
