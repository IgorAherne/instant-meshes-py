"""What a binary FBX says about its materials, straight from the file.

Assimp reads FBX geometry, its UVs and which material each mesh uses, but it
hands back ``*0`` for a texture the file embeds, with no way through its Python
binding to ask what ``*0`` holds, and it files every texture under one of its
own channel types, which loses what the exporter actually wrote.  Blender binds
its roughness map to ``ShininessExponent`` and its metallic map to
``ReflectionFactor``; 3ds Max binds glossiness to the first and an environment
map to ``ReflectionColor``.  Telling those apart needs the raw property name
and the program that wrote the file, so both are read here, along with the
embedded image bytes.

Only as much of the format as that needs is implemented.  A binary FBX is a
tree of records -- a length, a property list, then nested records.  The object
kinds that matter are ``Material`` (a name), ``Texture`` (the link between the
two, and the file name of the map), ``LayeredTexture`` (a stack of textures
bound as one) and ``Video`` (the bytes, under ``Content``).  ``Connections``
says which belongs to which, and names the material property each texture is
bound to.  ``FBXHeaderExtension`` names the exporter and ``GlobalSettings``
holds the file's axes and unit.

Anything unexpected -- an ASCII FBX, a version this does not know, a truncated
file -- returns None rather than raising.  The geometry has already loaded by
then, and a model without its maps is still a model to remesh.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

#: What a binary FBX starts with. An ASCII one does not, and is left alone.
MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"

#: Records switched from 32- to 64-bit offsets at this version.
WIDE_VERSION = 7500

#: Ceiling on the bytes one embedded image may claim, so that a corrupt length
#: cannot ask for a terabyte before anything has been read.
MAX_IMAGE_BYTES = 256 * 1024 * 1024

#: Scalar property types: their struct format and byte size.
_SCALARS: Dict[str, Tuple[str, int]] = {
    "Y": ("<h", 2),
    "C": ("<?", 1),
    "I": ("<i", 4),
    "F": ("<f", 4),
    "D": ("<d", 8),
    "L": ("<q", 8),
}


class TextureBinding(NamedTuple):
    """One texture bound to one property of a material."""

    #: The material property exactly as the file spells it: ``"DiffuseColor"``,
    #: ``"3dsMax|Parameters|bump_map"``, ``"Maya|normalCamera"``.
    property: str
    #: The image file the texture names (its RelativeFilename, else FileName),
    #: or the texture's own name when it names no file.  Paths are the
    #: exporter's, from another machine as often as not.
    file_name: str
    #: The image the file embeds, or None when it has to be found on disk.
    data: Optional[bytes]


@dataclass(frozen=True)
class Material:
    """One FBX material and the textures bound to it."""

    name: str
    textures: Tuple[TextureBinding, ...]


@dataclass(frozen=True)
class GlobalSettings:
    """The axis system and unit the file's coordinates are written in.

    FBX names three axes -- up, front and coord -- each as an index (0 = X,
    1 = Y, 2 = Z) and a sign.  The defaults are Maya's, which glTF shares:
    +Y up, +Z front, +X coord.  3ds Max writes +Z up and -Y front.
    """

    up_axis: int = 1
    up_sign: int = 1
    front_axis: int = 2
    front_sign: int = 1
    coord_axis: int = 0
    coord_sign: int = 1
    #: Centimetres per file unit (1 = centimetres, 100 = metres).
    unit_scale: float = 1.0

    def axes(self) -> np.ndarray:
        """``(3, 3)`` matrix taking file coordinates to glTF's frame.

        Its rows are the coord, up and front axes, so they land on +X, +Y
        and +Z.  A signed permutation: it only swaps and flips axes.  Its
        determinant is -1 when the file's frame is left-handed, and triangles
        then need their winding reversed to keep facing outward.
        """
        basis = np.zeros((3, 3))
        for row, (axis, sign) in enumerate(
            (
                (self.coord_axis, self.coord_sign),
                (self.up_axis, self.up_sign),
                (self.front_axis, self.front_sign),
            )
        ):
            basis[row, axis] = 1.0 if sign >= 0 else -1.0
        return basis

    @property
    def metres_per_unit(self) -> float:
        return self.unit_scale / 100.0

    @property
    def valid(self) -> bool:
        """Whether the three axes are distinct, i.e. :meth:`axes` is invertible."""
        return sorted((self.up_axis, self.front_axis, self.coord_axis)) == [0, 1, 2]


@dataclass(frozen=True)
class FbxMedia:
    """Everything :func:`read` took out of one file."""

    #: Every Material object, in the order the file lists them, with or
    #: without textures.
    materials: Tuple[Material, ...]
    #: ``FBXHeaderExtension`` Creator, e.g. "FBX SDK/FBX Plugins version 2012.2".
    creator: str
    #: SceneInfo ``Original|ApplicationName``, e.g. "3ds Max",
    #: "Blender (stable FBX IO)".  Often empty.
    application: str
    settings: GlobalSettings

    @property
    def exporter(self) -> str:
        """The program that wrote the file, as briefly as the file says it."""
        return self.application or self.creator


class _Record:
    """One node of the tree: a name, its properties and its children."""

    __slots__ = ("name", "props", "children")

    def __init__(self, name: str, props: List[Any], children: List["_Record"]) -> None:
        self.name = name
        self.props = props
        self.children = children

    def find(self, name: str) -> Optional["_Record"]:
        for child in self.children:
            if child.name == name:
                return child
        return None

    def every(self, name: str) -> List["_Record"]:
        return [child for child in self.children if child.name == name]

    def first_text(self, *names: str) -> str:
        """The first string property of the first child named, or ''."""
        for name in names:
            child = self.find(name)
            if child is not None and child.props:
                text = _text(child.props[0])
                if text:
                    return text
        return ""


class _Reader:
    """A cursor over the file, which parses records on demand."""

    def __init__(self, data: bytes, version: int) -> None:
        self.data = data
        self.wide = version >= WIDE_VERSION

    def _word(self, at: int) -> Tuple[int, int]:
        if self.wide:
            return struct.unpack_from("<Q", self.data, at)[0], at + 8
        return struct.unpack_from("<I", self.data, at)[0], at + 4

    def record(self, at: int) -> Tuple[Optional[_Record], int]:
        """Read one record. ``(None, end)`` marks the list terminator."""
        end, at = self._word(at)
        count, at = self._word(at)
        length, at = self._word(at)
        (name_length,) = struct.unpack_from("<B", self.data, at)
        at += 1
        if end == 0:
            return None, at + name_length
        if end > len(self.data):
            raise ValueError("record runs past the end of the file")
        name = self.data[at : at + name_length].decode("utf-8", "replace")
        at += name_length

        props = [None] * count
        stop = at + length
        for index in range(count):
            props[index], at = self.property(at)
            if at > stop:
                raise ValueError(f"{name}: property list overran its length")
        at = stop

        children: List[_Record] = []
        while at < end:
            child, at = self.record(at)
            if child is None:
                break
            children.append(child)
        return _Record(name, props, children), end

    def property(self, at: int) -> Tuple[Any, int]:
        kind = self.data[at : at + 1].decode("ascii", "replace")
        at += 1

        if kind in _SCALARS:
            fmt, size = _SCALARS[kind]
            return struct.unpack_from(fmt, self.data, at)[0], at + size

        if kind in ("S", "R"):
            (length,) = struct.unpack_from("<I", self.data, at)
            at += 4
            return self.data[at : at + length], at + length

        if kind in ("f", "d", "l", "i", "b"):
            # Skipped rather than decoded: the arrays in an FBX are its
            # geometry, which Assimp has already read, and inflating tens of
            # megabytes of vertex positions to throw them away is the whole
            # cost of this pass.
            _, _, compressed = struct.unpack_from("<III", self.data, at)
            at += 12
            return None, at + compressed

        raise ValueError(f"unknown property type {kind!r}")


def _object_name(raw: Any) -> str:
    """The name out of an object record's ``name\\x00\\x01Class`` property."""
    if not isinstance(raw, (bytes, bytearray)):
        return ""
    text = bytes(raw)
    if b"\x00\x01" in text:
        text = text.split(b"\x00\x01", 1)[0]
    name = text.decode("utf-8", "replace")
    # An ASCII-style "Material::skin" reaches here from some exporters too.
    return name.rsplit("::", 1)[-1].strip()


def _text(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return str(value or "")


def _properties70(record: Optional[_Record]) -> Dict[str, Any]:
    """``{name: first value}`` of a record's ``Properties70`` block."""
    block = record.find("Properties70") if record is not None else None
    values: Dict[str, Any] = {}
    for entry in block.every("P") if block is not None else ():
        # P: name, type, label, flags, value...
        if len(entry.props) >= 5:
            values[_text(entry.props[0])] = entry.props[4]
    return values


def _settings(record: Optional[_Record]) -> GlobalSettings:
    values = _properties70(record)
    defaults = GlobalSettings()

    def number(name: str, default: float) -> float:
        value = values.get(name, default)
        return float(value) if isinstance(value, (int, float)) else default

    settings = GlobalSettings(
        up_axis=int(number("UpAxis", defaults.up_axis)),
        up_sign=int(number("UpAxisSign", defaults.up_sign)),
        front_axis=int(number("FrontAxis", defaults.front_axis)),
        front_sign=int(number("FrontAxisSign", defaults.front_sign)),
        coord_axis=int(number("CoordAxis", defaults.coord_axis)),
        coord_sign=int(number("CoordAxisSign", defaults.coord_sign)),
        unit_scale=number("UnitScaleFactor", defaults.unit_scale) or defaults.unit_scale,
    )
    if not settings.valid:
        LOG.info("ignoring an FBX axis system with repeated axes: %s", settings)
        return GlobalSettings(unit_scale=settings.unit_scale)
    return settings


def _embedded(video: _Record) -> Optional[bytes]:
    holder = video.find("Content")
    data = holder.props[0] if holder is not None and holder.props else None
    if not isinstance(data, (bytes, bytearray)) or not data:
        return None
    if len(data) > MAX_IMAGE_BYTES:
        LOG.warning("skipping a %d byte embedded texture", len(data))
        return None
    return bytes(data)


def _base_name(named: str) -> str:
    return PureWindowsPath(named.replace("\\", "/")).name.lower()


def read(path: Path) -> Optional[FbxMedia]:
    """The materials, textures, exporter and axes of a binary FBX.

    None when the file is not a binary FBX this can parse -- an ASCII FBX, a
    truncated one, or no FBX at all.  Assimp is the reader of last resort for
    those.
    """
    try:
        return _read(Path(path))
    except Exception:
        LOG.info("could not read the FBX materials of %s", path, exc_info=True)
        return None


def _read(path: Path) -> Optional[FbxMedia]:
    data = path.read_bytes()
    if not data.startswith(MAGIC):
        return None  # ASCII FBX, or not an FBX at all
    (version,) = struct.unpack_from("<I", data, len(MAGIC))

    reader = _Reader(data, version)
    at = len(MAGIC) + 4
    top: Dict[str, _Record] = {}
    while at < len(data) - 16:
        record, at = reader.record(at)
        if record is None:
            break
        top.setdefault(record.name, record)
        if "Objects" in top and "Connections" in top:
            break  # everything after (Takes) is animation
    objects, connections = top.get("Objects"), top.get("Connections")
    if objects is None or connections is None:
        return None

    header = top.get("FBXHeaderExtension")
    creator = header.first_text("Creator") if header is not None else ""
    if not creator and "Creator" in top and top["Creator"].props:
        creator = _text(top["Creator"].props[0])
    scene_info = header.find("SceneInfo") if header is not None else None
    application = _text(_properties70(scene_info).get("Original|ApplicationName", ""))

    materials: Dict[int, str] = {}
    textures: Dict[int, Tuple[str, str]] = {}  # texture -> (the file it names, its name)
    videos: Dict[int, Tuple[str, Optional[bytes]]] = {}
    layered: Dict[int, List[int]] = {}  # layered texture -> its textures
    for child in objects.children:
        if not child.props or not isinstance(child.props[0], int):
            continue
        identity = child.props[0]
        name = _object_name(child.props[1] if len(child.props) > 1 else b"")
        if child.name == "Material":
            materials[identity] = name or f"material {len(materials)}"
        elif child.name == "Texture":
            textures[identity] = (child.first_text("RelativeFilename", "FileName"), name)
        elif child.name == "Video":
            named = child.first_text("RelativeFilename", "Filename", "FileName")
            videos[identity] = (named, _embedded(child))
        elif child.name == "LayeredTexture":
            layered[identity] = []

    # Some exporters embed one copy of an image that several Video records
    # name, so a Video without content borrows the bytes of one with the
    # same file name.
    by_file = {
        _base_name(named): content
        for named, content in videos.values()
        if content is not None and named
    }

    # Connections are written child-first: a Video hangs off a Texture, and
    # that Texture (or a LayeredTexture stacking it) hangs off the property of
    # the Material it feeds.
    video_of: Dict[int, int] = {}
    bindings: List[Tuple[int, int, str]] = []  # (texture or layered, material, property)
    for link in connections.every("C"):
        props = link.props
        if len(props) < 3 or not isinstance(props[1], int) or not isinstance(props[2], int):
            continue
        kind = _text(props[0])
        child, parent = props[1], props[2]
        if child in videos and parent in textures:
            video_of[parent] = child
        elif child in textures and parent in layered:
            layered[parent].append(child)
        elif (child in textures or child in layered) and parent in materials:
            binding = _text(props[3]) if kind == "OP" and len(props) > 3 else ""
            bindings.append((child, parent, binding))

    found: Dict[int, List[TextureBinding]] = {identity: [] for identity in materials}
    for source, material, binding in bindings:
        for texture in layered.get(source, [source]):
            named, label = textures[texture]
            content: Optional[bytes] = None
            if texture in video_of:
                video_named, content = videos[video_of[texture]]
                named = named or video_named
            if content is None and named:
                content = by_file.get(_base_name(named))
            found[material].append(TextureBinding(binding, named or label, content))

    return FbxMedia(
        materials=tuple(
            Material(name, tuple(found[identity])) for identity, name in materials.items()
        ),
        creator=creator,
        application=application,
        settings=_settings(top.get("GlobalSettings")),
    )
