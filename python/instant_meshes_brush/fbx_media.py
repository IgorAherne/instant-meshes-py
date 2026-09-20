"""The texture images an FBX file carries inside itself.

Assimp reads FBX geometry, its UVs and which material each mesh uses, but a
material's texture path comes back as ``*0`` for a map the file embeds and
there is no way through its Python binding to ask what ``*0`` holds.  Embedded
is the only kind that can work here: the browser uploads one file, not the
folder of maps that sat beside it in the exporter.  So the images are taken out
of the file directly.

Only as much of the format as that needs is implemented.  A binary FBX is a
tree of records -- a length, a property list, then nested records -- and the
three object kinds that matter are ``Material`` (a name), ``Texture`` (the link
between the two) and ``Video`` (the bytes, under ``Content``).  ``Connections``
says which belongs to which, and names the material property each texture is
bound to, which is what makes a map a normal map rather than a colour one.

Anything unexpected -- an ASCII FBX, a version this does not know, a truncated
file -- returns nothing rather than raising.  The geometry has already loaded
by then, and a model without its maps is still a model to remesh.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)

#: What a binary FBX starts with. An ASCII one does not, and is left alone.
MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"

#: Records switched from 32- to 64-bit offsets at this version.
WIDE_VERSION = 7500

#: Ceiling on the bytes one embedded image may claim, so that a corrupt length
#: cannot ask for a terabyte before anything has been read.
MAX_IMAGE_BYTES = 256 * 1024 * 1024

#: Material properties an FBX texture is bound to, and the channel each means.
#: Spelled lower case here and matched that way: exporters disagree about case
#: and about their vendor prefixes ("Maya|normalCamera"), and the tail after
#: the last '|' is the part they agree on.
_PROPERTIES: Dict[str, str] = {
    "diffusecolor": "base colour",
    "diffuse": "base colour",
    "basecolor": "base colour",
    "base_color": "base colour",
    "normalmap": "normal",
    "normalcamera": "normal",
    "bump": "normal",
    "bumpmap": "normal",
    "shininessexponent": "metal/rough",
    "shininess": "metal/rough",
    "roughness": "metal/rough",
    "metalness": "metal/rough",
    "specularcolor": "specular",
    "specularfactor": "specular",
    "reflectionfactor": "specular",
    "emissivecolor": "emissive",
    "emissive": "emissive",
    "ambientcolor": "occlusion",
    "ambientocclusion": "occlusion",
}

#: Channel for a texture whose binding property is not one of the above. A file
#: that names only one map almost always means the colour one.
_FALLBACK_CHANNEL = "base colour"


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

        scalars = {"Y": ("<h", 2), "C": ("<?", 1), "I": ("<i", 4),
                   "F": ("<f", 4), "D": ("<d", 8), "L": ("<q", 8)}
        if kind in scalars:
            fmt, size = scalars[kind]
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


def _channel(binding: str) -> str:
    """The channel a material property name stands for."""
    tail = binding.rsplit("|", 1)[-1].strip().lower()
    return _PROPERTIES.get(tail, _FALLBACK_CHANNEL)


def _content(video: _Record, beside: Optional[Path]) -> Optional[bytes]:
    """The image bytes a Video record stands for.

    Embedded first, because that is the only kind an upload can carry. A file
    opened from disk may still have its maps in the folder next to it, which is
    how an exporter that did not tick "embed media" leaves them.
    """
    holder = video.find("Content")
    data = holder.props[0] if holder is not None and holder.props else None
    if isinstance(data, (bytes, bytearray)) and data:
        if len(data) > MAX_IMAGE_BYTES:
            LOG.warning("skipping a %d byte embedded texture", len(data))
            return None
        return bytes(data)
    return _beside(video, beside)


def _beside(video: _Record, folder: Optional[Path]) -> Optional[bytes]:
    """An external map, looked for where the file that named it lives."""
    if folder is None:
        return None
    for field in ("RelativeFilename", "Filename"):
        named = _text(video.find(field).props[0]) if video.find(field) else ""
        if not named:
            continue
        # Only ever inside the folder the model came from: a path out of a
        # file is data, and "..\..\Windows\..." is a path like any other.
        name = PureWindowsPath(named.replace("\\", "/")).name
        if not name:
            continue
        for candidate in (folder / name, folder / "textures" / name):
            try:
                if candidate.is_file() and candidate.stat().st_size <= MAX_IMAGE_BYTES:
                    return candidate.read_bytes()
            except OSError:
                LOG.debug("could not read %s", candidate, exc_info=True)
    return None


def read(path: Path) -> Dict[str, List[Tuple[str, bytes]]]:
    """Embedded maps per material name: ``{name: [(channel, bytes), ...]}``.

    An empty result means the file has none this can reach, which is the
    ordinary answer for an FBX whose textures live beside it on disk.
    """
    try:
        return _read(Path(path))
    except Exception:
        LOG.info("no embedded textures read from %s", path, exc_info=True)
        return {}


def _read(path: Path) -> Dict[str, List[Tuple[str, bytes]]]:
    folder = path.parent
    data = path.read_bytes()
    if not data.startswith(MAGIC):
        return {}  # ASCII FBX, or not an FBX at all
    (version,) = struct.unpack_from("<I", data, len(MAGIC))

    reader = _Reader(data, version)
    at = len(MAGIC) + 4
    objects: Optional[_Record] = None
    connections: Optional[_Record] = None
    while at < len(data) - 16:
        record, at = reader.record(at)
        if record is None:
            break
        if record.name == "Objects":
            objects = record
        elif record.name == "Connections":
            connections = record
        if objects is not None and connections is not None:
            break
    if objects is None or connections is None:
        return {}

    materials: Dict[int, str] = {}
    videos: Dict[int, bytes] = {}
    textures: Dict[int, Optional[int]] = {}
    for child in objects.children:
        if not child.props:
            continue
        identity = child.props[0]
        if not isinstance(identity, int):
            continue
        if child.name == "Material":
            materials[identity] = _object_name(
                child.props[1] if len(child.props) > 1 else b""
            ) or f"material {len(materials)}"
        elif child.name == "Video":
            content = _content(child, folder)
            if content is not None:
                videos[identity] = content
        elif child.name == "Texture":
            textures[identity] = None

    # Connections are written child-first: a Video hangs off a Texture, and
    # that Texture hangs off the property of a Material it colours.
    bindings: List[Tuple[int, int, str]] = []  # (texture, material, property)
    for link in connections.every("C"):
        props = link.props
        if len(props) < 3 or not isinstance(props[1], int) or not isinstance(props[2], int):
            continue
        kind = _text(props[0])
        child, parent = props[1], props[2]
        if child in videos and parent in textures:
            textures[parent] = child
        elif child in textures and parent in materials:
            binding = _text(props[3]) if kind == "OP" and len(props) > 3 else ""
            bindings.append((child, parent, binding))

    found: Dict[str, List[Tuple[str, bytes]]] = {}
    for texture, material, binding in bindings:
        video = textures.get(texture)
        if video is None:
            continue
        found.setdefault(materials[material], []).append((_channel(binding), videos[video]))
    return found
