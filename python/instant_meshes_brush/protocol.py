"""Binary WebSocket framing shared by the server and the browser viewer.

Why not JSON
------------
A brushing session streams whole vertex fields: for a 200k-vertex working mesh
one field update is 2.4 MB of float32.  JSON would inflate that roughly 6x and
cost tens of milliseconds of parsing per frame on both ends.  Instead every
frame is a single binary blob whose typed arrays are handed straight to
``numpy.frombuffer`` on the server and to typed-array views on an ``ArrayBuffer``
in the browser -- zero copies, zero parsing.

Frame layout (all little-endian)::

    offset  size  field
    0       4     magic  = 0x53424D49 ('IMBS' read as LE uint32)
    4       2     version
    6       2     message type
    8       4     header length, always a multiple of 4
    12      H     UTF-8 JSON header, space-padded to a multiple of 4
    12+H    ...   array payloads, back to back, in header["arrays"] order

``header["arrays"]`` is a list of ``{"name", "dtype", "shape"}``.  Every dtype is
4 bytes wide, and the header is padded to a multiple of 4, so every payload
starts 4-byte aligned -- which is what ``Float32Array``/``Uint32Array`` views
over an ``ArrayBuffer`` require.

The matching JavaScript implementation is ``static/protocol.js``; the two must
be changed together, and ``PROTOCOL_VERSION`` bumped.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

MAGIC = 0x53424D49
PROTOCOL_VERSION = 1

_HEADER_STRUCT = struct.Struct("<IHHI")

# Only fixed-width 4-byte types are allowed, which is what keeps every payload
# aligned without explicit padding between arrays.
_DTYPES: Dict[str, np.dtype] = {
    "f32": np.dtype("<f4"),
    "u32": np.dtype("<u4"),
    "i32": np.dtype("<i4"),
}
_DTYPE_NAMES = {np.dtype(v): k for k, v in _DTYPES.items()}


class MessageType:
    """Wire message identifiers. Client->server below 100, server->client above."""

    # --- client -> server ---------------------------------------------------
    HELLO = 1
    LOAD_MESH = 2
    SET_CONFIG = 3
    STROKE = 4
    ERASE_STROKE = 5
    SOLVE = 6
    STOP = 7
    EXTRACT = 8
    EXPORT = 9
    CLEAR_STROKES = 10
    SUBSCRIBE = 11
    PING = 12

    # --- server -> client ---------------------------------------------------
    GEOMETRY = 100
    FIELD = 101
    STROKE_RESULT = 102
    SINGULARITIES = 103
    EXTRACTED = 104
    STATUS = 105
    ERROR = 106
    PROGRESS = 107
    EXPORT_READY = 108
    STROKE_LIST = 109
    PONG = 110


class ProtocolError(ValueError):
    """Raised when a frame is malformed or uses an unsupported version."""


@dataclass
class Message:
    """A decoded frame: a small JSON header plus named numpy arrays."""

    type: int
    header: Dict[str, Any] = field(default_factory=dict)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.header.get(key, default)


def encode(
    msg_type: int,
    header: Optional[Mapping[str, Any]] = None,
    arrays: Optional[Mapping[str, np.ndarray]] = None,
) -> bytes:
    """Serialise a message to a single ``bytes`` frame.

    Arrays are converted to little-endian and made contiguous if necessary; a
    correctly-typed contiguous array is written without being copied twice.
    """
    header_obj: Dict[str, Any] = dict(header or {})
    descriptors = []
    buffers = []

    for name, value in (arrays or {}).items():
        arr = np.asarray(value)
        dtype_name = _DTYPE_NAMES.get(arr.dtype.newbyteorder("<"))
        if dtype_name is None:
            raise ProtocolError(
                f"array {name!r} has unsupported dtype {arr.dtype}; "
                f"use one of {sorted(_DTYPES)}"
            )
        arr = np.ascontiguousarray(arr, dtype=_DTYPES[dtype_name])
        descriptors.append({"name": name, "dtype": dtype_name, "shape": list(arr.shape)})
        buffers.append(arr)

    if descriptors:
        header_obj["arrays"] = descriptors

    # ensure_ascii=False matches JSON.stringify on the JavaScript side, so an
    # identical header produces an identical frame in both directions.
    raw_header = json.dumps(
        header_obj, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    padding = (-len(raw_header)) % 4
    raw_header += b" " * padding

    parts = [
        _HEADER_STRUCT.pack(MAGIC, PROTOCOL_VERSION, msg_type, len(raw_header)),
        raw_header,
    ]
    parts.extend(arr.tobytes() for arr in buffers)
    return b"".join(parts)


def decode(data: bytes) -> Message:
    """Parse a frame produced by :func:`encode` (or its JavaScript twin)."""
    if len(data) < _HEADER_STRUCT.size:
        raise ProtocolError("frame shorter than its fixed header")

    magic, version, msg_type, header_len = _HEADER_STRUCT.unpack_from(data, 0)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic 0x{magic:08X}")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version {version} != {PROTOCOL_VERSION}; reload the page"
        )
    if header_len % 4:
        raise ProtocolError("header length is not a multiple of 4")

    start = _HEADER_STRUCT.size
    end = start + header_len
    if len(data) < end:
        raise ProtocolError("frame truncated inside its header")

    try:
        header_obj = json.loads(data[start:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"header is not valid JSON: {exc}") from exc
    if not isinstance(header_obj, dict):
        raise ProtocolError("header must be a JSON object")

    arrays: Dict[str, np.ndarray] = {}
    offset = end
    for descriptor in header_obj.pop("arrays", []):
        try:
            name = descriptor["name"]
            dtype = _DTYPES[descriptor["dtype"]]
            shape = tuple(int(n) for n in descriptor["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"malformed array descriptor: {descriptor!r}") from exc

        count = 1
        for n in shape:
            if n < 0:
                raise ProtocolError(f"array {name!r} has a negative dimension")
            count *= n
        nbytes = count * dtype.itemsize
        if len(data) < offset + nbytes:
            raise ProtocolError(f"frame truncated inside array {name!r}")

        arrays[name] = np.frombuffer(
            data, dtype=dtype, count=count, offset=offset
        ).reshape(shape)
        offset += nbytes

    return Message(type=msg_type, header=header_obj, arrays=arrays)


def error(message: str, *, fatal: bool = False) -> bytes:
    """Convenience builder for :data:`MessageType.ERROR` frames."""
    return encode(MessageType.ERROR, {"message": message, "fatal": fatal})
