"""Tests for the binary WebSocket framing in ``instant_meshes_brush.protocol``.

The browser half of this contract (``static/protocol.js``) reads payloads as
typed-array views straight onto the received ``ArrayBuffer``, which throws
unless every payload starts on a 4-byte boundary.  Python never notices -- it
copies -- so the alignment and dtype rules are checked here explicitly rather
than left to be discovered in a browser console.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

import instant_meshes_brush.protocol as protocol
from instant_meshes_brush.protocol import (
    MAGIC,
    PROTOCOL_VERSION,
    MessageType,
    ProtocolError,
    decode,
    encode,
    error,
)

#: The JavaScript half of the contract, shipped next to its Python twin.
PROTOCOL_JS = Path(protocol.__file__).resolve().parent / "static" / "protocol.js"

#: magic, version, message type, header length -- the frame prefix decode() reads
#: before it trusts anything else.
_FIXED_HEADER = struct.Struct("<IHHI")


#: Every dtype the protocol permits is exactly this wide, which is the whole
#: reason a padded header is enough to keep the payloads aligned.
_ITEMSIZE = 4


def _payload_offsets(frame: bytes) -> List[int]:
    """Byte offset of every array payload in ``frame``, in wire order."""
    header_len = _FIXED_HEADER.unpack_from(frame, 0)[3]
    start = _FIXED_HEADER.size + header_len
    header = json.loads(frame[_FIXED_HEADER.size : start])

    offsets = []
    offset = start
    for descriptor in header.get("arrays", []):
        offsets.append(offset)
        offset += int(np.prod(descriptor["shape"], dtype=np.int64)) * _ITEMSIZE
    return offsets


def _corrupt(frame: bytes, offset: int, value: bytes) -> bytes:
    """Overwrite ``len(value)`` bytes of a frame, leaving its length intact."""
    return frame[:offset] + value + frame[offset + len(value) :]


def _handmade(header_obj: Any, payload: bytes = b"") -> bytes:
    """A well-formed frame carrying an arbitrary header.

    ``encode()`` cannot produce the headers below; a buggy or hostile peer can,
    which is the only way to reach the rest of ``decode()``'s guards.
    """
    raw = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 4)
    prefix = _FIXED_HEADER.pack(MAGIC, PROTOCOL_VERSION, MessageType.FIELD, len(raw))
    return prefix + raw + payload


# ---------------------------------------------------------------------------
#  Round trips
# ---------------------------------------------------------------------------


def test_header_only_round_trip() -> None:
    frame = encode(MessageType.SOLVE, {"field": "orientations", "level": -1})

    message = decode(frame)

    assert message.type == MessageType.SOLVE
    assert message.header == {"field": "orientations", "level": -1}
    assert message.arrays == {}
    assert message.get("level") == -1
    assert message.get("missing", "fallback") == "fallback"


def test_single_array_round_trip() -> None:
    positions = np.arange(12, dtype=np.float32).reshape(4, 3) * 0.5

    message = decode(encode(MessageType.GEOMETRY, {"n": 4}, {"positions": positions}))

    assert message.header == {"n": 4}
    assert list(message.arrays) == ["positions"]
    assert message.arrays["positions"].shape == (4, 3)
    assert message.arrays["positions"].dtype == np.float32
    assert np.array_equal(message.arrays["positions"], positions)


def test_mixed_dtype_arrays_round_trip() -> None:
    arrays: Dict[str, np.ndarray] = {
        "vertices": np.linspace(-1.0, 1.0, 30, dtype=np.float32).reshape(10, 3),
        "faces": np.arange(24, dtype=np.uint32).reshape(6, 4),
        "shifts": np.array([[-1, 0], [2, -3]], dtype=np.int32),
        "flat": np.array([7], dtype=np.uint32),
    }

    message = decode(encode(MessageType.EXTRACTED, {}, arrays))

    # Order matters: the payloads are concatenated in header["arrays"] order.
    assert list(message.arrays) == list(arrays)
    for name, expected in arrays.items():
        decoded = message.arrays[name]
        assert decoded.shape == expected.shape, name
        assert decoded.dtype == expected.dtype, name
        assert np.array_equal(decoded, expected), name


def test_empty_array_round_trip() -> None:
    """A field update with nothing in it still has to name its shape."""
    empty = np.empty((0, 3), dtype=np.float32)

    message = decode(encode(MessageType.FIELD, {"kind": "Q"}, {"q": empty}))

    assert message.arrays["q"].shape == (0, 3)
    assert message.arrays["q"].dtype == np.float32
    assert message.arrays["q"].size == 0


def test_large_array_round_trip() -> None:
    """A realistic field update: 100k vertices is 1.2 MB in one frame."""
    field = np.linspace(-5.0, 5.0, 300_000, dtype=np.float32).reshape(100_000, 3)

    frame = encode(MessageType.FIELD, {"version": 42}, {"q": field})
    message = decode(frame)

    assert len(frame) == _payload_offsets(frame)[0] + field.nbytes
    assert message.arrays["q"].shape == (100_000, 3)
    assert np.array_equal(message.arrays["q"], field)


def test_error_helper_round_trip() -> None:
    message = decode(error("mesh has no faces", fatal=True))

    assert message.type == MessageType.ERROR
    assert message.header == {"message": "mesh has no faces", "fatal": True}


def test_non_contiguous_input_is_serialised_in_logical_order() -> None:
    """Fields are often sliced out of a bigger buffer before being sent."""
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    view = source[:, 1::2]
    assert not view.flags["C_CONTIGUOUS"]

    message = decode(encode(MessageType.FIELD, {}, {"o": view}))

    assert np.array_equal(message.arrays["o"], view)


# ---------------------------------------------------------------------------
#  Alignment and dtype rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filler", ["", "a", "ab", "abc", "abcd", "abcde"])
def test_every_payload_starts_four_byte_aligned(filler: str) -> None:
    """The filler walks the unpadded JSON length through every residue mod 4."""
    arrays = {
        "a": np.array([1.0], dtype=np.float32),
        "b": np.array([2, 3, 4], dtype=np.uint32),
        "c": np.array([[5, 6]], dtype=np.int32),
    }

    frame = encode(MessageType.GEOMETRY, {"pad": filler}, arrays)

    header_len = _FIXED_HEADER.unpack_from(frame, 0)[3]
    assert header_len % 4 == 0
    offsets = _payload_offsets(frame)
    assert len(offsets) == len(arrays)
    assert all(offset % 4 == 0 for offset in offsets)
    assert decode(frame).arrays.keys() == arrays.keys()


@pytest.mark.parametrize("dtype", ["float64", "uint8", "int16", "int64", "float16"])
def test_rejects_dtypes_that_are_not_four_bytes_wide(dtype: str) -> None:
    array = np.ones(4, dtype=dtype)

    with pytest.raises(ProtocolError) as excinfo:
        encode(MessageType.GEOMETRY, {}, {"payload": array})

    message = str(excinfo.value)
    assert "payload" in message
    assert dtype in message
    assert "f32" in message and "u32" in message and "i32" in message


def test_accepts_big_endian_input_and_writes_little_endian() -> None:
    """numpy will happily hand us a big-endian array; the wire is always LE."""
    array = np.arange(4, dtype=">u4")

    message = decode(encode(MessageType.GEOMETRY, {}, {"faces": array}))

    assert message.arrays["faces"].dtype == np.dtype("<u4")
    assert np.array_equal(message.arrays["faces"], array)


# ---------------------------------------------------------------------------
#  Malformed frames
# ---------------------------------------------------------------------------


def test_rejects_short_frame() -> None:
    with pytest.raises(ProtocolError, match="shorter than its fixed header"):
        decode(b"IMBS")


def test_rejects_bad_magic() -> None:
    frame = _corrupt(encode(MessageType.PING), 0, struct.pack("<I", 0xDEADBEEF))

    with pytest.raises(ProtocolError) as excinfo:
        decode(frame)

    assert "bad magic" in str(excinfo.value)
    assert "DEADBEEF" in str(excinfo.value).upper()
    assert isinstance(excinfo.value, ValueError)  # servers catch ValueError


def test_rejects_bad_version() -> None:
    bumped = struct.pack("<H", PROTOCOL_VERSION + 7)
    frame = _corrupt(encode(MessageType.PING), 4, bumped)

    with pytest.raises(ProtocolError) as excinfo:
        decode(frame)

    message = str(excinfo.value)
    assert str(PROTOCOL_VERSION + 7) in message
    assert str(PROTOCOL_VERSION) in message
    assert "reload" in message


def test_rejects_truncated_header() -> None:
    frame = encode(MessageType.SET_CONFIG, {"vertex_count": 4096, "rosy": 4})
    header_len = _FIXED_HEADER.unpack_from(frame, 0)[3]
    assert header_len > 4  # otherwise the cut below would not land in the header

    with pytest.raises(ProtocolError, match="truncated inside its header"):
        decode(frame[: _FIXED_HEADER.size + header_len - 4])


def test_rejects_header_length_that_breaks_alignment() -> None:
    frame = encode(MessageType.SET_CONFIG, {"rosy": 4})
    header_len = _FIXED_HEADER.unpack_from(frame, 0)[3]

    with pytest.raises(ProtocolError, match="not a multiple of 4"):
        decode(_corrupt(frame, 8, struct.pack("<I", header_len + 1)))


def test_rejects_truncated_payload() -> None:
    frame = encode(
        MessageType.STROKE, {}, {"points": np.zeros((32, 3), dtype=np.float32)}
    )

    with pytest.raises(ProtocolError, match="truncated inside array 'points'"):
        decode(frame[:-4])


def test_rejects_non_json_header() -> None:
    frame = _corrupt(encode(MessageType.PING, {"a": 1}), _FIXED_HEADER.size, b"{{{{")

    with pytest.raises(ProtocolError, match="not valid JSON"):
        decode(frame)


def test_magic_is_the_ascii_tag() -> None:
    """A frame is recognisable in a packet dump, which is half of why it exists."""
    assert struct.pack("<I", MAGIC) == b"IMBS"


def test_rejects_header_that_is_not_an_object() -> None:
    with pytest.raises(ProtocolError, match="must be a JSON object"):
        decode(_handmade([{"name": "q", "dtype": "f32", "shape": [1]}]))


@pytest.mark.parametrize(
    "descriptor",
    [
        {"dtype": "f32", "shape": [1]},  # no name
        {"name": "q", "shape": [1]},  # no dtype
        {"name": "q", "dtype": "f32"},  # no shape
        {"name": "q", "dtype": "f64", "shape": [1]},  # a dtype the wire has no room for
        {"name": "q", "dtype": "f32", "shape": ["two"]},  # shape is not numeric
    ],
    ids=["no-name", "no-dtype", "no-shape", "wide-dtype", "non-numeric-shape"],
)
def test_rejects_malformed_array_descriptor(descriptor: Dict[str, Any]) -> None:
    with pytest.raises(ProtocolError, match="malformed array descriptor"):
        decode(_handmade({"arrays": [descriptor]}, b"\0" * 16))


def test_rejects_negative_array_dimension() -> None:
    """A pair of negative extents multiplies out to a believable element count,
    so the guard has to sit on the dimensions and not just on the product."""
    frame = _handmade(
        {"arrays": [{"name": "q", "dtype": "f32", "shape": [-2, -3]}]}, b"\0" * 24
    )

    with pytest.raises(ProtocolError, match="negative dimension"):
        decode(frame)


# ---------------------------------------------------------------------------
#  The JavaScript half of the contract
# ---------------------------------------------------------------------------

#: Driver for ``static/protocol.js``: decode the frame Python wrote, then
#: re-encode the result.  Byte equality between the two frames is what pins the
#: header padding, the descriptor order and every payload offset to one layout;
#: the summary on stdout exists so a failure names the array that went wrong.
_BRIDGE_JS = """\
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [modulePath, inputPath, outputPath] = process.argv.slice(2);
const { decode, encode } = await import(pathToFileURL(modulePath).href);

const raw = readFileSync(inputPath);
const frame = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);

let message;
try {
  message = decode(frame);
} catch (err) {
  process.stdout.write(
    JSON.stringify({ error: err.constructor.name, message: err.message })
  );
  process.exit(0);
}

const arrays = {};
for (const [name, entry] of Object.entries(message.arrays)) {
  arrays[name] = { dtype: entry.data.constructor.name, shape: entry.shape };
}
const rebuilt = encode(message.type, message.header, message.arrays);
writeFileSync(outputPath, Buffer.from(rebuilt));
process.stdout.write(
  JSON.stringify({ type: message.type, header: message.header, arrays })
);
"""


def _through_javascript(
    tmp_path: Path, frame: bytes
) -> Tuple[Dict[str, Any], Optional[bytes]]:
    """Run ``frame`` through protocol.js, returning its report and its output.

    The output frame is None when the JavaScript decoder rejected the input, in
    which case the report carries the error class and message instead.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so the browser half cannot be run")

    bridge = tmp_path / "bridge.mjs"
    bridge.write_text(_BRIDGE_JS, encoding="utf-8")
    inbound = tmp_path / "inbound.bin"
    inbound.write_bytes(frame)
    outbound = tmp_path / "outbound.bin"

    result = subprocess.run(
        [node, str(bridge), str(PROTOCOL_JS), str(inbound), str(outbound)],
        capture_output=True,
        text=True,
        # Node writes UTF-8; without this Windows would decode it as cp1252 and
        # quietly mangle exactly the non-ASCII headers this test exists to check.
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    rebuilt = outbound.read_bytes() if outbound.exists() else None
    return json.loads(result.stdout), rebuilt


def test_javascript_reproduces_a_python_frame_byte_for_byte(tmp_path: Path) -> None:
    """The contract only holds if both halves agree on the exact bytes."""
    arrays: Dict[str, np.ndarray] = {
        "vertices": np.linspace(-1.0, 1.0, 30, dtype=np.float32).reshape(10, 3),
        "faces": np.arange(24, dtype=np.uint32).reshape(6, 4),
        "shifts": np.array([[-1, 0], [2, -3]], dtype=np.int32),
        "empty": np.empty((0, 3), dtype=np.float32),
    }
    # A non-ASCII value is the one place ensure_ascii=False has to match
    # JSON.stringify, and an odd length walks the header off a 4-byte boundary.
    header = {"name": "crête", "level": -1, "pure_quad": False}
    frame = encode(MessageType.EXTRACTED, header, arrays)

    report, rebuilt = _through_javascript(tmp_path, frame)

    assert report["type"] == MessageType.EXTRACTED
    assert report["header"] == header
    assert {name: entry["shape"] for name, entry in report["arrays"].items()} == {
        name: list(array.shape) for name, array in arrays.items()
    }
    assert {name: entry["dtype"] for name, entry in report["arrays"].items()} == {
        "vertices": "Float32Array",
        "faces": "Uint32Array",
        "shifts": "Int32Array",
        "empty": "Float32Array",
    }
    assert rebuilt == frame

    # ...and the frame JavaScript built has to come back through Python intact.
    message = decode(rebuilt)
    assert message.header == header
    for name, expected in arrays.items():
        assert message.arrays[name].dtype == expected.dtype, name
        assert np.array_equal(message.arrays[name], expected), name


def test_javascript_rejects_a_truncated_frame(tmp_path: Path) -> None:
    frame = encode(
        MessageType.STROKE, {}, {"points": np.zeros((8, 3), dtype=np.float32)}
    )

    report, rebuilt = _through_javascript(tmp_path, frame[:-8])

    assert rebuilt is None
    assert report["error"] == "ProtocolError"
    assert "truncated inside array" in report["message"]
    assert "points" in report["message"]


def test_javascript_rejects_a_future_protocol_version(tmp_path: Path) -> None:
    """Both halves must tell the user to reload rather than misread the frame."""
    bumped = struct.pack("<H", PROTOCOL_VERSION + 7)
    frame = _corrupt(encode(MessageType.PING), 4, bumped)

    report, rebuilt = _through_javascript(tmp_path, frame)

    assert rebuilt is None
    assert report["error"] == "ProtocolError"
    assert str(PROTOCOL_VERSION + 7) in report["message"]
    assert "reload" in report["message"]
