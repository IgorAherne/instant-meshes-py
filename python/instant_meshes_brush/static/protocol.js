/**
 * Binary WebSocket framing -- the JavaScript twin of protocol.py.
 *
 * Both files must change together, and PROTOCOL_VERSION must be bumped when
 * they do; the server rejects mismatched versions with an explicit error rather
 * than misreading a frame.
 *
 * Frame layout (little-endian):
 *   0   u32  magic = 0x53424D49
 *   4   u16  version
 *   6   u16  message type
 *   8   u32  header length (multiple of 4)
 *   12  ..   UTF-8 JSON header, space-padded to a multiple of 4
 *   ..  ..   array payloads back to back, in header.arrays order
 *
 * Decoded arrays are *views* onto the received ArrayBuffer, so reading a 2.4 MB
 * field update costs no copy at all.
 */

export const MAGIC = 0x53424d49;
export const PROTOCOL_VERSION = 2;

export const MessageType = Object.freeze({
  // client -> server
  HELLO: 1,
  LOAD_MESH: 2,
  SET_CONFIG: 3,
  STROKE: 4,
  ERASE_STROKE: 5,
  SOLVE: 6,
  STOP: 7,
  EXTRACT: 8,
  EXPORT: 9,
  CLEAR_STROKES: 10,
  SUBSCRIBE: 11,
  PING: 12,
  UNWRAP: 13,

  // server -> client
  GEOMETRY: 100,
  FIELD: 101,
  STROKE_RESULT: 102,
  SINGULARITIES: 103,
  EXTRACTED: 104,
  STATUS: 105,
  ERROR: 106,
  PROGRESS: 107,
  EXPORT_READY: 108,
  STROKE_LIST: 109,
  PONG: 110,
  UV_LAYOUT: 111,
});

const TYPED_ARRAYS = {
  f32: Float32Array,
  u32: Uint32Array,
  i32: Int32Array,
};

const DTYPE_OF = new Map([
  [Float32Array, 'f32'],
  [Uint32Array, 'u32'],
  [Int32Array, 'i32'],
]);

const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8');

export class ProtocolError extends Error {}

/**
 * Build a frame.
 *
 * @param {number} type      a MessageType value
 * @param {object} [header]  JSON-serialisable metadata
 * @param {object} [arrays]  name -> {data: TypedArray, shape: number[]}
 *                           (or a bare TypedArray, treated as 1-D)
 * @returns {ArrayBuffer}
 */
export function encode(type, header = {}, arrays = null) {
  const descriptors = [];
  const buffers = [];

  for (const [name, value] of Object.entries(arrays || {})) {
    const data = ArrayBuffer.isView(value) ? value : value.data;
    const dtype = DTYPE_OF.get(data.constructor);
    if (!dtype) {
      throw new ProtocolError(
        `array "${name}" has unsupported type ${data.constructor.name}`
      );
    }
    const shape =
      (ArrayBuffer.isView(value) ? null : value.shape) || [data.length];
    let count = 1;
    for (const n of shape) count *= n;
    if (count !== data.length) {
      throw new ProtocolError(
        `array "${name}" shape [${shape}] does not match length ${data.length}`
      );
    }
    descriptors.push({ name, dtype, shape });
    buffers.push(data);
  }

  const headerObj = { ...header };
  if (descriptors.length) headerObj.arrays = descriptors;

  let headerBytes = encoder.encode(JSON.stringify(headerObj));
  const padding = (4 - (headerBytes.length % 4)) % 4;
  if (padding) {
    const padded = new Uint8Array(headerBytes.length + padding);
    padded.set(headerBytes);
    padded.fill(0x20, headerBytes.length);
    headerBytes = padded;
  }

  let total = 12 + headerBytes.length;
  for (const buf of buffers) total += buf.byteLength;

  const out = new ArrayBuffer(total);
  const view = new DataView(out);
  view.setUint32(0, MAGIC, true);
  view.setUint16(4, PROTOCOL_VERSION, true);
  view.setUint16(6, type, true);
  view.setUint32(8, headerBytes.length, true);

  const bytes = new Uint8Array(out);
  bytes.set(headerBytes, 12);

  let offset = 12 + headerBytes.length;
  for (const buf of buffers) {
    bytes.set(
      new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength),
      offset
    );
    offset += buf.byteLength;
  }
  return out;
}

/**
 * Parse a frame. Arrays are zero-copy views onto `buffer`, so keep `buffer`
 * alive (or copy the arrays) for as long as you use them.
 *
 * @param {ArrayBuffer} buffer
 * @returns {{type: number, header: object, arrays: Record<string, {data: TypedArray, shape: number[]}>}}
 */
export function decode(buffer) {
  if (buffer.byteLength < 12) {
    throw new ProtocolError('frame shorter than its fixed header');
  }
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== MAGIC) {
    throw new ProtocolError(`bad magic 0x${magic.toString(16)}`);
  }
  const version = view.getUint16(4, true);
  if (version !== PROTOCOL_VERSION) {
    throw new ProtocolError(
      `protocol version ${version} != ${PROTOCOL_VERSION}; reload the page`
    );
  }
  const type = view.getUint16(6, true);
  const headerLength = view.getUint32(8, true);
  if (headerLength % 4 !== 0) {
    throw new ProtocolError('header length is not a multiple of 4');
  }
  if (buffer.byteLength < 12 + headerLength) {
    throw new ProtocolError('frame truncated inside its header');
  }

  const header = JSON.parse(
    decoder.decode(new Uint8Array(buffer, 12, headerLength))
  );

  const arrays = {};
  let offset = 12 + headerLength;
  for (const descriptor of header.arrays || []) {
    const Ctor = TYPED_ARRAYS[descriptor.dtype];
    if (!Ctor) {
      throw new ProtocolError(`unsupported dtype "${descriptor.dtype}"`);
    }
    let count = 1;
    for (const n of descriptor.shape) count *= n;
    const nbytes = count * Ctor.BYTES_PER_ELEMENT;
    if (buffer.byteLength < offset + nbytes) {
      throw new ProtocolError(`frame truncated inside array "${descriptor.name}"`);
    }
    arrays[descriptor.name] = {
      data: new Ctor(buffer, offset, count),
      shape: descriptor.shape,
    };
    offset += nbytes;
  }
  delete header.arrays;

  return { type, header, arrays };
}
