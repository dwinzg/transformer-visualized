/** A float32 tensor read from a safetensors file. `data` may share memory with the file. */
export interface Tensor {
  readonly shape: readonly number[];
  readonly data: Float32Array;
}

export interface SafetensorsFile {
  readonly tensors: ReadonlyMap<string, Tensor>;
  readonly metadata: Readonly<Record<string, string>>;
}

const IS_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/**
 * Parses a safetensors file: an 8-byte little-endian header length, a JSON header, then raw
 * little-endian tensor bytes (https://github.com/huggingface/safetensors). Only F32 is supported.
 * The returned tensors' `data` may be views into `buffer`, so it must not be transferred (for
 * example to a Worker) or mutated afterward.
 */
export function parseSafetensors(buffer: ArrayBuffer): SafetensorsFile {
  if (!IS_LITTLE_ENDIAN) throw new Error('safetensors: big-endian platforms are not supported');
  if (buffer.byteLength < 8) {
    throw new Error('safetensors: file is too short to hold a header length');
  }
  const headerLength = new DataView(buffer).getBigUint64(0, true);
  if (headerLength > BigInt(buffer.byteLength - 8)) {
    throw new Error(
      `safetensors: header length ${headerLength} exceeds the file size ${buffer.byteLength}`,
    );
  }
  const dataStart = 8 + Number(headerLength);
  const dataLength = buffer.byteLength - dataStart;
  const header = parseHeader(new Uint8Array(buffer, 8, Number(headerLength)));
  let metadata: Record<string, string> = {};
  const entries: { name: string; shape: number[]; begin: number; end: number }[] = [];
  for (const [name, entry] of Object.entries(header)) {
    if (name === '__metadata__') {
      metadata = parseMetadata(entry);
      continue;
    }
    const { shape, begin, end } = parseEntry(name, entry);
    const count = shape.reduce((product, size) => product * size, 1);
    if (end - begin !== count * 4) {
      throw new Error(
        `safetensors: tensor "${name}" has ${end - begin} bytes but its shape needs ${count * 4}`,
      );
    }
    if (end > dataLength) {
      throw new Error(
        `safetensors: tensor "${name}" ends at byte ${end}, past the end of the data (${dataLength})`,
      );
    }
    entries.push({ name, shape, begin, end });
  }
  // Model files are trusted exports, but a hand-edited file could alias two weights onto the
  // same bytes. Sort by start, then end, so an empty tensor sorts before a range starting at the
  // same byte whatever the header order, and require each range to begin at or after the previous end.
  const byBegin = [...entries].sort((a, b) => a.begin - b.begin || a.end - b.end);
  for (let i = 1; i < byBegin.length; i++) {
    if (byBegin[i].begin < byBegin[i - 1].end) {
      throw new Error(`safetensors: tensor "${byBegin[i].name}" overlaps "${byBegin[i - 1].name}"`);
    }
  }
  const tensors = new Map<string, Tensor>();
  for (const { name, shape, begin } of entries) {
    const count = shape.reduce((product, size) => product * size, 1);
    const offset = dataStart + begin;
    // Float32Array views need 4-byte alignment; copy when the header leaves the data unaligned.
    const data =
      offset % 4 === 0
        ? new Float32Array(buffer, offset, count)
        : new Float32Array(buffer.slice(offset, offset + count * 4));
    tensors.set(name, { shape, data });
  }
  return { tensors, metadata };
}

function parseHeader(bytes: Uint8Array): Record<string, unknown> {
  let header: unknown;
  try {
    header = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  } catch (cause) {
    throw new Error('safetensors: header is not valid UTF-8 JSON', { cause });
  }
  if (typeof header !== 'object' || header === null || Array.isArray(header)) {
    throw new Error('safetensors: header must be a JSON object');
  }
  return header as Record<string, unknown>;
}

function parseMetadata(entry: unknown): Record<string, string> {
  if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
    throw new Error('safetensors: __metadata__ must be an object');
  }
  const metadata: Record<string, string> = {};
  for (const [key, value] of Object.entries(entry)) {
    if (typeof value !== 'string') {
      throw new Error(`safetensors: metadata value "${key}" must be a string`);
    }
    metadata[key] = value;
  }
  return metadata;
}

function parseEntry(name: string, entry: unknown): { shape: number[]; begin: number; end: number } {
  if (typeof entry !== 'object' || entry === null) {
    throw new Error(`safetensors: tensor "${name}" has no description`);
  }
  const { dtype, shape, data_offsets: offsets } = entry as Record<string, unknown>;
  if (dtype !== 'F32') {
    throw new Error(
      `safetensors: tensor "${name}" has dtype ${String(dtype)}; only F32 is supported`,
    );
  }
  if (!Array.isArray(shape) || !shape.every(isNonNegativeInteger)) {
    throw new Error(`safetensors: tensor "${name}" has an invalid shape`);
  }
  if (
    !Array.isArray(offsets) ||
    offsets.length !== 2 ||
    !offsets.every(isNonNegativeInteger) ||
    offsets[0] > offsets[1]
  ) {
    throw new Error(`safetensors: tensor "${name}" has invalid data offsets`);
  }
  return { shape, begin: offsets[0], end: offsets[1] };
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}
