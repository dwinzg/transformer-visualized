export interface TensorSpec {
  name: string;
  shape: number[];
  values: number[];
}

/**
 * Encodes float32 tensors as a safetensors file. `headerPadding` appends spaces to the header,
 * which shifts where the tensor data starts (used to test unaligned data).
 */
export function encodeSafetensors(
  specs: readonly TensorSpec[],
  metadata?: Record<string, string>,
  headerPadding = 0,
): ArrayBuffer {
  const header: Record<string, unknown> = {};
  if (metadata !== undefined) header.__metadata__ = metadata;
  let offset = 0;
  for (const spec of specs) {
    const end = offset + spec.values.length * 4;
    header[spec.name] = { dtype: 'F32', shape: spec.shape, data_offsets: [offset, end] };
    offset = end;
  }
  const headerBytes = new TextEncoder().encode(JSON.stringify(header) + ' '.repeat(headerPadding));
  const buffer = new ArrayBuffer(8 + headerBytes.length + offset);
  const view = new DataView(buffer);
  view.setBigUint64(0, BigInt(headerBytes.length), true);
  new Uint8Array(buffer, 8, headerBytes.length).set(headerBytes);
  let position = 8 + headerBytes.length;
  for (const spec of specs) {
    for (const value of spec.values) {
      view.setFloat32(position, value, true);
      position += 4;
    }
  }
  return buffer;
}

/** A file with an arbitrary header text followed by `dataBytes` zero bytes. */
export function rawSafetensors(headerText: string, dataBytes = 0): ArrayBuffer {
  const headerBytes = new TextEncoder().encode(headerText);
  const buffer = new ArrayBuffer(8 + headerBytes.length + dataBytes);
  new DataView(buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  new Uint8Array(buffer, 8, headerBytes.length).set(headerBytes);
  return buffer;
}
