import { describe, expect, it } from 'vitest';
import { parseSafetensors } from '../src/safetensors';
import { encodeSafetensors, rawSafetensors } from './helpers/safetensors';

const specs = [
  { name: 'a', shape: [2, 2], values: [1, 2, 3, 4] },
  { name: 'b', shape: [3], values: [0.5, -0.25, 8] },
];

describe('parseSafetensors', () => {
  it('reads tensors, shapes and metadata', () => {
    const file = parseSafetensors(encodeSafetensors(specs, { format: 'test', note: 'hi' }));
    expect([...file.tensors.keys()].sort()).toEqual(['a', 'b']);
    expect(file.tensors.get('a')?.shape).toEqual([2, 2]);
    expect(Array.from(file.tensors.get('a')?.data ?? [])).toEqual([1, 2, 3, 4]);
    expect(Array.from(file.tensors.get('b')?.data ?? [])).toEqual([0.5, -0.25, 8]);
    expect(file.metadata).toEqual({ format: 'test', note: 'hi' });
  });

  it('reads data at any alignment', () => {
    for (let padding = 0; padding < 4; padding++) {
      const file = parseSafetensors(encodeSafetensors(specs, undefined, padding));
      expect(Array.from(file.tensors.get('b')?.data ?? [])).toEqual([0.5, -0.25, 8]);
    }
  });

  it('returns empty metadata when the file has none', () => {
    expect(parseSafetensors(encodeSafetensors(specs)).metadata).toEqual({});
  });

  it('keeps -Infinity values', () => {
    const file = parseSafetensors(
      encodeSafetensors([{ name: 'x', shape: [2], values: [-Infinity, 1] }]),
    );
    expect(Array.from(file.tensors.get('x')?.data ?? [])).toEqual([-Infinity, 1]);
  });

  it('rejects a file shorter than the header length field', () => {
    expect(() => parseSafetensors(new ArrayBuffer(4))).toThrow(/too short/);
  });

  it('rejects a header length larger than the file', () => {
    const buffer = new ArrayBuffer(16);
    new DataView(buffer).setBigUint64(0, 100n, true);
    expect(() => parseSafetensors(buffer)).toThrow(/exceeds/);
  });

  it('rejects a header that is not a JSON object', () => {
    expect(() => parseSafetensors(rawSafetensors('{not json'))).toThrow(/not valid UTF-8 JSON/);
    expect(() => parseSafetensors(rawSafetensors('[]'))).toThrow(/must be a JSON object/);
  });

  it('rejects dtypes other than F32', () => {
    const header = JSON.stringify({ x: { dtype: 'F16', shape: [1], data_offsets: [0, 2] } });
    expect(() => parseSafetensors(rawSafetensors(header, 2))).toThrow(/only F32/);
  });

  it('rejects a byte range that does not match the shape', () => {
    const header = JSON.stringify({ x: { dtype: 'F32', shape: [2], data_offsets: [0, 4] } });
    expect(() => parseSafetensors(rawSafetensors(header, 4))).toThrow(/shape needs 8/);
  });

  it('rejects a byte range past the end of the data', () => {
    const header = JSON.stringify({ x: { dtype: 'F32', shape: [2], data_offsets: [0, 8] } });
    expect(() => parseSafetensors(rawSafetensors(header, 4))).toThrow(/past the end/);
  });

  it('rejects malformed shapes and offsets', () => {
    const badShape = JSON.stringify({ x: { dtype: 'F32', shape: [-1], data_offsets: [0, 0] } });
    expect(() => parseSafetensors(rawSafetensors(badShape))).toThrow(/invalid shape/);
    const badOffsets = JSON.stringify({ x: { dtype: 'F32', shape: [1], data_offsets: [4, 0] } });
    expect(() => parseSafetensors(rawSafetensors(badOffsets, 4))).toThrow(/invalid data offsets/);
  });

  it('rejects metadata values that are not strings', () => {
    const header = JSON.stringify({ __metadata__: { n: 1 } });
    expect(() => parseSafetensors(rawSafetensors(header))).toThrow(/must be a string/);
  });
});
