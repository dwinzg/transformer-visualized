import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const FIXTURES = new URL('../fixtures/', import.meta.url);

/** Reads a file under test/fixtures into its own ArrayBuffer. */
export function readFixture(relativePath: string): ArrayBuffer {
  return new Uint8Array(readFileSync(fileURLToPath(new URL(relativePath, FIXTURES)))).buffer;
}

export function readFixtureJson<T>(relativePath: string): T {
  return JSON.parse(new TextDecoder().decode(readFixture(relativePath))) as T;
}
