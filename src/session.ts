import type { VoxelSurface } from "./surface";

// A saved prep: a small JSON header + the changed voxels of each tooth, gzip-compressed.
const FORMAT = "crown-prep-session";
const VERSION = 1;

export interface SessionTooth {
  name: string;
  nx: number;
  ny: number;
  nz: number;
  marginY: number;
  count: number; // number of changed voxels stored
}

interface SessionHeader {
  format: string;
  version: number;
  savedAt: string;
  teeth: SessionTooth[];
}

export async function saveSession(surfaces: VoxelSurface[]): Promise<Blob> {
  const header: SessionHeader = {
    format: FORMAT,
    version: VERSION,
    savedAt: new Date().toISOString(),
    teeth: [],
  };
  const parts: ArrayBuffer[] = [];

  for (const surface of surfaces) {
    const { grid } = surface;
    const { indices, values } = surface.diff();
    header.teeth.push({
      name: grid.prep?.name ?? "tooth",
      nx: grid.nx,
      ny: grid.ny,
      nz: grid.nz,
      marginY: grid.marginY ?? 0,
      count: indices.length,
    });
    parts.push(indices.buffer as ArrayBuffer, values.buffer as ArrayBuffer);
  }

  // Layout: [header length: 4 bytes][header JSON][indices 1][values 1][indices 2][values 2]...
  const json = new TextEncoder().encode(JSON.stringify(header));
  const length = new Uint32Array([json.length]);
  const raw = new Blob([length.buffer as ArrayBuffer, json.buffer as ArrayBuffer, ...parts]);

  // Compress in the browser (built-in CompressionStream)
  return new Response(raw.stream().pipeThrough(new CompressionStream("gzip"))).blob();
}

// Restores the cut surfaces. Returns the header so the caller can restore margins.
export async function loadSession(file: Blob, surfaces: VoxelSurface[]): Promise<SessionTooth[]> {
  // Unzip and read the header. Any failure here means it isn't one of our files.
  let buffer: ArrayBuffer;
  let length: number;
  let header: SessionHeader;
  try {
    buffer = await new Response(file.stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    length = new DataView(buffer).getUint32(0, true);
    header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, length))) as SessionHeader;
  } catch {
    throw new Error("This is not a crown prep session file");
  }
  if (header.format !== FORMAT || header.version !== VERSION) throw new Error("This is not a crown prep session file");

  // Check the file matches the teeth loaded now (same models, same voxel grid) BEFORE changing anything
  if (header.teeth.length !== surfaces.length) throw new Error("This session has a different number of teeth");
  header.teeth.forEach((tooth, n) => {
    const { grid } = surfaces[n];
    if (tooth.nx !== grid.nx || tooth.ny !== grid.ny || tooth.nz !== grid.nz) {
      throw new Error(`"${tooth.name}" was saved with a different tooth model`);
    }
  });

  // A truncated or damaged file must not half-load (it would write garbage into the teeth)
  const expectedBytes = 4 + length + header.teeth.reduce((sum, tooth) => sum + tooth.count * 8, 0);
  if (buffer.byteLength !== expectedBytes) throw new Error("The file is incomplete or damaged");

  // Read and check every tooth first...
  let offset = 4 + length;
  const diffs = header.teeth.map((tooth, n) => {
    const bytes = tooth.count * 4;
    // slice() copies into a fresh buffer, so the typed arrays are correctly aligned
    const indices = new Uint32Array(buffer.slice(offset, offset + bytes));
    offset += bytes;
    const values = new Float32Array(buffer.slice(offset, offset + bytes));
    offset += bytes;

    const voxels = surfaces[n].grid.data.length;
    for (let i = 0; i < indices.length; i++) {
      if (indices[i] >= voxels || !Number.isFinite(values[i])) throw new Error("The file is damaged");
    }
    return { indices, values };
  });

  // ...then apply, so a bad file changes nothing
  diffs.forEach(({ indices, values }, n) => surfaces[n].applyDiff(indices, values));
  return header.teeth;
}
