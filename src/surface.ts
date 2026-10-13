import * as THREE from "three";
import { type SDFGrid, type VoxelRange, type BurPose, burRange, carveBur, voxelIndex } from "./sdf";
import { meshRegion } from "./marchingCubes";

const CHUNK = 16; // cells per chunk along each axis
const BLOCK = 16; // voxels per side of an undo block
export const MAX_UNDO = 30; // strokes kept in the undo history, per tooth

// One stroke = the voxel values of every block it touched, saved just before it first touched them.
// Undo and redo both SWAP these with the current values, so the same record works both ways.
interface Stroke {
  blocks: Map<number, Float32Array>;
}

// The visible surface of a voxel tooth, split into chunks so carving only re-meshes what changed
export class VoxelSurface {
  readonly group = new THREE.Group();
  readonly grid: SDFGrid;
  readonly material: THREE.Material;
  private readonly chunks = new Map<number, THREE.Mesh>();
  private readonly dirty = new Set<number>();
  private readonly cx: number;
  private readonly cy: number;
  private readonly cz: number;

  // Undo / redo / reset
  private readonly original: Float32Array; // the uncut tooth
  private readonly undoStack: Stroke[] = [];
  private readonly redoStack: Stroke[] = [];
  private stroke: Stroke | null = null;
  private readonly bx: number;
  private readonly by: number;

  constructor(grid: SDFGrid, material: THREE.Material) {
    this.grid = grid;
    this.material = material;
    this.original = grid.data.slice();
    this.bx = Math.ceil(grid.nx / BLOCK);
    this.by = Math.ceil(grid.ny / BLOCK);

    // A grid of n voxels has n - 1 cells between them
    this.cx = Math.ceil((grid.nx - 1) / CHUNK);
    this.cy = Math.ceil((grid.ny - 1) / CHUNK);
    this.cz = Math.ceil((grid.nz - 1) / CHUNK);

    for (let c = 0; c < this.cx * this.cy * this.cz; c++) this.dirty.add(c);
    this.rebuild();
  }

  // Mark every chunk whose cells touch the changed voxels
  markDirty(range: VoxelRange): void {
    // A voxel is a corner of the cells on both sides of it, hence the - 1
    const a0 = Math.max(0, Math.floor((range.i0 - 1) / CHUNK));
    const b0 = Math.max(0, Math.floor((range.j0 - 1) / CHUNK));
    const c0 = Math.max(0, Math.floor((range.k0 - 1) / CHUNK));
    const a1 = Math.min(this.cx - 1, Math.floor(range.i1 / CHUNK));
    const b1 = Math.min(this.cy - 1, Math.floor(range.j1 / CHUNK));
    const c1 = Math.min(this.cz - 1, Math.floor(range.k1 / CHUNK));

    for (let c = c0; c <= c1; c++)
      for (let b = b0; b <= b1; b++)
        for (let a = a0; a <= a1; a++) this.dirty.add(a + this.cx * (b + this.cy * c));
  }

  // Re-mesh every dirty chunk. Returns how many chunks were rebuilt.
  rebuild(): number {
    const { nx, ny, nz } = this.grid;
    const count = this.dirty.size;

    for (const key of this.dirty) {
      // Decode the chunk key back into (a, b, c)
      const a = key % this.cx;
      const b = Math.floor(key / this.cx) % this.cy;
      const c = Math.floor(key / (this.cx * this.cy));

      const i0 = a * CHUNK;
      const j0 = b * CHUNK;
      const k0 = c * CHUNK;
      const geometry = meshRegion(
        this.grid,
        i0, j0, k0,
        Math.min(i0 + CHUNK, nx - 1), Math.min(j0 + CHUNK, ny - 1), Math.min(k0 + CHUNK, nz - 1),
      );

      // Throw away the old chunk mesh and free its GPU memory
      const old = this.chunks.get(key);
      if (old) {
        this.group.remove(old);
        old.geometry.dispose();
        this.chunks.delete(key);
      }

      if (geometry) {
        geometry.computeBoundsTree();
        const mesh = new THREE.Mesh(geometry, this.material);
        mesh.userData.surface = this;
        this.group.add(mesh);
        this.chunks.set(key, mesh);
      }
    }

    this.dirty.clear();
    return count;
  }

  // Re-mesh everything, e.g. after the colour mode changed (colours are stored in the vertices)
  rebuildAll(): void {
    // Every chunk, not just the non-empty ones: after an undo or reset, empty ones may have surface again
    for (let c = 0; c < this.cx * this.cy * this.cz; c++) this.dirty.add(c);
    this.rebuild();
  }

  // --- Carving with history ---------------------------------------------------

  beginStroke(): void {
    this.stroke = { blocks: new Map() };
  }

  // Carve one bur position. During a stroke, first save any block it is about to touch.
  carve(bur: BurPose): void {
    const range = burRange(this.grid, bur);
    if (!range) return;
    if (this.stroke) this.backup(this.stroke, range);

    const changed = carveBur(this.grid, bur);
    if (changed) this.markDirty(changed);
  }

  // Returns true if the stroke changed something (and was added to the history)
  endStroke(): boolean {
    const recorded = this.stroke !== null && this.stroke.blocks.size > 0;
    if (recorded) {
      this.undoStack.push(this.stroke!);
      if (this.undoStack.length > MAX_UNDO) this.undoStack.shift(); // forget the oldest
    }
    this.stroke = null;
    return recorded;
  }

  // A new cut makes the old "future" invalid
  clearRedo(): void {
    this.redoStack.length = 0;
  }

  get canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  get canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  undo(): boolean {
    const stroke = this.undoStack.pop();
    if (!stroke) return false;
    this.swap(stroke);
    this.redoStack.push(stroke);
    return true;
  }

  redo(): boolean {
    const stroke = this.redoStack.pop();
    if (!stroke) return false;
    this.swap(stroke);
    this.undoStack.push(stroke);
    return true;
  }

  // Back to the uncut tooth
  reset(): void {
    this.grid.data.set(this.original);
    this.undoStack.length = 0;
    this.redoStack.length = 0;
    this.rebuildAll();
  }

  // --- Save / load --------------------------------------------------------------

  // Only the voxels that differ from the uncut tooth - a prep changes a small fraction of the grid
  diff(): { indices: Uint32Array; values: Float32Array } {
    const { data } = this.grid;
    const indices: number[] = [];
    for (let id = 0; id < data.length; id++) if (data[id] !== this.original[id]) indices.push(id);
    return {
      indices: Uint32Array.from(indices),
      values: Float32Array.from(indices, (id) => data[id]),
    };
  }

  applyDiff(indices: Uint32Array, values: Float32Array): void {
    const { data } = this.grid;
    data.set(this.original);
    for (let n = 0; n < indices.length; n++) data[indices[n]] = values[n];
    this.undoStack.length = 0;
    this.redoStack.length = 0;
    this.rebuildAll();
  }

  // --- Undo internals -----------------------------------------------------------

  // Visit every voxel of one block: id = position in the grid, local = position in the saved copy
  private forEachInBlock(key: number, visit: (id: number, local: number) => void): VoxelRange {
    const { nx, ny, nz } = this.grid;
    const a = key % this.bx;
    const b = Math.floor(key / this.bx) % this.by;
    const c = Math.floor(key / (this.bx * this.by));
    const i0 = a * BLOCK;
    const j0 = b * BLOCK;
    const k0 = c * BLOCK;
    const i1 = Math.min(i0 + BLOCK, nx) - 1;
    const j1 = Math.min(j0 + BLOCK, ny) - 1;
    const k1 = Math.min(k0 + BLOCK, nz) - 1;

    for (let k = k0; k <= k1; k++)
      for (let j = j0; j <= j1; j++)
        for (let i = i0; i <= i1; i++)
          visit(voxelIndex(this.grid, i, j, k), (i - i0) + BLOCK * ((j - j0) + BLOCK * (k - k0)));

    return { i0, j0, k0, i1, j1, k1 };
  }

  private backup(stroke: Stroke, range: VoxelRange): void {
    const { data } = this.grid;
    for (let c = Math.floor(range.k0 / BLOCK); c <= Math.floor(range.k1 / BLOCK); c++)
      for (let b = Math.floor(range.j0 / BLOCK); b <= Math.floor(range.j1 / BLOCK); b++)
        for (let a = Math.floor(range.i0 / BLOCK); a <= Math.floor(range.i1 / BLOCK); a++) {
          const key = a + this.bx * (b + this.by * c);
          if (stroke.blocks.has(key)) continue; // already saved earlier in this stroke
          const saved = new Float32Array(BLOCK * BLOCK * BLOCK);
          this.forEachInBlock(key, (id, local) => (saved[local] = data[id]));
          stroke.blocks.set(key, saved);
        }
  }

  private swap(stroke: Stroke): void {
    const { data } = this.grid;
    for (const [key, saved] of stroke.blocks) {
      const range = this.forEachInBlock(key, (id, local) => {
        const current = data[id];
        data[id] = saved[local];
        saved[local] = current;
      });
      this.markDirty(range);
    }
    this.rebuild();
  }

  // The chunk meshes currently making up the surface (used by the prep report)
  meshes(): IterableIterator<THREE.Mesh> {
    return this.chunks.values();
  }

  triangleCount(): number {
    let total = 0;
    for (const mesh of this.chunks.values()) total += mesh.geometry.index!.count / 3;
    return total;
  }
}
