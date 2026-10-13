import * as THREE from "three";
import type { MeshBVH } from "three-mesh-bvh";
import type { PrepSpec } from "./prep";

export interface SDFGrid {
  nx: number; // voxels along X
  ny: number; // voxels along Y
  nz: number; // voxels along Z
  voxelSize: number; // mm per voxel
  origin: THREE.Vector3; // position of voxel (0, 0, 0)
  data: Float32Array; // signed distance in mm; negative = inside the tooth
  depth?: Float32Array; // mm below the ORIGINAL surface (set by buildDepthField)
  region?: Uint8Array; // which target region each voxel belongs to (set by classifyRegions)
  prep?: PrepSpec; // the reduction targets for this tooth
  nearestFace?: Int32Array; // nearest original triangle per voxel (kept so regions can be redone)
  marginY?: number; // current finish line height, mm
}

export function voxelIndex(grid: SDFGrid, i: number, j: number, k: number): number {
  return i + grid.nx * (j + grid.ny * k);
}

// Watertight = every edge is shared by exactly two triangles
export function checkWatertight(geometry: THREE.BufferGeometry) {
  const index = geometry.index;
  if (!index) throw new Error("Geometry has no index - run mergeVertices first");

  const vertexCount = geometry.attributes.position.count;
  const edgeUses = new Map<number, number>();

  const addEdge = (a: number, b: number) => {
    const key = a < b ? a * vertexCount + b : b * vertexCount + a;
    edgeUses.set(key, (edgeUses.get(key) ?? 0) + 1);
  };

  for (let f = 0; f < index.count; f += 3) {
    const a = index.getX(f);
    const b = index.getX(f + 1);
    const c = index.getX(f + 2);
    addEdge(a, b);
    addEdge(b, c);
    addEdge(c, a);
  }

  let boundary = 0;
  let nonManifold = 0;
  for (const uses of edgeUses.values()) {
    if (uses === 1) boundary++;
    else if (uses > 2) nonManifold++;
  }

  return {
    edges: edgeUses.size,
    boundary,
    nonManifold,
    watertight: boundary === 0 && nonManifold === 0,
  };
}

export function voxelize(geometry: THREE.BufferGeometry, voxelSize: number, padding = 3): SDFGrid {
  const bvh = geometry.boundsTree as MeshBVH | undefined;
  if (!bvh) throw new Error("Geometry has no BVH - call computeBoundsTree first");

  // Size the grid to the tooth, plus a few empty voxels of padding on every side
  geometry.computeBoundingBox();
  const box = geometry.boundingBox!;
  const size = new THREE.Vector3();
  box.getSize(size);

  const nx = Math.ceil(size.x / voxelSize) + 2 * padding + 1;
  const ny = Math.ceil(size.y / voxelSize) + 2 * padding + 1;
  const nz = Math.ceil(size.z / voxelSize) + 2 * padding + 1;
  const origin = box.min.clone().subScalar(padding * voxelSize);

  const data = new Float32Array(nx * ny * nz);
  const grid: SDFGrid = { nx, ny, nz, voxelSize, origin, data };

  const far = 2 * voxelSize; // placeholder distance for voxels away from the surface

  // PASS 1: one ray per row along +X, flipping inside/outside at each hit
  const ray = new THREE.Ray(new THREE.Vector3(), new THREE.Vector3(1, 0, 0));
  const hitXs: number[] = [];
  let oddRows = 0; // rows with an odd number of hits = parity broken by a mesh defect

  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      // tiny jitter so rays never run exactly along a triangle edge
      const y = origin.y + j * voxelSize + 1e-4;
      const z = origin.z + k * voxelSize + 1.7e-4;
      ray.origin.set(origin.x - voxelSize, y, z);

      const hits = bvh.raycast(ray, THREE.DoubleSide);
      hitXs.length = 0;
      for (const hit of hits) hitXs.push(hit.point.x);
      hitXs.sort((a, b) => a - b);

      let h = 0;
      let inside = false;
      for (let i = 0; i < nx; i++) {
        const x = origin.x + i * voxelSize;
        while (h < hitXs.length && hitXs[h] < x) {
          inside = !inside;
          h++;
        }
        data[voxelIndex(grid, i, j, k)] = inside ? -far : far;
      }
      if (inside) oddRows++; // a ray must leave the tooth as many times as it enters
    }
  }
  if (oddRows > 0) console.warn(`voxelize: ${oddRows} rows with broken inside/outside parity`);

  // PASS 2: exact distance only for voxels next to the surface (narrow band)
  const sx = 1; // step to the next voxel in i
  const sy = nx; // step to the next voxel in j
  const sz = nx * ny; // step to the next voxel in k
  const p = new THREE.Vector3();
  const target = { point: new THREE.Vector3(), distance: 0, faceIndex: 0 };
  let exactCount = 0;

  for (let k = 1; k < nz - 1; k++) {
    for (let j = 1; j < ny - 1; j++) {
      for (let i = 1; i < nx - 1; i++) {
        const id = voxelIndex(grid, i, j, k);
        const inside = data[id] < 0;

        const nearSurface =
          (data[id - sx] < 0) !== inside ||
          (data[id + sx] < 0) !== inside ||
          (data[id - sy] < 0) !== inside ||
          (data[id + sy] < 0) !== inside ||
          (data[id - sz] < 0) !== inside ||
          (data[id + sz] < 0) !== inside;
        if (!nearSurface) continue;

        p.set(origin.x + i * voxelSize, origin.y + j * voxelSize, origin.z + k * voxelSize);
        bvh.closestPointToPoint(p, target);
        data[id] = inside ? -target.distance : target.distance;
        exactCount++;
      }
    }
  }

  console.log(
    `${nx} x ${ny} x ${nz} = ${data.length.toLocaleString()} voxels, ${exactCount.toLocaleString()} exact, ${oddRows} broken rows`,
  );
  return grid;
}

// Debug view: blue = thin shell just inside the surface, orange = middle Z cross-section
export function createVoxelDebugPoints(grid: SDFGrid): THREE.Points {
  const { nx, ny, nz, voxelSize, origin, data } = grid;
  const midK = Math.floor(nz / 2);
  const positions: number[] = [];
  const colors: number[] = [];
  const shellColor = new THREE.Color(0x4fc3f7);
  const sliceColor = new THREE.Color(0xff9e40);

  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const d = data[voxelIndex(grid, i, j, k)];
        const isShell = d <= 0 && d > -voxelSize;
        const isSlice = k === midK && d < 0;
        if (!isShell && !isSlice) continue;

        positions.push(origin.x + i * voxelSize, origin.y + j * voxelSize, origin.z + k * voxelSize);
        const c = isSlice ? sliceColor : shellColor;
        colors.push(c.r, c.g, c.b);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({ size: voxelSize, vertexColors: true });
  const points = new THREE.Points(geometry, material);
  points.visible = false;
  return points;
}

// ---------------------------------------------------------------------------
// Carving
// ---------------------------------------------------------------------------

export interface VoxelRange {
  i0: number;
  j0: number;
  k0: number;
  i1: number;
  j1: number;
  k1: number;
}

export interface BurPose {
  tip: THREE.Vector3; // centre of the rounded tip, in grid (tooth) space
  axis: THREE.Vector3; // unit vector from the tip towards the shank
  length: number; // cutting length in mm
  tipRadius: number; // radius at the tip, mm
  topRadius: number; // radius at the top of the cutting head, mm
  // Optional: also remove everything the bur would pass through if slid this far along sweep
  // (unit vector at right angles to axis). Used by axial mode to clear the material outside the bur.
  sweep?: THREE.Vector3;
  sweepLength?: number;
}

// Reused scratch objects: carveBur runs every frame while drilling
const _end = new THREE.Vector3();
const _p = new THREE.Vector3();
const _q = new THREE.Vector3();
const _box = new THREE.Box3();
const _corner = new THREE.Vector3();

// The voxels a bur pose could touch: its bounding box plus the narrow-band width
export function burRange(grid: SDFGrid, bur: BurPose): VoxelRange | null {
  const { nx, ny, nz, voxelSize, origin } = grid;

  // Only voxels near the bur can change: its bounding box plus the band width
  _end.copy(bur.tip).addScaledVector(bur.axis, bur.length);
  _box.makeEmpty().expandByPoint(bur.tip).expandByPoint(_end);
  const sweepLength = bur.sweep ? (bur.sweepLength ?? 0) : 0;
  if (bur.sweep && sweepLength > 0) {
    _box.expandByPoint(_corner.copy(bur.tip).addScaledVector(bur.sweep, sweepLength));
    _box.expandByPoint(_corner.copy(_end).addScaledVector(bur.sweep, sweepLength));
  }
  _box.expandByScalar(Math.max(bur.tipRadius, bur.topRadius) + 2 * voxelSize);

  const i0 = Math.max(1, Math.floor((_box.min.x - origin.x) / voxelSize));
  const j0 = Math.max(1, Math.floor((_box.min.y - origin.y) / voxelSize));
  const k0 = Math.max(1, Math.floor((_box.min.z - origin.z) / voxelSize));
  const i1 = Math.min(nx - 2, Math.ceil((_box.max.x - origin.x) / voxelSize));
  const j1 = Math.min(ny - 2, Math.ceil((_box.max.y - origin.y) / voxelSize));
  const k1 = Math.min(nz - 2, Math.ceil((_box.max.z - origin.z) / voxelSize));
  if (i0 > i1 || j0 > j1 || k0 > k1) return null;
  return { i0, j0, k0, i1, j1, k1 };
}

// Subtract the bur shape from the volume. Returns the voxel range that changed, or null.
export function carveBur(grid: SDFGrid, bur: BurPose): VoxelRange | null {
  const { voxelSize, origin, data } = grid;
  const range = burRange(grid, bur);
  if (!range) return null;
  const { i0, j0, k0, i1, j1, k1 } = range;
  const sweepLength = bur.sweep ? (bur.sweepLength ?? 0) : 0;

  let changed = false;
  for (let k = k0; k <= k1; k++) {
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        _p.set(origin.x + i * voxelSize, origin.y + j * voxelSize, origin.z + k * voxelSize);

        // Distance from this voxel to the bur (negative = inside the bur):
        // find the closest point on the bur's centre line (or the strip it sweeps), then subtract the radius
        _q.subVectors(_p, bur.tip);
        const t = THREE.MathUtils.clamp(_q.dot(bur.axis), 0, bur.length);
        const radius = bur.tipRadius + (bur.topRadius - bur.tipRadius) * (t / bur.length);
        _q.addScaledVector(bur.axis, -t);
        if (bur.sweep && sweepLength > 0) {
          const w = THREE.MathUtils.clamp(_q.dot(bur.sweep), 0, sweepLength);
          _q.addScaledVector(bur.sweep, -w);
        }
        const burDistance = _q.length() - radius;

        // CSG subtraction: tooth minus bur = max(tooth, -bur)
        const id = voxelIndex(grid, i, j, k);
        const carved = Math.max(data[id], -burDistance);
        if (carved !== data[id]) {
          data[id] = carved;
          changed = true;
        }
      }
    }
  }

  return changed ? { i0, j0, k0, i1, j1, k1 } : null;
}

// ---------------------------------------------------------------------------
// Depth below the ORIGINAL surface (for the reduction heatmap)
// ---------------------------------------------------------------------------

// For every voxel inside the uncut tooth: distance (mm) to the original surface. 0 outside.
// Exact nearest points are found for the narrow band, then passed inward voxel by voxel
// ("closest-point propagation"), which is ~100x faster than querying every voxel.
export function buildDepthField(
  geometry: THREE.BufferGeometry,
  grid: SDFGrid,
): { depth: Float32Array; nearestFace: Int32Array } {
  const bvh = geometry.boundsTree as MeshBVH | undefined;
  if (!bvh) throw new Error("Geometry has no BVH - call computeBoundsTree first");

  const { nx, ny, nz, voxelSize, origin, data } = grid;
  const far = 2 * voxelSize;
  const count = data.length;

  const closest = new Float32Array(count * 3); // nearest surface point found so far, per voxel
  const best = new Float32Array(count).fill(Infinity); // squared distance to it
  const nearestFace = new Int32Array(count).fill(-1); // which triangle that point is on

  // 1. Seed: exact nearest points for inside voxels in the narrow band
  const p = new THREE.Vector3();
  const target = { point: new THREE.Vector3(), distance: 0, faceIndex: 0 };
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const id = voxelIndex(grid, i, j, k);
        if (data[id] >= 0 || data[id] <= -far) continue; // only exact, inside voxels
        p.set(origin.x + i * voxelSize, origin.y + j * voxelSize, origin.z + k * voxelSize);
        bvh.closestPointToPoint(p, target);
        closest[id * 3] = target.point.x;
        closest[id * 3 + 1] = target.point.y;
        closest[id * 3 + 2] = target.point.z;
        best[id] = target.distance * target.distance;
        nearestFace[id] = target.faceIndex;
      }
    }
  }

  // 2. Propagate inward: each voxel checks the nearest points its neighbours found.
  //    A forward sweep looks at the 13 neighbours already visited, a backward sweep at the other 13.
  const forward: number[] = [];
  for (let dk = -1; dk <= 0; dk++)
    for (let dj = -1; dj <= 1; dj++)
      for (let di = -1; di <= 1; di++)
        if (dk < 0 || (dk === 0 && dj < 0) || (dk === 0 && dj === 0 && di < 0))
          forward.push(di + nx * (dj + ny * dk));
  const backward = forward.map((offset) => -offset);

  const sweep = (offsets: number[], reverse: boolean) => {
    for (let kk = 0; kk < nz; kk++) {
      const k = reverse ? nz - 1 - kk : kk;
      for (let jj = 0; jj < ny; jj++) {
        const j = reverse ? ny - 1 - jj : jj;
        for (let ii = 0; ii < nx; ii++) {
          const i = reverse ? nx - 1 - ii : ii;
          const id = voxelIndex(grid, i, j, k);
          if (data[id] >= 0) continue; // inside voxels only (never on the grid border)

          const px = origin.x + i * voxelSize;
          const py = origin.y + j * voxelSize;
          const pz = origin.z + k * voxelSize;
          for (const offset of offsets) {
            const nb = id + offset;
            if (best[nb] === Infinity) continue;
            const dx = px - closest[nb * 3];
            const dy = py - closest[nb * 3 + 1];
            const dz = pz - closest[nb * 3 + 2];
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best[id]) {
              best[id] = d2;
              closest[id * 3] = closest[nb * 3];
              closest[id * 3 + 1] = closest[nb * 3 + 1];
              closest[id * 3 + 2] = closest[nb * 3 + 2];
              nearestFace[id] = nearestFace[nb];
            }
          }
        }
      }
    }
  };
  sweep(forward, false);
  sweep(backward, true);

  // 3. Squared distances -> mm, 0 for voxels outside the original tooth
  const depth = new Float32Array(count);
  // (A voxel the sweeps never reached would stay at Infinity; guard so a future model can't
  //  break the colours. Checked: it does not happen with the current two teeth.)
  for (let id = 0; id < count; id++) {
    depth[id] = data[id] < 0 && Number.isFinite(best[id]) ? Math.sqrt(best[id]) : 0;
  }
  return { depth, nearestFace };
}

// Smoothly interpolated value of a voxel field at any point (trilinear interpolation)
export function sampleField(grid: SDFGrid, field: Float32Array, point: THREE.Vector3): number {
  const { nx, ny, nz, voxelSize, origin } = grid;
  const fx = (point.x - origin.x) / voxelSize;
  const fy = (point.y - origin.y) / voxelSize;
  const fz = (point.z - origin.z) / voxelSize;

  const i = THREE.MathUtils.clamp(Math.floor(fx), 0, nx - 2);
  const j = THREE.MathUtils.clamp(Math.floor(fy), 0, ny - 2);
  const k = THREE.MathUtils.clamp(Math.floor(fz), 0, nz - 2);
  const tx = THREE.MathUtils.clamp(fx - i, 0, 1);
  const ty = THREE.MathUtils.clamp(fy - j, 0, 1);
  const tz = THREE.MathUtils.clamp(fz - k, 0, 1);

  const v = (a: number, b: number, c: number) => field[voxelIndex(grid, i + a, j + b, k + c)];
  const lerp = THREE.MathUtils.lerp;
  const x00 = lerp(v(0, 0, 0), v(1, 0, 0), tx);
  const x10 = lerp(v(0, 1, 0), v(1, 1, 0), tx);
  const x01 = lerp(v(0, 0, 1), v(1, 0, 1), tx);
  const x11 = lerp(v(0, 1, 1), v(1, 1, 1), tx);
  return lerp(lerp(x00, x10, ty), lerp(x01, x11, ty), tz);
}
