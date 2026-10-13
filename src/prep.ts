import * as THREE from "three";
import { type SDFGrid, voxelIndex } from "./sdf";

export const NO_REGION = 255; // voxel is not part of the original tooth

// occlusal = the biting surface/edge, axial = the walls, margin = below the finish line (never cut)
export type RegionKind = "occlusal" | "axial" | "margin";

export interface PrepRegion {
  name: string;
  kind: RegionKind;
  min: number; // mm of reduction
  max: number;
}

export interface PrepSpec {
  name: string;
  marginY: number; // default height of the finish line, mm above the base of the tooth
  regions: PrepRegion[]; // region number = index in this list
  // Original-surface normal (tooth space, +Y = towards the crown tip) -> region number
  classify: (normal: THREE.Vector3) => number;
}

// Typical textbook reduction ranges. Adjust them to the crown material you are teaching.
export const MOLAR_PREP: PrepSpec = {
  name: "Molar - full-coverage crown",
  marginY: 6.0, // ~1 mm above where the roots join
  regions: [
    { name: "Occlusal", kind: "occlusal", min: 1.5, max: 2.0 },
    { name: "Axial", kind: "axial", min: 1.0, max: 1.5 },
    { name: "Below margin", kind: "margin", min: 0, max: 0 },
  ],
  classify: (n) => (n.y > 0.5 ? 0 : 1),
};

export const INCISOR_PREP: PrepSpec = {
  name: "Central incisor - all-ceramic crown",
  marginY: 14.0, // at the neck of the tooth, where the crown narrows into the root
  regions: [
    { name: "Incisal", kind: "occlusal", min: 1.5, max: 2.0 },
    { name: "Labial / proximal", kind: "axial", min: 1.0, max: 1.5 },
    { name: "Lingual", kind: "axial", min: 0.7, max: 1.0 },
    { name: "Below margin", kind: "margin", min: 0, max: 0 },
  ],
  // After loading, the lingual (concave) side of this scan faces +Z
  classify: (n) => (n.y > 0.5 ? 0 : n.z > 0.5 ? 2 : 1),
};

// Give every voxel of the original tooth a region: below the margin line -> "margin",
// otherwise decided by the surface normal at its nearest original surface point
export function classifyRegions(
  geometry: THREE.BufferGeometry,
  nearestFace: Int32Array,
  prep: PrepSpec,
  marginY: number,
): Uint8Array {
  const marginRegion = prep.regions.findIndex((r) => r.kind === "margin");
  const position = geometry.attributes.position;
  const index = geometry.index!;
  const normal = geometry.attributes.normal;
  const regions = new Uint8Array(nearestFace.length).fill(NO_REGION);

  // Many voxels share the same nearest triangle, so classify each triangle only once
  const faceRegion = new Int16Array(index.count / 3).fill(-1);
  const n = new THREE.Vector3();
  const corner = new THREE.Vector3();

  for (let id = 0; id < nearestFace.length; id++) {
    const face = nearestFace[id];
    if (face < 0) continue;

    if (faceRegion[face] === -1) {
      // Average of the three corner normals = smoother than the raw triangle normal
      n.set(0, 0, 0);
      let centreY = 0;
      for (let c = 0; c < 3; c++) {
        const v = index.getX(face * 3 + c);
        n.add(corner.fromBufferAttribute(normal, v));
        centreY += position.getY(v) / 3;
      }
      faceRegion[face] = centreY < marginY ? marginRegion : prep.classify(n.normalize());
    }
    regions[id] = faceRegion[face];
  }
  return regions;
}

// Region at any point: the closest of the 8 surrounding voxels that belongs to the original tooth
export function regionAt(grid: SDFGrid, point: THREE.Vector3): number {
  const { nx, ny, nz, voxelSize, origin, region } = grid;
  if (!region) return NO_REGION;

  const i = THREE.MathUtils.clamp(Math.floor((point.x - origin.x) / voxelSize), 0, nx - 2);
  const j = THREE.MathUtils.clamp(Math.floor((point.y - origin.y) / voxelSize), 0, ny - 2);
  const k = THREE.MathUtils.clamp(Math.floor((point.z - origin.z) / voxelSize), 0, nz - 2);

  let best = NO_REGION;
  let bestDistance = Infinity;
  for (let c = 0; c < 8; c++) {
    const a = i + (c & 1);
    const b = j + ((c >> 1) & 1);
    const d = k + ((c >> 2) & 1);
    const r = region[voxelIndex(grid, a, b, d)];
    if (r === NO_REGION) continue;

    const dx = origin.x + a * voxelSize - point.x;
    const dy = origin.y + b * voxelSize - point.y;
    const dz = origin.z + d * voxelSize - point.z;
    const distance = dx * dx + dy * dy + dz * dz;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = r;
    }
  }
  return best;
}
