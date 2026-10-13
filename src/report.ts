import * as THREE from "three";
import type { VoxelSurface } from "./surface";
import { NO_REGION, type PrepRegion, type PrepSpec } from "./prep";
import { reductionStatus } from "./heatmap";

export interface RegionStats {
  region: PrepRegion;
  area: number; // mm2 of surface in this region
  untouched: number; // mm2 in each state
  under: number;
  on: number;
  over: number;
}

export interface TaperStats {
  wallArea: number; // mm2 of prepared axial wall measured
  meanDeg: number; // average lean of a wall towards the tooth's long axis
  undercutPct: number; // % of that wall leaning the wrong way
}

export interface PrepReport {
  prep: PrepSpec;
  regions: RegionStats[];
  onTargetPct: number; // % of the crown surface (above the margin) on target
  taper: TaperStats | null; // null until some wall has been prepared
}

const a = new THREE.Vector3();
const b = new THREE.Vector3();
const c = new THREE.Vector3();
const normal = new THREE.Vector3();

// Measure the current surface, triangle by triangle, weighting everything by area
export function buildReport(surface: VoxelSurface): PrepReport {
  const prep = surface.grid.prep!;
  const regions: RegionStats[] = prep.regions.map((region) => ({
    region,
    area: 0,
    untouched: 0,
    under: 0,
    on: 0,
    over: 0,
  }));

  let wallArea = 0;
  let taperSum = 0;
  let undercutArea = 0;

  for (const mesh of surface.meshes()) {
    const geometry = mesh.geometry;
    const position = geometry.attributes.position;
    const index = geometry.index!;
    const depth = geometry.userData.depth as Float32Array;
    const region = geometry.userData.region as Uint8Array;

    for (let f = 0; f < index.count; f += 3) {
      const ia = index.getX(f);
      const ib = index.getX(f + 1);
      const ic = index.getX(f + 2);
      const r = region[ia] !== NO_REGION ? region[ia] : region[ib] !== NO_REGION ? region[ib] : region[ic];
      if (r === NO_REGION) continue;

      // Area and direction of the triangle (grid space = tooth space, +Y = path of insertion)
      a.fromBufferAttribute(position, ia);
      b.fromBufferAttribute(position, ib).sub(a);
      c.fromBufferAttribute(position, ic).sub(a);
      normal.crossVectors(b, c);
      const area = normal.length() / 2;
      if (area === 0) continue;
      normal.normalize();

      const d = (depth[ia] + depth[ib] + depth[ic]) / 3;
      const stats = regions[r];
      stats.area += area;
      stats[reductionStatus(d, stats.region)] += area;

      // Taper: prepared axial walls only. Steep (|ny| < 0.5) so shelves/chamfers are not counted.
      // A wall that leans in towards the top has an outward normal tilted upward (ny > 0).
      if (stats.region.kind === "axial" && d > 0.3 && Math.abs(normal.y) < 0.5) {
        const angle = THREE.MathUtils.radToDeg(Math.asin(normal.y));
        wallArea += area;
        taperSum += angle * area;
        if (angle < -1) undercutArea += area; // leans outward: the crown could not seat
      }
    }
  }

  let crownArea = 0;
  let onTarget = 0;
  for (const stats of regions) {
    if (stats.region.kind === "margin") continue;
    crownArea += stats.area;
    onTarget += stats.on;
  }

  return {
    prep,
    regions,
    onTargetPct: crownArea > 0 ? (100 * onTarget) / crownArea : 0,
    taper:
      wallArea > 1 // at least 1 mm2 of prepared wall before reporting
        ? { wallArea, meanDeg: taperSum / wallArea, undercutPct: (100 * undercutArea) / wallArea }
        : null,
  };
}
