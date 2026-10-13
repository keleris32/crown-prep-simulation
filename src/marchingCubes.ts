import * as THREE from "three";
import { edgeTable, triTable } from "three/addons/objects/MarchingCubes.js";
import { type SDFGrid, voxelIndex } from "./sdf";
import { vertexColor } from "./heatmap";
import { NO_REGION } from "./prep";

// The 8 corners of a cell, as (di, dj, dk) offsets from its lowest corner
const CORNERS = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];

// The 12 edges of a cell, as pairs of corners (lower corner first)
const EDGES = [
  [0, 1], [1, 2], [3, 2], [0, 3],
  [4, 5], [5, 6], [7, 6], [4, 7],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

// Which axis each edge runs along: 0 = x, 1 = y, 2 = z
const EDGE_AXIS = [0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2];

// Surface direction at a voxel: the gradient of the distance field (points outward)
function gradient(grid: SDFGrid, i: number, j: number, k: number, out: THREE.Vector3): THREE.Vector3 {
  const { nx, ny, nz, data } = grid;
  const at = (a: number, b: number, c: number) =>
    data[voxelIndex(grid, Math.min(nx - 1, Math.max(0, a)), Math.min(ny - 1, Math.max(0, b)), Math.min(nz - 1, Math.max(0, c)))];
  return out.set(
    at(i + 1, j, k) - at(i - 1, j, k),
    at(i, j + 1, k) - at(i, j - 1, k),
    at(i, j, k + 1) - at(i, j, k - 1),
  );
}

// Build the surface for cells [i0, i1) x [j0, j1) x [k0, k1). Returns null if empty.
export function meshRegion(
  grid: SDFGrid,
  i0: number, j0: number, k0: number,
  i1: number, j1: number, k1: number,
): THREE.BufferGeometry | null {
  const { voxelSize, origin, data, depth, region, prep } = grid;

  const positions: number[] = [];
  const normals: number[] = [];
  const colors: number[] = [];
  const vertexDepths: number[] = []; // kept on the CPU for the prep report
  const vertexRegions: number[] = [];
  const color = new THREE.Color();
  const indices: number[] = [];
  const vertexOfEdge = new Map<number, number>(); // grid edge -> vertex number (shared vertices)

  const cornerIds = new Int32Array(8);
  const cornerValues = new Float32Array(8);
  const edgeVertex = new Int32Array(12);
  const ga = new THREE.Vector3();
  const gb = new THREE.Vector3();

  for (let k = k0; k < k1; k++) {
    for (let j = j0; j < j1; j++) {
      for (let i = i0; i < i1; i++) {
        // 1. Which corners are inside? One bit per corner -> a number from 0 to 255
        let cubeIndex = 0;
        for (let c = 0; c < 8; c++) {
          const id = voxelIndex(grid, i + CORNERS[c][0], j + CORNERS[c][1], k + CORNERS[c][2]);
          cornerIds[c] = id;
          cornerValues[c] = data[id];
          if (data[id] < 0) cubeIndex |= 1 << c;
        }

        // 2. Which edges does the surface cross? (0 = cell fully inside or outside)
        const crossed = edgeTable[cubeIndex];
        if (crossed === 0) continue;

        // 3. Place a vertex on each crossed edge, exactly where the distance hits zero
        for (let e = 0; e < 12; e++) {
          if ((crossed & (1 << e)) === 0) continue;
          const a = EDGES[e][0];
          const b = EDGES[e][1];

          const key = cornerIds[a] * 3 + EDGE_AXIS[e];
          let vertex = vertexOfEdge.get(key);
          if (vertex === undefined) {
            const da = cornerValues[a];
            const db = cornerValues[b];
            const t = da / (da - db); // 0..1 along the edge

            const ca = CORNERS[a];
            const cb = CORNERS[b];
            positions.push(
              origin.x + (i + ca[0] + t * (cb[0] - ca[0])) * voxelSize,
              origin.y + (j + ca[1] + t * (cb[1] - ca[1])) * voxelSize,
              origin.z + (k + ca[2] + t * (cb[2] - ca[2])) * voxelSize,
            );

            gradient(grid, i + ca[0], j + ca[1], k + ca[2], ga);
            gradient(grid, i + cb[0], j + cb[1], k + cb[2], gb);
            ga.lerp(gb, t).normalize();
            normals.push(ga.x, ga.y, ga.z);

            // Heatmap: how far below the original surface this vertex is, and in which region
            if (depth) {
              const vertexDepth = THREE.MathUtils.lerp(depth[cornerIds[a]], depth[cornerIds[b]], t);
              let vertexRegion = NO_REGION;
              if (region) {
                vertexRegion = region[cornerIds[a]] !== NO_REGION ? region[cornerIds[a]] : region[cornerIds[b]];
              }
              vertexColor(vertexDepth, vertexRegion, prep, color);
              colors.push(color.r, color.g, color.b);
              vertexDepths.push(vertexDepth);
              vertexRegions.push(vertexRegion);
            }

            vertex = positions.length / 3 - 1;
            vertexOfEdge.set(key, vertex);
          }
          edgeVertex[e] = vertex;
        }

        // 4. Connect the vertices into triangles using the lookup table
        const row = cubeIndex * 16;
        for (let t = row; triTable[t] !== -1; t += 3) {
          indices.push(edgeVertex[triTable[t]], edgeVertex[triTable[t + 2]], edgeVertex[triTable[t + 1]]);
        }
      }
    }
  }

  if (indices.length === 0) return null;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
  if (depth) {
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    // userData stays in JavaScript memory - it is not uploaded to the GPU
    geometry.userData.depth = new Float32Array(vertexDepths);
    geometry.userData.region = new Uint8Array(vertexRegions);
  }
  geometry.setIndex(indices);
  return geometry;
}
