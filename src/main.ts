import "./style.css";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { mergeVertices } from "three/addons/utils/BufferGeometryUtils.js";
import {
  computeBoundsTree,
  disposeBoundsTree,
  acceleratedRaycast,
} from "three-mesh-bvh";
import {
  checkWatertight,
  voxelize,
  createVoxelDebugPoints,
  buildDepthField,
  sampleField,
} from "./sdf";
import { DEPTH_LEGEND, REGION_LEGEND, getColorMode, setColorMode, reductionStatus } from "./heatmap";
import { buildReport } from "./report";
import {
  MOLAR_PREP,
  INCISOR_PREP,
  NO_REGION,
  classifyRegions,
  regionAt,
  type PrepSpec,
} from "./prep";
import { VoxelSurface, MAX_UNDO } from "./surface";
import { saveSession, loadSession } from "./session";

// Attach BVH (spatial index) support to every geometry and mesh
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document
  .querySelector<HTMLDivElement>("#app")!
  .appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1e1e24);

const camera = new THREE.PerspectiveCamera(
  45, // field of view (degrees)
  window.innerWidth / window.innerHeight, // aspect ratio
  0.1, // near clipping plane
  500, // far clipping plane
);
camera.position.set(0, 25, 55);

const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444455, 1.0);
scene.add(hemiLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
keyLight.position.set(20, 40, 30);
scene.add(keyLight);

const grid = new THREE.GridHelper(50, 50, 0x555566, 0x33333d);
scene.add(grid);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 10, 0);
controls.enableDamping = true;

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const stlLoader = new STLLoader();
const gltfLoader = new GLTFLoader();

async function loadGeometryFromGLTF(
  url: string,
): Promise<THREE.BufferGeometry> {
  const gltf = await gltfLoader.loadAsync(url);
  gltf.scene.updateMatrixWorld(true);

  const meshes: THREE.Mesh[] = [];
  gltf.scene.traverse((obj) => {
    if ((obj as THREE.Mesh).isMesh) meshes.push(obj as THREE.Mesh);
  });

  if (meshes.length === 0) throw new Error(`No mesh found in ${url}`);
  if (meshes.length > 1) {
    console.warn(
      `${url} has ${meshes.length} meshes:`,
      meshes.map((m) => m.name),
    );
  }

  const geometry = meshes[0].geometry.clone();
  geometry.applyMatrix4(meshes[0].matrixWorld);
  return geometry;
}

async function loadGeometry(url: string): Promise<THREE.BufferGeometry> {
  const ext = url.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "stl":
      return stlLoader.loadAsync(url);
    case "glb":
    case "gltf":
      return loadGeometryFromGLTF(url);
    default:
      throw new Error(`Unsupported format: .${ext}`);
  }
}

interface ToothOptions {
  rotateX?: number; // radians
  fit?: { axis: "x" | "y" | "z"; mm: number };
}

async function loadTooth(
  url: string,
  options: ToothOptions = {},
): Promise<THREE.Mesh> {
  let geometry = await loadGeometry(url);

  // 1. Keep only the shape, then merge and smooth
  for (const name of Object.keys(geometry.attributes)) {
    if (name !== "position") geometry.deleteAttribute(name);
  }
  geometry = mergeVertices(geometry);
  geometry.computeVertexNormals();

  // 2. Orientation first, so "y" really means height
  if (options.rotateX) geometry.rotateX(options.rotateX);

  // 3. Calibrate to real millimetres
  geometry.computeBoundingBox();
  const size = new THREE.Vector3();
  geometry.boundingBox!.getSize(size);

  if (options.fit) {
    const factor = options.fit.mm / size[options.fit.axis];
    geometry.scale(factor, factor, factor);
    size.multiplyScalar(factor);
  }

  console.log(
    url,
    `size: ${size.x.toFixed(2)} x ${size.y.toFixed(2)} x ${size.z.toFixed(2)} mm`,
    `vertices: ${geometry.attributes.position.count}`,
  );

  // 4. Centre it, then stand it on the grid
  geometry.center();
  geometry.computeBoundingBox();
  geometry.translate(0, -geometry.boundingBox!.min.y, 0);

  // 5. Build the spatial index AFTER the final vertex change
  geometry.computeBoundsTree();

  const material = new THREE.MeshPhysicalMaterial({
    color: 0xf2efe6,
    roughness: 0.35,
    clearcoat: 0.4,
    clearcoatRoughness: 0.2,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = url;
  return mesh;
}

const pointer = new THREE.Vector2();
let pointerInside = false;

renderer.domElement.addEventListener("pointermove", (e) => {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  pointerInside = true;
});

renderer.domElement.addEventListener("pointerleave", () => {
  pointerInside = false;
});

const teeth: THREE.Mesh[] = [];
const voxelViews: THREE.Points[] = [];
const surfaces: VoxelSurface[] = [];
const surfaceGroups: THREE.Group[] = [];

// --- Bur --------------------------------------------------------------------

// Bur dimensions in mm, shared by the 3D model and the carving maths
const BUR = { tipRadius: 0.5, topRadius: 0.6, length: 8 };

function createBur(): THREE.Group {
  const bur = new THREE.Group();

  const diamondMat = new THREE.MeshStandardMaterial({
    color: 0xb0aa98,
    roughness: 0.9,
  });
  const steelMat = new THREE.MeshStandardMaterial({
    color: 0xc0c4cc,
    roughness: 0.3,
    metalness: 0.6,
  });

  // Rounded tip: sphere sitting on the origin
  const tip = new THREE.Mesh(
    new THREE.SphereGeometry(BUR.tipRadius, 24, 16),
    diamondMat,
  );
  tip.position.y = BUR.tipRadius;

  // Tapered cutting head, widening slightly upward
  const head = new THREE.Mesh(
    new THREE.CylinderGeometry(BUR.topRadius, BUR.tipRadius, BUR.length, 24),
    diamondMat,
  );
  head.position.y = BUR.tipRadius + BUR.length / 2;

  // Smooth steel shank
  const shank = new THREE.Mesh(
    new THREE.CylinderGeometry(0.8, 0.8, 12, 24),
    steelMat,
  );
  shank.position.y = BUR.tipRadius + BUR.length + 6;

  bur.add(tip, head, shank);
  bur.visible = false;
  return bur;
}

const bur = createBur();
scene.add(bur);

const raycaster = new THREE.Raycaster();
raycaster.firstHitOnly = true;
const UP = new THREE.Vector3(0, 1, 0);

// --- Settings and drilling state --------------------------------------------

const FEED_RATE = 1.0; // mm per second the bur sinks while held
const DEPTH_STEP = 0.25; // mm per [ or ] press
const MARGIN_STEP = 0.25; // mm per , or . press
const BUR_TILT_DEG = 3; // axial mode: how far the bur leans in, which gives the walls their taper
let depthLimit = 1.5; // deepest the bur may go below the ORIGINAL surface, mm
let heatmapOn = true;
let axialMode = false; // A key: bur parallel to the path of insertion instead of along the surface normal

let drilling = false;
let drillMode: "normal" | "axial" = "normal"; // fixed for the whole stroke
let drillDepth = 0; // how far the bur currently is below the original surface, mm
let drillSurface: VoxelSurface | null = null;
const drillAxis = new THREE.Vector3(); // bur direction
const inward = new THREE.Vector3(); // axial mode: horizontal direction into the tooth
const lastTipCentre = new THREE.Vector3();
let hasLastTip = false;
let hover: THREE.Intersection | null = null; // latest hit on the visible surface
let hoverIsWall = false; // can axial mode be used where we are hovering?
let lastSurface: VoxelSurface | null = null; // the tooth most recently hovered (for margin keys)

// --- On-screen display (HUD) --------------------------------------------------

const hud = document.createElement("div");
hud.id = "hud";
const hudStatus = document.createElement("pre");
const legend = document.createElement("div");
legend.className = "legend";
hud.append(hudStatus, legend);
document.body.appendChild(hud);

function renderLegend(): void {
  const entries = getColorMode() === "region" ? REGION_LEGEND : DEPTH_LEGEND;
  legend.innerHTML = entries
    .map((entry) => `<span><i style="background:${entry.css}"></i>${entry.label}</span>`)
    .join("");
  legend.style.display = heatmapOn ? "" : "none";
}
renderLegend();

// What the bur is pointing at: which tooth's targets, which region, how deep
interface Probe {
  prep: PrepSpec;
  margin: number;
  region: number;
  depth: number;
}

const STATUS_TEXT = { untouched: "untouched", under: "UNDER target", on: "on target", over: "OVER target" };

let hudText = "";
function updateHud(probe: Probe | null): void {
  const lines = [probe ? probe.prep.name : "Hover over a tooth", "", ""];

  if (probe) {
    const label = drilling ? "Bur at" : "Here";
    if (probe.region === NO_REGION) {
      lines[1] = `${label}: ${probe.depth.toFixed(2)} mm`;
    } else {
      const region = probe.prep.regions[probe.region];
      lines[1] = `${label}: ${region.name}, ${probe.depth.toFixed(2)} mm deep`;
      const status = reductionStatus(probe.depth, region);
      lines[2] =
        region.kind === "margin"
          ? status === "untouched" ? "Below the margin - do not cut" : "CUT BELOW MARGIN"
          : `Target ${region.min.toFixed(1)}-${region.max.toFixed(1)} mm: ${STATUS_TEXT[status]}`;
    }
  }

  lines.push("", "Hold left mouse on a tooth to drill", "Drag elsewhere to rotate, scroll to zoom");
  const text = lines.join("\n");

  // Only touch the page when the text actually changed
  if (text !== hudText) {
    hudStatus.textContent = text;
    hudText = text;
  }
}

function setHeatmap(on: boolean): void {
  heatmapOn = on;
  for (const surface of surfaces) {
    surface.material.vertexColors = on;
    surface.material.needsUpdate = true; // the shader must be recompiled
  }
  renderLegend();
}

// --- Prep report panel (P) ------------------------------------------------------

const reportPanel = document.createElement("div");
reportPanel.id = "report";
reportPanel.style.display = "none";
document.body.appendChild(reportPanel);
let reportTimer = 0;

const pct = (part: number, whole: number) => (whole > 0 ? `${Math.round((100 * part) / whole)}%` : "-");

function renderReport(): void {
  const sections = surfaces.map((surface) => {
    const report = buildReport(surface);

    const rows = report.regions
      .map(({ region, area, untouched, under, on, over }) =>
        region.kind === "margin"
          ? `<tr class="margin"><td>${region.name}</td><td>none</td>
               <td colspan="4">${over > 0.05 ? `CUT: ${over.toFixed(1)} mm²` : "intact"}</td></tr>`
          : `<tr><td>${region.name}</td><td>${region.min.toFixed(1)}-${region.max.toFixed(1)}</td>
               <td>${pct(untouched, area)}</td><td>${pct(under, area)}</td>
               <td class="on">${pct(on, area)}</td><td>${pct(over, area)}</td></tr>`,
      )
      .join("");

    const taper = report.taper
      ? `Wall taper: ${report.taper.meanDeg.toFixed(1)}° per wall
         (~${(2 * report.taper.meanDeg).toFixed(0)}° total convergence; aim for 6-10°)<br>
         Undercut: ${report.taper.undercutPct.toFixed(0)}% of prepared wall (should be 0%)`
      : "Wall taper: no axial walls prepared yet";

    return `<h3>${report.prep.name}</h3>
      <div class="score">On target: ${report.onTargetPct.toFixed(0)}% of crown surface</div>
      <table>
        <tr><th>Region</th><th>Target mm</th><th>Untouched</th><th>Under</th><th>On</th><th>Over</th></tr>
        ${rows}
      </table>
      <p>${taper}</p>`;
  });
  reportPanel.innerHTML = sections.join("");
}

// --- Actions (shared by the keyboard and the toolbar) ---------------------------

// The tooth the margin buttons act on: the one last hovered, otherwise the first
const activeSurface = (): VoxelSurface | null => lastSurface ?? surfaces[0] ?? null;
const shortName = (surface: VoxelSurface) => surface.grid.prep!.name.split(" - ")[0];

// Move the finish line of a tooth: re-classify its regions and re-colour it
function setMargin(surface: VoxelSurface, marginY: number): void {
  const { grid } = surface;
  grid.marginY = THREE.MathUtils.clamp(marginY, 0, 30);
  const tooth = surface.group.parent as THREE.Mesh;
  grid.region = classifyRegions(tooth.geometry, grid.nearestFace!, grid.prep!, grid.marginY);
  surface.rebuildAll();
}

// Each tooth keeps its own history; these lists remember which tooth each stroke was on,
// so Undo always takes back the most recent stroke, whichever tooth it was made on
const undoOrder: VoxelSurface[] = [];
const redoOrder: VoxelSurface[] = [];

function undo(): void {
  if (drilling) return;
  while (undoOrder.length > 0) {
    const surface = undoOrder.pop()!;
    if (surface.undo()) {
      redoOrder.push(surface);
      break;
    }
  }
  refreshUi();
}

function redo(): void {
  if (drilling) return;
  const surface = redoOrder.pop();
  if (surface?.redo()) undoOrder.push(surface);
  refreshUi();
}

function resetAll(): void {
  if (!window.confirm("Reset both teeth to uncut? This cannot be undone.")) return;
  for (const surface of surfaces) surface.reset();
  undoOrder.length = 0;
  redoOrder.length = 0;
  refreshUi();
}

function toggleBurMode(): void {
  axialMode = !axialMode;
  refreshUi();
}

function changeDepthLimit(delta: number): void {
  depthLimit = THREE.MathUtils.clamp(depthLimit + delta, DEPTH_STEP, 3);
  refreshUi();
}

function changeMargin(delta: number): void {
  const surface = activeSurface();
  if (surface) setMargin(surface, surface.grid.marginY! + delta);
  refreshUi();
}

function toggleHeatmap(): void {
  setHeatmap(!heatmapOn);
  refreshUi();
}

function toggleRegions(): void {
  // Region view: colour each part of the tooth by which target zone it belongs to
  setColorMode(getColorMode() === "region" ? "depth" : "region");
  setHeatmap(true);
  for (const surface of surfaces) surface.rebuildAll(); // colours live in the vertices
  renderLegend();
  refreshUi();
}

function toggleReport(): void {
  reportPanel.style.display = reportPanel.style.display === "none" ? "" : "none";
  reportTimer = 0; // refresh straight away
  refreshUi();
}

function toggleVoxels(): void {
  // Voxel debug view (note: shows the grid as it was before any carving)
  const show = !voxelViews[0]?.visible;
  for (const points of voxelViews) points.visible = show;
  for (const surface of surfaces) surface.group.visible = !show;
}

function toggleOriginal(): void {
  // Original scan vs the voxel surface
  for (const tooth of teeth) {
    const material = tooth.material as THREE.Material;
    material.visible = !material.visible;
  }
  for (const surface of surfaces) surface.group.visible = !surface.group.visible;
}

// Save: build the file in memory, then "click" a temporary download link
async function saveToFile(): Promise<void> {
  const blob = await saveSession(surfaces);
  const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `crown-prep-${stamp}.prep`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000); // free the memory once the download started
}

// Load: a hidden file picker
const fileInput = document.createElement("input");
fileInput.type = "file";
fileInput.accept = ".prep";
fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  fileInput.value = ""; // so the same file can be picked again
  if (!file) return;
  try {
    const teethInFile = await loadSession(file, surfaces);
    teethInFile.forEach((tooth, n) => setMargin(surfaces[n], tooth.marginY));
    undoOrder.length = 0;
    redoOrder.length = 0;
  } catch (err) {
    window.alert(`Could not load this file: ${(err as Error).message}`);
  }
  refreshUi();
});

// --- Toolbar --------------------------------------------------------------------

const toolbar = document.createElement("div");
toolbar.id = "toolbar";
document.body.appendChild(toolbar);

function makeButton(text: string, tooltip: string, onClick: () => void): HTMLButtonElement {
  const button = document.createElement("button");
  button.textContent = text;
  button.title = tooltip;
  button.addEventListener("click", () => {
    onClick();
    button.blur(); // otherwise Space/Enter would press it again
  });
  return button;
}

function makeLabel(text = ""): HTMLSpanElement {
  const label = document.createElement("span");
  label.textContent = text;
  return label;
}

function makeGroup(...items: HTMLElement[]): HTMLDivElement {
  const group = document.createElement("div");
  group.className = "group";
  group.append(...items);
  return group;
}

const undoButton = makeButton("Undo", "Undo the last stroke (Ctrl/Cmd+Z)", undo);
const redoButton = makeButton("Redo", "Redo (Ctrl/Cmd+Shift+Z)", redo);
const burButton = makeButton("", "Switch the bur angle (A)", toggleBurMode);
const depthLabel = makeLabel();
const marginLabel = makeLabel();
const heatmapButton = makeButton("Heatmap", "Colour by reduction depth (H)", toggleHeatmap);
const regionsButton = makeButton("Regions", "Show the target regions and margin (R)", toggleRegions);
const reportButton = makeButton("Report", "Show the prep report (P)", toggleReport);

toolbar.append(
  makeGroup(undoButton, redoButton, makeButton("Reset", "Start again with uncut teeth", resetAll)),
  makeGroup(burButton),
  makeGroup(
    makeLabel("Depth limit"),
    makeButton("-", "Shallower ([)", () => changeDepthLimit(-DEPTH_STEP)),
    depthLabel,
    makeButton("+", "Deeper (])", () => changeDepthLimit(DEPTH_STEP)),
  ),
  makeGroup(
    marginLabel,
    makeButton("-", "Move the margin down (,)", () => changeMargin(-MARGIN_STEP)),
    makeButton("+", "Move the margin up (.)", () => changeMargin(MARGIN_STEP)),
  ),
  makeGroup(heatmapButton, regionsButton, reportButton),
  makeGroup(
    makeButton("Save", "Download this prep as a file", () => void saveToFile()),
    makeButton("Load", "Open a saved prep file", () => fileInput.click()),
  ),
);

// Bring the toolbar (and HUD) in line with the current state
function refreshUi(): void {
  undoButton.disabled = undoOrder.length === 0;
  redoButton.disabled = redoOrder.length === 0;
  burButton.textContent = axialMode ? "Bur: Axial" : "Bur: Normal";
  depthLabel.textContent = `${depthLimit.toFixed(2)} mm`;
  const surface = activeSurface();
  marginLabel.textContent = surface
    ? `Margin (${shortName(surface)}): ${surface.grid.marginY!.toFixed(2)} mm`
    : "Margin";
  heatmapButton.classList.toggle("on", heatmapOn);
  regionsButton.classList.toggle("on", getColorMode() === "region");
  reportButton.classList.toggle("on", reportPanel.style.display !== "none");
}

// --- Keys ---------------------------------------------------------------------

window.addEventListener("keydown", (e) => {
  const key = e.key.toLowerCase();

  if (e.ctrlKey || e.metaKey) {
    if (key === "z") {
      e.preventDefault();
      if (e.shiftKey) redo();
      else undo();
    } else if (key === "y") {
      e.preventDefault();
      redo();
    }
    return; // leave every other Ctrl/Cmd shortcut (copy, reload...) to the browser
  }

  // Developer views only (npm run dev): they hide the drillable surface, which would confuse a tester
  if (import.meta.env.DEV && key === "v") toggleVoxels();
  if (import.meta.env.DEV && key === "o") toggleOriginal();

  switch (key) {
    case "h":
      toggleHeatmap();
      break;
    case "r":
      toggleRegions();
      break;
    case "p":
      toggleReport();
      break;
    case "a":
      toggleBurMode();
      break;
    case "[":
      changeDepthLimit(-DEPTH_STEP);
      break;
    case "]":
      changeDepthLimit(DEPTH_STEP);
      break;
    case ",":
      changeMargin(-MARGIN_STEP);
      break;
    case ".":
      changeMargin(MARGIN_STEP);
      break;
  }
});

// --- Helpers ------------------------------------------------------------------

const toLocal = new THREE.Matrix4();
const localPoint = new THREE.Vector3();
const tipCentre = new THREE.Vector3();
const contact = new THREE.Vector3();
const sweepPoint = new THREE.Vector3();
const hoverNormal = new THREE.Vector3();
const hoverAxis = new THREE.Vector3();
const hoverInward = new THREE.Vector3();
const drillNormal = new THREE.Vector3();
const flatInward = new THREE.Vector3();
const targetQuat = new THREE.Quaternion();
const triA = new THREE.Vector3();
const triB = new THREE.Vector3();
const triC = new THREE.Vector3();
const normalA = new THREE.Vector3();
const normalB = new THREE.Vector3();
const normalC = new THREE.Vector3();
const bary = new THREE.Vector3();
const sweepDir = new THREE.Vector3();
const burPose = {
  tip: new THREE.Vector3(),
  axis: new THREE.Vector3(),
  ...BUR,
  sweep: new THREE.Vector3(),
  sweepLength: 0,
};

function surfacesVisible(): boolean {
  return surfaceGroups.length > 0 && surfaceGroups[0].visible;
}

// Region and depth below the original surface at a world-space point (one reused result object)
const probeResult: Probe = { prep: MOLAR_PREP, margin: 0, region: NO_REGION, depth: 0 };
function probe(surface: VoxelSurface, worldPoint: THREE.Vector3): Probe {
  const { grid } = surface;
  toLocal.copy(surface.group.parent!.matrixWorld).invert();
  localPoint.copy(worldPoint).applyMatrix4(toLocal);

  probeResult.prep = grid.prep!;
  probeResult.margin = grid.marginY!;
  probeResult.region = regionAt(grid, localPoint);
  probeResult.depth = sampleField(grid, grid.depth!, localPoint);
  return probeResult;
}

// Blend the corner normals of the hit triangle, so the bur doesn't jitter between tiny triangles
function smoothNormal(hit: THREE.Intersection, out: THREE.Vector3): THREE.Vector3 {
  const mesh = hit.object as THREE.Mesh;
  const face = hit.face!;
  const position = mesh.geometry.attributes.position;
  const normal = mesh.geometry.attributes.normal;

  toLocal.copy(mesh.matrixWorld).invert();
  localPoint.copy(hit.point).applyMatrix4(toLocal);
  triA.fromBufferAttribute(position, face.a);
  triB.fromBufferAttribute(position, face.b);
  triC.fromBufferAttribute(position, face.c);

  // Barycentric coordinates: how much of each corner the hit point is "made of"
  if (THREE.Triangle.getBarycoord(localPoint, triA, triB, triC, bary) === null) {
    return out.copy(face.normal).transformDirection(mesh.matrixWorld);
  }
  out
    .set(0, 0, 0)
    .addScaledVector(normalA.fromBufferAttribute(normal, face.a), bary.x)
    .addScaledVector(normalB.fromBufferAttribute(normal, face.b), bary.y)
    .addScaledVector(normalC.fromBufferAttribute(normal, face.c), bary.z);
  return out.transformDirection(mesh.matrixWorld); // also normalises
}

// Axial mode pose: bur parallel to the path of insertion (+Y), its top leaning BUR_TILT_DEG
// towards the tooth so the walls it cuts converge towards the occlusal surface.
// Returns false where the surface faces mostly up or down (not a wall).
function axialPose(normal: THREE.Vector3, axisOut: THREE.Vector3, inwardOut: THREE.Vector3): boolean {
  flatInward.set(-normal.x, 0, -normal.z);
  if (flatInward.lengthSq() < 0.3 * 0.3) return false;
  inwardOut.copy(flatInward).normalize();

  const tilt = THREE.MathUtils.degToRad(BUR_TILT_DEG);
  axisOut.copy(UP).multiplyScalar(Math.cos(tilt)).addScaledVector(inwardOut, Math.sin(tilt));
  return true;
}

// Show the bur with its tip centre at a point, pointing along an axis
function placeBur(centre: THREE.Vector3, axis: THREE.Vector3, smooth: boolean, dt: number): void {
  targetQuat.setFromUnitVectors(UP, axis);
  // Ease towards the new angle instead of snapping; snap when the bur first appears
  if (smooth && bur.visible) bur.quaternion.slerp(targetQuat, 1 - Math.exp(-15 * dt));
  else bur.quaternion.copy(targetQuat);
  bur.position.copy(centre).addScaledVector(axis, -BUR.tipRadius); // bur origin = bottom of the tip
  bur.visible = true;
}

// --- Drilling -----------------------------------------------------------------

// capture: true -> runs before OrbitControls' own pointerdown listener
renderer.domElement.addEventListener(
  "pointerdown",
  (e) => {
    if (e.button !== 0 || !hover) return; // left button, over a tooth
    drillSurface = hover.object.userData.surface as VoxelSurface;
    drillSurface.beginStroke(); // remember what this stroke changes, for Undo

    if (axialMode && hoverIsWall) {
      drillMode = "axial";
      drillAxis.copy(hoverAxis);
      inward.copy(hoverInward);
    } else {
      drillMode = "normal";
      drillAxis.copy(UP).applyQuaternion(bur.quaternion);
    }

    // Start from the depth already cut here, so drilling into an existing groove has no dead time
    drillDepth = Math.min(probe(drillSurface, hover.point).depth, depthLimit);
    hasLastTip = false;

    drilling = true;
    controls.enabled = false; // stop the camera rotating while we cut

    // Keep receiving this pointer's events even if it leaves the canvas or the browser window,
    // so releasing the button anywhere still ends the stroke
    renderer.domElement.setPointerCapture(e.pointerId);
  },
  { capture: true },
);

// End a stroke however it finishes: button released, pointer cancelled by the browser,
// or the window losing focus mid-stroke (e.g. Cmd/Alt+Tab)
function endDrilling(): void {
  const surface = drillSurface;
  if (surface?.endStroke()) {
    // A new cut: it can be undone, and anything previously undone can no longer be redone
    undoOrder.push(surface);
    // Each tooth only keeps its last MAX_UNDO strokes, so forget older entries here too
    if (undoOrder.filter((s) => s === surface).length > MAX_UNDO) {
      undoOrder.splice(undoOrder.indexOf(surface), 1);
    }
    redoOrder.length = 0;
    for (const other of surfaces) other.clearRedo();
    refreshUi();
  }
  drilling = false;
  drillSurface = null;
  controls.enabled = true;
}

window.addEventListener("pointerup", endDrilling);
renderer.domElement.addEventListener("pointercancel", endDrilling);
window.addEventListener("blur", endDrilling);

function carveAt(surface: VoxelSurface, centreWorld: THREE.Vector3): void {
  // The grid lives in the tooth's own coordinates, so convert the bur into them
  toLocal.copy(surface.group.parent!.matrixWorld).invert();
  burPose.tip.copy(centreWorld).applyMatrix4(toLocal);
  burPose.axis.copy(drillAxis).transformDirection(toLocal);

  if (drillMode === "axial") {
    // Clear everything between the bur and the outside, as you would clinically - otherwise a
    // leaning bur cutting deeper than its own width leaves a thin fin of enamel standing outside it.
    // Sweep direction: outward, at right angles to the bur. Far enough to clear any bulge.
    sweepDir.copy(inward).addScaledVector(drillAxis, -inward.dot(drillAxis)).normalize().negate();
    burPose.sweep.copy(sweepDir).transformDirection(toLocal);
    burPose.sweepLength = drillDepth + 2;
  } else {
    burPose.sweepLength = 0;
  }

  surface.carve(burPose); // saves what it is about to change, then carves
}

function updateDrilling(dt: number, surface: VoxelSurface): void {
  // Aim against the ORIGINAL scan. It never changes, so the bur can't jump onto the walls
  // of the groove it is cutting.
  const tooth = surface.group.parent as THREE.Mesh;
  const hit = raycaster.intersectObject(tooth, false)[0];
  if (!hit) return; // pointer is off the tooth: the bur waits where it is

  drillDepth = Math.min(depthLimit, drillDepth + FEED_RATE * dt);

  if (drillMode === "axial") {
    // The bur stands upright beside the wall; its side reaches drillDepth into the original surface.
    // The inward direction follows the original wall as you sweep around the tooth.
    axialPose(smoothNormal(hit, drillNormal), drillAxis, inward); // keeps the last pose on flat spots
    contact.copy(hit.point).addScaledVector(inward, drillDepth);
    tipCentre.copy(hit.point).addScaledVector(inward, drillDepth - BUR.tipRadius);
  } else {
    // The tip is pushed into the original surface along the locked axis
    contact.copy(hit.point).addScaledVector(drillAxis, -drillDepth);
    tipCentre.copy(contact).addScaledVector(drillAxis, BUR.tipRadius);
  }
  placeBur(tipCentre, drillAxis, false, dt);

  // Sweep from last frame's position, so a fast drag cuts a smooth channel, not a row of dents
  if (!hasLastTip) {
    lastTipCentre.copy(tipCentre);
    hasLastTip = true;
  }
  const distance = lastTipCentre.distanceTo(tipCentre);
  const steps = Math.min(12, Math.max(1, Math.ceil(distance / (BUR.tipRadius / 2))));
  for (let s = 1; s <= steps; s++) {
    sweepPoint.lerpVectors(lastTipCentre, tipCentre, s / steps);
    carveAt(surface, sweepPoint);
  }
  lastTipCentre.copy(tipCentre);

  surface.rebuild(); // once per frame, however many steps were carved
  updateHud(probe(surface, contact));
}

function updateBur(dt: number): void {
  if (!pointerInside || !surfacesVisible()) {
    bur.visible = false;
    hover = null;
    updateHud(null);
    return;
  }

  raycaster.setFromCamera(pointer, camera);

  if (drilling && drillSurface) {
    updateDrilling(dt, drillSurface);
    return;
  }

  // Hovering: follow the visible (carved) surface
  const hit = raycaster.intersectObjects(surfaceGroups, true)[0];
  if (!hit || !hit.face) {
    bur.visible = false;
    hover = null;
    updateHud(null);
    return;
  }
  hover = hit;
  const surface = hit.object.userData.surface as VoxelSurface;
  if (surface !== lastSurface) {
    lastSurface = surface;
    refreshUi(); // the margin buttons now act on this tooth
  }

  smoothNormal(hit, hoverNormal);
  hoverIsWall = axialPose(hoverNormal, hoverAxis, hoverInward);

  if (axialMode && hoverIsWall) {
    // Upright bur standing just outside the wall, its side touching the surface
    tipCentre.copy(hit.point).addScaledVector(hoverInward, -BUR.tipRadius);
    placeBur(tipCentre, hoverAxis, true, dt);
  } else {
    // Bur pointing straight out of the surface, its tip touching it
    tipCentre.copy(hit.point).addScaledVector(hoverNormal, BUR.tipRadius);
    placeBur(tipCentre, hoverNormal, true, dt);
  }

  updateHud(probe(surface, hit.point));
}

// Models are served from public/models. BASE_URL keeps this working when the site
// is hosted in a sub-folder (see vite.config.ts).
const MODELS = `${import.meta.env.BASE_URL}models/`;

const loading = document.createElement("div");
loading.id = "loading";
loading.textContent = "Preparing teeth...";
document.body.appendChild(loading);

async function init() {
  const [molar, incisor] = await Promise.all([
    loadTooth(`${MODELS}molar.stl`, {
      fit: { axis: "x", mm: 10.5 },
      rotateX: -Math.PI / 2,
    }),
    loadTooth(`${MODELS}maxillary_left_central_incisor.glb`, {
      fit: { axis: "y", mm: 23.5 },
      rotateX: Math.PI, // the scan is crown-down (as in the upper jaw); flip it crown-up like the molar
    }),
  ]);

  molar.position.x = -10;
  incisor.position.x = 10;
  scene.add(molar, incisor);

  teeth.push(molar, incisor);

  // Which reduction targets apply to which tooth
  const preps = new Map<THREE.Mesh, PrepSpec>([
    [molar, MOLAR_PREP],
    [incisor, INCISOR_PREP],
  ]);

  const VOXEL_SIZE = 0.1; // mm

  for (const tooth of teeth) {
    if (import.meta.env.DEV) console.log(tooth.name, checkWatertight(tooth.geometry));

    console.time(`voxelize ${tooth.name}`);
    const grid = voxelize(tooth.geometry, VOXEL_SIZE);
    console.timeEnd(`voxelize ${tooth.name}`);

    // Depth below the original surface, and which target region each voxel belongs to.
    // Both describe the ORIGINAL tooth, so they must be built before any carving.
    console.time(`depth field ${tooth.name}`);
    const { depth, nearestFace } = buildDepthField(tooth.geometry, grid);
    console.timeEnd(`depth field ${tooth.name}`);
    grid.depth = depth;
    grid.prep = preps.get(tooth)!;
    grid.nearestFace = nearestFace;
    grid.marginY = grid.prep.marginY;
    grid.region = classifyRegions(tooth.geometry, nearestFace, grid.prep, grid.marginY);

    // Replace the scanned mesh with a surface rebuilt from the voxels
    const material = (tooth.material as THREE.Material).clone();
    material.vertexColors = heatmapOn;
    (tooth.material as THREE.Material).visible = false;

    console.time(`mesh ${tooth.name}`);
    const surface = new VoxelSurface(grid, material);
    console.timeEnd(`mesh ${tooth.name}`);
    console.log(`${tooth.name}: ${surface.triangleCount().toLocaleString()} triangles`);

    tooth.add(surface.group);
    surfaces.push(surface);
    surfaceGroups.push(surface.group);

    // The voxel debug view (V) is only built in development - it costs load time and memory
    if (import.meta.env.DEV) {
      const points = createVoxelDebugPoints(grid);
      tooth.add(points);
      voxelViews.push(points);
    }
  }
}

// Start the render loop once everything it uses has been defined
let lastTime = 0;
renderer.setAnimationLoop((time) => {
  // Seconds since the last frame, capped so a pause never causes one giant cut
  const dt = Math.min((time - lastTime) / 1000, 0.05);
  lastTime = time;

  controls.update();
  updateBur(dt);

  // Keep the report fresh while it is open (twice a second is plenty)
  if (reportPanel.style.display !== "none") {
    reportTimer -= dt;
    if (reportTimer <= 0) {
      renderReport();
      reportTimer = 0.5;
    }
  }
  renderer.render(scene, camera);
});

init()
  .then(() => {
    loading.remove();
    refreshUi();
  })
  .catch((err) => {
    console.error("Failed to load teeth:", err);
    loading.textContent = "Could not load the teeth. Please reload the page.";
  });
