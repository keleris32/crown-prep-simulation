import * as THREE from "three";
import { NO_REGION, type PrepRegion, type PrepSpec } from "./prep";

// "depth": colour by reduction vs that region's target. "region": colour by region.
export type ColorMode = "depth" | "region";
let colorMode: ColorMode = "depth";
export const getColorMode = () => colorMode;
export const setColorMode = (mode: ColorMode) => {
  colorMode = mode;
};

const UNTOUCHED = new THREE.Color(0xffffff); // white = no tint on the enamel
const UNDER = new THREE.Color(0xffd54f);
const ON_TARGET = new THREE.Color(0x66bb6a);
const OVER = new THREE.Color(0xe53935);
const MARGIN_COLOR = new THREE.Color(0x8d8d8d);
const REGION_COLORS = [new THREE.Color(0x42a5f5), new THREE.Color(0xab47bc), new THREE.Color(0xffa726)];

const blend = (out: THREE.Color, from: THREE.Color, to: THREE.Color, t: number) =>
  out.copy(from).lerp(to, THREE.MathUtils.clamp(t, 0, 1));

// Colour of one surface vertex: depth = mm below the original surface, region = target zone
export function vertexColor(
  depth: number,
  region: number,
  prep: PrepSpec | undefined,
  out: THREE.Color,
): THREE.Color {
  const spec = prep && region !== NO_REGION ? prep.regions[region] : undefined;

  if (colorMode === "region") {
    if (!spec || spec.kind === "margin") return out.copy(MARGIN_COLOR);
    return out.copy(REGION_COLORS[region % REGION_COLORS.length]);
  }

  if (depth <= 0.05) return out.copy(UNTOUCHED);
  if (spec?.kind === "margin") return out.copy(OVER); // any cut below the finish line is wrong

  const { min, max } = spec ?? { min: 1.0, max: 2.0 }; // fallback if the region is unknown
  if (depth < min / 2) return blend(out, UNTOUCHED, UNDER, (depth - 0.05) / (min / 2 - 0.05));
  if (depth < min) return blend(out, UNDER, ON_TARGET, (depth - min / 2) / (min / 2));
  if (depth <= max) return out.copy(ON_TARGET);
  return blend(out, ON_TARGET, OVER, (depth - max) / 0.3);
}

export type Status = "untouched" | "under" | "on" | "over";

// Shared by the HUD and the report. A small tolerance absorbs the 0.1 mm voxel size.
export function reductionStatus(depth: number, region: PrepRegion): Status {
  if (depth <= 0.05) return "untouched";
  if (region.kind === "margin") return "over";
  if (depth < region.min - 0.05) return "under";
  if (depth > region.max + 0.05) return "over";
  return "on";
}

export const DEPTH_LEGEND = [
  { css: "#ffffff", label: "Untouched" },
  { css: "#ffd54f", label: "Under target" },
  { css: "#66bb6a", label: "On target" },
  { css: "#e53935", label: "Over target / below margin" },
];

export const REGION_LEGEND = [
  { css: "#42a5f5", label: "Occlusal / Incisal" },
  { css: "#ab47bc", label: "Axial / Labial & proximal" },
  { css: "#ffa726", label: "Lingual (incisor)" },
  { css: "#8d8d8d", label: "Below margin (do not cut)" },
];
