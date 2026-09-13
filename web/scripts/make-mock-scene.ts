#!/usr/bin/env tsx
/**
 * Generates a tiny mock 3DGS scene so the viewer's timeline, layers and
 * camera flythrough can be demonstrated end-to-end before the `recon` /
 * `procedural` / `camreg` workstreams publish real data.
 *
 * Rather than one file per epoch (which would re-encode the shared ground
 * plane six times over), each mock asset carries its own `t_start`/`t_end`
 * so the manifest's own time-gating produces the epoch transitions: both
 * towers standing -> one down -> both down -> rubble, exactly as `E0..E5`
 * in `wtc4d/timeline.py` describe. This exercises the same viewer code
 * path a real per-epoch manifest would.
 *
 * Run with `npm run make-mock-scene` from `web/`. Output goes to
 * `public/sample/`.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { hms, WTC1_COLLAPSE, WTC1_IMPACT, WTC2_COLLAPSE, WTC2_IMPACT } from "../src/lib/timeline";
import { towerBoxCorners, towerEnuCenter, WORLD_ORIGIN, WTC1, WTC2, type TowerSpec } from "../src/lib/world";
import type { CameraRef, SceneManifest, SplatAsset, TimelineEvent } from "../src/lib/schema";
import { mulberry32 } from "./mockRng";
import { type GaussianPoint, writePlyBinary } from "./ply";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(__dirname, "..", "public", "sample");

const rng = mulberry32(0x9110_9110);
const jitter = (amount: number) => (rng() * 2 - 1) * amount;

// --- Tower facades ---------------------------------------------------------

function towerFacade(tower: TowerSpec): GaussianPoint[] {
  const corners = towerBoxCorners(tower); // [4 bottom, 4 top]
  const points: GaussianPoint[] = [];
  const cols = 22; // along each face's width
  const rows = 90; // along height
  const colSpacing = tower.footprint_m / (cols - 1);
  const rowSpacing = tower.roof_height_m / (rows - 1);
  const scale: [number, number, number] = [colSpacing * 0.6, colSpacing * 0.6, rowSpacing * 0.7];

  for (let face = 0; face < 4; face++) {
    const a = corners[face];
    const b = corners[(face + 1) % 4];
    for (let u = 0; u < cols; u++) {
      const fu = u / (cols - 1);
      const bx = a.x + (b.x - a.x) * fu;
      const by = a.y + (b.y - a.y) * fu;
      for (let v = 0; v < rows; v++) {
        const fv = v / (rows - 1);
        const z = a.z + tower.roof_height_m * fv;
        // Sparse dark bands read as window rows against the lighter facade.
        const isWindowRow = v % 4 === 0;
        const shade = isWindowRow ? 0.32 : 0.58 + jitter(0.04);
        points.push({
          x: bx + jitter(0.15),
          y: by + jitter(0.15),
          z: z + jitter(0.15),
          color: [shade, shade + 0.02, shade + 0.05],
          opacity: 0.95,
          scale,
        });
      }
    }
  }

  // Roof cap.
  const center = towerEnuCenter(tower);
  const roofZ = center.z + tower.roof_height_m;
  const half = tower.footprint_m / 2;
  const roofGrid = 12;
  for (let i = 0; i < roofGrid; i++) {
    for (let j = 0; j < roofGrid; j++) {
      const fx = (i / (roofGrid - 1)) * 2 - 1;
      const fy = (j / (roofGrid - 1)) * 2 - 1;
      points.push({
        x: center.x + fx * half,
        y: center.y + fy * half,
        z: roofZ + jitter(0.1),
        color: [0.4, 0.41, 0.44],
        opacity: 0.95,
        scale: [half / roofGrid, half / roofGrid, 1],
      });
    }
  }

  // WTC1 has a rooftop antenna; a thin vertical line of points signals it.
  if (tower.top_height_m > tower.roof_height_m) {
    const antennaSteps = 24;
    for (let i = 0; i < antennaSteps; i++) {
      const f = i / (antennaSteps - 1);
      points.push({
        x: center.x,
        y: center.y,
        z: center.z + tower.roof_height_m + f * (tower.top_height_m - tower.roof_height_m),
        color: [0.7, 0.15, 0.1],
        opacity: 0.9,
        scale: [0.6, 0.6, 0.6],
      });
    }
  }

  return points;
}

// --- Rubble pile (post-collapse mound at a tower's footprint) --------------

function rubblePile(tower: TowerSpec, seed: number): GaussianPoint[] {
  const local = mulberry32(seed);
  const center = towerEnuCenter(tower);
  const radius = tower.footprint_m * 0.9;
  const maxHeight = 22; // metres, roughly the reported debris pile height
  const points: GaussianPoint[] = [];
  const n = 2200;
  for (let i = 0; i < n; i++) {
    const r = radius * Math.sqrt(local());
    const theta = local() * Math.PI * 2;
    const heightFalloff = 1 - r / radius;
    const z = center.z + Math.max(0, heightFalloff * maxHeight * (0.5 + local() * 0.5));
    const shade = 0.28 + local() * 0.12;
    points.push({
      x: center.x + Math.cos(theta) * r,
      y: center.y + Math.sin(theta) * r,
      z,
      color: [shade + 0.04, shade, shade - 0.03],
      opacity: 0.9,
      scale: [1.4, 1.4, 1.0],
    });
  }
  return points;
}

// --- Smoke plume (single mock puff, drifting SE per WIND_NOTES) -----------

function smokePlume(): GaussianPoint[] {
  const local = mulberry32(0x5a0c_e000);
  const origin = towerEnuCenter(WTC1);
  const points: GaussianPoint[] = [];
  const n = 1800;
  // Drift SE (world +X/-Y-ish blend) and rise, widening with height.
  const driftDir = { x: 0.6, y: -0.5 };
  for (let i = 0; i < n; i++) {
    const t = local(); // 0..1 along the plume's rise
    const rise = 30 + t * 260;
    const drift = t * 220;
    const spread = 20 + t * 90;
    const angle = local() * Math.PI * 2;
    const r = spread * Math.sqrt(local());
    points.push({
      x: origin.x + driftDir.x * drift + Math.cos(angle) * r,
      y: origin.y + driftDir.y * drift + Math.sin(angle) * r,
      z: origin.z + towerImpactHeight() + rise,
      color: [0.5 - t * 0.15, 0.49 - t * 0.15, 0.48 - t * 0.15],
      opacity: 0.35 * (1 - t * 0.4),
      scale: [spread * 0.25, spread * 0.25, spread * 0.2],
    });
  }
  return points;

  function towerImpactHeight(): number {
    const [lo, hi] = WTC1.impact_floors ?? [93, 99];
    const frac = (lo + hi) / 2 / WTC1.floors;
    return frac * WTC1.roof_height_m;
  }
}

// --- Ground disc -------------------------------------------------------

function groundDisc(): GaussianPoint[] {
  const local = mulberry32(0xd15c_0000);
  const points: GaussianPoint[] = [];
  const radius = 420;
  const n = 2600;
  for (let i = 0; i < n; i++) {
    const r = radius * Math.sqrt(local());
    const theta = local() * Math.PI * 2;
    const shade = 0.33 + local() * 0.05;
    points.push({
      x: Math.cos(theta) * r,
      y: Math.sin(theta) * r,
      z: -0.3,
      color: [shade, shade, shade + 0.01],
      opacity: 0.85,
      scale: [6, 6, 0.5],
    });
  }
  return points;
}

// --- Cameras (synthetic vantage points; no real footage yet) --------------
// Placeholder registry standing in for `camreg` output. Positions are
// plausible public vantage points; poses are a straight look-at toward the
// towers midpoint, not registered against real footage. Thumbnails are
// intentionally omitted (null) so the viewer's "no source frame available"
// fallback gets exercised.

interface Vec3 {
  x: number;
  y: number;
  z: number;
}
const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const cross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
const norm = (a: Vec3): Vec3 => {
  const len = Math.hypot(a.x, a.y, a.z) || 1;
  return { x: a.x / len, y: a.y / len, z: a.z / len };
};

/** Builds an OpenCV-convention (X right, Y down, Z forward) camera-to-world
 * matrix looking from `eye` toward `target`, assuming world +Z is up. */
function lookAtC2w(eye: Vec3, target: Vec3): number[] {
  const forward = norm(sub(target, eye));
  const worldUp: Vec3 = { x: 0, y: 0, z: 1 };
  const right = norm(cross(forward, worldUp));
  const down = norm(cross(forward, right));
  // prettier-ignore
  return [
    right.x, down.x, forward.x, eye.x,
    right.y, down.y, forward.y, eye.y,
    right.z, down.z, forward.z, eye.z,
    0, 0, 0, 1,
  ];
}

function makeCamera(
  id: string,
  label: string,
  t: number,
  eye: Vec3,
  target: Vec3,
  widthPx = 1280,
  heightPx = 720,
  hfovDeg = 40,
): CameraRef {
  const fx = widthPx / (2 * Math.tan((hfovDeg * Math.PI) / 360));
  return {
    id,
    shot_id: id,
    frame_idx: 0,
    t,
    c2w: lookAtC2w(eye, target),
    intrinsics: {
      width: widthPx,
      height: heightPx,
      fx,
      fy: fx,
      cx: widthPx / 2,
      cy: heightPx / 2,
      model: "OPENCV",
      dist: [],
    },
    thumbnail_url: null,
    source_url: null,
    label,
  };
}

function buildCameras(): CameraRef[] {
  const midpoint = {
    x: (towerEnuCenter(WTC1).x + towerEnuCenter(WTC2).x) / 2,
    y: (towerEnuCenter(WTC1).y + towerEnuCenter(WTC2).y) / 2,
    z: 200,
  };
  return [
    makeCamera(
      "cam_brooklyn_promenade",
      "Brooklyn Heights Promenade (mock)",
      WTC1_IMPACT.t + 120,
      { x: 1400, y: -1900, z: 35 },
      midpoint,
    ),
    makeCamera(
      "cam_jersey_city",
      "Jersey City waterfront (mock)",
      WTC2_IMPACT.t + 30,
      { x: -1600, y: 150, z: 20 },
      midpoint,
    ),
    makeCamera(
      "cam_helicopter",
      "News helicopter, orbiting (mock)",
      WTC2_IMPACT.t + 400,
      { x: 600, y: 1200, z: 650 },
      midpoint,
    ),
    makeCamera(
      "cam_street_level",
      "Church St, street level (mock)",
      WTC1_COLLAPSE.t - 60,
      { x: 250, y: 300, z: 2 },
      { ...towerEnuCenter(WTC1), z: 150 },
      1280,
      960,
      55,
    ),
    makeCamera(
      "cam_midtown",
      "Midtown rooftop, long lens (mock)",
      WTC2_COLLAPSE.t + 20,
      { x: 2200, y: 6400, z: 210 },
      midpoint,
      1280,
      720,
      12,
    ),
  ];
}

// --- Assemble manifest ---------------------------------------------------

const T_MIN = hms(8, 0, 0);
const T_MAX = hms(12, 30, 0);

function main() {
  mkdirSync(OUT_DIR, { recursive: true });

  const files: Record<string, GaussianPoint[]> = {
    "ground.ply": groundDisc(),
    "wtc1_tower.ply": towerFacade(WTC1),
    "wtc2_tower.ply": towerFacade(WTC2),
    "wtc1_rubble.ply": rubblePile(WTC1, 0x1001),
    "wtc2_rubble.ply": rubblePile(WTC2, 0x1002),
    "smoke_plume.ply": smokePlume(),
  };

  for (const [name, points] of Object.entries(files)) {
    const buf = writePlyBinary(points);
    writeFileSync(join(OUT_DIR, name), buf);
    const kb = (buf.length / 1024).toFixed(1);
    console.log(`wrote ${name}: ${points.length} points, ${kb} KiB`);
  }

  const events: TimelineEvent[] = [
    { id: "wtc1_impact", name: "AA11 strikes WTC1 (North Tower)", t: WTC1_IMPACT.t, sigma: WTC1_IMPACT.sigma },
    { id: "wtc2_impact", name: "UA175 strikes WTC2 (South Tower)", t: WTC2_IMPACT.t, sigma: WTC2_IMPACT.sigma },
    { id: "wtc2_collapse", name: "WTC2 collapse", t: WTC2_COLLAPSE.t, sigma: WTC2_COLLAPSE.sigma },
    { id: "wtc1_collapse", name: "WTC1 collapse", t: WTC1_COLLAPSE.t, sigma: WTC1_COLLAPSE.sigma },
  ];

  const assets: SplatAsset[] = [
    {
      id: "ground",
      url: "ground.ply",
      format: "ply",
      t_start: T_MIN,
      t_end: T_MAX,
      kind: "static",
      layer: "ground",
      epoch_id: null,
      notes: "Mock plaza/street disc, constant through the whole window.",
    },
    {
      id: "wtc1_tower",
      url: "wtc1_tower.ply",
      format: "ply",
      t_start: T_MIN,
      t_end: WTC1_COLLAPSE.t,
      kind: "static",
      layer: "towers",
      epoch_id: null,
      notes: "Mock WTC1 facade box; visible until its collapse initiation.",
    },
    {
      id: "wtc2_tower",
      url: "wtc2_tower.ply",
      format: "ply",
      t_start: T_MIN,
      t_end: WTC2_COLLAPSE.t,
      kind: "static",
      layer: "towers",
      epoch_id: null,
      notes: "Mock WTC2 facade box; visible until its collapse initiation.",
    },
    {
      id: "wtc1_rubble",
      url: "wtc1_rubble.ply",
      format: "ply",
      t_start: WTC1_COLLAPSE.t,
      t_end: T_MAX,
      kind: "static",
      layer: "debris",
      epoch_id: null,
      notes: "Mock debris mound at the WTC1 footprint after collapse.",
    },
    {
      id: "wtc2_rubble",
      url: "wtc2_rubble.ply",
      format: "ply",
      t_start: WTC2_COLLAPSE.t,
      t_end: T_MAX,
      kind: "static",
      layer: "debris",
      epoch_id: null,
      notes: "Mock debris mound at the WTC2 footprint after collapse.",
    },
    {
      id: "smoke_plume",
      url: "smoke_plume.ply",
      format: "ply",
      t_start: WTC1_IMPACT.t,
      t_end: T_MAX,
      kind: "static",
      layer: "smoke",
      epoch_id: null,
      notes: "Single mock plume puff; not time-varying. Real smoke is a `procedural` deliverable.",
    },
  ];

  const manifest: SceneManifest = {
    schema_version: 1,
    name: "911_4D sample scene (mock)",
    world_origin: WORLD_ORIGIN,
    t_min: T_MIN,
    t_max: T_MAX,
    events,
    assets,
    cameras: buildCameras(),
    notes:
      "Synthetic placeholder scene generated by web/scripts/make-mock-scene.ts. " +
      "Geometry is illustrative only (simple boxes/mounds/blobs at correct ENU " +
      "positions and heights from wtc4d/world.py) — not a reconstruction. " +
      "Camera poses are straight look-at shots from plausible vantage points, " +
      "not registered against real footage.",
  };

  writeFileSync(join(OUT_DIR, "manifest.json"), JSON.stringify(manifest, null, 2));
  console.log(`wrote manifest.json`);
}

main();
