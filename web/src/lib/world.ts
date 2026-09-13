/**
 * World coordinate frame, tower geometry and landmark registry.
 *
 * Ported from `wtc4d/world.py` — keep in sync with that module's constants
 * and maths. Do not diverge without updating both sides.
 *
 * World frame: local East-North-Up (ENU) tangent frame, in metres, with
 * origin at {@link WORLD_ORIGIN}. +X = east, +Y = north, +Z = up.
 *
 * Camera convention: OpenCV / COLMAP pinhole, camera +X right, +Y down,
 * +Z forward. Poses are stored camera-to-world (`c2w`), 4x4 row-major,
 * flattened to 16 numbers (see `src/lib/schema.ts`).
 */

export interface LatLonAlt {
  lat: number;
  lon: number;
  alt_m: number;
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

// --- WGS84 -------------------------------------------------------------
const A = 6378137.0;
const F = 1.0 / 298.257223563;
const E2 = F * (2.0 - F);

/** ENU origin: between the towers, at approximately mean sea level. */
export const WORLD_ORIGIN: LatLonAlt = { lat: 40.7112, lon: -74.0132, alt_m: 0.0 };

function toRad(deg: number): number {
  return (deg * Math.PI) / 180;
}
function toDeg(rad: number): number {
  return (rad * 180) / Math.PI;
}

export function geodeticToEcef(p: LatLonAlt): Vec3 {
  const lat = toRad(p.lat);
  const lon = toRad(p.lon);
  const n = A / Math.sqrt(1.0 - E2 * Math.sin(lat) ** 2);
  const x = (n + p.alt_m) * Math.cos(lat) * Math.cos(lon);
  const y = (n + p.alt_m) * Math.cos(lat) * Math.sin(lon);
  const z = (n * (1.0 - E2) + p.alt_m) * Math.sin(lat);
  return { x, y, z };
}

/** 3x3 rotation matrix (row-major, flattened) taking ECEF deltas to ENU. */
function enuRotation(origin: LatLonAlt): number[] {
  const lat = toRad(origin.lat);
  const lon = toRad(origin.lon);
  const sl = Math.sin(lat);
  const cl = Math.cos(lat);
  const so = Math.sin(lon);
  const co = Math.cos(lon);
  // prettier-ignore
  return [
    -so,       co,      0.0,
    -sl * co, -sl * so, cl,
     cl * co,  cl * so, sl,
  ];
}

function matVec3(m: number[], v: Vec3): Vec3 {
  return {
    x: m[0] * v.x + m[1] * v.y + m[2] * v.z,
    y: m[3] * v.x + m[4] * v.y + m[5] * v.z,
    z: m[6] * v.x + m[7] * v.y + m[8] * v.z,
  };
}

/** Transpose-multiply: `m^T @ v`, used to go from ENU back to ECEF deltas. */
function matTVec3(m: number[], v: Vec3): Vec3 {
  return {
    x: m[0] * v.x + m[3] * v.y + m[6] * v.z,
    y: m[1] * v.x + m[4] * v.y + m[7] * v.z,
    z: m[2] * v.x + m[5] * v.y + m[8] * v.z,
  };
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}
function add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

/** Geodetic -> world ENU (metres). */
export function latLonToEnu(p: LatLonAlt, origin: LatLonAlt = WORLD_ORIGIN): Vec3 {
  const d = sub(geodeticToEcef(p), geodeticToEcef(origin));
  return matVec3(enuRotation(origin), d);
}

/** World ENU (metres) -> geodetic. Iterative ECEF -> geodetic (Bowring-style). */
export function enuToLatLon(xyz: Vec3, origin: LatLonAlt = WORLD_ORIGIN): LatLonAlt {
  const rot = enuRotation(origin);
  const ecef = add(geodeticToEcef(origin), matTVec3(rot, xyz));
  const { x, y, z } = ecef;
  const lon = Math.atan2(y, x);
  const p = Math.hypot(x, y);
  let lat = Math.atan2(z, p * (1.0 - E2));
  let alt = 0;
  for (let i = 0; i < 10; i++) {
    const n = A / Math.sqrt(1.0 - E2 * Math.sin(lat) ** 2);
    alt = p / Math.cos(lat) - n;
    lat = Math.atan2(z, p * (1.0 - (E2 * n) / (n + alt)));
  }
  const n = A / Math.sqrt(1.0 - E2 * Math.sin(lat) ** 2);
  alt = p / Math.cos(lat) - n;
  return { lat: toDeg(lat), lon: toDeg(lon), alt_m: alt };
}

// --- Towers --------------------------------------------------------------
export interface TowerSpec {
  id: string;
  name: string;
  center: LatLonAlt;
  footprint_m: number;
  roof_height_m: number;
  top_height_m: number;
  rotation_deg: number;
  floors: number;
  impact_floors?: [number, number];
  notes?: string;
}

export function towerEnuCenter(t: TowerSpec): Vec3 {
  return latLonToEnu(t.center);
}

/** (8, 3) ENU corners of the tower as a rotated box from plaza to roof: 4
 * bottom corners followed by 4 top corners. */
export function towerBoxCorners(t: TowerSpec): Vec3[] {
  const c = towerEnuCenter(t);
  const h = t.footprint_m / 2.0;
  const r = toRad(t.rotation_deg);
  const cr = Math.cos(r);
  const sr = Math.sin(r);
  const square: [number, number][] = [
    [-h, -h],
    [h, -h],
    [h, h],
    [-h, h],
  ];
  const rotated = square.map(([sx, sy]): [number, number] => [
    sx * cr - sy * sr,
    sx * sr + sy * cr,
  ]);
  const bottom = rotated.map(([rx, ry]) => ({ x: rx + c.x, y: ry + c.y, z: c.z }));
  const top = rotated.map(([rx, ry]) => ({ x: rx + c.x, y: ry + c.y, z: c.z + t.roof_height_m }));
  return [...bottom, ...top];
}

// Both towers: 207 ft (63.1 m) square, 110 storeys. Centre coordinates are
// approx (memorial pools mark the footprints). See `wtc4d/world.py` for
// sourcing notes; the geo workstream owns verifying these.
export const WTC1: TowerSpec = {
  id: "WTC1",
  name: "One World Trade Center (North Tower)",
  center: { lat: 40.71195, lon: -74.01338, alt_m: 0.0 },
  footprint_m: 63.1,
  roof_height_m: 417.0,
  top_height_m: 526.3,
  rotation_deg: 0.0,
  floors: 110,
  impact_floors: [93, 99],
  notes: "Approx centre; verify against memorial North Pool footprint.",
};

export const WTC2: TowerSpec = {
  id: "WTC2",
  name: "Two World Trade Center (South Tower)",
  center: { lat: 40.71055, lon: -74.01285, alt_m: 0.0 },
  footprint_m: 63.1,
  roof_height_m: 415.1,
  top_height_m: 415.1,
  rotation_deg: 0.0,
  floors: 110,
  impact_floors: [77, 85],
  notes: "Approx centre; verify against memorial South Pool footprint.",
};

export const TOWERS: TowerSpec[] = [WTC1, WTC2];
