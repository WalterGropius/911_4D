/**
 * Minimal writer for the standard 3D Gaussian Splatting `.ply` format (the
 * one produced by the original INRIA `gaussian-splatting` code and read by
 * both `@sparkjsdev/spark` and `@mkkellogg/gaussian-splats-3d`). Binary
 * little-endian, SH degree 0 only (`f_dc_0..2`, no `f_rest_*`).
 *
 * Field encodings match the reference implementation:
 *  - color is stored as the degree-0 spherical harmonic coefficient:
 *    `f_dc = (linear_color - 0.5) / SH_C0`
 *  - opacity is stored pre-sigmoid: `raw = logit(opacity)`
 *  - scale is stored as `log(scale_metres)` (post-exp gives the Gaussian's
 *    std-dev along each local axis, in world metres)
 *  - rotation is an unnormalized (w,x,y,z) quaternion, camera/world axes
 *    agnostic (Gaussians are isotropic enough here that identity is fine)
 */

export const SH_C0 = 0.28209479177387814;

export interface GaussianPoint {
  x: number;
  y: number;
  z: number;
  /** Linear RGB, 0..1. */
  color: [number, number, number];
  /** 0..1. */
  opacity: number;
  /** Per-axis standard deviation, metres, in the point's local (usually
   * axis-aligned) frame. */
  scale: [number, number, number];
  /** (w, x, y, z), defaults to identity. */
  rot?: [number, number, number, number];
}

function logit(p: number): number {
  const clamped = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(clamped / (1 - clamped));
}

const PROPERTIES = [
  "x",
  "y",
  "z",
  "nx",
  "ny",
  "nz",
  "f_dc_0",
  "f_dc_1",
  "f_dc_2",
  "opacity",
  "scale_0",
  "scale_1",
  "scale_2",
  "rot_0",
  "rot_1",
  "rot_2",
  "rot_3",
] as const;

export function writePlyBinary(points: readonly GaussianPoint[]): Buffer {
  const header =
    `ply\n` +
    `format binary_little_endian 1.0\n` +
    `element vertex ${points.length}\n` +
    PROPERTIES.map((p) => `property float ${p}\n`).join("") +
    `end_header\n`;
  const headerBuf = Buffer.from(header, "ascii");
  const bytesPerPoint = PROPERTIES.length * 4;
  const body = Buffer.alloc(points.length * bytesPerPoint);

  points.forEach((pt, i) => {
    const off = i * bytesPerPoint;
    const [r, g, b] = pt.color;
    const [sx, sy, sz] = pt.scale;
    const [qw, qx, qy, qz] = pt.rot ?? [1, 0, 0, 0];
    const values = [
      pt.x,
      pt.y,
      pt.z,
      0,
      0,
      1, // normal: unused by splat renderers, filled as +Z
      (r - 0.5) / SH_C0,
      (g - 0.5) / SH_C0,
      (b - 0.5) / SH_C0,
      logit(pt.opacity),
      Math.log(Math.max(sx, 1e-6)),
      Math.log(Math.max(sy, 1e-6)),
      Math.log(Math.max(sz, 1e-6)),
      qw,
      qx,
      qy,
      qz,
    ];
    values.forEach((v, j) => body.writeFloatLE(v, off + j * 4));
  });

  return Buffer.concat([headerBuf, body]);
}
