/**
 * Conversion between the project's camera convention (OpenCV axes,
 * camera-to-world, see `wtc4d/schema/camera.py`) and Three.js.
 *
 * World frame: the viewer keeps Three.js world coordinates identical to the
 * project's ENU frame (+X east, +Y north, +Z up) — i.e. Z-up, not Three's
 * default Y-up. `THREE.Object3D.DefaultUp` is set to (0,0,1) at startup
 * (see `src/main.ts`) so new objects/cameras get the right default.
 *
 * Camera axis conversion: OpenCV camera space is +X right, +Y down, +Z
 * forward. Three.js camera space is +X right, +Y up, +Z backward (the
 * camera looks down -Z). Both share the same origin and world frame, so
 * converting only needs to flip the local Y and Z basis columns:
 *
 *   R_three = R_opencv @ diag(1, -1, -1)      (translation unchanged)
 *
 * Because `diag(1,-1,-1,1)` only scales columns of the 4x4 c2w, in row-major
 * form this is just negating the Y/Z entries of the rotation part (columns
 * 1 and 2) on every row, leaving column 0 (basis X) and column 3
 * (translation) untouched.
 */
import { Matrix4, Quaternion, Vector3 } from "three";
import type { CameraIntrinsics, CameraRef } from "./schema";

export interface PoseTransform {
  position: Vector3;
  quaternion: Quaternion;
}

/** `c2w`: 16 numbers, row-major, OpenCV camera axes, world = project ENU. */
export function c2wToThreePose(c2w: readonly number[]): PoseTransform {
  if (c2w.length !== 16) {
    throw new Error(`c2w must have 16 elements, got ${c2w.length}`);
  }
  const [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz] = c2w;
  const m = new Matrix4();
  // prettier-ignore
  m.set(
    r00, -r01, -r02, tx,
    r10, -r11, -r12, ty,
    r20, -r21, -r22, tz,
    0, 0, 0, 1,
  );
  const position = new Vector3();
  const quaternion = new Quaternion();
  const scale = new Vector3();
  m.decompose(position, quaternion, scale);
  return { position, quaternion };
}

/** Vertical FOV (degrees) implied by pixel intrinsics, matching
 * `THREE.PerspectiveCamera.fov` (vertical). */
export function verticalFovDeg(intrinsics: Pick<CameraIntrinsics, "height" | "fy">): number {
  return (2 * Math.atan(intrinsics.height / (2 * intrinsics.fy)) * 180) / Math.PI;
}

export function aspectFromIntrinsics(intrinsics: Pick<CameraIntrinsics, "width" | "height">): number {
  return intrinsics.width / intrinsics.height;
}

/** Everything needed to point a `THREE.PerspectiveCamera` at a registered
 * footage viewpoint: pose, fov and aspect. Does not touch a live camera
 * object so it stays unit-testable without a renderer. */
export function cameraRefToThree(ref: CameraRef): PoseTransform & { fov: number; aspect: number } {
  const pose = c2wToThreePose(ref.c2w);
  return {
    ...pose,
    fov: verticalFovDeg(ref.intrinsics),
    aspect: aspectFromIntrinsics(ref.intrinsics),
  };
}
