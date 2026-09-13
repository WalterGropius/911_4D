/**
 * Builds a small wireframe frustum (apex + far-plane rectangle) representing
 * a registered footage camera, for display in the 3D scene.
 */
import { BufferGeometry, Color, Float32BufferAttribute, LineBasicMaterial, LineSegments } from "three";
import { cameraRefToThree } from "../lib/cameraPose";
import type { CameraRef } from "../lib/schema";

const DEPTH_M = 18;

export function buildCameraFrustum(ref: CameraRef, color: Color = new Color(0x6fb3d8)): LineSegments {
  const { position, quaternion, fov, aspect } = cameraRefToThree(ref);
  const halfHeight = DEPTH_M * Math.tan((fov * Math.PI) / 360);
  const halfWidth = halfHeight * aspect;

  // prettier-ignore
  const localCorners = [
    [-halfWidth, -halfHeight, -DEPTH_M],
    [ halfWidth, -halfHeight, -DEPTH_M],
    [ halfWidth,  halfHeight, -DEPTH_M],
    [-halfWidth,  halfHeight, -DEPTH_M],
  ];

  const positions: number[] = [];
  const apex: [number, number, number] = [0, 0, 0];
  // Apex -> each far corner.
  for (const c of localCorners) {
    positions.push(...apex, ...c);
  }
  // Far-plane rectangle.
  for (let i = 0; i < 4; i++) {
    positions.push(...localCorners[i], ...localCorners[(i + 1) % 4]);
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(positions, 3));
  const material = new LineBasicMaterial({ color, transparent: true, opacity: 1 });
  const lines = new LineSegments(geometry, material);
  lines.position.copy(position);
  lines.quaternion.copy(quaternion);
  lines.name = `frustum:${ref.id}`;
  lines.userData.cameraId = ref.id;
  return lines;
}

/** Colour a frustum by how far its capture time is from the timeline's
 * overall span, so recent/near-current cameras read distinctly. Simple
 * warm (early) -> cool (late) ramp; callers also fade opacity separately. */
export function colorForCameraTime(t: number, tMin: number, tMax: number): Color {
  const f = tMax > tMin ? Math.min(1, Math.max(0, (t - tMin) / (tMax - tMin))) : 0;
  const c = new Color();
  c.setHSL(0.58 - f * 0.5, 0.65, 0.6);
  return c;
}
