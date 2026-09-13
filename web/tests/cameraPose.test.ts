import { describe, expect, it } from "vitest";
import {
  aspectFromIntrinsics,
  c2wToThreePose,
  cameraRefToThree,
  verticalFovDeg,
} from "../src/lib/cameraPose";
import type { CameraRef } from "../src/lib/schema";

// Identity-rotation camera: OpenCV +X right, +Y down, +Z forward, sitting at
// world (10, 20, 30). Hand-computed expectation: since Three camera space is
// +X right, +Y up, +Z backward, an identity-oriented OpenCV camera (looking
// along world +Z, "down" toward world +Y) must end up looking along world
// -Z... but here rotation is identity meaning camera axes == world axes, so
// the camera looks along +Z (world), "down" is world +Y, "right" is world +X.
// For Three to reproduce the same look direction (+Z world) and same "down"
// screen direction, its local -Z (forward) must equal world +Z, and its
// local +Y (up) must equal world -Y (since OpenCV's +Y-down = -1 * three's
// +Y-up direction in world space, for identity rotation both are simply
// world axes with sign flips baked into the columns).
describe("c2wToThreePose", () => {
  it("keeps translation unchanged", () => {
    // prettier-ignore
    const c2w = [
      1, 0, 0, 10,
      0, 1, 0, 20,
      0, 0, 1, 30,
      0, 0, 0, 1,
    ];
    const { position } = c2wToThreePose(c2w);
    expect(position.x).toBeCloseTo(10, 9);
    expect(position.y).toBeCloseTo(20, 9);
    expect(position.z).toBeCloseTo(30, 9);
  });

  it("maps an identity-rotation OpenCV camera to Three's forward/up convention", () => {
    // prettier-ignore
    const c2w = [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ];
    const { quaternion } = c2wToThreePose(c2w);
    // Three camera looks along local -Z and "up" is local +Y. Rotate those
    // by the resulting quaternion and check them against the OpenCV axes:
    // world forward must equal OpenCV local +Z (world +Z here), and world
    // up must equal OpenCV local -Y (world -Y here, since +Y is "down").
    const forward = { x: 0, y: 0, z: -1 };
    const rotated = rotateVec(quaternion, forward);
    expect(rotated.x).toBeCloseTo(0, 9);
    expect(rotated.y).toBeCloseTo(0, 9);
    expect(rotated.z).toBeCloseTo(1, 9);

    const up = { x: 0, y: 1, z: 0 };
    const rotatedUp = rotateVec(quaternion, up);
    expect(rotatedUp.x).toBeCloseTo(0, 9);
    expect(rotatedUp.y).toBeCloseTo(-1, 9);
    expect(rotatedUp.z).toBeCloseTo(0, 9);
  });

  it("a 90deg-about-world-Z OpenCV rotation preserves handedness (right stays right)", () => {
    // OpenCV camera rotated 90deg so its local +X (right) points along
    // world +Y, local +Y (down) points along world -X, local +Z (forward)
    // stays world +Z.
    // prettier-ignore
    const c2w = [
      0, -1, 0, 0,
      1,  0, 0, 0,
      0,  0, 1, 0,
      0,  0, 0, 1,
    ];
    const { quaternion } = c2wToThreePose(c2w);
    const right = rotateVec(quaternion, { x: 1, y: 0, z: 0 });
    expect(right.x).toBeCloseTo(0, 9);
    expect(right.y).toBeCloseTo(1, 9);
    expect(right.z).toBeCloseTo(0, 9);
  });

  it("throws on malformed input", () => {
    expect(() => c2wToThreePose([1, 2, 3])).toThrow();
  });
});

describe("verticalFovDeg / aspectFromIntrinsics", () => {
  it("computes vertical fov matching a hand-computed pinhole case", () => {
    // height=1080, fy=1080 -> vfov = 2*atan(0.5) ~= 53.13 deg
    const fov = verticalFovDeg({ height: 1080, fy: 1080 });
    expect(fov).toBeCloseTo(53.130102, 4);
  });

  it("computes aspect ratio", () => {
    expect(aspectFromIntrinsics({ width: 1920, height: 1080 })).toBeCloseTo(16 / 9, 9);
  });
});

describe("cameraRefToThree", () => {
  it("combines pose, fov and aspect", () => {
    const ref: CameraRef = {
      id: "cam1",
      shot_id: "shot1",
      frame_idx: 0,
      t: 100,
      // prettier-ignore
      c2w: [
        1, 0, 0, 5,
        0, 1, 0, 6,
        0, 0, 1, 7,
        0, 0, 0, 1,
      ],
      intrinsics: { width: 1920, height: 1080, fx: 1000, fy: 1000, cx: 960, cy: 540, model: "OPENCV", dist: [] },
      thumbnail_url: null,
      source_url: null,
      label: "",
    };
    const result = cameraRefToThree(ref);
    expect(result.position.x).toBeCloseTo(5, 9);
    expect(result.aspect).toBeCloseTo(1920 / 1080, 9);
    expect(result.fov).toBeCloseTo(verticalFovDeg(ref.intrinsics), 9);
  });
});

function rotateVec(q: { x: number; y: number; z: number; w: number }, v: { x: number; y: number; z: number }) {
  // Standard quaternion rotation v' = q * v * q^-1, implemented directly so
  // this test does not depend on three's own Vector3.applyQuaternion.
  const { x: qx, y: qy, z: qz, w: qw } = q;
  const { x, y, z } = v;
  const ix = qw * x + qy * z - qz * y;
  const iy = qw * y + qz * x - qx * z;
  const iz = qw * z + qx * y - qy * x;
  const iw = -qx * x - qy * y - qz * z;
  return {
    x: ix * qw + iw * -qx + iy * -qz - iz * -qy,
    y: iy * qw + iw * -qy + iz * -qx - ix * -qz,
    z: iz * qw + iw * -qz + ix * -qy - iy * -qx,
  };
}
