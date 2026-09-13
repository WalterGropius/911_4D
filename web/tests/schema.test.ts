import { describe, expect, it } from "vitest";
import { ManifestValidationError, parseManifest } from "../src/lib/schema";

function validManifest() {
  return {
    schema_version: 1,
    name: "test",
    world_origin: { lat: 40.7112, lon: -74.0132, alt_m: 0 },
    t_min: 0,
    t_max: 100,
    events: [{ id: "e1", name: "Event", t: 50, sigma: 1 }],
    assets: [
      {
        id: "a1",
        url: "a1.ply",
        format: "ply",
        t_start: 0,
        t_end: 100,
        kind: "static",
        layer: "towers",
      },
    ],
    cameras: [
      {
        id: "c1",
        shot_id: "s1",
        frame_idx: 0,
        t: 10,
        c2w: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        intrinsics: { width: 100, height: 100, fx: 100, fy: 100, cx: 50, cy: 50 },
      },
    ],
  };
}

describe("parseManifest", () => {
  it("accepts a well-formed manifest and fills in defaults", () => {
    const parsed = parseManifest(validManifest());
    expect(parsed.assets[0].format).toBe("ply");
    expect(parsed.cameras[0].label).toBe("");
    expect(parsed.notes).toBe("");
  });

  it("rejects a manifest missing required fields", () => {
    const bad = validManifest();
    // @ts-expect-error intentionally malformed for the test
    delete bad.world_origin;
    expect(() => parseManifest(bad)).toThrow(ManifestValidationError);
  });

  it("rejects a c2w matrix that isn't length 16", () => {
    const bad = validManifest();
    bad.cameras[0].c2w = [1, 0, 0];
    expect(() => parseManifest(bad)).toThrow();
  });

  it("rejects an out-of-range latitude", () => {
    const bad = validManifest();
    bad.world_origin.lat = 999;
    expect(() => parseManifest(bad)).toThrow();
  });

  it("error message includes the offending path", () => {
    const bad = validManifest();
    bad.assets[0].t_start = "not a number" as unknown as number;
    try {
      parseManifest(bad);
      throw new Error("expected parseManifest to throw");
    } catch (e) {
      expect(e).toBeInstanceOf(ManifestValidationError);
      expect((e as ManifestValidationError).message).toContain("assets.0.t_start");
    }
  });
});
