import { describe, expect, it } from "vitest";
import {
  distinctLayers,
  selectPreloadCandidates,
  selectVisibleAssets,
  selectVisibleCameras,
} from "../src/lib/sceneState";
import type { CameraRef, SplatAsset } from "../src/lib/schema";

function asset(
  id: string,
  t_start: number,
  t_end: number,
  layer: SplatAsset["layer"] = "scene",
): SplatAsset {
  return { id, url: `${id}.ply`, format: "ply", t_start, t_end, kind: "static", layer, epoch_id: null, notes: "" };
}

describe("selectVisibleAssets", () => {
  const assets = [asset("a", 0, 10), asset("b", 10, 20)];

  it("selects the asset whose window contains t", () => {
    const visible = selectVisibleAssets(assets, 5, 1);
    expect(visible.map((v) => v.asset.id)).toEqual(["a"]);
  });

  it("is half-open: t_end is exclusive, t_start is inclusive", () => {
    expect(selectVisibleAssets(assets, 10, 0).map((v) => v.asset.id)).toEqual(["b"]);
    expect(selectVisibleAssets(assets, 0, 0).map((v) => v.asset.id)).toEqual(["a"]);
  });

  it("fades in after t_start and out before t_end", () => {
    const [near_start] = selectVisibleAssets(assets, 0.5, 2);
    expect(near_start.opacity).toBeCloseTo(0.25, 6);

    const [near_end] = selectVisibleAssets(assets, 9, 2);
    expect(near_end.opacity).toBeCloseTo(0.5, 6);

    const [mid] = selectVisibleAssets(assets, 5, 2);
    expect(mid.opacity).toBeCloseTo(1, 6);
  });

  it("clamps opacity to [0,1] even with a crossfade longer than the asset", () => {
    const short = [asset("s", 0, 1)];
    const [v] = selectVisibleAssets(short, 0.5, 10);
    expect(v.opacity).toBeGreaterThanOrEqual(0);
    expect(v.opacity).toBeLessThanOrEqual(1);
  });

  it("returns nothing outside every asset window", () => {
    expect(selectVisibleAssets(assets, 25, 1)).toEqual([]);
  });
});

describe("selectPreloadCandidates", () => {
  it("returns assets starting soon but not yet visible", () => {
    const assets = [asset("a", 0, 10), asset("b", 12, 20)];
    const candidates = selectPreloadCandidates(assets, 9, 5);
    expect(candidates.map((a) => a.id)).toEqual(["b"]);
  });

  it("excludes assets already visible", () => {
    const assets = [asset("a", 0, 10)];
    expect(selectPreloadCandidates(assets, 5, 5)).toEqual([]);
  });
});

function camRef(id: string, t: number): CameraRef {
  return {
    id,
    shot_id: id,
    frame_idx: 0,
    t,
    c2w: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    intrinsics: { width: 100, height: 100, fx: 100, fy: 100, cx: 50, cy: 50, model: "OPENCV", dist: [] },
    thumbnail_url: null,
    source_url: null,
    label: "",
  };
}

describe("selectVisibleCameras", () => {
  it("selects cameras within the time window and fades by recency", () => {
    const cams = [camRef("near", 100), camRef("far", 200)];
    const visible = selectVisibleCameras(cams, 100, 20);
    expect(visible.map((v) => v.camera.id)).toEqual(["near"]);
    expect(visible[0].opacity).toBeCloseTo(1, 6);
  });

  it("computes partial opacity for cameras mid-window", () => {
    const cams = [camRef("c", 110)];
    const [v] = selectVisibleCameras(cams, 100, 20);
    expect(v.opacity).toBeCloseTo(0.5, 6);
  });
});

describe("distinctLayers", () => {
  it("preserves first-seen order and de-duplicates", () => {
    const assets = [asset("a", 0, 1, "towers"), asset("b", 0, 1, "smoke"), asset("c", 0, 1, "towers")];
    expect(distinctLayers(assets)).toEqual(["towers", "smoke"]);
  });
});
