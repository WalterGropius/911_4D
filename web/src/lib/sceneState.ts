/**
 * Pure functions selecting what should be visible/loaded at a given project
 * time `t`, given a validated `SceneManifest`. Kept renderer-agnostic and
 * side-effect free so it's easy to unit test; `src/scene/*` wires this into
 * Three.js and the splat renderer.
 */
import type { CameraRef, SplatAsset } from "./schema";

export interface VisibleAsset {
  asset: SplatAsset;
  /** 0..1 crossfade opacity, 1 away from the asset's boundaries. */
  opacity: number;
}

const DEFAULT_CROSSFADE_SECONDS = 1.5;

/** Assets whose `[t_start, t_end)` window contains `t`, with a crossfade
 * opacity computed against that asset's own boundaries (fades in just after
 * `t_start`, fades out just before `t_end`). Assets on different layers
 * don't affect each other's opacity — crossfade is meant for the handoff
 * between adjacent same-layer assets (e.g. consecutive epoch splats). */
export function selectVisibleAssets(
  assets: readonly SplatAsset[],
  t: number,
  crossfadeSeconds = DEFAULT_CROSSFADE_SECONDS,
): VisibleAsset[] {
  const out: VisibleAsset[] = [];
  for (const asset of assets) {
    if (!(asset.t_start <= t && t < asset.t_end)) continue;
    const fadeIn = crossfadeSeconds > 0 ? (t - asset.t_start) / crossfadeSeconds : 1;
    const fadeOut = crossfadeSeconds > 0 ? (asset.t_end - t) / crossfadeSeconds : 1;
    const opacity = Math.max(0, Math.min(1, Math.min(fadeIn, fadeOut)));
    out.push({ asset, opacity });
  }
  return out;
}

/** Assets not currently visible but starting within `lookaheadSeconds`,
 * worth preloading so the crossfade-in has no pop. */
export function selectPreloadCandidates(
  assets: readonly SplatAsset[],
  t: number,
  lookaheadSeconds = 5,
): SplatAsset[] {
  return assets.filter((a) => a.t_start > t && a.t_start - t <= lookaheadSeconds);
}

/** Cameras whose capture time is within `windowSeconds` of `t` — used to
 * decide which frusta to draw and how to fade them in/out by recency. */
export function selectVisibleCameras(
  cameras: readonly CameraRef[],
  t: number,
  windowSeconds: number,
): { camera: CameraRef; opacity: number }[] {
  const out: { camera: CameraRef; opacity: number }[] = [];
  for (const camera of cameras) {
    const dt = Math.abs(camera.t - t);
    if (dt > windowSeconds) continue;
    out.push({ camera, opacity: 1 - dt / windowSeconds });
  }
  return out;
}

/** Distinct `layer` values present in a manifest's assets, in first-seen
 * order, for building a layer-toggle UI. */
export function distinctLayers(assets: readonly SplatAsset[]): string[] {
  const seen = new Set<string>();
  for (const a of assets) seen.add(a.layer);
  return [...seen];
}
