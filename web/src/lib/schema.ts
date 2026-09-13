/**
 * Zod schema for `SceneManifest`, mirroring `wtc4d/schema/scene.py` and
 * `wtc4d/schema/camera.py`. Keep field names and defaults identical to the
 * Python side; this is the JSON contract produced by `recon`/`procedural`
 * and consumed by the viewer.
 */
import { z } from "zod";

export const LatLonAltSchema = z.object({
  lat: z.number().min(-90).max(90),
  lon: z.number().min(-180).max(180),
  alt_m: z.number().default(0),
});
export type LatLonAlt = z.infer<typeof LatLonAltSchema>;

export const CameraIntrinsicsSchema = z.object({
  width: z.number().int().positive(),
  height: z.number().int().positive(),
  fx: z.number(),
  fy: z.number(),
  cx: z.number(),
  cy: z.number(),
  model: z.string().default("OPENCV"),
  dist: z.array(z.number()).default([]),
});
export type CameraIntrinsics = z.infer<typeof CameraIntrinsicsSchema>;

export const SplatAssetSchema = z.object({
  id: z.string(),
  url: z.string(),
  format: z.enum(["ply", "splat", "ksplat", "spz", "sog"]).default("ply"),
  t_start: z.number(),
  t_end: z.number(),
  kind: z.enum(["static", "dynamic", "procedural"]).default("static"),
  layer: z.enum(["scene", "smoke", "debris", "towers", "ground"]).default("scene"),
  epoch_id: z.string().nullable().default(null),
  notes: z.string().default(""),
});
export type SplatAsset = z.infer<typeof SplatAssetSchema>;

export const CameraRefSchema = z.object({
  id: z.string(),
  shot_id: z.string(),
  frame_idx: z.number().int(),
  t: z.number(),
  c2w: z.array(z.number()).length(16),
  intrinsics: CameraIntrinsicsSchema,
  thumbnail_url: z.string().nullable().default(null),
  source_url: z.string().nullable().default(null),
  label: z.string().default(""),
});
export type CameraRef = z.infer<typeof CameraRefSchema>;

export const TimelineEventSchema = z.object({
  id: z.string(),
  name: z.string(),
  t: z.number(),
  sigma: z.number().default(0),
});
export type TimelineEvent = z.infer<typeof TimelineEventSchema>;

export const SceneManifestSchema = z.object({
  schema_version: z.number().int().default(1),
  name: z.string().default("911_4D"),
  world_origin: LatLonAltSchema,
  t_min: z.number(),
  t_max: z.number(),
  events: z.array(TimelineEventSchema).default([]),
  assets: z.array(SplatAssetSchema).default([]),
  cameras: z.array(CameraRefSchema).default([]),
  notes: z.string().default(""),
});
export type SceneManifest = z.infer<typeof SceneManifestSchema>;

export class ManifestValidationError extends Error {
  constructor(
    message: string,
    public readonly issues: z.ZodIssue[],
  ) {
    super(message);
    this.name = "ManifestValidationError";
  }
}

/** Parse and validate a manifest, throwing a friendly error with issue paths. */
export function parseManifest(data: unknown): SceneManifest {
  const result = SceneManifestSchema.safeParse(data);
  if (!result.success) {
    const issues = result.error.issues;
    const detail = issues.map((i) => `${i.path.join(".") || "<root>"}: ${i.message}`).join("; ");
    throw new ManifestValidationError(`Invalid scene manifest: ${detail}`, issues);
  }
  return result.data;
}
