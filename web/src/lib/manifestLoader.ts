/**
 * Resolves and loads a `SceneManifest` from a URL (default: the bundled
 * sample scene, or `?manifest=<url>` to point at another one — e.g. one
 * published by the `recon`/`procedural` workstreams next to their assets).
 */
import { ManifestValidationError, parseManifest, type SceneManifest } from "./schema";

export const DEFAULT_MANIFEST_URL = "sample/manifest.json";

export function manifestUrlFromLocation(location: Pick<Location, "search">): string {
  const params = new URLSearchParams(location.search);
  return params.get("manifest") ?? DEFAULT_MANIFEST_URL;
}

/** Base URL that asset/thumbnail `url` fields in the manifest are relative
 * to — the manifest's own location, per the `recon` publishing convention. */
export function resolveAssetUrl(manifestUrl: string, assetUrl: string): string {
  return new URL(assetUrl, new URL(manifestUrl, window.location.href)).toString();
}

export class ManifestLoadError extends Error {
  constructor(
    message: string,
    public readonly cause?: unknown,
  ) {
    super(message);
    this.name = "ManifestLoadError";
  }
}

export async function loadManifest(url: string): Promise<{ manifest: SceneManifest; url: string }> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (err) {
    throw new ManifestLoadError(`Could not fetch manifest at "${url}": ${String(err)}`, err);
  }
  if (!res.ok) {
    throw new ManifestLoadError(`Manifest fetch failed: ${res.status} ${res.statusText} (${url})`);
  }
  let data: unknown;
  try {
    data = await res.json();
  } catch (err) {
    throw new ManifestLoadError(`Manifest at "${url}" is not valid JSON: ${String(err)}`, err);
  }
  try {
    const manifest = parseManifest(data);
    return { manifest, url };
  } catch (err) {
    if (err instanceof ManifestValidationError) throw err;
    throw new ManifestLoadError(`Unexpected error validating manifest: ${String(err)}`, err);
  }
}
