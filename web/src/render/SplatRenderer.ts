/**
 * Small renderer-agnostic interface around a Gaussian-splat backend, so the
 * concrete library (currently `@sparkjsdev/spark`) can be swapped for
 * another (e.g. `@mkkellogg/gaussian-splats-3d`) without touching scene or
 * UI code. See `web/README.md` for why Spark was chosen.
 */
import type { Object3D, WebGLRenderer } from "three";

export interface SplatHandle {
  /** Add this to the THREE.Scene; it is the splat asset itself. */
  object: Object3D;
  /** Resolves once the file is loaded and the splat is ready to render. */
  ready: Promise<void>;
  setOpacity(opacity: number): void;
  setVisible(visible: boolean): void;
  dispose(): void;
}

export interface SplatRendererBackend {
  /** Wires the backend into an existing renderer/scene. Call once. */
  init(renderer: WebGLRenderer, scene: import("three").Scene): void;
  /** Call once per animation frame, before rendering. */
  update(camera: import("three").Camera, deltaTime: number): void;
  /** Begin loading a splat asset (`.ply`, `.splat`, `.ksplat`, `.spz`, `.sog`). */
  load(url: string): SplatHandle;
}
