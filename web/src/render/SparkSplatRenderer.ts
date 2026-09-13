/**
 * `@sparkjsdev/spark` backend for {@link SplatRendererBackend}. Spark patches
 * the given `THREE.WebGLRenderer` (via its `SparkRenderer` scene object,
 * `autoUpdate: true` by default) so no manual per-frame update call is
 * needed here; `update()` is a no-op kept only to satisfy the interface.
 */
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import type { Camera, Scene, WebGLRenderer } from "three";
import type { SplatHandle, SplatRendererBackend } from "./SplatRenderer";

export class SparkSplatRenderer implements SplatRendererBackend {
  private spark: SparkRenderer | null = null;

  init(renderer: WebGLRenderer, scene: Scene): void {
    this.spark = new SparkRenderer({ renderer });
    scene.add(this.spark);
  }

  update(_camera: Camera, _deltaTime: number): void {
    // Spark's SparkRenderer hooks the WebGLRenderer directly; nothing to do.
  }

  load(url: string): SplatHandle {
    const mesh = new SplatMesh({ url });
    let disposed = false;

    return {
      object: mesh,
      ready: mesh.initialized.then(() => undefined),
      setOpacity(opacity: number) {
        if (disposed) return;
        mesh.opacity = opacity;
      },
      setVisible(visible: boolean) {
        if (disposed) return;
        mesh.visible = visible;
      },
      dispose() {
        if (disposed) return;
        disposed = true;
        mesh.dispose();
      },
    };
  }
}
