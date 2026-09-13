/**
 * Three.js scene, camera, controls and splat-asset lifecycle. World
 * coordinates are the project's ENU frame directly (Z-up) — see
 * `src/lib/cameraPose.ts` for the axis-conversion rationale.
 */
import {
  AmbientLight,
  BufferGeometry,
  Color,
  DirectionalLight,
  Fog,
  GridHelper,
  Group,
  Line,
  LineBasicMaterial,
  Object3D,
  PerspectiveCamera,
  Scene,
  Vector3,
  WebGLRenderer,
} from "three";
import { FlyControls } from "three/examples/jsm/controls/FlyControls.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { cameraRefToThree } from "../lib/cameraPose";
import type { CameraRef, SceneManifest, SplatAsset } from "../lib/schema";
import { selectPreloadCandidates, selectVisibleAssets, selectVisibleCameras } from "../lib/sceneState";
import { towerBoxCorners, WTC1, WTC2 } from "../lib/world";
import type { SplatHandle, SplatRendererBackend } from "../render/SplatRenderer";
import { buildCameraFrustum, colorForCameraTime } from "./frustum";

// Camera frusta fade in/out over this many seconds either side of `t`.
const FRUSTUM_TIME_WINDOW_S = 90;
const PRELOAD_LOOKAHEAD_S = 6;

export type ControlMode = "orbit" | "fly";

export class SceneManager {
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(50, 1, 0.1, 20000);
  readonly renderer: WebGLRenderer;
  private orbitControls: OrbitControls;
  private flyControls: FlyControls;
  private controlMode: ControlMode = "orbit";

  private backend: SplatRendererBackend;
  private manifestBaseUrl = "";
  private assetHandles = new Map<string, SplatHandle>();
  private hiddenLayers = new Set<string>();

  private pinnedAspect: number | null = null;

  private frustaGroup = new Group();
  private frustumObjects = new Map<string, ReturnType<typeof buildCameraFrustum>>();
  private manifest: SceneManifest | null = null;

  private resizeObserver: ResizeObserver;

  constructor(
    private container: HTMLElement,
    backend: SplatRendererBackend,
  ) {
    // World is ENU (Z-up); new objects/cameras default to that "up".
    Object3D.DEFAULT_UP.set(0, 0, 1);

    this.renderer = new WebGLRenderer({ antialias: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.id = "viewer-canvas";
    this.renderer.domElement.setAttribute("aria-label", "3D reconstruction viewport");

    this.scene.background = new Color(0x05060a);
    this.scene.fog = new Fog(0x05060a, 400, 3000);

    this.camera.up.set(0, 0, 1);
    this.camera.position.set(300, -300, 220);

    const grid = new GridHelper(2000, 40, 0x2c2e33, 0x1c1d20);
    grid.rotateX(Math.PI / 2); // GridHelper is XZ by default; rotate into our XY ground plane
    this.scene.add(grid);
    this.scene.add(this.buildTowerFootprintGuides());

    this.scene.add(new AmbientLight(0xffffff, 0.7));
    const sun = new DirectionalLight(0xfff2d8, 0.6);
    sun.position.set(-500, 400, 900);
    this.scene.add(sun);

    this.backend = backend;
    this.backend.init(this.renderer, this.scene);

    this.orbitControls = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbitControls.target.set(0, 0, 150);
    this.orbitControls.enableDamping = true;
    this.orbitControls.maxDistance = 6000;
    this.orbitControls.update();

    this.flyControls = new FlyControls(this.camera, this.renderer.domElement);
    this.flyControls.movementSpeed = 80;
    this.flyControls.rollSpeed = Math.PI / 6;
    this.flyControls.dragToLook = true;
    this.flyControls.enabled = false;

    this.scene.add(this.frustaGroup);

    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(container);
    this.handleResize();
  }

  /** Faint footprint outlines for both towers, always visible, for spatial
   * orientation regardless of which epoch's splat assets are loaded. */
  private buildTowerFootprintGuides(): Group {
    const group = new Group();
    group.name = "tower-footprint-guides";
    for (const tower of [WTC1, WTC2]) {
      const corners = towerBoxCorners(tower).slice(0, 4);
      const pts = [...corners, corners[0]].map((c) => new Vector3(c.x, c.y, c.z + 0.05));
      const geometry = new BufferGeometry().setFromPoints(pts);
      const material = new LineBasicMaterial({ color: 0x3a3d44, transparent: true, opacity: 0.6 });
      group.add(new Line(geometry, material));
    }
    return group;
  }

  setManifestBaseUrl(url: string): void {
    this.manifestBaseUrl = url;
  }

  private resolveUrl(assetUrl: string): string {
    return new URL(assetUrl, new URL(this.manifestBaseUrl, window.location.href)).toString();
  }

  setManifest(manifest: SceneManifest): void {
    this.manifest = manifest;
    for (const ref of manifest.cameras) {
      const frustum = buildCameraFrustum(ref, colorForCameraTime(ref.t, manifest.t_min, manifest.t_max));
      frustum.visible = false;
      this.frustumObjects.set(ref.id, frustum);
      this.frustaGroup.add(frustum);
    }
  }

  setLayerVisible(layer: string, visible: boolean): void {
    if (visible) this.hiddenLayers.delete(layer);
    else this.hiddenLayers.add(layer);
  }

  isLayerVisible(layer: string): boolean {
    return !this.hiddenLayers.has(layer);
  }

  setFrustaVisible(visible: boolean): void {
    this.frustaGroup.visible = visible;
  }

  /** Called every tick with the current playback time; loads/shows the
   * right splat assets and camera frusta for `t`. */
  syncTime(t: number): void {
    if (!this.manifest) return;
    const visible = selectVisibleAssets(this.manifest.assets, t);
    const visibleIds = new Set(visible.map((v) => v.asset.id));

    for (const { asset, opacity } of visible) {
      const handle = this.getOrLoadAsset(asset);
      const layerOn = this.isLayerVisible(asset.layer);
      handle.setVisible(layerOn);
      handle.setOpacity(layerOn ? opacity : 0);
    }
    for (const [id, handle] of this.assetHandles) {
      if (!visibleIds.has(id)) handle.setVisible(false);
    }

    for (const asset of selectPreloadCandidates(this.manifest.assets, t, PRELOAD_LOOKAHEAD_S)) {
      this.getOrLoadAsset(asset);
    }

    const visibleCams = selectVisibleCameras(this.manifest.cameras, t, FRUSTUM_TIME_WINDOW_S);
    const visibleCamIds = new Set(visibleCams.map((v) => v.camera.id));
    for (const { camera, opacity } of visibleCams) {
      const obj = this.frustumObjects.get(camera.id);
      if (!obj) continue;
      obj.visible = true;
      (obj.material as import("three").LineBasicMaterial).opacity = 0.25 + opacity * 0.6;
    }
    for (const [id, obj] of this.frustumObjects) {
      if (!visibleCamIds.has(id)) obj.visible = false;
    }
  }

  private getOrLoadAsset(asset: SplatAsset): SplatHandle {
    let handle = this.assetHandles.get(asset.id);
    if (!handle) {
      handle = this.backend.load(this.resolveUrl(asset.url));
      handle.object.name = `asset:${asset.id}`;
      this.scene.add(handle.object);
      this.assetHandles.set(asset.id, handle);
    }
    return handle;
  }

  setControlMode(mode: ControlMode): void {
    this.controlMode = mode;
    this.orbitControls.enabled = mode === "orbit";
    this.flyControls.enabled = mode === "fly";
  }

  getControlMode(): ControlMode {
    return this.controlMode;
  }

  /** Points the camera at a registered footage viewpoint (switches to a
   * free orbit target at that position so the user can still look around).
   * Also pins the camera's aspect to the source frame's, letterboxing the
   * viewport, so the evidence-view overlay lines up pixel-for-pixel. */
  flyToCameraRef(ref: CameraRef): void {
    const { position, quaternion, fov, aspect } = cameraRefToThree(ref);
    this.camera.position.copy(position);
    this.camera.quaternion.copy(quaternion);
    this.camera.fov = fov;
    const forward = new Vector3(0, 0, -1).applyQuaternion(quaternion);
    this.orbitControls.target.copy(position).addScaledVector(forward, 30);
    this.orbitControls.update();
    this.setPinnedAspect(aspect);
  }

  /** Locks the renderer to a fixed aspect ratio (letterboxed within the
   * container) for matching a source frame exactly; `null` restores the
   * viewport to fill the container. */
  setPinnedAspect(aspect: number | null): void {
    this.pinnedAspect = aspect;
    this.handleResize();
  }

  private handleResize(): void {
    const cw = this.container.clientWidth;
    const ch = this.container.clientHeight;
    if (cw === 0 || ch === 0) return;
    let w = cw;
    let h = ch;
    if (this.pinnedAspect) {
      if (cw / ch > this.pinnedAspect) {
        h = ch;
        w = ch * this.pinnedAspect;
      } else {
        w = cw;
        h = cw / this.pinnedAspect;
      }
    }
    this.camera.aspect = this.pinnedAspect ?? cw / ch;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    const canvas = this.renderer.domElement;
    canvas.style.left = `${(cw - w) / 2}px`;
    canvas.style.top = `${(ch - h) / 2}px`;
  }

  render(deltaSeconds: number): void {
    if (this.controlMode === "orbit") {
      this.orbitControls.update();
    } else {
      this.flyControls.update(deltaSeconds);
    }
    this.backend.update(this.camera, deltaSeconds);
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.resizeObserver.disconnect();
    for (const handle of this.assetHandles.values()) handle.dispose();
    this.orbitControls.dispose();
    this.flyControls.dispose();
    this.renderer.dispose();
  }
}
