import "./style.css";
import { loadManifest, manifestUrlFromLocation, ManifestLoadError } from "./lib/manifestLoader";
import { ManifestValidationError } from "./lib/schema";
import { Playback } from "./lib/playback";
import type { CameraRef } from "./lib/schema";
import { SceneManager } from "./scene/SceneManager";
import { SparkSplatRenderer } from "./render/SparkSplatRenderer";
import { Timeline } from "./ui/Timeline";
import { CameraPanel } from "./ui/CameraPanel";
import { EvidenceOverlay } from "./ui/EvidenceOverlay";
import { MapInset } from "./ui/MapInset";
import { createSplash, hasSplashBeenDismissedThisSession } from "./ui/Splash";
import { createAboutModal } from "./ui/AboutPanel";
import { createFooter } from "./ui/Footer";

const appMaybe = document.getElementById("app");
if (!appMaybe) throw new Error("#app root element missing");
const app: HTMLElement = appMaybe;

function renderManifestError(err: unknown): void {
  const wrap = document.createElement("div");
  wrap.className = "manifest-error";
  const card = document.createElement("div");
  card.className = "manifest-error__card";
  const h1 = document.createElement("h1");
  h1.textContent = "Couldn't load the scene";
  card.appendChild(h1);
  const p = document.createElement("p");
  p.textContent =
    err instanceof ManifestValidationError
      ? "The scene manifest failed validation:"
      : err instanceof ManifestLoadError
        ? "The scene manifest could not be loaded:"
        : "An unexpected error occurred:";
  card.appendChild(p);
  const pre = document.createElement("pre");
  pre.textContent = err instanceof Error ? err.message : String(err);
  card.appendChild(pre);
  wrap.appendChild(card);
  app.appendChild(wrap);
}

async function boot(): Promise<void> {
  const manifestUrl = manifestUrlFromLocation(window.location);
  let loaded;
  try {
    loaded = await loadManifest(manifestUrl);
  } catch (err) {
    renderManifestError(err);
    return;
  }
  const { manifest, url } = loaded;

  const viewport = document.createElement("div");
  viewport.id = "viewport";
  viewport.style.position = "absolute";
  viewport.style.inset = "0";
  app.appendChild(viewport);

  const sceneManager = new SceneManager(viewport, new SparkSplatRenderer());
  sceneManager.setManifestBaseUrl(url);
  sceneManager.setManifest(manifest);

  const initialT = manifest.events[0]?.t ?? manifest.t_min;
  const playback = new Playback(manifest.t_min, manifest.t_max, initialT);
  playback.subscribe((t) => sceneManager.syncTime(t));
  sceneManager.syncTime(playback.t);

  // --- Top bar -------------------------------------------------------
  const topbar = document.createElement("div");
  topbar.className = "topbar";
  const title = document.createElement("span");
  title.className = "topbar__title";
  title.textContent = manifest.name;
  topbar.appendChild(title);

  const controlModeBtn = document.createElement("button");
  controlModeBtn.className = "chip";
  controlModeBtn.textContent = "Orbit";
  controlModeBtn.title = "Toggle between orbit and fly camera controls";
  controlModeBtn.addEventListener("click", () => {
    const next = sceneManager.getControlMode() === "orbit" ? "fly" : "orbit";
    sceneManager.setControlMode(next);
    controlModeBtn.textContent = next === "orbit" ? "Orbit" : "Fly";
  });
  topbar.appendChild(controlModeBtn);

  app.appendChild(topbar);

  // --- Evidence overlay (ghost image lives at the #app root so its
  // position can be synced to the canvas's on-screen rect) ------------
  const evidence = new EvidenceOverlay(() => {
    sceneManager.setPinnedAspect(null);
  });
  app.appendChild(evidence.ghostElement);
  app.appendChild(evidence.el);

  function selectCamera(ref: CameraRef): void {
    sceneManager.flyToCameraRef(ref);
    evidence.show(ref);
  }

  // --- Camera panel ----------------------------------------------------
  const cameraPanel = new CameraPanel(manifest, playback, { onSelect: selectCamera });
  topbar.appendChild(cameraPanel.toggleEl);
  app.appendChild(cameraPanel.el);

  // --- Map inset ---------------------------------------------------------
  const mapInset = new MapInset(manifest, playback, { onSelect: selectCamera });
  topbar.appendChild(mapInset.toggleEl);
  app.appendChild(mapInset.el);

  // --- About modal ---------------------------------------------------------
  const about = createAboutModal();
  app.appendChild(about.el);
  const aboutTopbarBtn = document.createElement("button");
  aboutTopbarBtn.className = "chip";
  aboutTopbarBtn.textContent = "About";
  aboutTopbarBtn.addEventListener("click", about.open);
  topbar.appendChild(aboutTopbarBtn);

  // --- Timeline ---------------------------------------------------------
  const timeline = new Timeline(manifest, playback, {
    onLayerToggle: (layer, visible) => sceneManager.setLayerVisible(layer, visible),
  });
  app.appendChild(timeline.el);

  // --- Footer ---------------------------------------------------------
  app.appendChild(createFooter(about.open));

  // --- Global keyboard shortcuts not owned by the timeline -------------
  window.addEventListener("keydown", (e) => {
    const target = e.target as HTMLElement | null;
    if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
    if (e.key === "m" || e.key === "M") mapInset.setOpen(mapInset.el.hidden);
    else if (e.key === "c" || e.key === "C") cameraPanel.setOpen(cameraPanel.el.hidden);
    else if (e.key === "Escape" && evidence.isOpen()) evidence.close();
  });

  // --- Render loop ---------------------------------------------------------
  let last = performance.now();
  function frame(now: number): void {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    playback.tick(dt);
    sceneManager.render(dt);
    if (evidence.isOpen()) {
      const rect = viewport.querySelector("canvas")?.getBoundingClientRect();
      if (rect) evidence.syncGhostRect(rect);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function start(): void {
  void boot();
}

if (hasSplashBeenDismissedThisSession()) {
  start();
} else {
  const splash = createSplash(() => {
    splash.remove();
    start();
  });
  app.appendChild(splash);
}
