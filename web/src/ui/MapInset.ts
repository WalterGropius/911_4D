/**
 * Small map inset showing the towers and registered camera positions.
 * Prefers Leaflet + OSM tiles; falls back to a static SVG plot (using the
 * project's own ENU coordinates directly, no tile server needed) if a
 * quick connectivity probe to the tile server fails.
 */
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { Playback } from "../lib/playback";
import type { CameraRef, SceneManifest } from "../lib/schema";
import { enuToLatLon, TOWERS, towerEnuCenter, WORLD_ORIGIN } from "../lib/world";

const OSM_PROBE_URL = "https://tile.openstreetmap.org/0/0/0.png";
const OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const NEARBY_WINDOW_S = 300;

export interface MapInsetOptions {
  onSelect: (ref: CameraRef) => void;
}

function c2wPosition(c2w: readonly number[]): { x: number; y: number; z: number } {
  return { x: c2w[3], y: c2w[7], z: c2w[11] };
}

export class MapInset {
  readonly el: HTMLElement;
  private leafletContainer: HTMLElement;
  private fallbackContainer: HTMLElement;
  private map: L.Map | null = null;
  private cameraMarkers = new Map<string, L.CircleMarker>();
  private toggleButton: HTMLButtonElement;

  constructor(
    private manifest: SceneManifest,
    private playback: Playback,
    private opts: MapInsetOptions,
  ) {
    this.el = document.createElement("div");
    this.el.className = "map-inset";
    this.el.setAttribute("aria-label", "Map of camera and tower positions");

    this.leafletContainer = document.createElement("div");
    this.leafletContainer.className = "map-inset__leaflet";
    this.el.appendChild(this.leafletContainer);

    this.fallbackContainer = document.createElement("div");
    this.fallbackContainer.hidden = true;
    this.el.appendChild(this.fallbackContainer);

    this.toggleButton = document.createElement("button");
    this.toggleButton.type = "button";
    this.toggleButton.textContent = "Map";
    this.toggleButton.setAttribute("aria-expanded", "true");
    this.toggleButton.addEventListener("click", () => this.setOpen(this.el.hidden));

    void this.init();
    playback.subscribe(() => this.syncTime());
  }

  get toggleEl(): HTMLButtonElement {
    return this.toggleButton;
  }

  setOpen(open: boolean): void {
    this.el.hidden = !open;
    this.toggleButton.setAttribute("aria-expanded", String(open));
    if (open) this.map?.invalidateSize();
  }

  private async init(): Promise<void> {
    const online = await this.probeTileServer();
    if (online) {
      this.initLeaflet();
    } else {
      this.leafletContainer.hidden = true;
      this.fallbackContainer.hidden = false;
      this.renderFallbackSvg();
    }
    this.syncTime();
  }

  private async probeTileServer(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2500);
      const res = await fetch(OSM_PROBE_URL, { signal: controller.signal, mode: "no-cors" });
      clearTimeout(timeout);
      // `no-cors` gives an opaque response (status 0) on success; a network
      // failure throws instead, which is what we're actually testing for.
      return res.type === "opaque" || res.ok;
    } catch {
      return false;
    }
  }

  private initLeaflet(): void {
    const origin = WORLD_ORIGIN;
    const map = L.map(this.leafletContainer, {
      attributionControl: false,
      zoomControl: false,
    }).setView([origin.lat, origin.lon], 15);
    L.control.attribution({ prefix: "" }).addAttribution("© OpenStreetMap contributors").addTo(map);
    L.tileLayer(OSM_TILE_URL, { maxZoom: 19 }).addTo(map);

    for (const tower of TOWERS) {
      const c = towerEnuCenter(tower);
      const ll = enuToLatLon(c);
      L.circleMarker([ll.lat, ll.lon], {
        radius: 5,
        color: "#c9a24a",
        fillColor: "#c9a24a",
        fillOpacity: 0.9,
      })
        .bindTooltip(tower.name)
        .addTo(map);
    }

    for (const ref of this.manifest.cameras) {
      const pos = c2wPosition(ref.c2w);
      const ll = enuToLatLon(pos);
      const marker = L.circleMarker([ll.lat, ll.lon], {
        radius: 4,
        color: "#6fb3d8",
        fillColor: "#6fb3d8",
        fillOpacity: 0.85,
      })
        .bindTooltip(ref.label || ref.id)
        .on("click", () => this.opts.onSelect(ref));
      marker.addTo(map);
      this.cameraMarkers.set(ref.id, marker);
    }

    this.map = map;
  }

  private renderFallbackSvg(): void {
    const width = 240;
    const height = 190;
    const scale = 0.09; // px per metre; tuned for the WTC-local area
    const cx = width / 2;
    const cy = height / 2;
    const project = (p: { x: number; y: number }) => [cx + p.x * scale, cy - p.y * scale];

    const svgNs = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNs, "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("class", "map-inset__fallback");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Static map (offline fallback) of tower and camera positions");

    const bg = document.createElementNS(svgNs, "rect");
    bg.setAttribute("width", String(width));
    bg.setAttribute("height", String(height));
    bg.setAttribute("fill", "#111214");
    svg.appendChild(bg);

    for (const tower of TOWERS) {
      const [x, y] = project(towerEnuCenter(tower));
      const rect = document.createElementNS(svgNs, "rect");
      rect.setAttribute("x", String(x - 3));
      rect.setAttribute("y", String(y - 3));
      rect.setAttribute("width", "6");
      rect.setAttribute("height", "6");
      rect.setAttribute("fill", "#c9a24a");
      const title = document.createElementNS(svgNs, "title");
      title.textContent = tower.name;
      rect.appendChild(title);
      svg.appendChild(rect);
    }

    for (const ref of this.manifest.cameras) {
      const [x, y] = project(c2wPosition(ref.c2w));
      const circle = document.createElementNS(svgNs, "circle");
      circle.setAttribute("cx", String(x));
      circle.setAttribute("cy", String(y));
      circle.setAttribute("r", "3");
      circle.setAttribute("fill", "#6fb3d8");
      circle.setAttribute("data-camera-id", ref.id);
      circle.style.cursor = "pointer";
      const title = document.createElementNS(svgNs, "title");
      title.textContent = ref.label || ref.id;
      circle.appendChild(title);
      circle.addEventListener("click", () => this.opts.onSelect(ref));
      svg.appendChild(circle);
    }

    const note = document.createElementNS(svgNs, "text");
    note.setAttribute("x", "4");
    note.setAttribute("y", String(height - 4));
    note.setAttribute("fill", "#6a6c72");
    note.setAttribute("font-size", "7");
    note.textContent = "offline map (no tile server reachable)";
    svg.appendChild(note);

    this.fallbackContainer.replaceChildren(svg);
  }

  private syncTime(): void {
    const t = this.playback.t;
    for (const [id, marker] of this.cameraMarkers) {
      const ref = this.manifest.cameras.find((c) => c.id === id);
      if (!ref) continue;
      const near = Math.abs(ref.t - t) <= NEARBY_WINDOW_S;
      marker.setStyle({ radius: near ? 6 : 4, fillOpacity: near ? 1 : 0.5 });
    }
    for (const circle of this.fallbackContainer.querySelectorAll<SVGCircleElement>("circle[data-camera-id]")) {
      const id = circle.getAttribute("data-camera-id");
      const ref = this.manifest.cameras.find((c) => c.id === id);
      if (!ref) continue;
      const near = Math.abs(ref.t - t) <= NEARBY_WINDOW_S;
      circle.setAttribute("r", near ? "4.5" : "3");
      circle.setAttribute("fill-opacity", near ? "1" : "0.55");
    }
  }
}
