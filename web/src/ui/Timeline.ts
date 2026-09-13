/**
 * Timeline scrubber: play/pause, speed presets, event markers, HH:MM:SS EDT
 * clock, layer toggles and keyboard shortcuts. Pure DOM, no framework.
 */
import { fmtLocal } from "../lib/timeline";
import { epochAt } from "../lib/timeline";
import type { Playback } from "../lib/playback";
import type { SceneManifest } from "../lib/schema";
import { distinctLayers } from "../lib/sceneState";

const SPEED_PRESETS = [0.25, 1, 4, 15, 60, 300];

export interface TimelineOptions {
  onLayerToggle: (layer: string, visible: boolean) => void;
}

const LAYER_LABELS: Record<string, string> = {
  towers: "Towers",
  smoke: "Smoke",
  debris: "Debris",
  ground: "Ground",
  scene: "Scene",
};

export class Timeline {
  readonly el: HTMLElement;
  private range: HTMLInputElement;
  private clock: HTMLElement;
  private epochLabel: HTMLElement;
  private playButton: HTMLButtonElement;
  private speedSelect: HTMLSelectElement;
  private layerButtons = new Map<string, HTMLButtonElement>();
  private seeking = false;

  constructor(
    private manifest: SceneManifest,
    private playback: Playback,
    private opts: TimelineOptions,
  ) {
    this.el = document.createElement("div");
    this.el.className = "timeline";

    const row = document.createElement("div");
    row.className = "timeline__row";
    this.el.appendChild(row);

    this.playButton = document.createElement("button");
    this.playButton.type = "button";
    this.playButton.setAttribute("aria-label", "Play or pause");
    this.playButton.textContent = "▶";
    this.playButton.addEventListener("click", () => this.playback.togglePlay());
    row.appendChild(this.playButton);

    this.speedSelect = document.createElement("select");
    this.speedSelect.className = "timeline__speed";
    this.speedSelect.setAttribute("aria-label", "Playback speed");
    for (const s of SPEED_PRESETS) {
      const option = document.createElement("option");
      option.value = String(s);
      option.textContent = `${s}×`;
      if (s === 1) option.selected = true;
      this.speedSelect.appendChild(option);
    }
    this.speedSelect.addEventListener("change", () => {
      this.playback.setSpeed(Number(this.speedSelect.value));
    });
    row.appendChild(this.speedSelect);

    this.clock = document.createElement("span");
    this.clock.className = "timeline__clock";
    this.clock.setAttribute("aria-live", "polite");
    row.appendChild(this.clock);

    this.epochLabel = document.createElement("span");
    this.epochLabel.className = "timeline__epoch";
    row.appendChild(this.epochLabel);

    const scrubber = document.createElement("div");
    scrubber.className = "timeline__scrubber";
    row.appendChild(scrubber);

    const markers = document.createElement("div");
    markers.className = "timeline__markers";
    markers.setAttribute("aria-hidden", "true");
    for (const event of manifest.events) {
      const pct = this.percentFor(event.t);
      const marker = document.createElement("div");
      marker.className = "timeline__marker";
      marker.style.left = `${pct}%`;
      const label = document.createElement("div");
      label.className = "timeline__marker-label";
      label.textContent = event.name;
      marker.appendChild(label);
      markers.appendChild(marker);
    }
    scrubber.appendChild(markers);

    this.range = document.createElement("input");
    this.range.type = "range";
    this.range.className = "timeline__range";
    this.range.min = String(manifest.t_min);
    this.range.max = String(manifest.t_max);
    this.range.step = "1";
    this.range.value = String(playback.t);
    this.range.setAttribute("aria-label", "Timeline scrubber");
    this.range.addEventListener("input", () => {
      this.seeking = true;
      this.playback.seek(Number(this.range.value));
      this.seeking = false;
    });
    scrubber.appendChild(this.range);

    const layerRow = document.createElement("div");
    layerRow.className = "chip-row";
    layerRow.setAttribute("role", "group");
    layerRow.setAttribute("aria-label", "Toggle scene layers");
    for (const layer of distinctLayers(manifest.assets)) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip";
      btn.textContent = LAYER_LABELS[layer] ?? layer;
      btn.setAttribute("aria-pressed", "true");
      btn.addEventListener("click", () => {
        const nowVisible = btn.getAttribute("aria-pressed") !== "true";
        btn.setAttribute("aria-pressed", String(nowVisible));
        this.opts.onLayerToggle(layer, nowVisible);
      });
      this.layerButtons.set(layer, btn);
      layerRow.appendChild(btn);
    }
    this.el.appendChild(layerRow);

    playback.subscribe((t) => this.onTimeChanged(t));
    this.onTimeChanged(playback.t);
    this.attachKeyboardShortcuts();
  }

  private percentFor(t: number): number {
    const { t_min, t_max } = this.manifest;
    if (t_max === t_min) return 0;
    return (100 * (t - t_min)) / (t_max - t_min);
  }

  private onTimeChanged(t: number): void {
    this.clock.textContent = fmtLocal(t) + " EDT";
    const epoch = epochAt(t);
    this.epochLabel.textContent = epoch ? epoch.name : "";
    if (!this.seeking) this.range.value = String(t);
    this.playButton.textContent = this.playback.isPlaying ? "⏸" : "▶";
    this.playButton.setAttribute("aria-pressed", String(this.playback.isPlaying));
  }

  private attachKeyboardShortcuts(): void {
    window.addEventListener("keydown", (e) => {
      const target = e.target as HTMLElement | null;
      const typing = target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
      if (e.code === "Space" && !typing) {
        e.preventDefault();
        this.playback.togglePlay();
        return;
      }
      if (typing) return;
      if (e.key === "ArrowRight") {
        this.playback.seekBy(e.shiftKey ? 60 : 10);
      } else if (e.key === "ArrowLeft") {
        this.playback.seekBy(e.shiftKey ? -60 : -10);
      } else if (e.key >= "1" && e.key <= "6") {
        const idx = Number(e.key) - 1;
        if (SPEED_PRESETS[idx] !== undefined) {
          this.playback.setSpeed(SPEED_PRESETS[idx]);
          this.speedSelect.value = String(SPEED_PRESETS[idx]);
        }
      }
    });
  }
}
