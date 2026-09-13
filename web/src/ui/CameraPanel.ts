/**
 * Panel listing registered footage viewpoints (`CameraRef`s), with a
 * time-proximity filter. Selecting one flies the 3D camera there and opens
 * the evidence overlay (via `onSelect`).
 */
import type { Playback } from "../lib/playback";
import type { CameraRef, SceneManifest } from "../lib/schema";
import { fmtLocal } from "../lib/timeline";

const NEARBY_WINDOW_S = 300;

export interface CameraPanelOptions {
  onSelect: (ref: CameraRef) => void;
}

export class CameraPanel {
  readonly el: HTMLElement;
  private list: HTMLElement;
  private nearbyOnly: HTMLInputElement;
  private toggleButton: HTMLButtonElement;

  constructor(
    private manifest: SceneManifest,
    private playback: Playback,
    private opts: CameraPanelOptions,
  ) {
    this.el = document.createElement("div");
    this.el.className = "panel panel--cameras";
    this.el.hidden = true;

    const header = document.createElement("div");
    header.className = "panel__header";
    const title = document.createElement("span");
    title.textContent = `Cameras (${manifest.cameras.length})`;
    header.appendChild(title);

    const filterLabel = document.createElement("label");
    filterLabel.style.display = "flex";
    filterLabel.style.alignItems = "center";
    filterLabel.style.gap = "0.3em";
    filterLabel.style.fontWeight = "400";
    filterLabel.style.fontSize = "0.72rem";
    this.nearbyOnly = document.createElement("input");
    this.nearbyOnly.type = "checkbox";
    this.nearbyOnly.checked = true;
    this.nearbyOnly.addEventListener("change", () => this.renderList());
    filterLabel.appendChild(this.nearbyOnly);
    filterLabel.appendChild(document.createTextNode("near current time"));
    header.appendChild(filterLabel);
    this.el.appendChild(header);

    this.list = document.createElement("div");
    this.list.className = "panel__body";
    this.el.appendChild(this.list);

    this.toggleButton = document.createElement("button");
    this.toggleButton.type = "button";
    this.toggleButton.textContent = "Cameras";
    this.toggleButton.setAttribute("aria-expanded", "false");
    this.toggleButton.addEventListener("click", () => this.setOpen(this.el.hidden));

    playback.subscribe(() => this.renderList());
    this.renderList();
  }

  get toggleEl(): HTMLButtonElement {
    return this.toggleButton;
  }

  setOpen(open: boolean): void {
    this.el.hidden = !open;
    this.toggleButton.setAttribute("aria-expanded", String(open));
  }

  private renderList(): void {
    const t = this.playback.t;
    const near = this.nearbyOnly.checked;
    this.list.replaceChildren();
    const cameras = [...this.manifest.cameras].sort((a, b) => Math.abs(a.t - t) - Math.abs(b.t - t));
    let shown = 0;
    for (const ref of cameras) {
      if (near && Math.abs(ref.t - t) > NEARBY_WINDOW_S) continue;
      shown++;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "camera-item";
      const title = document.createElement("div");
      title.textContent = ref.label || ref.id;
      btn.appendChild(title);
      const meta = document.createElement("small");
      meta.textContent = `${fmtLocal(ref.t)} EDT · ${ref.intrinsics.width}×${ref.intrinsics.height}`;
      btn.appendChild(meta);
      btn.addEventListener("click", () => this.opts.onSelect(ref));
      this.list.appendChild(btn);
    }
    if (shown === 0) {
      const empty = document.createElement("p");
      empty.style.color = "var(--fg-dim)";
      empty.style.fontSize = "0.78rem";
      empty.textContent = near
        ? "No cameras within 5 minutes of the current time. Uncheck the filter to see all."
        : "No cameras in this manifest.";
      this.list.appendChild(empty);
    }
  }
}
