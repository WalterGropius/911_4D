/**
 * "Evidence view": shows the source frame for a selected `CameraRef`
 * alongside (or ghosted over) the 3D reconstruction, so the two can be
 * compared. Two modes:
 *  - overlay: the thumbnail is pinned exactly over the (letterboxed)
 *    canvas rect with an adjustable opacity, for direct ghosting/alignment.
 *  - side-by-side: the thumbnail sits in this panel instead, full opacity.
 * Handles the no-thumbnail case (most cameras, until `camreg` publishes
 * real frames) gracefully rather than erroring.
 */
import type { CameraRef } from "../lib/schema";
import { fmtLocal } from "../lib/timeline";

export type EvidenceMode = "overlay" | "side-by-side";

export class EvidenceOverlay {
  readonly el: HTMLElement;
  private ghostImg: HTMLImageElement;
  private inlineThumb: HTMLElement;
  private opacitySlider: HTMLInputElement;
  private modeButton: HTMLButtonElement;
  private mode: EvidenceMode = "overlay";
  private current: CameraRef | null = null;
  private titleEl!: HTMLElement;
  private metaEl!: HTMLElement;
  private sourceLink!: HTMLAnchorElement;

  constructor(private onClose: () => void) {
    this.el = document.createElement("div");
    this.el.className = "evidence";
    this.el.hidden = true;
    this.el.setAttribute("role", "region");
    this.el.setAttribute("aria-label", "Evidence view");

    const header = document.createElement("div");
    header.className = "evidence__header";
    const title = document.createElement("strong");
    header.appendChild(title);
    this.titleEl = title;
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.textContent = "✕";
    closeBtn.setAttribute("aria-label", "Close evidence view");
    closeBtn.addEventListener("click", () => this.close());
    header.appendChild(closeBtn);
    this.el.appendChild(header);

    const body = document.createElement("div");
    body.className = "evidence__body";
    this.el.appendChild(body);

    this.inlineThumb = document.createElement("div");
    this.inlineThumb.className = "evidence__thumb-wrap";
    body.appendChild(this.inlineThumb);

    this.metaEl = document.createElement("div");
    body.appendChild(this.metaEl);

    const controls = document.createElement("div");
    controls.className = "evidence__controls";
    body.appendChild(controls);

    this.modeButton = document.createElement("button");
    this.modeButton.type = "button";
    this.modeButton.textContent = "Side-by-side";
    this.modeButton.addEventListener("click", () => this.setMode(this.mode === "overlay" ? "side-by-side" : "overlay"));
    controls.appendChild(this.modeButton);

    const opacityLabel = document.createElement("label");
    opacityLabel.style.display = "flex";
    opacityLabel.style.alignItems = "center";
    opacityLabel.style.gap = "0.4em";
    opacityLabel.textContent = "Overlay opacity";
    this.opacitySlider = document.createElement("input");
    this.opacitySlider.type = "range";
    this.opacitySlider.min = "0";
    this.opacitySlider.max = "1";
    this.opacitySlider.step = "0.01";
    this.opacitySlider.value = "0.6";
    this.opacitySlider.setAttribute("aria-label", "Ghost overlay opacity");
    this.opacitySlider.addEventListener("input", () => this.updateGhostOpacity());
    opacityLabel.appendChild(this.opacitySlider);
    controls.appendChild(opacityLabel);

    this.sourceLink = document.createElement("a");
    this.sourceLink.target = "_blank";
    this.sourceLink.rel = "noopener noreferrer";
    this.sourceLink.textContent = "Open source";
    this.sourceLink.style.fontSize = "0.78rem";
    controls.appendChild(this.sourceLink);

    // Ghost image lives outside this panel, positioned over the canvas.
    this.ghostImg = document.createElement("img");
    this.ghostImg.className = "evidence-ghost";
    this.ghostImg.style.position = "absolute";
    this.ghostImg.style.pointerEvents = "none";
    this.ghostImg.style.zIndex = "12";
    this.ghostImg.style.objectFit = "fill";
    this.ghostImg.hidden = true;
    this.ghostImg.alt = "";
    this.ghostImg.setAttribute("aria-hidden", "true");
  }

  /** Must be appended to the same positioned ancestor as the canvas. */
  get ghostElement(): HTMLElement {
    return this.ghostImg;
  }

  show(ref: CameraRef): void {
    this.current = ref;
    this.el.hidden = false;
    this.titleEl.textContent = ref.label || ref.id;
    this.metaEl.textContent = `${fmtLocal(ref.t)} EDT · shot ${ref.shot_id} frame ${ref.frame_idx}`;

    this.inlineThumb.replaceChildren();
    this.ghostImg.hidden = true;
    this.ghostImg.removeAttribute("src");

    if (ref.thumbnail_url) {
      const inlineImg = document.createElement("img");
      inlineImg.src = ref.thumbnail_url;
      inlineImg.alt = `Source frame for ${ref.label || ref.id}`;
      this.inlineThumb.appendChild(inlineImg);
      this.ghostImg.src = ref.thumbnail_url;
    } else {
      this.inlineThumb.textContent =
        "No source frame published yet for this camera. The camreg workstream will add thumbnail_url once registered.";
    }

    if (ref.source_url) {
      this.sourceLink.href = ref.source_url;
      this.sourceLink.hidden = false;
    } else {
      this.sourceLink.hidden = true;
    }

    this.setMode(this.mode);
  }

  close(): void {
    this.current = null;
    this.el.hidden = true;
    this.ghostImg.hidden = true;
    this.onClose();
  }

  setMode(mode: EvidenceMode): void {
    this.mode = mode;
    this.modeButton.textContent = mode === "overlay" ? "Side-by-side" : "Overlay";
    const hasThumb = !!this.current?.thumbnail_url;
    this.inlineThumb.style.display = mode === "side-by-side" ? "flex" : "none";
    this.ghostImg.hidden = !(mode === "overlay" && hasThumb);
    this.updateGhostOpacity();
  }

  private updateGhostOpacity(): void {
    this.ghostImg.style.opacity = this.opacitySlider.value;
  }

  /** Call every frame (or on resize) with the canvas's current on-screen
   * rect, relative to the ghost image's positioned ancestor. */
  syncGhostRect(rect: { left: number; top: number; width: number; height: number }): void {
    if (this.ghostImg.hidden) return;
    this.ghostImg.style.left = `${rect.left}px`;
    this.ghostImg.style.top = `${rect.top}px`;
    this.ghostImg.style.width = `${rect.width}px`;
    this.ghostImg.style.height = `${rect.height}px`;
  }

  isOpen(): boolean {
    return !this.el.hidden;
  }
}
