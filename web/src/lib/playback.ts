/**
 * Playback clock driving the timeline: current project time `t`, play state
 * and speed. Renderer/UI-agnostic and unit-testable; `src/ui/Timeline.ts`
 * and `src/scene/SceneManager.ts` both subscribe to it.
 */

export type PlaybackListener = (t: number) => void;

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

export class Playback {
  private _t: number;
  private _playing = false;
  private _speed = 1;
  private listeners = new Set<PlaybackListener>();

  constructor(
    public readonly tMin: number,
    public readonly tMax: number,
    initial: number = tMin,
  ) {
    this._t = clamp(initial, tMin, tMax);
  }

  get t(): number {
    return this._t;
  }

  get isPlaying(): boolean {
    return this._playing;
  }

  get speed(): number {
    return this._speed;
  }

  setSpeed(speed: number): void {
    this._speed = speed;
  }

  play(): void {
    if (this._t >= this.tMax) this._t = this.tMin;
    this._playing = true;
  }

  pause(): void {
    this._playing = false;
  }

  togglePlay(): void {
    if (this._playing) this.pause();
    else this.play();
  }

  seek(t: number): void {
    this._t = clamp(t, this.tMin, this.tMax);
    this.emit();
  }

  seekBy(deltaSeconds: number): void {
    this.seek(this._t + deltaSeconds);
  }

  /** Advance the clock by `deltaSeconds` of wall time (scaled by `speed`)
   * if playing; a no-op otherwise. Stops at the manifest's time bounds. */
  tick(deltaSeconds: number): void {
    if (!this._playing) return;
    let next = this._t + deltaSeconds * this._speed;
    let stop = false;
    if (next >= this.tMax) {
      next = this.tMax;
      stop = true;
    } else if (next <= this.tMin) {
      next = this.tMin;
      stop = true;
    }
    this._t = next;
    if (stop) this._playing = false;
    this.emit();
  }

  subscribe(fn: PlaybackListener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(): void {
    for (const fn of this.listeners) fn(this._t);
  }
}
