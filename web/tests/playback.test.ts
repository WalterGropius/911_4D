import { describe, expect, it } from "vitest";
import { Playback } from "../src/lib/playback";

describe("Playback", () => {
  it("clamps the initial time to [tMin, tMax]", () => {
    expect(new Playback(0, 100, -50).t).toBe(0);
    expect(new Playback(0, 100, 500).t).toBe(100);
  });

  it("advances time only while playing, scaled by speed", () => {
    const p = new Playback(0, 100, 0);
    p.tick(1);
    expect(p.t).toBe(0);
    p.play();
    p.setSpeed(4);
    p.tick(1);
    expect(p.t).toBe(4);
  });

  it("stops at tMax and clamps", () => {
    const p = new Playback(0, 10, 8);
    p.play();
    p.tick(5);
    expect(p.t).toBe(10);
    expect(p.isPlaying).toBe(false);
  });

  it("stops at tMin when seeking backward while playing", () => {
    const p = new Playback(0, 10, 2);
    p.play();
    p.setSpeed(-1);
    p.tick(5);
    expect(p.t).toBe(0);
    expect(p.isPlaying).toBe(false);
  });

  it("restarts from tMin if play() is called after reaching the end", () => {
    const p = new Playback(0, 10, 10);
    p.play();
    expect(p.t).toBe(0);
    expect(p.isPlaying).toBe(true);
  });

  it("notifies subscribers on seek and tick", () => {
    const p = new Playback(0, 100, 0);
    const seen: number[] = [];
    p.subscribe((t) => seen.push(t));
    p.seek(50);
    p.play();
    p.tick(1);
    expect(seen).toEqual([50, 51]);
  });

  it("unsubscribes correctly", () => {
    const p = new Playback(0, 100, 0);
    const seen: number[] = [];
    const unsub = p.subscribe((t) => seen.push(t));
    p.seek(10);
    unsub();
    p.seek(20);
    expect(seen).toEqual([10]);
  });
});
