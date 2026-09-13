/**
 * Absolute time for 2001-09-11 and the epoch structure of the
 * reconstruction. Ported from `wtc4d/timeline.py` — keep constants in sync.
 *
 * Time convention: all times in this project are seconds since local
 * midnight, 2001-09-11 00:00:00 EDT (UTC-4), stored as plain numbers.
 * 08:46:30 -> 31590.
 */

export function hms(h: number, m: number, s = 0): number {
  return h * 3600 + m * 60 + s;
}

/** Project seconds -> "HH:MM:SS" (EDT wall-clock), with a day suffix past midnight. */
export function fmtLocal(t: number): string {
  const totalSeconds = Math.floor(t);
  const days = Math.floor(totalSeconds / 86400);
  const secOfDay = ((totalSeconds % 86400) + 86400) % 86400;
  const h = Math.floor(secOfDay / 3600);
  const m = Math.floor((secOfDay % 3600) / 60);
  const s = Math.floor(secOfDay % 60);
  const pad = (n: number) => n.toString().padStart(2, "0");
  const base = `${pad(h)}:${pad(m)}:${pad(s)}`;
  return days !== 0 ? `${base} +${days}d` : base;
}

export interface EventAnchor {
  id: string;
  name: string;
  t: number;
  sigma: number;
  source: string;
  notes?: string;
}

// --- Canonical anchors (NIST NCSTAR 1, Table of key times) ----------------
export const WTC1_IMPACT: EventAnchor = {
  id: "wtc1_impact",
  name: "AA11 strikes WTC1 (North Tower), floors 93-99, north face",
  t: hms(8, 46, 30),
  sigma: 5.0,
  source: "NIST NCSTAR 1 (08:46:30). 9/11 Commission: 08:46:40.",
};
export const WTC2_IMPACT: EventAnchor = {
  id: "wtc2_impact",
  name: "UA175 strikes WTC2 (South Tower), floors 77-85, south face",
  t: hms(9, 2, 59),
  sigma: 5.0,
  source: "NIST NCSTAR 1 (09:02:59). 9/11 Commission: 09:03:11.",
};
export const WTC2_COLLAPSE: EventAnchor = {
  id: "wtc2_collapse",
  name: "WTC2 (South Tower) collapse initiation",
  t: hms(9, 58, 59),
  sigma: 2.0,
  source: "NIST NCSTAR 1 (09:58:59).",
  notes: "Total collapse duration roughly 10-15 s to ground level; dust cloud persists for minutes.",
};
export const WTC1_COLLAPSE: EventAnchor = {
  id: "wtc1_collapse",
  name: "WTC1 (North Tower) collapse initiation",
  t: hms(10, 28, 22),
  sigma: 2.0,
  source: "NIST NCSTAR 1 (10:28:22).",
};
export const WTC7_COLLAPSE: EventAnchor = {
  id: "wtc7_collapse",
  name: "WTC7 collapse (out of core scope, optional late epoch)",
  t: hms(17, 20, 52),
  sigma: 2.0,
  source: "NIST NCSTAR 1A (17:20:52).",
};

export const EVENTS: EventAnchor[] = [
  WTC1_IMPACT,
  WTC2_IMPACT,
  WTC2_COLLAPSE,
  WTC1_COLLAPSE,
  WTC7_COLLAPSE,
];

export interface Epoch {
  id: string;
  name: string;
  t_start: number;
  t_end: number;
  description?: string;
}

export function epochContains(ep: Epoch, t: number): boolean {
  return ep.t_start <= t && t < ep.t_end;
}

export const EPOCHS: Epoch[] = [
  {
    id: "E0",
    name: "Both towers intact",
    t_start: hms(0, 0, 0),
    t_end: WTC1_IMPACT.t,
    description:
      "Pre-impact. Sparse same-morning footage; historical photos and the geo prior supply static geometry.",
  },
  { id: "E1", name: "WTC1 burning, WTC2 intact", t_start: WTC1_IMPACT.t, t_end: WTC2_IMPACT.t },
  {
    id: "E2",
    name: "Both towers burning",
    t_start: WTC2_IMPACT.t,
    t_end: WTC2_COLLAPSE.t,
    description: "Densest footage coverage of the day.",
  },
  {
    id: "E3",
    name: "WTC2 collapsed, WTC1 burning",
    t_start: WTC2_COLLAPSE.t,
    t_end: WTC1_COLLAPSE.t,
  },
  { id: "E4", name: "Both collapsed, dust cloud", t_start: WTC1_COLLAPSE.t, t_end: hms(12, 0, 0) },
  {
    id: "E5",
    name: "Rubble pile / aftermath",
    t_start: hms(12, 0, 0),
    t_end: hms(48, 0, 0),
    description: "Extends into 9/12+ for aerial imagery and LiDAR of the debris field.",
  },
];

export function epochAt(t: number): Epoch | undefined {
  return EPOCHS.find((ep) => epochContains(ep, t));
}

export interface DynamicWindow {
  id: string;
  name: string;
  t_start: number;
  t_end: number;
}

export const DYNAMIC_WINDOWS: DynamicWindow[] = [
  { id: "D_impact2", name: "Second impact", t_start: WTC2_IMPACT.t - 10, t_end: WTC2_IMPACT.t + 40 },
  {
    id: "D_collapse2",
    name: "WTC2 collapse",
    t_start: WTC2_COLLAPSE.t - 10,
    t_end: WTC2_COLLAPSE.t + 60,
  },
  {
    id: "D_collapse1",
    name: "WTC1 collapse",
    t_start: WTC1_COLLAPSE.t - 10,
    t_end: WTC1_COLLAPSE.t + 60,
  },
];
