import { describe, expect, it } from "vitest";
import { enuToLatLon, latLonToEnu, towerBoxCorners, towerEnuCenter, WTC1, WTC2 } from "../src/lib/world";

// Reference values computed from `wtc4d/world.py` (numpy/pydantic) on the
// same constants, to keep the TS port numerically in sync with Python.
const EMPIRE_STATE = { lat: 40.74844, lon: -73.98565, alt_m: 443.0 };
const EMPIRE_STATE_ENU = { x: 2326.8760267503667, y: 4136.105710497521, z: 441.23191584819176 };

const STATUE_OF_LIBERTY = { lat: 40.68925, lon: -74.0445, alt_m: 93.0 };
const STATUE_OF_LIBERTY_ENU = { x: -2645.798996726013, y: -2437.0703908045707, z: 91.98528986102251 };

const WTC1_CENTER_ENU = { x: -15.210072479506566, y: 83.28626837368827, z: -0.0005632179128327231 };
const WTC2_CENTER_ENU = { x: 29.575760370620433, y: -72.18135138399344, z: -0.0004779093160785479 };

describe("latLonToEnu", () => {
  it("matches the Python reference for the Empire State Building spire", () => {
    const enu = latLonToEnu(EMPIRE_STATE);
    expect(enu.x).toBeCloseTo(EMPIRE_STATE_ENU.x, 3);
    expect(enu.y).toBeCloseTo(EMPIRE_STATE_ENU.y, 3);
    expect(enu.z).toBeCloseTo(EMPIRE_STATE_ENU.z, 3);
  });

  it("matches the Python reference for the Statue of Liberty torch", () => {
    const enu = latLonToEnu(STATUE_OF_LIBERTY);
    expect(enu.x).toBeCloseTo(STATUE_OF_LIBERTY_ENU.x, 3);
    expect(enu.y).toBeCloseTo(STATUE_OF_LIBERTY_ENU.y, 3);
    expect(enu.z).toBeCloseTo(STATUE_OF_LIBERTY_ENU.z, 3);
  });

  it("matches the Python reference for the tower centers", () => {
    const wtc1 = towerEnuCenter(WTC1);
    expect(wtc1.x).toBeCloseTo(WTC1_CENTER_ENU.x, 3);
    expect(wtc1.y).toBeCloseTo(WTC1_CENTER_ENU.y, 3);
    expect(wtc1.z).toBeCloseTo(WTC1_CENTER_ENU.z, 6);

    const wtc2 = towerEnuCenter(WTC2);
    expect(wtc2.x).toBeCloseTo(WTC2_CENTER_ENU.x, 3);
    expect(wtc2.y).toBeCloseTo(WTC2_CENTER_ENU.y, 3);
    expect(wtc2.z).toBeCloseTo(WTC2_CENTER_ENU.z, 6);
  });
});

describe("enuToLatLon", () => {
  it("round-trips lat/lon/alt through ENU", () => {
    const enu = latLonToEnu(EMPIRE_STATE);
    const back = enuToLatLon(enu);
    expect(back.lat).toBeCloseTo(EMPIRE_STATE.lat, 6);
    expect(back.lon).toBeCloseTo(EMPIRE_STATE.lon, 6);
    expect(back.alt_m).toBeCloseTo(EMPIRE_STATE.alt_m, 2);
  });

  it("round-trips an arbitrary ENU point far from the origin", () => {
    const p = { x: 12345.6, y: -6789.1, z: 250.0 };
    const back = enuToLatLon(p);
    const forward = latLonToEnu(back);
    expect(forward.x).toBeCloseTo(p.x, 3);
    expect(forward.y).toBeCloseTo(p.y, 3);
    expect(forward.z).toBeCloseTo(p.z, 3);
  });
});

describe("towerBoxCorners", () => {
  it("produces 8 corners: 4 at plaza level, 4 at roof height", () => {
    const corners = towerBoxCorners(WTC1);
    expect(corners).toHaveLength(8);
    const center = towerEnuCenter(WTC1);
    for (const c of corners.slice(0, 4)) {
      expect(c.z).toBeCloseTo(center.z, 6);
    }
    for (const c of corners.slice(4, 8)) {
      expect(c.z).toBeCloseTo(center.z + WTC1.roof_height_m, 6);
    }
  });

  it("centers the footprint on the tower center in plan", () => {
    const corners = towerBoxCorners(WTC2);
    const center = towerEnuCenter(WTC2);
    const bottom = corners.slice(0, 4);
    const meanX = bottom.reduce((s, c) => s + c.x, 0) / 4;
    const meanY = bottom.reduce((s, c) => s + c.y, 0) / 4;
    expect(meanX).toBeCloseTo(center.x, 6);
    expect(meanY).toBeCloseTo(center.y, 6);
    const half = WTC2.footprint_m / 2;
    for (const c of bottom) {
      expect(Math.hypot(c.x - center.x, c.y - center.y)).toBeCloseTo(half * Math.SQRT2, 4);
    }
  });
});
