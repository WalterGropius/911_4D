import { defineConfig } from "@playwright/test";
import { existsSync } from "node:fs";

const bundledChromium = "/opt/pw-browsers/chromium";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["html", { open: "never" }], ["list"]] : "list",
  use: {
    baseURL: "http://127.0.0.1:4173",
    launchOptions: existsSync(bundledChromium) ? { executablePath: bundledChromium } : {},
  },
  webServer: {
    // Serves the production build (already produced by `npm run build`,
    // which `npm run e2e` runs first) rather than `vite dev`: no cold
    // dependency-pre-bundling step, so it's ready in well under a second
    // instead of the 1-2+ minutes a cold `vite dev` can take in CI.
    // --host 127.0.0.1 pins the bind address explicitly: some CI runners'
    // default DNS resolution order for "localhost" prefers the IPv6
    // loopback, which left `vite preview`'s default bind unreachable at
    // the IPv4 address this config health-checks, causing the webServer
    // wait to exhaust its timeout with no explicit connection error.
    command: "npm run preview -- --strictPort --host 127.0.0.1",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
