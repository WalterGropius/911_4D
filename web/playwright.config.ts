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
    command: "npm run preview -- --port 4173 --strictPort",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
