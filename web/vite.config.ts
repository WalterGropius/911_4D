import { defineConfig } from "vitest/config";

// Base path: GitHub Pages serves this project at /<repo>/ unless a custom
// domain is configured. Override with VITE_BASE at build time if needed.
export default defineConfig({
  base: process.env.VITE_BASE ?? "/",
  build: {
    target: "es2022",
    sourcemap: true,
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
