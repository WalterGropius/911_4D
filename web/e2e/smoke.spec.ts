import { expect, test } from "@playwright/test";

test("loads the sample scene, dismisses the content warning, and renders the viewer", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await page.goto("/");

  await expect(page.getByRole("dialog", { name: "Before you continue" })).toBeVisible();
  await page.getByRole("button", { name: "I understand, continue" }).click();

  const canvas = page.locator("#viewer-canvas");
  await expect(canvas).toBeVisible();

  // The timeline clock should show an EDT wall-clock time once the manifest
  // loads and playback initialises (not stuck on a loading/error state).
  await expect(page.locator(".timeline__clock")).toHaveText(/\d{2}:\d{2}:\d{2} EDT/, { timeout: 15_000 });

  // Layer toggle chips from the sample manifest's asset layers.
  await expect(page.getByRole("button", { name: "Towers" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Ground" })).toBeVisible();

  // No manifest-load or runtime error surfaced.
  await expect(page.locator(".manifest-error")).toHaveCount(0);

  await page.waitForTimeout(1000); // let a few frames render before the screenshot
  const screenshot = await page.screenshot();
  await test.info().attach("scene-screenshot", { body: screenshot, contentType: "image/png" });

  expect(pageErrors, `unexpected page errors: ${pageErrors.join("; ")}`).toEqual([]);
});

test("camera panel lists cameras and flying to one opens the evidence view", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "I understand, continue" }).click();
  await expect(page.locator("#viewer-canvas")).toBeVisible();

  await page.getByRole("button", { name: /^Cameras/ }).click();
  const firstCamera = page.locator(".camera-item").first();
  await expect(firstCamera).toBeVisible();
  await firstCamera.click();

  await expect(page.getByRole("region", { name: "Evidence view" })).toBeVisible();
});

test("manifest validation errors are shown gracefully for a bad manifest URL", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await page.goto("/?manifest=sample/does-not-exist.json");
  await page.getByRole("button", { name: "I understand, continue" }).click();

  await expect(page.locator(".manifest-error")).toBeVisible();
  expect(pageErrors).toEqual([]);
});
