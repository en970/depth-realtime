import { chromium } from "playwright";

const PROFILE =
  "/private/tmp/claude-501/-Users-enes-Developer-depth-realtime/914d624e-f8a0-4f36-a0c6-39e63e8a2cd2/scratchpad/chrome-profile";
const OUT =
  "/private/tmp/claude-501/-Users-enes-Developer-depth-realtime/914d624e-f8a0-4f36-a0c6-39e63e8a2cd2/scratchpad";
const URL =
  process.env.URL ?? "http://127.0.0.1:4173/depth-realtime/#smooth=0.35&mode=split&split=50";

const context = await chromium.launchPersistentContext(PROFILE, {
  headless: false,
  args: [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--enable-unsafe-webgpu",
  ],
  permissions: ["camera"],
  viewport: { width: 1440, height: 900 },
});
context.setDefaultTimeout(300_000);
const page = context.pages()[0] ?? (await context.newPage());
await page.setViewportSize({ width: 1440, height: 900 });

await page.goto(URL, { waitUntil: "load" });
await page.click("#overlay-action");
await page.waitForFunction(
  () => document.getElementById("stat-fps")?.textContent !== "—",
  { timeout: 300_000 },
);
await page.waitForTimeout(6000);
await page.screenshot({ path: `${OUT}/v2-desktop-split.png` });

await page.keyboard.press("c");
await page.waitForTimeout(1500);
await page.screenshot({ path: `${OUT}/v2-desktop-compare.png` });
await page.keyboard.press("c");

await page.setViewportSize({ width: 390, height: 844 });
await page.waitForTimeout(2000);
await page.screenshot({ path: `${OUT}/v2-mobile.png` });

await context.close();
console.log("screenshots written");
