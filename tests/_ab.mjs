import { chromium } from "playwright";
import { join } from "node:path";
import { HEADLESS, GPU_ARGS, sceneArgs } from "./browser.mjs";

const OUT =
  "/private/tmp/claude-501/-Users-enes-Developer-depth-realtime/914d624e-f8a0-4f36-a0c6-39e63e8a2cd2/scratchpad";
const scene = process.env.SCENE ?? "demo01";
const CONTROL = process.env.CONTROL ?? "contour-range";
const VALUES = (process.env.VALUES ?? "0,0.6").split(",");
const COMMON =
  process.env.COMMON ??
  "mirror=0&adaptive=0&res=350&smooth=0.35&guide=0.85&structure=0.6&tone=0.7&mode=split&split=50";

const context = await chromium.launchPersistentContext(
  process.env.PROFILE ?? `${OUT}/chrome-profile`,
  {
    headless: HEADLESS,
    args: [...GPU_ARGS, ...sceneArgs(join(`${OUT}/refdata`, `${scene}.y4m`))],
    permissions: ["camera"],
    viewport: { width: 1400, height: 800 },
  },
);
context.setDefaultTimeout(300_000);
const page = context.pages()[0] ?? (await context.newPage());
await page.addInitScript(() => localStorage.clear());

await page.goto(`http://127.0.0.1:4173/depth-realtime/#${COMMON}`, { waitUntil: "load" });
await page.click("#overlay-action");
await page.waitForFunction(
  () => document.getElementById("stat-fps")?.textContent !== "—",
  { timeout: 300_000 },
);
await page.waitForTimeout(9000);

for (const value of VALUES) {
  await page.evaluate(
    ({ id, v }) => {
      const input = document.getElementById(id);
      input.value = v;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    },
    { id: CONTROL, v: value },
  );
  await page.waitForTimeout(3500);
  const label = `${scene}-${CONTROL.replace("-range", "")}${value.replace(".", "")}`;
  const pane = await page.locator(".pane--depth").boundingBox();
  await page.screenshot({ path: `${OUT}/ab-${label}.png`, clip: pane });
  console.log(`${label} @ ${await page.locator("#stat-fps").innerText()} fps`);
}

await context.close();
