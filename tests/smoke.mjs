/**
 * End-to-end smoke test.
 *
 * Drives the built site in Chromium with a synthetic camera, waits for the model
 * to download and warm up, then verifies that the depth canvas contains a
 * non-uniform image. A blank canvas is the failure mode unit tests cannot catch:
 * the pipeline can be wired correctly end to end and still render nothing if the
 * texture format, the tensor layout or the shader is wrong.
 *
 *   npm run preview
 *   node tests/smoke.mjs                 # headless; falls back to WASM
 *   HEADED=1 node tests/smoke.mjs        # real GPU, exercises the WebGPU path
 *
 * Headless Chromium exposes a software WebGPU adapter on which ONNX Runtime
 * hangs at first inference. That is a property of the test environment, not of
 * the site — and the run below is also how the fallback path gets tested.
 */

import { chromium } from "playwright";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const URL_UNDER_TEST =
  process.env.URL ?? "http://127.0.0.1:4173/depth-realtime/#res=182&smooth=0";
const HEADLESS = process.env.HEADED !== "1";
const READY_TIMEOUT = Number(process.env.READY_TIMEOUT ?? 300_000);
// A persistent profile keeps the browser cache between runs, so repeat runs skip
// the weight download entirely.
const PROFILE = process.env.PROFILE ?? mkdtempSync(join(tmpdir(), "depth-smoke-"));

const context = await chromium.launchPersistentContext(PROFILE, {
  headless: HEADLESS,
  args: [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--enable-unsafe-webgpu",
  ],
  permissions: ["camera"],
});
context.setDefaultTimeout(READY_TIMEOUT);
const page = context.pages()[0] ?? (await context.newPage());

const consoleErrors = [];
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});
page.on("pageerror", (error) => consoleErrors.push(`pageerror: ${error.message}`));

let failed = false;
const check = (label, ok, detail = "") => {
  if (!ok) failed = true;
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${detail}` : ""}`);
};

await page.goto(URL_UNDER_TEST, { waitUntil: "load" });
check("page loads", true);
check(
  "both panes present",
  (await page.locator(".pane--camera").count()) === 1 &&
    (await page.locator(".pane--depth").count()) === 1,
);

await page.click("#overlay-action");

await page.waitForFunction(() => Boolean(document.querySelector("video")?.videoWidth), {
  timeout: 30_000,
});
check("camera stream attached", true, await page.locator("#meta-camera").innerText());

await page.waitForFunction(
  () => document.getElementById("info-backend")?.textContent !== "—",
  { timeout: READY_TIMEOUT },
);
check(
  "inference session warmed up",
  true,
  `${await page.locator("#info-backend").innerText()} / ${await page.locator("#info-dtype").innerText()}`,
);

await page.waitForFunction(
  () => document.getElementById("stat-fps")?.textContent !== "—",
  { timeout: READY_TIMEOUT },
);
check("depth tensor produced", true, await page.locator("#info-output").innerText());

// Read the drawing buffer inside an animation frame: with
// preserveDrawingBuffer disabled the contents are only valid until compositing.
const stats = await page.evaluate(
  () =>
    new Promise((resolve) => {
      requestAnimationFrame(() => {
        const canvas = document.getElementById("depth");
        const gl = canvas.getContext("webgl2");
        const w = Math.min(canvas.width, 256);
        const h = Math.min(canvas.height, 256);
        const pixels = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

        let min = 255;
        let max = 0;
        let sum = 0;
        const unique = new Set();
        for (let i = 0; i < pixels.length; i += 4) {
          const luma =
            (pixels[i] * 299 + pixels[i + 1] * 587 + pixels[i + 2] * 114) / 1000;
          if (luma < min) min = luma;
          if (luma > max) max = luma;
          sum += luma;
          unique.add((pixels[i] << 16) | (pixels[i + 1] << 8) | pixels[i + 2]);
        }
        resolve({ w, h, min, max, mean: sum / (pixels.length / 4), unique: unique.size });
      });
    }),
);

check("depth canvas has a drawing buffer", stats.w > 0 && stats.h > 0, `${stats.w}x${stats.h}`);
check(
  "depth field has structure",
  stats.max - stats.min > 20,
  `luma ${stats.min.toFixed(0)}..${stats.max.toFixed(0)}, ${stats.unique} distinct colours`,
);
check("depth field is not black", stats.mean > 4, `mean luma ${stats.mean.toFixed(1)}`);

const fps = await page.locator("#stat-fps").innerText();
const latency = await page.locator("#stat-latency").innerText();
check("telemetry reports numbers", fps !== "—" && latency !== "—", `${fps} fps, ${latency} ms`);

await page.keyboard.press("c");
check(
  "comparison layout engages",
  (await page.locator("#viewer").getAttribute("data-mode")) === "compare",
);
await page.keyboard.press("c");

await page.keyboard.press("m");
check(
  "colour scale cycles",
  (await page.locator("#colormap-select").inputValue()) !== "ice",
  await page.locator("#colormap-select").inputValue(),
);
await page.keyboard.press("m");
await page.keyboard.press("m");

for (const width of [390, 768, 1024, 1440]) {
  await page.setViewportSize({ width, height: 820 });
  // Two frames, so the canvas has resized its drawing buffer before measuring.
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  await page.waitForTimeout(300);

  const overflow = await page.evaluate(() => {
    const limit = document.documentElement.clientWidth;
    const offenders = [];
    for (const element of document.querySelectorAll("*")) {
      const rect = element.getBoundingClientRect();
      if (rect.right > limit + 1) {
        offenders.push(
          `${element.tagName.toLowerCase()}.${String(element.className).split(" ")[0]}@${Math.round(rect.right)}`,
        );
      }
    }
    return {
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: limit,
      offenders: offenders.slice(0, 4),
    };
  });

  check(
    `no horizontal overflow at ${width}px`,
    overflow.scrollWidth <= overflow.clientWidth + 1,
    overflow.offenders.join(", "),
  );
}

check("no console errors", consoleErrors.length === 0, consoleErrors.slice(0, 2).join(" | "));

await context.close();
console.log(failed ? "\nSMOKE TEST FAILED" : "\nSMOKE TEST PASSED");
process.exit(failed ? 1 : 0);
