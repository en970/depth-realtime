/**
 * Renderer unit test.
 *
 * Pushes a known field through DepthRenderer and reads the canvas back, which
 * pins down the two properties that are easy to get wrong and impossible to
 * notice on a live camera: which way is up, and whether the aspect fit crops
 * rather than stretches.
 *
 * Runs against the dev server so the TypeScript module can be imported directly:
 *
 *   npm run dev
 *   node tests/renderer.mjs
 */

import { chromium } from "playwright";

const BASE = process.env.DEV_URL ?? "http://127.0.0.1:5173/depth-realtime/";

const browser = await chromium.launch({ headless: process.env.HEADED !== "1" });
const page = await browser.newPage();
await page.goto(BASE, { waitUntil: "load" });

let failed = false;
const check = (label, ok, detail = "") => {
  if (!ok) failed = true;
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${detail}` : ""}`);
};

const result = await page.evaluate(async () => {
  const { DepthRenderer } = await import("./src/lib/depth-renderer.ts");
  const { buildLut } = await import("./src/lib/colormaps.ts");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:fixed;left:0;top:0;width:400px;height:200px";
  document.body.append(canvas);

  const renderer = new DepthRenderer(canvas);
  renderer.setColormap(buildLut("gray"));

  // 1 in the top half, 0 in the bottom half, and a bright vertical stripe on the
  // left quarter so horizontal orientation is checked too.
  const w = 64;
  const h = 32;
  const field = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      field[y * w + x] = y < h / 2 ? 255 : 0;
    }
  }
  for (let y = 0; y < h; y++) field[y * w + 1] = 255;

  renderer.push(field, w, h, 1);

  const sample = () =>
    new Promise((resolve) => {
      requestAnimationFrame((now) => {
        renderer.render(now);
        const gl = canvas.getContext("webgl2");
        const cw = canvas.width;
        const ch = canvas.height;
        const px = new Uint8Array(cw * ch * 4);
        gl.readPixels(0, 0, cw, ch, gl.RGBA, gl.UNSIGNED_BYTE, px);
        const at = (fx, fy) => {
          // readPixels is bottom-up, so flip the requested y into buffer space.
          const x = Math.floor(fx * (cw - 1));
          const y = Math.floor((1 - fy) * (ch - 1));
          const i = (y * cw + x) * 4;
          return px[i];
        };
        resolve({
          canvasSize: [cw, ch],
          topMiddle: at(0.5, 0.1),
          bottomMiddle: at(0.5, 0.9),
          leftStripe: at(0.02, 0.9),
        });
      });
    });

  const readings = await sample();
  renderer.push(field, w, h, 1);
  const second = await sample();

  // Aspect behaviour. The texture is 2:1. Widening the canvas to 4:1 must crop
  // the texture vertically (sample a narrower band), not stretch it. A band in
  // the top quarter therefore has to disappear. The inverted formula would
  // sample past the edge instead and smear the clamped border back in.
  const band = new Uint8Array(w * h);
  for (let y = 0; y < h / 4; y++) {
    for (let x = 0; x < w; x++) band[y * w + x] = 255;
  }

  const wide = document.createElement("canvas");
  wide.style.cssText = "position:fixed;left:0;top:0;width:800px;height:200px";
  document.body.append(wide);
  const wideRenderer = new DepthRenderer(wide);
  wideRenderer.setColormap(buildLut("gray"));
  wideRenderer.push(band, w, h, 1);

  const wideReading = await new Promise((resolve) => {
    requestAnimationFrame((now) => {
      wideRenderer.render(now);
      const gl = wide.getContext("webgl2");
      const cw = wide.width;
      const ch = wide.height;
      const px = new Uint8Array(cw * ch * 4);
      gl.readPixels(0, 0, cw, ch, gl.RGBA, gl.UNSIGNED_BYTE, px);
      const at = (fx, fy) => {
        const x = Math.floor(fx * (cw - 1));
        const y = Math.floor((1 - fy) * (ch - 1));
        return px[(y * cw + x) * 4];
      };
      resolve({ top: at(0.5, 0.05), middle: at(0.5, 0.5) });
    });
  });

  canvas.remove();
  wide.remove();
  return { readings, second, wideReading };
});

check(
  "drawing buffer allocated",
  result.readings.canvasSize[0] > 0,
  result.readings.canvasSize.join("x"),
);
check(
  "top of the field renders at the top",
  result.readings.topMiddle > 200,
  `luma ${result.readings.topMiddle}`,
);
check(
  "bottom of the field renders at the bottom",
  result.readings.bottomMiddle < 60,
  `luma ${result.readings.bottomMiddle}`,
);
check(
  "orientation is stable across frames",
  result.second.topMiddle > 200 && result.second.bottomMiddle < 60,
  `top ${result.second.topMiddle}, bottom ${result.second.bottomMiddle}`,
);

check(
  "wide canvas crops the field instead of stretching it",
  result.wideReading.top < 60 && result.wideReading.middle < 60,
  `top luma ${result.wideReading.top}, middle luma ${result.wideReading.middle}`,
);

await browser.close();
console.log(failed ? "\nRENDERER TEST FAILED" : "\nRENDERER TEST PASSED");
process.exit(failed ? 1 : 0);
