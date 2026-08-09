/**
 * Depth quality measurement.
 *
 * "More detail" is not something to judge by eye: the impression changes with
 * the scene, the lighting and what you were looking at a second ago. This drives
 * the built site with a *fixed* frame — Chromium replays a single-frame y4m as
 * the camera — so every run sees identical input and the numbers are comparable
 * across commits.
 *
 * Three metrics, each answering a different question:
 *
 *   edgeAlignment  Do depth edges land on object silhouettes? Computed as the
 *                  ratio of mean depth-gradient magnitude on strong image edges
 *                  to that on flat image regions. Bilinear magnification blurs
 *                  depth across silhouettes and pushes this down; edge-aware
 *                  upsampling pushes it up.
 *
 *   highFrequency  Laplacian variance of the depth field: how much fine
 *                  structure survives. Rises with real detail, but also with
 *                  noise, so it is only meaningful read together with the two
 *                  others.
 *
 *   temporalDrift  Mean absolute change per pixel between consecutive reads of
 *                  a *static* scene. Input is frozen, so anything non-zero is
 *                  flicker introduced by the pipeline. Detail improvements must
 *                  not buy their gain here.
 *
 * Usage:
 *   npm run preview
 *   node tests/quality.mjs                       # measures, prints JSON
 *   node tests/quality.mjs --save baseline       # also writes a named record
 *   node tests/quality.mjs --compare baseline    # diffs against that record
 */

import { chromium } from "playwright";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const RECORDS = join(HERE, "..", ".continuum", "quality");

const SCENES = (process.env.SCENES ?? "demo01,demo20,hf-depth").split(",");
const SCENE_DIR =
  process.env.SCENE_DIR ??
  "/private/tmp/claude-501/-Users-enes-Developer-depth-realtime/914d624e-f8a0-4f36-a0c6-39e63e8a2cd2/scratchpad/refdata";
const PROFILE =
  process.env.PROFILE ??
  "/private/tmp/claude-501/-Users-enes-Developer-depth-realtime/914d624e-f8a0-4f36-a0c6-39e63e8a2cd2/scratchpad/chrome-profile";
const BASE_URL = process.env.URL ?? "http://127.0.0.1:4173/depth-realtime/";
// Mirroring is off and the resolution is pinned: the measurement must not move
// when the adaptive controller decides the machine is busy.
const HASH = process.env.HASH ?? "#mirror=0&adaptive=0&res=350&smooth=0&mode=split&split=50";

const args = process.argv.slice(2);
const saveAs = args.includes("--save") ? args[args.indexOf("--save") + 1] : null;
const compareTo = args.includes("--compare") ? args[args.indexOf("--compare") + 1] : null;

/** Sampling grid for the metrics. Independent of canvas or tensor size. */
const GRID_W = 320;
const GRID_H = 180;

async function measureScene(scene) {
  const y4m = join(SCENE_DIR, `${scene}.y4m`);
  if (!existsSync(y4m)) throw new Error(`missing scene file: ${y4m}`);

  const context = await chromium.launchPersistentContext(PROFILE, {
    headless: process.env.HEADED !== "1" ? false : false, // always headed: software WebGPU hangs
    args: [
      `--use-file-for-fake-video-capture=${y4m}`,
      "--use-fake-ui-for-media-stream",
      "--enable-unsafe-webgpu",
    ],
    permissions: ["camera"],
    viewport: { width: 1440, height: 900 },
  });
  context.setDefaultTimeout(300_000);
  const page = context.pages()[0] ?? (await context.newPage());

  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto(`${BASE_URL}${HASH}`, { waitUntil: "load" });
  await page.click("#overlay-action");
  await page.waitForFunction(
    () => document.getElementById("stat-fps")?.textContent !== "—",
    { timeout: 300_000 },
  );
  // Let the temporal smoothing settle on the frozen frame.
  await page.waitForTimeout(6000);

  const result = await page.evaluate(
    async ({ gw, gh }) => {
      const video = document.querySelector("video");
      const depth = document.getElementById("depth");

      const grid = document.createElement("canvas");
      grid.width = gw;
      grid.height = gh;
      const ctx = grid.getContext("2d", { willReadFrequently: true });

      /** Samples a source into the fixed grid and returns its luma plane. */
      const sampleLuma = (source) => {
        ctx.clearRect(0, 0, gw, gh);
        ctx.drawImage(source, 0, 0, gw, gh);
        const { data } = ctx.getImageData(0, 0, gw, gh);
        const luma = new Float32Array(gw * gh);
        for (let i = 0, p = 0; i < luma.length; i++, p += 4) {
          luma[i] = (data[p] * 299 + data[p + 1] * 587 + data[p + 2] * 114) / 1000;
        }
        return luma;
      };

      const sobel = (plane) => {
        const out = new Float32Array(gw * gh);
        for (let y = 1; y < gh - 1; y++) {
          for (let x = 1; x < gw - 1; x++) {
            const i = y * gw + x;
            const gx =
              -plane[i - gw - 1] - 2 * plane[i - 1] - plane[i + gw - 1] +
              plane[i - gw + 1] + 2 * plane[i + 1] + plane[i + gw + 1];
            const gy =
              -plane[i - gw - 1] - 2 * plane[i - gw] - plane[i - gw + 1] +
              plane[i + gw - 1] + 2 * plane[i + gw] + plane[i + gw + 1];
            out[i] = Math.hypot(gx, gy);
          }
        }
        return out;
      };

      const laplacianVariance = (plane) => {
        const values = [];
        for (let y = 1; y < gh - 1; y++) {
          for (let x = 1; x < gw - 1; x++) {
            const i = y * gw + x;
            values.push(
              4 * plane[i] - plane[i - 1] - plane[i + 1] - plane[i - gw] - plane[i + gw],
            );
          }
        }
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        return values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
      };

      const waitFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

      // Two reads a second apart: on a frozen input the difference is drift.
      await waitFrame();
      const depthA = sampleLuma(depth);
      const image = sampleLuma(video);
      await new Promise((r) => setTimeout(r, 1000));
      await waitFrame();
      const depthB = sampleLuma(depth);

      const imageGrad = sobel(image);
      const depthGrad = sobel(depthA);

      // Split image pixels into "on an edge" and "flat" by the 85th percentile
      // of image gradient, then compare mean depth gradient across the two.
      const sorted = Float32Array.from(imageGrad).sort();
      const threshold = sorted[Math.floor(sorted.length * 0.85)];
      let edgeSum = 0;
      let edgeCount = 0;
      let flatSum = 0;
      let flatCount = 0;
      for (let i = 0; i < imageGrad.length; i++) {
        if (imageGrad[i] >= threshold) {
          edgeSum += depthGrad[i];
          edgeCount++;
        } else {
          flatSum += depthGrad[i];
          flatCount++;
        }
      }
      const edgeMean = edgeCount ? edgeSum / edgeCount : 0;
      const flatMean = flatCount ? flatSum / flatCount : 0;

      let drift = 0;
      for (let i = 0; i < depthA.length; i++) drift += Math.abs(depthA[i] - depthB[i]);
      drift /= depthA.length;

      let depthMin = 255;
      let depthMax = 0;
      for (let i = 0; i < depthA.length; i++) {
        if (depthA[i] < depthMin) depthMin = depthA[i];
        if (depthA[i] > depthMax) depthMax = depthA[i];
      }

      return {
        edgeAlignment: flatMean > 0.001 ? edgeMean / flatMean : 0,
        edgeGradient: edgeMean,
        flatGradient: flatMean,
        highFrequency: laplacianVariance(depthA),
        temporalDrift: drift,
        dynamicRange: depthMax - depthMin,
        fps: Number(document.getElementById("stat-fps").textContent),
        inferenceMs: Number(document.getElementById("stat-latency").textContent),
        output: document.getElementById("info-output").textContent,
        backend: document.getElementById("info-backend").textContent,
      };
    },
    { gw: GRID_W, gh: GRID_H },
  );

  await context.close();
  if (errors.length) result.pageErrors = errors.slice(0, 3);
  return result;
}

const record = { scenes: {} };
for (const scene of SCENES) {
  process.stdout.write(`measuring ${scene} ... `);
  const metrics = await measureScene(scene);
  record.scenes[scene] = metrics;
  console.log(
    `edgeAlignment=${metrics.edgeAlignment.toFixed(3)} ` +
      `highFreq=${metrics.highFrequency.toFixed(1)} ` +
      `drift=${metrics.temporalDrift.toFixed(3)} ` +
      `${metrics.fps}fps`,
  );
}

const keys = Object.keys(record.scenes);
record.mean = {
  edgeAlignment:
    keys.reduce((a, k) => a + record.scenes[k].edgeAlignment, 0) / keys.length,
  highFrequency:
    keys.reduce((a, k) => a + record.scenes[k].highFrequency, 0) / keys.length,
  temporalDrift:
    keys.reduce((a, k) => a + record.scenes[k].temporalDrift, 0) / keys.length,
  fps: keys.reduce((a, k) => a + record.scenes[k].fps, 0) / keys.length,
};

console.log("\nmean:", JSON.stringify(record.mean, null, 1));

if (saveAs) {
  mkdirSync(RECORDS, { recursive: true });
  writeFileSync(join(RECORDS, `${saveAs}.json`), JSON.stringify(record, null, 2));
  console.log(`saved as ${saveAs}`);
}

if (compareTo) {
  const path = join(RECORDS, `${compareTo}.json`);
  if (!existsSync(path)) {
    console.error(`no record named ${compareTo}`);
    process.exit(1);
  }
  const before = JSON.parse(readFileSync(path, "utf8"));
  const pct = (now, then) => (then === 0 ? "n/a" : `${(((now - then) / then) * 100).toFixed(1)}%`);
  console.log(`\nagainst ${compareTo}:`);
  console.log(`  edgeAlignment ${before.mean.edgeAlignment.toFixed(3)} -> ${record.mean.edgeAlignment.toFixed(3)}  (${pct(record.mean.edgeAlignment, before.mean.edgeAlignment)})`);
  console.log(`  highFrequency ${before.mean.highFrequency.toFixed(1)} -> ${record.mean.highFrequency.toFixed(1)}  (${pct(record.mean.highFrequency, before.mean.highFrequency)})`);
  console.log(`  temporalDrift ${before.mean.temporalDrift.toFixed(3)} -> ${record.mean.temporalDrift.toFixed(3)}  (${pct(record.mean.temporalDrift, before.mean.temporalDrift)})`);
  console.log(`  fps           ${before.mean.fps.toFixed(1)} -> ${record.mean.fps.toFixed(1)}  (${pct(record.mean.fps, before.mean.fps)})`);
}
