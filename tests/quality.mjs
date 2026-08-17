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
 *   farGradient    Mean depth-gradient magnitude over the far half of the
 *                  scene, and farSpread its standard deviation. The other
 *                  metrics are dominated by the near subject; these are the
 *                  ones that answer "is the background just a black mass".
 *
 *   temporalDrift  Mean absolute change per pixel between consecutive reads of
 *                  a *static* scene. Input is frozen, so anything non-zero is
 *                  flicker introduced by the pipeline. Detail improvements must
 *                  not buy their gain here.
 *
 * Two modes:
 *
 *   Absolute — one measurement of the current settings, saved or compared
 *   against a stored record. Use for tracking the project over time.
 *
 *   Paired (--vary) — several settings measured inside ONE browser session,
 *   with the control changed in place. Use for deciding whether a change helps.
 *   Separate sessions are not comparable: the model output drifts, the
 *   normalisation settles differently, and the machine load moves underneath.
 *   Measured, the same settings across sessions gave edge alignment anywhere
 *   between 3.9 and 7.5. In one session the same comparison is stable.
 *
 *   The variants are measured forwards then backwards and averaged, so that any
 *   drift over the session — thermal, cache, adaptive state — falls on all of
 *   them equally instead of favouring whichever went first.
 *
 * Usage:
 *   npm run preview
 *   node tests/quality.mjs                        # measure current settings
 *   node tests/quality.mjs --save baseline        # ... and store it
 *   node tests/quality.mjs --compare baseline     # ... and diff against it
 *   node tests/quality.mjs --vary tone=0,0.7      # paired comparison
 *   node tests/quality.mjs --vary structure=0,0.6,1
 */

import { chromium } from "playwright";
import { HEADLESS, GPU_ARGS, sceneArgs } from "./browser.mjs";
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
const varySpec = args.includes("--vary") ? args[args.indexOf("--vary") + 1] : null;

/** Settings that can be changed in place, without rebuilding the session. */
const CONTROLS = {
  tone: "tone-range",
  structure: "structure-range",
  guide: "guide-range",
  smooth: "smoothing-range",
  motion: "motion-range",
  contours: "contour-range",
  parallax: "parallax-range",
  res: "resolution-range",
};

/**
 * Sampling grid for the metrics.
 *
 * It has to out-resolve the thing being measured. At 320x180 a comparison
 * between a 322 px field and a 518 px one showed the higher resolution scoring
 * *lower* on fine structure, which is an artefact of the grid: everything the
 * larger field resolved beyond the grid was averaged away before it was counted.
 * 640x360 covers the whole ladder with room to spare.
 */
const GRID_W = Number(process.env.GRID_W ?? 640);
const GRID_H = Number(process.env.GRID_H ?? 360);

async function openScene(scene) {
  const y4m = join(SCENE_DIR, `${scene}.y4m`);
  if (!existsSync(y4m)) throw new Error(`missing scene file: ${y4m}`);

  const context = await chromium.launchPersistentContext(PROFILE, {
    headless: HEADLESS,
    args: [...GPU_ARGS, ...sceneArgs(y4m)],
    permissions: ["camera"],
    viewport: { width: 1440, height: 900 },
  });
  context.setDefaultTimeout(300_000);
  const page = context.pages()[0] ?? (await context.newPage());

  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  // A stored preset from an earlier run must not silently override the settings
  // under test.
  await page.addInitScript(() => localStorage.clear());

  await page.goto(`${BASE_URL}${HASH}`, { waitUntil: "load" });
  await page.click("#overlay-action");
  await page.waitForFunction(
    () => document.getElementById("stat-fps")?.textContent !== "—",
    { timeout: 300_000 },
  );
  // The first seconds are not representative: the adaptive controller, the
  // normalisation bounds and the browser's own warm-up are all still moving.
  await page.waitForTimeout(10_000);
  return { context, page, errors };
}

/** Sets a control and waits for the pipeline to settle on the new value. */
async function applySetting(page, name, value) {
  const control = CONTROLS[name];
  if (!control) throw new Error(`no in-place control for "${name}"`);
  await page.evaluate(
    ({ id, v }) => {
      const input = document.getElementById(id);
      input.value = v;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    },
    { id: control, v: String(value) },
  );
  // The tone curve and the normalisation bounds are both smoothed across
  // frames, so a changed setting needs time before it is what is measured.
  await page.waitForTimeout(Number(process.env.SETTLE ?? 5000));
}

async function collect(page) {
  const REPEATS = Number(process.env.REPEATS ?? 5);
  const readings = [];
  for (let attempt = 0; attempt < REPEATS; attempt++) {
    readings.push(await readOnce(page));
    await page.waitForTimeout(700);
  }

  const median = (values) => {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  };
  const spread = (values) => {
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const mid = median(values);
    return mid === 0 ? 0 : (hi - lo) / mid;
  };

  const result = {};
  for (const key of Object.keys(readings[0])) {
    const values = readings.map((r) => r[key]);
    result[key] = typeof values[0] === "number" ? median(values) : values[0];
  }
  result.spreadEdgeAlignment = spread(readings.map((r) => r.edgeAlignment));
  result.spreadFarGradient = spread(readings.map((r) => r.farGradient));
  result.spreadHighFrequency = spread(readings.map((r) => r.highFrequency));
  return result;
}

async function measureScene(scene) {
  const { context, page, errors } = await openScene(scene);
  const result = await collect(page);
  await context.close();
  if (errors.length) result.pageErrors = errors.slice(0, 3);
  return result;
}

/**
 * Measures several values of one setting inside a single session, forwards then
 * backwards, averaging the two directions so that drift over the session falls
 * on every variant equally instead of favouring whichever went first.
 */
async function measurePaired(scene, name, values) {
  const { context, page, errors } = await openScene(scene);
  const order = [...values, ...[...values].reverse()];
  const collected = new Map(values.map((v) => [v, []]));

  for (const value of order) {
    await applySetting(page, name, value);
    collected.get(value).push(await collect(page));
  }

  await context.close();

  const out = {};
  for (const [value, runs] of collected) {
    const averaged = {};
    for (const key of Object.keys(runs[0])) {
      const numbers = runs.map((r) => r[key]).filter((n) => typeof n === "number");
      averaged[key] = numbers.length
        ? numbers.reduce((a, b) => a + b, 0) / numbers.length
        : runs[0][key];
    }
    out[value] = averaged;
  }
  if (errors.length) out.pageErrors = errors.slice(0, 3);
  return out;
}

async function readOnce(page) {
  return page.evaluate(
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

      // Three reads: one immediately after another (frame-scale flicker) and one
      // a second later (slow drift). Temporal smoothing suppresses the first but
      // cannot touch the second, so separating them says which one to fix.
      await waitFrame();
      const depthA = sampleLuma(depth);
      const image = sampleLuma(video);
      await waitFrame();
      await waitFrame();
      const depthShort = sampleLuma(depth);
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

      const meanOf = (plane) => {
        let sum = 0;
        for (let i = 0; i < plane.length; i++) sum += plane[i];
        return sum / plane.length;
      };

      /** Splits the change into a whole-image brightness shift and the residual
       *  per-pixel variation. A shift means the normalisation bounds are moving;
       *  residual means the field itself is unstable. The fix differs. */
      const compare = (a, b) => {
        const ma = meanOf(a);
        const mb = meanOf(b);
        let total = 0;
        let residual = 0;
        for (let i = 0; i < a.length; i++) {
          total += Math.abs(a[i] - b[i]);
          residual += Math.abs(a[i] - ma - (b[i] - mb));
        }
        return {
          total: total / a.length,
          shift: Math.abs(ma - mb),
          residual: residual / a.length,
        };
      };

      const shortTerm = compare(depthA, depthShort);
      const longTerm = compare(depthA, depthB);
      const drift = longTerm.total;

      let depthMin = 255;
      let depthMax = 0;
      for (let i = 0; i < depthA.length; i++) {
        if (depthA[i] < depthMin) depthMin = depthA[i];
        if (depthA[i] > depthMax) depthMax = depthA[i];
      }

      // Detail in the far half of the scene, which the other metrics miss
      // entirely: they are dominated by the near subject, where contrast was
      // never the problem. Inverse depth compresses everything distant into a
      // few levels, so this is where "the background is just black" shows up.
      const sortedDepth = Float32Array.from(depthA).sort();
      const median = sortedDepth[sortedDepth.length >> 1];
      let farGradient = 0;
      let farCount = 0;
      let farSum = 0;
      let farSumSq = 0;
      for (let i = 0; i < depthA.length; i++) {
        if (depthA[i] <= median) {
          farGradient += depthGrad[i];
          farSum += depthA[i];
          farSumSq += depthA[i] * depthA[i];
          farCount++;
        }
      }
      const farMean = farCount ? farSum / farCount : 0;
      const farVariance = farCount ? farSumSq / farCount - farMean * farMean : 0;

      return {
        edgeAlignment: flatMean > 0.001 ? edgeMean / flatMean : 0,
        edgeGradient: edgeMean,
        flatGradient: flatMean,
        highFrequency: laplacianVariance(depthA),
        temporalDrift: drift,
        driftShort: shortTerm.total,
        driftShortShift: shortTerm.shift,
        driftShortResidual: shortTerm.residual,
        driftLongShift: longTerm.shift,
        driftLongResidual: longTerm.residual,
        dynamicRange: depthMax - depthMin,
        farGradient: farCount ? farGradient / farCount : 0,
        farSpread: Math.sqrt(Math.max(0, farVariance)),
        fps: Number(document.getElementById("stat-fps").textContent),
        inferenceMs: Number(document.getElementById("stat-latency").textContent),
        output: document.getElementById("info-output").textContent,
        backend: document.getElementById("info-backend").textContent,
      };
    },
    { gw: GRID_W, gh: GRID_H },
  );
}

if (varySpec) {
  const [name, listed] = varySpec.split("=");
  const values = listed.split(",");
  if (!CONTROLS[name]) {
    console.error(`unknown setting "${name}". Available: ${Object.keys(CONTROLS).join(", ")}`);
    process.exit(1);
  }

  for (const scene of SCENES) {
    console.log(`\n${scene}, varying ${name}:`);
    const results = await measurePaired(scene, name, values);
    const rows = values.map((v) => ({ value: v, ...results[v] }));
    const first = rows[0];
    for (const row of rows) {
      const delta = (now, then) =>
        then === 0 ? "" : ` (${now >= then ? "+" : ""}${(((now - then) / then) * 100).toFixed(0)}%)`;
      console.log(
        `  ${name}=${row.value.padEnd(5)} ` +
          `edge=${row.edgeAlignment.toFixed(2)}${row === first ? "" : delta(row.edgeAlignment, first.edgeAlignment)} ` +
          `far=${row.farGradient.toFixed(2)}${row === first ? "" : delta(row.farGradient, first.farGradient)} ` +
          `highFreq=${row.highFrequency.toFixed(1)}${row === first ? "" : delta(row.highFrequency, first.highFrequency)} ` +
          `drift=${row.temporalDrift.toFixed(2)} ` +
          `fps=${row.fps.toFixed(1)}`,
      );
    }
  }
  process.exit(0);
}

const record = { scenes: {} };
for (const scene of SCENES) {
  process.stdout.write(`measuring ${scene} ... `);
  const metrics = await measureScene(scene);
  record.scenes[scene] = metrics;
  console.log(
    `edgeAlignment=${metrics.edgeAlignment.toFixed(3)} ` +
      `highFreq=${metrics.highFrequency.toFixed(1)} ` +
      `farGrad=${metrics.farGradient.toFixed(2)} ` +
      `farSpread=${metrics.farSpread.toFixed(1)} ` +
      `drift=${metrics.temporalDrift.toFixed(2)} ` +
      `(shift ${metrics.driftLongShift.toFixed(2)} / resid ${metrics.driftLongResidual.toFixed(2)}) ` +
      `frame=${metrics.driftShort.toFixed(2)} ` +
      `${metrics.fps}fps ` +
      `[spread ${(metrics.spreadEdgeAlignment * 100).toFixed(0)}%]`,
  );
}

const keys = Object.keys(record.scenes);
record.mean = {
  edgeAlignment:
    keys.reduce((a, k) => a + record.scenes[k].edgeAlignment, 0) / keys.length,
  farGradient:
    keys.reduce((a, k) => a + record.scenes[k].farGradient, 0) / keys.length,
  farSpread:
    keys.reduce((a, k) => a + record.scenes[k].farSpread, 0) / keys.length,
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
