/**
 * Builds the measurement scenes.
 *
 * The quality harness feeds Chromium a y4m file as the camera, so every run sees
 * identical input. Those files live outside the repository — they are large, and
 * the photographs are not ours to redistribute — which means they vanish
 * whenever the scratch directory is cleared. This rebuilds them from their
 * sources in one step.
 *
 *   node tests/fixtures.mjs
 *
 * Produces, at 1280x720:
 *   demo01, demo20, hf-depth   single frozen frames, for absolute measurements
 *   pan                        60 frames panning slowly, for anything temporal
 *
 * A frozen frame is the right input for measuring detail, because every change
 * in the output is then the pipeline's doing. It is the wrong input for
 * measuring anything about motion, which is what `pan` is for.
 */

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdirSync, existsSync, statSync } from "node:fs";
import { join } from "node:path";

const run = promisify(execFile);

const DIR =
  process.env.SCENE_DIR ??
  "/private/tmp/claude-501/-Users-enes-Developer-depth-realtime/914d624e-f8a0-4f36-a0c6-39e63e8a2cd2/scratchpad/refdata";

const SOURCES = {
  demo01:
    "https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/assets/examples/demo01.jpg",
  demo20:
    "https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/assets/examples/demo20.jpg",
  "hf-depth":
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-estimation-example.jpg",
};

mkdirSync(DIR, { recursive: true });

for (const [name, url] of Object.entries(SOURCES)) {
  const jpg = join(DIR, `${name}.jpg`);
  if (!existsSync(jpg)) {
    await run("curl", ["-sL", "-o", jpg, "--max-time", "120", url]);
    console.log(`fetched ${name}.jpg (${(statSync(jpg).size / 1e3).toFixed(0)} kB)`);
  }

  const y4m = join(DIR, `${name}.y4m`);
  if (!existsSync(y4m)) {
    await run("ffmpeg", [
      "-y", "-loglevel", "error", "-i", jpg,
      "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
      "-pix_fmt", "yuv420p", "-frames:v", "1", "-f", "yuv4mpegpipe", y4m,
    ]);
    console.log(`built ${name}.y4m (${(statSync(y4m).size / 1e6).toFixed(1)} MB)`);
  }
}

// A slow pan, for measuring anything that depends on the scene moving:
// temporal smoothing, motion trails, accumulation.
const pan = join(DIR, "pan.y4m");
if (!existsSync(pan)) {
  await run("ffmpeg", [
    "-y", "-loglevel", "error", "-loop", "1", "-i", join(DIR, "demo20.jpg"),
    "-vf",
    "crop=1400:788:x='min(iw-1400,t*90)':y='min(ih-788,t*20)',scale=1280:720",
    "-t", "2", "-r", "30", "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", pan,
  ]);
  console.log(`built pan.y4m (${(statSync(pan).size / 1e6).toFixed(1)} MB)`);
}

console.log(`scenes ready in ${DIR}`);
