/**
 * Shared browser launch settings for the test scripts.
 *
 * Headless Chromium picks SwiftShader by default, and ONNX Runtime's WebGPU
 * path hangs at first inference on it — which is why these tests used to need a
 * visible window. Asking ANGLE for Metal gives the real GPU (vendor "apple",
 * architecture "metal-3", shader-f16 available) with no window at all, so runs
 * stay out of the way. Verified with tests/_gpuprobe.mjs.
 *
 * HEADED=1 still opens a window when something needs to be watched.
 */

export const HEADLESS = process.env.HEADED !== "1";

export const GPU_ARGS = [
  "--enable-unsafe-webgpu",
  "--use-angle=metal",
  // If a window is opened deliberately, keep it off-screen and stop the
  // compositor from throttling it; an occluded window would otherwise pause
  // rendering and ruin every timing measurement.
  ...(HEADLESS
    ? []
    : [
        "--window-position=-4000,-4000",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--disable-features=CalculateNativeWinOcclusion",
      ]),
];

export const CAMERA_ARGS = ["--use-fake-ui-for-media-stream"];

/** Synthetic moving pattern, for tests that only need *a* camera. */
export const FAKE_CAMERA_ARGS = [...CAMERA_ARGS, "--use-fake-device-for-media-stream"];

/** A frozen scene, for measurements that must be reproducible. */
export const sceneArgs = (y4mPath) => [...CAMERA_ARGS, `--use-file-for-fake-video-capture=${y4mPath}`];
