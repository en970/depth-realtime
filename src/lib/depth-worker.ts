/// <reference lib="webworker" />

/**
 * Inference worker.
 *
 * Everything expensive happens here: weight download, session creation,
 * inference, normalisation and temporal smoothing. The page thread is left with
 * camera playback and a single WebGL draw call, which is what keeps the UI
 * responsive on phones.
 *
 * Model: onnx-community/depth-anything-v2-small (Apache-2.0). Within the Depth
 * Anything V2 family only the Small checkpoint is permissively licensed; Base,
 * Large and Giant are CC-BY-NC-4.0 and cannot ship in an MIT project. The model
 * emits affine-invariant inverse depth, so larger values are nearer and no
 * inversion is applied anywhere in this codebase.
 */

import { AutoModelForDepthEstimation, Tensor } from "@huggingface/transformers";
import type {
  Backend,
  Dtype,
  FromWorker,
  ToWorker,
} from "./protocol";

const MODEL_ID = "onnx-community/depth-anything-v2-small";

/** Percentile clipping bounds. Tighter than min/max, which a single specular
 *  pixel can hijack: measured dynamic-range usage 16% with min/max versus 97%
 *  with p2/p98 on the same frame. */
const P_LOW = 0.02;
const P_HIGH = 0.98;
const HIST_BINS = 1024;

/** Asymmetric smoothing of the bounds: expand quickly so nothing clips, contract
 *  slowly so the image does not pump when a bright object leaves the frame. */
const EMA_EXPAND = 0.35;
const EMA_CONTRACT = 0.08;

/** Tone curve resolution. */
const TONE_BINS = 256;
/**
 * Contrast limit for the adaptive tone curve, as a multiple of the flat
 * histogram. Unlimited equalisation would stretch whatever is most common —
 * often a flat wall — until its quantisation noise became visible structure.
 */
const TONE_CLIP = 3.0;
/** How fast the curve follows the scene. Slow enough that walking across the
 *  room does not make the whole image pulse. */
const TONE_EMA = 0.05;
/* Every pixel is counted. Sampling fewer is cheaper but noisier, and the noise
 * lands directly in the curve: measured at stride 3 the whole-image brightness
 * drift nearly doubled. */

/** A hung WebGPU submission cannot be cancelled, but it can be raced against a
 *  timer so the app falls back instead of showing a permanently empty pane.
 *  WASM gets far longer: its first call also downloads and compiles the ONNX
 *  Runtime binary and runs graph optimisation, which legitimately takes tens of
 *  seconds on a slow connection. */
const TIMEOUTS: Record<Backend, { warmup: number; frame: number }> = {
  webgpu: { warmup: 20_000, frame: 15_000 },
  wasm: { warmup: 90_000, frame: 60_000 },
};

class InferenceTimeout extends Error {
  constructor(label: string) {
    super(`TIMEOUT:${label}`);
    this.name = "InferenceTimeout";
  }
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new InferenceTimeout(label)), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

/** ImageNet statistics, as declared in the model's preprocessor_config.json. */
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

interface Session {
  model: any;
  backend: Backend;
  dtype: Dtype;
}

interface Candidate {
  backend: Backend;
  dtype: Dtype;
}

let session: Session | null = null;
let resolution = 392;
let smoothing = 0.35;
let mobile = false;
let emaLo: number | null = null;
let emaHi: number | null = null;
let smoothed: Float32Array | null = null;
let histogram = new Uint32Array(HIST_BINS);
let toneHistogram = new Uint32Array(TONE_BINS);
let toneCurve: Float32Array | null = null;
let toneStrength = 0;
let busy = false;
let queue: Candidate[] = [];

function post(message: FromWorker, transfer?: Transferable[]): void {
  (self as DedicatedWorkerGlobalScope).postMessage(message, transfer ?? []);
}

/**
 * Ordered list of configurations to try. Each one is created and warmed up
 * before it is accepted, so a driver that fails only at first inference is
 * caught here rather than in front of the user.
 */
async function candidates(
  force?: "webgpu" | "wasm",
  forceDtype?: Dtype,
): Promise<Candidate[]> {
  const wasm: Candidate = { backend: "wasm", dtype: forceDtype ?? "q8" };
  if (force === "wasm") return [wasm];

  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  let adapter: GPUAdapter | null = null;
  if (gpu) {
    try {
      adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    } catch {
      adapter = null;
    }
  }
  if (!adapter) return force === "webgpu" ? [] : [wasm];

  const list: Candidate[] = [];
  if (forceDtype) return [{ backend: "webgpu", dtype: forceDtype }, wasm];
  if (adapter.features.has("shader-f16")) {
    // 19 MB against 50 MB. On a phone the download dominates time-to-first-frame
    // and the quality difference is not visible at these input resolutions.
    list.push({ backend: "webgpu", dtype: mobile ? "q4f16" : "fp16" });
  }
  // fp16 overflow has been observed on some WebGPU drivers, and fp32 is the
  // retry — but only on desktop: the fp32 weights are 99 MB, which is not a
  // reasonable thing to pull over a mobile connection as a second attempt.
  if (!mobile) list.push({ backend: "webgpu", dtype: "fp32" });
  if (force !== "webgpu") list.push(wasm);
  return list;
}

async function createSession(
  backend: Backend,
  dtype: Dtype,
  report: boolean,
): Promise<Session> {
  const seen = new Map<string, { loaded: number; total: number }>();

  const progress_callback = (info: any) => {
    if (!report) return;
    if (info.status === "progress" && info.file?.endsWith(".onnx")) {
      seen.set(info.file, { loaded: info.loaded ?? 0, total: info.total ?? 0 });
      let loaded = 0;
      let total = 0;
      for (const entry of seen.values()) {
        loaded += entry.loaded;
        total += entry.total;
      }
      if (total > 0) {
        post({
          type: "status",
          stage: "downloading",
          progress: Math.min(1, loaded / total),
          detail: `${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`,
        });
      }
    } else if (info.status === "ready" || info.status === "done") {
      post({ type: "status", stage: "compiling" });
    }
  };

  // Deliberately no session_options. Measured on WebGPU/fp16 at 350x196:
  //   enableGraphCapture      needs IO binding, which the library does not use
  //   preferredOutputLocation 'gpu-buffer' makes tensor.data throw, and the
  //                           percentile histogram below runs on the CPU anyway
  //   freeDimensionOverrides  breaks outright: the resolution ladder walks
  //                           182..518 at runtime and warm-up runs at 126x98
  //   graphOptimizationLevel  'all' measured within noise of the default
  // The cost is dominated by resolution, not by dispatch overhead.
  const model = await AutoModelForDepthEstimation.from_pretrained(MODEL_ID, {
    device: backend,
    dtype,
    progress_callback,
  } as any);

  return { model, backend, dtype };
}

/**
 * RGBA bytes to a normalised NCHW tensor, in one pass.
 *
 * The library's image processor would do the same work, but it also re-runs a
 * bicubic resize that the capture canvas has already performed and allocates
 * several intermediate images along the way; measured at 5-8 ms per frame
 * against roughly 1 ms here. The caller is responsible for supplying dimensions
 * that are multiples of the ViT patch size.
 */
function toTensor(rgba: Uint8ClampedArray, width: number, height: number): Tensor {
  const pixels = width * height;
  const data = new Float32Array(3 * pixels);
  const rScale = 1 / (255 * IMAGENET_STD[0]);
  const gScale = 1 / (255 * IMAGENET_STD[1]);
  const bScale = 1 / (255 * IMAGENET_STD[2]);
  const rShift = IMAGENET_MEAN[0] / IMAGENET_STD[0];
  const gShift = IMAGENET_MEAN[1] / IMAGENET_STD[1];
  const bShift = IMAGENET_MEAN[2] / IMAGENET_STD[2];

  for (let i = 0, p = 0; i < pixels; i++, p += 4) {
    data[i] = rgba[p] * rScale - rShift;
    data[pixels + i] = rgba[p + 1] * gScale - gShift;
    data[2 * pixels + i] = rgba[p + 2] * bScale - bShift;
  }

  return new Tensor("float32", data, [1, 3, height, width]);
}

/** Percentile bounds via a 1024-bin histogram: ~0.36 ms for a 518x518 field. */
function computeBounds(data: Float32Array): [number, number] {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || max - min < 1e-9) {
    return [min, min + 1];
  }

  histogram.fill(0);
  const scale = (HIST_BINS - 1) / (max - min);
  for (let i = 0; i < data.length; i++) {
    histogram[((data[i] - min) * scale) | 0]++;
  }

  const lowTarget = data.length * P_LOW;
  const highTarget = data.length * P_HIGH;
  let cumulative = 0;
  let lowBin = 0;
  let highBin = HIST_BINS - 1;
  for (let b = 0; b < HIST_BINS; b++) {
    cumulative += histogram[b];
    if (cumulative >= lowTarget) {
      lowBin = b;
      break;
    }
  }
  cumulative = 0;
  for (let b = 0; b < HIST_BINS; b++) {
    cumulative += histogram[b];
    if (cumulative >= highTarget) {
      highBin = b;
      break;
    }
  }

  return [min + lowBin / scale, min + highBin / scale];
}

/** Last raw percentile bounds, before smoothing. Reported so the stability of
 *  the model output can be told apart from the stability of the smoothing. */
let rawLo = 0;
let rawHi = 1;

/**
 * Rebuilds the tone curve from the histogram collected on the previous frame.
 *
 * The network emits inverse depth, so value is proportional to 1/distance. A
 * person at half a metre and a wall at three metres differ by a factor of six,
 * which puts everything past a couple of metres into the last few percent of
 * the range. Linear scaling cannot fix that — the problem is the shape of the
 * distribution, not its tails — so the curve equalises against the scene's own
 * histogram and spends output range where the data actually is.
 *
 * The clip keeps it honest: unbounded equalisation would take whatever surface
 * is most common, usually a flat wall, and stretch it until its quantisation
 * noise looked like texture.
 *
 * Built from the previous frame's histogram on purpose. The curve is smoothed
 * across frames anyway, so one frame of lag is invisible, and it lets the whole
 * normalisation run as a single pass instead of two.
 */
function rebuildToneCurve(sampleCount: number): void {
  if (sampleCount === 0) return;

  const limit = (sampleCount / TONE_BINS) * TONE_CLIP;
  let excess = 0;
  for (let b = 0; b < TONE_BINS; b++) {
    if (toneHistogram[b] > limit) {
      excess += toneHistogram[b] - limit;
      toneHistogram[b] = limit;
    }
  }
  const share = excess / TONE_BINS;

  let cumulative = 0;
  const target = toneCurve ?? new Float32Array(TONE_BINS);
  const fresh = toneCurve === null;
  for (let b = 0; b < TONE_BINS; b++) {
    cumulative += toneHistogram[b] + share;
    const mapped = cumulative / sampleCount;
    target[b] = fresh ? mapped : target[b] + TONE_EMA * (mapped - target[b]);
  }
  toneCurve = target;
}

/** Samples the tone curve with linear interpolation between bins. */
function applyToneCurve(t: number): number {
  const curve = toneCurve!;
  const x = t * (TONE_BINS - 1);
  const i = Math.min(TONE_BINS - 2, x | 0);
  const f = x - i;
  return curve[i] + (curve[i + 1] - curve[i]) * f;
}

/**
 * Normalises into [0, 1], applies the tone curve and the per-pixel exponential
 * average, in one pass. The smoothing state stays in float; only the
 * transported copy is quantised.
 */
function normalise(data: Float32Array): Float32Array {
  const [lo, hi] = computeBounds(data);
  rawLo = lo;
  rawHi = hi;

  if (emaLo === null || emaHi === null) {
    emaLo = lo;
    emaHi = hi;
  } else {
    emaLo += (lo < emaLo ? EMA_EXPAND : EMA_CONTRACT) * (lo - emaLo);
    emaHi += (hi > emaHi ? EMA_EXPAND : EMA_CONTRACT) * (hi - emaHi);
  }
  const span = Math.max(emaHi - emaLo, 1e-6);
  const base = emaLo;

  const fresh = !smoothed || smoothed.length !== data.length;
  if (fresh) smoothed = new Float32Array(data.length);
  const out = smoothed!;

  const toning = toneStrength > 0.001;
  const curveReady = toning && toneCurve !== null;
  if (toning) toneHistogram.fill(0);
  const binScale = TONE_BINS - 1;

  const a = smoothing;
  const b = 1 - a;
  const blend = !fresh && a > 0.001;

  for (let i = 0; i < data.length; i++) {
    let t = (data[i] - base) / span;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    // Collected before the curve is applied: the curve describes the linear
    // distribution, and feeding it its own output would compound every frame.
    if (toning) toneHistogram[(t * binScale) | 0]++;
    if (curveReady) t += toneStrength * (applyToneCurve(t) - t);
    out[i] = blend ? out[i] * a + t * b : t;
  }

  if (toning) rebuildToneCurve(data.length);
  return out;
}

async function handleFrame(
  id: number,
  buffer: ArrayBuffer,
  width: number,
  height: number,
): Promise<void> {
  if (!session || busy) return;
  busy = true;
  const started = performance.now();

  try {
    const preprocessStart = performance.now();
    const pixel_values = toTensor(new Uint8ClampedArray(buffer), width, height);
    const preprocessMs = performance.now() - preprocessStart;
    const output: any = await withTimeout(
      session.model({ pixel_values }),
      TIMEOUTS[session.backend].frame,
      "inference",
    );
    const tensor = output.predicted_depth ?? output.depth ?? output.logits;
    if (!tensor) throw new Error("Model returned no depth tensor.");

    const dims: number[] = tensor.dims;
    const outHeight = dims[dims.length - 2];
    const outWidth = dims[dims.length - 1];
    const raw = tensor.data as Float32Array;

    // fp16 overflow on some WebGPU drivers yields NaN or a constant field.
    if (!Number.isFinite(raw[0]) || !Number.isFinite(raw[raw.length - 1])) {
      throw new Error("NON_FINITE_OUTPUT");
    }

    const normalised = normalise(raw);

    // Transported as 8-bit: a quarter of the bytes to copy, transfer and upload,
    // and the shader dithers by half an LSB anyway so the quantisation is not
    // visible. The field is a normalised [0, 1] ramp, not measurement data.
    const copy = new Uint8Array(normalised.length);
    for (let i = 0; i < normalised.length; i++) {
      copy[i] = (normalised[i] * 255 + 0.5) | 0;
    }
    const ms = performance.now() - started;

    post(
      {
        type: "result",
        id,
        buffer: copy.buffer,
        width: outWidth,
        height: outHeight,
        ms,
        preprocessMs,
        lo: emaLo ?? 0,
        hi: emaHi ?? 1,
        rawLo,
        rawHi,
      },
      [copy.buffer],
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // A hung submission will hang the retry too, so drop the remaining WebGPU
    // candidates and go straight to WASM.
    if (message.startsWith("TIMEOUT")) {
      queue = queue.filter((candidate) => candidate.backend !== "webgpu");
    }
    session = null;
    post({
      type: "error",
      message: `Inference failed (${message}); switching backend.`,
      fatal: false,
    });
    await activateNext();
  } finally {
    busy = false;
  }
}

/**
 * Runs one synthetic frame through a freshly created session. A configuration
 * is only announced as ready once it has actually produced a finite field:
 * several failure modes (driver shader compilation, fp16 overflow, the ONNX
 * Runtime WebGPU faults reported on WebKit) appear at first inference rather
 * than at session creation.
 */
async function warmUp(candidate: Session): Promise<void> {
  const width = 126; // 9 x 14
  const height = 98; // 7 x 14
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      rgba[i] = (x / width) * 255;
      rgba[i + 1] = (y / height) * 255;
      rgba[i + 2] = 128;
      rgba[i + 3] = 255;
    }
  }

  const budget = TIMEOUTS[candidate.backend].warmup;
  const pixel_values = toTensor(rgba, width, height);
  const output: any = await withTimeout(
    candidate.model({ pixel_values }),
    budget,
    "warmup",
  );
  const tensor = output.predicted_depth ?? output.depth ?? output.logits;
  if (!tensor) throw new Error("Model returned no depth tensor.");
  const data = tensor.data as Float32Array;
  if (data.length === 0 || !Number.isFinite(data[0])) {
    throw new Error("Warm-up produced a non-finite field.");
  }
}

/** Walks the candidate list until one configuration warms up successfully. */
async function activateNext(): Promise<void> {
  while (queue.length > 0) {
    const candidate = queue.shift()!;
    try {
      post({ type: "status", stage: "starting" });
      const created = await createSession(candidate.backend, candidate.dtype, true);
      post({ type: "status", stage: "compiling" });
      await warmUp(created);
      session = created;
      resetTemporalState();
      post({ type: "status", stage: "ready" });
      post({
        type: "ready",
        backend: candidate.backend,
        dtype: candidate.dtype,
        model: MODEL_ID,
      });
      return;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      // A submission that never returns will not return on the next WebGPU
      // configuration either, and each retry costs another weight download.
      if (message.startsWith("TIMEOUT")) {
        queue = queue.filter((entry) => entry.backend !== "webgpu");
      }
      post({
        type: "error",
        message: `${candidate.backend}/${candidate.dtype} unusable: ${message}`,
        fatal: queue.length === 0,
        poisoned: candidate.backend,
      });
    }
  }
  session = null;
}

function resetTemporalState(): void {
  emaLo = null;
  emaHi = null;
  smoothed = null;
  toneCurve = null;
}

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const message = event.data;

  if (message.type === "init") {
    resolution = message.resolution;
    smoothing = message.smoothing;
    toneStrength = message.tone ?? 0;
    mobile = message.mobile;
    post({ type: "status", stage: "starting" });
    queue = await candidates(message.force, message.forceDtype);
    if (queue.length === 0) {
      post({
        type: "error",
        message: "No inference backend is available in this browser.",
        fatal: true,
      });
      return;
    }
    await activateNext();
    return;
  }

  if (message.type === "config") {
    if (typeof message.smoothing === "number") smoothing = message.smoothing;
    if (typeof message.tone === "number") toneStrength = message.tone;
    if (typeof message.resolution === "number" && message.resolution !== resolution) {
      // The page decides the framing; the worker only needs to drop its
      // temporal state, whose buffers are sized to the previous field.
      resolution = message.resolution;
      resetTemporalState();
    }
    return;
  }

  if (message.type === "frame") {
    await handleFrame(message.id, message.buffer, message.width, message.height);
  }
};
