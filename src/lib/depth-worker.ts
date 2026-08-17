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

/**
 * Two networks, selectable for comparison.
 *
 * V2 emits affine-invariant *inverse* depth: larger is nearer. V3 exponentiates
 * log depth in its head, so it emits depth directly and larger is farther — the
 * opposite convention, which has to be undone before anything downstream sees
 * it. V3 also takes a rank-5 input, with a leading image count.
 *
 * V3 resolves distant structure far better: measured on a portrait with a deep
 * background, the far third of the frame kept 179 distinct levels against V2's
 * 10. It is not the default yet because only an 8-bit export exists, and 8-bit
 * weights are not a fast path on WebGPU.
 */
const MODELS = {
  v2: {
    id: "onnx-community/depth-anything-v2-small",
    rank5: false,
    /** Inverse depth: already large-is-near. */
    invert: false,
  },
  v3: {
    id: "en970/depth-anything-v3-small-onnx",
    rank5: true,
    /** Depth: large-is-far, so it must be flipped. */
    invert: true,
  },
} as const;

let modelKey: keyof typeof MODELS = "v2";
const modelSpec = () => MODELS[modelKey];

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

/**
 * Floor for the logarithm. The DPT head ends in ReLU, so the field is
 * non-negative and can reach exactly zero.
 */
const LOG_FLOOR = 1e-4;

/**
 * How sharply temporal smoothing backs off where the field is moving.
 *
 * The first attempt ramped linearly from zero, which failed for a measurable
 * reason: the frame-to-frame difference of a still pixel is not zero, it is the
 * noise floor (0.0035 of the range, measured). A linear ramp therefore let noise
 * exempt itself from the smoothing that would have quieted it, and the pixel got
 * noisier — drift rose from 5.9 to 8.4 as sensitivity went up.
 *
 * A threshold fixes that. Below the noise floor nothing counts as motion and the
 * full rate applies; above a silhouette-sized change the smoothing is off. The
 * point of getting this right is that it lets the rate itself go up: measured,
 * 0.35 removes 31% of the flicker and 0.8 removes 67%, and the high rate is only
 * safe once still and moving pixels are told apart.
 */
let motionSensitivity = 8;

/** Below this fraction of the range, a change is noise rather than motion. */
const MOTION_FLOOR = 0.006;
/** Above this, it is a moving silhouette and smoothing should be off entirely. */
const MOTION_CEILING = 0.06;

/**
 * Frame interval the temporal settings are calibrated against, in milliseconds.
 *
 * Both the smoothing rate and the motion sensitivity are defined per frame, but
 * the interval between frames varies by a factor of six across devices: about
 * 48 ms on the desktop this was tuned on, around 280 ms on a Galaxy S22. Left
 * uncorrected that means two different things. A rate of 0.35 gives a 31 ms
 * half-life here and 189 ms there — the same slider, six times the lag. And the
 * motion term is the change between two inferences, which grows with the
 * interval, so a sensitivity of 5 behaves like 30 on the slower device and
 * switches the smoothing off almost everywhere.
 *
 * Scaling both by the observed interval makes the settings mean the same thing
 * on any device.
 */
const REFERENCE_INTERVAL_MS = 48;
let lastFrameAt = 0;

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
    // fp16 everywhere, including mobile.
    //
    // 4-bit weights were the mobile default, for a 19 MB download against 50 MB.
    // They are not worth it. On a Galaxy S22 they produced diagonal banding with
    // no relation to the scene — arithmetic failing quietly rather than erroring
    // — and even where they work they are slower than fp16, not faster:
    // measured 12.4 fps against 19.4 on the same machine, because dequantising
    // costs more than the narrower weights save.
    list.push({ backend: "webgpu", dtype: "fp16" });
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
  const model = await AutoModelForDepthEstimation.from_pretrained(modelSpec().id, {
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

  return new Tensor(
    "float32",
    data,
    modelSpec().rank5 ? [1, 1, 3, height, width] : [1, 3, height, width],
  );
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
 * Normalises into [0, 1], applies the distance transfer curve and the per-pixel
 * exponential average, in one pass.
 *
 * The network emits inverse depth: value is proportional to 1/distance. Scaling
 * that linearly is what made the background a flat dark mass — a person at half
 * a metre and a wall at three metres differ by a factor of six, so everything
 * past a couple of metres lands in the last few percent of the range. Measured
 * on a real scene, the far third of the image came out with ten distinct levels
 * out of 256, at a mean brightness of 0.6.
 *
 * Taking the logarithm inverts exactly the transform that caused it: log(1/z)
 * is linear in log distance, so equal ratios of distance get equal shares of
 * the output. The same scene goes from ten levels to seventy-nine.
 *
 * Contrast-limited histogram equalisation was tried here first and does not
 * work, at any histogram resolution: equalisation cannot separate pixels that
 * share a bin, and after linear scaling the entire background shares one.
 * Measured, it moved that scene from ten levels to twelve.
 *
 * Blended rather than switched, because the curve spends on distance what it
 * takes from the foreground, and which one matters depends on the scene.
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

  const logBase = Math.log(Math.max(emaLo, LOG_FLOOR));
  const logSpan = Math.max(Math.log(Math.max(emaHi, LOG_FLOOR)) - logBase, 1e-6);

  const fresh = !smoothed || smoothed.length !== data.length;
  if (fresh) smoothed = new Float32Array(data.length);
  const out = smoothed!;

  const toning = toneStrength > 0.001;
  const inverted = modelSpec().invert;

  // Rescale the temporal settings to the interval actually being achieved, so
  // the time constant is the same on a phone as on a laptop.
  const timestamp = performance.now();
  const interval = lastFrameAt
    ? Math.min(400, Math.max(8, timestamp - lastFrameAt))
    : REFERENCE_INTERVAL_MS;
  lastFrameAt = timestamp;
  const intervalRatio = interval / REFERENCE_INTERVAL_MS;

  // a^ratio: two frames at half the rate leave the same weight behind as one at
  // the full rate.
  const a = smoothing > 0.001 ? Math.pow(smoothing, intervalRatio) : 0;
  // The motion term grows with the interval, so the threshold has to as well.
  const sensitivity = motionSensitivity / intervalRatio;
  // The thresholds are expressed against a sensitivity of 1, so they scale with
  // it rather than being re-tuned every time the slider moves.
  const floor = MOTION_FLOOR * motionSensitivity;
  const ceiling = MOTION_CEILING * motionSensitivity;
  const blend = !fresh && a > 0.001;

  for (let i = 0; i < data.length; i++) {
    const raw = data[i];
    let t = (raw - base) / span;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    if (inverted) t = 1 - t;

    // Only meaningful for inverse depth: V3 already emits linear depth, and
    // taking the logarithm again re-compresses the background it just opened up.
    if (toning && !inverted) {
      let curved = (Math.log(raw > LOG_FLOOR ? raw : LOG_FLOOR) - logBase) / logSpan;
      curved = curved < 0 ? 0 : curved > 1 ? 1 : curved;
      t += toneStrength * (curved - t);
    }

    if (blend) {
      const previous = out[i];
      let rate = a;
      if (sensitivity > 0) {
        // Smoothing a still scene removes noise for free; smoothing a moving one
        // drags the old position behind the new, which costs detail exactly
        // where the eye is looking.
        const motion = previous > t ? previous - t : t - previous;
        const scaled = motion * sensitivity;
        if (scaled >= ceiling) {
          rate = 0;
        } else if (scaled > floor) {
          // Smoothstep between the two, so the transition does not itself show
          // up as an edge in the output.
          const x = (scaled - floor) / (ceiling - floor);
          rate = a * (1 - x * x * (3 - 2 * x));
        }
      }
      out[i] = rate > 0 ? previous * rate + t * (1 - rate) : t;
    } else {
      out[i] = t;
    }
  }

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

  /** Two different synthetic frames, one varying across, one down. */
  const build = (vertical: boolean) => {
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        const ramp = vertical ? y / height : x / width;
        rgba[i] = ramp * 255;
        rgba[i + 1] = (1 - ramp) * 255;
        rgba[i + 2] = ((x + y) % 32) * 8;
        rgba[i + 3] = 255;
      }
    }
    return rgba;
  };

  const budget = TIMEOUTS[candidate.backend].warmup;
  const infer = async (rgba: Uint8ClampedArray) => {
    const pixel_values = toTensor(rgba, width, height);
    const output: any = await withTimeout(candidate.model({ pixel_values }), budget, "warmup");
    const tensor = output.predicted_depth ?? output.depth ?? output.logits;
    if (!tensor) throw new Error("Model returned no depth tensor.");
    return tensor.data as Float32Array;
  };

  const first = await infer(build(false));
  if (first.length === 0 || !Number.isFinite(first[0])) {
    throw new Error("Warm-up produced a non-finite field.");
  }

  // A depth field is smooth: neighbouring samples differ by a small fraction of
  // the range, because surfaces are continuous. Arithmetic that has gone wrong
  // on a particular driver produces something with the same range and the same
  // responsiveness, but full of high-frequency structure — on one phone, a
  // diagonal banding pattern with no relation to the scene.
  //
  // Measured on a working model: 0.003 for both synthetic ramps and a real
  // photograph. White noise gives 0.33, diagonal banding 0.28. The threshold
  // sits an order of magnitude above healthy and well below either failure.
  const columns = Math.round(Math.sqrt((first.length * width) / height));
  const rows = Math.max(1, Math.floor(first.length / Math.max(columns, 1)));
  let range = { min: first[0], max: first[0] };
  for (let i = 1; i < first.length; i++) {
    if (first[i] < range.min) range.min = first[i];
    if (first[i] > range.max) range.max = first[i];
  }
  const spread = range.max - range.min;
  if (!(spread > 1e-4)) throw new Error("Warm-up output is constant.");

  let variation = 0;
  let samples = 0;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x + 1 < columns; x++) {
      const i = y * columns + x;
      if (i + 1 >= first.length) break;
      variation += Math.abs(first[i + 1] - first[i]);
      samples++;
    }
  }
  const roughness = samples > 0 ? variation / samples / spread : 0;
  if (roughness > 0.05) {
    throw new Error(`Warm-up output is not a depth field (roughness ${roughness.toFixed(3)}).`);
  }

  // A finite field is not the same as a working one. Reduced-precision weights
  // have been reported to produce plausible-looking noise on some drivers rather
  // than failing outright, and a viewer cannot tell the difference by looking at
  // a single frame. These two checks catch both ways that goes wrong: a field
  // that never varies, and one that ignores its input.
  const second = await infer(build(true));
  if (second.length !== first.length) throw new Error("Warm-up output size changed.");

  // Correlation between the responses to two different inputs. A working model
  // reacts differently to a horizontal ramp than to a vertical one; weights that
  // have collapsed produce nearly the same field whatever they are shown.
  let sumA = 0;
  let sumB = 0;
  for (let i = 0; i < first.length; i++) {
    sumA += first[i];
    sumB += second[i];
  }
  const meanA = sumA / first.length;
  const meanB = sumB / second.length;
  let cov = 0;
  let varA = 0;
  let varB = 0;
  for (let i = 0; i < first.length; i++) {
    const da = first[i] - meanA;
    const db = second[i] - meanB;
    cov += da * db;
    varA += da * da;
    varB += db * db;
  }
  const correlation = cov / Math.sqrt(Math.max(varA * varB, 1e-12));
  if (correlation > 0.995) {
    throw new Error(`Warm-up output barely responds to input (r=${correlation.toFixed(4)}).`);
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
        model: modelSpec().id,
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
}

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const message = event.data;

  if (message.type === "init") {
    resolution = message.resolution;
    smoothing = message.smoothing;
    toneStrength = message.tone ?? 0;
    motionSensitivity = message.motion ?? 8;
    mobile = message.mobile;
    modelKey = message.model ?? "v2";
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
    if (typeof message.motion === "number") motionSensitivity = message.motion;
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
