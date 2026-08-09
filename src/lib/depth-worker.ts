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

import {
  AutoModelForDepthEstimation,
  AutoProcessor,
  RawImage,
} from "@huggingface/transformers";
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

interface Session {
  model: any;
  processor: any;
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
async function candidates(force?: "webgpu" | "wasm"): Promise<Candidate[]> {
  const wasm: Candidate = { backend: "wasm", dtype: "q8" };
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

  const model = await AutoModelForDepthEstimation.from_pretrained(MODEL_ID, {
    device: backend,
    dtype,
    progress_callback,
  } as any);

  const processor: any = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback,
  } as any);

  applyResolution(processor, resolution);
  return { model, processor, backend, dtype };
}

/**
 * The image processor keeps the aspect ratio and rounds each edge to a multiple
 * of 14, so setting a square target yields a rectangle with the camera's aspect
 * ratio whose longest edge is `size`.
 */
function applyResolution(processor: any, size: number): void {
  const target = processor?.feature_extractor ?? processor?.image_processor ?? processor;
  if (!target) return;
  target.size = { width: size, height: size };
  if ("keep_aspect_ratio" in target) target.keep_aspect_ratio = true;
  if ("ensure_multiple_of" in target) target.ensure_multiple_of = 14;
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

/** Normalises into [0, 1] and applies the per-pixel exponential average. The
 *  smoothing state stays in float; only the transported copy is quantised. */
function normalise(data: Float32Array): Float32Array {
  const [lo, hi] = computeBounds(data);

  if (emaLo === null || emaHi === null) {
    emaLo = lo;
    emaHi = hi;
  } else {
    emaLo += (lo < emaLo ? EMA_EXPAND : EMA_CONTRACT) * (lo - emaLo);
    emaHi += (hi > emaHi ? EMA_EXPAND : EMA_CONTRACT) * (hi - emaHi);
  }
  const span = Math.max(emaHi - emaLo, 1e-6);

  if (!smoothed || smoothed.length !== data.length) {
    smoothed = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      smoothed[i] = clamp01((data[i] - emaLo) / span);
    }
    return smoothed;
  }

  const a = smoothing;
  if (a <= 0.001) {
    for (let i = 0; i < data.length; i++) {
      smoothed[i] = clamp01((data[i] - emaLo) / span);
    }
  } else {
    const b = 1 - a;
    for (let i = 0; i < data.length; i++) {
      const t = clamp01((data[i] - emaLo) / span);
      smoothed[i] = smoothed[i] * a + t * b;
    }
  }
  return smoothed;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
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
    const image = new RawImage(new Uint8ClampedArray(buffer), width, height, 4);
    const preprocessStart = performance.now();
    const inputs = await session.processor(image);
    const preprocessMs = performance.now() - preprocessStart;
    const output: any = await withTimeout(
      session.model(inputs),
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
  const size = 98; // 7 x 14
  const pixels = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      pixels[i] = (x / size) * 255;
      pixels[i + 1] = (y / size) * 255;
      pixels[i + 2] = 128;
      pixels[i + 3] = 255;
    }
  }

  const image = new RawImage(pixels, size, size, 4);
  const budget = TIMEOUTS[candidate.backend].warmup;
  const inputs = await withTimeout(candidate.processor(image), budget, "preprocess");
  const output: any = await withTimeout(candidate.model(inputs), budget, "warmup");
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
    mobile = message.mobile;
    post({ type: "status", stage: "starting" });
    queue = await candidates(message.force);
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
    if (typeof message.resolution === "number" && message.resolution !== resolution) {
      resolution = message.resolution;
      if (session) applyResolution(session.processor, resolution);
      resetTemporalState();
    }
    return;
  }

  if (message.type === "frame") {
    await handleFrame(message.id, message.buffer, message.width, message.height);
  }
};
