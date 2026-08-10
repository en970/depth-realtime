import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/layout.css";
import "./styles/components.css";

import { Camera } from "./lib/camera";
import { buildLut, COLORMAPS, cssGradient, type ColormapId } from "./lib/colormaps";
import { DepthRenderer } from "./lib/depth-renderer";
import { PATCH, snapToPatch, type FromWorker, type ToWorker } from "./lib/protocol";

/* -------------------------------------------------------------------------- */
/* Elements                                                                    */
/* -------------------------------------------------------------------------- */

const $ = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing element #${id}`);
  return element as T;
};

const video = $<HTMLVideoElement>("camera");
const depthCanvas = $<HTMLCanvasElement>("depth");
const viewer = $<HTMLElement>("viewer");
const stage = document.querySelector<HTMLElement>(".stage")!;

const overlay = $<HTMLElement>("overlay");
const overlayKicker = $<HTMLElement>("overlay-kicker");
const overlayTitle = $<HTMLElement>("overlay-title");
const overlayText = $<HTMLElement>("overlay-text");
const overlayAction = $<HTMLButtonElement>("overlay-action");
const overlayProgress = $<HTMLElement>("overlay-progress");
const overlayProgressBar = $<HTMLElement>("overlay-progress-bar");

const statusBadge = $<HTMLElement>("badge-status");
const statusText = $<HTMLElement>("badge-status-text");
const statFps = $<HTMLElement>("stat-fps");
const statLatency = $<HTMLElement>("stat-latency");
const liveSummary = $<HTMLElement>("live-summary");

const metaCamera = $<HTMLElement>("meta-camera");
const metaDepth = $<HTMLElement>("meta-depth");
const legendScale = $<HTMLElement>("legend-scale");

const toggleStream = $<HTMLButtonElement>("toggle-stream");
const cameraSelect = $<HTMLSelectElement>("camera-select");
const mirrorToggle = $<HTMLInputElement>("mirror-toggle");
const colormapSelect = $<HTMLSelectElement>("colormap-select");
const resolutionRange = $<HTMLInputElement>("resolution-range");
const resolutionValue = $<HTMLElement>("resolution-value");
const smoothingRange = $<HTMLInputElement>("smoothing-range");
const smoothingValue = $<HTMLElement>("smoothing-value");
const adaptiveToggle = $<HTMLInputElement>("adaptive-toggle");
const qualityToggle = $<HTMLInputElement>("quality-toggle");
const guideRange = $<HTMLInputElement>("guide-range");
const guideValue = $<HTMLElement>("guide-value");
const structureRange = $<HTMLInputElement>("structure-range");
const structureValue = $<HTMLElement>("structure-value");
const compareInput = $<HTMLInputElement>("compare-input");
const infoBackend = $<HTMLElement>("info-backend");
const infoDtype = $<HTMLElement>("info-dtype");
const infoOutput = $<HTMLElement>("info-output");

/* -------------------------------------------------------------------------- */
/* State                                                                       */
/* -------------------------------------------------------------------------- */

type Mode = "split" | "stacked" | "compare";

interface Settings {
  mode: Mode;
  colormap: ColormapId;
  resolution: number;
  smoothing: number;
  mirror: boolean;
  adaptive: boolean;
  split: number;
  /** Strength of camera-guided edge-aware upsampling, 0 to 1. */
  guide: number;
  guideSigma: number;
  /** Relief shading strength: depth darkening plus curvature. */
  structure: number;
  quality: boolean;
}

const isMobile =
  matchMedia("(pointer: coarse)").matches && matchMedia("(max-width: 1024px)").matches;

const defaults: Settings = {
  mode: "split",
  colormap: "ice",
  resolution: isMobile ? 224 : 322,
  smoothing: 0.35,
  mirror: true,
  adaptive: true,
  split: 50,
  guide: 0.85,
  guideSigma: 0.06,
  structure: 0.6,
  quality: false,
};

const settings: Settings = { ...defaults, ...readSettings() };

function readSettings(): Partial<Settings> {
  const out: Partial<Settings> = {};
  const source = new URLSearchParams(location.hash.replace(/^#/, ""));
  const stored = (() => {
    try {
      return JSON.parse(localStorage.getItem("depth-realtime:settings") ?? "{}");
    } catch {
      return {};
    }
  })();

  const pick = (key: string): string | null => source.get(key) ?? (stored[key] ?? null);

  /** Number(null) is 0, which silently overrides defaults with a valid-looking
   *  zero. Absent keys must stay absent. */
  const pickNumber = (key: string, min: number, max: number): number | null => {
    const raw = pick(key);
    if (raw === null || raw === "") return null;
    const value = Number(raw);
    return Number.isFinite(value) && value >= min && value <= max ? value : null;
  };

  const mode = pick("mode");
  if (mode === "split" || mode === "stacked" || mode === "compare") out.mode = mode;

  const colormap = pick("cmap");
  if (colormap && colormap in COLORMAPS) out.colormap = colormap as ColormapId;

  const resolution = pickNumber("res", 182, 518);
  if (resolution !== null) out.resolution = snapToPatch(resolution);

  const smoothing = pickNumber("smooth", 0, 0.85);
  if (smoothing !== null) out.smoothing = smoothing;

  const split = pickNumber("split", 0, 100);
  if (split !== null) out.split = split;

  const guide = pickNumber("guide", 0, 1);
  if (guide !== null) out.guide = guide;

  const structure = pickNumber("structure", 0, 1);
  if (structure !== null) out.structure = structure;

  // Advanced, URL only: how different the camera pixel has to be before a depth
  // sample stops counting. Exposed for tuning runs rather than for the panel.
  const sigma = pickNumber("sigma", 0.005, 0.5);
  if (sigma !== null) out.guideSigma = sigma;

  const mirror = pick("mirror");
  if (mirror !== null) out.mirror = mirror === "1" || mirror === "true";

  const adaptive = pick("adaptive");
  if (adaptive !== null) out.adaptive = adaptive === "1" || adaptive === "true";

  const quality = pick("quality");
  if (quality !== null) out.quality = quality === "1" || quality === "true";

  return out;
}

let persistTimer = 0;
function persistSettings(): void {
  clearTimeout(persistTimer);
  persistTimer = window.setTimeout(() => {
    const params = new URLSearchParams({
      mode: settings.mode,
      cmap: settings.colormap,
      res: String(settings.resolution),
      smooth: settings.smoothing.toFixed(2),
      split: settings.split.toFixed(0),
      guide: settings.guide.toFixed(2),
      structure: settings.structure.toFixed(2),
      mirror: settings.mirror ? "1" : "0",
      adaptive: settings.adaptive ? "1" : "0",
      quality: settings.quality ? "1" : "0",
    });
    history.replaceState(null, "", `#${params.toString()}`);
    try {
      localStorage.setItem(
        "depth-realtime:settings",
        JSON.stringify(Object.fromEntries(params)),
      );
    } catch {
      /* private browsing */
    }
  }, 250);
}

/* -------------------------------------------------------------------------- */
/* Rendering + inference plumbing                                              */
/* -------------------------------------------------------------------------- */

const renderer = new DepthRenderer(depthCanvas);
renderer.setColormap(buildLut(settings.colormap));
renderer.setGuide(video);

let worker = spawnWorker();
let restarts = 0;

/**
 * The worker is replaceable, not a constant.
 *
 * Transformers.js serialises web inference through a module-scoped promise
 * chain that has no rejection handler, so a genuine session.run() rejection
 * leaves the chain permanently rejected: every subsequent call, including one
 * on a freshly created session, fails immediately with the original error. That
 * state cannot be cleared from inside the realm, so recovery means discarding
 * the worker and starting a new one.
 */
function spawnWorker(): Worker {
  const instance = new Worker(new URL("./lib/depth-worker.ts", import.meta.url), {
    type: "module",
  });
  instance.onmessage = handleWorkerMessage;
  instance.onerror = handleWorkerError;
  instance.onmessageerror = () => {
    inFlight = false;
  };
  return instance;
}

function restartWorker(poisoned?: "webgpu" | "wasm"): boolean {
  if (restarts >= 1) return false;
  restarts++;
  worker.terminate();
  modelReady = false;
  inFlight = false;
  recovering = true;
  worker = spawnWorker();
  send({
    type: "init",
    resolution: settings.resolution,
    smoothing: settings.smoothing,
    mobile: isMobile,
    // A new worker would otherwise re-derive the same candidate list and walk
    // straight back into the backend that just failed.
    force: poisoned === "webgpu" ? "wasm" : undefined,
  });
  return true;
}

const capture = document.createElement("canvas");
const captureCtx = capture.getContext("2d", { willReadFrequently: true })!;

const camera = new Camera(video, {
  onInterrupted: () => setStatus("warn", "Interrupted"),
  onResumed: () => setStatus("live", "Live"),
  onEnded: () => {
    running = false;
    setStatus("error", "Disconnected");
    showOverlay({
      kicker: "Camera",
      title: "The camera stream ended",
      text: "The device was disconnected or claimed by another application.",
      action: "Reconnect",
      tone: "error",
    });
  },
});

let modelReady = false;
let running = false;
let paused = false;
/** Set while the worker is rebuilding a session. Frames captured during that
 *  window would be dropped by the worker without a reply, and the in-flight
 *  flag would stay raised for ever. */
let recovering = false;
let frameId = 0;
let inFlight = false;
let lastResultAt = 0;
let interval = 100;
let latencyEma = 0;
let preprocessEma = 0;
let normLo = 0;
let normHi = 1;
let rawLo = 0;
let rawHi = 1;
let lastCaptureChecksum = 0;
let fpsEma = 0;
let lastAdaptation = 0;

// Stops at 518, the resolution Depth Anything V2 was trained at. Measured at
// 644 the output is visibly *worse* — facial structure flattens out — because
// the input leaves the distribution the model saw. Going higher is not a
// quality knob, it is a cliff.
const LADDER = [182, 196, 224, 252, 280, 322, 350, 392, 434, 476, 518];

/** Must match --space-4 and --space-3 in the stylesheet. */
const GUTTER = 16;
const GAP = 12;

/* -------------------------------------------------------------------------- */
/* Worker messages                                                             */
/* -------------------------------------------------------------------------- */

const send = (message: ToWorker, transfer?: Transferable[]) =>
  worker.postMessage(message, transfer ?? []);

// Without this the worker can die silently and the page just stops updating.
function handleWorkerError(event: ErrorEvent): void {
  inFlight = false;
  running = false;
  setStatus("error", "Failed");
  showOverlay({
    kicker: "Inference",
    title: "The inference worker stopped",
    text: event.message || "The worker terminated unexpectedly.",
    action: "Reload",
    tone: "error",
    onAction: () => location.reload(),
  });
}

/** Diagnostic hook. Call `depthDiagnostics()` from the browser console to see
 *  why the pipeline is idle. Documented in the README. */
(globalThis as unknown as { depthDiagnostics: () => unknown }).depthDiagnostics = () => ({
  running,
  paused,
  inFlight,
  modelReady,
  frameId,
  latencyEma,
  fpsEma,
  preprocessEma,
  normLo,
  normHi,
  rawLo,
  rawHi,
  captureChecksum: lastCaptureChecksum,
  videoReady: video.readyState,
  videoSize: [video.videoWidth, video.videoHeight],
});

function handleWorkerMessage(event: MessageEvent<FromWorker>): void {
  const message = event.data;

  switch (message.type) {
    case "status": {
      if (message.stage === "downloading") {
        overlayProgress.hidden = false;
        overlayProgressBar.style.width = `${((message.progress ?? 0) * 100).toFixed(1)}%`;
        showOverlay({
          kicker: "Model",
          title: "Downloading network weights",
          text:
            message.detail !== undefined
              ? `${message.detail} — cached locally, so later visits start immediately.`
              : "Cached locally, so later visits start immediately.",
          action: null,
        });
      } else if (message.stage === "compiling") {
        showOverlay({
          kicker: "Model",
          title: "Preparing the inference session",
          text: "Compiling shaders for this device.",
          action: null,
        });
      }
      break;
    }

    case "ready": {
      modelReady = true;
      recovering = false;
      // A recovery leaves a frame permanently in flight: the worker dropped it
      // while it had no session, so no result or error ever came back.
      inFlight = false;
      overlayProgress.hidden = true;
      infoBackend.textContent = message.backend === "webgpu" ? "WebGPU" : "WASM (1 thread)";
      infoDtype.textContent = message.dtype;
      if (message.backend === "wasm") {
        setStatus("warn", "Compatibility");
      }
      if (camera.active) {
        hideOverlay();
        if (running) {
          setStatus("live", "Live");
          scheduleCapture();
        } else {
          startLoop();
        }
      }
      break;
    }

    case "result": {
      inFlight = false;
      const now = performance.now();
      if (lastResultAt > 0) {
        const delta = now - lastResultAt;
        interval = delta;
        fpsEma = fpsEma === 0 ? 1000 / delta : fpsEma * 0.8 + (1000 / delta) * 0.2;
      }
      lastResultAt = now;
      latencyEma = latencyEma === 0 ? message.ms : latencyEma * 0.8 + message.ms * 0.2;
      preprocessEma =
        preprocessEma === 0
          ? message.preprocessMs
          : preprocessEma * 0.8 + message.preprocessMs * 0.2;

      renderer.push(new Uint8Array(message.buffer), message.width, message.height, interval);
      normLo = message.lo;
      normHi = message.hi;
      rawLo = message.rawLo;
      rawHi = message.rawHi;
      metaDepth.textContent = `${message.width}x${message.height}`;
      infoOutput.textContent = `${message.width}x${message.height}`;
      adapt(now);

      // Start the next inference immediately rather than waiting for the next
      // camera callback. At 63 ms per inference and a 30 fps camera that wait
      // added ~16 ms of dead time to every cycle, costing roughly a third of
      // the achievable frame rate.
      captureFrame();
      break;
    }

    case "error": {
      inFlight = false;
      if (message.fatal) {
        if (restartWorker(message.poisoned)) {
          setStatus("warn", "Restarting");
          showOverlay({
            kicker: "Inference",
            title: "Restarting the inference worker",
            text: "The previous session could not run on this device; retrying on a different backend.",
            action: null,
          });
          break;
        }
        running = false;
        setStatus("error", "Failed");
        showOverlay({
          kicker: "Inference",
          title: "The model could not run on this device",
          text: message.message,
          action: "Reload",
          tone: "error",
          onAction: () => location.reload(),
        });
      } else {
        recovering = true;
        setStatus("warn", "Recovering");
        console.warn(`[depth] ${message.message}`);
      }
      break;
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Frame loop                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Two independent clocks. Capture is driven by requestVideoFrameCallback so it
 * fires once per camera frame rather than once per display refresh — on a 30 fps
 * camera and a 60 Hz screen that halves the work. Presentation runs on
 * requestAnimationFrame and interpolates between the last two depth fields, so
 * the depth pane stays smooth even when inference is much slower than capture.
 */
function startLoop(): void {
  if (running) return;
  running = true;
  setStatus("live", "Live");
  scheduleCapture();
  requestAnimationFrame(present);
}

function scheduleCapture(): void {
  if (!running) return;
  const useVideoCallback = typeof video.requestVideoFrameCallback === "function";
  if (useVideoCallback) {
    video.requestVideoFrameCallback(() => {
      captureFrame();
      scheduleCapture();
    });
  } else {
    requestAnimationFrame(() => {
      captureFrame();
      scheduleCapture();
    });
  }
}

function captureFrame(): void {
  if (!running || paused || inFlight || recovering || !modelReady) return;
  if (video.readyState < 2 || !video.videoWidth) return;

  // The worker feeds these dimensions straight to the network, so both edges
  // must be multiples of the ViT patch size. Rounding here also keeps the
  // camera's aspect ratio, which is what makes the two panes frame the same
  // region.
  const scale = settings.resolution / Math.max(video.videoWidth, video.videoHeight);
  const width = Math.max(PATCH, Math.round((video.videoWidth * scale) / PATCH) * PATCH);
  const height = Math.max(PATCH, Math.round((video.videoHeight * scale) / PATCH) * PATCH);

  if (capture.width !== width || capture.height !== height) {
    capture.width = width;
    capture.height = height;
  }
  captureCtx.drawImage(video, 0, 0, width, height);
  const image = captureCtx.getImageData(0, 0, width, height);

  // Diagnostic only: a cheap fingerprint of the captured frame, so a drifting
  // model output can be told apart from a drifting input.
  let checksum = 0;
  for (let i = 0; i < image.data.length; i += 997) checksum += image.data[i];
  lastCaptureChecksum = checksum;

  inFlight = true;
  frameId++;
  send(
    { type: "frame", id: frameId, buffer: image.data.buffer, width, height },
    [image.data.buffer],
  );
}

function present(now: number): void {
  if (!running) return;
  renderer.render(now);
  requestAnimationFrame(present);
}

/**
 * Adaptive quality. Sustained GPU load throttles phones hard — published
 * measurements show ~40% throughput loss within a minute — so a fixed
 * resolution either wastes headroom on desktops or collapses on mobile. The
 * hysteresis (drop after 2.5 s, raise after 6 s) keeps it from oscillating.
 */
function adapt(now: number): void {
  if (!settings.adaptive) return;
  // Budgets are per inference. They are set below the frame interval on
  // purpose: a depth view that updates 20 times a second reads as continuous,
  // whereas 10 updates a second reads as stuttering however sharp each one is.
  const budget = isMobile ? 80 : 45;

  // Nearest rung rather than exact match: the slider steps by 14 and reaches 25
  // values, the ladder holds 11 of them. indexOf returned -1 for the rest, which
  // silently disabled adaptation as soon as anyone touched the slider.
  const index = LADDER.reduce(
    (best, value, i) =>
      Math.abs(value - settings.resolution) < Math.abs(LADDER[best] - settings.resolution)
        ? i
        : best,
    0,
  );

  if (latencyEma > budget * 1.3 && now - lastAdaptation > 2500) {
    if (index > 0) {
      lastAdaptation = now;
      applyResolution(LADDER[index - 1], false);
    }
  } else if (latencyEma < budget * 0.55 && now - lastAdaptation > 6000) {
    if (index < LADDER.length - 1) {
      lastAdaptation = now;
      applyResolution(LADDER[index + 1], false);
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Overlay + status                                                            */
/* -------------------------------------------------------------------------- */

let overlayAction_: (() => void) | null = null;

function showOverlay(options: {
  kicker: string;
  title: string;
  text: string;
  action: string | null;
  tone?: "neutral" | "error";
  onAction?: () => void;
}): void {
  overlay.hidden = false;
  overlay.dataset.tone = options.tone ?? "neutral";
  overlayKicker.textContent = options.kicker;
  overlayTitle.textContent = options.title;
  overlayText.textContent = options.text;
  overlayAction.hidden = options.action === null;
  if (options.action) overlayAction.textContent = options.action;
  overlayAction_ = options.onAction ?? overlayAction_;
}

function hideOverlay(): void {
  overlay.hidden = true;
  overlayProgress.hidden = true;
}

function setStatus(state: "idle" | "live" | "warn" | "error", label: string): void {
  statusBadge.dataset.state = state;
  statusText.textContent = label;
}

/* -------------------------------------------------------------------------- */
/* Controls                                                                    */
/* -------------------------------------------------------------------------- */

function applyMode(mode: Mode): void {
  settings.mode = mode;
  viewer.dataset.mode = mode;
  fitViewer();
  document.querySelectorAll<HTMLButtonElement>(".segmented__item[data-mode]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  });
  persistSettings();
}

function applyColormap(id: ColormapId): void {
  settings.colormap = id;
  renderer.setColormap(buildLut(id));
  legendScale.style.backgroundImage = cssGradient(id);
  colormapSelect.value = id;
  persistSettings();
}

function applyResolution(value: number, fromUser: boolean): void {
  const snapped = snapToPatch(value);
  settings.resolution = snapped;
  resolutionRange.value = String(snapped);
  resolutionValue.textContent = `${snapped} px`;
  setRangeFill(resolutionRange);
  send({ type: "config", resolution: snapped });
  if (fromUser) {
    lastAdaptation = performance.now();
    persistSettings();
  }
}

function applySmoothing(value: number): void {
  settings.smoothing = value;
  smoothingRange.value = String(value);
  smoothingValue.textContent = value.toFixed(2);
  setRangeFill(smoothingRange);
  send({ type: "config", smoothing: value });
  persistSettings();
}

function applyGuide(value: number): void {
  settings.guide = value;
  guideRange.value = String(value);
  guideValue.textContent = value.toFixed(2);
  setRangeFill(guideRange);
  renderer.setGuideStrength(value, settings.guideSigma);
  persistSettings();
}

/**
 * Quality is a preset, not a separate pipeline: it pins the input to the
 * resolution the network was trained at, turns the guided upsampling fully on
 * and stops the adaptive controller from trading that away. It costs roughly
 * half the frame rate, which is why it is opt-in.
 */
function applyQuality(enabled: boolean): void {
  settings.quality = enabled;
  qualityToggle.checked = enabled;
  document.body.classList.toggle("is-quality", enabled);

  settings.adaptive = !enabled;
  adaptiveToggle.checked = settings.adaptive;
  adaptiveToggle.disabled = enabled;

  applyResolution(enabled ? 518 : isMobile ? 224 : 322, true);
  applyGuide(enabled ? 1 : 0.85);
  applyStructure(enabled ? 0.8 : 0.6);
  persistSettings();
}

function applyStructure(value: number): void {
  settings.structure = value;
  structureRange.value = String(value);
  structureValue.textContent = value.toFixed(2);
  setRangeFill(structureRange);
  renderer.setStructure(value);
  persistSettings();
}

function applyMirror(mirror: boolean): void {
  settings.mirror = mirror;
  mirrorToggle.checked = mirror;
  video.classList.toggle("pane__media--mirrored", mirror);
  depthCanvas.classList.toggle("pane__media--mirrored", mirror);
  persistSettings();
}

function applySplit(value: number): void {
  settings.split = value;
  viewer.style.setProperty("--split", `${value}%`);
  compareInput.value = String(value);
  compareInput.setAttribute("aria-valuetext", `${Math.round(value)}% depth`);
  persistSettings();
}

/**
 * Sizes the viewer so every pane has exactly the camera's aspect ratio.
 *
 * Doing this in script rather than CSS because the constraint is
 * two-dimensional: the panes are limited by the stage width in one layout and
 * by its height in another, and `aspect-ratio` with `max-width`/`max-height`
 * does not preserve the ratio once the second limit binds.
 */
function fitViewer(): void {
  const source = camera.resolution;
  const aspect = source ? source.width / source.height : 16 / 9;
  const availableWidth = stage.clientWidth - GUTTER * 2;
  const availableHeight = stage.clientHeight - GUTTER * 2;
  if (availableWidth <= 0 || availableHeight <= 0) return;

  // Narrow viewports always stack, whatever the chosen layout, because two
  // panes side by side would each be too small to read.
  const vertical = settings.mode === "stacked" || window.innerWidth <= 767;
  const columns = settings.mode === "compare" ? 1 : vertical ? 1 : 2;
  const rows = settings.mode === "compare" ? 1 : vertical ? 2 : 1;

  // Try filling the width first, then fall back to filling the height.
  let paneWidth = (availableWidth - GAP * (columns - 1)) / columns;
  let paneHeight = paneWidth / aspect;
  if (paneHeight * rows + GAP * (rows - 1) > availableHeight) {
    paneHeight = (availableHeight - GAP * (rows - 1)) / rows;
    paneWidth = paneHeight * aspect;
  }

  viewer.style.width = `${Math.floor(paneWidth * columns + GAP * (columns - 1))}px`;
  viewer.style.height = `${Math.floor(paneHeight * rows + GAP * (rows - 1))}px`;
}

/** Fills the track to the left of the thumb; WebKit has no ::-moz-range-progress. */
function setRangeFill(input: HTMLInputElement): void {
  const min = Number(input.min);
  const max = Number(input.max);
  const ratio = ((Number(input.value) - min) / (max - min)) * 100;
  input.style.setProperty("--fill", `${ratio}%`);
}

async function refreshDevices(): Promise<void> {
  const devices = await camera.devices();
  cameraSelect.innerHTML = "";
  if (devices.length === 0) {
    cameraSelect.disabled = true;
    cameraSelect.append(new Option("No camera found", ""));
    return;
  }
  for (const device of devices) {
    const option = new Option(device.label, device.id);
    option.selected = device.id === camera.currentDeviceId;
    cameraSelect.append(option);
  }
  cameraSelect.disabled = devices.length < 2;
}

async function startCamera(deviceId?: string): Promise<void> {
  overlayAction.disabled = true;
  const previousLabel = overlayAction.textContent;
  overlayAction.textContent = "Waiting for permission";

  const failure = await camera.start(deviceId);

  overlayAction.disabled = false;
  overlayAction.textContent = previousLabel;

  if (failure) {
    setStatus("error", "Blocked");
    showOverlay({
      kicker: "Camera",
      title: failure.title,
      text: failure.detail,
      action: "Try again",
      tone: "error",
      onAction: () => startCamera(deviceId),
    });
    return;
  }

  await refreshDevices();
  const resolution = camera.resolution;
  if (resolution) metaCamera.textContent = `${resolution.width}x${resolution.height}`;
  fitViewer();
  toggleStream.textContent = "Stop";

  if (!modelReady) {
    showOverlay({
      kicker: "Model",
      title: "Loading the depth network",
      text: "Weights are fetched once and cached in the browser.",
      action: null,
    });
    return;
  }
  hideOverlay();
  startLoop();
}

function stopCamera(): void {
  running = false;
  camera.stop();
  toggleStream.textContent = "Start";
  setStatus("idle", "Idle");
  showOverlay({
    kicker: "Camera",
    title: "Camera stopped",
    text: "The stream is released. Nothing was recorded or transmitted.",
    action: "Start camera",
    onAction: () => startCamera(cameraSelect.value || undefined),
  });
}

/* -------------------------------------------------------------------------- */
/* Wiring                                                                      */
/* -------------------------------------------------------------------------- */

overlayAction.addEventListener("click", () => {
  if (overlayAction_) {
    overlayAction_();
    return;
  }
  void begin();
});

async function begin(): Promise<void> {
  // Weight download and permission prompt run concurrently: both are slow and
  // neither depends on the other.
  if (!modelReady) {
    const forced = new URLSearchParams(location.hash.replace(/^#/, "")).get("backend");
    send({
      type: "init",
      resolution: settings.resolution,
      smoothing: settings.smoothing,
      mobile: isMobile,
      force: forced === "wasm" || forced === "webgpu" ? forced : undefined,
    });
  }
  await startCamera(cameraSelect.value || undefined);
}

toggleStream.addEventListener("click", () => {
  if (camera.active) stopCamera();
  else void begin();
});

cameraSelect.addEventListener("change", () => {
  if (camera.active) void startCamera(cameraSelect.value);
});

mirrorToggle.addEventListener("change", () => applyMirror(mirrorToggle.checked));

colormapSelect.addEventListener("change", () =>
  applyColormap(colormapSelect.value as ColormapId),
);

document.querySelectorAll<HTMLButtonElement>(".segmented__item[data-mode]").forEach((button) => {
  button.addEventListener("click", () => applyMode(button.dataset.mode as Mode));
});

resolutionRange.addEventListener("input", () => {
  applyResolution(Number(resolutionRange.value), true);
});

smoothingRange.addEventListener("input", () => {
  applySmoothing(Number(smoothingRange.value));
});

guideRange.addEventListener("input", () => applyGuide(Number(guideRange.value)));

structureRange.addEventListener("input", () =>
  applyStructure(Number(structureRange.value)),
);

qualityToggle.addEventListener("change", () => applyQuality(qualityToggle.checked));

adaptiveToggle.addEventListener("change", () => {
  settings.adaptive = adaptiveToggle.checked;
  persistSettings();
});

compareInput.addEventListener("input", () => applySplit(Number(compareInput.value)));

document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
    return;
  }
  switch (event.key) {
    case " ":
      event.preventDefault();
      paused = !paused;
      setStatus(paused ? "warn" : "live", paused ? "Paused" : "Live");
      break;
    case "c":
    case "C": {
      const order: Mode[] = ["split", "stacked", "compare"];
      applyMode(order[(order.indexOf(settings.mode) + 1) % order.length]);
      break;
    }
    case "m":
    case "M": {
      const ids = Object.keys(COLORMAPS) as ColormapId[];
      applyColormap(ids[(ids.indexOf(settings.colormap) + 1) % ids.length]);
      break;
    }
    case "ArrowLeft":
      if (settings.mode === "compare") applySplit(Math.max(0, settings.split - 2));
      break;
    case "ArrowRight":
      if (settings.mode === "compare") applySplit(Math.min(100, settings.split + 2));
      break;
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    paused = true;
  } else {
    paused = false;
    void camera.resume();
  }
});

window.addEventListener("pagehide", () => camera.stop());

new ResizeObserver(() => fitViewer()).observe(stage);
// The camera can renegotiate its resolution mid-stream, most often when a phone
// is rotated, so neither the fit nor the readout can be computed once at
// start-up.
video.addEventListener("resize", () => {
  fitViewer();
  const resolution = camera.resolution;
  if (resolution) metaCamera.textContent = `${resolution.width}x${resolution.height}`;
});

/* Telemetry, updated on a slow clock so screen readers are not flooded. */
setInterval(() => {
  statFps.textContent = fpsEma > 0 ? fpsEma.toFixed(1) : "—";
  statLatency.textContent = latencyEma > 0 ? latencyEma.toFixed(0) : "—";
}, 500);

setInterval(() => {
  if (!running) return;
  liveSummary.textContent = `${fpsEma.toFixed(0)} frames per second, ${latencyEma.toFixed(0)} milliseconds per inference, input ${settings.resolution} pixels.`;
}, 5000);

/* -------------------------------------------------------------------------- */
/* Boot                                                                        */
/* -------------------------------------------------------------------------- */

applyMode(settings.mode);
fitViewer();
applyColormap(settings.colormap);
applyMirror(settings.mirror);
applySplit(settings.split);
applyGuide(settings.guide);
applyStructure(settings.structure);

// Reflect a stored quality flag without re-running the preset: the values it
// would set are already restored, and re-applying them would overwrite anything
// given explicitly in the URL.
qualityToggle.checked = settings.quality;
adaptiveToggle.disabled = settings.quality;
document.body.classList.toggle("is-quality", settings.quality);
adaptiveToggle.checked = settings.adaptive;
resolutionRange.value = String(settings.resolution);
resolutionValue.textContent = `${settings.resolution} px`;
smoothingRange.value = String(settings.smoothing);
smoothingValue.textContent = settings.smoothing.toFixed(2);
setRangeFill(resolutionRange);
setRangeFill(smoothingRange);

if (Camera.insecureContext()) {
  showOverlay({
    kicker: "Camera",
    title: "A secure context is required",
    text: "Camera access needs HTTPS or localhost.",
    action: null,
    tone: "error",
  });
} else if (!Camera.supported()) {
  showOverlay({
    kicker: "Camera",
    title: "This browser cannot open a camera",
    text: "Open the page in Safari, Chrome or Edge rather than an in-app browser.",
    action: null,
    tone: "error",
  });
}
