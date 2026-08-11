/** Message contract between the page and the inference worker. */

export type Backend = "webgpu" | "wasm";
export type Dtype = "fp32" | "fp16" | "q8" | "q4f16";

export interface InitMessage {
  type: "init";
  /** Longest edge of the network input, in pixels. Multiples of 14 only. */
  resolution: number;
  /** Mobile devices default to smaller weights: 19 MB instead of 50 MB. */
  mobile: boolean;
  smoothing: number;
  /** Adaptive tone curve strength, 0 to 1. */
  tone?: number;
  /** Diagnostic override, set through the `backend` URL parameter. */
  force?: "webgpu" | "wasm";
  /** Diagnostic override, set through the `dtype` URL parameter. */
  forceDtype?: Dtype;
  /** Which network to run, set through the `model` URL parameter. */
  model?: "v2" | "v3";
}

export interface FrameMessage {
  type: "frame";
  id: number;
  buffer: ArrayBuffer;
  width: number;
  height: number;
}

export interface ConfigMessage {
  type: "config";
  resolution?: number;
  smoothing?: number;
  tone?: number;
}

export type ToWorker = InitMessage | FrameMessage | ConfigMessage;

export interface StatusMessage {
  type: "status";
  stage: "starting" | "downloading" | "compiling" | "ready";
  /** 0-1 while weights download, undefined otherwise. */
  progress?: number;
  detail?: string;
}

export interface ReadyMessage {
  type: "ready";
  backend: Backend;
  dtype: Dtype;
  model: string;
}

export interface ResultMessage {
  type: "result";
  id: number;
  /** Normalised depth quantised to 8 bits; 255 is nearest. */
  buffer: ArrayBuffer;
  width: number;
  height: number;
  /** Wall-clock time of the inference call, in milliseconds. */
  ms: number;
  /** Preprocessing time (resize, normalise, layout) inside that total. */
  preprocessMs: number;
  /** Smoothed normalisation bounds actually used for this frame. */
  lo: number;
  hi: number;
  /** The same bounds before smoothing, for stability diagnosis. */
  rawLo: number;
  rawHi: number;
}

export interface ErrorMessage {
  type: "error";
  message: string;
  fatal: boolean;
  /** Backend that must not be retried if the worker is restarted. */
  poisoned?: Backend;
}

export type FromWorker = StatusMessage | ReadyMessage | ResultMessage | ErrorMessage;

/** The network input edge must be a multiple of the ViT patch size. */
export const PATCH = 14;

export function snapToPatch(value: number): number {
  return Math.max(PATCH * 4, Math.round(value / PATCH) * PATCH);
}
