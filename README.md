# Depth / Realtime

Real-time monocular depth estimation from a live camera stream, running entirely
in the browser. The raw camera image is shown next to a depth map in which
distant regions are dark blue and near regions approach white.

**Live site:** https://en970.github.io/depth-realtime/

There is no server component. Frames are captured, analysed and rendered on the
device; no image data is uploaded at any point.

---

## What it does

A single monocular frame carries no direct distance information. A depth network
infers it from monocular cues — perspective, occlusion, texture gradient,
relative size — that the model learned from a large corpus of images with known
geometry. This project runs such a network on every camera frame and renders the
result as a continuous colour field.

The estimate is **relative and affine-invariant**, not metric. Within one frame
the ordering of depths is meaningful; across frames the absolute scale is not.
The same shade of white in two different frames does not imply the same
distance. The interface states this explicitly, because a smooth colour ramp
invites a numerical reading that the model does not support.

## Method

```
  camera ──► <video>                            (playback, always visible)
                │
                │ requestVideoFrameCallback      one callback per camera frame
                ▼
           2D canvas ──► ImageData ──────────►  Web Worker
                                                    │
                                    image processor │ resize to a multiple of 14,
                                                    │ aspect ratio preserved,
                                                    │ ImageNet normalisation
                                                    ▼
                                    Depth Anything V2 Small (ONNX)
                                       WebGPU, or WASM where unavailable
                                                    │
                                                    │ inverse depth, float32
                                                    ▼
                              p2/p98 histogram bounds ──► asymmetric EMA
                                                    │
                                       per-pixel exponential average
                                                    │
                                    normalised [0,1] field (transferred)
                                                    ▼
  depth pane ◄── WebGL2 fragment shader ◄──── R16F texture + 256×1 colour LUT
```

Design decisions worth stating explicitly:

**No inversion is applied.** Depth Anything V2 emits affine-invariant *inverse*
depth, so larger values are already nearer. Applying `1 - t` — the reflex when
coming from datasets such as KITTI, where smaller values are nearer — would
invert the entire visualisation. Note that Depth Anything 3 reverses this
convention again, and the V2 metric checkpoints return metres; switching
checkpoints therefore requires revisiting this decision.

**Percentile normalisation, not min/max.** The reference implementations
normalise each frame between its own minimum and maximum. A single specular
highlight then compresses the rest of the scene into a fraction of the output
range, and because that outlier moves between frames the whole image flickers.
Clipping at the 2nd and 98th percentiles and smoothing those bounds with an
asymmetric exponential average — expanding quickly so nothing clips, contracting
slowly so the image does not pump — removes most of the flicker at a cost of
about 0.4 ms per frame.

**Colouring on the GPU, statistics on the CPU.** Computing the histogram over
the small network output is cheap. Colouring at display resolution on the CPU is
not: roughly 10 ms per 1080p frame, several times that on a phone. The depth
field is therefore uploaded as a small single-channel `R16F` texture and
magnified by the sampler, with the colour ramp applied in a fragment shader.

**Two clocks.** Capture is driven by `requestVideoFrameCallback`, which fires
once per camera frame rather than once per display refresh; on a 30 fps camera
and a 60 Hz screen this halves the work. Presentation runs on
`requestAnimationFrame` and interpolates between the last two depth fields, so
the depth pane remains smooth even when inference is several times slower than
capture.

**WebGL2 for display, WebGPU only for inference.** WebGPU reached Safari with
version 26; making the visualisation depend on it would exclude every device
still on iOS 18–25, as well as Firefox on Linux. WebGL2 is available on
essentially all of them.

## Colour scales

All three scales are monotonic in lightness. A non-monotonic ramp — jet, turbo,
Spectral — introduces perceived edges that are absent from the data and destroys
the ordering for viewers with colour vision deficiency.

| Scale | Source | Notes |
| --- | --- | --- |
| Ice (default) | cmocean `ice`, MIT | L\* 1.8 → 98.0, monotonic; dark navy to near-white |
| Greyscale | — | Neutral reference for measurement or publication |
| Magma | matplotlib, CC0 | Monotonic, higher chroma |

## Model

| | |
| --- | --- |
| Checkpoint | [`onnx-community/depth-anything-v2-small`](https://huggingface.co/onnx-community/depth-anything-v2-small) |
| Parameters | 24.8 M |
| Licence | Apache-2.0 |
| Output | Affine-invariant inverse relative depth |
| Weights | `fp16` (49.6 MB) on desktop WebGPU, `q4f16` (19.1 MB) on mobile, `q8` (27.3 MB) on WASM |

Within the Depth Anything V2 family only the Small checkpoint is permissively
licensed. Base, Large and Giant are CC-BY-NC-4.0 and cannot be used in an
MIT-licensed project. The quality gap is modest — NYU-D δ1 of 0.973 for Small
against 0.979 for Large — so the licence constraint costs little here.

Weights are fetched from the Hugging Face CDN on first use and cached by the
browser, so subsequent visits start without a download. They are deliberately
not committed to this repository: Git LFS pointers are served verbatim by GitHub
Pages, which would break the site in a way that is difficult to diagnose.

## Browser support

| Environment | Inference | Notes |
| --- | --- | --- |
| Chrome / Edge 113+, desktop | WebGPU | Primary target |
| Safari 26+, macOS and iOS | WebGPU | WebGPU enabled by default from version 26 |
| Firefox 141+ (Windows), 147+ (macOS) | WebGPU | Linux still falls back |
| iOS 18–25, Firefox/Linux, older Android | WASM, single thread | Runs at a few frames per second; the interface says so |

Multi-threaded WASM is unavailable because GitHub Pages cannot set the
`COOP`/`COEP` headers that `SharedArrayBuffer` requires. The `coi-serviceworker`
workaround is deliberately not used: it forces a reload on first visit and its
`require-corp` policy blocks the cross-origin weight download.

## Performance

Adaptive quality is on by default. The controller tracks an exponential average
of inference time and moves along a resolution ladder (182 … 518 px, multiples
of 14) when it leaves the budget, with hysteresis to prevent oscillation. This
is not premature optimisation: sustained GPU load throttles phones severely, and
published measurements show roughly 40 % throughput loss within a minute of
continuous inference. A fixed resolution would either waste desktop headroom or
collapse on mobile.

Input resolution, temporal smoothing and the colour scale can be set manually;
the state is written to the URL fragment, so a particular configuration can be
shared or cited.

## Local development

```bash
npm install
npm run dev        # http://127.0.0.1:5173
```

`getUserMedia` requires a secure context. `127.0.0.1` qualifies; a LAN address
over plain HTTP does not.

```bash
npm run build      # type check, then bundle into dist/
npm run preview
```

### Verification

```bash
npm run preview
npm run smoke              # headless: exercises the WASM fallback path
HEADED=1 npm run smoke     # real GPU: exercises the WebGPU path
```

The smoke test drives the built site with a synthetic camera, waits for the
model to download and warm up, then reads the depth canvas back and requires it
to contain a non-uniform image. That is the failure mode static checks cannot
catch: the pipeline can be wired correctly end to end and still render nothing
if the texture format, the tensor layout or the shader is wrong.

Headless Chromium exposes a software WebGPU adapter on which ONNX Runtime hangs
at first inference, so the headless run falls back to WASM. That is a property
of the test environment rather than of the site — and it is also how the
fallback path gets exercised.

For diagnosis in a live browser, `depthDiagnostics()` in the console reports
loop state, in-flight status, measured latency and video readiness.

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds the site
and publishes it with GitHub Pages. The repository setting
*Settings → Pages → Source* must be set to **GitHub Actions**.

`vite.config.ts` sets `base` to the repository name because the site is served
from a subdirectory. The workflow overrides it from the repository name at build
time, so renaming the repository does not break asset paths.

Note that a GitHub Pages site is public even when its repository is private.
This is expected here: the site has no secrets, and camera frames never leave
the visitor's device.

## Privacy

No analytics, no cookies, no server. The camera stream is opened only after an
explicit click, is released when the page is hidden or closed, and is processed
in the same browser tab that captured it. The only network requests after page
load are the model weights fetched from the Hugging Face CDN.

## Licence and attribution

This project is released under the [MIT License](LICENSE), © 2026 Enes Öz.

Third-party components retain their own licences:

- **Depth Anything V2 Small** — Apache-2.0. Yang, L., Kang, B., Huang, Z., Zhao,
  Z., Xu, X., Feng, J., Zhao, H. *Depth Anything V2*. NeurIPS 2024.
  arXiv:2406.09414
- **Transformers.js** (`@huggingface/transformers`) — Apache-2.0
- **ONNX Runtime Web**, bundled through Transformers.js — MIT
- **cmocean** colour maps — MIT. Thyng, K. M., Greene, C. A., Hetland, R. D.,
  Zimmerle, H. M., DiMarco, S. F. *True colors of oceanography*. Oceanography
  29(3), 2016.
- **matplotlib** `magma` — CC0
