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
  camera ──► <video>                           (playback, always visible)
                │
                │ requestVideoFrameCallback, plus an immediate re-arm each time
                │ a result arrives, so inference never waits for the camera
                ▼
      2D canvas, scaled to a multiple of 14 ──► ImageData ──► Web Worker
                                                                 │
                                        one pass: RGBA to NCHW,  │
                                        rescale and ImageNet     │
                                        normalisation folded in  │
                                                                 ▼
                                          Depth Anything V2 Small (ONNX)
                                            WebGPU, or WASM where unavailable
                                                                 │
                                                                 │ inverse depth
                                                                 ▼
                                    p2/p98 histogram bounds ──► asymmetric EMA
                                                                 │
                                                    per-pixel exponential average
                                                                 │
                                              8-bit normalised field (transferred)
                                                                 ▼
  depth pane ◄── WebGL2 fragment shader ◄──── R8 texture + 256×1 colour LUT
```

Design decisions worth stating explicitly:

**No inversion is applied.** Depth Anything V2 emits affine-invariant *inverse*
depth, so larger values are already nearer. Applying `1 - t` — the reflex when
coming from datasets such as KITTI, where smaller values are nearer — would
invert the entire visualisation. Note that Depth Anything 3 reverses this
convention again, and the V2 metric checkpoints return metres; switching
checkpoints therefore requires revisiting this decision.

**Distance is mapped logarithmically.** The network emits inverse depth, so
value is proportional to 1/distance: a person at half a metre and a wall at
three metres differ by a factor of six, which puts everything past a couple of
metres into the last few percent of the range — about fifteen levels once
quantised, which is why the background used to read as a flat black mass. In log
space equal ratios of distance get equal shares of the output. Measured on a
room scene, the far third goes from ten distinct levels to seventy-nine.

Contrast-limited histogram equalisation was tried first and does not work at any
histogram resolution: equalisation cannot separate pixels that share a bin, and
after linear scaling the entire background shares one.

**Temporal accumulation follows the camera and backs off where things move.**
Smoothing a still scene removes noise for free; smoothing a moving one drags the
old position behind the new. Both need care to get right. The motion test needs
a threshold at the noise floor, or a noisy pixel exempts itself from the
smoothing that would have quieted it and gets noisier — measured, that mistake
sent drift from 5.9 to 8.4. And the comparison has to follow the scene: hand
held, the image travels several field texels per update, and comparing screen
positions makes every edge look like motion. With both, on a panning scene, edge
alignment rises 4.6 to 7.8 and drift falls 12.5 to 2.9 — sharper and steadier at
once.

The settings are scaled by the frame interval actually being achieved, because
it varies six-fold across devices. Uncorrected, the same slider gave a 31 ms
half-life on a laptop and 189 ms on a phone.

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
field is therefore uploaded as a small single-channel `R8` texture and magnified
by the sampler, with the colour ramp applied in a fragment shader. Eight bits
are enough because the field is a normalised ramp rather than measurement data,
and the shader dithers by half a least significant bit to break up banding.

**Preprocessing by hand.** The library's image processor re-ran a bicubic resize
that the capture canvas had already performed and allocated several intermediate
images on the way to an NCHW tensor: 5.4 ms per frame measured. Converting the
RGBA bytes directly, with the rescale and the ImageNet normalisation folded into
one multiply-add per channel, costs 0.33 ms.

**Two clocks.** Capture is driven by `requestVideoFrameCallback`, which fires
once per camera frame rather than once per display refresh; on a 30 fps camera
and a 60 Hz screen this halves the work. It is also re-armed the moment a result
arrives rather than only on the next camera callback — waiting for one cost
about 16 ms of dead time per cycle, roughly a third of the achievable frame
rate. Presentation runs on `requestAnimationFrame`, interpolates between the
last two depth fields so the pane stays smooth when inference is slower than
capture, and stops redrawing once that blend has finished, since the image is
static between results.

**Orientation and aspect are pinned by tests.** Row 0 of the tensor is the top
of the image, but `v = 0` is the bottom of the viewport in clip space, so the
vertex stage inverts `v`. The cover scale is always at most 1: covering means
sampling a smaller window of the texture, never sampling past its edge. Neither
property is noticeable on a live camera until someone points at it, so
`tests/renderer.mjs` asserts both against a known field.

**The viewer is sized in script, not stretched.** Two 16:9 panes side by side in
a stage wider than it is tall gives each pane a box of roughly 0.64:1 — far
taller than the camera frame — and `object-fit: cover` then discards about two
thirds of the horizontal field of view. The viewer is therefore fitted to the
camera's aspect ratio so the whole frame survives and the unused space falls
outside the panes. CSS alone cannot express this: the panes are limited by the
stage width in one layout and by its height in another, and `aspect-ratio` stops
preserving the ratio once the second limit binds.

**One worker restart on a fatal failure.** Transformers.js serialises web
inference through a module-scoped promise chain with no rejection handler, so a
genuine `session.run()` rejection leaves that chain permanently rejected: every
later call fails immediately with the original error, including calls on a
freshly created session. The state cannot be cleared from inside the realm, so
the page discards the worker and starts a new one, carrying forward which
backend must not be tried again.

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
| Weights | `fp16` (49.6 MB) on WebGPU, `q8` (27.3 MB) on WASM |

Within the Depth Anything V2 family only the Small checkpoint is permissively
licensed. Base, Large and Giant are CC-BY-NC-4.0 and cannot be used in an
MIT-licensed project. The quality gap is modest — NYU-D δ1 of 0.973 for Small
against 0.979 for Large — so the licence constraint costs little here.

4-bit weights were once the mobile default, for a 19 MB download instead of
50 MB. They are gone: on a Galaxy S22 they produced diagonal banding unrelated
to the scene — arithmetic failing quietly rather than erroring — and even where
they work they are slower than fp16 rather than faster, since dequantising costs
more than the narrower weights save.

A model is only accepted after it has produced a plausible field from two
synthetic frames. Finite output is not enough: it must vary, must respond
differently to different inputs, and must be smooth, since a depth field is
continuous while broken arithmetic is not. Measured, a working model scores
0.003 on that last test and the banding scored 0.28.

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

## Views

Beyond the depth map itself, three ways of reading the same field:

**Relief shading** darkens the far side of every silhouette and shades hollows
and ridges, so surface curvature reads as shape rather than as a flat band of
colour. Without it a face is one white patch; with it the nose, eye sockets and
jaw line are legible.

**Iso-depth contours** draw a line at every fixed step of depth, as on a
topographic map. A smooth ramp shows that two surfaces differ but not by how
much; lines make it countable and reveal gradients too shallow to see as colour.

**Parallax** replaces the map with the camera image swayed by depth. Near things
move against far ones, which reads as three dimensions on a flat screen. The
warp is backward, so occluded edges stretch rather than tearing holes.

**Point cloud** shows the estimate as geometry rather than as a picture of
geometry, turned by dragging or by a slow idle sway. A wall that bulges or an
object floating off its background is obvious here and invisible on a colour map.

Three layouts are available: side by side, stacked, and a comparison slider that
overlays the two and wipes between them. Stacked gives each pane roughly 1.8x
the area on a typical laptop screen, since a 16:9 pane is limited by the stage
height rather than its width when two of them share a row.

Input resolution, temporal smoothing and the colour scale can be set manually;
the state is written to the URL fragment, so a particular configuration can be
shared or cited.

Measured on an Apple Silicon laptop, WebGPU with fp16 weights, 350x196 network
input:

| | Before | After |
| --- | --- | --- |
| Throughput | 10 fps | 20 fps |
| Time per inference | 74 ms | 47 ms |
| Preprocessing within that | 5.4 ms | 0.33 ms |
| Main-thread frame gap, median | 16.6 ms | 16.6 ms |
| Main-thread frame gap, p95 | 20.4 ms | 17.7 ms |

The main thread was never the bottleneck; the gains came from removing dead time
between inferences, from the preprocessing rewrite, and from not redrawing a
static image sixty times a second.

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
