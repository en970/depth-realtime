/**
 * WebGL2 renderer for the depth map.
 *
 * Work is split by how often it actually needs to happen.
 *
 * The compose pass runs once per inference result. It magnifies the small
 * single-channel field the network produces, steers that magnification by the
 * camera image so silhouettes stay sharp, shades relief from the field, and
 * maps the result through the colour table — writing a finished RGB frame.
 *
 * The display pass runs on every animation frame and does one thing: cross-fade
 * between the last two composed frames. Inference at 12 fps therefore still
 * presents as continuous motion, and the blend also suppresses inter-frame
 * flicker. Doing the expensive work here instead measured as a third of the
 * frame rate, for no visible gain: nothing in it changes between results.
 *
 * WebGL2 rather than WebGPU: the visualisation must also run on iOS 18-25 and
 * Firefox/Linux, where WebGPU is unavailable but WebGL2 is universal (~96%).
 * WebGPU is used for inference only, where it actually matters.
 */

/**
 * Compose pass vertex stage: flips v.
 *
 * Row 0 of the uploaded tensor is the top of the image, but v = 0 is the bottom
 * of the viewport in clip space, so without this the field would be sampled
 * upside down.
 */
const VERT = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = vec2(aPos.x, -aPos.y) * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

/**
 * Display pass vertex stage: does NOT flip.
 *
 * The compose pass already wrote the frame the right way up into a texture, and
 * rendering to a framebuffer puts row 0 at the bottom. Flipping again here
 * would cancel the first flip out and stand the image on its head — which is
 * exactly what tests/renderer.mjs caught when both passes shared one stage.
 */
const VERT_DISPLAY = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uPrevColor;
uniform sampler2D uCurrColor;
uniform sampler2D uCamera;
uniform float uMix;
uniform float uParallax;   // 0 = show the depth map, > 0 = warped camera
uniform vec2 uShift;       // current view offset, in uv units
uniform vec2 uCoverCam;    // uv scale that frames the camera like the panes do

void main() {
  vec4 previous = texture(uPrevColor, vUv);
  vec4 current = texture(uCurrColor, vUv);
  vec4 blended = mix(previous, current, uMix);

  if (uParallax > 0.001) {
    // Backward warp: for each output pixel, look up where it came from. Warping
    // forwards would tear holes open behind foreground objects; going backwards
    // there is nothing to fill, because every output pixel gets a source. The
    // cost is that hidden surfaces stretch rather than being revealed, which
    // reads as a soft smear at silhouettes instead of a black gap.
    //
    // Depth is inverse, so it is already large-is-near: subtracting the midpoint
    // makes near objects move opposite to far ones, which is what sells the
    // effect.
    vec2 source = vUv + uShift * (blended.a - 0.5);
    vec2 camera = clamp((source - 0.5) * uCoverCam + 0.5, 0.0, 1.0);
    // This stage does not flip v (the composed frames are already the right way
    // up), but the camera texture is uploaded unflipped and needs it.
    camera.y = 1.0 - camera.y;
    vec3 warped = texture(uCamera, camera).rgb;
    fragColor = vec4(mix(blended.rgb, warped, uParallax), 1.0);
    return;
  }

  fragColor = vec4(blended.rgb, 1.0);
}`;

const COMPOSE_FRAG = `#version 300 es
precision highp float;

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uField;
uniform sampler2D uGuideTex;
uniform sampler2D uLutTex;
uniform vec2 uTexel;
uniform vec2 uCoverS;
uniform vec3 uLight;
uniform float uExaggeration;
uniform float uBase;
uniform float uNoiseFloor;
uniform float uGuideAmountS;
uniform float uRangeSigmaS;
uniform float uStructureS;
uniform float uContours;      // 0 = off, 1 = full strength
uniform float uContourBands;  // number of iso-depth steps across the range

float hashS(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float lumaS(vec3 c) {
  return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
  vec2 uv = clamp((vUv - 0.5) * uCoverS + 0.5, 0.0, 1.0);

  // Guided upsampling, done here rather than in the display pass. The field is
  // perhaps 350 px across and the pane three times that, so plain magnification
  // smears every silhouette; re-weighting the samples by how similar the camera
  // pixel is at the same place pulls the edge back onto the outline.
  float t;
  if (uGuideAmountS <= 0.001) {
    t = texture(uField, uv).r;
  } else {
    float guideCentre = lumaS(texture(uGuideTex, uv).rgb);
    float sum = 0.0;
    float weightSum = 0.0;
    for (int y = -2; y <= 2; y++) {
      for (int x = -2; x <= 2; x++) {
        vec2 offset = vec2(float(x), float(y)) * uTexel;
        vec2 sampleUv = clamp(uv + offset, 0.0, 1.0);
        float spatial = exp(-float(x * x + y * y) / 4.0);
        float difference = lumaS(texture(uGuideTex, sampleUv).rgb) - guideCentre;
        float range = exp(-(difference * difference) / (2.0 * uRangeSigmaS * uRangeSigmaS));
        float weight = spatial * mix(1.0, range, uGuideAmountS);
        sum += weight * texture(uField, sampleUv).r;
        weightSum += weight;
      }
    }
    t = sum / max(weightSum, 1e-5);
  }

  // Wide radius: depth darkening (Luft, Colditz and Deussen 2006). Where the
  // field sits below its own neighbourhood there is a silhouette with a far
  // side; darkening only that side draws a halo around every object. It
  // separates two things painted the same shade without needing colour
  // contrast, and unlike surface shading it holds up at distance because it
  // keys on a large depth step rather than a small slope.
  // Measured: deriving these from the guided value instead of a raw sample
  // trades 8% of edge alignment for 22% less drift. Edge alignment is the
  // target here, so the raw sample wins.
  float centre = texture(uField, uv).r;
  float wide = mix(textureLod(uField, uv, 2.0).r, textureLod(uField, uv, 4.0).r, 0.5);
  float tight = textureLod(uField, uv, 1.0).r;


  float halo = min(centre - wide, 0.0) * 6.0;

  // Tight radius: the same quantity at one texel, which is curvature. Hollows
  // darken and ridges lift, so knuckles, folds and a table edge stop being flat.
  float curvature = centre - tight;
  float cavity = min(curvature, 0.0) * 8.0 + max(curvature, 0.0) * 4.0;

  // Surface shading: a Sobel estimate of the gradient builds a normal, which a
  // fixed light then shades. The gate below is not optional — see the note on
  // the noise floor where the uniform is set.
  vec2 tap = uTexel * uBase;
  float ul = texture(uField, uv + vec2(-tap.x, -tap.y)).r;
  float uc = texture(uField, uv + vec2(0.0, -tap.y)).r;
  float ur = texture(uField, uv + vec2(tap.x, -tap.y)).r;
  float ml = texture(uField, uv + vec2(-tap.x, 0.0)).r;
  float mr = texture(uField, uv + vec2(tap.x, 0.0)).r;
  float dl = texture(uField, uv + vec2(-tap.x, tap.y)).r;
  float dc = texture(uField, uv + vec2(0.0, tap.y)).r;
  float dr = texture(uField, uv + vec2(tap.x, tap.y)).r;

  // 1-2-1 weights average three rows, dividing quantisation noise by about
  // sqrt(3). Dividing by the baseline keeps the exaggeration meaning the same
  // if the baseline is widened.
  float gx = ((ur + 2.0 * mr + dr) - (ul + 2.0 * ml + dl)) * 0.125 / uBase;
  float gy = ((ul + 2.0 * uc + ur) - (dl + 2.0 * dc + dr)) * 0.125 / uBase;

  vec3 normal = normalize(vec3(-gx * uExaggeration, -gy * uExaggeration, 1.0));
  float shade = dot(normal, uLight) * 0.5 + 0.5;
  shade *= shade;

  float trust = smoothstep(uNoiseFloor, 4.0 * uNoiseFloor, length(vec2(gx, gy)));
  float relief = trust * (shade - 0.5) * 0.16;

  // Bounded gain, applied in linear light. Never an overlay blend: the colour
  // table is the only carrier of the value, and a full-swing overlay would
  // invert lightness order across half the range.
  float gain = clamp(1.0 + uStructureS * (halo + cavity + relief), 0.86, 1.10);

  // Iso-depth contours, drawn from the undithered value: a topographic map
  // reading of the same field. A smooth ramp shows that two surfaces differ but
  // not by how much; a line every fixed step turns that into something
  // countable, which is also how a shallow gradient becomes visible at all.
  float contour = 1.0;
  if (uContours > 0.001) {
    float bands = t * uContourBands;
    float distance = min(fract(bands), 1.0 - fract(bands));
    // fwidth keeps the line one pixel wide wherever the gradient takes it,
    // instead of thin on steep surfaces and a wide band on flat ones.
    float width = fwidth(bands);
    contour = smoothstep(0.0, max(width, 1e-5), distance);
    contour = mix(1.0, mix(0.55, 1.0, contour), uContours);
  }

  float shaded = t + (hashS(gl_FragCoord.xy) - 0.5) / 255.0;
  float lutWidth = float(textureSize(uLutTex, 0).x);
  float lutU = (clamp(shaded, 0.0, 1.0) * (lutWidth - 1.0) + 0.5) / lutWidth;
  vec3 rgb = texture(uLutTex, vec2(lutU, 0.5)).rgb;

  if (uStructureS > 0.001 || uContours > 0.001) {
    vec3 linear = pow(rgb, vec3(2.2)) * gain * contour;
    rgb = pow(max(linear, 0.0), vec3(1.0 / 2.2));
  }

  // Alpha carries the depth value itself. The display pass needs it to warp the
  // camera image for parallax, and the channel is otherwise wasted.
  fragColor = vec4(rgb, clamp(t, 0.0, 1.0));
}`;

/**
 * Point cloud view.
 *
 * One point per texel of the depth field, placed in space and projected. This
 * is the only view where the depth estimate is shown as geometry rather than as
 * a picture of geometry, which makes errors obvious in a way a colour map hides:
 * a wall that bulges, or an object floating off its background, is visible the
 * moment the view turns.
 *
 * Positions are computed in the vertex stage from gl_VertexID, so no vertex
 * buffer is uploaded and nothing has to be re-sent when the field resizes.
 */
const CLOUD_VERT = `#version 300 es
precision highp float;

uniform sampler2D uField;
uniform sampler2D uCloudCamera;
uniform sampler2D uCloudLut;
uniform vec2 uFieldSize;
uniform vec2 uCoverCloud;
uniform vec2 uRotation;    // yaw, pitch in radians
uniform float uDepthScale;
uniform float uAspect;
uniform float uPointSize;
uniform float uColourMix;  // 0 = colour scale, 1 = camera colour

const float NEAR_PLANE = 1.35;
const float FAR_PLANE = 3.1;
const float PIVOT = 2.2;
const float FOCAL = 1.5;
/** Chosen so the cloud fills the pane head-on with room to turn without
 *  clipping at the edges. */
const float SPREAD = 0.72;

out vec3 vColour;

void main() {
  int index = gl_VertexID;
  int column = index % int(uFieldSize.x);
  int row = index / int(uFieldSize.x);
  vec2 uv = (vec2(float(column), float(row)) + 0.5) / uFieldSize;

  float depth = texture(uField, uv).r;

  // Drop points that straddle a depth discontinuity. A silhouette in a depth
  // map is a ramp a texel or two wide, and those in-between samples have no
  // surface to belong to: seen head-on they hide behind the edge, but as soon
  // as the view turns they string out into a veil between foreground and
  // background. Discarding them costs a thin outline and removes the veil.
  vec2 texel = 1.0 / uFieldSize;
  float dx = abs(texture(uField, uv + vec2(texel.x, 0.0)).r
               - texture(uField, uv - vec2(texel.x, 0.0)).r);
  float dy = abs(texture(uField, uv + vec2(0.0, texel.y)).r
               - texture(uField, uv - vec2(0.0, texel.y)).r);
  if (max(dx, dy) > 0.06) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);   // outside the clip volume
    gl_PointSize = 0.0;
    vColour = vec3(0.0);
    return;
  }

  // Distance from a bounded range rather than 1/depth. The estimate is
  // affine-invariant, so its absolute scale is arbitrary anyway, and taking the
  // reciprocal literally sends the far plane to infinity: the background
  // stretches into a long tail that dominates the view and hides the subject.
  float distance = mix(FAR_PLANE, NEAR_PLANE, clamp(depth, 0.0, 1.0)) * uDepthScale;

  // Back-projection: x and y scale with distance, which is what makes a flat
  // wall stay flat when the view turns. The vertical extent is divided by the
  // field's aspect, since uv is normalised on both axes and a 16:9 field would
  // otherwise come out square.
  vec2 centred = (uv - 0.5) * vec2(2.0, -2.0);
  centred.y /= uFieldSize.x / uFieldSize.y;
  vec3 position = vec3(centred * distance * SPREAD, -distance);

  // Turn about the middle of the cloud, not about the camera.
  position.z += PIVOT;
  float cy = cos(uRotation.x), sy = sin(uRotation.x);
  float cp = cos(uRotation.y), sp = sin(uRotation.y);
  position.xz = mat2(cy, -sy, sy, cy) * position.xz;
  position.yz = mat2(cp, -sp, sp, cp) * position.yz;
  position.z -= PIVOT;

  float w = max(-position.z, 0.05);
  gl_Position = vec4(position.x * FOCAL / uAspect, position.y * FOCAL, 0.0, w);
  // Points shrink with distance, as they would if they had physical size.
  gl_PointSize = clamp(uPointSize / w, 1.0, 9.0);

  vec2 camera = clamp((uv - 0.5) * uCoverCloud + 0.5, 0.0, 1.0);
  vec3 fromCamera = texture(uCloudCamera, camera).rgb;
  float lutWidth = float(textureSize(uCloudLut, 0).x);
  vec3 fromScale = texture(uCloudLut,
      vec2((clamp(depth, 0.0, 1.0) * (lutWidth - 1.0) + 0.5) / lutWidth, 0.5)).rgb;
  vColour = mix(fromScale, fromCamera, uColourMix);
}`;

const CLOUD_FRAG = `#version 300 es
precision highp float;
in vec3 vColour;
out vec4 fragColor;
void main() {
  // Round points: square ones read as a grid artefact rather than as samples.
  vec2 offset = gl_PointCoord - 0.5;
  if (dot(offset, offset) > 0.25) discard;
  fragColor = vec4(vColour, 1.0);
}`;


interface Field {
  texture: WebGLTexture;
  width: number;
  height: number;
}

export class DepthRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly canvas: HTMLCanvasElement;

  private readonly displayProgram: WebGLProgram;
  private readonly displayUniforms: Record<string, WebGLUniformLocation | null> = {};
  private readonly composeProgram: WebGLProgram;
  private readonly composeUniforms: Record<string, WebGLUniformLocation | null> = {};

  /** Raw depth fields as uploaded by the worker. */
  private field: Field;
  private previousField: Field;

  /** Finished RGB frames, one per inference result. */
  private currColor: WebGLTexture;
  private prevColor: WebGLTexture;
  private colorWidth = 0;
  private colorHeight = 0;
  private framebuffer: WebGLFramebuffer;

  private readonly lutTexture: WebGLTexture;
  private readonly guideTexture: WebGLTexture;
  private guideSource: HTMLVideoElement | null = null;

  private guideAmount = 0;
  private rangeSigma = 0.06;
  private structure = 0;
  private contours = 0;
  private contourBands = 16;
  private parallax = 0;
  private needsCamera = false;
  private cloudProgram!: WebGLProgram;
  private cloudUniforms: Record<string, WebGLUniformLocation | null> = {};
  private cloud = false;
  private cloudYaw = 0;
  private cloudPitch = 0;
  private cloudColourMix = 1;
  private mixStart = 0;
  private mixDuration = 1;
  private frames = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      desynchronized: true,
      powerPreference: "high-performance",
      preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error("WebGL2 is not available in this browser.");
    this.gl = gl;

    this.displayProgram = link(gl, VERT_DISPLAY, FRAG);
    this.composeProgram = link(gl, VERT, COMPOSE_FRAG);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

    // Both programs read the same full-screen triangle, but the attribute has
    // to be enabled once per program.
    for (const program of [this.displayProgram, this.composeProgram]) {
      gl.useProgram(program);
      const location = gl.getAttribLocation(program, "aPos");
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
    }

    gl.useProgram(this.displayProgram);
    for (const name of [
      "uPrevColor", "uCurrColor", "uCamera", "uMix", "uParallax", "uShift", "uCoverCam",
    ]) {
      this.displayUniforms[name] = gl.getUniformLocation(this.displayProgram, name);
    }
    gl.uniform1i(this.displayUniforms.uPrevColor, 5);
    gl.uniform1i(this.displayUniforms.uCurrColor, 6);
    gl.uniform1i(this.displayUniforms.uCamera, 3);

    gl.useProgram(this.composeProgram);
    for (const name of [
      "uField", "uGuideTex", "uLutTex", "uTexel", "uCoverS", "uLight",
      "uExaggeration", "uBase", "uNoiseFloor", "uGuideAmountS", "uRangeSigmaS",
      "uStructureS", "uContours", "uContourBands",
    ]) {
      this.composeUniforms[name] = gl.getUniformLocation(this.composeProgram, name);
    }
    gl.uniform1i(this.composeUniforms.uField, 1);
    gl.uniform1i(this.composeUniforms.uLutTex, 2);
    gl.uniform1i(this.composeUniforms.uGuideTex, 3);

    this.field = this.createField();
    this.previousField = this.createField();
    this.lutTexture = this.createLut();
    this.guideTexture = this.createGuide();
    this.currColor = this.createColorTarget();
    this.prevColor = this.createColorTarget();
    this.framebuffer = gl.createFramebuffer()!;

    this.cloudProgram = link(gl, CLOUD_VERT, CLOUD_FRAG);
    gl.useProgram(this.cloudProgram);
    for (const name of [
      "uField", "uCloudCamera", "uCloudLut", "uFieldSize", "uCoverCloud",
      "uRotation", "uDepthScale", "uAspect", "uPointSize", "uColourMix",
    ]) {
      this.cloudUniforms[name] = gl.getUniformLocation(this.cloudProgram, name);
    }
    gl.uniform1i(this.cloudUniforms.uField, 1);
    gl.uniform1i(this.cloudUniforms.uCloudLut, 2);
    gl.uniform1i(this.cloudUniforms.uCloudCamera, 3);
    gl.useProgram(this.displayProgram);
  }

  /**
   * Shows the field as geometry instead of as a picture. Rotation is in
   * radians; the caller drives it from pointer drag or from a slow idle spin.
   */
  setCloud(enabled: boolean, colourMix = 1): void {
    this.cloud = enabled;
    this.cloudColourMix = colourMix;
    this.needsCamera = enabled || this.parallax > 0.001;
  }

  setCloudRotation(yaw: number, pitch: number): void {
    this.cloudYaw = yaw;
    this.cloudPitch = pitch;
  }

  private renderCloud(): void {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.027, 0.035, 0.063, 1);   // --bg-inset
    gl.clear(gl.COLOR_BUFFER_BIT);

    this.uploadGuide();

    gl.useProgram(this.cloudProgram);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.field.texture);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.lutTexture);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.guideTexture);

    gl.uniform2f(this.cloudUniforms.uFieldSize, this.field.width, this.field.height);
    gl.uniform2f(this.cloudUniforms.uRotation, this.cloudYaw, this.cloudPitch);
    gl.uniform1f(this.cloudUniforms.uDepthScale, 1);
    gl.uniform1f(this.cloudUniforms.uAspect, this.canvas.width / this.canvas.height);
    gl.uniform1f(this.cloudUniforms.uPointSize, this.canvas.height / 90);
    gl.uniform1f(this.cloudUniforms.uColourMix, this.cloudColourMix);

    const source = this.guideSource;
    const cameraAspect =
      source && source.videoWidth ? source.videoWidth / source.videoHeight : 16 / 9;
    const fieldAspect = this.field.width / this.field.height;
    const cover: [number, number] =
      fieldAspect > cameraAspect
        ? [1, cameraAspect / fieldAspect]
        : [fieldAspect / cameraAspect, 1];
    gl.uniform2f(this.cloudUniforms.uCoverCloud, cover[0], cover[1]);

    gl.drawArrays(gl.POINTS, 0, this.field.width * this.field.height);
  }

  private createField(): Field {
    const gl = this.gl;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // R8 is filterable everywhere and a quarter of the bandwidth of R16F. R32F
    // would need OES_texture_float_linear and silently yields an incomplete
    // texture on several mobile drivers. Mipmapped, because relief shading
    // reads blurred copies of the field from the mip chain instead of running
    // a separate blur pass.
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array(1));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return { texture, width: 1, height: 1 };
  }

  private createColorTarget(): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }

  private createLut(): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }

  private createGuide(): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 0, 255]),
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }

  setColormap(lut: Uint8Array): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.lutTexture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, 256, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, lut);
    this.recompose();
  }

  /**
   * Supplies the camera frame used to steer edge-aware upsampling. Uploading
   * straight from the video element avoids the intermediate canvas draw and the
   * readback a CPU copy would cost.
   */
  setGuide(source: HTMLVideoElement | null): void {
    this.guideSource = source;
  }

  setGuideStrength(amount: number, rangeSigma = 0.06): void {
    this.guideAmount = amount;
    this.rangeSigma = Math.max(0.005, rangeSigma);
    this.recompose();
  }

  setStructure(amount: number): void {
    this.structure = amount;
    this.recompose();
  }

  setContours(amount: number, bands = 16): void {
    this.contours = amount;
    this.contourBands = bands;
    this.recompose();
  }

  /**
   * Parallax amplitude, 0 to 1. Above zero the pane shows the camera image
   * warped by depth rather than the depth map itself, which is the only version
   * of this that works: shifting a colourised depth map moves a silhouette
   * around but carries no texture for the eye to track.
   */
  setParallax(amount: number): void {
    this.parallax = amount;
    // The guide texture is only uploaded while guided upsampling is on, but
    // parallax needs the camera too.
    this.needsCamera = amount > 0.001;
  }

  get hasFrame(): boolean {
    return this.frames > 0;
  }

  /**
   * Uploads a new depth field and composes it. `transitionMs` should be the
   * observed interval between results, so the cross-fade finishes exactly as
   * the next one arrives.
   */
  push(data: Uint8Array, width: number, height: number, transitionMs: number): void {
    const gl = this.gl;
    const target = this.previousField;

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, target.texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    if (target.width !== width || target.height !== height) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, data);
      target.width = width;
      target.height = height;
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RED, gl.UNSIGNED_BYTE, data);
    }
    gl.generateMipmap(gl.TEXTURE_2D);

    this.previousField = this.field;
    this.field = target;

    this.uploadGuide();

    // Swap the colour targets first: what was composed last time becomes the
    // frame being faded out.
    const outgoing = this.prevColor;
    this.prevColor = this.currColor;
    this.currColor = outgoing;
    this.compose();

    if (this.frames === 0) {
      // Nothing to fade from on the very first frame; compose into both so the
      // first displayed image is not a blend with an empty texture.
      const spare = this.prevColor;
      this.prevColor = this.currColor;
      this.currColor = spare;
      this.compose();
      this.mixDuration = 1;
    } else {
      this.mixDuration = Math.max(16, Math.min(transitionMs, 400));
    }
    this.mixStart = performance.now();
    this.frames++;
  }

  private uploadGuide(): void {
    const source = this.guideSource;
    if (!source || (this.guideAmount <= 0.001 && !this.needsCamera)) return;
    if (source.readyState < 2 || source.videoWidth === 0) return;
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.guideTexture);
    // No flip: the depth texture is uploaded unflipped too, and the vertex
    // stage already inverts v. Flipping only one of the two would sample the
    // guide upside down relative to the depth it steers.
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, source);
  }

  /** Re-runs the compose pass for a settings change, between results. */
  private recompose(): void {
    if (this.frames === 0) return;
    this.compose();
  }

  private compose(): void {
    const gl = this.gl;
    this.resize();
    const width = this.canvas.width;
    const height = this.canvas.height;
    if (width === 0 || height === 0) return;

    if (this.colorWidth !== width || this.colorHeight !== height) {
      for (const texture of [this.currColor, this.prevColor]) {
        gl.activeTexture(gl.TEXTURE5);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      }
      this.colorWidth = width;
      this.colorHeight = height;
    }

    // The target must not stay bound to a sampler unit while it is the colour
    // attachment; that is a feedback loop and the draw becomes undefined.
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_2D, null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.currColor, 0,
    );
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return;
    }
    gl.viewport(0, 0, width, height);

    gl.useProgram(this.composeProgram);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.field.texture);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.lutTexture);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.guideTexture);

    // object-fit: cover in uv space. The scale is always <= 1: covering means
    // sampling a smaller window of the texture, never sampling past its edge.
    const canvasAspect = width / height;
    const fieldAspect = this.field.width / this.field.height;
    const cover: [number, number] =
      canvasAspect > fieldAspect
        ? [1, fieldAspect / canvasAspect]
        : [canvasAspect / fieldAspect, 1];

    gl.uniform2f(this.composeUniforms.uCoverS, cover[0], cover[1]);
    gl.uniform2f(
      this.composeUniforms.uTexel,
      1 / Math.max(1, this.field.width),
      1 / Math.max(1, this.field.height),
    );
    gl.uniform1f(this.composeUniforms.uGuideAmountS, this.guideSource ? this.guideAmount : 0);
    gl.uniform1f(this.composeUniforms.uRangeSigmaS, this.rangeSigma);
    gl.uniform1f(this.composeUniforms.uStructureS, this.structure);
    gl.uniform1f(this.composeUniforms.uContours, this.contours);
    gl.uniform1f(this.composeUniforms.uContourBands, this.contourBands);
    // Light from the north-north-west at 45 degrees elevation. Above-left is the
    // cartographic convention for shaded relief; from below, hollows read as
    // bumps.
    gl.uniform3f(this.composeUniforms.uLight, -0.2706, 0.6533, 0.7071);
    gl.uniform1f(this.composeUniforms.uExaggeration, 14);
    gl.uniform1f(this.composeUniforms.uBase, 1.5);
    // 8-bit transport puts gradient noise near 4.9e-4 per texel, scaled by the
    // baseline. Below that there is no slope to shade, only quantisation.
    gl.uniform1f(this.composeUniforms.uNoiseFloor, 4.9e-4 / 1.5);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Matches the drawing buffer to the CSS box. Capped at 1.5x device pixels:
   *  the field is magnified from roughly 350 px across, so denser sampling buys
   *  nothing while costing fragments linearly. */
  private resize(): boolean {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width === width && this.canvas.height === height) return false;
    this.canvas.width = width;
    this.canvas.height = height;
    return true;
  }

  render(now: number): void {
    if (this.frames === 0) return;
    const gl = this.gl;

    // A resize invalidates both composed frames, so rebuild them before drawing.
    if (this.canvas.clientWidth * Math.min(window.devicePixelRatio || 1, 1.5) !== this.colorWidth) {
      this.compose();
    }

    if (this.cloud) {
      this.resize();
      this.renderCloud();
      return;
    }

    // performance.now() and the animation-frame timestamp are not the same
    // clock; measured, 11-18% of pushes see a difference of up to -1.1 ms.
    const mix = Math.min(1, Math.max(0, (now - this.mixStart) / this.mixDuration));

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.displayProgram);
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, this.prevColor);
    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_2D, this.currColor);
    gl.uniform1f(this.displayUniforms.uMix, mix);
    gl.uniform1f(this.displayUniforms.uParallax, this.parallax);

    if (this.parallax > 0.001) {
      // Keep the camera upload going even between inference results: the warp is
      // animated, so a stale frame would visibly lag the motion.
      this.uploadGuide();
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, this.guideTexture);

      // A slow sway of about one and a half percent of the width. Larger reads
      // as a wobble rather than as depth, and stretches the occluded edges
      // until the smear becomes the thing you notice.
      const phase = (now / 1400) * Math.PI * 2;
      gl.uniform2f(this.displayUniforms.uShift, Math.sin(phase) * 0.03, 0);

      const source = this.guideSource;
      const cameraAspect =
        source && source.videoWidth ? source.videoWidth / source.videoHeight : 16 / 9;
      const canvasAspect = this.canvas.width / this.canvas.height;
      const cover: [number, number] =
        canvasAspect > cameraAspect
          ? [1, cameraAspect / canvasAspect]
          : [canvasAspect / cameraAspect, 1];
      gl.uniform2f(this.displayUniforms.uCoverCam, cover[0], cover[1]);
    }

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compilation failed: ${log}`);
  }
  return shader;
}

function link(gl: WebGL2RenderingContext, vert: string, frag: string): WebGLProgram {
  const program = gl.createProgram()!;
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vert));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, frag));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link failed: ${log}`);
  }
  return program;
}
