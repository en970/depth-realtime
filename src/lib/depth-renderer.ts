/**
 * WebGL2 renderer for the depth map.
 *
 * The network emits a small single-channel field (typically 392x224). Uploading
 * that as an R8 texture and letting the hardware do the bilinear magnification
 * is effectively free, whereas colouring at display resolution on the CPU costs
 * ~10 ms per 1080p frame and would consume the entire frame budget on mobile.
 *
 * Two textures are kept. When a new inference result arrives it becomes the
 * target and the previous one stays as the source; the shader interpolates
 * between them over the measured inference interval. Inference at 12 fps
 * therefore still presents as continuous motion at display refresh rate, and
 * the same blend suppresses inter-frame flicker.
 *
 * WebGL2 rather than WebGPU: the visualisation must also run on iOS 18-25 and
 * Firefox/Linux, where WebGPU is unavailable but WebGL2 is universal (~96%).
 * WebGPU is used for inference only, where it actually matters.
 */

const VERT = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  // Flip v: row 0 of the uploaded tensor is the top of the image, but v = 0 is
  // the bottom of the viewport in clip space. Without this the depth pane is
  // upside down relative to the video next to it.
  vUv = vec2(aPos.x, -aPos.y) * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uPrev;
uniform sampler2D uCurr;
uniform sampler2D uLut;
uniform vec2 uPrevRange;   // normalisation bounds captured with uPrev
uniform vec2 uCurrRange;   // normalisation bounds captured with uCurr
uniform float uMix;        // 0 = show uPrev, 1 = show uCurr
uniform float uGamma;
uniform vec2 uCover;       // uv scale that reproduces CSS object-fit: cover

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float normalised(float raw, vec2 range) {
  return clamp((raw - range.x) / max(range.y - range.x, 1e-6), 0.0, 1.0);
}

void main() {
  vec2 uv = (vUv - 0.5) * uCover + 0.5;
  uv = clamp(uv, 0.0, 1.0);

  float prev = normalised(texture(uPrev, uv).r, uPrevRange);
  float curr = normalised(texture(uCurr, uv).r, uCurrRange);
  float t = mix(prev, curr, uMix);

  t = pow(t, uGamma);

  // Half-LSB dither. The ice ramp is smooth enough that flat walls would
  // otherwise show visible bands in an 8-bit canvas.
  t += (hash(gl_FragCoord.xy) - 0.5) / 255.0;

  vec3 rgb = texture(uLut, vec2(clamp(t, 0.0, 1.0), 0.5)).rgb;
  fragColor = vec4(rgb, 1.0);
}`;

interface Slot {
  texture: WebGLTexture;
  width: number;
  height: number;
  lo: number;
  hi: number;
}

export class DepthRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private readonly uniforms: Record<string, WebGLUniformLocation | null>;
  private readonly lutTexture: WebGLTexture;
  private prev: Slot;
  private curr: Slot;
  private mixStart = 0;
  private mixDuration = 1;
  private gamma = 1;
  private frames = 0;
  private dirty = false;
  private readonly canvas: HTMLCanvasElement;

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

    this.program = link(gl, VERT, FRAG);
    gl.useProgram(this.program);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW,
    );
    const loc = gl.getAttribLocation(this.program, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    this.uniforms = {};
    for (const name of [
      "uPrev", "uCurr", "uLut", "uPrevRange", "uCurrRange",
      "uMix", "uGamma", "uCover",
    ]) {
      this.uniforms[name] = gl.getUniformLocation(this.program, name);
    }

    this.prev = this.createSlot();
    this.curr = this.createSlot();
    this.lutTexture = this.createLutTexture();

    gl.uniform1i(this.uniforms.uPrev, 0);
    gl.uniform1i(this.uniforms.uCurr, 1);
    gl.uniform1i(this.uniforms.uLut, 2);
  }

  private createSlot(): Slot {
    const gl = this.gl;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // R8 is filterable everywhere and a quarter of the bandwidth of R16F. R32F
    // would need OES_texture_float_linear and silently yields an incomplete
    // texture on several mobile drivers.
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array(1));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return { texture, width: 1, height: 1, lo: 0, hi: 1 };
  }

  private createLutTexture(): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }

  setColormap(lut: Uint8Array): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.lutTexture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, 256, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, lut);
    this.dirty = true;
  }

  setGamma(value: number): void {
    this.gamma = value;
    this.dirty = true;
  }

  get hasFrame(): boolean {
    return this.frames > 0;
  }

  /**
   * Uploads a new depth field. `transitionMs` should be the observed interval
   * between results so the blend finishes exactly as the next one arrives.
   */
  push(
    data: Uint8Array,
    width: number,
    height: number,
    lo: number,
    hi: number,
    transitionMs: number,
  ): void {
    const gl = this.gl;
    const target = this.prev; // the slot not currently displayed becomes the new target

    gl.bindTexture(gl.TEXTURE_2D, target.texture);
    // Row alignment must be 1: single-channel rows are rarely a multiple of 4
    // (a 350 px wide field is not), and the default of 4 would shear the image.
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    if (target.width !== width || target.height !== height) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, data);
      target.width = width;
      target.height = height;
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RED, gl.UNSIGNED_BYTE, data);
    }
    target.lo = lo;
    target.hi = hi;

    // Swap: the freshly written slot becomes `curr`, the previously shown one
    // becomes `prev` and is blended out.
    this.prev = this.curr;
    this.curr = target;

    if (this.frames === 0) {
      // Seed the other slot with the same field. Writing only its metadata
      // would leave a 1x1 texture behind a 392x224 descriptor, and the next
      // texSubImage2D would fail with an offset overflow.
      const other = this.prev;
      gl.bindTexture(gl.TEXTURE_2D, other.texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, data);
      other.width = width;
      other.height = height;
      other.lo = lo;
      other.hi = hi;
      this.mixDuration = 1;
    } else {
      this.mixDuration = Math.max(16, Math.min(transitionMs, 400));
    }
    this.mixStart = performance.now();
    this.frames++;
    this.dirty = true;
  }

  /** Matches the drawing buffer to the CSS box. Capped at 1.5x device pixels:
   *  the depth field is magnified from roughly 350 px across, so denser
   *  sampling buys nothing while costing fragments linearly. */
  resize(): boolean {
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

    const mix = Math.min(1, (now - this.mixStart) / this.mixDuration);
    const resized = this.resize();

    // The image is static between inference results, so redrawing at display
    // refresh rate burns a full-screen fragment pass for an identical frame.
    if (!resized && !this.dirty && mix >= 1) return;
    if (mix >= 1) this.dirty = false;

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    // object-fit: cover, computed in uv space so the depth pane frames exactly
    // the same region as the video element next to it.
    // object-fit: cover in uv space. The scale is always <= 1: covering means
    // sampling a *smaller* window of the texture, never sampling past its edge.
    // Getting this backwards magnifies the centre and smears the clamped border.
    const canvasAspect = this.canvas.width / this.canvas.height;
    const texAspect = this.curr.width / this.curr.height;
    const cover: [number, number] =
      canvasAspect > texAspect
        ? [1, texAspect / canvasAspect]
        : [canvasAspect / texAspect, 1];

    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.prev.texture);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.curr.texture);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.lutTexture);

    gl.uniform2f(this.uniforms.uPrevRange, this.prev.lo, this.prev.hi);
    gl.uniform2f(this.uniforms.uCurrRange, this.curr.lo, this.curr.hi);
    gl.uniform1f(this.uniforms.uMix, mix);
    gl.uniform1f(this.uniforms.uGamma, this.gamma);
    gl.uniform2f(this.uniforms.uCover, cover[0], cover[1]);

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
