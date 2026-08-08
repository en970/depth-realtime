/**
 * WebGL2 renderer for the depth map.
 *
 * The network emits a small single-channel field (typically 392x224). Uploading
 * that as an R16F texture and letting the hardware do the bilinear magnification
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
  vUv = aPos * 0.5 + 0.5;
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
    // R16F is filterable in core WebGL2; R32F would need OES_texture_float_linear
    // and silently produces an incomplete texture on several mobile drivers.
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, 1, 1, 0, gl.RED, gl.FLOAT, new Float32Array(1));
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
  }

  setGamma(value: number): void {
    this.gamma = value;
  }

  get hasFrame(): boolean {
    return this.frames > 0;
  }

  /**
   * Uploads a new depth field. `transitionMs` should be the observed interval
   * between results so the blend finishes exactly as the next one arrives.
   */
  push(
    data: Float32Array,
    width: number,
    height: number,
    lo: number,
    hi: number,
    transitionMs: number,
  ): void {
    const gl = this.gl;
    const target = this.prev; // the slot not currently displayed becomes the new target

    gl.bindTexture(gl.TEXTURE_2D, target.texture);
    if (target.width !== width || target.height !== height) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, width, height, 0, gl.RED, gl.FLOAT, data);
      target.width = width;
      target.height = height;
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RED, gl.FLOAT, data);
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
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, width, height, 0, gl.RED, gl.FLOAT, data);
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
  }

  /** Matches the drawing buffer to the CSS box; capped at 2x device pixels. */
  resize(): boolean {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
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
    this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    const mix = Math.min(1, (now - this.mixStart) / this.mixDuration);

    // object-fit: cover, computed in uv space so the depth pane frames exactly
    // the same region as the video element next to it.
    const canvasAspect = this.canvas.width / this.canvas.height;
    const texAspect = this.curr.width / this.curr.height;
    const cover: [number, number] =
      canvasAspect > texAspect
        ? [1, canvasAspect / texAspect]
        : [texAspect / canvasAspect, 1];

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
