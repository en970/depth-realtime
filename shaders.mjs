export const VERT = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = vec2(aPos.x, -aPos.y) * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

/* ------------------------------------------------------------------ */
/* 1. Joint bilateral upsampling, full-resolution single pass          */
/* ------------------------------------------------------------------ */
export const JBU = `#version 300 es
precision highp float;
precision highp sampler2D;

#define RADIUS 2            // 2 -> 5x5, 3 -> 7x7

in  vec2 vUv;
out vec4 fragColor;

uniform sampler2D uDepth;   // RG8 low-res: .r = blended depth, .g = guide luma
uniform sampler2D uGuideHi; // RGBA8 camera frame, full resolution
uniform sampler2D uLut;     // RGB8 256x1 colour ramp

uniform vec2  uCover;
uniform float uSigmaS;      // low-res pixel units (Kopf et al.: 0.5)
uniform float uSigmaR;      // luma units in [0,1] (Kopf et al.: 0.1)

const vec3 REC601 = vec3(0.299, 0.587, 0.114);

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  vec2 uv = clamp((vUv - 0.5) * uCover + 0.5, 0.0, 1.0);

  ivec2 lrSize = textureSize(uDepth, 0);
  vec2  size   = vec2(lrSize);

  // Guide value at the *output* pixel: this is I~_p in Kopf eq. (3).
  float ip = dot(texture(uGuideHi, uv).rgb, REC601);

  // Position of this output pixel in low-resolution texel coordinates.
  vec2  pd     = uv * size - 0.5;
  ivec2 centre = ivec2(floor(pd + 0.5));
  vec2  base   = vec2(centre);

  float invS2 = 1.0 / (2.0 * uSigmaS * uSigmaS);
  float invR2 = 1.0 / (2.0 * uSigmaR * uSigmaR);

  float acc  = 0.0;
  float wacc = 0.0;

  for (int dy = -RADIUS; dy <= RADIUS; dy++) {
    for (int dx = -RADIUS; dx <= RADIUS; dx++) {
      ivec2 q = clamp(centre + ivec2(dx, dy), ivec2(0), lrSize - 1);
      vec2  t = texelFetch(uDepth, q, 0).rg;

      // f(||p_down - q_down||): spatial term, measured on the low-res lattice.
      vec2  d  = (base + vec2(dx, dy)) - pd;
      float ws = exp(-dot(d, d) * invS2);

      // g(||I~_p - I~_q||): range term against the box-averaged guide.
      float dr = ip - t.g;
      float wr = exp(-dr * dr * invR2);

      float w = ws * wr;
      acc  += w * t.r;
      wacc += w;
    }
  }

  float v = acc / max(wacc, 1e-6);
  v += (hash(gl_FragCoord.xy) - 0.5) / 255.0;

  float lutW = float(textureSize(uLut, 0).x);
  float lutU = (clamp(v, 0.0, 1.0) * (lutW - 1.0) + 0.5) / lutW;
  fragColor = vec4(texture(uLut, vec2(lutU, 0.5)).rgb, 1.0);
}`;

/* ------------------------------------------------------------------ */
/* 2. Fast guided filter, pass A: pack (depth, guide luma) at LR       */
/* ------------------------------------------------------------------ */
export const FGF_PACK = `#version 300 es
precision highp float;
precision highp sampler2D;

in  vec2 vUv;
out vec4 fragColor;

uniform sampler2D uPrev;     // R8 low-res depth, previous result
uniform sampler2D uCurr;     // R8 low-res depth, latest result
uniform sampler2D uGuideHi;  // RGBA8 camera frame, full resolution
uniform float     uMix;

const vec3 REC601 = vec3(0.299, 0.587, 0.114);

void main() {
  float p = mix(texture(uPrev, vUv).r, texture(uCurr, vUv).r, uMix);

  // Box-average the guide over exactly one low-resolution cell. A single
  // bilinear tap would alias: at 1280 -> 350 the cell spans 3.7 source
  // pixels, and the aliased high-frequency content is precisely what shows
  // up later as texture copying.
  vec2  cell = 1.0 / vec2(textureSize(uPrev, 0));
  float lum  = 0.0;
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      vec2 o = ((vec2(float(i), float(j)) + 0.5) / 4.0 - 0.5) * cell;
      lum += dot(texture(uGuideHi, vUv + o).rgb, REC601);
    }
  }
  fragColor = vec4(p, lum / 16.0, 0.0, 1.0);
}`;

/* ------------------------------------------------------------------ */
/* 3. Fast guided filter, pass B: local linear coefficients a, b       */
/* ------------------------------------------------------------------ */
export const FGF_COEF = `#version 300 es
precision highp float;
precision highp sampler2D;

#define RADIUS 2            // r' in He & Sun's Algorithm 2

in  vec2 vUv;
out vec2 fragColor;         // (a, b)

uniform sampler2D uPack;    // RG8: .r = depth p, .g = guide I
uniform float     uEps;     // He et al.: plays the role of sigma_r^2
uniform float     uAClamp;  // hard cap on |a|; 0 disables

void main() {
  ivec2 size = textureSize(uPack, 0);
  ivec2 c    = ivec2(vUv * vec2(size));
  c = clamp(c, ivec2(0), size - 1);

  // Shifted-data accumulation. Subtracting the centre sample before
  // squaring keeps var = E[d^2] - E[d]^2 away from catastrophic
  // cancellation, which matters because I lives in [0,1] and a textured
  // but flat-depth wall has var_I of order 1e-3.
  vec2 o = texelFetch(uPack, c, 0).rg;

  float n    = 0.0;
  float sd   = 0.0;   // sum (I - I0)
  float sdd  = 0.0;   // sum (I - I0)^2
  float sp   = 0.0;   // sum (p - p0)
  float sdp  = 0.0;   // sum (I - I0)(p - p0)

  for (int dy = -RADIUS; dy <= RADIUS; dy++) {
    for (int dx = -RADIUS; dx <= RADIUS; dx++) {
      ivec2 q = clamp(c + ivec2(dx, dy), ivec2(0), size - 1);
      vec2  s = texelFetch(uPack, q, 0).rg - o;
      sd  += s.g;
      sdd += s.g * s.g;
      sp  += s.r;
      sdp += s.g * s.r;
      n   += 1.0;
    }
  }

  float mI = sd / n;
  float mP = sp / n;
  float varI  = sdd / n - mI * mI;
  float covIp = sdp / n - mI * mP;

  float a = covIp / (varI + uEps);
  if (uAClamp > 0.0) a = clamp(a, -uAClamp, uAClamp);

  // b is expressed against the un-shifted means: mean_p = mP + p0,
  // mean_I = mI + I0.
  float b = (mP + o.r) - a * (mI + o.g);

  fragColor = vec2(a, b);
}`;

/* ------------------------------------------------------------------ */
/* 4. Fast guided filter, pass C: box mean of (a, b)                   */
/* ------------------------------------------------------------------ */
export const FGF_BOX = `#version 300 es
precision highp float;
precision highp sampler2D;

#define RADIUS 2

in  vec2 vUv;
out vec2 fragColor;

uniform sampler2D uCoef;

void main() {
  ivec2 size = textureSize(uCoef, 0);
  ivec2 c    = clamp(ivec2(vUv * vec2(size)), ivec2(0), size - 1);

  vec2  acc = vec2(0.0);
  float n   = 0.0;
  for (int dy = -RADIUS; dy <= RADIUS; dy++) {
    for (int dx = -RADIUS; dx <= RADIUS; dx++) {
      ivec2 q = clamp(c + ivec2(dx, dy), ivec2(0), size - 1);
      acc += texelFetch(uCoef, q, 0).rg;
      n   += 1.0;
    }
  }
  fragColor = acc / n;
}`;

/* ------------------------------------------------------------------ */
/* 5. Fast guided filter, pass D: q = mean_a * I + mean_b, full res    */
/* ------------------------------------------------------------------ */
export const FGF_APPLY = `#version 300 es
precision highp float;
precision highp sampler2D;

in  vec2 vUv;
out vec4 fragColor;

uniform sampler2D uCoef;     // RG16F (mean_a, mean_b), low resolution
uniform sampler2D uGuideHi;  // RGBA8 camera frame, full resolution
uniform sampler2D uLut;
uniform vec2      uCover;
uniform float     uDetail;   // 0 = plain bilinear depth, 1 = full guided

const vec3 REC601 = vec3(0.299, 0.587, 0.114);

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  vec2 uv = clamp((vUv - 0.5) * uCover + 0.5, 0.0, 1.0);

  // Both coefficient maps are smooth by construction, so the hardware
  // bilinear tap is the correct interpolant here — this is the whole point
  // of the fast guided filter: interpolate the model, not the signal.
  vec2  ab = texture(uCoef, uv).rg;
  float I  = dot(texture(uGuideHi, uv).rgb, REC601);

  float q = ab.x * I + ab.y;

  // uDetail lets the coefficient pair fall back to a pure local mean
  // (a = 0, b = mean_p), which is what the current renderer already shows.
  q = mix(ab.y + ab.x * 0.0, q, uDetail);

  q = clamp(q, 0.0, 1.0);
  q += (hash(gl_FragCoord.xy) - 0.5) / 255.0;

  float lutW = float(textureSize(uLut, 0).x);
  float lutU = (clamp(q, 0.0, 1.0) * (lutW - 1.0) + 0.5) / lutW;
  fragColor = vec4(texture(uLut, vec2(lutU, 0.5)).rgb, 1.0);
}`;

/* ------------------------------------------------------------------ */
/* 6. Catmull-Rom bicubic, 9 bilinear taps, with a ringing clamp       */
/* ------------------------------------------------------------------ */
export const CATMULL = `#version 300 es
precision highp float;
precision highp sampler2D;

in  vec2 vUv;
out vec4 fragColor;

uniform sampler2D uDepth;    // R8 low-res
uniform sampler2D uLut;
uniform vec2      uCover;
uniform float     uClampRing; // 1 = clamp to the 2x2 neighbourhood

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float catmullRom(sampler2D tex, vec2 uv, vec2 size) {
  vec2 sp  = uv * size;
  vec2 tp1 = floor(sp - 0.5) + 0.5;
  vec2 f   = sp - tp1;

  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  vec2 w3 = f * f * (-0.5 + 0.5 * f);

  vec2 w12 = w1 + w2;
  vec2 off = w2 / w12;

  vec2 t0  = (tp1 - 1.0) / size;
  vec2 t3  = (tp1 + 2.0) / size;
  vec2 t12 = (tp1 + off) / size;

  float r = 0.0;
  r += texture(tex, vec2(t0.x,  t0.y )).r * w0.x  * w0.y;
  r += texture(tex, vec2(t12.x, t0.y )).r * w12.x * w0.y;
  r += texture(tex, vec2(t3.x,  t0.y )).r * w3.x  * w0.y;
  r += texture(tex, vec2(t0.x,  t12.y)).r * w0.x  * w12.y;
  r += texture(tex, vec2(t12.x, t12.y)).r * w12.x * w12.y;
  r += texture(tex, vec2(t3.x,  t12.y)).r * w3.x  * w12.y;
  r += texture(tex, vec2(t0.x,  t3.y )).r * w0.x  * w3.y;
  r += texture(tex, vec2(t12.x, t3.y )).r * w12.x * w3.y;
  r += texture(tex, vec2(t3.x,  t3.y )).r * w3.x  * w3.y;
  return r;
}

void main() {
  vec2  uv   = clamp((vUv - 0.5) * uCover + 0.5, 0.0, 1.0);
  ivec2 isz  = textureSize(uDepth, 0);
  vec2  size = vec2(isz);

  float v = catmullRom(uDepth, uv, size);

  if (uClampRing > 0.5) {
    // The Catmull-Rom kernel has negative lobes, so a depth step produces an
    // over- and undershoot that reads as a painted outline around every
    // object. Clamping to the four nearest samples removes it at the cost of
    // flattening the very sharpening the filter was added for.
    ivec2 c  = ivec2(floor(uv * size - 0.5));
    float a0 = texelFetch(uDepth, clamp(c,               ivec2(0), isz - 1), 0).r;
    float a1 = texelFetch(uDepth, clamp(c + ivec2(1, 0), ivec2(0), isz - 1), 0).r;
    float a2 = texelFetch(uDepth, clamp(c + ivec2(0, 1), ivec2(0), isz - 1), 0).r;
    float a3 = texelFetch(uDepth, clamp(c + ivec2(1, 1), ivec2(0), isz - 1), 0).r;
    float lo = min(min(a0, a1), min(a2, a3));
    float hi = max(max(a0, a1), max(a2, a3));
    v = clamp(v, lo, hi);
  }

  v += (hash(gl_FragCoord.xy) - 0.5) / 255.0;
  float lutW = float(textureSize(uLut, 0).x);
  float lutU = (clamp(v, 0.0, 1.0) * (lutW - 1.0) + 0.5) / lutW;
  fragColor = vec4(texture(uLut, vec2(lutU, 0.5)).rgb, 1.0);
}`;

/* ------------------------------------------------------------------ */
/* 7. Guide-weighted bilinear: 4 depth taps, the cheapest edge-aware   */
/* ------------------------------------------------------------------ */
export const GUIDED_BILINEAR = `#version 300 es
precision highp float;
precision highp sampler2D;

in  vec2 vUv;
out vec4 fragColor;

uniform sampler2D uPack;     // RG8 low-res: .r depth, .g box-averaged luma
uniform sampler2D uGuideHi;
uniform sampler2D uLut;
uniform vec2      uCover;
uniform float     uSigmaR;

const vec3 REC601 = vec3(0.299, 0.587, 0.114);

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  vec2  uv   = clamp((vUv - 0.5) * uCover + 0.5, 0.0, 1.0);
  ivec2 isz  = textureSize(uPack, 0);
  vec2  size = vec2(isz);

  float ip = dot(texture(uGuideHi, uv).rgb, REC601);

  vec2  sp = uv * size - 0.5;
  ivec2 c  = ivec2(floor(sp));
  vec2  f  = sp - vec2(c);

  vec2 s0 = texelFetch(uPack, clamp(c,               ivec2(0), isz - 1), 0).rg;
  vec2 s1 = texelFetch(uPack, clamp(c + ivec2(1, 0), ivec2(0), isz - 1), 0).rg;
  vec2 s2 = texelFetch(uPack, clamp(c + ivec2(0, 1), ivec2(0), isz - 1), 0).rg;
  vec2 s3 = texelFetch(uPack, clamp(c + ivec2(1, 1), ivec2(0), isz - 1), 0).rg;

  vec4 wb = vec4((1.0 - f.x) * (1.0 - f.y),
                 f.x * (1.0 - f.y),
                 (1.0 - f.x) * f.y,
                 f.x * f.y);

  float invR2 = 1.0 / (2.0 * uSigmaR * uSigmaR);
  vec4  dr = vec4(s0.g, s1.g, s2.g, s3.g) - ip;
  vec4  wr = exp(-dr * dr * invR2);
  vec4  w  = wb * wr;

  float wsum = dot(w, vec4(1.0));
  float v = dot(w, vec4(s0.r, s1.r, s2.r, s3.r)) / max(wsum, 1e-6);

  v += (hash(gl_FragCoord.xy) - 0.5) / 255.0;
  float lutW = float(textureSize(uLut, 0).x);
  float lutU = (clamp(v, 0.0, 1.0) * (lutW - 1.0) + 0.5) / lutW;
  fragColor = vec4(texture(uLut, vec2(lutU, 0.5)).rgb, 1.0);
}`;
