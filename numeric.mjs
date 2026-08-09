import { chromium } from "playwright";
import * as S from "./shaders.mjs";

const VERT_NOFLIP = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.goto("about:blank");

const out = await page.evaluate(
  ({ vert, PACK, COEF, BOX, APPLY, JBU, GB }) => {
    const LR = 16, HR = 64, SCALE = HR / LR;
    const canvas = document.createElement("canvas");
    canvas.width = HR; canvas.height = HR;
    const gl = canvas.getContext("webgl2", { antialias: false });
    gl.getExtension("EXT_color_buffer_float");
    gl.getExtension("EXT_color_buffer_half_float");

    const mk = (src, type) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const prog = (fs) => { const p = gl.createProgram();
      gl.attachShader(p, mk(vert, gl.VERTEX_SHADER)); gl.attachShader(p, mk(fs, gl.FRAGMENT_SHADER));
      gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p; };

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);

    const bindQuad = (p) => { gl.useProgram(p); const l = gl.getAttribLocation(p, "aPos");
      gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 2, gl.FLOAT, false, 0, 0); };

    const tex = (internal, w, h, fmt, type, data) => {
      const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, fmt, type, data ?? null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    const fbo = (t) => { const f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { f, complete: st === gl.FRAMEBUFFER_COMPLETE }; };

    /* --- synthetic scene ------------------------------------------------ */
    // Case A: a depth step at x = 8 LR px, smeared over 3 LR px (what a
    // 350 px network output actually looks like), and a razor-sharp guide
    // edge at the same place.
    // Case B (rows >= 8): depth is perfectly flat, guide carries a 2px
    // checkerboard -> tests texture copying.
    const depth = new Uint8Array(LR * LR);
    const guide = new Uint8Array(HR * HR * 4);
    for (let y = 0; y < LR; y++) for (let x = 0; x < LR; x++) {
      let d;
      if (y < 8) { const t = Math.min(1, Math.max(0, (x - 6.5) / 3.0)); d = 0.25 + 0.5 * t; }
      else       { d = 0.5; }
      depth[y * LR + x] = Math.round(d * 255);
    }
    for (let y = 0; y < HR; y++) for (let x = 0; x < HR; x++) {
      const ly = y / SCALE;
      let g;
      if (ly < 8) g = x < 32 ? 0.2 : 0.8;
      else        g = (((x >> 1) + (y >> 1)) & 1) ? 0.85 : 0.15;
      const v = Math.round(g * 255), i = (y * HR + x) * 4;
      guide[i] = v; guide[i + 1] = v; guide[i + 2] = v; guide[i + 3] = 255;
    }
    // Rec.601 of a gray value is the value itself, so guide luma == g.

    const lut = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) { lut[i * 3] = i; lut[i * 3 + 1] = i; lut[i * 3 + 2] = i; }

    const tDepth = tex(gl.R8, LR, LR, gl.RED, gl.UNSIGNED_BYTE, depth);
    const tGuide = tex(gl.RGBA8, HR, HR, gl.RGBA, gl.UNSIGNED_BYTE, guide);
    const tLut   = tex(gl.RGB8, 256, 1, gl.RGB, gl.UNSIGNED_BYTE, lut);
    const tPack  = tex(gl.RG8,  LR, LR, gl.RG, gl.UNSIGNED_BYTE, null);
    const tCoef  = tex(gl.RG16F, LR, LR, gl.RG, gl.HALF_FLOAT, null);
    const tCoefB = tex(gl.RG16F, LR, LR, gl.RG, gl.HALF_FLOAT, null);
    // nearest for the coefficient probe would defeat the point; keep linear.

    const fPack = fbo(tPack), fCoef = fbo(tCoef), fCoefB = fbo(tCoefB);

    const pPack = prog(PACK), pCoef = prog(COEF), pBox = prog(BOX), pApply = prog(APPLY);
    const pJbu = prog(JBU), pGb = prog(GB);

    const draw = (p, target, w, h, setup) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target);
      gl.viewport(0, 0, w, h);
      bindQuad(p); setup(p);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    const bind = (p, name, unit, t) => { gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t); gl.uniform1i(gl.getUniformLocation(p, name), unit); };

    draw(pPack, fPack.f, LR, LR, (p) => {
      bind(p, "uPrev", 0, tDepth); bind(p, "uCurr", 1, tDepth); bind(p, "uGuideHi", 2, tGuide);
      gl.uniform1f(gl.getUniformLocation(p, "uMix"), 1.0);
    });
    draw(pCoef, fCoef.f, LR, LR, (p) => {
      bind(p, "uPack", 0, tPack);
      gl.uniform1f(gl.getUniformLocation(p, "uEps"), 1e-3);
      gl.uniform1f(gl.getUniformLocation(p, "uAClamp"), 0.0);
    });
    draw(pBox, fCoefB.f, LR, LR, (p) => { bind(p, "uCoef", 0, tCoef); });

    const readRow = (row) => {
      const px = new Uint8Array(HR * 4);
      gl.readPixels(0, row, HR, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      const r = []; for (let x = 0; x < HR; x++) r.push(px[x * 4] / 255);
      return r;
    };

    const results = {};

    // Guided filter output.
    draw(pApply, null, HR, HR, (p) => {
      bind(p, "uCoef", 0, tCoefB); bind(p, "uGuideHi", 1, tGuide); bind(p, "uLut", 2, tLut);
      gl.uniform2f(gl.getUniformLocation(p, "uCover"), 1, 1);
      gl.uniform1f(gl.getUniformLocation(p, "uDetail"), 1.0);
    });
    results.gf_edgeRow = readRow(8);     // inside the step region
    results.gf_flatRow = readRow(48);    // inside the checkerboard region

    // JBU output.
    draw(pJbu, null, HR, HR, (p) => {
      bind(p, "uDepth", 0, tPack); bind(p, "uGuideHi", 1, tGuide); bind(p, "uLut", 2, tLut);
      gl.uniform2f(gl.getUniformLocation(p, "uCover"), 1, 1);
      gl.uniform1f(gl.getUniformLocation(p, "uSigmaS"), 0.8);
      gl.uniform1f(gl.getUniformLocation(p, "uSigmaR"), 0.1);
    });
    results.jbu_edgeRow = readRow(8);
    results.jbu_flatRow = readRow(48);

    // Guide-weighted bilinear.
    draw(pGb, null, HR, HR, (p) => {
      bind(p, "uPack", 0, tPack); bind(p, "uGuideHi", 1, tGuide); bind(p, "uLut", 2, tLut);
      gl.uniform2f(gl.getUniformLocation(p, "uCover"), 1, 1);
      gl.uniform1f(gl.getUniformLocation(p, "uSigmaR"), 0.1);
    });
    results.gb_edgeRow = readRow(8);
    results.gb_flatRow = readRow(48);

    results.fbo = { pack: fPack.complete, coef: fCoef.complete, box: fCoefB.complete };
    return results;
  },
  { vert: VERT_NOFLIP, PACK: S.FGF_PACK, COEF: S.FGF_COEF, BOX: S.FGF_BOX,
    APPLY: S.FGF_APPLY, JBU: S.JBU, GB: S.GUIDED_BILINEAR },
);

const fmt = (a) => a.map((v) => v.toFixed(2)).join(" ");
const transition = (row) => {
  // width in HR pixels over which the value crosses 10%..90% of its range
  const lo = Math.min(...row), hi = Math.max(...row);
  const a = lo + 0.1 * (hi - lo), b = lo + 0.9 * (hi - lo);
  let first = -1, last = -1;
  for (let i = 0; i < row.length; i++) {
    if (row[i] > a && first < 0) first = i;
    if (row[i] < b) last = i;
  }
  return last - first + 1;
};
const ripple = (row) => (Math.max(...row) - Math.min(...row));

console.log("FBO completeness:", out.fbo);
console.log("\n--- step edge (LR edge smeared over 3 LR px = 12 HR px) ---");
for (const k of ["gf_edgeRow", "jbu_edgeRow", "gb_edgeRow"]) {
  console.log(k.padEnd(12), "10-90% width (HR px):", transition(out[k]));
  console.log("   ", fmt(out[k].slice(24, 40)));
}
console.log("\n--- flat depth under a checkerboard guide (texture copying) ---");
for (const k of ["gf_flatRow", "jbu_flatRow", "gb_flatRow"]) {
  console.log(k.padEnd(12), "peak-to-peak ripple:", ripple(out[k]).toFixed(4));
  console.log("   ", fmt(out[k].slice(24, 40)));
}

await browser.close();
