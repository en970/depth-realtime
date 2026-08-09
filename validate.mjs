import { chromium } from "playwright";
import * as S from "./shaders.mjs";

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.goto("about:blank");

const shaders = {
  JBU: S.JBU,
  FGF_PACK: S.FGF_PACK,
  FGF_COEF: S.FGF_COEF,
  FGF_BOX: S.FGF_BOX,
  FGF_APPLY: S.FGF_APPLY,
  CATMULL: S.CATMULL,
  GUIDED_BILINEAR: S.GUIDED_BILINEAR,
};

const result = await page.evaluate(
  ({ vert, shaders }) => {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const gl = canvas.getContext("webgl2");
    if (!gl) return { error: "no webgl2" };

    const out = {
      renderer: gl.getParameter(gl.RENDERER),
      version: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
      halfFloat: !!gl.getExtension("EXT_color_buffer_half_float"),
      colorBufferFloat: !!gl.getExtension("EXT_color_buffer_float"),
      programs: {},
      rg8Renderable: null,
      rg16fRenderable: null,
    };

    const compile = (type, src) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      const ok = gl.getShaderParameter(sh, gl.COMPILE_STATUS);
      const log = ok ? "" : gl.getShaderInfoLog(sh);
      return { sh, ok, log };
    };

    const v = compile(gl.VERTEX_SHADER, vert);
    if (!v.ok) return { error: "vertex: " + v.log };

    for (const [name, src] of Object.entries(shaders)) {
      const f = compile(gl.FRAGMENT_SHADER, src);
      if (!f.ok) {
        out.programs[name] = { ok: false, stage: "compile", log: f.log };
        continue;
      }
      const p = gl.createProgram();
      gl.attachShader(p, v.sh);
      gl.attachShader(p, f.sh);
      gl.linkProgram(p);
      const linked = gl.getProgramParameter(p, gl.LINK_STATUS);
      out.programs[name] = linked
        ? { ok: true, uniforms: gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS) }
        : { ok: false, stage: "link", log: gl.getProgramInfoLog(p) };
    }

    // Renderability probes.
    const probe = (internal) => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texStorage2D(gl.TEXTURE_2D, 1, internal, 16, 16);
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return status === gl.FRAMEBUFFER_COMPLETE;
    };
    out.rg8Renderable = probe(gl.RG8);
    out.r8Renderable = probe(gl.R8);
    out.rg16fRenderable = probe(gl.RG16F);

    return out;
  },
  { vert: S.VERT, shaders },
);

console.log(JSON.stringify(result, null, 2));
await browser.close();
