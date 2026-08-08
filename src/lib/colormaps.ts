/**
 * Colour scales for the depth map.
 *
 * Every scale here is monotonic in lightness. That is not an aesthetic
 * preference: a non-monotonic ramp (jet, turbo, Spectral) invents edges that
 * are not present in the data and inverts the perceived ordering for viewers
 * with colour vision deficiency.
 *
 * Stops are converted once into a 256x1 RGB lookup texture; the shader then
 * samples it. Recomputing a polynomial approximation per pixel would be both
 * slower and less faithful than the table it approximates.
 */

export type ColormapId = "ice" | "gray" | "magma";

interface Colormap {
  id: ColormapId;
  label: string;
  /** Evenly spaced stops, t = 0 far, t = 1 near. */
  stops: string[];
}

/**
 * cmocean "ice" (MIT licensed), reconstructed from 17 evenly spaced samples of
 * the reference table. Maximum deviation from the full 256-entry table is
 * 1.4/255, which is below the quantisation step of the output canvas.
 */
const ICE = [
  "#040613", "#13132a", "#212041", "#2e2c5a", "#383975", "#3e478f",
  "#3f57a3", "#3e69b0", "#427bb7", "#4b8bbd", "#579cc3", "#67adc9",
  "#79bed0", "#92ced7", "#afdde1", "#cdecef", "#eafdfd",
];

/** matplotlib "magma" (CC0), sampled at 10 stops. */
const MAGMA = [
  "#000004", "#180f3d", "#440f76", "#721f81", "#9e2f7f",
  "#cd4071", "#f1605d", "#fd9668", "#feca8d", "#fcfdbf",
];

const GRAY = ["#050505", "#ffffff"];

export const COLORMAPS: Record<ColormapId, Colormap> = {
  ice: { id: "ice", label: "Ice", stops: ICE },
  gray: { id: "gray", label: "Greyscale", stops: GRAY },
  magma: { id: "magma", label: "Magma", stops: MAGMA },
};

function parseHex(hex: string): [number, number, number] {
  const v = Number.parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/** Builds the 256x1 RGB table used as the shader lookup texture. */
export function buildLut(id: ColormapId): Uint8Array {
  const stops = COLORMAPS[id].stops.map(parseHex);
  const lut = new Uint8Array(256 * 3);
  const segments = stops.length - 1;

  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * segments;
    const lo = Math.min(Math.floor(t), segments - 1);
    const f = t - lo;
    const a = stops[lo];
    const b = stops[lo + 1];
    lut[i * 3 + 0] = Math.round(a[0] + (b[0] - a[0]) * f);
    lut[i * 3 + 1] = Math.round(a[1] + (b[1] - a[1]) * f);
    lut[i * 3 + 2] = Math.round(a[2] + (b[2] - a[2]) * f);
  }
  return lut;
}

/** CSS gradient for the legend, so it cannot drift from the shader table. */
export function cssGradient(id: ColormapId): string {
  const stops = COLORMAPS[id].stops;
  const parts = stops.map(
    (hex, i) => `${hex} ${((i / (stops.length - 1)) * 100).toFixed(2)}%`,
  );
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}
