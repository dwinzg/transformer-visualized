/** A CIELAB color with the D65 white point. */
export type Lab = readonly [l: number, a: number, b: number];

/** Linear-light sRGB channels of a #rrggbb color. */
function linearChannels(hex: string): [number, number, number] {
  const match = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!match) throw new Error(`Expected a #rrggbb color, got "${hex}"`);
  const value = Number.parseInt(match[1], 16);
  const [r, g, b] = [(value >> 16) & 255, (value >> 8) & 255, value & 255].map((c) => {
    const s = c / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return [r, g, b];
}

/** WCAG 2.x relative luminance of a #rrggbb color. */
export function relativeLuminance(hex: string): number {
  const [r, g, b] = linearChannels(hex);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG 2.x contrast ratio between two #rrggbb colors, from 1 to 21. */
export function contrastRatio(a: string, b: string): number {
  const [light, dark] = [relativeLuminance(a), relativeLuminance(b)].sort((x, y) => y - x);
  return (light + 0.05) / (dark + 0.05);
}

/** CIELAB coordinates of a #rrggbb color. */
export function hexToLab(hex: string): Lab {
  const [r, g, b] = linearChannels(hex);
  const x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
  const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
  const f = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : ((24389 / 27) * t + 16) / 116);
  return [116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))];
}

const DEG = Math.PI / 180;

/** Hue angle in degrees, from 0 to 360. */
function hueAngle(b: number, a: number): number {
  return a === 0 && b === 0 ? 0 : (Math.atan2(b, a) / DEG + 360) % 360;
}

/**
 * CIEDE2000 color difference, following Sharma, Wu and Dalal (2005). A difference of about 1 is
 * just noticeable side by side.
 */
export function deltaE2000Lab([l1, a1, b1]: Lab, [l2, a2, b2]: Lab): number {
  const meanC = (Math.hypot(a1, b1) + Math.hypot(a2, b2)) / 2;
  const g = 0.5 * (1 - Math.sqrt(meanC ** 7 / (meanC ** 7 + 25 ** 7)));
  const a1p = (1 + g) * a1;
  const a2p = (1 + g) * a2;
  const c1 = Math.hypot(a1p, b1);
  const c2 = Math.hypot(a2p, b2);
  const h1 = hueAngle(b1, a1p);
  const h2 = hueAngle(b2, a2p);

  let dh = 0;
  if (c1 * c2 !== 0) {
    dh = h2 - h1;
    if (dh > 180) dh -= 360;
    else if (dh < -180) dh += 360;
  }
  const dL = l2 - l1;
  const dC = c2 - c1;
  const dH = 2 * Math.sqrt(c1 * c2) * Math.sin((dh / 2) * DEG);

  const meanL = (l1 + l2) / 2;
  const meanCp = (c1 + c2) / 2;
  let meanH = h1 + h2;
  if (c1 * c2 !== 0) {
    if (Math.abs(h1 - h2) <= 180) meanH /= 2;
    else meanH = meanH < 360 ? (meanH + 360) / 2 : (meanH - 360) / 2;
  }

  const t =
    1 -
    0.17 * Math.cos((meanH - 30) * DEG) +
    0.24 * Math.cos(2 * meanH * DEG) +
    0.32 * Math.cos((3 * meanH + 6) * DEG) -
    0.2 * Math.cos((4 * meanH - 63) * DEG);
  const rotation = 30 * Math.exp(-(((meanH - 275) / 25) ** 2));
  const rc = 2 * Math.sqrt(meanCp ** 7 / (meanCp ** 7 + 25 ** 7));
  const sl = 1 + (0.015 * (meanL - 50) ** 2) / Math.sqrt(20 + (meanL - 50) ** 2);
  const sc = 1 + 0.045 * meanCp;
  const sh = 1 + 0.015 * meanCp * t;
  const rt = -Math.sin(2 * rotation * DEG) * rc;
  return Math.sqrt((dL / sl) ** 2 + (dC / sc) ** 2 + (dH / sh) ** 2 + rt * (dC / sc) * (dH / sh));
}

/** CIEDE2000 difference between two #rrggbb colors. */
export function deltaE2000(a: string, b: string): number {
  return deltaE2000Lab(hexToLab(a), hexToLab(b));
}
