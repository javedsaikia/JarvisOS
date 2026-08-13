// Dense concentric HUD ring system around the orb, matching the layered
// reticle look of the reference: tick ring, segmented arcs, dashed ring,
// node dots, and an outer level arc driven by real TTS audio amplitude.
//
// Built as a 2D SVG overlay, deliberately NOT as Three.js geometry — an
// earlier attempt to add ring geometry into the WebGL scene bloomed into
// light-streak artifacts through UnrealBloomPass. SVG composited on top
// sidesteps that entirely and is far cheaper to iterate on.

const SVG_NS = "http://www.w3.org/2000/svg";
const C = 200; // viewBox centre
const LEVEL_R = 194;

function el<K extends keyof SVGElementTagNameMap>(
  name: K,
  attrs: Record<string, string | number>
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

/** An arc path between two angles at a given radius. */
function arcPath(radius: number, startDeg: number, endDeg: number): string {
  const s = (startDeg * Math.PI) / 180;
  const e = (endDeg * Math.PI) / 180;
  const x1 = C + Math.cos(s) * radius;
  const y1 = C + Math.sin(s) * radius;
  const x2 = C + Math.cos(e) * radius;
  const y2 = C + Math.sin(e) * radius;
  const large = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${radius} ${radius} 0 ${large} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

export class HudRing {
  private levelArc: SVGCircleElement;
  private circumference: number;

  constructor() {
    const svg = document.getElementById("hud-ring") as unknown as SVGSVGElement;

    this.buildTicks(svg.querySelector(".ring-ticks") as SVGGElement, 72, 150);
    this.buildSegments(svg.querySelector(".ring-segments") as SVGGElement, 172, 8, 30);
    this.buildSegments(svg.querySelector(".ring-segments-inner") as SVGGElement, 128, 4, 62);
    this.buildNodes(svg.querySelector(".ring-nodes") as SVGGElement, 12, 186);
    this.buildBrackets(svg.querySelector(".ring-brackets") as SVGGElement, 160);

    this.levelArc = svg.querySelector("#ring-level-arc") as SVGCircleElement;
    this.circumference = 2 * Math.PI * LEVEL_R;
    this.levelArc.setAttribute("stroke-dasharray", `0 ${this.circumference}`);
  }

  /** Fine graduation marks, every 6th one longer — a measuring reticle. */
  private buildTicks(container: SVGGElement, count: number, radius: number): void {
    for (let i = 0; i < count; i++) {
      const angle = ((360 / count) * i * Math.PI) / 180;
      const major = i % 6 === 0;
      const len = major ? 11 : 5;
      container.appendChild(
        el("line", {
          x1: (C + Math.cos(angle) * radius).toFixed(2),
          y1: (C + Math.sin(angle) * radius).toFixed(2),
          x2: (C + Math.cos(angle) * (radius - len)).toFixed(2),
          y2: (C + Math.sin(angle) * (radius - len)).toFixed(2),
          stroke: "var(--blue)",
          "stroke-width": major ? 1.6 : 0.8,
          opacity: major ? 0.85 : 0.4,
        })
      );
    }
  }

  /** Thick arc chunks separated by gaps — the heavy "armour segment" ring. */
  private buildSegments(container: SVGGElement, radius: number, count: number, sweep: number): void {
    const step = 360 / count;
    for (let i = 0; i < count; i++) {
      const start = i * step;
      container.appendChild(
        el("path", {
          d: arcPath(radius, start, start + sweep),
          fill: "none",
          stroke: "var(--blue)",
          "stroke-width": 5,
          "stroke-linecap": "butt",
          opacity: 0.5,
        })
      );
    }
  }

  /** Small dots at intervals, like connection nodes on the outer ring. */
  private buildNodes(container: SVGGElement, count: number, radius: number): void {
    for (let i = 0; i < count; i++) {
      const angle = ((360 / count) * i * Math.PI) / 180;
      container.appendChild(
        el("circle", {
          cx: (C + Math.cos(angle) * radius).toFixed(2),
          cy: (C + Math.sin(angle) * radius).toFixed(2),
          r: 2,
          fill: "var(--blue)",
          opacity: 0.7,
        })
      );
    }
  }

  /** Four corner arc brackets at the diagonals, framing the core. */
  private buildBrackets(container: SVGGElement, radius: number): void {
    [45, 135, 225, 315].forEach((start) => {
      container.appendChild(
        el("path", {
          d: arcPath(radius, start - 13, start + 13),
          fill: "none",
          stroke: "var(--blue-bright)",
          "stroke-width": 2.5,
          "stroke-linecap": "round",
          opacity: 0.9,
        })
      );
    });
  }

  /** amplitude in [0, 1] — outer arc fills as JARVIS speaks, from the same
   * real analyser data that drives the orb's glow. */
  setAmplitude(amplitude: number): void {
    const clamped = Math.max(0, Math.min(1, amplitude));
    this.levelArc.setAttribute(
      "stroke-dasharray",
      `${this.circumference * clamped} ${this.circumference}`
    );
  }
}
