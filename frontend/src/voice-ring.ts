// Circular waveform wrapped around the core.
//
// The orb glows and swells with audio, but glow alone is a poor read of
// "Max is mid-sentence" — the eye needs shape change, not just
// brightness. This draws the last ~2 seconds of measured amplitude as
// bars around the circle, newest at the top, mirrored left/right so the
// figure stays symmetric. Speech becomes a visible ripple travelling
// outward from the core; silence collapses it back to a thin ring.
//
// Every bar is a real sample: browser-played audio comes from the same
// AnalyserNode the orb uses, and voice-loop speech comes from the
// envelope measured off the actual WAV being played on the Mac's
// speakers (max/voice_events.py). When nothing is playing, the samples
// really are zero and the ring really does go flat — no idle wiggle.

export type RingStatus = "idle" | "listening" | "thinking" | "speaking" | "error" | "offline";

const STATUS_COLOR: Record<RingStatus, string> = {
  idle: "0, 212, 255",
  listening: "0, 212, 255",
  thinking: "217, 70, 239",
  speaking: "44, 232, 184",
  error: "255, 77, 94",
  offline: "255, 77, 94",
};

// Half the ring; the other half is this buffer mirrored. 96 bars per side
// at 60fps is a little over 1.5s of history on screen at once.
const BARS = 96;

export class VoiceRing {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private samples: number[] = new Array(BARS).fill(0);
  private latest = 0;
  private smoothed = 0;
  private status: RingStatus = "idle";
  private spin = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.resize();
    window.addEventListener("resize", () => this.resize());
    requestAnimationFrame(this.frame);
  }

  private resize(): void {
    const dpr = Math.min(window.devicePixelRatio, 2);
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /** amplitude in [0, 1] — the same value the orb is driven with. */
  setAmplitude(amplitude: number): void {
    this.latest = Math.max(0, Math.min(1, amplitude));
  }

  setStatus(status: RingStatus): void {
    this.status = status;
  }

  private frame = (): void => {
    // Ease toward the incoming level so a dropped frame doesn't punch a
    // notch into the ring, then record it as this frame's sample.
    this.smoothed += (this.latest - this.smoothed) * 0.35;
    this.samples.push(this.smoothed);
    this.samples.shift();

    this.draw();
    requestAnimationFrame(this.frame);
  };

  private draw(): void {
    const { width, height } = this.canvas.getBoundingClientRect();
    const ctx = this.ctx;
    ctx.clearRect(0, 0, width, height);

    const cx = width / 2;
    const cy = height / 2;
    // Just outside the reticle's outer level arc (r=194 in its 400-unit
    // viewBox, and that SVG is sized to 62vmin against this canvas's
    // 82vmin) — close enough to read as part of the core, clear of every
    // ring that is already drawn there.
    const baseR = Math.min(width, height) * 0.4;
    const maxBar = Math.min(width, height) * 0.075;
    const rgb = STATUS_COLOR[this.status];

    // Slow drift so the ring never looks like a frozen still frame while
    // idle. Rotation only — it moves nothing that encodes data.
    this.spin += 0.0015;

    // The ring is never absent, only flat: with nothing playing this
    // circle is all that's left, which is the honest picture of silence
    // and still reads as a live instrument rather than a blank area.
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = `rgba(${rgb}, 0.3)`;
    ctx.beginPath();
    ctx.arc(cx, cy, baseR, 0, Math.PI * 2);
    ctx.stroke();

    ctx.lineCap = "round";
    ctx.shadowColor = `rgba(${rgb}, 0.8)`;

    for (let i = 0; i < BARS; i++) {
      // Newest sample at the top (-90°), history sweeping backwards down
      // both sides.
      const offset = (i / BARS) * Math.PI;
      const level = this.samples[BARS - 1 - i];
      const len = level * maxBar;
      if (len < 0.4) continue;

      const alpha = 0.25 + level * 0.75;
      ctx.strokeStyle = `rgba(${rgb}, ${alpha.toFixed(3)})`;
      ctx.lineWidth = 1.8;
      ctx.shadowBlur = 6 * level;

      for (const dir of [1, -1]) {
        const angle = -Math.PI / 2 + dir * offset + this.spin;
        const x1 = cx + Math.cos(angle) * baseR;
        const y1 = cy + Math.sin(angle) * baseR;
        const x2 = cx + Math.cos(angle) * (baseR + len);
        const y2 = cy + Math.sin(angle) * (baseR + len);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
    ctx.shadowBlur = 0;
  }
}
