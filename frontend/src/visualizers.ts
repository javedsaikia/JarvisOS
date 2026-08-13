// Canvas HUD visualizers driven by the SAME real AnalyserNode the orb uses.
// Nothing here synthesizes a fake signal: when no audio is playing they
// render an explicit flat "no signal" baseline rather than idle noise that
// would imply the mic/output is live when it isn't.

import type { AudioPlayer } from "./audio";

const BLUE = "#00d4ff";
const BLUE_SOFT = "rgba(0, 212, 255, 0.5)";
const BLUE_FAINT = "rgba(0, 212, 255, 0.18)";

function fitCanvas(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext("2d")!;
  const dpr = Math.min(window.devicePixelRatio, 2);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.scale(dpr, dpr);
  return ctx;
}

export class Visualizers {
  private waveCtx: CanvasRenderingContext2D;
  private specCtx: CanvasRenderingContext2D;
  private histCtx: CanvasRenderingContext2D;
  private waveCanvas: HTMLCanvasElement;
  private specCanvas: HTMLCanvasElement;
  private histCanvas: HTMLCanvasElement;
  private audio: AudioPlayer;

  /** Rolling history of real output levels, one sample per animation frame
   * batch — drives the bar-chart panel. Starts empty (all zero), fills only
   * from genuine measured amplitude. */
  private history: number[] = new Array(28).fill(0);
  private frameCount = 0;

  constructor(audio: AudioPlayer) {
    this.audio = audio;
    this.waveCanvas = document.getElementById("viz-wave") as HTMLCanvasElement;
    this.specCanvas = document.getElementById("viz-spectrum") as HTMLCanvasElement;
    this.histCanvas = document.getElementById("viz-history") as HTMLCanvasElement;

    this.waveCtx = fitCanvas(this.waveCanvas);
    this.specCtx = fitCanvas(this.specCanvas);
    this.histCtx = fitCanvas(this.histCanvas);

    window.addEventListener("resize", () => {
      this.waveCtx = fitCanvas(this.waveCanvas);
      this.specCtx = fitCanvas(this.specCanvas);
      this.histCtx = fitCanvas(this.histCanvas);
    });

    this.loop();
  }

  private loop = (): void => {
    const playing = this.audio.isPlaying();
    this.drawWave(playing);
    this.drawSpectrum(playing);

    this.frameCount++;
    if (this.frameCount % 6 === 0) {
      const level = playing ? this.averageLevel() : 0;
      this.history.push(level);
      this.history.shift();
      this.drawHistory();
    }

    requestAnimationFrame(this.loop);
  };

  private averageLevel(): number {
    const data = this.audio.getFrequencyData();
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i];
    return sum / data.length / 255;
  }

  private drawWave(playing: boolean): void {
    const ctx = this.waveCtx;
    const { width, height } = this.waveCanvas.getBoundingClientRect();
    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = BLUE_FAINT;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    ctx.strokeStyle = BLUE;
    ctx.lineWidth = 1.5;
    ctx.shadowColor = BLUE;
    ctx.shadowBlur = 6;
    ctx.beginPath();

    if (!playing) {
      // Honest "no signal": a flat centre line, not synthetic wiggle.
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
    } else {
      const data = this.audio.getWaveformData();
      const step = width / data.length;
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128;
        const y = height / 2 + v * (height / 2) * 0.9;
        if (i === 0) ctx.moveTo(0, y);
        else ctx.lineTo(i * step, y);
      }
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  private drawSpectrum(playing: boolean): void {
    const ctx = this.specCtx;
    const { width, height } = this.specCanvas.getBoundingClientRect();
    ctx.clearRect(0, 0, width, height);

    const bars = 32;
    const gap = 2;
    const barWidth = (width - gap * (bars - 1)) / bars;
    const data = playing ? this.audio.getFrequencyData() : null;

    for (let i = 0; i < bars; i++) {
      const value = data ? data[Math.floor((i / bars) * data.length)] / 255 : 0;
      const h = Math.max(1, value * height);
      const x = i * (barWidth + gap);
      const y = height - h;

      ctx.fillStyle = value > 0.02 ? BLUE : BLUE_FAINT;
      if (value > 0.02) {
        ctx.shadowColor = BLUE;
        ctx.shadowBlur = 5;
      }
      ctx.fillRect(x, y, barWidth, h);
      ctx.shadowBlur = 0;
    }
  }

  private drawHistory(): void {
    const ctx = this.histCtx;
    const { width, height } = this.histCanvas.getBoundingClientRect();
    ctx.clearRect(0, 0, width, height);

    const bars = this.history.length;
    const gap = 2;
    const barWidth = (width - gap * (bars - 1)) / bars;

    for (let i = 0; i < bars; i++) {
      const value = this.history[i];
      const h = Math.max(1, value * height);
      const x = i * (barWidth + gap);
      ctx.fillStyle = value > 0.02 ? BLUE_SOFT : BLUE_FAINT;
      ctx.fillRect(x, height - h, barWidth, h);
    }
  }
}
