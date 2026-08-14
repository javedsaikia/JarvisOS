// Canvas HUD visualizers driven by real measured audio only.
//
// Two sources, never mixed and never faked:
//   - Audio played in this tab: the same AnalyserNode the orb reads, which
//     gives a true waveform and a true frequency spectrum.
//   - Audio played by the voice loop on the Mac's speakers: only an
//     amplitude envelope exists (jarvis/voice_events.py), so these panels
//     show level-over-time and say so, rather than inventing 32 bins of
//     spectrum out of one number.
// With neither playing, they render an explicit flat "no signal" baseline.

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
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

export class Visualizers {
  private waveCtx: CanvasRenderingContext2D;
  private specCtx: CanvasRenderingContext2D;
  private waveCanvas: HTMLCanvasElement;
  private specCanvas: HTMLCanvasElement;
  private caption: HTMLElement | null;
  private audio: AudioPlayer;

  /** Rolling history of real output level, one sample per frame batch.
   * Starts empty and only ever fills from measured amplitude. */
  private history: number[] = new Array(64).fill(0);
  private frameCount = 0;

  // Level relayed from the voice loop while it speaks through the Mac's
  // speakers — real, just measured in another process.
  private externalLevel = 0;
  private externalActive = false;

  constructor(audio: AudioPlayer) {
    this.audio = audio;
    this.waveCanvas = document.getElementById("viz-wave") as HTMLCanvasElement;
    this.specCanvas = document.getElementById("viz-spectrum") as HTMLCanvasElement;
    this.caption = document.getElementById("viz-caption");

    this.waveCtx = fitCanvas(this.waveCanvas);
    this.specCtx = fitCanvas(this.specCanvas);

    window.addEventListener("resize", () => {
      this.waveCtx = fitCanvas(this.waveCanvas);
      this.specCtx = fitCanvas(this.specCanvas);
    });

    this.loop();
  }

  setExternalLevel(level: number): void {
    this.externalLevel = Math.max(0, Math.min(1, level));
  }

  setExternalActive(active: boolean): void {
    this.externalActive = active;
    if (!active) this.externalLevel = 0;
    this.updateCaption();
  }

  private updateCaption(): void {
    if (!this.caption) return;
    if (this.audio.isPlaying()) this.caption.textContent = "Output spectrum";
    else if (this.externalActive) this.caption.textContent = "Speech level · voice loop";
    else this.caption.textContent = "No signal";
  }

  private loop = (): void => {
    const localPlaying = this.audio.isPlaying();
    this.drawWave(localPlaying);
    this.drawSpectrum(localPlaying);

    this.frameCount++;
    if (this.frameCount % 4 === 0) {
      let level = 0;
      if (localPlaying) level = this.averageLevel();
      else if (this.externalActive) level = this.externalLevel;
      this.history.push(level);
      this.history.shift();
      this.updateCaption();
    }

    requestAnimationFrame(this.loop);
  };

  private averageLevel(): number {
    const data = this.audio.getFrequencyData();
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i];
    return sum / data.length / 255;
  }

  private drawWave(localPlaying: boolean): void {
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

    if (localPlaying) {
      const data = this.audio.getWaveformData();
      const step = width / data.length;
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128;
        const y = height / 2 + v * (height / 2) * 0.9;
        if (i === 0) ctx.moveTo(0, y);
        else ctx.lineTo(i * step, y);
      }
    } else if (this.externalActive) {
      // No per-sample waveform exists for speakers audio, so this is the
      // measured envelope mirrored about the centre line — the shape of
      // the speech, drawn from real levels rather than invented samples.
      const step = width / this.history.length;
      for (let i = 0; i < this.history.length; i++) {
        const y = height / 2 - (this.history[i] * height) / 2;
        if (i === 0) ctx.moveTo(0, y);
        else ctx.lineTo(i * step, y);
      }
      for (let i = this.history.length - 1; i >= 0; i--) {
        ctx.lineTo(i * step, height / 2 + (this.history[i] * height) / 2);
      }
    } else {
      // Honest "no signal": a flat centre line, not synthetic wiggle.
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  private drawSpectrum(localPlaying: boolean): void {
    const ctx = this.specCtx;
    const { width, height } = this.specCanvas.getBoundingClientRect();
    ctx.clearRect(0, 0, width, height);

    const bars = localPlaying ? 32 : this.history.length;
    const gap = 2;
    const barWidth = (width - gap * (bars - 1)) / bars;
    const data = localPlaying ? this.audio.getFrequencyData() : null;

    for (let i = 0; i < bars; i++) {
      // Frequency bins while this tab has the audio; level-over-time while
      // the voice loop has it. Same panel, different real measurement.
      const value = data
        ? data[Math.floor((i / bars) * data.length)] / 255
        : this.history[i];
      const h = Math.max(1, value * height);
      const x = i * (barWidth + gap);

      const lit = value > 0.02;
      ctx.fillStyle = lit ? (data ? BLUE : BLUE_SOFT) : BLUE_FAINT;
      if (lit) {
        ctx.shadowColor = BLUE;
        ctx.shadowBlur = 5;
      }
      ctx.fillRect(x, height - h, barWidth, h);
      ctx.shadowBlur = 0;
    }
  }
}
