// Decodes and plays TTS audio the bridge forwards from Piper, and exposes
// real-time amplitude from a Web Audio AnalyserNode so the orb can react to
// actual audio energy while Orin is speaking (not a fake/synthetic pulse).

export class AudioPlayer {
  private ctx: AudioContext;
  private analyser: AnalyserNode;
  private dataArray: Uint8Array<ArrayBuffer>;
  private waveArray: Uint8Array<ArrayBuffer>;
  private source: AudioBufferSourceNode | null = null;
  private rafId: number | null = null;
  private onAmplitude: (amplitude: number) => void;
  private onPlaybackEnd: () => void;
  // A reply can now arrive as several per-sentence clips (streamed turns)
  // instead of one clip for the whole response. Queuing plays them
  // back-to-back in order; without it, each new clip's playBase64Wav would
  // call stop() and cut off whichever sentence was still playing.
  private queue: string[] = [];
  private playing = false;

  constructor(onAmplitude: (amplitude: number) => void, onPlaybackEnd: () => void) {
    this.onAmplitude = onAmplitude;
    this.onPlaybackEnd = onPlaybackEnd;
    this.ctx = new AudioContext();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 256;
    this.analyser.smoothingTimeConstant = 0.6;
    this.dataArray = new Uint8Array(new ArrayBuffer(this.analyser.frequencyBinCount));
    this.waveArray = new Uint8Array(new ArrayBuffer(this.analyser.fftSize));
  }

  // --- Accessors for the HUD visualizers. These expose the SAME real
  // AnalyserNode the orb already reacts to — no synthesized/fake signal.
  // isPlaying() lets visualizers render an honest "no signal" idle state
  // instead of pretending to show audio that isn't there. ---

  isPlaying(): boolean {
    return this.source !== null;
  }

  /** Live frequency-domain spectrum, 0-255 per bin. */
  getFrequencyData(): Uint8Array<ArrayBuffer> {
    this.analyser.getByteFrequencyData(this.dataArray);
    return this.dataArray;
  }

  /** Live time-domain waveform, 0-255 centered on 128. */
  getWaveformData(): Uint8Array<ArrayBuffer> {
    this.analyser.getByteTimeDomainData(this.waveArray);
    return this.waveArray;
  }

  /** Queue a clip for playback. Plays immediately if nothing is already
   * playing; otherwise waits its turn behind whatever's queued ahead of it. */
  enqueueBase64Wav(base64: string): void {
    this.queue.push(base64);
    if (!this.playing) void this._playNext();
  }

  private async _playNext(): Promise<void> {
    const next = this.queue.shift();
    if (!next) {
      this.playing = false;
      this.stopAnalysisLoop();
      this.onPlaybackEnd();
      return;
    }
    this.playing = true;

    if (this.ctx.state === "suspended") {
      await this.ctx.resume();
    }

    const binary = atob(next);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    let audioBuffer: AudioBuffer;
    try {
      audioBuffer = await this.ctx.decodeAudioData(bytes.buffer);
    } catch {
      // Skip an unplayable clip rather than stalling the rest of the queue.
      void this._playNext();
      return;
    }

    const source = this.ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);
    source.onended = () => {
      this.source = null;
      void this._playNext();
    };
    this.source = source;
    source.start();

    this.startAnalysisLoop();
  }

  stop(): void {
    this.queue = [];
    if (this.source) {
      try {
        this.source.onended = null;
        this.source.stop();
      } catch {
        // already stopped
      }
      this.source = null;
    }
    this.playing = false;
    this.stopAnalysisLoop();
  }

  private startAnalysisLoop(): void {
    const loop = () => {
      this.analyser.getByteFrequencyData(this.dataArray);
      let sum = 0;
      let peak = 0;
      for (let i = 0; i < this.dataArray.length; i++) {
        const v = this.dataArray[i];
        sum += v;
        if (v > peak) peak = v;
      }
      const average = sum / this.dataArray.length / 255;
      const peakNorm = peak / 255;
      // A plain full-spectrum average dilutes typical speech down to
      // ~0.05-0.15 (most of the 128 bins are near-silent for a voice
      // signal), which barely moved the orb — verified live, motion was
      // there but imperceptible. Blending in peak (catches syllable
      // onsets) and a gain curve makes normal speech actually swing
      // through a visible range without clipping on loud moments.
      const raw = average * 0.4 + peakNorm * 0.6;
      const boosted = Math.min(1, Math.pow(raw, 0.6) * 1.6);
      this.onAmplitude(boosted);
      this.rafId = requestAnimationFrame(loop);
    };
    loop();
  }

  private stopAnalysisLoop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.onAmplitude(0);
  }
}
