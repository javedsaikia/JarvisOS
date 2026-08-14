// Replays the voice loop's speech envelope against the browser's clock.
//
// A spoken reply is played by afplay on the Mac's speakers and never
// reaches this tab, so there is no AnalyserNode to read. What does arrive
// (bridge -> voice_audio) is the real amplitude envelope measured off the
// exact WAV being played, plus the moment playback began. Walking that
// envelope in real time gives the orb and ring genuine per-syllable
// motion for spoken turns.
//
// The clock starts when the message lands rather than at the WAV's own
// timestamp: the two processes are on the same machine and the envelope
// is computed before playback starts, so the offset is the bridge's
// 100ms poll at worst — below the threshold where a viewer could tell the
// orb from the speaker.

const SAMPLE_INTERPOLATION = true;

export class EnvelopePlayer {
  private envelope: number[] = [];
  private hz = 25;
  private startedAt = 0;
  private rafId: number | null = null;
  private onLevel: (amplitude: number) => void;
  private onSpeakingChange: (speaking: boolean) => void;

  constructor(
    onLevel: (amplitude: number) => void,
    onSpeakingChange: (speaking: boolean) => void
  ) {
    this.onLevel = onLevel;
    this.onSpeakingChange = onSpeakingChange;
  }

  get speaking(): boolean {
    return this.rafId !== null;
  }

  /** Start (or replace) playback of an envelope. Replacing matters: a
   * reply is spoken as several per-sentence clips arriving back to back,
   * and the next one should take over immediately rather than queue. */
  play(envelope: number[], hz: number): void {
    if (!envelope.length) return;
    const wasSpeaking = this.speaking;
    this.envelope = envelope;
    this.hz = hz > 0 ? hz : 25;
    this.startedAt = performance.now();
    if (!wasSpeaking) this.onSpeakingChange(true);
    if (this.rafId === null) this.rafId = requestAnimationFrame(this.tick);
  }

  /** Playback ended early — barge-in, or the user hitting Stop. */
  stop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.envelope = [];
    this.onLevel(0);
    this.onSpeakingChange(false);
  }

  private tick = (): void => {
    const elapsed = (performance.now() - this.startedAt) / 1000;
    const position = elapsed * this.hz;
    const index = Math.floor(position);

    if (index >= this.envelope.length) {
      // Ran to the end of the clip: that is the clip finishing normally.
      this.rafId = null;
      this.envelope = [];
      this.onLevel(0);
      this.onSpeakingChange(false);
      return;
    }

    let level = this.envelope[index];
    if (SAMPLE_INTERPOLATION && index + 1 < this.envelope.length) {
      // 25Hz samples would step visibly at 60fps; interpolating between
      // them keeps the motion continuous without inventing detail.
      const next = this.envelope[index + 1];
      level += (next - level) * (position - index);
    }
    this.onLevel(level);
    this.rafId = requestAnimationFrame(this.tick);
  };
}
