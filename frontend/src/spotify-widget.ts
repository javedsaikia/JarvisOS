// Now Playing widget — reflects real local Spotify.app state, pushed by
// max/bridge.py's _spotify_poll_loop over the same websocket as
// everything else. Purely presentational; its only "polling" is a local
// requestAnimationFrame tick that interpolates the progress bar between
// the server's ~2s updates so it doesn't visibly stall between them.
import type { ServerMessage } from "./socket";

type SpotifyState = Extract<ServerMessage, { type: "spotify_state" }>;

function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export class SpotifyWidget {
  private card: HTMLElement;
  private status: HTMLElement;
  private art: HTMLImageElement;
  private track: HTMLElement;
  private artist: HTMLElement;
  private progressFill: HTMLElement;
  private positionLabel: HTMLElement;
  private durationLabel: HTMLElement;

  private playing = false;
  private duration = 0;
  private lastPosition = 0;
  private lastUpdateAt = 0;

  constructor() {
    this.card = document.getElementById("spotify-card")!;
    this.status = document.getElementById("spotify-status")!;
    this.art = document.getElementById("spotify-art") as HTMLImageElement;
    this.track = document.getElementById("spotify-track")!;
    this.artist = document.getElementById("spotify-artist")!;
    this.progressFill = document.getElementById("spotify-progress-fill")!;
    this.positionLabel = document.getElementById("spotify-position")!;
    this.durationLabel = document.getElementById("spotify-duration")!;
    requestAnimationFrame(this.tick);
  }

  update(state: SpotifyState): void {
    if (!state.active) {
      this.card.hidden = true;
      this.playing = false;
      return;
    }
    this.card.hidden = false;
    this.status.textContent = state.playing ? "Playing" : "Paused";
    if (state.artwork_url) this.art.src = state.artwork_url;
    this.track.textContent = state.track || "—";
    this.artist.textContent = state.artist || "—";
    this.playing = !!state.playing;
    this.duration = state.duration ?? 0;
    this.lastPosition = state.position ?? 0;
    this.lastUpdateAt = performance.now();
    this.render(this.lastPosition);
  }

  private render(position: number): void {
    const pct = this.duration > 0 ? Math.min(100, (position / this.duration) * 100) : 0;
    this.progressFill.style.width = `${pct}%`;
    this.positionLabel.textContent = formatTime(position);
    this.durationLabel.textContent = formatTime(this.duration);
  }

  private tick = (): void => {
    if (this.playing && !this.card.hidden) {
      const elapsed = (performance.now() - this.lastUpdateAt) / 1000;
      this.render(Math.min(this.duration, this.lastPosition + elapsed));
    }
    requestAnimationFrame(this.tick);
  };
}
