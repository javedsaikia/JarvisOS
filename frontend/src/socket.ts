// Thin WebSocket client for the JARVIS bridge (jarvis/bridge.py).
// Mirrors the message protocol exactly — no logic/decisions happen here,
// just relaying. Routing/cost/confirmation decisions all happen server-side.

export type ServerMessage =
  | { type: "reply"; backend: string; label: string; text: string }
  | { type: "reply_start"; backend: string; label: string }
  | { type: "reply_chunk"; text: string }
  | { type: "reply_end"; backend: string; label: string }
  | { type: "confirm_request"; prompt: string }
  | { type: "audio"; data: string }
  | { type: "muted"; enabled: boolean }
  | { type: "voice_state"; running: boolean; state: string }
  | { type: "error"; message: string }
  | { type: "transcript_turn"; role: "user" | "assistant"; content: string; label: string }
  | {
      type: "spotify_state";
      active: boolean;
      playing?: boolean;
      track?: string;
      artist?: string;
      artwork_url?: string;
      position?: number;
      duration?: number;
    };

type Listener = (msg: ServerMessage) => void;

export class JarvisSocket {
  private ws: WebSocket | null = null;
  private listeners: Listener[] = [];
  private url: string;
  private reconnectDelay = 1000;
  private onStatusChange: (connected: boolean) => void;

  constructor(url: string, onStatusChange: (connected: boolean) => void) {
    this.url = url;
    this.onStatusChange = onStatusChange;
    this.connect();
  }

  private connect(): void {
    const ws = new WebSocket(this.url);
    this.ws = ws;

    ws.onopen = () => {
      this.reconnectDelay = 1000;
      this.onStatusChange(true);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as ServerMessage;
        this.listeners.forEach((fn) => fn(msg));
      } catch {
        // ignore malformed frames
      }
    };

    ws.onclose = () => {
      this.onStatusChange(false);
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, 10000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }

  onMessage(fn: Listener): void {
    this.listeners.push(fn);
  }

  private send(obj: unknown): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
    }
  }

  sendMessage(text: string): void {
    this.send({ type: "message", text });
  }

  sendConfirm(answer: boolean): void {
    this.send({ type: "confirm_response", answer });
  }

  mute(): void {
    this.send({ type: "mute" });
  }

  unmute(): void {
    this.send({ type: "unmute" });
  }

  startVoice(): void {
    this.send({ type: "voice_start" });
  }

  stopVoice(): void {
    this.send({ type: "voice_stop" });
  }

  wakeVoice(): void {
    this.send({ type: "voice_wake" });
  }
}
