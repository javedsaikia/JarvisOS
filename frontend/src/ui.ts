export type UIStatus = "idle" | "listening" | "thinking" | "speaking" | "error" | "offline";

const STATUS_TEXT: Record<UIStatus, string> = {
  idle: "Idle",
  listening: "Listening",
  thinking: "Thinking…",
  speaking: "Speaking",
  error: "Error",
  offline: "Disconnected — reconnecting…",
};

export class UI {
  private transcript: HTMLElement;
  private input: HTMLInputElement;
  private sendBtn: HTMLButtonElement;
  private muteBtn: HTMLButtonElement;
  private stopBtn: HTMLButtonElement;
  private voiceBtn: HTMLButtonElement;
  private wakeBtn: HTMLButtonElement;
  private forceLocalBtn: HTMLButtonElement;
  private forceClaudeBtn: HTMLButtonElement;
  private statusDot: HTMLElement;
  private statusLabel: HTMLElement;
  private banner: HTMLElement;

  // HUD readout panel — purely additive, independent of status-dot/statusLabel above.
  private readoutLink: HTMLElement;
  private readoutBackend: HTMLElement;
  private readoutAudio: HTMLElement;
  private readoutAudioBar: HTMLElement;
  private readoutSession: HTMLElement;
  private readoutMessages: HTMLElement;

  private muted = false;
  private awaitingReply = false;
  private awaitingConfirm = false;
  private voiceRunning = false;
  private voiceBusy = false;

  onSend: ((text: string) => void) | null = null;
  onConfirm: ((answer: boolean) => void) | null = null;
  onMuteToggle: ((mute: boolean) => void) | null = null;
  onStop: (() => void) | null = null;
  onVoiceToggle: ((running: boolean) => void) | null = null;
  onWake: (() => void) | null = null;

  constructor() {
    this.transcript = document.getElementById("transcript")!;
    this.input = document.getElementById("text-input") as HTMLInputElement;
    this.sendBtn = document.getElementById("send-btn") as HTMLButtonElement;
    this.muteBtn = document.getElementById("mute-btn") as HTMLButtonElement;
    this.stopBtn = document.getElementById("stop-btn") as HTMLButtonElement;
    this.voiceBtn = document.getElementById("voice-btn") as HTMLButtonElement;
    this.wakeBtn = document.getElementById("wake-btn") as HTMLButtonElement;
    this.forceLocalBtn = document.getElementById("force-local-btn") as HTMLButtonElement;
    this.forceClaudeBtn = document.getElementById("force-claude-btn") as HTMLButtonElement;
    this.statusDot = document.getElementById("status-dot")!;
    this.statusLabel = document.getElementById("status-label")!;
    this.banner = document.getElementById("connection-banner")!;

    this.readoutLink = document.getElementById("readout-link")!;
    this.readoutBackend = document.getElementById("readout-backend")!;
    this.readoutAudio = document.getElementById("readout-audio")!;
    this.readoutAudioBar = document.getElementById("readout-bar-fill")!;
    this.readoutSession = document.getElementById("readout-session")!;
    this.readoutMessages = document.getElementById("readout-messages")!;

    this.sendBtn.addEventListener("click", () => this.trySend());
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.trySend();
    });
    this.muteBtn.addEventListener("click", () => {
      this.muted = !this.muted;
      this.muteBtn.textContent = this.muted ? "🔇" : "🔊";
      this.onMuteToggle?.(this.muted);
    });
    this.stopBtn.addEventListener("click", () => {
      this.onStop?.();
    });
    this.voiceBtn.addEventListener("click", () => {
      if (this.voiceBusy) return;
      this.voiceBusy = true;
      this.renderVoiceButton();
      this.onVoiceToggle?.(!this.voiceRunning);
    });
    this.wakeBtn.addEventListener("click", () => {
      this.onWake?.();
    });
    this.forceLocalBtn.addEventListener("click", () => this.trySend("/ollama "));
    this.forceClaudeBtn.addEventListener("click", () => this.trySend("/claude "));
    this.renderVoiceButton();
  }

  private trySend(prefix = ""): void {
    if (this.awaitingReply || this.awaitingConfirm) return;
    const text = this.input.value.trim();
    if (!text) return;
    this.appendUserMessage(text);
    this.input.value = "";
    this.setAwaitingReply(true);
    this.onSend?.(prefix + text);
  }

  setAwaitingReply(waiting: boolean): void {
    this.awaitingReply = waiting;
    this.updateInputEnabled();
    if (waiting) this.setStatus("thinking");
  }

  private updateInputEnabled(): void {
    const disabled = this.awaitingReply || this.awaitingConfirm;
    this.input.disabled = disabled;
    this.sendBtn.disabled = disabled;
    this.forceLocalBtn.disabled = disabled;
    this.forceClaudeBtn.disabled = disabled;
  }

  private renderVoiceButton(): void {
    if (this.voiceBusy) {
      this.voiceBtn.textContent = this.voiceRunning ? "Stopping…" : "Starting…";
      this.voiceBtn.disabled = true;
      this.wakeBtn.disabled = true;
      return;
    }
    this.voiceBtn.disabled = false;
    this.wakeBtn.disabled = false;
    this.voiceBtn.textContent = this.voiceRunning ? "Stop Voice" : "Start Voice";
    this.voiceBtn.title = this.voiceRunning
      ? "Stop the always-on voice loop"
      : "Start the always-on voice loop";
    this.wakeBtn.textContent = "Wake";
    this.wakeBtn.title = this.voiceRunning
      ? "Wake the voice loop and begin listening now"
      : "Start the voice loop, then wake it";
  }

  appendUserMessage(text: string): void {
    const el = document.createElement("div");
    el.className = "msg msg-user";
    el.textContent = text;
    this.transcript.appendChild(el);
    this.scrollToBottom();
  }

  // "[Ollama]" -> "ollama", "[Claude Code]" -> "claude-code" — used as a
  // data-backend attribute so the transcript can color-code each reply by
  // which backend actually answered, the same way the tier strip already does.
  private backendKey(label: string): string {
    return label.replace(/[[\]]/g, "").toLowerCase().replace(/\s+/g, "-");
  }

  appendReply(label: string, text: string): void {
    this.setAwaitingReply(false);
    const el = document.createElement("div");
    el.className = "msg msg-reply";
    el.dataset.backend = this.backendKey(label);
    const labelEl = document.createElement("span");
    labelEl.className = "msg-label";
    labelEl.textContent = label;
    const textEl = document.createElement("span");
    textEl.className = "msg-text";
    el.appendChild(labelEl);
    el.appendChild(textEl);
    this.transcript.appendChild(el);
    this.scrollToBottom();
    this.setStatusFromLabel();
    this.typewriterReveal(textEl, text);
  }

  // Backends that don't stream token-by-token (tools, Claude Code, Sarvam,
  // and anything relayed from voice_loop.py — see appendVoiceTurn) used to
  // dump their full reply into the DOM instantly, which read as "typed
  // ahead of" the TTS audio rather than alongside it. This paces the
  // reveal at a fixed rate approximating natural speech, so text lands on
  // screen roughly in step with JARVIS actually saying it. Streamed Ollama
  // replies (appendReplyChunk) already arrive incrementally in real
  // generation time and are left alone — they don't need a synthetic pace.
  private static readonly TYPE_CHARS_PER_SEC = 16;

  // Chrome throttles setTimeout heavily in backgrounded tabs (often to
  // ~1/sec or less) — for a voice assistant meant to run hands-free while
  // the tab isn't focused, that made replies look permanently "stuck"
  // mid-word rather than just paced. Skips the animation entirely (full
  // text immediately) whenever the tab isn't visible, and jumps any
  // reveal already in progress to completion the moment it's backgrounded
  // mid-animation, so nothing is ever left dangling on an unfocused tab.
  private typewriterReveal(textEl: HTMLElement, fullText: string): void {
    if (!fullText) return;
    if (document.hidden) {
      textEl.textContent = fullText;
      return;
    }
    const intervalMs = 1000 / UI.TYPE_CHARS_PER_SEC;
    let i = 0;
    const finish = () => {
      document.removeEventListener("visibilitychange", onHidden);
      textEl.textContent = fullText;
      this.scrollToBottom();
    };
    const onHidden = () => {
      if (document.hidden) finish();
    };
    document.addEventListener("visibilitychange", onHidden);
    const step = () => {
      if (document.hidden) {
        finish();
        return;
      }
      i++;
      textEl.textContent = fullText.slice(0, i);
      this.scrollToBottom();
      if (i < fullText.length) {
        setTimeout(step, intervalMs);
      } else {
        document.removeEventListener("visibilitychange", onHidden);
      }
    };
    step();
  }

  // Renders a turn relayed from voice_loop.py (a separate OS process —
  // see bridge.py's _transcript_poll_loop) into the same Comms Log as
  // web-typed turns. User turns appear immediately (the user already said
  // it — nothing to "type" on their behalf); assistant turns get the same
  // typewriter pacing as appendReply, since voice-loop audio plays on the
  // Mac's speakers directly and never reaches the browser, so there's no
  // real audio duration here to sync against — this is an estimate.
  appendVoiceTurn(role: "user" | "assistant", label: string, text: string): void {
    if (role === "user") {
      this.appendUserMessage(text);
      return;
    }
    const el = document.createElement("div");
    el.className = "msg msg-reply";
    el.dataset.backend = this.backendKey(label);
    const labelEl = document.createElement("span");
    labelEl.className = "msg-label";
    labelEl.textContent = label;
    const textEl = document.createElement("span");
    textEl.className = "msg-text";
    el.appendChild(labelEl);
    el.appendChild(textEl);
    this.transcript.appendChild(el);
    this.scrollToBottom();
    this.typewriterReveal(textEl, text);
  }

  // --- Streaming reply lifecycle: begin/chunk/finish, used for backends
  // that stream token-by-token (currently only plain Ollama chat — see
  // bridge.py's reply_start/reply_chunk/reply_end). Same bubble markup as
  // appendReply above, just filled in incrementally instead of all at once,
  // so text appears as it's generated instead of only once the full reply
  // and its TTS clip are both ready. ---

  private streamTextEl: HTMLElement | null = null;

  beginReplyStream(label: string): void {
    const el = document.createElement("div");
    el.className = "msg msg-reply";
    el.dataset.backend = this.backendKey(label);
    const labelEl = document.createElement("span");
    labelEl.className = "msg-label";
    labelEl.textContent = label;
    const textEl = document.createElement("span");
    textEl.className = "msg-text";
    el.appendChild(labelEl);
    el.appendChild(textEl);
    this.transcript.appendChild(el);
    this.streamTextEl = textEl;
    this.scrollToBottom();
  }

  appendReplyChunk(text: string): void {
    if (!this.streamTextEl) return;
    this.streamTextEl.textContent += text;
    this.scrollToBottom();
  }

  finishReplyStream(): void {
    this.streamTextEl = null;
    this.setAwaitingReply(false);
    this.setStatusFromLabel();
  }

  appendConfirmPrompt(prompt: string): void {
    this.awaitingConfirm = true;
    this.updateInputEnabled();

    const el = document.createElement("div");
    el.className = "msg msg-confirm";

    const promptEl = document.createElement("div");
    promptEl.className = "confirm-prompt";
    promptEl.textContent = `[Router] ${prompt}`;
    el.appendChild(promptEl);

    const btnRow = document.createElement("div");
    btnRow.className = "confirm-buttons";

    const yesBtn = document.createElement("button");
    yesBtn.textContent = "Yes";
    yesBtn.className = "confirm-yes";
    const noBtn = document.createElement("button");
    noBtn.textContent = "No";
    noBtn.className = "confirm-no";

    const resolve = (answer: boolean) => {
      yesBtn.disabled = true;
      noBtn.disabled = true;
      promptEl.textContent += answer ? "  →  Yes" : "  →  No";
      this.awaitingConfirm = false;
      this.updateInputEnabled();
      this.setAwaitingReply(true);
      this.onConfirm?.(answer);
    };
    yesBtn.addEventListener("click", () => resolve(true));
    noBtn.addEventListener("click", () => resolve(false));

    btnRow.appendChild(yesBtn);
    btnRow.appendChild(noBtn);
    el.appendChild(btnRow);

    this.transcript.appendChild(el);
    this.scrollToBottom();
  }

  appendError(message: string): void {
    const el = document.createElement("div");
    el.className = "msg msg-error";
    el.textContent = message;
    this.transcript.appendChild(el);
    this.scrollToBottom();
  }

  private setStatusFromLabel(): void {
    // A completed turn in an active voice session means JARVIS is ready for
    // the next utterance, not idle and waiting for another wake phrase.
    this.setStatus(this.voiceRunning ? "listening" : "idle");
  }

  setStatus(status: UIStatus): void {
    this.statusDot.className = status;
    this.statusLabel.textContent = STATUS_TEXT[status];
  }

  setConnectionState(connected: boolean): void {
    this.banner.classList.toggle("hidden", connected);
    this.banner.textContent = "Disconnected from JARVIS bridge — reconnecting…";
    if (!connected) this.setStatus("offline");
  }

  setVoiceState(running: boolean, busy = false): void {
    this.voiceRunning = running;
    this.voiceBusy = busy;
    this.setStatus(running ? "listening" : "idle");
    this.renderVoiceButton();
  }

  clearVoiceBusy(): void {
    this.voiceBusy = false;
    this.renderVoiceButton();
  }

  // --- HUD readout panel — all values passed in are real data from main.ts
  // (connection state, actual backend labels, live audio amplitude, elapsed
  // time), nothing fabricated. ---

  setLink(url: string, connected: boolean): void {
    this.readoutLink.textContent = connected ? url : `${url} (offline)`;
  }

  setBackendReadout(label: string): void {
    this.readoutBackend.textContent = label;
  }

  setAudioLevel(amplitude: number): void {
    const pct = Math.round(Math.max(0, Math.min(1, amplitude)) * 100);
    this.readoutAudio.textContent = `${pct}%`;
    this.readoutAudioBar.style.width = `${pct}%`;
  }

  setSessionTime(seconds: number): void {
    const m = Math.floor(seconds / 60)
      .toString()
      .padStart(2, "0");
    const s = Math.floor(seconds % 60)
      .toString()
      .padStart(2, "0");
    this.readoutSession.textContent = `${m}:${s}`;
  }

  setMessageCount(count: number): void {
    this.readoutMessages.textContent = String(count);
  }

  private scrollToBottom(): void {
    this.transcript.scrollTop = this.transcript.scrollHeight;
  }
}
