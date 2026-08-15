export type UIStatus = "idle" | "listening" | "thinking" | "speaking" | "error" | "offline";

const STATUS_TEXT: Record<UIStatus, string> = {
  idle: "Idle",
  listening: "Listening",
  thinking: "Thinking…",
  speaking: "Speaking",
  error: "Error",
  // Kept short: this sits in the status card's header row, where a longer
  // phrase wraps and shoves the card's own title onto two lines.
  offline: "Reconnecting…",
};

// The caption under the wordmark in the middle of the reticle. Shorter and
// louder than the status card's wording — it's read at a glance from across
// the room, which is the whole point of putting it in the core.
const CORE_TEXT: Record<UIStatus, string> = {
  idle: "CORE ONLINE",
  listening: "LISTENING",
  thinking: "PROCESSING",
  speaking: "SPEAKING",
  error: "FAULT",
  offline: "LINK LOST",
};

export class UI {
  private transcript: HTMLElement;
  private input: HTMLInputElement;
  private sendBtn: HTMLButtonElement;
  private muteBtn: HTMLButtonElement | null;
  private stopBtn: HTMLButtonElement;
  private voiceBtn: HTMLButtonElement;
  private wakeBtn: HTMLButtonElement | null;
  private screenBtn: HTMLButtonElement;
  private micBtn: HTMLButtonElement;
  private forceLocalBtn: HTMLButtonElement | null;
  private forceClaudeBtn: HTMLButtonElement | null;
  private statusDot: HTMLElement;
  private statusLabel: HTMLElement;
  private banner: HTMLElement;

  // Command-rail button internals — each large button carries its own
  // label/sub-label/indicator rather than just swapping its text.
  private voiceLabel: HTMLElement;
  private voiceSub: HTMLElement;
  private muteLabel: HTMLElement | null;
  private screenLabel: HTMLElement;
  private screenStatus: HTMLElement;
  private screenImg: HTMLImageElement;
  private screenPlaceholder: HTMLElement;
  private coreState: HTMLElement;
  private modelRoles: HTMLElement;
  private modelsStatus: HTMLElement;
  private frontCard: HTMLElement | null;
  private frontShutter: HTMLButtonElement | null;
  private memoryFacts: HTMLElement | null;
  private memoryStatus: HTMLElement | null;
  private micLabel: HTMLElement;

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
  private screenFeedOn = true;
  private micMuted = false;
  private connected = true;

  onSend: ((text: string) => void) | null = null;
  onConfirm: ((answer: boolean) => void) | null = null;
  onMuteToggle: ((mute: boolean) => void) | null = null;
  onStop: (() => void) | null = null;
  onVoiceToggle: ((running: boolean) => void) | null = null;
  onWake: (() => void) | null = null;
  onScreenToggle: ((enabled: boolean) => void) | null = null;
  onModelChange: ((role: string, model: string) => void) | null = null;
  onMicToggle: ((muted: boolean) => void) | null = null;

  constructor() {
    this.transcript = document.getElementById("transcript")!;
    this.input = document.getElementById("text-input") as HTMLInputElement;
    this.sendBtn = document.getElementById("send-btn") as HTMLButtonElement;
    this.muteBtn = document.getElementById("mute-btn") as HTMLButtonElement | null;
    this.stopBtn = document.getElementById("stop-btn") as HTMLButtonElement;
    this.voiceBtn = document.getElementById("voice-btn") as HTMLButtonElement;
    this.wakeBtn = document.getElementById("wake-btn") as HTMLButtonElement | null;
    this.screenBtn = document.getElementById("screen-btn") as HTMLButtonElement;
    this.micBtn = document.getElementById("mic-btn") as HTMLButtonElement;
    this.forceLocalBtn = document.getElementById("force-local-btn") as HTMLButtonElement | null;
    this.forceClaudeBtn = document.getElementById("force-claude-btn") as HTMLButtonElement | null;
    this.statusDot = document.getElementById("status-dot")!;
    this.statusLabel = document.getElementById("status-label")!;
    this.banner = document.getElementById("connection-banner")!;

    this.voiceLabel = document.getElementById("voice-label")!;
    this.voiceSub = document.getElementById("voice-sub")!;
    this.muteLabel = document.getElementById("mute-label");
    this.screenLabel = document.getElementById("screen-label")!;
    this.screenStatus = document.getElementById("screen-status")!;
    this.screenImg = document.getElementById("screen-frame") as HTMLImageElement;
    this.screenPlaceholder = document.getElementById("screen-placeholder")!;
    this.coreState = document.getElementById("core-state")!;
    this.modelRoles = document.getElementById("model-roles")!;
    this.modelsStatus = document.getElementById("models-status")!;
    this.frontCard = document.getElementById("front-card");
    this.frontShutter = document.getElementById("front-shutter") as HTMLButtonElement | null;
    this.memoryFacts = document.getElementById("memory-facts");
    this.memoryStatus = document.getElementById("memory-status");
    this.micLabel = document.getElementById("mic-label")!;

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
    this.muteBtn?.addEventListener("click", () => {
      this.muted = !this.muted;
      this.renderMuteButton();
      this.onMuteToggle?.(this.muted);
    });
    this.micBtn.addEventListener("click", () => {
      // Optimistic only for the label; setMicState() below is what
      // actually confirms it, once the voice loop has closed the device.
      this.onMicToggle?.(!this.micMuted);
    });
    this.screenBtn.addEventListener("click", () => {
      this.setScreenFeedState(!this.screenFeedOn);
      this.onScreenToggle?.(this.screenFeedOn);
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
    this.wakeBtn?.addEventListener("click", () => {
      this.onWake?.();
    });
    this.forceLocalBtn?.addEventListener("click", () => this.trySend("/ollama "));
    this.forceClaudeBtn?.addEventListener("click", () => this.trySend("/claude "));
    this.frontShutter?.addEventListener("click", () => {
      const open = this.frontCard?.classList.toggle("unlocked") ?? false;
      this.frontShutter?.setAttribute("aria-expanded", open ? "true" : "false");
      this.modelsStatus.textContent = open ? "Inspecting" : "Sealed";
    });
    this.frontCard?.addEventListener("mouseenter", () => {
      if (!this.frontCard?.classList.contains("unlocked")) {
        this.modelsStatus.textContent = "Inspecting";
      }
    });
    this.frontCard?.addEventListener("mouseleave", () => {
      if (!this.frontCard?.classList.contains("unlocked")) {
        this.modelsStatus.textContent = "Sealed";
      }
    });
    this.renderVoiceButton();
    this.renderMuteButton();
  }

  /** Vendor names stay off the glass unless Front is open. Tools keep
   *  their own labels because those are capabilities, not models. */
  static publicLabel(label: string): string {
    const key = label.replace(/[[\]]/g, "").toLowerCase().replace(/\s+/g, "-");
    const tools = new Set([
      "calendar", "notes", "email", "github", "youtube", "drive", "cal",
      "spotify", "files", "shell", "screen", "browser", "location", "call", "news",
    ]);
    if (tools.has(key)) return label;
    if (key === "claude-code" || key === "agent") return "[Agent]";
    return "[Max]";
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
    const disabled = this.awaitingReply || this.awaitingConfirm || !this.connected;
    this.input.disabled = disabled;
    this.sendBtn.disabled = disabled;
    if (this.forceLocalBtn) this.forceLocalBtn.disabled = disabled;
    if (this.forceClaudeBtn) this.forceClaudeBtn.disabled = disabled;
  }

  private renderVoiceButton(): void {
    if (this.voiceBusy) {
      // Models load on a cold start, which takes tens of seconds — the
      // button says so rather than looking like a dead control.
      this.voiceLabel.textContent = this.voiceRunning ? "Stopping…" : "Starting…";
      this.voiceSub.textContent = this.voiceRunning
        ? "Shutting down"
        : "Loading models";
      this.voiceBtn.classList.add("busy");
      this.voiceBtn.disabled = true;
      if (this.wakeBtn) this.wakeBtn.disabled = true;
      return;
    }
    this.voiceBtn.classList.remove("busy");
    this.voiceBtn.classList.toggle("live", this.voiceRunning);
    this.voiceBtn.disabled = !this.connected;
    if (this.wakeBtn) this.wakeBtn.disabled = !this.connected;
    this.voiceLabel.textContent = this.voiceRunning ? "Stop Voice" : "Start Voice";
    this.voiceSub.textContent = this.voiceRunning ? "Loop online" : "Loop offline";
    this.voiceBtn.title = this.voiceRunning
      ? "Stop the always-on voice loop"
      : "Start the always-on voice loop";
    if (this.wakeBtn) this.wakeBtn.title = this.voiceRunning
      ? "Wake the voice loop and begin listening now"
      : "Start the voice loop, then wake it";
  }

  private renderMuteButton(): void {
    // The sound toggle was removed from the rail: it was constantly
    // mistaken for the microphone mute sitting next to it, and Stop
    // already silences a reply in progress. The handler stays so a
    // layout that wants the button back needs no code change.
    if (!this.muteBtn || !this.muteLabel) return;
    this.muteBtn?.classList.toggle("off", this.muted);
    this.muteLabel.textContent = this.muted ? "Muted" : "Sound On";
    if (this.muteBtn) this.muteBtn.title = this.muted
      ? "Unmute spoken replies"
      : "Mute spoken replies";
  }

  /** Live screen card. `enabled` reflects the bridge's actual capture
   * state, so the button can't drift out of sync with what is really
   * being captured (the config can veto it server-side). */
  setScreenFeedState(enabled: boolean): void {
    this.screenFeedOn = enabled;
    this.screenBtn.classList.toggle("off", !enabled);
    this.screenLabel.textContent = enabled ? "Screen On" : "Screen Off";
    if (!enabled) {
      this.screenStatus.textContent = "Off";
      this.screenImg.removeAttribute("src");
      this.screenImg.classList.remove("visible");
      this.screenPlaceholder.textContent = "Screen feed off";
      this.screenPlaceholder.classList.remove("error");
    } else if (!this.screenImg.getAttribute("src")) {
      this.screenStatus.textContent = "Standby";
      this.screenPlaceholder.textContent = "Awaiting first capture…";
    }
  }

  /** Render the model picker from the bridge's catalog.
   *
   * One row per role, because "which model" is not one choice: the spoken
   * loop wants the fastest thing available and a typed question usually
   * wants the best one. Models whose API key is missing stay visible but
   * disabled, with the key name shown — a picker that silently hides an
   * option gives no clue what to do about it.
   */
  setModels(
    models: {
      id: string; label: string; local: boolean; available: boolean; needs_key: string;
    }[],
    roles: Record<string, string>,
    roleLabels: Record<string, string>
  ): void {
    this.modelRoles.replaceChildren();

    for (const [role, description] of Object.entries(roleLabels)) {
      const row = document.createElement("label");
      row.className = "model-row";
      row.title = description;

      const name = document.createElement("span");
      name.className = "model-role";
      name.textContent = role;

      const select = document.createElement("select");
      select.className = "model-select";
      for (const model of models) {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = model.available
          ? model.label
          : `${model.label} — needs ${model.needs_key}`;
        option.disabled = !model.available;
        option.selected = roles[role] === model.id;
        select.appendChild(option);
      }
      select.addEventListener("change", () => this.onModelChange?.(role, select.value));

      row.appendChild(name);
      row.appendChild(select);
      this.modelRoles.appendChild(row);
    }

    const inspecting = Boolean(
      this.frontCard?.classList.contains("unlocked")
      || this.frontCard?.matches(":hover")
      || this.frontCard?.contains(document.activeElement)
    );
    this.modelsStatus.textContent = inspecting ? "Inspecting" : "Sealed";
    this.modelsStatus.classList.remove("cloud");
  }

  setMemory(facts: string[], style: string[] = []): void {
    if (!this.memoryFacts || !this.memoryStatus) return;
    this.memoryFacts.replaceChildren();
    const items = [...facts, ...style.map((s) => s)];
    if (!items.length) {
      const empty = document.createElement("li");
      empty.className = "memory-empty";
      empty.textContent = "Nothing stored yet. Say remember this…";
      this.memoryFacts.appendChild(empty);
      this.memoryStatus.textContent = "Empty";
      return;
    }
    for (const fact of items.slice(0, 12)) {
      const li = document.createElement("li");
      li.textContent = fact;
      this.memoryFacts.appendChild(li);
    }
    this.memoryStatus.textContent = String(items.length);
  }

  /** Microphone state, as reported by the voice loop actually opening or
   * closing the device — not by the button being clicked. For a privacy
   * control the UI must show what is true, not what was asked for.
   */
  setMicState(muted: boolean): void {
    this.micMuted = muted;
    this.micBtn.classList.toggle("off", muted);
    this.micLabel.textContent = muted ? "Mic Off" : "Mic On";
    this.micBtn.title = muted
      ? "Microphone is closed — Max hears nothing. Click to listen again."
      : "Privacy: close the microphone completely — Max stops listening";
    document.documentElement.dataset.mic = muted ? "off" : "on";
  }

  /** Blank the HUD while the backend screenshots the desktop.
   *
   * A web page cannot minimise its own window, so "hiding" is the most a
   * browser front end can do — the CSS drops every HUD layer to nothing
   * and leaves one small caption, so a capture that does include this
   * window shows an almost-empty screen instead of a wall of glowing orb.
   * Normally this never fires: the native capture path composites the
   * screen without this window in the first place.
   */
  setCapturing(capturing: boolean): void {
    document.documentElement.dataset.capturing = capturing ? "on" : "off";
  }

  setScreenFrame(base64Jpeg: string): void {
    // A frame already in flight when the feed is switched off would
    // otherwise land afterwards and quietly flip the card back to "Live"
    // — showing a capture taken after the user said stop.
    if (!this.screenFeedOn) return;
    this.screenImg.src = `data:image/jpeg;base64,${base64Jpeg}`;
    this.screenImg.classList.add("visible");
    this.screenPlaceholder.classList.remove("error");
    this.screenStatus.textContent = "Live";
  }

  setScreenError(message: string): void {
    this.screenImg.removeAttribute("src");
    this.screenImg.classList.remove("visible");
    this.screenStatus.textContent = "Blocked";
    this.screenPlaceholder.textContent = message;
    this.screenPlaceholder.classList.add("error");
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
    const face = UI.publicLabel(label);
    el.dataset.backend = this.backendKey(face);
    const labelEl = document.createElement("span");
    labelEl.className = "msg-label";
    labelEl.textContent = face;
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
  // screen roughly in step with Max actually saying it. Streamed Ollama
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
    const key = `${role}:${label}:${text}`;
    if (key === this.lastVoiceKey) return;
    this.lastVoiceKey = key;
    if (role === "user") {
      this.appendUserMessage(text);
      return;
    }
    const el = document.createElement("div");
    el.className = "msg msg-reply";
    const face = UI.publicLabel(label);
    el.dataset.backend = this.backendKey(face);
    const labelEl = document.createElement("span");
    labelEl.className = "msg-label";
    labelEl.textContent = face;
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
  private lastVoiceKey = "";

  beginReplyStream(label: string): void {
    const el = document.createElement("div");
    el.className = "msg msg-reply";
    const face = UI.publicLabel(label);
    el.dataset.backend = this.backendKey(face);
    const labelEl = document.createElement("span");
    labelEl.className = "msg-label";
    labelEl.textContent = face;
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
    // A completed turn in an active voice session means Max is ready for
    // the next utterance, not idle and waiting for another wake phrase.
    this.setStatus(this.voiceRunning ? "listening" : "idle");
  }

  setStatus(status: UIStatus): void {
    this.statusDot.className = status;
    this.statusLabel.textContent = STATUS_TEXT[status];
    this.coreState.textContent = CORE_TEXT[status];
    // Published on the root element so the whole HUD can shift with the
    // state — ring tint, card accents, corner brackets — from one place
    // instead of every component subscribing separately.
    document.documentElement.dataset.status = status;
  }

  setConnectionState(connected: boolean): void {
    this.banner.classList.toggle("hidden", connected);
    this.banner.textContent = "Disconnected from Max bridge — reconnecting…";
    this.connected = connected;
    if (!connected) this.setStatus("offline");
    // Every command in the rail is a message to the bridge, so with the
    // socket down they can do nothing at all. Disabling them says that,
    // instead of leaving buttons that look live and silently swallow the
    // click.
    if (this.wakeBtn) this.wakeBtn.disabled = !connected;
    this.stopBtn.disabled = !connected;
    this.screenBtn.disabled = !connected;
    // Mute included: the mute state lives on the bridge (cfg.tts_enabled),
    // so toggling it with the socket down would change the button and
    // nothing else.
    if (this.muteBtn) this.muteBtn.disabled = !connected;
    this.updateInputEnabled();
    this.renderVoiceButton();
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
    this.readoutBackend.textContent = UI.publicLabel(label);
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
