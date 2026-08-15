import "./style.css";
import { Orb } from "./orb";
import { AudioPlayer } from "./audio";
import { UI } from "./ui";
import { HudRing } from "./hud-ring";
import { Visualizers } from "./visualizers";
import { VoiceRing } from "./voice-ring";
import { EnvelopePlayer } from "./envelope-player";
import { OrinSocket, type ServerMessage } from "./socket";
import { SpotifyWidget } from "./spotify-widget";

// Overridable with ?bridge=ws://host:port — the bridge is normally on this
// machine, but this makes it possible to open the HUD from another device
// on the LAN (or against a second bridge) without a rebuild.
const BRIDGE_URL =
  new URLSearchParams(location.search).get("bridge") || "ws://localhost:8765";

const canvas = document.getElementById("orb-canvas") as HTMLCanvasElement;
const orb = new Orb(canvas);
const ui = new UI();
const spotifyWidget = new SpotifyWidget();
const hudRing = new HudRing();
const voiceRing = new VoiceRing(document.getElementById("voice-ring") as HTMLCanvasElement);

// Single funnel for audio level, so it makes no difference to the visuals
// whether the sound is playing in this tab (AnalyserNode) or out of the
// Mac's speakers (envelope relayed from the voice loop). Both are measured
// from real audio; neither is synthesized here.
function applyAmplitude(amplitude: number): void {
  orb.setAmplitude(amplitude);
  ui.setAudioLevel(amplitude);
  hudRing.setAmplitude(amplitude);
  voiceRing.setAmplitude(amplitude);
}

function setStatus(status: "idle" | "listening" | "thinking" | "speaking" | "error"): void {
  orb.setStatus(status);
  voiceRing.setStatus(status);
  ui.setStatus(status);
}

const audio = new AudioPlayer(applyAmplitude, () => setStatus(voiceRunning ? "listening" : "idle"));

// Canvas telemetry panels — read the same real analyser as the orb.
const visualizers = new Visualizers(audio);

// Spoken replies never reach this tab as audio, only as the envelope
// measured from the clip the voice loop is playing. See envelope-player.ts.
const envelopePlayer = new EnvelopePlayer(
  (amplitude) => {
    applyAmplitude(amplitude);
    visualizers.setExternalLevel(amplitude);
  },
  (speaking) => {
    visualizers.setExternalActive(speaking);
    if (speaking) {
      setStatus("speaking");
    } else {
      applyAmplitude(0);
      visualizers.setExternalLevel(0);
      setStatus(voiceRunning ? "listening" : "idle");
    }
  }
);

let connected = false;
let voiceRunning = false;
const socket = new OrinSocket(BRIDGE_URL, (isConnected) => {
  connected = isConnected;
  ui.setConnectionState(isConnected);
  ui.setLink(BRIDGE_URL, isConnected);
  if (isConnected) {
    setStatus(voiceRunning ? "listening" : "idle");
  } else {
    orb.setStatus("error");
    voiceRing.setStatus("offline");
  }
});
ui.setLink(BRIDGE_URL, connected);

// Routing-tier strip: highlights whichever backend actually answered, using
// the label the core itself produced (cli.label()) — not a guess made here.
const tiers = Array.from(document.querySelectorAll<HTMLElement>(".tier"));
function highlightTier(label: string): void {
  tiers.forEach((t) => t.classList.toggle("active", t.dataset.label === label));
}

socket.onMessage((msg: ServerMessage) => {
  switch (msg.type) {
    case "reply": {
      ui.appendReply(msg.label, msg.text);
      ui.setBackendReadout(msg.label);
      highlightTier(msg.label);
      setStatus(msg.backend === "claude_code" ? "thinking" : "idle");
      break;
    }
    case "reply_start": {
      ui.beginReplyStream(msg.label);
      ui.setBackendReadout(msg.label);
      highlightTier(msg.label);
      break;
    }
    case "reply_chunk": {
      ui.appendReplyChunk(msg.text);
      break;
    }
    case "reply_end": {
      ui.finishReplyStream();
      setStatus(voiceRunning ? "listening" : "idle");
      break;
    }
    case "confirm_request": {
      ui.appendConfirmPrompt(msg.prompt);
      setStatus("listening");
      break;
    }
    case "audio": {
      setStatus("speaking");
      audio.enqueueBase64Wav(msg.data);
      break;
    }
    case "error": {
      ui.appendError(msg.message);
      setStatus("error");
      break;
    }
    case "muted": {
      // Server ack of our own toggle — button already reflects local state.
      break;
    }
    case "voice_state": {
      voiceRunning = msg.running;
      ui.setVoiceState(msg.running);
      setStatus(msg.running ? "listening" : "idle");
      break;
    }
    case "voice_audio": {
      envelopePlayer.play(msg.envelope, msg.hz);
      break;
    }
    case "voice_audio_end": {
      envelopePlayer.stop();
      break;
    }
    case "screen_frame": {
      if (msg.error) ui.setScreenError(msg.error);
      else if (msg.data) ui.setScreenFrame(msg.data);
      break;
    }
    case "screen_feed_state": {
      ui.setScreenFeedState(msg.enabled);
      break;
    }
    case "models_state": {
      ui.setModels(msg.models, msg.roles, msg.role_labels);
      break;
    }
    case "screen_capture": {
      ui.setCapturing(msg.phase === "start");
      break;
    }
    case "spotify_state": {
      spotifyWidget.update(msg);
      break;
    }
    case "transcript_turn": {
      ui.appendVoiceTurn(msg.role, msg.label, msg.content);
      if (msg.role === "assistant") highlightTier(msg.label);
      break;
    }
  }
});

// Brief "targeting lock" pulse on the corner brackets when sending.
const corners = document.querySelectorAll<HTMLElement>(".hud-corner");
function pulseCorners(): void {
  corners.forEach((el) => {
    el.classList.remove("lock");
    void el.offsetWidth; // force reflow so the animation restarts
    el.classList.add("lock");
  });
}

let messageCount = 0;
ui.onSend = (text) => {
  messageCount++;
  ui.setMessageCount(messageCount);
  pulseCorners();
  socket.sendMessage(text);
};
ui.onConfirm = (answer) => socket.sendConfirm(answer);
ui.onMuteToggle = (muted) => (muted ? socket.mute() : socket.unmute());
ui.onStop = () => {
  // Both halves of "stop talking": audio.stop() kills what this tab is
  // playing, and voice_interrupt reaches the voice loop, whose audio is
  // coming out of the Mac's speakers where nothing in this tab can touch
  // it. audio.stop() intentionally clears source.onended (see audio.ts),
  // so onPlaybackEnd doesn't fire — set the status here instead.
  audio.stop();
  envelopePlayer.stop();
  socket.interruptVoice();
  setStatus(voiceRunning ? "listening" : "idle");
};
ui.onVoiceToggle = (running) => {
  pulseCorners();
  running ? socket.startVoice() : socket.stopVoice();
};
ui.onWake = () => {
  pulseCorners();
  socket.wakeVoice();
};
ui.onScreenToggle = (enabled) => socket.setScreenFeed(enabled);
ui.onModelChange = (role, model) => socket.setModel(role, model);

// Real elapsed session time since page load.
const sessionStart = performance.now();
setInterval(() => {
  ui.setSessionTime((performance.now() - sessionStart) / 1000);
}, 1000);

// Decorative edge columns — stylistic motion graphics only, not data.
// Content duplicated so the CSS translateY(-50%) scroll loops seamlessly.
function buildDataStream(el: Element | null): void {
  if (!el) return;
  const chars = "0123456789ABCDEF";
  const lines: string[] = [];
  for (let i = 0; i < 44; i++) {
    lines.push(chars[Math.floor(Math.random() * 16)] + chars[Math.floor(Math.random() * 16)]);
  }
  el.textContent = lines.join("\n") + "\n" + lines.join("\n");
}
buildDataStream(document.querySelector(".data-stream.left .data-stream-inner"));
buildDataStream(document.querySelector(".data-stream.right .data-stream-inner"));
