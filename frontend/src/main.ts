import "./style.css";
import { Orb } from "./orb";
import { AudioPlayer } from "./audio";
import { UI } from "./ui";
import { HudRing } from "./hud-ring";
import { Visualizers } from "./visualizers";
import { JarvisSocket, type ServerMessage } from "./socket";
import { SpotifyWidget } from "./spotify-widget";

const BRIDGE_URL = "ws://localhost:8765";

const canvas = document.getElementById("orb-canvas") as HTMLCanvasElement;
const orb = new Orb(canvas);
const ui = new UI();
const spotifyWidget = new SpotifyWidget();
const hudRing = new HudRing();
const visionVideo = document.querySelector(".vision-video") as HTMLVideoElement | null;

const audio = new AudioPlayer(
  (amplitude) => {
    orb.setAmplitude(amplitude);
    ui.setAudioLevel(amplitude);
    hudRing.setAmplitude(amplitude);
  },
  () => orb.setStatus("idle")
);

// Canvas telemetry panels — read the same real analyser as the orb.
new Visualizers(audio);

function startVisionMotionSampler(video: HTMLVideoElement | null): void {
  if (!video) return;

  const sampleCanvas = document.createElement("canvas");
  sampleCanvas.width = 48;
  sampleCanvas.height = 86;
  const ctx = sampleCanvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return;

  let previous: Uint8ClampedArray | null = null;
  let smoothed = 0;
  let smoothedBrightness = 0;
  let rafId = 0;

  const step = () => {
    if (video.readyState >= 2 && !video.paused && !video.ended) {
      ctx.drawImage(video, 0, 0, sampleCanvas.width, sampleCanvas.height);
      const data = ctx.getImageData(0, 0, sampleCanvas.width, sampleCanvas.height).data;
      let brightnessSum = 0;
      const pixelCount = sampleCanvas.width * sampleCanvas.height;
      for (let i = 0; i < data.length; i += 4) {
        brightnessSum += (data[i] + data[i + 1] + data[i + 2]) / 3;
      }
      const brightness = brightnessSum / pixelCount / 255;
      smoothedBrightness += (brightness - smoothedBrightness) * 0.08;
      if (previous && previous.length === data.length) {
        let diff = 0;
        for (let i = 0; i < data.length; i += 4) {
          const current = (data[i] + data[i + 1] + data[i + 2]) / 3;
          const last = (previous[i] + previous[i + 1] + previous[i + 2]) / 3;
          diff += Math.abs(current - last);
        }
        const normalized = Math.min(1, diff / (sampleCanvas.width * sampleCanvas.height * 36));
        smoothed += (normalized - smoothed) * 0.12;
      }
      orb.setVideoSignal(smoothedBrightness, smoothed);
      previous = new Uint8ClampedArray(data);
    }
    rafId = requestAnimationFrame(step);
  };

  const syncPlayback = () => {
    if (!video.paused && !video.ended) {
      if (rafId === 0) rafId = requestAnimationFrame(step);
    } else {
      orb.setVideoSignal(0, 0);
    }
  };

  video.addEventListener("play", syncPlayback);
  video.addEventListener("pause", () => orb.setVideoSignal(0, 0));
  video.addEventListener("ended", () => orb.setVideoSignal(0, 0));
  syncPlayback();
}

startVisionMotionSampler(visionVideo);

let connected = false;
const socket = new JarvisSocket(BRIDGE_URL, (isConnected) => {
  connected = isConnected;
  ui.setConnectionState(isConnected);
  ui.setLink(BRIDGE_URL, isConnected);
  orb.setStatus(isConnected ? "idle" : "error");
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
      orb.setStatus(msg.backend === "claude_code" ? "thinking" : "idle");
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
      orb.setStatus("idle");
      break;
    }
    case "confirm_request": {
      ui.appendConfirmPrompt(msg.prompt);
      orb.setStatus("listening");
      break;
    }
    case "audio": {
      orb.setStatus("speaking");
      audio.enqueueBase64Wav(msg.data);
      break;
    }
    case "error": {
      ui.appendError(msg.message);
      orb.setStatus("error");
      break;
    }
    case "muted": {
      // Server ack of our own toggle — button already reflects local state.
      break;
    }
    case "voice_state": {
      ui.setVoiceState(msg.running);
      orb.setStatus(msg.running ? "listening" : "idle");
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
  // Client-side only, deliberately: playback happens in the browser, so
  // there's no round trip needed to cut it off instantly. Doesn't cancel an
  // in-flight LLM generation on the server — just stops what's already
  // queued/playing here. audio.stop() intentionally clears source.onended
  // before stopping (see audio.ts), so onPlaybackEnd doesn't fire on its
  // own — set the status here instead.
  audio.stop();
  orb.setStatus("idle");
};
ui.onVoiceToggle = (running) => {
  pulseCorners();
  running ? socket.startVoice() : socket.stopVoice();
};
ui.onWake = () => {
  pulseCorners();
  socket.wakeVoice();
};

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
