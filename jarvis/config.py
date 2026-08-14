"""Config load/save. JSON file next to this package, human-editable."""
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"

DEFAULTS = {
    "user_name": "Sir",
    "default_backend": "ollama",
    "ollama_host": "http://localhost:11434",
    "ollama_model": "qwen2.5-coder:7b",
    "voice_ollama_model": "qwen2.5-coder:1.5b",
    "voice_ollama_max_tokens": 96,
    "bridge_ollama_model": "qwen2.5-coder:1.5b",
    "bridge_ollama_max_tokens": 400,
    "ollama_keep_alive": "30m",
    "claude_code_enabled": True,
    "claude_code_command": "claude",
    "claude_code_confirm": True,
    "tools_confirm_writes": True,
    "tts_enabled": True,
    "tts_backend": "piper",
    "tts_voice_model": "jarvis/voices/en_GB-alan-medium.onnx",
    "voice_wake_word": "hey_jarvis",
    "voice_stt_model": "small.en",
    "voice_context_turns": 6,
    # How much trailing silence ends an utterance. Was 0.45s, which cut
    # people off mid-sentence whenever they paused to think — seen live,
    # "run this command: sleep 30" was endpointed after "run this command."
    # and the actual command was lost. 0.8s is roughly where mainstream
    # assistants sit and costs only a third of a second more per turn.
    "voice_silence_seconds": 0.8,
    "voice_no_speech_bail_seconds": 3.0,
    "voice_wake_listen_seconds": 10.0,
    # Push-to-talk (see jarvis/hotkey.py). Hold the combo and speak — no
    # wake word, no VAD gate, no timeout, so it stays reliable over music
    # and room noise where the hands-free path is probabilistic.
    "push_to_talk_enabled": True,
    "push_to_talk_combo": ["ctrl", "alt"],
    "voice_silence_rms_threshold": 300,
    # Was 180 — seen live, a window that long meant any speech near the
    # mic (e.g. dictating a message to something else entirely) got picked
    # up as a Orin follow-up command for minutes after the last real
    # exchange, with no wake word required to re-arm it.
    "voice_conversation_window_seconds": 25,
    # Master switch for everything that looks at the screen — "what's on
    # my screen?", and the live screen card below. False disables the lot.
    "vision_enabled": True,
    # Live screen card in the web UI (bridge._screen_feed_loop). Real
    # screenshots of your desktop, pushed to the browser every 2s while a
    # tab is open — set false to disable it outright, regardless of the
    # UI's own toggle.
    "screen_feed_enabled": True,
    # --- "What's on my screen?" (jarvis/screen_capture.py + tools/vision.py)
    # macOS reads the screen text on-device (Vision framework) and the
    # local text model answers from it. A small vision model cannot read a
    # Retina screenshot — measured, see tools/vision.py — so OCR is the
    # primary path and the vision model is the fallback for screens with
    # no text on them.
    "screen_ocr_enabled": True,
    # Answers about screen text. Small and fast beats clever here: this
    # runs on every "what's on my screen", including spoken ones.
    "screen_text_model": "qwen2.5:1.5b",
    "screen_max_tokens": 220,
    # Window title marked as Orin's own and left out of the capture, so
    # the HUD is never what Orin describes.
    "screen_hud_window_marker": "Orin HUD",
    # Only used when native window-excluding capture is unavailable: ask
    # the web UI to blank itself, wait, then screenshot everything.
    "screen_hide_ui": True,
    "screen_hide_delay_ms": 400,
    # Paid cloud vision fallback. Off by default, and even when on it
    # still asks before sending anything — a screenshot is about the most
    # sensitive thing this machine can send anywhere. Needs GEMINI_API_KEY
    # in jarvis/.env.
    "screen_cloud_fallback": False,
    "screen_cloud_model": "gemini-2.0-flash",
    "vision_ollama_model": "moondream",
    "browser_enabled": True,
    "location_enabled": True,
    "location_default_radius_m": 3000,
    "calling_enabled": True,
    "spotify_max_volume": 75,
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        cfg = DEFAULTS.copy()
        cfg.update(json.loads(CONFIG_PATH.read_text()))
        return cfg
    save_config(DEFAULTS)
    return DEFAULTS.copy()


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
