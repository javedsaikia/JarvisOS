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
    # Models the dashboard offers, and which one answers for each job.
    # Cloud entries need their key in jarvis/.env (GROQ_API_KEY,
    # GEMINI_API_KEY) — without it they still appear, marked unavailable,
    # so it is obvious why one can't be picked. Roles are assigned in the
    # web UI; see jarvis/llm.py.
    "models": [
        {"id": "local-fast", "provider": "ollama", "model": "qwen2.5:1.5b",
         "label": "Local · Qwen 1.5B"},
        {"id": "local", "provider": "ollama", "model": "qwen2.5-coder:7b",
         "label": "Local · Qwen Coder 7B"},
        {"id": "groq-8b", "provider": "groq", "model": "llama-3.1-8b-instant",
         "label": "Groq · Llama 3.1 8B"},
        {"id": "groq-70b", "provider": "groq", "model": "llama-3.3-70b-versatile",
         "label": "Groq · Llama 3.3 70B"},
        {"id": "gemini-flash", "provider": "gemini", "model": "gemini-2.0-flash",
         "label": "Gemini · 2.0 Flash"},
    ],
    # Defaults stay local, so nothing starts costing money on its own —
    # a cloud model is used only for a role you assign it to.
    "model_roles": {"chat": "local", "voice": "local-fast", "screen": "local-fast",
                    # Not "local-fast": reflection runs in the background
                    # where nobody is waiting, so it should use the most
                    # capable local model rather than the quickest. A
                    # small model invents, and an invented profile line is
                    # permanent context.
                    "memory": "local"},
    # --- Memory (jarvis/memory.py + jarvis/reflection.py)
    # Raw turns stay loadable for this many days, then move to
    # conversation_archive.jsonl. Nothing is deleted.
    "memory_retention_days": 21,
    # Automatic profile-building. Off means Orin only knows what you told
    # it explicitly via /remember and facts.md.
    "memory_reflection_enabled": True,
    # Reflect after this many new turns, or after this long with at least
    # a few turns waiting — whichever comes first.
    "memory_reflect_every_turns": 20,
    "memory_reflect_min_seconds": 1200,
    "memory_reflect_max_turns": 60,
    "memory_reflect_max_tokens": 500,
    "claude_code_enabled": True,
    "claude_code_command": "claude",
    "claude_code_confirm": True,
    "tools_confirm_writes": True,
    "tts_enabled": True,
    "tts_backend": "piper",
    "tts_voice_model": "jarvis/voices/en_GB-alan-medium.onnx",
    # How Orin is woken.
    #   "phrase" — VAD + a small local Whisper spot whatever phrase you put
    #              in wake_phrases. No model to train, and nothing named
    #              after somebody else's product. See jarvis/wake_phrase.py.
    #   "model"  — an openWakeWord detector named by voice_wake_word. Lower
    #              CPU and it fires mid-phrase, but the pretrained set is
    #              hey_jarvis / alexa / hey_mycroft / hey_rhasspy only.
    "wake_mode": "phrase",
    "wake_phrases": ["hey orin"],
    # Empty means "share voice_stt_model", which is the default and costs
    # nothing extra. Set a size (e.g. "tiny.en") to load a separate,
    # faster model for spotting — measured at 0.1s per utterance against
    # small.en's 0.6s, but it renders an invented name inconsistently
    # enough to miss real wakes. See jarvis/wake_phrase.py.
    "wake_stt_model": "",
    # Only used when wake_mode is "model".
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
    # Tap a shortcut to wake Orin, instead of holding one to talk (see
    # jarvis/hotkey.py). Works from any app, so you never have to find the
    # window or say the phrase.
    #   "double_tap" — tap wake_hotkey_key twice, the way macOS dictation
    #                  works. Collides with nothing: a lone modifier means
    #                  nothing to any application.
    #   "combo"      — a chord like ["ctrl", "alt", "space"]. pynput
    #                  watches keys without consuming them, so the chord
    #                  ALSO reaches whatever app has focus — pick one no
    #                  app uses.
    #   "off"        — no wake hotkey.
    "wake_hotkey_mode": "double_tap",
    # Right Command: the least-used modifier on an Apple keyboard, and not
    # part of push-to-talk's Control+Option, so the two can't trip over
    # each other.
    "wake_hotkey_key": "cmd_r",
    "wake_hotkey_combo": ["ctrl", "alt", "space"],
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
    # Pins a specific local model for screen questions. Empty (the
    # default) means the dashboard's "screen" role decides, which is
    # where this is meant to be changed now.
    "screen_text_model": "",
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


# Settings whose value is a dict of independent sub-settings. A plain
# update() replaces the whole dict, so a config.json written before a new
# sub-key existed would silently drop it — that is how the "memory" role
# went missing the moment anyone saved a model choice from the dashboard.
_MERGED_KEYS = ("model_roles",)


def load_config() -> dict:
    if CONFIG_PATH.exists():
        cfg = DEFAULTS.copy()
        saved = json.loads(CONFIG_PATH.read_text())
        for key in _MERGED_KEYS:
            if isinstance(saved.get(key), dict) and isinstance(DEFAULTS.get(key), dict):
                saved[key] = {**DEFAULTS[key], **saved[key]}
        cfg.update(saved)
        return cfg
    save_config(DEFAULTS)
    return DEFAULTS.copy()


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def update_config(**changes) -> None:
    """Change specific settings on disk, leaving the rest of the file alone.

    Deliberately not save_config(load_config()): that writes the *merged*
    config, freezing today's defaults into the user's file, so every later
    improvement to a default would silently never reach them. This touches
    only the keys it was given.
    """
    current = {}
    if CONFIG_PATH.exists():
        try:
            current = json.loads(CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            current = {}
    current.update(changes)
    CONFIG_PATH.write_text(json.dumps(current, indent=2))
