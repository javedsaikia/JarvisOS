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
    "voice_ollama_max_tokens": 80,
    "bridge_ollama_model": "qwen2.5-coder:1.5b",
    "bridge_ollama_max_tokens": 120,
    # Hard cap on ordinary chat (HUD + voice fallback). Stops Groq writing
    # a lecture when the user asked a one-line question.
    "chat_max_tokens": 120,
    "ollama_keep_alive": "30m",
    # Models the dashboard offers, and which one answers for each job.
    # Cloud entries need their key in bella/.env (GROQ_API_KEY,
    # GEMINI_API_KEY) — without it they still appear, marked unavailable,
    # so it is obvious why one can't be picked. Roles are assigned in the
    # web UI; see bella/llm.py.
    "models": [
        {"id": "local-fast", "provider": "ollama", "model": "qwen2.5:1.5b",
         "label": "Local · Qwen 1.5B"},
        {"id": "local", "provider": "ollama", "model": "qwen2.5-coder:7b",
         "label": "Local · Qwen Coder 7B"},
        # Groq decommissioned both Llama models this catalog used to
        # name. Measured live: every call returned "HTTP 404: The model
        # `llama-3.3-70b-versatile` does not exist or you do not have
        # access to it", so chat, voice AND screen were all silently
        # falling back to the local model — one turn took 32s. These two
        # are what the key can actually reach; check with
        # GET https://api.groq.com/openai/v1/models if they ever go the
        # same way.
        {"id": "groq-20b", "provider": "groq", "model": "openai/gpt-oss-20b",
         "label": "Groq · GPT-OSS 20B"},
        {"id": "groq-120b", "provider": "groq", "model": "openai/gpt-oss-120b",
         "label": "Groq · GPT-OSS 120B"},
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
    # --- Memory (bella/memory.py + bella/reflection.py)
    # Raw turns stay loadable for this many days, then move to
    # conversation_archive.jsonl. Nothing is deleted.
    "memory_retention_days": 21,
    # Automatic profile-building. Off means Bella only knows what you told
    # it explicitly via /remember and facts.md.
    "memory_reflection_enabled": True,
    # Reflect after this many new turns, or after this long with at least
    # a few turns waiting — whichever comes first.
    "memory_reflect_every_turns": 20,
    "memory_reflect_min_seconds": 1200,
    "memory_reflect_max_turns": 60,
    "memory_reflect_max_tokens": 500,
    # Longest thing Bella will write for you in one go (bella/compose.py) —
    # an email, a prompt, a covering letter. Generous because a draft that
    # stops mid-sentence is worse than a slightly slower one.
    "compose_max_tokens": 700,
    "claude_code_enabled": True,
    "claude_code_command": "claude",
    "claude_code_confirm": True,
    "tools_confirm_writes": True,
    "tts_enabled": True,
    # --- Which voice Bella speaks in (bella/speech.py)
    #   "auto"   — a local macOS voice for English, Sarvam for Indian
    #              languages. Local is faster and cannot fail: measured on
    #              the same sentence, Rishi 0.88s against Sarvam's 2.35s,
    #              with no network in the path at all.
    #   "macos"  — always a locally installed voice (`say -v '?'` lists them)
    #   "sarvam" — always Bulbul, for when its voice is worth the wait
    # Ask "add X to your calendar?" before writing. Off: the write is
    # about a second and the spoken yes/no cost several times that, and a
    # calendar entry is the most reversible thing Bella writes — "remove
    # that" takes it straight back out. Email sends and shell commands
    # still confirm regardless of this.
    "confirm_calendar_writes": False,
    "tts_engine": "auto",
    # Which provider speaks ENGLISH. Sarvam is bypassed entirely on this
    # path — it stays for Assamese, which is the only thing it is uniquely
    # able to say.
    #   "elevenlabs" — realtime-oriented cloud voice, needs
    #                  ELEVENLABS_API_KEY in the environment or bella/.env
    #   "sarvam"     — Bulbul, using tts_voice
    #   "macos"      — a locally installed voice, no network at all
    "tts_provider_english": "elevenlabs",
    # Where English goes when ElevenLabs cannot answer. "sarvam" (neural)
    # or "macos" (local, but concatenative and robotic).
    "tts_english_fallback": "sarvam",
    # ElevenLabs voice ID — a voice from your own ElevenLabs library.
    # `python3 -c "from bella import elevenlabs_client as e;
    # print(e.voices())"` lists what the key can reach.
    "tts_voice_id": "Ix8C14HEHgIQkJswik2o",
    # Flash is the low-latency line, which is the point of using
    # ElevenLabs for a live voice; eleven_turbo_v2_5 trades speed for
    # quality if that is ever the better bargain.
    "elevenlabs_model": "eleven_flash_v2_5",
    # English voice. A macOS voice name under "auto"/"macos" (Rishi, Tara
    # and Aman are the Indian-English ones this Mac ships), or a Bulbul
    # speaker name under "sarvam". See bella/speech.py for the full list.
    "tts_voice": "ritu",
    # Assamese/Hindi voice. Always a Bulbul speaker — macOS has no Indian
    # language voices installed, so this is not interchangeable with the
    # English one above.
    "tts_voice_indic": "ritu",
    # Assamese ears + voice. "auto" uses Bhashini when BHASHINI_INFERENCE_KEY
    # is set, otherwise Sarvam Bulbul / Saaras.
    "assamese_speech_provider": "auto",
    # Append Bhashini ASR/TTS feedback to bella/memory/assamese_eval_dataset.jsonl
    "bhashini_feedback_enabled": True,
    # Words per minute for the macOS voices. 180 is `say`'s default.
    "tts_rate": 180,
    "tts_backend": "piper",
    "tts_voice_model": "bella/voices/en_GB-alan-medium.onnx",
    # Digital gain for spoken replies (afplay -v), 5–100. Separate from
    # the Mac's output volume — Piper is loud on earphones at 100.
    "tts_volume": 35,
    # How Bella is woken.
    #   "phrase" — VAD + a small local Whisper spot whatever phrase you put
    #              in wake_phrases. No model to train, and nothing named
    #              after somebody else's product. See bella/wake_phrase.py.
    #   "model"  — an openWakeWord detector named by voice_wake_word. Lower
    #              CPU and it fires mid-phrase, but the pretrained set is
    #              hey_jarvis / alexa / hey_mycroft / hey_rhasspy only.
    "wake_mode": "phrase",
    "wake_phrases": ["hey bella", "bella"],
    # Empty means "share voice_stt_model", which is the default and costs
    # nothing extra. Set a size (e.g. "tiny.en") to load a separate,
    # faster model for spotting — measured at 0.1s per utterance against
    # small.en's 0.6s, but it renders an invented name inconsistently
    # enough to miss real wakes. See bella/wake_phrase.py.
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
    # Push-to-talk (see bella/hotkey.py). Hold the combo and speak — no
    # wake word, no VAD gate, no timeout, so it stays reliable over music
    # and room noise where the hands-free path is probabilistic.
    "push_to_talk_enabled": True,
    "push_to_talk_combo": ["cmd", "alt"],
    # Tap a shortcut to wake Bella, instead of holding one to talk (see
    # bella/hotkey.py). Works from any app, so you never have to find the
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
    # 0 = every spoken command must start with the wake phrase. A leftover
    # open window used to answer anything said near the mic after a turn.
    "voice_conversation_window_seconds": 0,
    # Master switch for everything that looks at the screen — "what's on
    # my screen?", and the live screen card below. False disables the lot.
    "vision_enabled": True,
    # Live screen card in the web UI (bridge._screen_feed_loop). Real
    # screenshots of your desktop, pushed to the browser every 2s while a
    # tab is open — set false to disable it outright, regardless of the
    # UI's own toggle.
    "screen_feed_enabled": True,
    # --- "What's on my screen?" (bella/screen_capture.py + tools/vision.py)
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
    # 120 cut medical SBA answers mid-sentence ("patient with…"). The
    # Groq call hit max_tokens and the spoken reply just stopped.
    "screen_max_tokens": 400,
    # Window title marked as Bella's own and left out of the capture, so
    # the HUD is never what Bella describes.
    "screen_hud_window_marker": "Bella HUD",
    # Only used when native window-excluding capture is unavailable: ask
    # the web UI to blank itself, wait, then screenshot everything.
    "screen_hide_ui": True,
    "screen_hide_delay_ms": 400,
    # Paid cloud vision fallback. Off by default, and even when on it
    # still asks before sending anything — a screenshot is about the most
    # sensitive thing this machine can send anywhere. Needs GEMINI_API_KEY
    # in bella/.env.
    "screen_cloud_fallback": False,
    "screen_cloud_model": "gemini-3-flash-preview",
    # Locally installed model that accepts images, for when there is no
    # Gemini key. Blank means "find one" — anything with moondream/llava/
    # vl/vision in its name. It genuinely sees the pixels, but it is small:
    # measured, it describes a window accurately at 1024px wide and cannot
    # read a chess position at all.
    "screen_local_vision_model": "",
    "vision_ollama_model": "moondream",
    # Isolated Playwright Chromium is not the user's real desktop browser
    # and is not daily computer control. Off unless you turn it on.
    "browser_enabled": False,
    # Native click/type via the macos-harness CLI (already-open apps only).
    # Off until a live check. Teaching still does not click.
    "desktop_enabled": False,
    "location_enabled": True,
    "location_default_radius_m": 3000,
    "calling_enabled": True,
    # Javed's iPhone — Continuity uses this handset as the cellular radio.
    # Outgoing restaurant calls are never placed as this number.
    "user_phone": "",
    "news_enabled": True,
    "omdb_enabled": True,
    # Public listing search for movies / hotels / flights. Read-only.
    # Opening a listing URL is confirmed; checkout is never automated.
    "discovery_enabled": True,
    "spotify_max_volume": 75,
    # Stable Composio user id — connections are stored against this.
    # Overwritten at runtime with the user id that already owns
    # Composio connections, if one exists.
    "composio_user_id": "bella",
    # --- Laya System-1 router (optional, bella/laya_router.py)
    # Requires: pip install laya (adds torch). Off by default — zero-shot
    # accuracy on Bella's tool schema is ~28% until fine-tuned. See
    # scripts/laya_poc/FINETUNE_RECOMMENDATION.md.
    "laya_enabled": False,
    "laya_preload": False,
    "laya_device": "mps",
    "laya_confidence_threshold": 0.85,
    "laya_guard_enabled": False,
    "laya_guard_threshold": 0.85,
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
