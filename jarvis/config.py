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
    "voice_silence_seconds": 0.45,
    "voice_silence_rms_threshold": 300,
    "voice_conversation_window_seconds": 180,
    "vision_enabled": True,
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
