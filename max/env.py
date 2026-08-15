"""Minimal .env loader — stdlib only, no extra deps needed for a single
KEY=VALUE file. Keeps API keys (Sarvam, ElevenLabs, etc.) out of
config.json, which is a plain preferences file with no secret-handling
story and no reason to ever hold a credential.
"""
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"


def load_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    try:
        import os
        os.chmod(ENV_PATH, 0o600)
    except OSError:
        pass
    env = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env
