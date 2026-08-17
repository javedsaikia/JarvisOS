"""Minimal .env loader — stdlib only, no extra deps needed for a single
KEY=VALUE file. Keeps API keys (Sarvam, ElevenLabs, etc.) out of
config.json, which is a plain preferences file with no secret-handling
story and no reason to ever hold a credential.
"""
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"


def load_env() -> dict[str, str]:
    """Keys from max/.env, with the real environment taking precedence.

    The file used to be the only source, so exporting a key in the shell —
    the ordinary way to hand a process a secret, and what the ElevenLabs
    setup asks for — had no effect at all.
    """
    import os

    env: dict[str, str] = {}
    if ENV_PATH.exists():
        env.update(_from_file())
    for key in _ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            env[key] = value
    return env


# Real environment variables win over the file, so a key exported for one
# run overrides whatever is on disk without editing it.
_ENV_KEYS = (
    "ELEVENLABS_API_KEY", "SARVAM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
    "COMPOSIO_API_KEY", "OMDB_API_KEY", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET",
)


def _from_file() -> dict[str, str]:
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
