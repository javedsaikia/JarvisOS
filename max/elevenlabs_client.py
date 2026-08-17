"""ElevenLabs text-to-speech. Stdlib urllib only, same shape as the other
API clients here (sarvam_client, omdb_client, news_client) — one POST does
not justify an SDK.

The key comes from ELEVENLABS_API_KEY, read from the real environment or
from max/.env (see max/env.py). Never from config.json, which is a plain
preferences file with no secret-handling story.

Audio comes back as raw 16-bit PCM (`output_format=pcm_22050`) and is
wrapped in a WAV header here, because everything downstream expects a
playable WAV: afplay plays it, and max/voice_events.py opens the same
bytes with the `wave` module to measure the amplitude envelope the HUD
orb animates from. Asking for MP3 would save bandwidth and break the orb.
"""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
import wave

from max import env

API_ROOT = "https://api.elevenlabs.io/v1"

# The Flash family is the low-latency line and the reason to use
# ElevenLabs for a realtime voice at all. eleven_turbo_v2_5 is the
# quality-leaning sibling if the extra latency is ever worth it.
DEFAULT_MODEL = "eleven_flash_v2_5"

# 16-bit PCM at 22.05 kHz — matches what the macOS voices already produce,
# so nothing downstream has to care which provider spoke.
OUTPUT_FORMAT = "pcm_22050"
SAMPLE_RATE = 22050


class ElevenLabsError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def api_key() -> str:
    return (env.load_env().get("ELEVENLABS_API_KEY") or "").strip()


def configured() -> bool:
    return bool(api_key())


def _pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buf.getvalue()


def text_to_speech(
    text: str,
    voice_id: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 30,
) -> bytes:
    """WAV bytes for `text` in `voice_id`. Raises ElevenLabsError."""
    body = (text or "").strip()
    if not body:
        return b""
    key = api_key()
    if not key:
        raise ElevenLabsError("ELEVENLABS_API_KEY is not set")
    if not (voice_id or "").strip():
        raise ElevenLabsError("no tts_voice_id set in config")

    url = (
        f"{API_ROOT}/text-to-speech/{urllib.parse.quote(voice_id)}"
        f"?output_format={OUTPUT_FORMAT}"
    )
    payload = json.dumps({"text": body, "model_id": model}).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "xi-api-key": key},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            pcm = response.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:200]
        raise ElevenLabsError(f"ElevenLabs error {e.code}: {detail}", status=e.code) from e
    # TimeoutError is NOT a URLError — urllib only wraps a timeout that
    # happens while connecting, so a slow read escapes bare. Same trap
    # that once hung the whole voice loop; see max/sarvam_client.py.
    except (urllib.error.URLError, TimeoutError) as e:
        raise ElevenLabsError(f"could not reach ElevenLabs ({e})") from e
    if not pcm:
        raise ElevenLabsError("ElevenLabs returned no audio")
    return _pcm_to_wav(pcm)


def voices() -> list[dict]:
    """The voices this key can use — for picking a tts_voice_id."""
    key = api_key()
    request = urllib.request.Request(
        f"{API_ROOT}/voices", headers={"xi-api-key": key} if key else {}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError) as e:
        raise ElevenLabsError(f"could not list voices ({e})") from e
    out = []
    for v in data.get("voices", []):
        labels = v.get("labels") or {}
        out.append({
            "voice_id": v.get("voice_id", ""),
            "name": v.get("name", ""),
            "accent": labels.get("accent", ""),
            "gender": labels.get("gender", ""),
        })
    return out
