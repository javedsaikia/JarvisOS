"""Minimal Sarvam chat client (Sarvam-105B). Stdlib only — same pattern as
ollama_client.py, just pointed at Sarvam's OpenAI-compatible chat
completions endpoint instead of a local Ollama server. Used for
Hindi/Assamese conversations, where the local Ollama model isn't reliable
(seen live: hallucinated movie titles/directors, drifted into broken
Hindi-ish text mid-conversation).
"""
import base64
import json
import urllib.error
import urllib.request

from jarvis.env import load_env

API_URL = "https://api.sarvam.ai/v1/chat/completions"
TTS_URL = "https://api.sarvam.ai/text-to-speech"
DEFAULT_MODEL = "sarvam-105b"
DEFAULT_TTS_MODEL = "bulbul:v3"
DEFAULT_TTS_SPEAKER = "shubh"


class SarvamError(Exception):
    pass


def _api_key() -> str:
    key = load_env().get("SARVAM_API_KEY", "")
    if not key:
        raise SarvamError("SARVAM_API_KEY not set in jarvis/.env")
    return key


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    timeout: int = 90,
) -> str:
    """Sarvam-105B is a reasoning model — it spends completion tokens on an
    internal chain-of-thought (returned separately as reasoning_content)
    before ever writing the final answer. Verified live: a plain "say
    hello" burned ~940 completion tokens and 7.9s just to reach "Hello.".
    A too-small max_tokens truncates mid-reasoning (finish_reason="length")
    and message.content comes back None — not an error, just an empty
    answer — so the default here is deliberately generous, and a None/empty
    content is treated as exactly that failure mode rather than silently
    returned as if it were a real (blank) reply.
    """
    body = {
        "messages": messages,
        "model": model,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    payload = json.dumps(body).encode()
    key = _api_key()
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            # Sarvam accepts either header; sending both since docs disagree
            # on which is authoritative and this costs nothing.
            "Authorization": f"Bearer {key}",
            "api-subscription-key": key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SarvamError(f"Sarvam API error {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise SarvamError(f"Could not reach Sarvam API ({e})") from e

    choice = data["choices"][0]
    content = choice["message"].get("content")
    if not content:
        reason = choice.get("finish_reason", "unknown")
        raise SarvamError(
            f"Sarvam returned no content (finish_reason={reason!r}) — likely truncated "
            "mid-reasoning; try a larger max_tokens."
        )
    return content


def text_to_speech(
    text: str,
    language_code: str = "hi-IN",
    model: str = DEFAULT_TTS_MODEL,
    speaker: str = DEFAULT_TTS_SPEAKER,
    timeout: int = 60,
) -> bytes:
    """Returns raw WAV bytes — same shape as tts.synthesize()'s Piper
    output, so callers (voice_loop's StreamingSpeech, bridge.py) can treat
    this as a drop-in alternative source of audio for the same play()/send
    paths, not a parallel code path. language_code is BCP-47 (hi-IN,
    as-IN, en-IN, ...); bulbul:v3 caps input at 2500 characters."""
    body = {
        "text": text,
        "language_code": language_code,
        "model": model,
        "speaker": speaker,
    }
    payload = json.dumps(body).encode()
    key = _api_key()
    req = urllib.request.Request(
        TTS_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "api-subscription-key": key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SarvamError(f"Sarvam TTS error {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise SarvamError(f"Could not reach Sarvam TTS API ({e})") from e

    audios = data.get("audios") or []
    if not audios:
        raise SarvamError("Sarvam TTS returned no audio")
    return base64.b64decode(audios[0])
