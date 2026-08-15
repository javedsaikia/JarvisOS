"""Minimal Sarvam chat client (Sarvam-105B). Stdlib only — same pattern as
ollama_client.py, just pointed at Sarvam's OpenAI-compatible chat
completions endpoint instead of a local Ollama server. Used for
Hindi/Assamese conversations, where the local Ollama model isn't reliable
(seen live: hallucinated movie titles/directors, drifted into broken
Hindi-ish text mid-conversation).
"""
import base64
import io
import json
import urllib.error
import urllib.request
import wave

from max.env import load_env

API_URL = "https://api.sarvam.ai/v1/chat/completions"
TTS_URL = "https://api.sarvam.ai/text-to-speech"
STT_URL = "https://api.sarvam.ai/speech-to-text"
DEFAULT_MODEL = "sarvam-105b"
DEFAULT_TTS_MODEL = "bulbul:v3"
DEFAULT_TTS_SPEAKER = "shubh"
DEFAULT_STT_MODEL = "saaras:v3"


class SarvamError(Exception):
    pass


def _api_key() -> str:
    key = load_env().get("SARVAM_API_KEY", "")
    if not key:
        raise SarvamError("SARVAM_API_KEY not set in max/.env")
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


_as_in_tts_fallback_noted = False


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
    as-IN, en-IN, ...); bulbul:v3 caps input at 2500 characters.

    as-IN (Assamese) is documented but still beta-gated on some keys.
    When that happens we speak the same text with bn-IN — same script,
    close enough to hear — rather than going silent.
    """
    try:
        return _tts_request(text, language_code, model, speaker, timeout)
    except SarvamError as e:
        if language_code != "as-IN" or "beta" not in str(e).lower():
            raise
        global _as_in_tts_fallback_noted
        if not _as_in_tts_fallback_noted:
            print(
                "(Sarvam Assamese voice needs beta access — "
                "speaking with the Bengali voice until that is granted.)"
            )
            _as_in_tts_fallback_noted = True
        return _tts_request(text, "bn-IN", model, speaker, timeout)


def _tts_request(
    text: str,
    language_code: str,
    model: str,
    speaker: str,
    timeout: int,
) -> bytes:
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


def pcm16_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw mono 16-bit PCM in a WAV header. Saaras accepts WAV and
    wants 16 kHz — the same format the voice loop already records."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buf.getvalue()


def speech_to_text(
    wav_bytes: bytes,
    language_code: str = "unknown",
    model: str = DEFAULT_STT_MODEL,
    mode: str = "transcribe",
    timeout: int = 30,
) -> tuple[str, str | None]:
    """Saaras transcription. Returns (transcript, detected_language).

    language_code is BCP-47 (as-IN, hi-IN, en-IN, unknown). Assamese is
    as-IN. mode=transcribe keeps the original language instead of
    translating into English, which is what the Assamese conversation
    path needs.
    """
    boundary = "----MaxSarvamSTT"
    parts = bytearray()

    def add_field(name: str, value: str) -> None:
        parts.extend(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n'
                f"\r\n"
                f"{value}\r\n"
            ).encode()
        )

    add_field("model", model)
    add_field("mode", mode)
    add_field("language_code", language_code)
    parts.extend(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="utterance.wav"\r\n'
            f"Content-Type: audio/wav\r\n"
            f"\r\n"
        ).encode()
    )
    parts.extend(wav_bytes)
    parts.extend(f"\r\n--{boundary}--\r\n".encode())
    body = bytes(parts)

    key = _api_key()
    req = urllib.request.Request(
        STT_URL,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {key}",
            "api-subscription-key": key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SarvamError(f"Sarvam STT error {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise SarvamError(f"Could not reach Sarvam STT API ({e})") from e

    transcript = (data.get("transcript") or "").strip()
    detected = data.get("language_code") or None
    return transcript, detected
