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
import re
import struct
import time
import urllib.error
import urllib.request
import wave

from max.env import load_env

API_URL = "https://api.sarvam.ai/v1/chat/completions"
TTS_URL = "https://api.sarvam.ai/text-to-speech"
# Same synthesis, but the audio comes back as a raw binary stream instead
# of base64 inside a JSON body that is only sent once the whole clip is
# finished. A/B interleaved on this key, median of 5 each (interleaved
# because sequential runs drift badly — a naive back-to-back comparison
# made the 12-char case look like a regression when it is not):
#
#   chars    REST    stream    delta
#      12   1.45s     1.05s   -0.39s
#      64   3.67s     2.61s   -1.06s
#     196   7.02s     4.72s   -2.30s
#
# Faster at every length, and the tail is what really differs: REST's worst
# of five at 196 chars was 16.3s, the stream's was 5.8s. For something the
# user is sitting in silence waiting for, that consistency matters more
# than the median.
#
# We still read the whole response before playing. Time-to-first-BYTE is a
# flat ~0.5s regardless of length, so piping straight into a player would
# be faster again — but afplay cannot read stdin, and introducing a second
# audio subsystem into the voice loop is precisely what has repeatedly
# killed the microphone before (see _ack_tone_path in voice_loop.py). Not
# worth 0.3s. Buffered streaming keeps every byte of that risk at zero.
TTS_STREAM_URL = "https://api.sarvam.ai/text-to-speech/stream"
STT_URL = "https://api.sarvam.ai/speech-to-text"

# Kill switch. Flip to False to go back to the plain REST endpoint if the
# streamed audio ever sounds worse — auto-fallback below only covers
# errors, not quality.
_TTS_USE_STREAM = True
_STREAM_DISABLED_UNTIL = 0.0
_STREAM_DISABLE_SECONDS = 300
# Conversations variant is post-trained for voice/chat. Flagship 105b
# thinks first (reasoning_effort default medium) and cost us 3–8s of
# silence on every Assamese turn. Fall back if this ID is not on the key.
DEFAULT_MODEL = "sarvam-105b-conversations"
FALLBACK_MODEL = "sarvam-105b"
DEFAULT_TTS_MODEL = "bulbul:v3"
DEFAULT_TTS_SPEAKER = "shubh"
DEFAULT_STT_MODEL = "saaras:v3"


class SarvamError(Exception):
    """`status` is the HTTP status when the failure came from a response.

    It exists so callers can tell "this request was wrong" (4xx — the same
    request will fail the same way anywhere) from "the transport had a bad
    day" (5xx, timeout, DNS). The TTS stream/REST fallback below depends on
    that distinction: retrying a 400 on the other endpoint just doubles the
    latency of a failure that was always going to happen.
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _api_key() -> str:
    key = load_env().get("SARVAM_API_KEY", "")
    if not key:
        raise SarvamError("SARVAM_API_KEY not set in max/.env")
    return key


def _post_json(url: str, body: dict, timeout: int) -> dict:
    payload = json.dumps(body).encode()
    key = _api_key()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "api-subscription-key": key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SarvamError(f"Sarvam API error {e.code}: {detail}") from e
    # TimeoutError is NOT a URLError. urllib only wraps a timeout that
    # happens while connecting; one that happens mid-read comes out bare,
    # so leaving it uncaught let a slow response escape as something no
    # caller was written to handle.
    except (urllib.error.URLError, TimeoutError) as e:
        raise SarvamError(f"Could not reach Sarvam API ({e})") from e
    except json.JSONDecodeError as e:
        raise SarvamError(f"Sarvam returned a malformed response ({e})") from e


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 400,
    timeout: int = 25,
    temperature: float = 0.6,
    reasoning: bool = False,
) -> str:
    """Voice/Assamese chat. Reasoning is off by default — Sarvam-105B
    otherwise burns seconds on a hidden chain-of-thought before the
    spoken line (seen live: 3–8s llm, ~940 tokens to say hello).
    """
    last_err: SarvamError | None = None
    for try_model in (model, FALLBACK_MODEL) if model != FALLBACK_MODEL else (model,):
        try:
            return _chat_request(
                messages,
                try_model,
                max_tokens,
                timeout,
                temperature,
                reasoning,
            )
        except SarvamError as e:
            last_err = e
            detail = str(e).lower()
            if try_model != FALLBACK_MODEL and re.search(
                r"unknown model|invalid model|model_not_found|does not exist",
                detail,
            ):
                print(f"(Sarvam {try_model} unavailable — {FALLBACK_MODEL})", flush=True)
                continue
            raise
    raise last_err or SarvamError("Sarvam chat failed")


def _chat_request(
    messages: list[dict],
    model: str,
    max_tokens: int,
    timeout: int,
    temperature: float,
    reasoning: bool,
) -> str:
    body = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Docs: disable thinking by sending null. Some keys reject the
        # field — retry without it rather than sit in medium reasoning.
        "reasoning_effort": "low" if reasoning else None,
    }
    started = time.perf_counter()
    try:
        data = _post_json(API_URL, body, timeout)
    except SarvamError as e:
        if not reasoning and "reasoning_effort" in body and "400" in str(e):
            body.pop("reasoning_effort", None)
            data = _post_json(API_URL, body, timeout)
        else:
            raise

    choice = data["choices"][0]
    content = choice["message"].get("content")
    if not content:
        reason = choice.get("finish_reason", "unknown")
        raise SarvamError(
            f"Sarvam returned no content (finish_reason={reason!r}) — likely truncated "
            "mid-reasoning; try a larger max_tokens."
        )
    usage = data.get("usage") or {}
    elapsed = time.perf_counter() - started
    print(
        f"  (sarvam chat {model} {elapsed:.2f}s "
        f"out={usage.get('completion_tokens', '?')} "
        f"in={usage.get('prompt_tokens', '?')})",
        flush=True,
    )
    return content


_as_in_tts_ok: bool | None = None
_tts_wav_cache: dict[tuple[str, str], bytes] = {}
# language_code -> (effective_lang, model, speaker). Without this, every
# spoken sentence retried 3 dead as-IN voices (shubh is not on v2) then
# bn-IN. Live: ttfw 2–4.5s and gaps between joke lines.
_tts_route: dict[str, tuple[str, str, str]] = {}
# as-IN is only on bulbul:v3. Extra models just add failed round-trips
# before the bn-IN fallback (seen live: v2 said "as-IN is only supported
# by bulbul:v3").
_AS_IN_TRIES = (("bulbul:v3", "shubh"),)


def text_to_speech(
    text: str,
    language_code: str = "hi-IN",
    model: str = DEFAULT_TTS_MODEL,
    speaker: str = DEFAULT_TTS_SPEAKER,
    timeout: int = 30,
) -> bytes:
    """Returns raw WAV bytes. as-IN is probed once, then the working
    route is reused so later sentences do not wait on failed retries.
    """
    cache_key = (language_code, text.strip())
    if len(cache_key[1]) <= 80 and cache_key in _tts_wav_cache:
        return _tts_wav_cache[cache_key]

    def _keep(wav: bytes) -> bytes:
        if len(cache_key[1]) <= 80:
            _tts_wav_cache[cache_key] = wav
        return wav

    cached = _tts_route.get(language_code)
    if cached:
        use_lang, use_model, use_speaker = cached
        return _keep(_tts_request(text, use_lang, use_model, use_speaker, timeout))

    if language_code != "as-IN":
        wav = _tts_request(text, language_code, model, speaker, timeout)
        _tts_route[language_code] = (language_code, model, speaker)
        return _keep(wav)

    global _as_in_tts_ok
    last_err = None
    for try_model, try_speaker in _AS_IN_TRIES:
        try:
            wav = _tts_request(text, "as-IN", try_model, try_speaker, timeout)
            _tts_route["as-IN"] = ("as-IN", try_model, try_speaker)
            _as_in_tts_ok = True
            return _keep(wav)
        except SarvamError as e:
            last_err = e
            continue
    print(
        f"(as-IN TTS blocked: {last_err} — using bn-IN for Assamese script)",
        flush=True,
    )
    wav = _tts_request(text, "bn-IN", model, speaker, timeout)
    _tts_route["as-IN"] = ("bn-IN", model, speaker)
    _as_in_tts_ok = False
    return _keep(wav)


def _tts_body(text: str, language_code: str, model: str, speaker: str) -> dict:
    return {
        "text": text,
        "language_code": language_code,
        "model": model,
        "speaker": speaker,
    }


def _tts_headers() -> dict:
    key = _api_key()
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "api-subscription-key": key,
    }


def _fix_wav_header(audio: bytes) -> bytes:
    """Write the real lengths into a streamed WAV header.

    The streaming endpoint cannot know the clip length when it sends the
    header, so it ships the standard placeholder: RIFF size 0xFFFFFFFF,
    which reads back as 2147483647 frames — `wave` reports a 3-second clip
    as 97391 seconds long. afplay copes (verified: exit 0, correct
    duration), but max/voice_events.py opens the same bytes with `wave` to
    measure the amplitude envelope for the HUD orb, and anything else that
    trusts getnframes() would be equally misled. Patching the two size
    fields makes the buffer an ordinary, self-describing WAV — identical in
    shape to what the REST path has always returned.
    """
    if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        return audio
    out = bytearray(audio)
    struct.pack_into("<I", out, 4, len(out) - 8)
    pos = 12
    while pos + 8 <= len(out):
        chunk_id = bytes(out[pos:pos + 4])
        size = struct.unpack_from("<I", out, pos + 4)[0]
        if chunk_id == b"data":
            struct.pack_into("<I", out, pos + 4, len(out) - (pos + 8))
            break
        if size in (0, 0xFFFFFFFF):
            break  # another unbounded chunk; leave the rest alone
        pos += 8 + size + (size & 1)
    return bytes(out)


def _tts_stream_request(
    text: str, language_code: str, model: str, speaker: str, timeout: int
) -> bytes:
    body = _tts_body(text, language_code, model, speaker)
    # Without this the endpoint returns MP3, which `wave` cannot open and
    # the orb's envelope needs.
    body["output_audio_codec"] = "wav"
    req = urllib.request.Request(
        TTS_STREAM_URL, data=json.dumps(body).encode(), headers=_tts_headers()
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            audio = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SarvamError(f"Sarvam TTS error {e.code}: {detail}", status=e.code) from e
    except (urllib.error.URLError, TimeoutError) as e:  # see _post_json
        raise SarvamError(f"Could not reach Sarvam TTS stream ({e})") from e
    if not audio:
        raise SarvamError("Sarvam TTS stream returned no audio")
    return _fix_wav_header(audio)


def _tts_rest_request(
    text: str, language_code: str, model: str, speaker: str, timeout: int
) -> bytes:
    req = urllib.request.Request(
        TTS_URL,
        data=json.dumps(_tts_body(text, language_code, model, speaker)).encode(),
        headers=_tts_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SarvamError(f"Sarvam TTS error {e.code}: {detail}", status=e.code) from e
    except (urllib.error.URLError, TimeoutError) as e:  # see _post_json
        raise SarvamError(f"Could not reach Sarvam TTS API ({e})") from e
    except json.JSONDecodeError as e:
        raise SarvamError(f"Sarvam TTS returned a malformed response ({e})") from e

    audios = data.get("audios") or []
    if not audios:
        raise SarvamError("Sarvam TTS returned no audio")
    try:
        return base64.b64decode(audios[0])
    except (ValueError, TypeError) as e:
        raise SarvamError(f"Sarvam TTS returned undecodable audio ({e})") from e


def _tts_request(
    text: str,
    language_code: str,
    model: str,
    speaker: str,
    timeout: int,
) -> bytes:
    """Streaming endpoint first, REST as the safety net."""
    global _STREAM_DISABLED_UNTIL
    if _TTS_USE_STREAM and time.monotonic() >= _STREAM_DISABLED_UNTIL:
        try:
            return _tts_stream_request(text, language_code, model, speaker, timeout)
        except SarvamError as e:
            # 4xx is a verdict on the request itself — a rejected speaker,
            # or the as-IN beta gate that text_to_speech below relies on
            # seeing. REST would reject it identically, so re-trying only
            # doubles the wait before the caller's own fallback runs.
            if e.status is not None and 400 <= e.status < 500:
                raise
            _STREAM_DISABLED_UNTIL = time.monotonic() + _STREAM_DISABLE_SECONDS
            print(f"(Sarvam TTS stream unavailable: {e} — using REST for "
                  f"{_STREAM_DISABLE_SECONDS // 60} minutes)", flush=True)
    return _tts_rest_request(text, language_code, model, speaker, timeout)


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
    except (urllib.error.URLError, TimeoutError) as e:  # see _post_json
        raise SarvamError(f"Could not reach Sarvam STT API ({e})") from e
    except json.JSONDecodeError as e:
        raise SarvamError(f"Sarvam STT returned a malformed response ({e})") from e

    transcript = (data.get("transcript") or "").strip()
    detected = data.get("language_code") or None
    # Saaras often emits Odia (or-IN) for Assamese speech — same
    # Brahmi shape, wrong block. Convert before anyone else sees it.
    if _has_odia(transcript) or (detected or "").startswith(("or", "od")):
        transcript = odia_to_assamese(transcript)
        if detected and not str(detected).startswith("as"):
            print(f"  (Saaras wrote {detected} — folded to Assamese script)", flush=True)
            detected = "as-IN"
    return transcript, detected


_ODIA_START, _ODIA_END = 0x0B00, 0x0B7F
_BENG_START = 0x0980


def _has_odia(text: str) -> bool:
    return any(_ODIA_START <= ord(ch) <= _ODIA_END for ch in text or "")


def odia_to_assamese(text: str) -> str:
    """Map Odia code points onto the Bengali/Assamese block.

    The two blocks are laid out in parallel (offset 0x80). Assamese
    uses that block; Odia on an Assamese turn is a recognizer mix-up,
    not a language switch.
    """
    out: list[str] = []
    for ch in text or "":
        code = ord(ch)
        if _ODIA_START <= code <= _ODIA_END:
            out.append(chr(code - _ODIA_START + _BENG_START))
        else:
            out.append(ch)
    return "".join(out)
