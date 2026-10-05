"""One place that turns text into speech, and one setting that picks the voice.

The voice used to be decided in two unrelated places and neither was a
setting. Spoken turns went to Sarvam's Bulbul with the speaker hardcoded
to "shubh" (bella/sarvam_client.py), while the terminal and the web UI used
Piper's en_GB-alan-medium — a British voice — so Bella sounded like two
different people depending on which front end you talked to.

Now: `tts_voice` in config.json, and everything asks this module.

The rule, by language:

    Assamese / Hindi  ->  Sarvam Bulbul, `tts_voice_indic`
    English           ->  `tts_provider_english`

Engines
-------
- **elevenlabs** — the English voice. Flash is built for realtime, and a
  voice you actually want to listen to is the whole point.
- **sarvam** — Bulbul, `tts_voice_indic`. The only provider here that can
  say the Indian languages at all, and the English fallback while
  ElevenLabs is unreachable.
- **macos** — locally installed voices, no network in the path. Fastest
  by a wide margin (Rishi 0.88s against Sarvam's 1.64s on the same
  sentence) and it cannot rate-limit or time out — but it is
  concatenative and sounds it, so it is opt-in, not a default.

Nothing here decides WHAT to say, only how it sounds. Routing, memory and
the streaming logic are untouched by this file.
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
import wave

from bella import bhashini_client, bhashini_feedback, elevenlabs_client, sarvam_client
from bella.config import load_config

# Voices this Mac ships that are worth defaulting to. `say -v '?'` lists
# what is actually installed; these are the en_IN ones.
MACOS_INDIAN_VOICES = ("Rishi", "Tara", "Aman")

# Speakers bulbul:v3 accepts. NOT the same as the list Sarvam returns for
# an unknown name — that one has 44 entries covering every model, and six
# of them (anushka, abhilash, manisha, vidya, arya, karun, hitesh) are
# rejected by v3 with "not compatible with model bulbul:v3". Taken from
# the compatibility error itself, so it is the list that actually works.
SARVAM_SPEAKERS = (
    "aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja", "rohan",
    "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun",
    "manan", "sumit", "roopa", "kabir", "aayan", "shubh", "advait", "anand",
    "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti", "suhani",
    "mohit", "kavitha", "rehan", "soham", "rupali", "niharika",
)


class SpeechError(Exception):
    pass


# An ElevenLabs 401/403 is a fact about the key, not a blip: it will fail
# identically on every turn. Retrying it cost a measured 3.5s of dead
# round-trip before each fallback, so the first one latches and the rest
# of the process skips straight to the fallback voice. Cleared by a
# restart, which is also what enabling the key's permissions needs.
_elevenlabs_denied = ""


def _reset_elevenlabs_denial() -> None:
    """For tests and checkpoints, which change the key mid-process."""
    global _elevenlabs_denied
    _elevenlabs_denied = ""


def _is_english(language_code: str) -> bool:
    return (language_code or "en-IN").lower().startswith("en")


def assamese_speech_backend(cfg: dict | None = None) -> str:
    """'bhashini' or 'sarvam' — which provider hears/speaks Assamese."""
    cfg = cfg if cfg is not None else load_config()
    pref = (cfg.get("assamese_speech_provider") or "auto").lower()
    if pref == "sarvam":
        return "sarvam"
    if pref == "bhashini":
        return "bhashini" if bhashini_client.configured() else "sarvam"
    return "bhashini" if bhashini_client.configured() else "sarvam"


def describe(cfg: dict | None = None) -> str:
    cfg = cfg if cfg is not None else load_config()
    provider = (cfg.get("tts_provider_english") or "elevenlabs").lower()
    if provider == "elevenlabs":
        english = f"ElevenLabs · {cfg.get('tts_voice_id')} ({cfg.get('elevenlabs_model')})"
        fallback = (cfg.get("tts_english_fallback") or "sarvam").lower()
        where = (f"Sarvam · {cfg.get('tts_voice')}" if fallback == "sarvam"
                 else f"macOS · {cfg.get('tts_voice')}")
        if not elevenlabs_client.configured():
            english += f" [no key — falling back to {where}]"
        else:
            english += f" (falls back to {where})"
    elif provider == "sarvam":
        english = f"Sarvam · {cfg.get('tts_voice')}"
    else:
        english = f"macOS · {cfg.get('tts_voice')}"
    if assamese_speech_backend(cfg) == "bhashini":
        assamese = "Bhashini · Udyat ASR/TTS"
    else:
        assamese = f"Sarvam · {cfg.get('tts_voice_indic')}"
    return f"English: {english}; Assamese: {assamese}"


def installed_macos_voices() -> list[str]:
    """Voice names `say` will actually accept on this Mac."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    names = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and "_" in parts[1]:
            names.append(parts[0])
    return names


def resolve_macos_voice(wanted: str) -> str:
    """The voice to actually use, warning if the configured one is absent.

    `say` does NOT fail on an unknown voice — measured: `say -v Nonexistent`
    exits 0 and quietly synthesizes in the system default, which is how a
    typo in config would silently put an American or British voice back in
    Bella's mouth with nothing in the log to explain it.
    """
    installed = installed_macos_voices()
    if not installed or wanted in installed:
        return wanted
    for candidate in MACOS_INDIAN_VOICES:
        if candidate in installed:
            print(f"[speech] voice {wanted!r} is not installed — using {candidate!r}. "
                  f"`say -v '?'` lists what is available.", flush=True)
            return candidate
    print(f"[speech] voice {wanted!r} is not installed and no Indian-English "
          f"voice was found — using the system default.", flush=True)
    return wanted


def _macos_say(text: str, voice: str, rate: int, timeout: int = 30) -> bytes:
    """Synthesize with a locally installed macOS voice.

    LEI16@22050 rather than the default AIFF: bella/voice_events.py opens
    these bytes with the `wave` module to measure the amplitude envelope
    for the HUD orb, and it only accepts 16-bit PCM.
    """
    path = tempfile.mktemp(suffix=".wav")
    try:
        result = subprocess.run(
            ["say", "-v", voice, "-r", str(int(rate)),
             "--data-format=LEI16@22050", "-o", path, text],
            capture_output=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise SpeechError(
                (result.stderr or b"").decode(errors="replace").strip()
                or f"`say` could not use the voice {voice!r}"
            )
        data = open(path, "rb").read()
        if not data:
            raise SpeechError("`say` produced no audio")
        return data
    except subprocess.TimeoutExpired as e:
        raise SpeechError("`say` timed out") from e
    except OSError as e:
        raise SpeechError(f"could not run `say` ({e})") from e
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def playable(wav_bytes: bytes) -> bool:
    """True if these bytes are 16-bit PCM the rest of the pipeline can use."""
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            return w.getsampwidth() == 2 and w.getframerate() > 0
    except Exception:
        return False


def synthesize(
    text: str,
    language_code: str = "en-IN",
    cfg: dict | None = None,
    source: str = "voice",
) -> bytes:
    """WAV bytes for `text`, in whichever voice the language calls for.

    The whole routing rule, in one place:

        Assamese  ->  Bhashini (when configured) or Sarvam Bulbul
        Hindi     ->  Sarvam Bulbul
        English   ->  tts_provider_english

    macOS ships no Assamese voice and the ElevenLabs voice configured for
    English is an English voice, so Indian-language speech goes to cloud
    providers. Assamese prefers Bhashini when the Udyat inference key is
    in bella/.env; Sarvam remains the fallback and handles Hindi.

    When ElevenLabs cannot answer, English falls back to whatever
    `tts_english_fallback` names. It is "sarvam" rather than "macos"
    because the macOS voices are concatenative and audibly robotic; a
    fallback nobody wants to listen to is not a fallback. Set it to
    "macos" for a path with no network in it at all.
    """
    clean = (text or "").strip()
    if not clean:
        return b""
    cfg = cfg if cfg is not None else load_config()

    if not _is_english(language_code):
        if language_code == "as-IN" and assamese_speech_backend(cfg) == "bhashini":
            try:
                sid = bhashini_client.service_id("tts")
                wav = bhashini_client.text_to_speech(clean)
                if playable(wav):
                    bhashini_feedback.record_call(
                        "tts",
                        input_text=clean,
                        output_text="[audio]",
                        service_id=sid,
                        source=source,
                        cfg=cfg,
                    )
                    return wav
                raise bhashini_client.BhashiniError(
                    "returned audio that is not 16-bit PCM"
                )
            except bhashini_client.BhashiniError as e:
                print(f"[speech] Bhashini TTS unavailable ({e}) — Sarvam", flush=True)
                try:
                    bhashini_feedback.record_fallback(
                        "tts",
                        str(e),
                        input_text=clean,
                        service_id=bhashini_client.service_id("tts"),
                        source=source,
                        cfg=cfg,
                    )
                except bhashini_client.BhashiniError:
                    pass
        return sarvam_client.text_to_speech(
            clean, language_code=language_code,
            speaker=(cfg.get("tts_voice_indic") or sarvam_client.DEFAULT_TTS_SPEAKER),
        )

    global _elevenlabs_denied
    provider = (cfg.get("tts_provider_english") or "elevenlabs").lower()
    fallback = (cfg.get("tts_english_fallback") or "sarvam").lower()
    if provider == "elevenlabs" and _elevenlabs_denied:
        provider = fallback
    if provider == "elevenlabs":
        try:
            wav = elevenlabs_client.text_to_speech(
                clean,
                cfg.get("tts_voice_id") or "",
                cfg.get("elevenlabs_model") or elevenlabs_client.DEFAULT_MODEL,
            )
            if playable(wav):
                return wav
            raise elevenlabs_client.ElevenLabsError("returned audio that is not 16-bit PCM")
        except elevenlabs_client.ElevenLabsError as e:
            if e.status in (401, 403):
                _elevenlabs_denied = str(e)
                print(f"[speech] ElevenLabs rejected the key ({e}) — using "
                      f"{fallback} for the rest of this session. Enable the "
                      f"key's text_to_speech permission and restart.", flush=True)
            else:
                print(f"[speech] ElevenLabs unavailable ({e}) — falling back to "
                      f"{fallback}", flush=True)
            if fallback == "sarvam":
                return _sarvam_english(clean, language_code, cfg)
    elif provider == "sarvam":
        return _sarvam_english(clean, language_code, cfg)

    return _macos_english(clean, cfg)


def _sarvam_english(text: str, language_code: str, cfg: dict) -> bytes:
    return sarvam_client.text_to_speech(
        text, language_code=language_code,
        speaker=(cfg.get("tts_voice") or sarvam_client.DEFAULT_TTS_SPEAKER),
    )


def _macos_english(text: str, cfg: dict) -> bytes:
    voice = resolve_macos_voice(cfg.get("tts_voice") or MACOS_INDIAN_VOICES[0])
    try:
        wav = _macos_say(text, voice, cfg.get("tts_rate", 180))
        # Checked rather than assumed: afplay and the HUD orb's envelope
        # both need 16-bit PCM, and a voice producing anything else would
        # fail later and further away, where it is harder to attribute.
        if not playable(wav):
            raise SpeechError(f"{voice} produced audio that is not 16-bit PCM")
        return wav
    except SpeechError as e:
        print(f"[speech] {voice} unavailable ({e})", flush=True)
        raise


