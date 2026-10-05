"""Always-on voice loop: continuous mic capture -> wake phrase ("Hey Bella")
-> record utterance -> local transcription -> normal Bella pipeline -> TTS.

Everything here is local: the wake phrase is spotted by VAD + a small
Whisper (bella/wake_phrase.py, so the phrase is whatever config says),
and faster-whisper transcribes the utterance. Wispr Flow isn't used in this
mode — it's a dictation tool bound to a focused text field, not an ambient
background listener, so it can't drive this loop. The terminal `cli.py`
loop (where Wispr dictation IS used) is unaffected by any of this.

Run with: python3 -m bella.voice_loop
"""
import os
import queue
import subprocess
import sys
import tempfile
import time
import threading
import traceback
import re
import signal
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel

from bella import cli, composio_client, control, hotkey, llm, sarvam_client, speech, voice_events, wake_phrase
from bella.config import load_config
from bella.memory import MemoryStore
from bella.tools import spotify
from bella.vad import SileroVAD, CHUNK_SAMPLES as VAD_CHUNK_SAMPLES

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80ms — openWakeWord's native chunk size
WAKE_THRESHOLD = 0.5

# The InputStream's callback should fire roughly every FRAME_SAMPLES /
# SAMPLE_RATE (~80ms) forever, regardless of room silence — a real audio
# frame still gets delivered even when nothing's making sound. A gap this
# long with zero frames delivered is unambiguous evidence the stream died,
# not that the room is quiet. Seen live: unplugging a mic/headset produces
# a macOS CoreAudio error (-10863, kAudioUnitErr_CannotDoInCurrentContext)
# that isn't raised as a catchable Python exception — the callback just
# stops firing forever, and the old code (a single blocking
# audio_q.get() with no timeout) hung the whole process indefinitely.
_STREAM_DEAD_TIMEOUT = 5.0


class _StreamDead(Exception):
    """Raised by VoiceLoop._next_frame when the input stream has stopped
    delivering frames — caught in run() to close the dead stream and open
    a fresh one against whatever the current default input device is now
    (the OS will have already picked a new default after a device change)."""


class _MicMuted(Exception):
    """Raised the moment the microphone is switched off, to unwind out of
    whatever the loop was doing and close the input stream.

    Muting has to actually release the device, not merely ignore what it
    hears. A flag that drops frames still leaves an open InputStream, the
    macOS microphone indicator still lit, and no way for the user to
    verify the claim. Closing the stream turns the indicator off — which
    is the only evidence of "not listening" that doesn't require trusting
    this code."""


class _VoiceLoopShutdown(Exception):
    """Raised when the user asks to fully exit (SHUTDOWN_RE) — distinct
    from _StreamDead so run()'s outer retry loop knows to actually stop
    instead of reopening the stream and continuing."""
# Silero's own recommended operating point — unlike the RMS threshold this
# replaces for utterance-recording, it's a property of the trained model,
# not something that needs per-room calibration.
VAD_SPEECH_THRESHOLD = 0.5

MAX_UTTERANCE_SECONDS = 15
CONFIRM_MAX_SECONDS = 6
# How long to wait for speech to START before giving up on a listening
# phase. Without this, record_utterance's only exit was "heard speech, then
# heard silence" — so a phase where the user simply didn't speak ran the
# FULL max_seconds. Measured live over 416 turns: 293 of them (70%) ended
# in "didn't catch anything", each burning 15s of recording plus 10-29s of
# Whisper chewing through 15s of pure silence — a 25-44 second window where
# Bella looks completely dead before it says anything at all. That is the
# "Bella is sleeping / sometimes it just doesn't respond" symptom, and it
# is a bug, not a tuning problem. Real assistants give you a few seconds to
# start talking and then quietly stand down.
NO_SPEECH_BAIL_SECONDS = 3.0
# ...but only for speculative listens (the follow-up window, where usually
# nobody is there). Straight after a wake word the user has explicitly said
# they want to talk, so cutting them off at 3s is wrong — that is what
# "I say the wake phrase and it doesn't respond" actually was: the wake
# fired, the user paused to gather the sentence, and Bella had already
# stopped listening with no cue that it ever started. Field-tested
# Whisper-assistant configs sit around 12s before returning to idle.
WAKE_LISTEN_SECONDS = 6.0
# How long after Bella stops speaking its own voice can still be arriving
# at the microphone. Beyond this the room is quiet and anything heard is
# the user, however much it happens to sound like the last reply.
ECHO_WINDOW_SECONDS = 4.0
# Shortest buffered clip a push-to-talk press may be treated as a complete
# command. Not a tuning knob: below this there is physically no room for
# one, and accepting it costs a whole wasted turn. See the use site.
_PTT_MIN_COMMAND_SAMPLES = int(0.8 * SAMPLE_RATE)
# Word-boundary contains-match, not exact set membership — seen live:
# exact matching missed "Hey stop!" (normalizes to "hey stop", which isn't
# in the set) and sent it to Ollama as a normal query instead of
# interrupting. People naturally wrap "stop" in filler ("hey", "please",
# "can you"), so the phrase needs to be found anywhere in the utterance,
# not be the whole thing. SHUTDOWN is checked first in run() and `break`s
# immediately, so "stop listening" is claimed by SHUTDOWN_RE before
# INTERRUPT_RE's bare "stop" ever gets a chance to also match it.
INTERRUPT_RE = re.compile(
    r"\b(?:stop(?:\s+(?:it|that|talking))?|be\s+quiet|quiet|silence|shut\s+up|enough)\b")

# "Stop" with an object is a command about that object, not an interrupt.
# Checked before INTERRUPT_RE, because the interrupt check happens before
# routing: without this, "stop the music" was swallowed as "be quiet" and
# never reached the Spotify pause path it matches in router.py — the
# music kept playing. "Don't stop" is here for the obvious reason.
_NOT_AN_INTERRUPT_RE = re.compile(
    r"\b(?:do\s*n'?t|dont|never)\s+stop\b"
    r"|\bstop\s+(?:the\s+|my\s+|this\s+)?"
    r"(?:music|song|track|playback|spotify|player|timer|alarm|recording|sharing)\b"
)
SHUTDOWN_RE = re.compile(r"\b(?:stop\s+listening|quit|exit|shutdown|go\s+to\s+sleep)\b")

# Spoken "speak in Assamese / English" is mapped to /assamese and
# /english in cli.apply_language_trigger (same function process_turn
# uses, so HUD-typed Assamese "স্পিক ইংলিছ" also switches).


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", "", text.lower()).strip()


def _fold_wake_text(text: str) -> str:
    """Keep Assamese letters. Latin-only normalize used to turn
    'বেলা।' into '' so name-only never counted as a wake."""
    t = (text or "").replace("\u0964", " ")
    t = t.lower()
    # Keep the Bengali/Assamese block: virama and matras are not \w,
    # so a generic strip turned বেলা into ম য ক স.
    t = re.sub(r"[^\w\s\u0980-\u09FF]+", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


_ASSAMESE_NAME_ONLY_RE = re.compile(
    r"^(?:(?:হে(?:ই)?|হায়)\s*)?(?:বেলা|বেল্লা|বেলাহ)$"
)
_PUNCT_ONLY_RE = re.compile(r"^[\s!,.?\u0964\-–—]+$")


# Exact normalized forms only (not a fuzzy/substring check) — deliberately
# narrow, so a genuine follow-up that happens to start with the name (e.g.
# "Bella, can you check my email") is never swallowed by this, only a
# bare repeat of the wake phrase itself. Covers the STT variants actually
# seen live for the wake phrase (accent-stripped, fused words).
# Rebuilt from config at startup by configure_wake_phrases(); these are
# the defaults for "hey bella", including how Whisper actually renders it
# (measured: "Hey Oren", "Hey, Orange.", "Our in").
_BARE_WAKE_PHRASES: set[str] = set()
_LEADING_WAKE_PHRASE_RE: re.Pattern | None = None

# One table, shared with the detector — these are the same STT spellings,
# and two copies would drift the moment either is tuned.
_NAME_MISHEARINGS = wake_phrase.NAME_MISHEARINGS


def configure_wake_phrases(phrases: list[str]) -> None:
    """Point the follow-up/echo handling at the configured wake phrase.

    These two checks are about what the user *says mid-conversation*, not
    about waking: repeating the name during an open window shouldn't be
    sent to the LLM as a question, and a repeated name at the start of a
    command shouldn't end up baked into the command text. Both need the
    same list of spellings the STT produces, so they are derived from the
    same config as the detector rather than hardcoded a second time.
    """
    global _BARE_WAKE_PHRASES, _LEADING_WAKE_PHRASE_RE
    leads = ["hey", "hi", "hej", "hello", "ok", "okay"]
    bare: set[str] = set()
    names: set[str] = set()
    for phrase in phrases:
        words = _normalize_for_compare(phrase).split()
        if not words:
            continue
        name = words[-1]
        extra = list(_NAME_MISHEARINGS.get(name, []))
        if name == "bella":
            extra.extend(["বেলা", "বেল্লা", "বেলাহ"])
        for spelling in [name, *extra]:
            names.add(spelling)
            bare.add(spelling)
            bare.add(spelling.replace(" ", ""))
            for lead in leads:
                bare.add(f"{lead} {spelling}")
                bare.add(f"{lead}{spelling}".replace(" ", ""))
            for lead in ("হে", "হেই", "হায়"):
                if re.search(r"[\u0980-\u09FF]", spelling):
                    bare.add(f"{lead} {spelling}")
                    bare.add(f"{lead}{spelling}")
    _BARE_WAKE_PHRASES = bare
    escaped = "|".join(sorted((re.escape(n) for n in names), key=len, reverse=True))
    as_leads = r"হেই|হে|হায়"
    punct = r"[!,.?\u0964]*"
    _LEADING_WAKE_PHRASE_RE = re.compile(
        rf"^(?:(?:{'|'.join(leads)}|{as_leads})[,!.\u0964]?\s*)?"
        rf"(?:{escaped})(?:\b|{punct}){punct}\s*",
        re.IGNORECASE,
    )


configure_wake_phrases(["hey bella", "bella"])


def _is_bare_wake_phrase(text: str) -> bool:
    """True if text is just the wake phrase repeated and nothing else.

    Seen live: saying the wake phrase again while a conversation window is
    already open gets transcribed as ordinary follow-up speech (no wake-
    word model involved — that only runs when NOT already in an active
    window) and sent straight to the LLM, which answers as if freshly
    greeted ("Hello! How can I assist you today?") — redundant and a bit
    jarring when the conversation was already live. Only checked when
    already in an active window (see _run_one_turn); saying the wake
    phrase to genuinely start a fresh conversation is unaffected.
    """
    folded = _fold_wake_text(text)
    if not folded:
        return False
    if folded in _BARE_WAKE_PHRASES:
        return True
    if _normalize_for_compare(text) in _BARE_WAKE_PHRASES:
        return True
    return bool(_ASSAMESE_NAME_ONLY_RE.fullmatch(folded))


def _is_empty_command(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if _PUNCT_ONLY_RE.fullmatch(t):
        return True
    return _is_bare_wake_phrase(t)


_EN_CHAT_RE = re.compile(
    r"\b(?:how are you|how're you|what's up|whats up|thank(?:s| you)|"
    r"good (?:morning|afternoon|evening|night)|hello|hi there|"
    r"i love you|love you)\b",
    re.IGNORECASE,
)


def _command_from_wake_transcript(wake_tr: str, token: str = "") -> str | None:
    """The command already in the wake clip, if it is safe to skip a second STT.

    Seen live: wake heard "Bella, what's on my screen?" in 0.91s, then
    Saaras ran 2.5–8s on the same audio and wrote Assamese that missed
    the screen tool. Reuse the wake line when it is already a real
    English command. Mangled Assamese ("Matoumak Bhaal Pao") must still
    go to Saaras.
    """
    if not (wake_tr or "").strip():
        return None
    text = _strip_matched_name(wake_tr, token)
    text = _strip_leading_wake_phrase(text)
    text = wake_phrase.strip_address_name(text)
    if not text or _is_empty_command(text):
        return None
    if any("\u0980" <= c <= "\u09FF" for c in text):
        return text
    from bella import router
    from bella.tools import vision
    if router.detect_tool(text):
        return text
    if vision.is_question_help(text) or vision.is_howto(text) or vision.is_advice_request(text):
        return text
    if _EN_CHAT_RE.search(text):
        return text
    return None


def _strip_matched_name(text: str, token: str) -> str:
    """Remove the exact word the detector accepted as our name.

    The spelling list can only cover mishearings someone has already seen.
    This covers the rest: if the matcher decided a word was the name, that
    word is the name, and it must not reach the model as if the user had
    introduced themselves.
    """
    if not token:
        return text
    # The punctuation between lead word and name matters: speech-to-text
    # writes "Hey, Oren." with a comma, and a pattern that only allowed
    # whitespace there silently matched nothing — which is how a name-only
    # utterance reached the model intact.
    pattern = re.compile(
        rf"^(?:(?:hey|hi|hej|hello|ok|okay)[,!.]?\s*)?{re.escape(token)}\b[!,.?]*\s*",
        re.IGNORECASE,
    )
    # Deliberately no `or text` fallback: an utterance that was *only*
    # the name must come back empty so the caller can treat it as a wake
    # rather than a command. Returning the original text here is what put
    # "Hey, Oren" in front of the model as something to reply to.
    return pattern.sub("", text.strip(), count=1).strip()


def _strip_leading_wake_phrase(text: str) -> str:
    """Strips a re-said wake phrase from the start of an utterance that has
    more to it than just the phrase — same STT variants as
    _BARE_WAKE_PHRASES (accent-stripped, fused words, STT variants of
    "Bella"), but as a prefix rather than a whole-utterance match. Seen
    live: users often repeat the wake phrase out of habit mid-conversation,
    and the misheard variant then ends up baked into the actual command
    text, corrupting deterministic extraction downstream — a shell command
    literally starting with the mangled name plus "run the command, sleep 30" ran
    `Hail` as the command and failed instead of ever running `sleep 30`.
    """
    return _LEADING_WAKE_PHRASE_RE.sub("", text.strip(), count=1).strip()


_ACK_TONE_PATH: str | None = None


def _ack_tone_path() -> str:
    """Path to a short rising two-note chime, rendered once on first use.

    Bella had no way of telling the user it had started listening, so
    after saying the wake phrase you were talking into a system whose state
    you couldn't see — and if you paused first, it had already given up.
    Siri, Alexa and Google all play a tone here for exactly this reason.
    Synthesized with numpy rather than shipping an audio asset.

    Played by handing a file to afplay rather than through sounddevice.
    sd.play() is much quicker (0.29s vs ~1.15s of process spawn) but opens
    an output stream on the shared PortAudio instance, and on a combined
    mic-and-speaker device such as EarPods that makes CoreAudio renegotiate
    and kills our input stream — seen live as "PaMacCore (AUHAL) Error
    -10851: Audio Unit: Invalid Property Value" immediately after the wake
    word, taking the microphone down with it. afplay is a separate process
    and has never disturbed the input stream, which is also why every
    spoken reply already goes out that way.
    """
    global _ACK_TONE_PATH
    if _ACK_TONE_PATH is not None and os.path.exists(_ACK_TONE_PATH):
        return _ACK_TONE_PATH

    def note(freq: float, ms: int, amp: float = 0.22) -> np.ndarray:
        n = int(SAMPLE_RATE * ms / 1000)
        samples = np.sin(2 * np.pi * freq * np.arange(n) / SAMPLE_RATE) * amp
        # Fade the edges or the discontinuity clicks audibly.
        fade = min(int(SAMPLE_RATE * 0.008), n // 2)
        if fade:
            samples[:fade] *= np.linspace(0.0, 1.0, fade)
            samples[-fade:] *= np.linspace(1.0, 0.0, fade)
        return samples

    pcm = (np.concatenate([note(660, 70), note(990, 90)]) * 32767).astype(np.int16)
    path = os.path.join(tempfile.gettempdir(), "max_ack_tone.wav")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    _ACK_TONE_PATH = path
    return path


# Closing quotes/brackets may sit between the terminator and the space.
# Without them a line like: ...Schrodinger's cat?" The librarian replied...
# had NO split point at all, so the "first chunk" was the entire joke and
# the user waited for all of it to synthesize before hearing a word.
# Two branches so the closer stays attached to the sentence it closes —
# consuming it in the split silently dropped the quote from the text.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?][\"'”’)\]])\s+|(?<=[.!?])\s+")
# Fallback break points inside a sentence that is still too long.
_CLAUSE_END_RE = re.compile(r"[,;:—–)\]]\s")

# The models still emit markdown despite being asked not to ("Today is
# **Thursday**"), and it was going to the speech engine verbatim. Strip the
# markup but keep the words — this only ever touches text on its way to
# TTS, never the transcript shown or logged.
_MD_FENCE_RE = re.compile(r"```[\s\S]*?```")
_MD_INLINE_RE = re.compile(r"[*_`~]{1,3}")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def _clean_for_speech(text: str) -> str:
    text = _MD_FENCE_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_BULLET_RE.sub("", text)
    text = _MD_INLINE_RE.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _break_point(sentence: str, limit: int) -> int:
    """Where to cut an over-long sentence: last clause end inside the
    window, else the last word boundary, else the limit."""
    window = sentence[: limit + 1]
    ends = [m.end() for m in _CLAUSE_END_RE.finditer(window)]
    if ends:
        return ends[-1]
    space = window.rfind(" ")
    return space if space > limit // 2 else limit


def _speech_pieces(text: str, limit: int) -> list[str]:
    """Sentences, with any sentence longer than `limit` broken at clause
    and word boundaries. Nothing is discarded — the pieces rejoin to the
    original words."""
    pieces: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split((text or "").strip()):
        sentence = " ".join(sentence.split())
        while len(sentence) > limit:
            cut = _break_point(sentence, limit)
            head, sentence = sentence[:cut].strip(), sentence[cut:].strip()
            if head:
                pieces.append(head)
        if sentence:
            pieces.append(sentence)
    return pieces


# How long the opening chunk may get before it is cut at a clause. Only a
# ceiling for an over-long FIRST sentence — a shorter one is left whole,
# because breaking mid-clause ("...partly cloudy in" / "Jorhat.") to save
# a second sounds worse than the second is worth.
_FIRST_MAX_CHARS = 110

# Per-chunk size ceiling, ramping up. Re-measured against the live Sarvam
# API, median of 5 calls each (the old note here claimed 0.6s + 0.02s/char,
# which no longer holds — and single samples are useless, one 12-char call
# came back in 7.0s against a 1.18s median):
#
#     12 chars  1.18s      104 chars  6.88s
#     52 chars  2.49s      196 chars  7.86s
#
# Steeply superlinear to ~100 chars, then flat, and worst on address- or
# digit-heavy text. So time-to-first-word is decided almost entirely by
# the size of chunk one, and the old code set a *minimum* on it with no
# maximum at all — a reply that was one long sentence became one 200+ char
# chunk and the user sat in silence for 9-11s (measured live from the
# ttfw= field in /tmp/voice_loop.log). These are ceilings instead. The ramp
# matters too: chunk N+1 is synthesized while chunk N plays, so a short
# opener can only cover a similarly small follow-up before the pipeline
# stalls and the reply audibly gaps.
#
_CHUNK_CEILINGS = (60, 130, 260)


def _split_for_speech(
    text: str,
    first_max_chars: int = _FIRST_MAX_CHARS,
    ceilings: tuple[int, ...] = _CHUNK_CEILINGS,
    min_chars: int = 140,
) -> list[str]:
    """Break a reply into chunks for streaming synthesis, smallest first."""
    # Sentences stay whole unless they are enormous; only the opening one
    # is trimmed further, and only if it needs it.
    pieces = _speech_pieces(text, ceilings[-1])
    if not pieces:
        return []
    if len(pieces[0]) > first_max_chars:
        cut = _break_point(pieces[0], first_max_chars)
        head, tail = pieces[0][:cut].strip(), pieces[0][cut:].strip()
        if head and tail:
            pieces[:1] = [head, tail]

    # The opening chunk is exactly one piece. It is the only thing standing
    # between the user and the first word, and a second round trip for the
    # rest is free — it happens while this one is already playing.
    chunks: list[str] = [pieces[0]]
    buf = ""
    for piece in pieces[1:]:
        ceiling = ceilings[min(len(chunks), len(ceilings) - 1)]
        if buf and len(buf) + 1 + len(piece) > ceiling:
            chunks.append(buf)
            buf = piece
            continue
        buf = f"{buf} {piece}".strip() if buf else piece
        if len(buf) >= min_chars:
            chunks.append(buf)
            buf = ""
    if buf:
        # A tiny trailing fragment rides along with the previous chunk
        # rather than costing its own round-trip — but never into the
        # opening chunk, which is the one the user is waiting on. Merging
        # there grew a 64-char opener to 88 to save a round trip that was
        # already free, and paid for it in time-to-first-word.
        if len(chunks) > 1 and len(buf) < 25:
            chunks[-1] = f"{chunks[-1]} {buf}"
        else:
            chunks.append(buf)
    return chunks


# Utterances shorter than this are never classified as an echo at all, and
# below _FUZZY_ECHO_MIN_WORDS the bag-of-words overlap check specifically
# is noise rather than signal. See the use sites in _looks_like_self_echo.
_ECHO_MIN_WORDS = 5
_FUZZY_ECHO_MIN_WORDS = 5


def _looks_like_self_echo(new_text: str, last_reply: str) -> bool:
    """True if new_text is plausibly Bella's own just-spoken reply bleeding
    back through the mic, not a new thing the user said.

    Seen live: no acoustic echo cancellation means the mic can pick up
    Bella's own speaker output. That gets transcribed as if it were a
    fresh utterance, gets sent back through the LLM, gets a reply, and the
    reply itself bleeds through again — a self-sustaining conversation loop
    with nobody talking, each reply drifting further from anything the user
    actually said (this is what "the flow isn't maintained, it's writing
    gibberish" looks like from the outside — it's not incoherent generation,
    it's Bella answering a mangled transcript of itself).

    An exact contiguous-word-run match was the first attempt, but STT on
    bleed-through audio (picked up off-axis, at speaker distance, often
    over the tail of the reply) is noisier than a direct utterance — seen
    live, "As an AI language model" bled through and transcribed as "As in
    the air language," sharing no exact run of 2+ words with the original
    despite obviously being the same echo. Falls back to a majority-of-
    words-appear-anywhere check; the trade-off is a real but rare new
    command that happens to reuse most of the same words as the last reply
    could get misread as an echo. Given the alternative is full
    conversations derailing into self-talk, that trade-off is worth it.

    No longer capped to short utterances — that cap was the bigger bug.
    Seen live: Bella's own longer replies (especially tool results like a
    multi-sentence email summary) bleed through just as often as short
    ones, and a MISSED long echo does far more damage than a missed short
    one. It gets processed as a genuine new request (sometimes
    re-triggering the same tool call, producing the same reply again,
    which then bleeds through again), and it pollutes the LLM's recent
    context with a confused meta-exchange the model then keeps circling
    back to for several further turns regardless of what's actually said
    next — one missed email-summary echo derailed five or six subsequent
    turns into an email/hallucination loop even after the topic moved on.
    Longer text also makes the fuzzy check safer, not riskier: a
    coincidental 60%+ word overlap between a genuine new command and an
    unrelated prior reply gets less likely as length grows, not more.
    The actual fix would be real acoustic echo cancellation (feeding the
    outgoing TTS audio back as a reference signal to subtract from the mic
    input) — a real DSP project, not a text heuristic; this stays a
    mitigation until that exists.
    """
    # Word-level containment, not character-level: a plain substring check
    # matches "yo" inside "...assist YOu today" — same class of mistake as
    # the voice_confirm substring bug (see CONFIRM_YES_RE above).
    new_words = _normalize_for_compare(new_text).split()
    reply_words = _normalize_for_compare(last_reply).split()
    if not new_words or not reply_words:
        return False
    n = len(new_words)
    # Nothing this short is ever treated as an echo. Natural conversation
    # answers a question using the question's own words — Bella asked
    # "...or exploring some music recommendations?", the user said
    # "recommendations", and it was discarded as an echo of the very reply
    # it was answering. Same for "yes", "stop", "30", "command". A one- or
    # two-word bleed-through does get through as a result, but that is a
    # far cheaper mistake than ignoring what the user actually said, which
    # is the complaint that mattered.
    if n < _ECHO_MIN_WORDS:
        return False
    # "yes, can you show me" after Bella offered to show something is a
    # real follow-up. Fuzzy overlap used to throw it away because the
    # reply contained show/you/me — seen live, the pointer never moved.
    joined = " ".join(new_words)
    if re.search(r"\b(show|point|guide|cursor)\b", joined):
        return False
    # Exact run-of-words match: precise, so it needs no further length
    # guard beyond the floor above.
    if any(reply_words[i : i + n] == new_words for i in range(len(reply_words) - n + 1)):
        return True
    # Fuzzy bag-of-words overlap, but only for utterances long enough for
    # 60% overlap to actually mean something. On a 1-3 word utterance it
    # means almost nothing: a lone "30" or "command" scores 1.0 against any
    # reply containing that word and was discarded as an echo — measured
    # live, 36 genuine commands were thrown away this way, which is the
    # "it just ignores what I say" half of the problem. Short utterances
    # now rely on the exact check above.
    if n < _FUZZY_ECHO_MIN_WORDS:
        return False
    reply_word_set = set(reply_words)
    overlap = sum(1 for w in new_words if w in reply_word_set)
    return overlap / len(new_words) >= 0.6


# voice_confirm gates shell commands and file writes, so a false accept here
# is a real-world "it ran a command nobody approved" incident, not just a
# misheard chat reply. Word-boundary regex, not substring "in" — a plain
# `"yes" in text` check matches "yes" inside "yesterday", "sure" inside
# "unsure"/"insure"/"assure", "confirm" inside "unconfirmed"/"reconfirm".
# Negative words are checked first and win outright (covers "no, don't do
# it" and colloquial "yeah, no").
CONFIRM_YES_RE = re.compile(
    r"\b(?:yes|yeah|yep|yup|confirm(?:ed)?|go ahead|do it|please do|sure|affirmative|correct)\b"
)
CONFIRM_NO_RE = re.compile(
    r"\b(?:no|nope|nah|don'?t|dont|cancel|negative|stop|never ?mind|not now|wait)\b"
)
# A deliberate yes/no answer is short. A long transcript that happens to
# contain an affirmative word somewhere — open-mic background chatter, or a
# Whisper hallucination from silence/noise — is not the same thing as
# someone actually answering the question, so it's treated as a decline
# rather than executed.
CONFIRM_MAX_WORDS = 8
# How many times to ask before giving up. A confirmation that heard only
# Bella's own voice has not been answered, and one re-ask costs a few
# seconds where cancelling costs the user the whole request.
CONFIRM_ATTEMPTS = 2
# Let the speakers finish before opening the mic. See the use site.
CONFIRM_SETTLE_SECONDS = 0.35

# The literal instruction appended to every confirm prompt, which is what
# the mic most often catches. Written as a pattern because Whisper
# punctuates it freely: "yes or no.", "yes or no?", "say yes or no".
_CONFIRM_ECHO_RE = re.compile(
    r"\b(?:say\s+)?yes\s+or\s+no\b[\s,.?!—-]*", re.IGNORECASE
)


def _strip_confirm_echo(heard: str, spoken: str) -> str:
    """`heard` with Bella's own confirm prompt removed.

    Measured live: with no acoustic echo cancellation the mic picks up the
    tail of the prompt, so "yes or no? yes." arrived as the transcript of
    a turn the user had answered YES. CONFIRM_NO_RE matched the "no" that
    Bella itself had said, negatives win outright, and the event was
    silently dropped. Removing the prompt's own words first means the
    yes/no test only ever sees what the user contributed.
    """
    text = _CONFIRM_ECHO_RE.sub(" ", (heard or "").lower())
    # Whole words the prompt itself used are also echo, but only strip
    # them from the front — a user answering "no, not the 11 labs one"
    # must keep their words. Anything the user adds beyond the echo stops
    # the strip.
    prompt_words = set(re.findall(r"[a-z0-9']+", (spoken or "").lower()))
    prompt_words -= {"yes", "no"}
    words = text.split()
    while words and re.sub(r"[^a-z0-9']", "", words[0]) in prompt_words:
        words.pop(0)
    return " ".join(words).strip(" ,.?!-")



class _StreamingSpeaker:
    """Speaks the reply while the model is still writing it.

    The voice loop used to pass stream_callback=None, so the FULL answer
    was generated, then split, then synthesized — the user heard nothing
    until every one of those had finished. The machinery to do better
    already existed on both sides: llm.py streams Groq and Gemini over
    SSE (the web bridge has used it all along), and _speak_streaming
    already overlaps synthesis with playback. This joins the two.

    Three stages, each on its own thread so none blocks the next:

        model tokens -> feed() -> text_q -> synthesize -> wav_q -> play

    Only the plain-chat path streams; tool answers arrive complete and
    keep using _speak_streaming, exactly as the bridge distinguishes them.
    """

    # Flush the opening line early even mid-sentence. The first chunk is
    # the only thing between the user and the first word, and a model that
    # opens with a 200-character sentence would otherwise hold it all back.
    FIRST_FLUSH_CHARS = 70

    def __init__(self, loop, language_code, barge_event, playback_active):
        self.loop = loop
        self.language_code = language_code
        self.barge_event = barge_event
        self.playback_active = playback_active
        self.text_q: queue.Queue = queue.Queue()
        self.wav_q: queue.Queue = queue.Queue()
        self.buffer = ""
        self.lock = threading.Lock()
        self.spoke = False
        self.interrupted = False
        self.started_at = time.perf_counter()
        self._synth = threading.Thread(target=self._synthesize, daemon=True)
        self._play = threading.Thread(target=self._playback, daemon=True)
        self._synth.start()
        self._play.start()

    # --- producer ------------------------------------------------------

    def feed(self, piece: str) -> None:
        """Called by llm.chat for every token as it arrives."""
        if not piece or self.barge_event.is_set():
            return
        with self.lock:
            self.buffer += piece
            sentences, self.buffer = cli.tts.split_ready_sentences(self.buffer)
            if not sentences and not self.spoke and len(self.buffer) >= self.FIRST_FLUSH_CHARS:
                cut = self.buffer.rfind(" ", 0, self.FIRST_FLUSH_CHARS)
                if cut > 20:
                    sentences, self.buffer = [self.buffer[:cut]], self.buffer[cut:].lstrip()
        for sentence in sentences:
            self.spoke = True
            self.text_q.put(sentence)

    def finish(self) -> bool:
        """Flush the tail and wait for playback. False if barged in."""
        with self.lock:
            tail, self.buffer = self.buffer.strip(), ""
        if tail:
            self.spoke = True
            self.text_q.put(tail)
        self.text_q.put(None)
        self._synth.join(timeout=90)
        self._play.join(timeout=90)
        return not self.interrupted

    def abandon(self) -> None:
        """Shut the threads down without speaking anything more."""
        self.barge_event.set()
        self.text_q.put(None)

    # --- stages --------------------------------------------------------

    def _synthesize(self) -> None:
        # Same contract as _speak_streaming.produce: the sentinel goes out
        # on every exit path, or the playback thread waits forever.
        try:
            while True:
                text = self.text_q.get()
                if text is None or self.barge_event.is_set():
                    return
                clean = _clean_for_speech(text)
                if not clean:
                    continue
                try:
                    self.wav_q.put(
                        speech.synthesize(
                            clean, self.language_code, self.loop.cfg, source="voice"
                        )
                    )
                except sarvam_client.SarvamError as e:
                    print(f"(Sarvam TTS unavailable: {e})")
                    return
                except Exception as e:
                    print(f"(TTS synthesis failed: {type(e).__name__}: {e})")
                    traceback.print_exc()
                    return
        finally:
            self.wav_q.put(None)

    def _playback(self) -> None:
        first = True
        try:
            while True:
                wav_bytes = self.wav_q.get()
                if wav_bytes is None:
                    return
                if self.barge_event.is_set():
                    if not first:
                        voice_events.speaking_stopped()
                    self.interrupted = True
                    return
                if first:
                    self.playback_active.set()
                    first = False
                    self.loop._last_ttfa = time.perf_counter() - self.started_at
                voice_events.speaking(wav_bytes)
                if not cli.tts.play(wav_bytes, stop_event=self.barge_event):
                    voice_events.speaking_stopped()
                    self.interrupted = True
                    return
        finally:
            # Same reason as _speak_streaming: these frames are Bella's own
            # voice coming back through the mic and must not become the
            # opening of the next recording.
            with self.loop._barge_frames_lock:
                self.loop._barge_frames.clear()


class VoiceLoop:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.wake_word = cfg["voice_wake_word"]
        self.silence_rms_threshold = cfg["voice_silence_rms_threshold"]
        # Kept so calibrate() always re-derives from the configured floor
        # rather than from its own previous (possibly inflated) result.
        self._base_silence_rms_threshold = cfg["voice_silence_rms_threshold"]
        self.push_to_talk = None
        if cfg.get("push_to_talk_enabled", True):
            combo = tuple(cfg.get("push_to_talk_combo", hotkey.DEFAULT_COMBO))
            self.push_to_talk = hotkey.PushToTalk(combo, on_press=self._on_push_to_talk)
            if not self.push_to_talk.start():
                self.push_to_talk = None

        # Tap-to-wake, as opposed to hold-to-talk above. Same permission,
        # separate listener: if this one can't start, push-to-talk is
        # unaffected.
        self.wake_hotkey = None
        if cfg.get("wake_hotkey_mode", "double_tap") != "off":
            self.wake_hotkey = hotkey.WakeHotkey(
                mode=cfg.get("wake_hotkey_mode", "double_tap"),
                key=cfg.get("wake_hotkey_key", hotkey.DEFAULT_WAKE_KEY),
                combo=tuple(cfg.get("wake_hotkey_combo", hotkey.DEFAULT_WAKE_COMBO)),
                on_wake=self._on_wake_hotkey,
            )
            if not self.wake_hotkey.start():
                self.wake_hotkey = None
        self.volume_keys = hotkey.VolumeKeys(on_change=self._on_volume_key)
        if not self.volume_keys.start():
            self.volume_keys = None
        self.silence_seconds = cfg.get("voice_silence_seconds", 0.7)
        self.no_speech_bail_seconds = cfg.get("voice_no_speech_bail_seconds", NO_SPEECH_BAIL_SECONDS)
        self.wake_listen_seconds = cfg.get("voice_wake_listen_seconds", WAKE_LISTEN_SECONDS)
        self.audio_q: queue.Queue = queue.Queue()
        self._last_frame_at = time.monotonic()
        self.wake_event = threading.Event()
        # Unused for multi-turn follow-ups. Every command needs "Hey Bella".
        # Kept so older call sites that still write it do not crash.
        # Monotonic deadline for staying in "listening for follow-up" mode
        # without requiring the wake word again. None means "need the wake
        # word." Deliberately a timestamp, not the single-shot boolean this
        # used to be — the old boolean got consumed at the top of each loop
        # iteration and only re-armed at the end of a *completed* turn, so
        # any single "didn't catch anything" (a pause, speaking too quietly)
        # silently dropped back to requiring the wake phrase again. That was
        # the actual mechanical cause of the conversation feeling like it
        # kept turning on and off. A deadline that persists across empty
        # attempts and only expires after real inactivity fixes that.
        self.conversation_active_until: float | None = None
        self._awaiting_command_after_name = False
        self._woke_by_hotkey = False
        # When Bella last stopped talking — the echo filter's time window.
        self._last_speech_ended_at = 0.0
        # Set by the barge-in watcher when the user said "stop <name>".
        # Distinct from a wake-word barge-in, which means "listen to me
        # now"; a stop means "be quiet", so the loop must not immediately
        # open a listening window and wait for a command that isn't coming.
        self._stop_requested = False
        # Live switches shared with the web UI (bella/control.py).
        self._control = control.Watcher()
        self._barge_frames: deque[np.ndarray] = deque(maxlen=12)
        self._barge_frames_lock = threading.Lock()
        # Two ways to be woken, same interface (reset/predict) so nothing
        # downstream cares which is in use:
        #   "phrase" — VAD segments speech, a small Whisper transcribes it,
        #              and the configured phrase is matched in the text.
        #              Any phrase at all, no model to train.
        #   "model"  — an openWakeWord detector. Lower CPU and it fires
        #              mid-phrase, but the pretrained set is hey_jarvis /
        #              alexa / hey_mycroft / hey_rhasspy, all of them
        #              someone else's product name.
        self.wake_mode = cfg.get("wake_mode", "phrase")
        # Key the detector's score dict by. In model mode it has to be the
        # openWakeWord model's own name; in phrase mode nothing external
        # dictates it, and "hey_jarvis" appearing as a live dict key in a
        # product named Bella is exactly the sort of leftover that confuses
        # someone reading a log six months from now.
        self.wake_key = "wake" if self.wake_mode == "phrase" else self.wake_word
        configure_wake_phrases(cfg.get("wake_phrases") or ["hey bella", "bella"])
        print(f'Loading local speech-to-text model (faster-whisper {cfg["voice_stt_model"]})...')
        # Two threads, not "all cores". Idle wake listening was measured at
        # 424% CPU (ctranslate2 GEMM) after an Assamese turn, which made
        # the loop look gone: HUD still said running, but nothing could
        # get a word in.
        self.whisper = WhisperModel(
            cfg["voice_stt_model"], device="cpu", compute_type="int8", cpu_threads=2
        )
        if self.wake_mode == "phrase":
            phrases = cfg.get("wake_phrases") or ["hey bella", "bella"]
            print(
                'Loading wake phrase detector ("'
                + " or ".join(p.title() for p in phrases)
                + '")...'
            )
            # An empty wake_stt_model means "share the command model above"
            # — no second download, no extra memory, and better recall than
            # a smaller one (see bella/wake_phrase.py).
            wake_size = cfg.get("wake_stt_model", "")
            self.wake_model = wake_phrase.PhraseWake(
                phrases,
                stt_model_size=wake_size,
                key="wake",
                model=None if wake_size else self.whisper,
            )
        else:
            print(f'Loading wake word model ("{self.wake_word}")...')
            self.wake_model = WakeWordModel(
                wakeword_models=[self.wake_word], inference_framework="onnx"
            )
        print("Loading voice activity detector (Silero VAD, onnx)...")
        self.vad = SileroVAD()
        self._vad_buffer = np.zeros(0, dtype=np.float32)

    def trigger_wake(self) -> None:
        self.wake_event.set()

    def _on_wake_hotkey(self) -> None:
        """Shortcut tapped: start a turn now, from wherever they are.

        Deliberately not the push-to-talk path. Push-to-talk records for
        exactly as long as the key is held; this is the same thing as
        saying the wake phrase — Bella chimes and listens for one utterance
        — so it goes through trigger_wake() and nothing else.
        """
        if self._control.mic_muted:
            # Pressing a key is as deliberate as clicking the button, and
            # it means the user is at the machine. Without this, a muted
            # Bella can only be revived from the web UI, which is exactly
            # where you are not when you want to talk to it.
            print("(Wake hotkey — turning the microphone back on.)", flush=True)
            control.set_mic_muted(False)
        print("(Wake hotkey — listening.)", flush=True)
        self._play_ack_tone(clear_preroll=False, block=False)
        self.trigger_wake()

    def _dismiss_screen_help(self) -> None:
        """Take the instruction card down. Stop means the lesson is over."""
        try:
            from bella import instructions
            from bella.tools import vision
            instructions.hide()
            vision.clear_pending_show()
        except Exception as e:
            print(f"[panel] could not dismiss: {e}", flush=True)

    def trigger_interrupt(self) -> None:
        """Stop talking / stop working, from outside this process.

        Exactly what the push-to-talk key does (see _on_push_to_talk),
        reached instead by SIGUSR2 from the bridge so the web UI's Stop
        button is a real stop. It used to only silence audio inside the
        browser tab, which does nothing at all for a spoken turn — that
        audio is playing out of the Mac's speakers via afplay, and no
        amount of stopping a WebAudio node touches it.
        """
        self._dismiss_screen_help()
        self._interrupt_source = "the web UI's Stop button"
        self._ptt_interrupt = True
        self.trigger_wake()

    def _on_volume_key(self, direction: str) -> None:
        from bella import tts
        delta = tts.VOLUME_STEP if direction == "up" else -tts.VOLUME_STEP
        level = tts.adjust_volume(delta)
        print(f"  voice volume: {level}%", flush=True)

    def _on_push_to_talk(self) -> None:
        """Hotkey pressed: start a turn immediately, wherever we were.

        trigger_wake() releases listen_for_wake_word the same way the
        bridge's SIGUSR1 does. _ptt_interrupt lets a press double as
        barge-in, so pressing the key while Bella is talking stops it and
        starts listening — the same gesture for "go" and "stop".
        """
        self._interrupt_source = "Push-to-talk"
        self._ptt_interrupt = True
        print("(Push-to-talk — listening.)", flush=True)
        self._play_ack_tone(clear_preroll=False, block=False)
        self.trigger_wake()

    def _audio_callback(self, indata, frames, time_info, status):
        self._last_frame_at = time.monotonic()
        self.audio_q.put(indata[:, 0].copy())

    def _next_frame(self) -> np.ndarray:
        while True:
            # Checked on the one path every listening state goes through,
            # so mute takes effect mid-wake-word, mid-recording, anywhere.
            if self._control.mic_muted:
                raise _MicMuted()
            try:
                return self.audio_q.get(timeout=1.0)
            except queue.Empty:
                if time.monotonic() - self._last_frame_at > _STREAM_DEAD_TIMEOUT:
                    raise _StreamDead()

    def _drain_queue(self) -> None:
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    def _rms(self, frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

    def _frame_speech_prob(self, frame: np.ndarray) -> float:
        """Run frame (FRAME_SAMPLES int16 samples) through Silero VAD.

        The model requires exactly VAD_CHUNK_SAMPLES (512) per call, which
        FRAME_SAMPLES (1280, openWakeWord's native size) isn't a multiple
        of — leftover samples carry over in self._vad_buffer to the next
        call rather than being dropped or requiring two incompatible frame
        sizes. Returns the bella speech probability across whatever whole
        512-sample chunks this frame completed (0.0 if none did yet).
        """
        audio_float = frame.astype(np.float32) / 32768.0
        self._vad_buffer = np.concatenate([self._vad_buffer, audio_float])
        probs = []
        while len(self._vad_buffer) >= VAD_CHUNK_SAMPLES:
            chunk = self._vad_buffer[:VAD_CHUNK_SAMPLES]
            self._vad_buffer = self._vad_buffer[VAD_CHUNK_SAMPLES:]
            probs.append(self.vad.speech_probability(chunk))
        return max(probs) if probs else 0.0

    def calibrate(self, seconds: float = 2.5) -> None:
        """Set the silence threshold from real ambient noise instead of a
        guessed constant — a fixed default can't know your room/mic gain in
        advance. Uses the median (not bella) of the sampled window: bella is
        fragile against a single transient spike (a click, a cough) that
        isn't representative of the true steady-state noise floor —
        observed this directly, a one-off spike inflated the threshold
        above real speech volume on the previous attempt.
        """
        print(f"Calibrating mic to ambient noise ({seconds:.1f}s, please stay quiet)...")
        self._drain_queue()
        chunks_needed = max(1, int(seconds * SAMPLE_RATE / FRAME_SAMPLES))
        levels = [float(np.sqrt(np.mean(self._next_frame().astype(np.float32) ** 2))) for _ in range(chunks_needed)]
        levels.sort()
        median = levels[len(levels) // 2] if levels else 0.0
        p90 = levels[int(len(levels) * 0.9)] if levels else 0.0
        # A real room is never digitally silent. All-zero frames mean the
        # device is delivering nothing — seen live after EarPods were
        # plugged in and removed, where the process kept a stale PortAudio
        # device and read zeros forever. Treat it as a dead stream so the
        # caller re-initializes the audio layer instead of calibrating
        # against nothing and then sitting there deaf.
        if p90 <= 0.0:
            print("  mic is delivering pure silence — treating the audio device as dead")
            raise _StreamDead()
        # Baseline from the CONFIGURED value, not the last calibrated one.
        # This used to be max(self.silence_rms_threshold, ...) where the
        # left side was the previous calibration, so the threshold could
        # only ever ratchet upward across stream reopens and never recover
        # — observed stuck at 5673 (config baseline is 300) after a run of
        # bad calibrations, high enough to reject ordinary speech long
        # after the microphone itself was fine again.
        base = self._base_silence_rms_threshold
        # Calibrating while music or speech is already in the room used
        # to push the floor to 5000+ (seen live: 2389 median -> 5255).
        # After that, follow-up listens treated the song as speech and
        # Bella answered lyrics instead of the user. If the "ambient"
        # window is already loud, keep the configured floor.
        if median > base * 3 or p90 > base * 5:
            print(
                f"  ambient RMS median {median:.0f} / p90 {p90:.0f} looks like "
                f"speech/music — keeping floor {base:.0f}"
            )
            self.silence_rms_threshold = base
        else:
            cap = base * 2.5
            calibrated = min(max(base, median * 2.2), cap)
            print(
                f"  ambient RMS median {median:.0f} / p90 {p90:.0f} "
                f"-> silence threshold set to {calibrated:.0f}"
            )
            self.silence_rms_threshold = calibrated
        print(f'  barge-in: say "{self.wake_label()}" to interrupt '
              '(phrase-gated, not volume)')
        if self.wake_hotkey is not None:
            print(f"  wake hotkey: {self.wake_hotkey.describe()} to start a turn "
                  "from any app")
        if self.push_to_talk is not None:
            print(f"  push-to-talk: hold {self.push_to_talk.describe()} and say "
                  f'"{self.wake_label()}" then the command')
        if self.volume_keys is not None:
            from bella import tts
            print(f"  voice volume: {tts.get_volume()}% — keyboard volume keys "
                  "or the HUD slider, say 'quieter' / 'louder'")
            # Both listeners are optional — push_to_talk is None whenever
            # push_to_talk_enabled is false or the tap failed to start, and
            # dereferencing it here took the whole startup down with an
            # AttributeError inside calibrate().
            ptt_silent = self.push_to_talk is None or not self.push_to_talk.saw_any_event
            wake_silent = self.wake_hotkey is None or not self.wake_hotkey.saw_any_event
            if ptt_silent and wake_silent:
                # pynput's listener starts fine without permission and then
                # never receives anything, so say it out loud rather than
                # letting the key appear to do nothing.
                print("     if nothing happens, grant Input Monitoring: System Settings >")
                print("     Privacy & Security > Input Monitoring > enable your terminal")

    def _voice_model_override(self) -> str | None:
        pinned = self.cfg.get("voice_ollama_model")
        if not pinned:
            return None
        return pinned if llm.resolve(self.cfg, "voice")["local"] else None

    def wake_label(self) -> str:
        """What to tell the user to say. Comes from whichever detector is
        actually loaded, so the banner can never advertise a phrase that
        isn't the one being listened for."""
        if self.wake_mode == "phrase":
            return self.wake_model.describe()
        return self.wake_word.replace("_", " ").title()

    def listen_for_wake_word(self) -> None:
        self.wake_model.reset()
        self._woke_by_hotkey = False
        if self.wake_event.is_set():
            self.wake_event.clear()
            self._woke_by_hotkey = True
            return
        while True:
            if self.wake_event.is_set():
                self.wake_event.clear()
                self._woke_by_hotkey = True
                return
            frame = self._next_frame()
            prediction = self.wake_model.predict(frame)
            if prediction.get(self.wake_key, 0.0) > WAKE_THRESHOLD:
                self._woke_by_hotkey = False
                return

    def _watch_for_barge_in(
        self,
        stop_event: threading.Event,
        playback_active: threading.Event,
        processing_active: threading.Event | None = None,
    ) -> None:
        """Interrupt on the WAKE WORD, not on loudness.

        This machine has no acoustic echo cancellation, so the mic hears
        Bella's own speakers. An energy threshold cannot tell that apart
        from the user interrupting, and measured over 416 real turns it
        fired 179 times (43%) — Bella cutting its own replies into
        fragments, which is what "it says three or four lines and then
        drops" actually was. Chasing it with a higher threshold and a
        fuzzy transcript filter also ate 36 genuine user commands.

        Gating on the wake phrase makes the problem structurally
        impossible instead: Bella does not say its own wake phrase, so its
        voice cannot trigger this at any volume. Same contract Alexa uses
        — say the phrase to interrupt. (In phrase mode the check is a
        transcript match rather than a wake-word model, so the guarantee
        is "Bella never says it" rather than "no model can fire on it" —
        which is why the persona is told not to greet itself.)
        """
        # Fed every frame, including while disarmed, so the model always
        # has continuous context and a wake word spanning the moment
        # playback starts is still caught.
        self.wake_model.reset()
        while not stop_event.is_set():
            try:
                frame = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._barge_frames_lock:
                self._barge_frames.append(frame)
            if getattr(self, "_ptt_interrupt", False):
                self._ptt_interrupt = False
                source = getattr(self, "_interrupt_source", "Push-to-talk")
                print(f"({source} — stopping and listening.)")
                if processing_active is not None and processing_active.is_set() and not playback_active.is_set():
                    cli.claude_handoff.cancel_current()
                    cli.shell.cancel_current()
                stop_event.set()
                return
            if self.wake_mode == "phrase":
                # Only here: "stop Bella" while it is talking or working has
                # to land immediately, without waiting for a turn to end.
                score = self.wake_model.predict(frame, include_stop=True).get(self.wake_key, 0.0)
                if score > WAKE_THRESHOLD and self.wake_model.last_match_kind == "stop":
                    print("(Stop heard — stopping and standing by.)")
                    self._dismiss_screen_help()
                    if processing_active is not None and processing_active.is_set():
                        cli.claude_handoff.cancel_current()
                        cli.shell.cancel_current()
                    self._stop_requested = True
                    stop_event.set()
                    return
            else:
                score = self.wake_model.predict(frame).get(self.wake_key, 0.0)
            processing = processing_active is not None and processing_active.is_set()
            if not playback_active.is_set() and not processing:
                # This thread starts before the response is ready, so that
                # playback can be interrupted the instant it begins.
                # processing_active (set only around cli.process_turn) is
                # the deliberate exception: it enables real "stop while
                # it's still working" cancellation of an in-flight task.
                continue
            if score > WAKE_THRESHOLD:
                if processing and not playback_active.is_set():
                    print(f"(Wake word heard while working — cancelling that task. score {score:.2f})")
                    cli.claude_handoff.cancel_current()
                    cli.shell.cancel_current()
                else:
                    print(f"(Wake word heard — stopping speech. score {score:.2f})")
                stop_event.set()
                return

    def _play_ack_tone(self, *, clear_preroll: bool = True, block: bool = True) -> None:
        def _run() -> None:
            try:
                subprocess.run(["afplay", _ack_tone_path()], capture_output=True, timeout=5)
            except Exception:
                return
            # Hotkey chime must not wipe speech that started on the press.
            if clear_preroll:
                with self._barge_frames_lock:
                    self._barge_frames.clear()

        if block:
            _run()
        else:
            threading.Thread(target=_run, daemon=True).start()

    def _take_queued_frames(self) -> np.ndarray:
        frames: list[np.ndarray] = []
        while not self.audio_q.empty():
            try:
                frames.append(self.audio_q.get_nowait())
            except queue.Empty:
                break
        if not frames:
            return np.array([], dtype=np.int16)
        return np.concatenate(frames)

    def record_while_held(self, max_seconds: float = MAX_UTTERANCE_SECONDS) -> tuple[np.ndarray, bool]:
        """Record for exactly as long as the push-to-talk key is held.

        No wake word, no VAD gate, no silence timeout, no bail — the key
        being down IS the signal, so none of the probabilistic machinery
        that has been dropping utterances can drop one here. Whisper still
        decides whether there were words in it.
        """
        # Do not drain: the command is already in the queue from the
        # moment they pressed the key. Drain was why "good morning" had
        # to be said five times.
        frames: list[np.ndarray] = []
        queued = self._take_queued_frames()
        if queued.size:
            frames.append(queued)
        max_chunks = int(max_seconds * SAMPLE_RATE / FRAME_SAMPLES)
        # A short floor so an accidental quick tap still yields something
        # transcribable rather than a zero-length buffer.
        min_chunks = max(1, int(0.5 * SAMPLE_RATE / FRAME_SAMPLES))
        while len(frames) < max_chunks:
            if not self.push_to_talk.held.is_set() and len(frames) >= min_chunks:
                break
            frames.append(self._next_frame())
        # Tail past the release: people finish the last syllable as they
        # let go, and clipping it turns "play Metallica" into "play Metall".
        for _ in range(int(0.3 * SAMPLE_RATE / FRAME_SAMPLES)):
            frames.append(self._next_frame())

        recorded = np.concatenate(frames) if frames else np.array([], dtype=np.int16)
        peak = self._rms(recorded) if recorded.size else 0.0
        self._last_peak_rms = peak
        print(f"  (debug: push-to-talk, {recorded.size / SAMPLE_RATE:.1f}s recorded, RMS {peak:.0f})")
        return recorded, recorded.size > 0

    def record_utterance(
        self, max_seconds: float = MAX_UTTERANCE_SECONDS, bail_seconds: float | None = None
    ) -> tuple[np.ndarray, bool]:
        # Drain first: the InputStream callback queues frames continuously,
        # including the entire time Bella was mid-reply (LLM latency +
        # TTS playback, seen live running 20-40+ seconds). Nothing here
        # used to discard that backlog before listening again, so a fresh
        # "recording" could start by consuming hundreds of queued frames
        # of Bella's OWN just-finished speech before ever reaching
        # genuinely current audio — the actual mechanism behind most of
        # what looked like random mis-hearing, confirm prompts hearing an
        # unrelated older reply instead of yes/no, and the self-echo text
        # filter (which only runs after this) having stale content to
        # filter in the first place. This is a bug fix, not a tuning
        # knob — a listening phase should always start from silence.
        self._drain_queue()
        # heard_speech is now decided by Silero VAD, not the RMS threshold —
        # a real speech classifier doesn't need per-room amplitude
        # calibration the way raw RMS does, which is exactly what kept
        # drifting out of tune tonight (auto-relaxing the RMS threshold to
        # fix missed speech had the side effect of also making barge-in
        # oversensitive, since both derived from the same base value). RMS
        # is still computed for the debug line and is still what barge-in
        # and voice_confirm's silence gate use — this only replaces the
        # utterance-recording gate specifically.
        self.vad.reset()
        self._vad_buffer = np.zeros(0, dtype=np.float32)
        with self._barge_frames_lock:
            frames = list(self._barge_frames)
            self._barge_frames.clear()
        silence_chunks = 0
        needed_silence_chunks = max(1, int(self.silence_seconds * SAMPLE_RATE / FRAME_SAMPLES))
        max_chunks = int(max_seconds * SAMPLE_RATE / FRAME_SAMPLES)
        if bail_seconds is None:
            bail_seconds = self.no_speech_bail_seconds
        bail_chunks = max(1, int(bail_seconds * SAMPLE_RATE / FRAME_SAMPLES))
        heard_speech = False
        max_rms = 0.0
        max_vad_prob = 0.0
        chunks_used = 0

        for frame in frames:
            chunks_used += 1
            max_rms = max(max_rms, self._rms(frame))
            prob = self._frame_speech_prob(frame)
            max_vad_prob = max(max_vad_prob, prob)
            if prob > VAD_SPEECH_THRESHOLD:
                heard_speech = True
                silence_chunks = 0
            else:
                silence_chunks += 1

        for _ in range(max(0, max_chunks - len(frames))):
            frame = self._next_frame()
            frames.append(frame)
            chunks_used += 1
            max_rms = max(max_rms, self._rms(frame))
            prob = self._frame_speech_prob(frame)
            max_vad_prob = max(max_vad_prob, prob)
            if prob > VAD_SPEECH_THRESHOLD:
                heard_speech = True
                silence_chunks = 0
            else:
                silence_chunks += 1
            if heard_speech and silence_chunks >= needed_silence_chunks:
                break
            # Speech never started — stand down instead of recording (and
            # then transcribing) the full max_seconds of silence. See
            # NO_SPEECH_BAIL_SECONDS.
            if not heard_speech and chunks_used >= bail_chunks:
                break

        elapsed = chunks_used * FRAME_SAMPLES / SAMPLE_RATE
        print(
            f"  (debug: {elapsed:.1f}s recorded, peak RMS {max_rms:.0f}, "
            f"peak VAD prob {max_vad_prob:.2f}, heard_speech={heard_speech})"
        )
        self._last_peak_rms = max_rms
        self._last_max_vad = max_vad_prob
        recorded = np.concatenate(frames) if frames else np.array([], dtype=np.int16)
        return recorded, heard_speech

    def _speak_streaming(
        self,
        text: str,
        language_code: str,
        barge_event: threading.Event,
        playback_active: threading.Event,
    ) -> bool:
        """Speak `text` sentence-by-sentence. Returns False if interrupted.

        The whole reply used to go to Sarvam in one call, so nothing was
        audible until the entire thing had been synthesized — measured live
        across 40 turns at a ~7s median and a 21.7s worst case of dead
        silence before Bella said a single word. Synthesis of chunk N+1
        overlaps playback of chunk N, so the only thing standing between
        the user and the first word is one short sentence.
        """
        chunks = _split_for_speech(_clean_for_speech(text))
        self._last_ttfa = None
        if not chunks:
            return True

        started_at = time.perf_counter()
        wav_q: queue.Queue = queue.Queue()

        def produce() -> None:
            # The sentinel MUST be posted on every exit path. The consumer
            # below blocks on wav_q.get() with no timeout, so a producer that
            # dies without posting it hangs the turn — and the whole voice
            # loop — forever, with no traceback anywhere the user can see and
            # no way for barge-in to break it (the consumer is blocked before
            # it ever checks barge_event). That is exactly what happened: a
            # urllib *read* timeout raises a bare TimeoutError, which is not a
            # URLError and so was never wrapped as SarvamError; it escaped
            # this thread and Bella went permanently silent. The bare `except
            # Exception` is deliberate — anything at all coming out of
            # synthesis costs one spoken reply, never the process.
            try:
                for chunk in chunks:
                    if barge_event.is_set():
                        break
                    try:
                        wav_q.put(
                            speech.synthesize(
                                chunk, language_code, self.cfg, source="voice"
                            )
                        )
                    except sarvam_client.SarvamError as e:
                        print(f"(Sarvam TTS unavailable: {e})")
                        break
                    except Exception as e:
                        print(f"(TTS synthesis failed: {type(e).__name__}: {e})")
                        traceback.print_exc()
                        break
            finally:
                wav_q.put(None)

        threading.Thread(target=produce, daemon=True).start()

        first = True
        try:
            while True:
                wav_bytes = wav_q.get()
                if wav_bytes is None:
                    return True
                if barge_event.is_set():
                    # Only if an envelope already went out: the UI holds the
                    # last one until it runs out, so a barge-in between
                    # chunks leaves the orb mouthing a sentence that stopped.
                    if not first:
                        voice_events.speaking_stopped()
                    return False
                if first:
                    # Arm barge-in only once there's real audio, not during
                    # the network call that produces it.
                    playback_active.set()
                    first = False
                    self._last_ttfa = time.perf_counter() - started_at
                # Published for the web UI's orb, which otherwise has no
                # signal at all for a spoken turn — this audio goes to the
                # Mac's speakers, never to the browser. See voice_events.
                voice_events.speaking(wav_bytes)
                if not cli.tts.play(wav_bytes, stop_event=barge_event):
                    voice_events.speaking_stopped()
                    return False
        finally:
            # THE source of self-echo, and it was never a text problem.
            # _watch_for_barge_in appends every frame it sees into
            # _barge_frames, including all the frames that are Bella's own
            # voice coming back through the mic during playback. That deque
            # is then used verbatim as the opening ~1s of the NEXT
            # recording (see record_utterance) — _drain_queue() empties
            # audio_q but has never touched this. So the tail of Bella's
            # own sentence was being prepended to whatever the user said
            # next, which is what the transcript-similarity heuristics have
            # been trying and failing to clean up after ever since.
            # Dropping it here means the pre-roll only ever holds audio
            # captured after Bella stopped talking, while still doing its
            # real job of catching a user who starts speaking early.
            with self._barge_frames_lock:
                self._barge_frames.clear()

    def transcribe(self, audio: np.ndarray, language_code: str | None = None) -> str:
        if audio.size == 0:
            return ""
        # Local English whisper first on short clips. "Good morning" was
        # waiting 1.4s on Saaras every time even though the wake model
        # already heard it in English.
        audio_float = audio.astype(np.float32) / 32768.0
        if language_code in ("as-IN", "hi-IN") and audio.size < 2.5 * SAMPLE_RATE:
            try:
                segs, _ = self.whisper.transcribe(audio_float, language="en", beam_size=1)
                local = " ".join(s.text.strip() for s in segs).strip()
            except Exception:
                local = ""
            if local and cli._assamese_is_greeting(local):
                print(f"  (local stt, skip Saaras: {local!r})")
                return local
        if language_code in ("as-IN", "hi-IN"):
            wav_bytes = sarvam_client.pcm16_to_wav(
                np.ascontiguousarray(audio, dtype=np.int16).tobytes(),
                SAMPLE_RATE,
            )
            if language_code == "as-IN" and speech.assamese_speech_backend(self.cfg) == "bhashini":
                from bella import bhashini_client, bhashini_feedback

                try:
                    sid = bhashini_client.service_id("asr")
                    text = bhashini_client.speech_to_text(
                        wav_bytes, sample_rate=SAMPLE_RATE
                    )
                    text = sarvam_client.odia_to_assamese(text)
                    bhashini_feedback.record_call(
                        "asr",
                        input_text="[audio]",
                        output_text=text,
                        service_id=sid,
                        source="voice",
                        cfg=self.cfg,
                    )
                    return text
                except bhashini_client.BhashiniError as e:
                    print(f"(Bhashini STT unavailable: {e} — Sarvam)", flush=True)
                    try:
                        bhashini_feedback.record_fallback(
                            "asr",
                            str(e),
                            service_id=bhashini_client.service_id("asr"),
                            source="voice",
                            cfg=self.cfg,
                        )
                    except bhashini_client.BhashiniError:
                        pass
            try:
                text, detected = sarvam_client.speech_to_text(
                    wav_bytes, language_code=language_code, mode="transcribe"
                )
            except sarvam_client.SarvamError as e:
                print(f"(Sarvam STT unavailable: {e})")
                return ""
            if detected and detected != language_code:
                print(f"  (Saaras heard {detected})")
            return sarvam_client.odia_to_assamese(text)
        segments, _ = self.whisper.transcribe(audio_float, language="en", beam_size=1)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def voice_confirm(self, prompt: str) -> bool:
        # Nothing is actually running yet anywhere in this function — the
        # task this prompt is asking about only starts AFTER it returns
        # True. Seen live: processing_active (armed for cancel-while-
        # executing, see _handle_turn_audio) being armed for any part of
        # this — the prompt's own TTS, or just answering "yes" out loud —
        # let ordinary interaction with the confirmation itself satisfy the
        # barge-in sustain threshold and get misread as "stop, cancel the
        # task" that hadn't started yet. Paused for the function's entire
        # duration, not just around record_utterance, closes the gap
        # completely without touching the real cancel-while-executing path.
        processing_active = getattr(self, "_processing_active", None)
        was_armed = processing_active is not None and processing_active.is_set()
        if was_armed:
            processing_active.clear()
        try:
            return self._voice_confirm_inner(prompt)
        finally:
            if was_armed:
                processing_active.set()

    def _voice_confirm_inner(self, prompt: str) -> bool:
        for attempt in range(CONFIRM_ATTEMPTS):
            answer = self._ask_once(prompt, reask=attempt > 0)
            if answer is not None:
                return answer
        print("  (no clear answer after asking twice — treating as no.)")
        return False

    def _ask_once(self, prompt: str, reask: bool) -> bool | None:
        """True/False for a clear answer, None if nothing usable was heard.

        None is NOT the same as no: a confirmation that only picked up
        Bella's own prompt coming back through the mic has not been answered
        at all, and cancelling on it threw away work the user had asked
        for. The caller asks again instead.
        """
        spoken = ("Sorry — yes or no?" if reask else prompt + " Say yes or no.")
        print(f"[Router] {spoken if reask else prompt}")
        # Was self.speak() (cli.speak_response -> Piper), left over from
        # before the Bulbul-everywhere switch — every other reply plays
        # through Sarvam now, so the confirm prompt suddenly speaking in a
        # different voice (Piper's British accent) was jarring and
        # confusing on its own, on top of everything else. No Piper
        # fallback on failure, same as every other Sarvam call site since
        # that switch — just report and move on.
        if self.cfg.get("tts_enabled"):
            try:
                wav_bytes = speech.synthesize(spoken, "en-IN", self.cfg)
                voice_events.speaking(wav_bytes)
                cli.tts.play(wav_bytes)
                # cli.tts.play returns when the player exits, but the last
                # fraction of a second is still leaving the speakers. This
                # Mac has no acoustic echo cancellation, so without a
                # settle the recording below opens on Bella's own trailing
                # words — measured live, the mic caught "yes or no" from
                # this very prompt and CONFIRM_NO_RE cancelled the turn.
                time.sleep(CONFIRM_SETTLE_SECONDS)
            except sarvam_client.SarvamError as e:
                print(f"(Sarvam TTS unavailable: {e})")
        print("(listening for yes/no...)")
        audio, heard_speech = self.record_utterance(max_seconds=CONFIRM_MAX_SECONDS)
        if not heard_speech:
            print("  (heard: nothing above the noise floor — treating as no.)")
            return False

        raw = self.transcribe(audio).lower().strip()
        text = _strip_confirm_echo(raw, spoken)
        if text != raw:
            print(f"  (heard: {raw!r} -> {text!r} after removing Bella's own prompt)")
        else:
            print(f"  (heard: {raw!r})")
        if not text:
            # Everything heard was the prompt echoing back. Nothing was
            # answered, so ask again rather than cancel.
            return None
        if CONFIRM_NO_RE.search(text):
            return False
        if len(text.split()) > CONFIRM_MAX_WORDS:
            print("  (heard something too long/ambiguous to count as a clear yes — treating as no.)")
            return False
        if CONFIRM_YES_RE.search(text):
            return True
        return None

    def run(self) -> None:
        mem = MemoryStore()
        # Same reason as the bridge: the Composio session costs ~2.9s to
        # build, and it should not be built while the user is waiting for
        # their first answer.
        threading.Thread(target=composio_client.warm, daemon=True).start()
        wake_label = self.wake_label()
        first_start = True
        # Chosen explicitly rather than left to sd.default.device, which
        # goes stale the moment a microphone is plugged in or removed.
        self._input_device = self._select_input_device()
        while True:
            self._last_frame_at = time.monotonic()
            try:
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                    blocksize=FRAME_SAMPLES,
                    callback=self._audio_callback,
                    device=self._input_device,
                ):
                    self._drain_queue()
                    self.calibrate()
                    if first_start:
                        first_start = False
                        from bella import screen_capture
                        if screen_capture.ensure_permission():
                            print("  screen: Screen Recording is on")
                        else:
                            print("  screen: " + screen_capture.PERMISSION_HELP)
                        print(f'Bella voice loop online, {self.cfg["user_name"]}. Say "{wake_label}" to begin.\n')
                    else:
                        print(f'Bella back online, {self.cfg["user_name"]}. Say "{wake_label}" to begin.\n')
                    self._conversation_loop(mem)
            except _MicMuted:
                # The `with` above has now exited, so the InputStream is
                # closed and the microphone is genuinely released.
                print("\n🔇 Microphone OFF — Bella is not listening. "
                      "Nothing is recorded or transcribed until you turn it back on.\n",
                      flush=True)
                self._drain_queue()
                voice_events.mic_state(True)
                while self._control.mic_muted:
                    time.sleep(0.3)
                print("🎙  Microphone ON — listening again.\n", flush=True)
                voice_events.mic_state(False)
                self._last_frame_at = time.monotonic()
                continue
            except _StreamDead:
                print(
                    "\n(Audio stream stopped responding — likely a microphone/audio "
                    "device change. Reopening against the current default input...)"
                )
                self._reinit_audio()
                continue
            except _VoiceLoopShutdown:
                # Never leave the user's music paused because we exited
                # mid-conversation; pause/resume is per conversation now,
                # so shutdown is a conversation ending like any other.
                spotify.resume_after_conversation()
                print("Stopping voice loop on request.")
                return

    def _select_input_device(self) -> int | None:
        """Pick an input device that actually delivers audio.

        sd.default.device materializes an integer index the first time it
        is read and then never tracks devices appearing or disappearing.
        The loop started with no EarPods connected, cached that index, and
        after they were plugged in and removed the same index pointed at
        something that returned pure silence — it then reported itself
        "listening on MacBook Pro Microphone" 714 times while being deaf.
        So: ask the host API for the CURRENT default, and verify the device
        really produces samples before committing to it, falling back to
        any other input that does.
        """
        candidates: list[int] = []
        try:
            hostapi = sd.query_hostapis(sd.default.hostapi)
            idx = hostapi.get("default_input_device", -1)
            if idx is not None and idx >= 0:
                candidates.append(idx)
        except Exception:
            pass
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0 and i not in candidates:
                    candidates.append(i)
        except Exception:
            pass

        for idx in candidates:
            try:
                name = sd.query_devices(idx)["name"]
                probe = sd.rec(
                    int(0.3 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                    channels=1, dtype="int16", device=idx,
                )
                sd.wait()
                if int(np.abs(probe).max()) > 0:
                    print(f"  input device: {name}")
                    return idx
                print(f"  skipping {name} — delivering pure silence")
            except Exception as e:
                print(f"  skipping device {idx} — {type(e).__name__}")
        print("  (no input device produced audio; falling back to the system default)")
        return None

    def _reinit_audio(self) -> None:
        """Re-enumerate audio devices after a stream failure.

        PortAudio caches its device list at initialization, so simply
        reopening an InputStream after the user plugs in or unplugs
        headphones hands back a device index that no longer refers to what
        it did — or refers to nothing. Seen live: EarPods were connected
        and removed mid-session, after which the loop reopened onto a stale
        device that returned pure silence, failed its liveness check,
        reopened onto the same stale device, and span in that loop
        indefinitely while appearing to be "online and listening".
        Terminating and re-initializing forces a fresh enumeration.
        """
        try:
            sd._terminate()
            sd._initialize()
        except Exception as e:
            print(f"  (could not reinitialize the audio layer: {type(e).__name__}: {e})")
        # Small pause so a device that is genuinely mid-transition (or
        # permanently gone) can't spin this into a hot retry loop.
        time.sleep(1.0)
        self._input_device = self._select_input_device()

    def _conversation_loop(self, mem: MemoryStore) -> None:
        last_reply_text = ""
        while True:
            try:
                last_reply_text = self._run_one_turn(mem, last_reply_text)
            except (_StreamDead, _VoiceLoopShutdown, _MicMuted):
                # Must propagate — _StreamDead needs run()'s outer handler
                # to reopen the audio stream, _VoiceLoopShutdown needs it
                # to actually stop, and _MicMuted needs the `with` in
                # run() to exit so the microphone is really released.
                # None is a per-turn problem to swallow here; caught by
                # the generic handler below, muting turned into an endless
                # retry loop that never closed the device.
                raise
            except Exception as e:
                # Seen live: the whole process died silently mid-turn with
                # no traceback anywhere (not even a Python exception in the
                # log), leaving the user with a Bella that looked "on" but
                # never responded again. Whatever the cause, one bad turn
                # should not be able to take down the entire voice loop —
                # log it loudly and keep listening instead.
                print(f"\n(Unexpected error during that turn — recovering and continuing. {type(e).__name__}: {e})")
                traceback.print_exc()
                last_reply_text = ""

    def _keep_listening(self) -> None:
        """They said Hey Bella and nothing else — wait once for the command.

        Does not open a follow-up window. The next speech is the rest of
        this line. After the answer, Hey Bella is required again.
        """
        self._force_wake_listen = True
        self._awaiting_command_after_name = True
        self.conversation_active_until = None

    def _close_listen_window(self) -> None:
        self.conversation_active_until = None
        self._force_wake_listen = False
        self._awaiting_command_after_name = False

    def _run_one_turn(self, mem: MemoryStore, last_reply_text: str) -> str:
        already_in_conversation = bool(
            getattr(self, "_force_wake_listen", False)
            or getattr(self, "_awaiting_command_after_name", False)
        )
        if already_in_conversation:
            # listen_for_wake_word() is skipped on this path, and that is
            # the only place this flag was ever cleared — so a hotkey wake
            # that fell through to "just the name, listening for the rest"
            # left it set, and the follow-up turn took the hotkey branch
            # below: no drain, no chime, for speech nobody pressed a key
            # for. Continuing a line is never a hotkey wake.
            self._woke_by_hotkey = False
            listening_banner = f"(Heard {self.wake_label()} — listening for the rest of the line...)"
        else:
            # About to go idle on the wake word, so hand the music back
            # before waiting. Pause/resume is per CONVERSATION, not per
            # turn: doing it per turn meant every 3s follow-up listen
            # paused and resumed Spotify, and because the AppleScript state
            # read lags the write, pause_for_conversation eventually read
            # "already paused", recorded nothing to restore, and the music
            # never came back at all — observed live, stuck paused after
            # the conversation window expired.
            spotify.resume_after_conversation()
            self.conversation_active_until = None
            self.listen_for_wake_word()
            listening_banner = "(Wake word detected — listening...)"
        turn_start = time.perf_counter()
        wake_detected_at = time.perf_counter()
        # Marked HERE, at the wake, not when the turn finishes. Background
        # reflection arms its timer on turn end, so nothing pushed it out
        # while the user was mid-sentence — and a 7B model starting then
        # took the next transcription from 0.73s to 28.55s.
        control.note_activity()
        print(listening_banner)
        if not already_in_conversation:
            # Quiet the music for the whole conversation before cueing the
            # user, so neither the chime nor their reply competes with it.
            spotify.pause_for_conversation()
            # Do not chime here. Phrase wake already has the full line
            # ("Bella, what's the weather") — a chime before we use that
            # audio used to eat the request, especially after a short
            # "Bella". Chime only when we still need them to speak.
        return self._handle_turn_audio(
            mem, last_reply_text, already_in_conversation, turn_start, wake_detected_at,
            woke_by_name=not already_in_conversation and not getattr(self, "_woke_by_hotkey", False),
        )

    def _streaming_helps(self) -> bool:
        """True only when the assigned voice model streams progressively.

        A reasoning model does its thinking first and then emits the whole
        answer in a burst, so there is nothing to overlap: measured on
        gpt-oss-20b, first content token 0.96s against 1.01s for the full
        reply. Streaming it was actively WORSE end to end — 3.14s to the
        first spoken word against 2.62s buffered — because the reply got
        split across two synthesis round trips to gain 0.05s of overlap.

        The buffered path already sends a deliberately tiny opening chunk
        (see _split_for_speech), which is where that speed comes from.
        So: stream when the model really streams, otherwise don't. The
        machinery stays wired, and switching the voice role to a
        non-reasoning model turns it on by itself.
        """
        entry = llm.resolve(self.cfg, "voice")
        if entry["provider"] == "ollama":
            return True          # local models emit token by token
        return not llm._is_reasoning_model(entry.get("model", ""))

    def _reply_language(self, mem) -> str:
        """Which voice to synthesize in, decided BEFORE the reply exists.

        The streaming speaker has to be built before the first token
        arrives, so it cannot look at the finished text the way the
        non-streaming path does. The conversation language is the same
        signal process_turn itself routes on, so it agrees by construction.
        """
        return "as-IN" if cli.get_conversation_language(mem) == "as-IN" else "en-IN"

    def _handle_turn_audio(
        self,
        mem: MemoryStore,
        last_reply_text: str,
        already_in_conversation: bool,
        turn_start: float,
        wake_detected_at: float,
        woke_by_name: bool = False,
    ) -> str:
        # Straight after a wake word the user has said they intend to
        # speak, so wait properly for them; a follow-up listen is
        # speculative and should give up quickly. _force_wake_listen is set
        # when they called Bella by name mid-conversation, which is the
        # same explicit "I'm about to talk" signal as a fresh wake.
        # An interrupt only counts while a turn is actually running. This
        # flag is set from two places (push-to-talk, and the web UI's Stop
        # button via SIGUSR2) and was only ever cleared by the barge-in
        # watcher — so a press that arrived with no turn in flight sat
        # there, and then killed the speech of the NEXT reply. Seen live:
        # every turn logging "stopping and listening" straight after the
        # user finished speaking, ttfw=n/a, speak=0.00s, and no audible
        # answer at all. Clearing it here means only an interrupt during
        # this turn can stop this turn.
        self._ptt_interrupt = False
        self._awaiting_rest_of_line = False
        # Reset per turn: a turn that never speaks used to report the
        # PREVIOUS turn's figure, so the log showed ttfw=1.57s next to
        # speak=0.00s four times running.
        self._last_ttfa = None

        pending = None
        open_seg = None
        used_pending = False
        if self.wake_mode == "phrase":
            pending = self.wake_model.take_pending_audio()
            open_seg = self.wake_model.take_open_segment()
            # Short clip whose English wake transcript is only the name:
            # they said "Bella" and the request is still coming. Do not send
            # "বেলা।" to Sarvam — that produced the looping
            # "তুমি আকৌ একো নিলিখিলা" line. Keep longer clips: English
            # Whisper often writes only "Bella." while the same audio still
            # has the Assamese command.
            if pending is not None and pending.size:
                dur = pending.size / SAMPLE_RATE
                wake_tr = self.wake_model.last_transcript or ""
                if _is_bare_wake_phrase(wake_tr) and dur < 1.6:
                    print("(Just the name — listening for the rest of the line.)")
                    pending = None
                    self._keep_listening()
                    already_in_conversation = True
                    # The next recording is the command, not ambient music.
                    # Live: VAD 0.02 / RMS 1106 after "Bella!" was dropped
                    # as "loud but not speech" and the rest of the line died.
                    self._awaiting_rest_of_line = True

        if pending is not None and pending.size:
            # The phrase detector fires at the END of the utterance, so
            # "hey bella, what's the weather" / "bella, ..." is already
            # recorded in full. Reuse it — same words, no second pass.
            self._force_wake_listen = False
            audio, heard_speech = pending, True
            used_pending = True
        elif getattr(self, "_woke_by_hotkey", False):
            # They already pressed the key and usually already spoke.
            # Stitch detector leftover + queued mic frames. Do not drain.
            self._force_wake_listen = False
            parts = [p for p in (open_seg, self._take_queued_frames()) if p is not None and p.size]
            if self.push_to_talk is not None and self.push_to_talk.held.is_set():
                more, _ = self.record_while_held()
                if more.size:
                    parts.append(more)
            buffered = np.concatenate(parts) if parts else np.array([], dtype=np.int16)
            if buffered.size >= _PTT_MIN_COMMAND_SAMPLES:
                audio = buffered
                heard_speech = True
                print(f"  (debug: ptt-keep {audio.size / SAMPLE_RATE:.1f}s, no drain)")
            else:
                # Too short to be a command, so do not pretend it is one.
                #
                # This branch used to accept ANY non-empty buffer and set
                # heard_speech=True. Tapping the key rather than holding it
                # leaves `held` already clear by the time we look, so all
                # that existed was ~0.2s of pre-roll — measured live at 0.2,
                # 0.8, 0.2s. Whisper returns nothing useful for that, so the
                # turn burned a round trip, announced "Just my name", and
                # only THEN began listening — by which point the user had
                # already said the command into a closed window and had to
                # repeat it. Seven of those detours in one session.
                #
                # Keep the pre-roll (they may have started talking on the
                # press) and actually listen for the rest.
                if buffered.size:
                    print(f"  (debug: ptt pre-roll {buffered.size / SAMPLE_RATE:.1f}s "
                          f"— too short for a command, listening)")
                    with self._barge_frames_lock:
                        self._barge_frames.appendleft(buffered)
                audio, heard_speech = self.record_utterance(
                    bail_seconds=self.wake_listen_seconds
                )
        else:
            expecting_speech = not already_in_conversation or getattr(self, "_force_wake_listen", False)
            self._force_wake_listen = False
            if not already_in_conversation:
                self._play_ack_tone()
            audio, heard_speech = self.record_utterance(
                bail_seconds=self.wake_listen_seconds if expecting_speech else None
            )
        utterance_done_at = time.perf_counter()
        # Skip Whisper when nothing was said. This return value used to be
        # discarded here (`audio, _ = ...`), so every no-speech phase still
        # paid the full transcription cost — measured at 10-29s for a 15s
        # silent buffer, on top of the 15s spent recording it.
        #
        # But the VAD does not get the last word, because when it is wrong
        # this drops the turn outright rather than merely slowing it down —
        # seen live scoring 0.03 on audio peaking at RMS 3553, i.e. real
        # sound it refused to call speech. Whisper on a few seconds of
        # audio costs well under a second, so anything louder than the
        # calibrated noise floor gets transcribed regardless and Whisper,
        # which is the better judge, decides whether there were words.
        if not heard_speech:
            peak = getattr(self, "_last_peak_rms", 0.0)
            max_vad = getattr(self, "_last_max_vad", 0.0)
            if peak <= self.silence_rms_threshold:
                print(f"(Didn't catch anything — say {self.wake_label()} again.)\n")
                self._close_listen_window()
                return last_reply_text
            # High RMS + low VAD is the room (Spotify, TV), not a missed
            # utterance. Transcribing it produced live lyric fragments
            # ("I'll keep on feeling high") that looked like Bella had
            # gone mad — and blocked the real command behind them.
            # Exception: they just said the name and we asked for the rest.
            awaiting_rest = getattr(self, "_awaiting_rest_of_line", False)
            self._awaiting_rest_of_line = False
            if max_vad < 0.35 and not awaiting_rest:
                print(
                    f"(Ignoring — loud but not speech, VAD {max_vad:.2f} "
                    f"RMS {peak:.0f}.)\n"
                )
                self._close_listen_window()
                return last_reply_text
            if awaiting_rest:
                print(
                    f"  (name then sound, VAD {max_vad:.2f} RMS {peak:.0f} "
                    f"— transcribing the rest of the line)"
                )
            print(f"  (VAD said no speech but peak RMS {peak:.0f} / VAD {max_vad:.2f} "
                  f"is above the noise floor {self.silence_rms_threshold:.0f} — transcribing anyway)")
        lang = cli.get_conversation_language(mem)
        text = ""
        if used_pending:
            reused = _command_from_wake_transcript(
                getattr(self.wake_model, "last_transcript", "") or "",
                getattr(self.wake_model, "last_match_token", "") or "",
            )
            if reused:
                print(f"  (reused wake stt, skipped second pass: {reused!r})")
                text = reused
        if not text:
            text = self.transcribe(audio, language_code=lang)
        wake_tr = getattr(self.wake_model, "last_transcript", "") or ""
        # Wake already heard English "Good morning Bella" while Saaras
        # wrote Odia. Prefer the wake line when it is a greeting.
        if cli._assamese_is_greeting(wake_tr) and (
            not text or sarvam_client._has_odia(text) or cli._assamese_is_greeting(text)
        ):
            text = wake_tr
        transcription_done_at = time.perf_counter()
        # ALWAYS say what came back, including nothing. This used to print
        # "Didn't catch anything" and discard the text unseen, which made
        # the most common real failure undiagnosable from the log: four
        # utterances in one session were recorded at RMS 2300-3700 with VAD
        # 1.00 — unambiguous speech — and dropped, with no way to tell
        # whether Whisper returned empty or _is_empty_command() classified
        # a real command as the bare wake word. Those are opposite bugs
        # with opposite fixes, and guessing between them is how this got
        # worse before.
        stt_seconds = transcription_done_at - utterance_done_at
        if not text or _is_empty_command(text):
            why = "empty transcript" if not text.strip() else "matched the wake phrase only"
            print(
                f"  (stt {stt_seconds:.2f}s on {audio.size / SAMPLE_RATE:.1f}s of audio"
                f" -> {text!r}: {why})"
            )
            if getattr(self, "_awaiting_command_after_name", False):
                print(f"(Didn't catch anything — say {self.wake_label()} again.)\n")
                self._close_listen_window()
                return last_reply_text
            print("(Just my name — listening for what you need.)\n")
            self._keep_listening()
            return last_reply_text
        if stt_seconds > 2.0:
            # Whisper on a few seconds of audio is normally well under a
            # second; anything near ten means something else is eating the
            # CPU (measured: 8.54s while a local 7B reflection ran).
            print(f"  (stt took {stt_seconds:.2f}s — something else is using the CPU)")

        if already_in_conversation and _is_bare_wake_phrase(text):
            print("(Already listening — go ahead.)\n")
            self._keep_listening()
            return last_reply_text

        if self.wake_mode == "phrase":
            text = _strip_matched_name(text, self.wake_model.last_match_token)
        text = _strip_leading_wake_phrase(text)
        text = wake_phrase.strip_address_name(text)

        if _is_empty_command(text):
            print("(Just my name — listening for what you need.)\n")
            self._keep_listening()
            return last_reply_text

        print(f"You said: {text}")

        # The echo filter only applies to speech that could physically BE
        # an echo: Bella's voice reaching the mic while, or just after, it
        # was speaking. Two guards, both learned from the log.
        #
        # Seen live: "Hey Oren, what's the weather in Bangalore?" asked
        # four times in a row, every one discarded as an echo, because the
        # previous reply was about the weather and shared most of its
        # words. From the user's side Bella had simply gone deaf. A
        # follow-up question naturally reuses the words of the answer it
        # follows — that is what a conversation is — so word overlap alone
        # can never be enough.
        #
        #   1. If they woke Bella by name for this turn, it is not an echo.
        #      Saying the wake phrase is proof of intent; Bella's own voice
        #      never contains it.
        #   2. Otherwise it must arrive within ECHO_WINDOW_SECONDS of Bella
        #      actually finishing speaking. Past that the speakers are
        #      silent and there is nothing left to bleed through.
        since_speech = time.monotonic() - getattr(self, "_last_speech_ended_at", 0.0)
        could_be_echo = (
            already_in_conversation
            and not woke_by_name
            and since_speech <= ECHO_WINDOW_SECONDS
        )
        if could_be_echo and last_reply_text and _looks_like_self_echo(text, last_reply_text):
            print("(Ignoring — sounds like my own voice bleeding into the mic, not a new command.)\n")
            return last_reply_text

        normalized = _normalize_for_compare(text)
        if SHUTDOWN_RE.search(normalized):
            self._dismiss_screen_help()
            raise _VoiceLoopShutdown()
        if INTERRUPT_RE.search(normalized) and not _NOT_AN_INTERRUPT_RE.search(normalized):
            # "Stop" used to be lumped in with the shutdown phrases
            # above, which meant saying it killed the whole voice
            # loop process — you'd have to restart Bella just to
            # get it to stop talking. This just cancels the current
            # turn and goes back to listening.
            print("(Stopping — standing by. Say Hey Bella when you need me.)\n")
            self._dismiss_screen_help()
            self._close_listen_window()
            return last_reply_text

        barge_event = threading.Event()
        # Cleared per turn; only gates whether _watch_for_barge_in's
        # detection logic is armed, not when the thread itself
        # starts (that still has to start early, before the
        # response exists, so real-time streaming playback can be
        # barged in on the instant it begins).
        playback_active = threading.Event()
        # Armed for the entire cli.process_turn() call below (LLM
        # generation, Claude Code handoff, shell execution — not just TTS
        # playback) — explicitly requested: saying "stop" should actually
        # kill an in-flight task (Claude Code can run for minutes), not
        # just silence the eventual reply once it's done. See the
        # cancel_current() calls in _watch_for_barge_in. An instance
        # attribute, not a local — voice_confirm() (below) needs to reach
        # it too, to pause arming while it's asking its own yes/no
        # question (see voice_confirm's own comment for why).
        self._processing_active = threading.Event()
        processing_active = self._processing_active
        barge_monitor = None
        # No headphones/speakers distinction any more: _watch_for_barge_in
        # triggers on the wake word rather than on loudness, and Bella's
        # own voice can never contain the wake word, so echo through open
        # speakers is harmless by construction.
        if self.cfg.get("tts_enabled"):
            barge_monitor = threading.Thread(
                target=self._watch_for_barge_in,
                args=(barge_event, playback_active, processing_active),
                daemon=True,
            )
            barge_monitor.start()

        # Speak while the model is still writing. Only the plain-chat
        # path calls stream_callback (see cli.handle_ollama); a tool answer
        # arrives complete and falls through to _speak_streaming below,
        # exactly as the web bridge distinguishes the two.
        speaker = None
        if self.cfg.get("tts_enabled") and self._streaming_helps():
            speaker = _StreamingSpeaker(
                self, self._reply_language(mem), barge_event, playback_active
            )

        processing_active.set()
        try:
            backend, response = cli.process_turn(
                cli.apply_language_trigger(text),
                self.cfg,
                mem,
                interactive=True,
                confirm_fn=self.voice_confirm,
                recent_turns_limit=self.cfg.get("voice_context_turns", 6),
                source="voice",
                # voice_ollama_model pins a smaller *local* model for
                # spoken turns. It must not survive a dashboard choice of
                # a cloud model for the voice role, or picking one would
                # appear to do nothing at all — so it applies only while
                # the voice role still resolves to something local.
                model_override=self._voice_model_override(),
                model_role="voice",
                ollama_options={"num_predict": self.cfg.get("voice_ollama_max_tokens", 128)},
                stream_callback=speaker.feed if speaker else None,
            )
        except (cli.claude_handoff.Cancelled, cli.shell.Cancelled):
            print("(Cancelled — say Hey Bella for the next instruction.)\n")
            if speaker is not None:
                speaker.abandon()
            barge_event.set()
            if barge_monitor is not None:
                barge_monitor.join(timeout=1)
            self._close_listen_window()
            return ""
        finally:
            processing_active.clear()
        response_ready_at = time.perf_counter()
        print(f"{cli.label(backend)} {response}\n")
        last_reply_text = response
        interrupted = barge_event.is_set()
        # Everything goes out through Bulbul (Sarvam's TTS) — one
        # consistent voice rather than Piper's British one.
        if interrupted:
            if speaker is not None:
                speaker.abandon()
            print("(Skipping speech output because you already barged in.)")
        elif speaker is not None and speaker.spoke:
            # The reply streamed: playback started while the model was
            # still writing, and finish() only waits for the tail.
            interrupted = not speaker.finish()
        elif self.cfg.get("tts_enabled"):
            if speaker is not None:
                speaker.abandon()
            language_code = backend.split(":", 1)[1] if backend.startswith("sarvam:") else "en-IN"
            if re.search(r"[\u0980-\u09FF]", response or ""):
                language_code = "as-IN"
            interrupted = not self._speak_streaming(
                response, language_code, barge_event, playback_active
            )

        barge_event.set()
        self._last_speech_ended_at = time.monotonic()
        if barge_monitor is not None:
            barge_monitor.join(timeout=1)

        # A barge-in that was the WAKE PHRASE is a new command, not just a
        # request for silence — "Bella, can you play something" said while
        # Bella is still working means both "stop" and "here is the next
        # thing". The wake detector already captured that utterance in
        # pending_audio, but the turn used to end by closing the window,
        # which sent the loop back to waiting for a wake word and threw
        # the command away. Measured live: four replies in a row spoke
        # nothing and every follow-up request vanished — Bella looked like
        # it was running with nothing connected to anything.
        #
        # An explicit "Bella stop" is the opposite and must still go quiet,
        # which is what _stop_requested distinguishes.
        pending_command = (
            self.wake_mode == "phrase"
            and not self._stop_requested
            and getattr(self.wake_model, "pending_audio", None) is not None
            and self.wake_model.pending_audio.size > 0
        )
        self._stop_requested = False
        if pending_command:
            print("(Heard a new request while working — taking it now.)")
            self._keep_listening()
        else:
            self._close_listen_window()
        spoken_done_at = time.perf_counter()
        # ttfw (time to first word) is the number that actually describes
        # how responsive this feels; the old single "tts=" figure lumped it
        # together with how long the reply took to say out loud, so a long
        # answer looked identical to a slow one. speak= is that speaking
        # duration, which is supposed to be long for a long answer.
        ttfa = getattr(self, "_last_ttfa", None)
        ttfw = f"{ttfa:.2f}s" if ttfa is not None else "n/a"
        print(
            "  (timing: "
            f"wake={wake_detected_at - turn_start:.2f}s, "
            f"record={utterance_done_at - wake_detected_at:.2f}s, "
            f"stt={transcription_done_at - utterance_done_at:.2f}s, "
            f"llm={response_ready_at - transcription_done_at:.2f}s, "
            f"ttfw={ttfw}, "
            f"speak={spoken_done_at - response_ready_at:.2f}s)"
        )
        return last_reply_text


_SINGLETON_PID = Path("/tmp/bella_voice_loop.pid")


def _acquire_singleton() -> bool:
    """Refuse to start a second listener. Two loops on one mic is how
    the HUD 'Start Voice' button made Bella randomly deaf."""
    if _SINGLETON_PID.exists():
        try:
            old = int(_SINGLETON_PID.read_text().strip())
        except ValueError:
            old = 0
        if old and old != os.getpid():
            try:
                os.kill(old, 0)
            except OSError:
                pass
            else:
                print(
                    f"Another voice loop is already running (pid {old}). "
                    "Not starting a second one."
                )
                return False
    _SINGLETON_PID.write_text(str(os.getpid()))

    def _release() -> None:
        try:
            if _SINGLETON_PID.read_text().strip() == str(os.getpid()):
                _SINGLETON_PID.unlink(missing_ok=True)
        except OSError:
            pass

    import atexit
    atexit.register(_release)
    return True


def main():
    if not _acquire_singleton():
        sys.exit(0)
    from bella.security import harden_home
    harden_home()
    cfg = load_config()
    active_loop = None
    pending_wake = False

    def _handle_wake_signal(signum, frame):
        nonlocal pending_wake
        if active_loop is None:
            pending_wake = True
        else:
            active_loop.trigger_wake()

    def _handle_interrupt_signal(signum, frame):
        # No pending-interrupt equivalent to pending_wake: there is
        # nothing to interrupt before the loop exists, and queueing a
        # stop until after startup would silence the first real reply.
        if active_loop is not None:
            active_loop.trigger_interrupt()

    signal.signal(signal.SIGUSR1, _handle_wake_signal)
    signal.signal(signal.SIGUSR2, _handle_interrupt_signal)
    try:
        loop = VoiceLoop(cfg)
        active_loop = loop
        if pending_wake:
            loop.trigger_wake()
        loop.run()
    except KeyboardInterrupt:
        print("\nVoice loop stopped.")
        sys.exit(0)
    except Exception:
        # Last-resort safety net for anything outside _run_one_turn's own
        # per-turn recovery (model loading, calibration, stream setup).
        # Seen live: the process died once with zero traceback anywhere —
        # not in this log, not in a macOS crash report — leaving no way to
        # diagnose what actually happened. If this fires, at minimum
        # there's now a full traceback on record before the process exits.
        print("\n(Voice loop crashed at the top level — see traceback below.)")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
