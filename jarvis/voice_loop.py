"""Always-on voice loop: continuous mic capture -> wake word ("Hey Jarvis")
-> record utterance -> local transcription -> normal JARVIS pipeline -> TTS.

Everything here is local: openWakeWord for the wake word, faster-whisper for
transcription of the triggered utterance. Wispr Flow isn't used in this
mode — it's a dictation tool bound to a focused text field, not an ambient
background listener, so it can't drive this loop. The terminal `cli.py`
loop (where Wispr dictation IS used) is unaffected by any of this.

Run with: python3 -m jarvis.voice_loop
"""
import queue
import sys
import time
import threading
import traceback
import re
import signal
from collections import deque

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel

from jarvis import cli, sarvam_client
from jarvis.config import load_config
from jarvis.memory import MemoryStore
from jarvis.tools import spotify
from jarvis.vad import SileroVAD, CHUNK_SAMPLES as VAD_CHUNK_SAMPLES

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
# JARVIS looks completely dead before it says anything at all. That is the
# "JARVIS is sleeping / sometimes it just doesn't respond" symptom, and it
# is a bug, not a tuning problem. Real assistants give you a few seconds to
# start talking and then quietly stand down.
NO_SPEECH_BAIL_SECONDS = 3.0
# Word-boundary contains-match, not exact set membership — seen live:
# exact matching missed "Hey stop!" (normalizes to "hey stop", which isn't
# in the set) and sent it to Ollama as a normal query instead of
# interrupting. People naturally wrap "stop" in filler ("hey", "please",
# "can you"), so the phrase needs to be found anywhere in the utterance,
# not be the whole thing. SHUTDOWN is checked first in run() and `break`s
# immediately, so "stop listening" is claimed by SHUTDOWN_RE before
# INTERRUPT_RE's bare "stop" ever gets a chance to also match it.
INTERRUPT_RE = re.compile(r"\b(?:stop(?:\s+jarvis)?|jarvis\s+stop|be\s+quiet|quiet|silence)\b")
SHUTDOWN_RE = re.compile(r"\b(?:stop\s+listening|quit|exit|shutdown|go\s+to\s+sleep)\b")

# cli.process_turn's /hindi and /assamese routing needs a literal slash
# prefix — fine for typed text, impossible to actually *say*. These map
# spoken phrasing onto the same routing: strip the trigger phrase out of
# the utterance and prepend the slash command cli.py already understands,
# rather than inventing a second routing mechanism.
_HINDI_TRIGGER_RE = re.compile(r"\b(?:in\s+hindi|hindi\s+mein|speak\s+(?:in\s+)?hindi)\b", re.IGNORECASE)
_ASSAMESE_TRIGGER_RE = re.compile(r"\b(?:in\s+assamese|assamese\s+mein|speak\s+(?:in\s+)?assamese)\b", re.IGNORECASE)


def _apply_language_trigger(text: str) -> str:
    if _HINDI_TRIGGER_RE.search(text):
        return "/hindi " + _HINDI_TRIGGER_RE.sub("", text).strip()
    if _ASSAMESE_TRIGGER_RE.search(text):
        return "/assamese " + _ASSAMESE_TRIGGER_RE.sub("", text).strip()
    return text


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", "", text.lower()).strip()


# Exact normalized forms only (not a fuzzy/substring check) — deliberately
# narrow, so a genuine follow-up that happens to start with the name (e.g.
# "Jarvis, can you check my email") is never swallowed by this, only a
# bare repeat of the wake phrase itself. Covers the STT variants actually
# seen live for "Hey Jarvis" (accent-stripped, fused words, "Jervis").
_BARE_WAKE_PHRASES = {
    "hey jarvis", "hey jervis", "hey jrvis", "hey jarviss",
    "hi jarvis", "hej jarvis", "hej jrvis", "hej jervis",
    "jarvis", "jervis", "jrvis", "hejervis",
}


def _is_bare_wake_phrase(text: str) -> bool:
    """True if text is just the wake phrase repeated and nothing else.

    Seen live: saying "Hey Jarvis" again while a conversation window is
    already open gets transcribed as ordinary follow-up speech (no wake-
    word model involved — that only runs when NOT already in an active
    window) and sent straight to the LLM, which answers as if freshly
    greeted ("Hello! How can I assist you today?") — redundant and a bit
    jarring when the conversation was already live. Only checked when
    already in an active window (see _run_one_turn); saying the wake
    phrase to genuinely start a fresh conversation is unaffected.
    """
    return _normalize_for_compare(text) in _BARE_WAKE_PHRASES


_LEADING_WAKE_PHRASE_RE = re.compile(
    r"^(?:hey|hi|hej|hail)\s+(?:jarvis|jervis|jrvis)\b[!,.]?\s*",
    re.IGNORECASE,
)


def _strip_leading_wake_phrase(text: str) -> str:
    """Strips a re-said wake phrase from the start of an utterance that has
    more to it than just the phrase — same STT variants as
    _BARE_WAKE_PHRASES (accent-stripped, fused words, "Jervis"/"Hail
    Jervis"), but as a prefix rather than a whole-utterance match. Seen
    live: users often repeat the wake phrase out of habit mid-conversation,
    and the misheard variant then ends up baked into the actual command
    text, corrupting deterministic extraction downstream — a shell command
    literally starting with "Hail Jervis, run the command, sleep 30" ran
    `Hail` as the command and failed instead of ever running `sleep 30`.
    """
    stripped = _LEADING_WAKE_PHRASE_RE.sub("", text.strip(), count=1)
    return stripped or text


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

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


def _split_for_speech(
    text: str, first_min_chars: int = 40, min_chars: int = 140, max_chars: int = 300
) -> list[str]:
    """Break a reply into sentence-ish chunks for streaming synthesis.

    The FIRST chunk is deliberately short and every later one is longer.
    Measured against the live Sarvam API, synthesis cost is roughly a 0.6s
    fixed overhead plus ~0.02s per character (4 chars 0.79s, 38 chars 1.46s,
    142 chars 3.35s, 217 chars 5.34s) — so the only thing that decides how
    long the user waits in silence is the length of the first chunk. Later
    chunks are synthesized while earlier audio is still playing, where
    there is plenty of slack (a 140-char chunk is ~9s of speech), so making
    them bigger costs nothing and avoids choppy sentence-by-sentence
    delivery. min_chars also stops us paying a whole round-trip for "Yes."
    """
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for part in parts:
        buf = f"{buf} {part}".strip() if buf else part
        threshold = first_min_chars if not chunks else min_chars
        if len(buf) >= threshold:
            chunks.append(buf)
            buf = ""
    if buf:
        # A tiny trailing fragment rides along with the previous chunk
        # rather than costing its own round-trip.
        if chunks and len(buf) < 25:
            chunks[-1] = f"{chunks[-1]} {buf}"
        else:
            chunks.append(buf)
    return [c[:max_chars] for c in chunks]


# Utterances shorter than this are never classified as an echo at all, and
# below _FUZZY_ECHO_MIN_WORDS the bag-of-words overlap check specifically
# is noise rather than signal. See the use sites in _looks_like_self_echo.
_ECHO_MIN_WORDS = 5
_FUZZY_ECHO_MIN_WORDS = 5


def _looks_like_self_echo(new_text: str, last_reply: str) -> bool:
    """True if new_text is plausibly JARVIS's own just-spoken reply bleeding
    back through the mic, not a new thing the user said.

    Seen live: no acoustic echo cancellation means the mic can pick up
    JARVIS's own speaker output. That gets transcribed as if it were a
    fresh utterance, gets sent back through the LLM, gets a reply, and the
    reply itself bleeds through again — a self-sustaining conversation loop
    with nobody talking, each reply drifting further from anything the user
    actually said (this is what "the flow isn't maintained, it's writing
    gibberish" looks like from the outside — it's not incoherent generation,
    it's JARVIS answering a mangled transcript of itself).

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
    Seen live: JARVIS's own longer replies (especially tool results like a
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
    # answers a question using the question's own words — JARVIS asked
    # "...or exploring some music recommendations?", the user said
    # "recommendations", and it was discarded as an echo of the very reply
    # it was answering. Same for "yes", "stop", "30", "command". A one- or
    # two-word bleed-through does get through as a result, but that is a
    # far cheaper mistake than ignoring what the user actually said, which
    # is the complaint that mattered.
    if n < _ECHO_MIN_WORDS:
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



class VoiceLoop:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.wake_word = cfg["voice_wake_word"]
        self.silence_rms_threshold = cfg["voice_silence_rms_threshold"]
        self.silence_seconds = cfg.get("voice_silence_seconds", 0.7)
        self.no_speech_bail_seconds = cfg.get("voice_no_speech_bail_seconds", NO_SPEECH_BAIL_SECONDS)
        self.audio_q: queue.Queue = queue.Queue()
        self._last_frame_at = time.monotonic()
        self.wake_event = threading.Event()
        # Monotonic deadline for staying in "listening for follow-up" mode
        # without requiring the wake word again. None means "need the wake
        # word." Deliberately a timestamp, not the single-shot boolean this
        # used to be — the old boolean got consumed at the top of each loop
        # iteration and only re-armed at the end of a *completed* turn, so
        # any single "didn't catch anything" (a pause, speaking too quietly)
        # silently dropped back to requiring "Hey Jarvis" again. That was
        # the actual mechanical cause of the conversation feeling like it
        # kept turning on and off. A deadline that persists across empty
        # attempts and only expires after real inactivity fixes that.
        self.conversation_active_until: float | None = None
        self._barge_frames: deque[np.ndarray] = deque(maxlen=12)
        self._barge_frames_lock = threading.Lock()
        print(f'Loading wake word model ("{self.wake_word}")...')
        self.wake_model = WakeWordModel(wakeword_models=[self.wake_word], inference_framework="onnx")
        print(f'Loading local speech-to-text model (faster-whisper {cfg["voice_stt_model"]})...')
        self.whisper = WhisperModel(cfg["voice_stt_model"], device="cpu", compute_type="int8")
        print("Loading voice activity detector (Silero VAD, onnx)...")
        self.vad = SileroVAD()
        self._vad_buffer = np.zeros(0, dtype=np.float32)

    def trigger_wake(self) -> None:
        self.wake_event.set()

    def _audio_callback(self, indata, frames, time_info, status):
        self._last_frame_at = time.monotonic()
        self.audio_q.put(indata[:, 0].copy())

    def _next_frame(self) -> np.ndarray:
        while True:
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
        sizes. Returns the max speech probability across whatever whole
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
        advance. Uses the median (not max) of the sampled window: max is
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
        calibrated = max(self.silence_rms_threshold, median * 2.2)
        print(f"  ambient RMS median {median:.0f} / p90 {p90:.0f} -> silence threshold set to {calibrated:.0f}")
        self.silence_rms_threshold = calibrated
        print('  barge-in: say "Hey Jarvis" to interrupt (wake-word gated, not volume)')

    def listen_for_wake_word(self) -> None:
        self.wake_model.reset()
        if self.wake_event.is_set():
            self.wake_event.clear()
            return
        while True:
            if self.wake_event.is_set():
                self.wake_event.clear()
                return
            frame = self._next_frame()
            prediction = self.wake_model.predict(frame)
            if prediction.get(self.wake_word, 0.0) > WAKE_THRESHOLD:
                return

    def _watch_for_barge_in(
        self,
        stop_event: threading.Event,
        playback_active: threading.Event,
        processing_active: threading.Event | None = None,
    ) -> None:
        """Interrupt on the WAKE WORD, not on loudness.

        This machine has no acoustic echo cancellation, so the mic hears
        JARVIS's own speakers. An energy threshold cannot tell that apart
        from the user interrupting, and measured over 416 real turns it
        fired 179 times (43%) — JARVIS cutting its own replies into
        fragments, which is what "it says three or four lines and then
        drops" actually was. Chasing it with a higher threshold and a
        fuzzy transcript filter also ate 36 genuine user commands.

        openWakeWord makes the problem structurally impossible instead:
        JARVIS's own speech never contains "Hey Jarvis", so it cannot
        trigger this, at any volume. Same contract Alexa uses — say the
        wake word to interrupt.
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
            score = self.wake_model.predict(frame).get(self.wake_word, 0.0)
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

    def record_utterance(self, max_seconds: float = MAX_UTTERANCE_SECONDS) -> tuple[np.ndarray, bool]:
        # Drain first: the InputStream callback queues frames continuously,
        # including the entire time JARVIS was mid-reply (LLM latency +
        # TTS playback, seen live running 20-40+ seconds). Nothing here
        # used to discard that backlog before listening again, so a fresh
        # "recording" could start by consuming hundreds of queued frames
        # of JARVIS's OWN just-finished speech before ever reaching
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
        bail_chunks = max(1, int(self.no_speech_bail_seconds * SAMPLE_RATE / FRAME_SAMPLES))
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
        silence before JARVIS said a single word. Synthesis of chunk N+1
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
            for chunk in chunks:
                if barge_event.is_set():
                    break
                try:
                    wav_q.put(sarvam_client.text_to_speech(chunk, language_code=language_code))
                except sarvam_client.SarvamError as e:
                    print(f"(Sarvam TTS unavailable: {e})")
                    break
            wav_q.put(None)

        threading.Thread(target=produce, daemon=True).start()

        first = True
        try:
            while True:
                wav_bytes = wav_q.get()
                if wav_bytes is None:
                    return True
                if barge_event.is_set():
                    return False
                if first:
                    # Arm barge-in only once there's real audio, not during
                    # the network call that produces it.
                    playback_active.set()
                    first = False
                    self._last_ttfa = time.perf_counter() - started_at
                if not cli.tts.play(wav_bytes, stop_event=barge_event):
                    return False
        finally:
            # THE source of self-echo, and it was never a text problem.
            # _watch_for_barge_in appends every frame it sees into
            # _barge_frames, including all the frames that are JARVIS's own
            # voice coming back through the mic during playback. That deque
            # is then used verbatim as the opening ~1s of the NEXT
            # recording (see record_utterance) — _drain_queue() empties
            # audio_q but has never touched this. So the tail of JARVIS's
            # own sentence was being prepended to whatever the user said
            # next, which is what the transcript-similarity heuristics have
            # been trying and failing to clean up after ever since.
            # Dropping it here means the pre-roll only ever holds audio
            # captured after JARVIS stopped talking, while still doing its
            # real job of catching a user who starts speaking early.
            with self._barge_frames_lock:
                self._barge_frames.clear()

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        audio_float = audio.astype(np.float32) / 32768.0
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
        print(f"[Router] {prompt}")
        # Was self.speak() (cli.speak_response -> Piper), left over from
        # before the Bulbul-everywhere switch — every other reply plays
        # through Sarvam now, so the confirm prompt suddenly speaking in a
        # different voice (Piper's British accent) was jarring and
        # confusing on its own, on top of everything else. No Piper
        # fallback on failure, same as every other Sarvam call site since
        # that switch — just report and move on.
        if self.cfg.get("tts_enabled"):
            try:
                wav_bytes = sarvam_client.text_to_speech(prompt + " Say yes or no.", language_code="en-IN")
                cli.tts.play(wav_bytes)
            except sarvam_client.SarvamError as e:
                print(f"(Sarvam TTS unavailable: {e})")
        print("(listening for yes/no...)")
        audio, heard_speech = self.record_utterance(max_seconds=CONFIRM_MAX_SECONDS)
        if not heard_speech:
            print("  (heard: nothing above the noise floor — treating as no.)")
            return False

        text = self.transcribe(audio).lower().strip()
        print(f"  (heard: {text!r})")
        if not text:
            return False
        if CONFIRM_NO_RE.search(text):
            return False
        if len(text.split()) > CONFIRM_MAX_WORDS:
            print("  (heard something too long/ambiguous to count as a clear yes — treating as no.)")
            return False
        return bool(CONFIRM_YES_RE.search(text))

    def run(self) -> None:
        mem = MemoryStore()
        wake_phrase = self.wake_word.replace("_", " ").title()
        first_start = True
        while True:
            self._last_frame_at = time.monotonic()
            try:
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                    blocksize=FRAME_SAMPLES,
                    callback=self._audio_callback,
                ):
                    self._drain_queue()
                    self.calibrate()
                    if first_start:
                        first_start = False
                        print(f'JARVIS voice loop online, {self.cfg["user_name"]}. Say "{wake_phrase}" to begin.\n')
                    else:
                        print(f'JARVIS back online, {self.cfg["user_name"]}. Say "{wake_phrase}" to begin.\n')
                    self._conversation_loop(mem)
            except _StreamDead:
                print(
                    "\n(Audio stream stopped responding — likely a microphone/audio "
                    "device change. Reopening against the current default input...)\n"
                )
                continue
            except _VoiceLoopShutdown:
                print("Stopping voice loop on request.")
                return

    def _conversation_loop(self, mem: MemoryStore) -> None:
        last_reply_text = ""
        while True:
            try:
                last_reply_text = self._run_one_turn(mem, last_reply_text)
            except (_StreamDead, _VoiceLoopShutdown):
                # Must propagate — _StreamDead needs run()'s outer handler
                # to reopen the audio stream, _VoiceLoopShutdown needs it
                # to actually stop, neither is a per-turn problem to
                # swallow here.
                raise
            except Exception as e:
                # Seen live: the whole process died silently mid-turn with
                # no traceback anywhere (not even a Python exception in the
                # log), leaving the user with a JARVIS that looked "on" but
                # never responded again. Whatever the cause, one bad turn
                # should not be able to take down the entire voice loop —
                # log it loudly and keep listening instead.
                print(f"\n(Unexpected error during that turn — recovering and continuing. {type(e).__name__}: {e})")
                traceback.print_exc()
                last_reply_text = ""

    def _run_one_turn(self, mem: MemoryStore, last_reply_text: str) -> str:
        already_in_conversation = (
            self.conversation_active_until is not None and time.monotonic() < self.conversation_active_until
        )
        if already_in_conversation:
            remaining = self.conversation_active_until - time.monotonic()
            listening_banner = f"(Listening for your follow-up — window open {remaining:.0f}s more...)"
        else:
            self.conversation_active_until = None
            self.listen_for_wake_word()
            listening_banner = "(Wake word detected — listening...)"
        turn_start = time.perf_counter()
        wake_detected_at = time.perf_counter()
        print(listening_banner)
        # Paused for the whole turn — recording through the spoken reply —
        # not just the recording, since background music would otherwise
        # talk over JARVIS's own reply too. resume_after_conversation() in
        # `finally` guarantees playback resumes no matter which exit path
        # below fires (nothing heard, self-echo filtered, "stop", a normal
        # completed reply, or even a shutdown).
        spotify.pause_for_conversation()
        try:
            return self._handle_turn_audio(mem, last_reply_text, already_in_conversation, turn_start, wake_detected_at)
        finally:
            spotify.resume_after_conversation()

    def _handle_turn_audio(
        self,
        mem: MemoryStore,
        last_reply_text: str,
        already_in_conversation: bool,
        turn_start: float,
        wake_detected_at: float,
    ) -> str:
        audio, heard_speech = self.record_utterance()
        utterance_done_at = time.perf_counter()
        # Skip Whisper entirely when the VAD is confident nothing was said.
        # This return value used to be discarded here (`audio, _ = ...`),
        # so every no-speech phase still paid the full transcription cost —
        # measured live at 10-29s for a 15s silent buffer, on top of the
        # 15s spent recording it. Silero already answered the only question
        # transcription would have answered.
        if not heard_speech:
            print("(Didn't catch anything.)\n")
            return last_reply_text
        text = self.transcribe(audio)
        transcription_done_at = time.perf_counter()

        if not text:
            # Used to auto-relax silence_rms_threshold here after
            # repeated misses, to correct for a miscalibrated RMS
            # threshold. VAD-based heard_speech (see
            # record_utterance) doesn't need per-room calibration in
            # the first place, so that workaround — and its side
            # effect of also lowering barge_in_threshold, observed
            # live making barge-in fire on near-ambient noise — is
            # gone along with the problem it was working around.
            print("(Didn't catch anything.)\n")
            return last_reply_text

        print(f"You said: {text}")

        if already_in_conversation and _is_bare_wake_phrase(text):
            print("(Already listening — no need to say the wake word again, go ahead.)\n")
            return last_reply_text

        text = _strip_leading_wake_phrase(text)

        if last_reply_text and _looks_like_self_echo(text, last_reply_text):
            print("(Ignoring — sounds like my own voice bleeding into the mic, not a new command.)\n")
            return last_reply_text

        normalized = _normalize_for_compare(text)
        if SHUTDOWN_RE.search(normalized):
            raise _VoiceLoopShutdown()
        if INTERRUPT_RE.search(normalized):
            # "Stop" used to be lumped in with the shutdown phrases
            # above, which meant saying it killed the whole voice
            # loop process — you'd have to restart JARVIS just to
            # get it to stop talking. This just cancels the current
            # turn and goes back to listening.
            print("(Stopping — still listening, just not talking.)\n")
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
        # triggers on the wake word rather than on loudness, and JARVIS's
        # own voice can never contain the wake word, so echo through open
        # speakers is harmless by construction.
        if self.cfg.get("tts_enabled"):
            barge_monitor = threading.Thread(
                target=self._watch_for_barge_in,
                args=(barge_event, playback_active, processing_active),
                daemon=True,
            )
            barge_monitor.start()

        processing_active.set()
        try:
            backend, response = cli.process_turn(
                _apply_language_trigger(text),
                self.cfg,
                mem,
                interactive=True,
                confirm_fn=self.voice_confirm,
                recent_turns_limit=self.cfg.get("voice_context_turns", 6),
                model_override=self.cfg.get("voice_ollama_model", self.cfg["ollama_model"]),
                ollama_options={"num_predict": self.cfg.get("voice_ollama_max_tokens", 128)},
                stream_callback=None,
                source="voice",
            )
        except (cli.claude_handoff.Cancelled, cli.shell.Cancelled):
            print("(Cancelled — ready for your next instruction.)\n")
            barge_event.set()
            if barge_monitor is not None:
                barge_monitor.join(timeout=1)
            window = self.cfg.get("voice_conversation_window_seconds", 180)
            self.conversation_active_until = time.monotonic() + window
            return ""
        finally:
            processing_active.clear()
        response_ready_at = time.perf_counter()
        print(f"{cli.label(backend)} {response}\n")
        last_reply_text = response
        interrupted = barge_event.is_set()
        # Every reply now goes through Bulbul (Sarvam's TTS), not
        # just Sarvam-backed turns — user explicitly chose one
        # consistent Indian-accented voice over Piper's British
        # voice + per-sentence streaming, accepting the resulting
        # wait for full non-streaming synthesis on every turn.
        if interrupted:
            print("(Skipping speech output because you already barged in.)")
        elif self.cfg.get("tts_enabled"):
            language_code = backend.split(":", 1)[1] if backend.startswith("sarvam:") else "en-IN"
            interrupted = not self._speak_streaming(
                response, language_code, barge_event, playback_active
            )

        barge_event.set()
        if barge_monitor is not None:
            barge_monitor.join(timeout=1)
        # A wake starts a conversational session. Every completed
        # turn (interrupted or not) refreshes the window so the user
        # doesn't need to repeat the wake phrase for as long as the
        # conversation stays active — including through any number
        # of quiet/missed attempts in between, unlike the old
        # single-shot flag this replaced.
        window = self.cfg.get("voice_conversation_window_seconds", 180)
        self.conversation_active_until = time.monotonic() + window
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


def main():
    cfg = load_config()
    active_loop = None
    pending_wake = False

    def _handle_wake_signal(signum, frame):
        nonlocal pending_wake
        if active_loop is None:
            pending_wake = True
        else:
            active_loop.trigger_wake()

    signal.signal(signal.SIGUSR1, _handle_wake_signal)
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
