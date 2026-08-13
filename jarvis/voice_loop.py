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
    words-appear-anywhere check for short utterances, which catches that
    case; the trade-off is a real but rare new command that happens to
    reuse most of the same words as the last reply could get misread as an
    echo. Given the alternative is full conversations derailing into
    self-talk, that trade-off is worth it. The actual fix would be real
    acoustic echo cancellation (feeding the outgoing TTS audio back as a
    reference signal to subtract from the mic input) — a real DSP project,
    not a text heuristic; this stays a mitigation until that exists.
    """
    # Word-level containment, not character-level: a plain substring check
    # matches "yo" inside "...assist YOu today" — same class of mistake as
    # the voice_confirm substring bug (see CONFIRM_YES_RE above).
    new_words = _normalize_for_compare(new_text).split()
    reply_words = _normalize_for_compare(last_reply).split()
    if not new_words or not reply_words or len(new_words) > 6:
        return False
    n = len(new_words)
    if any(reply_words[i : i + n] == new_words for i in range(len(reply_words) - n + 1)):
        return True
    reply_word_set = set(reply_words)
    overlap = sum(1 for w in new_words if w in reply_word_set)
    return overlap / len(new_words) >= 0.6


# Anything NOT matching one of these is assumed to be open speakers in the
# same room as the mic — the case where JARVIS's own TTS output can bleed
# back into its own recording. On headphones that path doesn't exist, so
# barge-in can trust the normal calibrated threshold; on speakers it needs
# extra margin. Checked by device *name*, since there's no cross-platform
# API that just says "is this headphones" — sounddevice only exposes
# whatever CoreAudio calls the device.
_HEADPHONE_NAME_MARKERS = (
    "headphone", "headset", "earpods", "earbuds", "airpods", "beats",
    "earphone", "iem", "in-ear",
)


def _headphones_connected() -> bool:
    try:
        idx = sd.default.device[1]
        name = sd.query_devices(idx)["name"].lower()
    except Exception:
        return False
    return any(marker in name for marker in _HEADPHONE_NAME_MARKERS)


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
        self.barge_in_threshold, self.barge_in_sustain_chunks = self._compute_barge_in_params()
        self.barge_in_grace_seconds = 0.35
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

    def _compute_barge_in_params(self) -> tuple[float, int]:
        """Barge-in RMS threshold + required sustained chunks, adjusted for
        whether output is currently routing through headphones.

        On headphones, JARVIS's own voice can't re-enter the mic, so the
        normal calibrated-ambient threshold is trustworthy — a loud,
        sustained sound really is someone interrupting. Through open
        speakers (a laptop's default), the mic WILL pick up JARVIS's own
        playback — seen live producing RMS 4000-8000 against a calibrated
        threshold of only ~2000-3500 — so a higher bar and a longer sustain
        requirement cuts down how often that gets mistaken for a real
        interruption. _looks_like_self_echo() in run() is the backstop for
        whatever still gets through either way.
        """
        if _headphones_connected():
            return max(self.silence_rms_threshold * 2.0, self.silence_rms_threshold + 200), 3
        return max(self.silence_rms_threshold * 4.0, self.silence_rms_threshold + 800), 6

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
        # barge_in_threshold was derived from this at __init__ time, before
        # calibration — recompute it now, or it stays stuck at whatever the
        # pre-calibration config default (voice_silence_rms_threshold, 300)
        # produced (600) regardless of this room's actual noise floor. Seen
        # live: barge-in firing on ordinary ambient sound because its
        # threshold never moved off that stale 600 while the real silence
        # threshold calibrated to 2500+.
        self.barge_in_threshold, self.barge_in_sustain_chunks = self._compute_barge_in_params()
        headphones = _headphones_connected()
        print(
            f"  barge-in threshold set to {self.barge_in_threshold:.0f} "
            f"(sustain={self.barge_in_sustain_chunks} chunks, "
            f"output={'headphones' if headphones else 'speakers — extra margin applied'})"
        )

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

    def _watch_for_barge_in(self, stop_event: threading.Event, playback_active: threading.Event) -> None:
        grace_deadline = time.perf_counter() + self.barge_in_grace_seconds
        sustained = 0
        while not stop_event.is_set():
            try:
                frame = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._barge_frames_lock:
                self._barge_frames.append(frame)
            if not playback_active.is_set():
                # This thread starts before the response is ready (needed
                # so streaming Ollama playback can be barged in on the
                # instant it begins). For slow, non-streaming backends —
                # confirmed live with Sarvam's 10-30s+ reasoning-model
                # latency — "before playback" can be the entire turn. Any
                # speech during that dead air used to get misread as
                # barging in on a reply that hadn't made a sound yet,
                # silently skipping it before the user ever heard anything.
                sustained = 0
                continue
            if time.perf_counter() < grace_deadline:
                sustained = 0
                continue
            rms = self._rms(frame)
            if rms > self.barge_in_threshold:
                sustained += 1
                if sustained >= self.barge_in_sustain_chunks:
                    print(f"(Barge-in detected — stopping speech. RMS {rms:.0f} vs threshold {self.barge_in_threshold:.0f})")
                    stop_event.set()
                    return
            else:
                sustained = 0

    def record_utterance(self, max_seconds: float = MAX_UTTERANCE_SECONDS) -> tuple[np.ndarray, bool]:
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

        elapsed = chunks_used * FRAME_SAMPLES / SAMPLE_RATE
        print(
            f"  (debug: {elapsed:.1f}s recorded, peak RMS {max_rms:.0f}, "
            f"peak VAD prob {max_vad_prob:.2f}, heard_speech={heard_speech})"
        )
        recorded = np.concatenate(frames) if frames else np.array([], dtype=np.int16)
        return recorded, heard_speech

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        audio_float = audio.astype(np.float32) / 32768.0
        segments, _ = self.whisper.transcribe(audio_float, language="en", beam_size=1)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def speak(self, text: str, stop_event: threading.Event | None = None) -> bool:
        return cli.speak_response(text, self.cfg, stop_event=stop_event)

    def voice_confirm(self, prompt: str) -> bool:
        print(f"[Router] {prompt}")
        self.speak(prompt + " Say yes or no.")
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
        if self.conversation_active_until is not None and time.monotonic() < self.conversation_active_until:
            remaining = self.conversation_active_until - time.monotonic()
            listening_banner = f"(Listening for your follow-up — window open {remaining:.0f}s more...)"
        else:
            self.conversation_active_until = None
            self.listen_for_wake_word()
            listening_banner = "(Wake word detected — listening...)"
        turn_start = time.perf_counter()
        wake_detected_at = time.perf_counter()
        print(listening_banner)
        audio, _ = self.record_utterance()
        utterance_done_at = time.perf_counter()
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
        barge_monitor = None
        if self.cfg.get("tts_enabled"):
            # Re-checked every turn, not just at calibration — headphones
            # can get plugged in or removed mid-session, and the
            # threshold/sustain requirement should track that immediately.
            self.barge_in_threshold, self.barge_in_sustain_chunks = self._compute_barge_in_params()
            barge_monitor = threading.Thread(
                target=self._watch_for_barge_in,
                args=(barge_event, playback_active),
                daemon=True,
            )
            barge_monitor.start()

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
            try:
                wav_bytes = sarvam_client.text_to_speech(response, language_code=language_code)
                # Arm barge-in only once there's real audio, not
                # during the (here, up to ~30s) network call that
                # produces it.
                playback_active.set()
                interrupted = not cli.tts.play(wav_bytes, stop_event=barge_event)
            except sarvam_client.SarvamError as e:
                print(f"(Sarvam TTS unavailable: {e})")

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
        print(
            "  (timing: "
            f"wake={wake_detected_at - turn_start:.2f}s, "
            f"record={utterance_done_at - wake_detected_at:.2f}s, "
            f"stt={transcription_done_at - utterance_done_at:.2f}s, "
            f"llm={response_ready_at - transcription_done_at:.2f}s, "
            f"tts={spoken_done_at - response_ready_at:.2f}s)"
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
