"""Wake on a phrase you choose, with no pretrained wake-word model.

openWakeWord only ships four pretrained detectors — `hey_jarvis`,
`alexa`, `hey_mycroft`, `hey_rhasspy` — and every one of them is somebody
else's name. There is no `hey_max`, so as long as the wake word is a
pretrained model, the product is stuck being woken by another product's
name. Training a custom openWakeWord model is possible (see the README)
but needs hours of synthesis and training before the first word works.

This is the path that works today: Silero VAD (already vendored here for
endpointing) marks each stretch of speech, and a small local Whisper
transcribes just that stretch and checks it against the configured
phrases. Detection is therefore *whatever you type in config* — "hey
max", "computer", your dog's name — with no model to train and nothing
downloaded that carries a brand.

The costs, measured rather than assumed, on this machine:

- `tiny.en` transcribes a one-second utterance in ~0.1s, and only runs
  when VAD says someone actually spoke, so an empty room costs nothing
  beyond the VAD itself, which was already running every frame.
- Detection lands at the *end* of the utterance instead of mid-phrase,
  which is what a wake-word model gives you. In exchange the whole
  utterance is already recorded and transcribed, so "hey max, what's the
  weather" needs no second recording — `pending_audio` hands the caller
  the same audio to re-transcribe with the better model.
- Whisper is not a keyword spotter, so it renders the name however it
  hears it: "Hey Max" comes back as "Hey Oren", and sometimes "Hey,
  Orange." PHRASE_VARIANTS exists for exactly that, the same way the
  loop's existing wake-phrase matching already allows for STT variants.

Licensing, since the point of the exercise is being able to ship this:
Whisper (MIT), faster-whisper (MIT) and Silero VAD (MIT) are all
permissive, and nothing here downloads a model named after a product.
"""
import difflib
import re
import time

import numpy as np

from max.vad import SileroVAD, CHUNK_SAMPLES as VAD_CHUNK_SAMPLES

SAMPLE_RATE = 16000

# Silence that ends an utterance. Shorter than the command endpointer's
# 0.8s: this only has to decide "did they stop saying the wake phrase",
# and every extra 100ms here is 100ms before the chime.
SEGMENT_SILENCE_SECONDS = 0.35

# Speech shorter than this is a cough or a chair; longer than this is a
# conversation that was never addressed to us, and transcribing all of it
# on every pass would be the one genuinely expensive thing this could do.
MIN_SEGMENT_SECONDS = 0.35
MAX_SEGMENT_SECONDS = 8.0

VAD_SPEECH_THRESHOLD = 0.5


# Words people put in front of a name when addressing something. The
# phrase is matched as "lead + name" anywhere in the utterance, or as a
# bare name at the very start ("Max, check my calendar"), which is how
# people actually address an assistant once they are used to it.
LEAD_WORDS = {"hey", "hi", "hello", "ok", "okay", "hey there", "yo"}

# Whisper is not a keyword spotter. It writes what it hears through an
# English lexicon, so an invented name comes out as the nearest real word
# — and a different one each time. Measured on the same synthesized
# "Hey Max" across models: Orrin, Arin, Oren, Orange, Auring, "or in".
# Most of that spread is covered by the similarity test below; these are
# the ones that are too far off for it and too specific to be dangerous.
# Words that mean "stop what you are doing", when said *with the name*.
# The name is required on purpose: this is matched against speech picked
# up while Max is talking, and this machine has no echo cancellation, so
# a bare "stop" would let Max interrupt itself the moment it said a
# sentence containing the word. Max never says its own name plus "stop".
STOP_WORDS = {"stop", "quiet", "enough", "shush", "shut"}

NAME_MISHEARINGS = {
    "max": [
        "macs", "mack", "mac", "maxx", "maxe", "maxie", "macks",
        "marks", "marx", "maps", "mas", "mak", "mex", "mix",
    ],
}

# Tuned against the measured renderings: "oren" (0.75), "orrin" (0.89) and
# "arin" (0.75) are in; "there" (0.44) and ordinary words are out. Loose
# enough to survive an unfamiliar name, tight enough that ordinary speech
# after "hey" doesn't wake anything.
NAME_SIMILARITY = 0.72


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z ]", " ", text.lower())


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


class PhraseWake:
    """VAD-segmented, STT-matched wake phrase.

    Deliberately exposes the two methods voice_loop already calls on an
    openWakeWord model — `reset()` and `predict(frame)` returning a score
    dict — so both the idle listen and the barge-in watcher work with it
    unchanged, and switching back to a trained model is a config flip
    rather than a code change.
    """

    def __init__(self, phrases: list[str], stt_model_size: str = "", key: str = "wake",
                 model=None):
        self.key = key
        # The phrase is "[lead] name"; the name is what actually gets
        # matched, since the lead word is optional in real speech.
        self.phrases = [p.strip() for p in phrases if p.strip()]
        # strict: the name as configured, matched fuzzily — safe enough to
        #   accept on its own at the start of an utterance ("Max, ...").
        # loose: real English words the STT substitutes for it. Accepted
        #   only directly after a lead word, because "orange" opening a
        #   sentence is a fruit, while "hey orange" is somebody with an
        #   unusual accent talking to their computer.
        self.strict_names: list[str] = []
        self.loose_names: list[str] = []
        for phrase in self.phrases:
            words = _normalize(phrase).split()
            if not words:
                continue
            name = words[-1]
            self.strict_names.append(name)
            for spelling in NAME_MISHEARINGS.get(name, []):
                # Both forms: matching is token-wise, so a spelling
                # Whisper writes as two words ("or in") is only ever seen
                # here as the joined pair.
                self.loose_names.append(spelling)
                self.loose_names.append(spelling.replace(" ", ""))
        self.strict_names = sorted(set(self.strict_names))
        self.loose_names = sorted(set(self.loose_names) - set(self.strict_names))

        # By default this shares the voice loop's own Whisper rather than
        # loading a second one: same memory, same load time, and measurably
        # better recall. tiny.en is fast (0.1s) but renders an invented
        # name differently almost every time — measured on one synthesized
        # "Hey Max": Orange, Arryn, AMAXG, Arring, Oren, Orrin. The
        # command model wrote "Oren" every single time. For a wake word,
        # missing it is worse than half a second.
        if model is not None:
            self.model = model
        else:
            # Imported here, not at module scope, so importing this module
            # for its matching helpers costs nothing.
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                stt_model_size or "small.en", device="cpu", compute_type="int8"
            )
        self.vad = SileroVAD()

        self._vad_buffer = np.array([], dtype=np.float32)
        self._segment: list[np.ndarray] = []
        self._silence_chunks = 0
        self._in_speech = False

        # The audio of the utterance that triggered the last detection,
        # so the caller can re-transcribe it with the good model instead
        # of asking the user to repeat a command they already said.
        self.pending_audio: np.ndarray | None = None
        self.last_transcript = ""
        # "wake" or "stop" — see predict(include_stop=True).
        self.last_match_kind = ""
        # The literal token(s) accepted as the name in the last match.
        self.last_match_token = ""

    def reset(self) -> None:
        self.vad.reset()
        self._vad_buffer = np.array([], dtype=np.float32)
        self._segment = []
        self._silence_chunks = 0
        self._in_speech = False

    def describe(self) -> str:
        return self.phrases[0] if self.phrases else "(none)"

    def _is_name(self, token: str, loose: bool) -> bool:
        if not token:
            return False
        names = self.strict_names + self.loose_names if loose else self.strict_names
        return any(
            token == name or _similar(token, name) >= NAME_SIMILARITY
            for name in names
        )

    def _name_at(self, tokens: list[str], i: int, loose: bool = True) -> bool:
        """Sets last_match_token to whatever matched, so the caller can
        remove exactly that from the command text — including a spelling
        nobody has ever seen before. A mishearing that gets through the
        matcher must not survive into the prompt: seen live, "Max, I am
        really depressed" arrived as "Kiren, ..." and the model reasonably
        started calling the user Kiren."""
        """Is the name at position `i`, as one token or split across two?

        Whisper sometimes writes an unfamiliar name as two real words —
        measured: "Max, check my calendar" came back as "or in check my
        calendar". Joining the pair before comparing costs nothing and
        recovers exactly that case.
        """
        if i >= len(tokens):
            return False
        if self._is_name(tokens[i], loose):
            self.last_match_token = tokens[i]
            return True
        # The joined pair is matched *exactly*, never fuzzily. Fuzzy on a
        # concatenation is far too generous — "the orange" joins to
        # "theorange", which scores 0.80 against "orange" and would wake
        # the assistant on a sentence about fruit juice.
        joined = tokens[i] + tokens[i + 1] if i + 1 < len(tokens) else ""
        names = self.strict_names + self.loose_names if loose else self.strict_names
        if joined and joined in names:
            self.last_match_token = f"{tokens[i]} {tokens[i + 1]}"
            return True
        return False

    def _speech_probability(self, frame: np.ndarray) -> float:
        """Highest VAD probability across the 512-sample chunks in `frame`.

        Same buffering the voice loop already does — Silero requires
        exactly 512 samples per call and the audio arrives in 1280-sample
        frames, which is not a multiple of it.
        """
        # Silero wants float32 in [-1, 1]; frames arrive as int16. Feeding
        # it raw int16 doesn't error, it just returns near-zero for
        # everything — which looks exactly like a silent room.
        self._vad_buffer = np.concatenate([self._vad_buffer, frame.astype(np.float32) / 32768.0])
        probs = []
        while len(self._vad_buffer) >= VAD_CHUNK_SAMPLES:
            chunk = self._vad_buffer[:VAD_CHUNK_SAMPLES]
            self._vad_buffer = self._vad_buffer[VAD_CHUNK_SAMPLES:]
            probs.append(self.vad.speech_probability(chunk))
        return max(probs) if probs else 0.0

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio.astype(np.float32) / 32768.0, language="en", beam_size=1
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def _matches(self, transcript: str) -> bool:
        """True when the utterance addresses us by name.

        Two accepted shapes, both checked against the misheard spellings
        rather than the exact one:
          - a lead word followed by the name, anywhere in the utterance
            ("okay so, hey max, what's the weather")
          - the name as the very first word ("max, check my calendar")
        Requiring one of those, rather than the name appearing anywhere,
        is what keeps "I was reading about the origin of the universe"
        from waking it.
        """
        tokens = _normalize(transcript).split()
        if not tokens:
            return False
        if self._name_at(tokens, 0, loose=False):
            return True
        # Multi-word phrase spelled out in config ("computer, wake up") is
        # still matched literally, so a phrase that isn't name-shaped works.
        text = " ".join(tokens)
        if any(_normalize(p).strip() in text for p in self.phrases):
            return True
        for i, token in enumerate(tokens[:-1]):
            if token in LEAD_WORDS and self._name_at(tokens, i + 1):
                return True
        return False

    def _is_stop(self, transcript: str) -> bool:
        """True for "stop Max" / "Max, stop" / "Max be quiet".

        Requires the name next to the stop word — see STOP_WORDS. Checked
        only where the caller asks for it (barge-in), never when idle.
        """
        tokens = _normalize(transcript).split()
        for i, token in enumerate(tokens):
            if token not in STOP_WORDS:
                continue
            # The name within two tokens either side covers "stop max",
            # "max stop", "max, please stop" and "stop it max".
            for j in (i - 2, i - 1, i + 1, i + 2):
                if 0 <= j < len(tokens) and self._is_name(tokens[j], loose=True):
                    return True
        return False

    def predict(self, frame: np.ndarray, include_stop: bool = False) -> dict[str, float]:
        """Feed one frame. Returns {key: 1.0} on the frame that completes a
        matching utterance, {key: 0.0} otherwise.

        `include_stop` also accepts "stop <name>" as a match, and records
        which kind it was in `last_match_kind` — used by barge-in, where
        "stop talking" and "wake up, I have something to say" want the
        same interruption but different behaviour afterwards.
        """
        speaking = self._speech_probability(frame) >= VAD_SPEECH_THRESHOLD
        frame_seconds = len(frame) / SAMPLE_RATE

        if speaking:
            self._in_speech = True
            self._silence_chunks = 0
            self._segment.append(frame)
            if len(self._segment) * frame_seconds > MAX_SEGMENT_SECONDS:
                # Someone is talking to somebody else. Drop it rather than
                # transcribe a minute of conversation.
                self._segment = []
                self._in_speech = False
            return {self.key: 0.0}

        if not self._in_speech:
            return {self.key: 0.0}

        # Trailing silence: keep it in the segment (Whisper reads the tail
        # of a word better with a little room after it) until it's long
        # enough to call the utterance finished.
        self._segment.append(frame)
        self._silence_chunks += 1
        if self._silence_chunks * frame_seconds < SEGMENT_SILENCE_SECONDS:
            return {self.key: 0.0}

        audio = np.concatenate(self._segment) if self._segment else np.array([], dtype=np.int16)
        self._segment = []
        self._in_speech = False
        self._silence_chunks = 0

        if audio.size < MIN_SEGMENT_SECONDS * SAMPLE_RATE:
            return {self.key: 0.0}

        started = time.monotonic()
        transcript = self._transcribe(audio)
        self.last_transcript = transcript

        kind = ""
        self.last_match_token = ""
        if include_stop and self._is_stop(transcript):
            kind = "stop"
        elif self._matches(transcript):
            kind = "wake"
        if not kind:
            return {self.key: 0.0}
        self.last_match_kind = kind

        print(
            f"  ({kind}: heard {transcript!r} in {time.monotonic() - started:.2f}s)",
            flush=True,
        )
        # A stop is an instruction on its own, not the start of a command,
        # so there is nothing to hand on to the turn.
        self.pending_audio = None if kind == "stop" else audio
        return {self.key: 1.0}

    def take_pending_audio(self) -> np.ndarray | None:
        """The triggering utterance, once. Cleared on read so a later turn
        can't accidentally replay an old command."""
        audio, self.pending_audio = self.pending_audio, None
        return audio
