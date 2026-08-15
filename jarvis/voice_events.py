"""Real-time speech telemetry from the voice loop to the web UI.

The web UI's orb reacts to real audio energy, but only ever saw the audio
it played itself. Voice-loop replies never reach the browser at all: they
are synthesized and handed to `afplay`, which plays them on the Mac's
speakers (see tts.play). So during an actual spoken conversation — the
main way this thing is used — the orb had no signal and sat still while
Orin was talking.

Rather than inventing motion for those turns, this publishes the real
thing: the amplitude envelope measured from the exact WAV bytes about to
be played, plus the moment playback starts. The browser replays that
envelope against its own clock, so the orb moves with the actual syllables
of the actual sentence, not a synthetic pulse.

Transport is a JSONL file the bridge tails, matching how voice-loop
transcript turns already reach the UI (bridge._transcript_poll_loop):
voice_loop.py is a separate OS process that must keep working with no
bridge running at all, so it publishes to a file and nothing here ever
depends on anyone reading it. That property is why this is also the
channel for screen-capture phases (see screen_capture_phase) — a spoken
"what's on my screen?" is handled entirely inside voice_loop.py, which
has no connection to the browser at all.
"""
import io
import json
import time
import wave
from pathlib import Path

import numpy as np

EVENTS_PATH = Path("/tmp/jarvis_voice_events.jsonl")

# Envelope sample rate. 25Hz (one value per 40ms) is well past what the
# eye resolves in a glowing orb and keeps a 10s reply to 250 floats.
ENVELOPE_HZ = 25

# Truncate rather than grow without bound — this is a live telemetry
# stream, not a log; anything already read is worthless. The bridge
# detects the file shrinking and resets its own read offset.
MAX_FILE_BYTES = 256 * 1024


def _envelope(wav_bytes: bytes) -> tuple[list[float], float]:
    """RMS amplitude envelope of a WAV, normalized to roughly [0, 1].

    Returns (envelope, duration_seconds). Empty envelope for anything
    unreadable — callers treat that as "no telemetry", never as an error.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())

    if width != 2 or rate <= 0:
        # Both Piper and Sarvam return 16-bit PCM; anything else is
        # unexpected enough that guessing at the format is worse than
        # sending nothing.
        return [], 0.0

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if samples.size == 0:
        return [], 0.0

    duration = samples.size / rate
    window = max(1, rate // ENVELOPE_HZ)
    count = samples.size // window
    if count == 0:
        return [], duration

    blocks = samples[: count * window].reshape(count, window) / 32768.0
    rms = np.sqrt((blocks ** 2).mean(axis=1))
    # Speech RMS sits around 0.05-0.2 even at a healthy level, which would
    # barely move anything on a 0-1 scale. Same shaping the browser-side
    # analyser already applies to its own signal (see audio.ts): gain,
    # then a curve that opens up the quiet end without clipping the loud.
    # Gain measured against real Piper output rather than guessed — 2.2
    # puts a normal sentence at mean 0.46 / p90 0.76 with nothing clipped,
    # so loud syllables still have somewhere to go. (3.4 flattened 10% of
    # the clip against the ceiling, losing the peaks that read as stress.)
    level = np.clip(np.power(np.clip(rms * 2.2, 0, 1), 0.6), 0, 1)
    return [round(float(v), 3) for v in level], duration


def _write(event: dict) -> None:
    """Best-effort append. Telemetry must never break speech, so every
    failure here is swallowed — a missing frame costs an orb animation."""
    try:
        event["at"] = time.time()
        if EVENTS_PATH.exists() and EVENTS_PATH.stat().st_size > MAX_FILE_BYTES:
            EVENTS_PATH.write_text("")
        with EVENTS_PATH.open("a") as f:
            f.write(json.dumps(event) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def speaking(wav_bytes: bytes) -> None:
    """Announce that playback of `wav_bytes` is starting now."""
    if not wav_bytes:
        return
    try:
        envelope, duration = _envelope(wav_bytes)
    except (wave.Error, EOFError, ValueError):
        return
    if not envelope:
        return
    _write({
        "type": "speaking",
        "envelope": envelope,
        "hz": ENVELOPE_HZ,
        "duration": round(duration, 3),
    })


def screen_capture_phase(phase: str) -> None:
    """Tell the web UI to get out of the way of a screenshot, and later
    that it can come back. `phase` is "start" or "end".

    Only used on the fallback capture path: the native path composites the
    screen without the HUD window (see jarvis/screen_capture.py), so there
    is nothing to hide and the UI never flickers. A browser page cannot
    minimise its own window, so "hide" here means the HUD blanks itself —
    enough to keep a wall of glowing orb out of the screenshot.
    """
    _write({"type": "screen_capture", "phase": phase})


def mic_state(muted: bool) -> None:
    """Report that the microphone was actually released or reopened.

    Sent by the voice loop rather than the button that asked for it: the
    UI should show "mic off" when the device is really closed, not when a
    request was sent. For a privacy control, the difference matters.
    """
    _write({"type": "mic_state", "muted": bool(muted)})


def speaking_stopped() -> None:
    """Playback ended early (barge-in, stop, or a failed clip). The UI
    holds the last envelope until it runs out, so an interruption has to
    be said out loud or the orb keeps mouthing a sentence that stopped."""
    _write({"type": "speaking_end"})
