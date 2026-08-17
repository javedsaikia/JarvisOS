"""Local TTS via Piper (isolated in max/.venv — never touches system Python).
No cloud calls, no cost. Falls back silently disabled if Piper isn't set up.
"""
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

# Digital gain for Max's spoken voice (afplay -v), independent of the
# Mac's output volume. Piper WAVs are normalised hot; at 100% they are
# piercing on earphones even when the system volume looks modest.
DEFAULT_VOLUME = 35
MIN_VOLUME = 5
MAX_VOLUME = 100
VOLUME_STEP = 10

PROJECT_ROOT = Path(__file__).parent.parent
PIPER_BIN = PROJECT_ROOT / "max" / ".venv" / "bin" / "piper"

SENTENCE_END_RE = re.compile(r"(.+?[.!?][\"')\]]*)(?:\s+|\n+)", re.S)


class TTSError(Exception):
    pass


def split_ready_sentences(buffer: str) -> tuple[list[str], str]:
    """Pull complete sentences off the front of a streamed-text buffer.

    Shared by any caller that feeds an LLM's streaming output into TTS
    sentence-by-sentence (voice_loop's local playback, the web bridge's
    websocket streaming) so both synthesize as soon as a sentence is ready
    instead of waiting for the full reply. Returns (sentences, remainder) —
    remainder is the trailing partial sentence still waiting on more text.
    """
    sentences = []
    while True:
        match = SENTENCE_END_RE.search(buffer)
        if not match:
            return sentences, buffer
        sentence = match.group(1).strip()
        buffer = buffer[match.end():]
        if sentence:
            sentences.append(sentence)


def _resolve_model(model_path: str) -> Path:
    model = Path(model_path)
    if not model.is_absolute():
        model = PROJECT_ROOT / model_path
    if not model.exists():
        raise TTSError(f"Voice model not found at {model}")
    return model


def synthesize(text: str, model_path: str, timeout: int = 60) -> bytes:
    """Run Piper and return the synthesized WAV bytes — no playback.

    Split out from speak() so callers that need the raw audio (e.g. the
    web UI bridge, which forwards it to the browser for playback +
    real-time analysis) don't have to also play it server-side via afplay.
    """
    text = text.strip()
    if not text:
        return b""

    if not PIPER_BIN.exists():
        raise TTSError(
            f"Piper not found at {PIPER_BIN}. Set up with: "
            f"python3 -m venv max/.venv && max/.venv/bin/pip install piper-tts"
        )

    model = _resolve_model(model_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        synth = subprocess.run(
            [str(PIPER_BIN), "-m", str(model), "-f", wav_path],
            input=text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if synth.returncode != 0:
            raise TTSError(synth.stderr.strip() or "piper synthesis failed")
        return Path(wav_path).read_bytes()
    finally:
        Path(wav_path).unlink(missing_ok=True)


def get_volume() -> int:
    """Current spoken-voice level, 5–100. Reads the live config each time
    so a key press or HUD slider applies to the next sentence."""
    from max.config import load_config

    try:
        raw = load_config().get("tts_volume", DEFAULT_VOLUME)
        return max(MIN_VOLUME, min(MAX_VOLUME, int(raw)))
    except (TypeError, ValueError):
        return DEFAULT_VOLUME


def set_volume(level: int) -> int:
    from max.config import update_config

    try:
        level = int(level)
    except (TypeError, ValueError):
        level = DEFAULT_VOLUME
    level = max(MIN_VOLUME, min(MAX_VOLUME, level))
    update_config(tts_volume=level)
    try:
        from max import voice_events
        voice_events.tts_volume(level)
    except Exception:
        pass
    return level


def adjust_volume(delta: int) -> int:
    return set_volume(get_volume() + int(delta))


def _afplay_gain() -> float:
    return get_volume() / 100.0


def play(wav_bytes: bytes, timeout: int = 60, stop_event: threading.Event | None = None) -> bool:
    """Play WAV bytes on this machine's speakers via afplay.

    Returns True when playback completed normally. Returns False if a caller
    asked us to stop early via stop_event.
    """
    if not wav_bytes:
        return True
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        wav_path = f.name
    try:
        proc = subprocess.Popen(["afplay", "-v", f"{_afplay_gain():.3f}", wav_path])
        deadline = time.monotonic() + timeout
        while True:
            if stop_event is not None and stop_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1)
                return False
            if proc.poll() is not None:
                return True
            if time.monotonic() >= deadline:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1)
                return False
            time.sleep(0.05)
    finally:
        Path(wav_path).unlink(missing_ok=True)


def speak(text: str, model_path: str, timeout: int = 60, stop_event: threading.Event | None = None) -> bool:
    if stop_event is not None and stop_event.is_set():
        return False
    wav_bytes = synthesize(text, model_path, timeout)
    if stop_event is not None and stop_event.is_set():
        return False
    return play(wav_bytes, timeout, stop_event=stop_event)
