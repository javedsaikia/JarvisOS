"""Screen understanding: a real screenshot + a local vision model, no
cloud calls. `screencapture` is macOS's built-in screenshot CLI — same
"native OS tool over a library" choice already made for AppleScript
(Calendar/Notes) and mdfind (file search). The description comes from
Ollama's own /api/chat with an `images` field on the message; that's
already a plain pass-through in jarvis/ollama_client.py, so no changes
were needed there for a vision model to work.
"""
import base64
import json
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_QUESTION = "Describe what's on the screen in 2-4 clear, concise sentences."


class VisionError(Exception):
    pass


def capture_screenshot() -> bytes:
    """Full-screen capture. `-x` suppresses the camera-shutter sound —
    JARVIS taking a screenshot on request shouldn't announce itself
    audibly every time, same reasoning as calibrate() staying quiet.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = Path(f.name)
    try:
        result = subprocess.run(
            ["screencapture", "-x", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not path.exists():
            stderr = result.stderr.strip()
            if "could not create image from display" in stderr.lower():
                raise VisionError(
                    "macOS is refusing to let JARVIS take a screenshot — this app needs "
                    "\"Screen Recording\" permission. Grant it once in System Settings > "
                    "Privacy & Security > Screen Recording, then restart the voice loop/bridge."
                )
            raise VisionError(f"screencapture failed: {stderr or 'unknown error'}")
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def capture_screen_jpeg(max_width: int = 640) -> bytes:
    """Small JPEG of the screen, for the web UI's live screen card.

    Same `screencapture` as capture_screenshot, but straight to JPEG and
    downscaled with `sips` (macOS's own image tool — no Pillow dependency,
    matching the "native OS tool over a library" choice this file already
    makes). A full-resolution Retina PNG is several megabytes; this is
    tens of kilobytes, which is what makes it sane to push over the
    WebSocket every couple of seconds.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = Path(f.name)
    try:
        result = subprocess.run(
            ["screencapture", "-x", "-t", "jpg", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not path.exists():
            stderr = result.stderr.strip()
            if "could not create image from display" in stderr.lower():
                raise VisionError(
                    "macOS is refusing to let JARVIS capture the screen — this app needs "
                    "\"Screen Recording\" permission in System Settings > Privacy & "
                    "Security > Screen Recording."
                )
            raise VisionError(f"screencapture failed: {stderr or 'unknown error'}")
        # -Z fits the longest edge, preserving aspect ratio. A failure here
        # is not fatal: the full-size capture is still a valid JPEG, just
        # a heavier one.
        subprocess.run(
            ["sips", "-Z", str(max_width), "-s", "formatOptions", "60", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def describe_screen(question: str | None, cfg: dict) -> str:
    """Captures the screen and asks a local Ollama vision model about it.
    `question` is the user's own phrasing when there is one (e.g. "what's
    the error message on screen") so the model answers the actual thing
    asked, not just a generic description.
    """
    png_bytes = capture_screenshot()
    b64_image = base64.b64encode(png_bytes).decode("ascii")
    prompt = question.strip() if question and question.strip() else DEFAULT_QUESTION

    model = cfg.get("vision_ollama_model", "moondream")
    host = cfg["ollama_host"]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [b64_image]}],
        "stream": False,
    }
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise VisionError(f"Could not reach Ollama at {host} ({e}). Is `ollama serve` running?") from e

    content = data.get("message", {}).get("content", "").strip()
    if not content:
        raise VisionError(
            f"The vision model ({model}) returned nothing — is it pulled? "
            f"Run `ollama pull {model}`."
        )
    return content
