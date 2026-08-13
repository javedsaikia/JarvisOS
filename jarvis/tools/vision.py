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
