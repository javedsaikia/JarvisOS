"""Run AppleScript via osascript. Shared helper for Calendar/Notes tools."""
import subprocess
import tempfile
import time
from pathlib import Path


class AppleScriptError(Exception):
    pass


# Seen live: "Calendar got an error: Application isn't running. (-600)" when
# Calendar.app isn't open. Without this, that raw AppleEvent error text gets
# wrapped as the tool's "answer" and read aloud verbatim over TTS — a
# confusing thing to hear. `tell application "X"` is supposed to auto-launch
# the app, but that doesn't always happen reliably, so when the app_name of
# the target app is known, launch it explicitly and retry once instead of
# surfacing the raw error.
_APP_NOT_RUNNING_MARKERS = ("isn't running", "is not running", "(-600)")


def _is_not_running_error(message: str) -> bool:
    return any(marker in message for marker in _APP_NOT_RUNNING_MARKERS)


def _launch_and_wait(app_name: str, timeout: int = 15) -> None:
    subprocess.run(["open", "-a", app_name], capture_output=True, text=True, timeout=timeout)
    time.sleep(2)  # give it a moment to finish launching before the retry


def _run_once(script: str, timeout: int) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".scpt", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        result = subprocess.run(
            ["osascript", path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        Path(path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise AppleScriptError(result.stderr.strip() or "osascript failed")
    return result.stdout.strip()


def run(script: str, timeout: int = 20, app_name: str = None) -> str:
    try:
        return _run_once(script, timeout)
    except AppleScriptError as e:
        if app_name and _is_not_running_error(str(e)):
            _launch_and_wait(app_name)
            return _run_once(script, timeout)
        raise


def as_string(s: str) -> str:
    """Embed a Python string as a safe AppleScript string expression.

    Escapes quotes/backslashes and splices real newlines together with the
    `return` constant, since AppleScript string literals can't contain a
    literal line break.
    """
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    lines = escaped.split("\n")
    return " & return & ".join(f'"{line}"' for line in lines)
