"""Shell execution — 'carefully', per the mission spec. Not sandboxed: safety
comes from mandatory human confirmation before every run (enforced by the
caller in cli.py), not from technical restriction of what the command can do.
"""
import os
import signal
import subprocess
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class ShellError(Exception):
    pass


class Cancelled(ShellError):
    """Raised when cancel_current() killed the in-flight command."""


_lock = threading.Lock()
_current_proc: subprocess.Popen | None = None
_cancel_requested = threading.Event()


def cancel_current() -> bool:
    """Kills the in-flight shell command, if any — same barge-in-triggered
    mechanism as claude_handoff.cancel_current(). Kills the whole process
    group, not just the immediate child: run() uses shell=True, so the
    actual command is a grandchild of this Popen (child of the /bin/sh
    wrapper); terminating just the Popen's own pid would leave the real
    command running as an orphan.
    """
    with _lock:
        proc = _current_proc
    if proc is None or proc.poll() is not None:
        return False
    _cancel_requested.set()
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=3)
    return True


def run(command: str, timeout: int = 30) -> str:
    global _current_proc
    _cancel_requested.clear()
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # own process group, see cancel_current()
    )
    with _lock:
        _current_proc = proc
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise ShellError(f"Command timed out after {timeout}s.") from None
    finally:
        with _lock:
            if _current_proc is proc:
                _current_proc = None

    if _cancel_requested.is_set():
        raise Cancelled("Cancelled by user.")

    output = (stdout + stderr).strip()
    if proc.returncode != 0:
        return f"(exit code {proc.returncode}) {output or '(no output)'}"
    return output or "(no output)"
