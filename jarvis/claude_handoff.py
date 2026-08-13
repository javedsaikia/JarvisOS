"""Hand off a task to Claude Code (non-interactive `-p` mode) when the router escalates."""
import os
import signal
import subprocess
import threading


class ClaudeCodeError(Exception):
    pass


class Cancelled(ClaudeCodeError):
    """Raised when cancel_current() killed the in-flight process — distinct
    from a real failure so callers can respond differently (e.g. skip
    speaking a stale result instead of reporting an error)."""


_lock = threading.Lock()
_current_proc: subprocess.Popen | None = None
_cancel_requested = threading.Event()


def cancel_current() -> bool:
    """Kills the in-flight Claude Code process, if any. Called by the voice
    loop's barge-in monitor when "stop" is detected while a task is still
    running — a plain subprocess.run() call can only be silenced once it
    finally finishes; this actually terminates it, since Claude Code can
    run for minutes on real agentic work. No-op (returns False) if nothing
    is currently running.

    Kills the whole process group, not just the immediate child — verified
    live that plain proc.terminate() isn't enough: if the child (or
    anything it launches, e.g. Claude Code invoking a tool) spawns its own
    subprocess inheriting the stdout/stderr pipes, killing only the direct
    child leaves that grandchild running as an orphan holding the pipes
    open, and communicate() in invoke() blocks until it exits on its own
    (measured: up to the full original timeout) rather than returning
    promptly. start_new_session=True in invoke() puts the whole tree in one
    killable process group.
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


def invoke(task: str, command: str = "claude", timeout: int = 600) -> str:
    global _current_proc
    _cancel_requested.clear()
    try:
        proc = subprocess.Popen(
            [command, "-p", task],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # own process group, see cancel_current()
        )
    except FileNotFoundError as e:
        raise ClaudeCodeError(f"`{command}` CLI not found on PATH.") from e

    with _lock:
        _current_proc = proc
    try:
        # communicate(), not wait() — unlike subprocess.run() (which this
        # replaced), this can be raced with cancel_current() terminating
        # the process from another thread: once the process actually dies
        # (for any reason), communicate() returns immediately rather than
        # blocking for the rest of `timeout`.
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise ClaudeCodeError(f"Claude Code timed out after {timeout}s.") from None
    finally:
        with _lock:
            if _current_proc is proc:
                _current_proc = None

    if _cancel_requested.is_set():
        raise Cancelled("Cancelled by user.")
    if proc.returncode != 0:
        raise ClaudeCodeError(stderr.strip() or "Claude Code exited with an error.")

    return stdout.strip()
