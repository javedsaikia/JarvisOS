"""Hand off a task to Claude Code (non-interactive `-p` mode) when the router escalates."""
import subprocess


class ClaudeCodeError(Exception):
    pass


def invoke(task: str, command: str = "claude", timeout: int = 600) -> str:
    try:
        result = subprocess.run(
            [command, "-p", task],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise ClaudeCodeError(f"`{command}` CLI not found on PATH.") from e
    except subprocess.TimeoutExpired as e:
        raise ClaudeCodeError(f"Claude Code timed out after {timeout}s.") from e

    if result.returncode != 0:
        raise ClaudeCodeError(result.stderr.strip() or "Claude Code exited with an error.")

    return result.stdout.strip()
