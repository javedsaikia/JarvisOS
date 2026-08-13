"""Shell execution — 'carefully', per the mission spec. Not sandboxed: safety
comes from mandatory human confirmation before every run (enforced by the
caller in cli.py), not from technical restriction of what the command can do.
"""
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class ShellError(Exception):
    pass


def run(command: str, timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise ShellError(f"Command timed out after {timeout}s.") from e

    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return f"(exit code {result.returncode}) {output or '(no output)'}"
    return output or "(no output)"
