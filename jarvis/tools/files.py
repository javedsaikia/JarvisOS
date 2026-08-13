"""File read/write scoped to the JarvisOS project directory only. Path
traversal outside the project root is rejected — this is the 'file
read/write in project' tool from the mission spec, deliberately not a
general-purpose filesystem tool.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class FileToolError(Exception):
    pass


def _resolve(path: str) -> Path:
    raw = Path(path)
    candidate = (raw if raw.is_absolute() else PROJECT_ROOT / raw).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        raise FileToolError(f"'{path}' is outside the project directory — refusing.")
    return candidate


def read_file(path: str, max_chars: int = 8000) -> str:
    target = _resolve(path)
    if not target.exists():
        raise FileToolError(f"{path} does not exist.")
    if not target.is_file():
        raise FileToolError(f"{path} is not a file.")
    content = target.read_text(errors="replace")
    if len(content) > max_chars:
        return content[:max_chars] + f"\n... (truncated, {len(content)} chars total)"
    return content


def write_file(path: str, content: str) -> str:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} chars to {path}."


def list_dir(path: str = ".") -> str:
    target = _resolve(path)
    if not target.exists():
        raise FileToolError(f"{path} does not exist.")
    if not target.is_dir():
        raise FileToolError(f"{path} is not a directory.")

    skip = {".venv", "__pycache__", ".git"}
    entries = sorted(e for e in target.iterdir() if e.name not in skip)
    if not entries:
        return "(empty)"
    return "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in entries)
