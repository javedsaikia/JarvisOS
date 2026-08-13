"""File access for JARVIS. Write stays scoped to the JarvisOS project
directory only (path traversal outside it is rejected) — this is the
'file read/write in project' tool from the mission spec. Read and search
are scoped wider, to the project plus the user's common personal folders
(Desktop, Documents, Downloads) — an explicit, deliberate choice (not the
whole home directory) so JARVIS can actually answer questions about real
personal files without reaching into ~/.ssh, ~/Library (app data, browser
profiles, anything keychain-adjacent), or other apps' private storage.
"""
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

READ_ONLY_ROOTS = [
    PROJECT_ROOT,
    Path.home() / "Desktop",
    Path.home() / "Documents",
    Path.home() / "Downloads",
]


class FileToolError(Exception):
    pass


def _resolve_write(path: str) -> Path:
    raw = Path(path)
    candidate = (raw if raw.is_absolute() else PROJECT_ROOT / raw).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        raise FileToolError(f"'{path}' is outside the project directory — refusing to write there.")
    return candidate


def _resolve_read(path: str) -> Path:
    raw = Path(path).expanduser()
    candidate = (raw if raw.is_absolute() else PROJECT_ROOT / raw).resolve()
    for root in READ_ONLY_ROOTS:
        try:
            candidate.relative_to(root.resolve())
            return candidate
        except ValueError:
            continue
    raise FileToolError(
        f"'{path}' is outside the folders JARVIS can read "
        "(the project, Desktop, Documents, Downloads) — refusing."
    )


def read_file(path: str, max_chars: int = 8000) -> str:
    target = _resolve_read(path)
    if not target.exists():
        raise FileToolError(f"{path} does not exist.")
    if not target.is_file():
        raise FileToolError(f"{path} is not a file.")
    content = target.read_text(errors="replace")
    if len(content) > max_chars:
        return content[:max_chars] + f"\n... (truncated, {len(content)} chars total)"
    return content


def write_file(path: str, content: str) -> str:
    target = _resolve_write(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} chars to {path}."


def list_dir(path: str = ".") -> str:
    target = _resolve_read(path)
    if not target.exists():
        raise FileToolError(f"{path} does not exist.")
    if not target.is_dir():
        raise FileToolError(f"{path} is not a directory.")

    skip = {".venv", "__pycache__", ".git"}
    entries = sorted(e for e in target.iterdir() if e.name not in skip)
    if not entries:
        return "(empty)"
    return "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in entries)


_SEARCH_EXCLUDE_SEGMENTS = ("/node_modules/", "/.venv/", "/__pycache__/", "/.git/", "/dist/")


def search_files(query: str, max_results: int = 15) -> str:
    """Spotlight search (mdfind) across the same allowed read folders —
    matches filenames and, for indexed document types, file contents too.
    AppleScript's own Finder search is much slower for this; mdfind is the
    native CLI for exactly this job on macOS. Dependency/build directories
    are filtered out — verified live, a generic query like "pdf" otherwise
    drowns real personal files in hundreds of node_modules source files
    that just happen to mention the word.
    """
    results: list[str] = []
    seen: set[str] = set()
    for root in READ_ONLY_ROOTS:
        if not root.exists():
            continue
        try:
            proc = subprocess.run(
                ["mdfind", "-onlyin", str(root), query],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line or line in seen:
                continue
            if any(segment in line for segment in _SEARCH_EXCLUDE_SEGMENTS):
                continue
            seen.add(line)
            results.append(line)

    if not results:
        return f'No files found matching "{query}".'
    return "\n".join(results[:max_results])
