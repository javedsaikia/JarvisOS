"""File access for Max. Write stays scoped to the Max project
directory only (path traversal outside it is rejected) — this is the
'file read/write in project' tool from the mission spec. Read and search
are scoped wider, to the project plus the user's common personal folders
(Desktop, Documents, Downloads) — an explicit, deliberate choice (not the
whole home directory) so Max can actually answer questions about real
personal files without reaching into ~/.ssh, ~/Library (app data, browser
profiles, anything keychain-adjacent), or other apps' private storage.
"""
import os
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


# "Desktop", "/Desktop", "~/Desktop" and "Documents/foo.pdf" all mean the
# one in the user's home folder. Models reliably emit the leading-slash
# form — measured, qwen2.5-coder answers "list the files on my Desktop"
# with {"path": "/Desktop"} — which resolves to a root directory that does
# not exist, gets refused as outside the allowed roots, and the model then
# tells the user their Desktop is off limits. It is not; the path was.
_HOME_FOLDERS = ("Desktop", "Documents", "Downloads")


def _rewrite_home_folder(path: str) -> str:
    stripped = path.strip().lstrip("/")
    if not stripped:
        return path
    first, _, rest = stripped.partition("/")
    for folder in _HOME_FOLDERS:
        if first.lower() == folder.lower():
            return str(Path.home() / folder / rest) if rest else str(Path.home() / folder)
    return path


def _resolve_read(path: str) -> Path:
    raw = Path(_rewrite_home_folder(path)).expanduser()
    candidate = (raw if raw.is_absolute() else PROJECT_ROOT / raw).resolve()
    for root in READ_ONLY_ROOTS:
        try:
            candidate.relative_to(root.resolve())
            return candidate
        except ValueError:
            continue
    raise FileToolError(
        f"'{path}' is outside the folders Max can read "
        "(the project, Desktop, Documents, Downloads) — refusing."
    )


def find_named(name: str, limit: int = 3) -> list[str]:
    """Locate a file by its basename under the readable folders.

    Used by screen-teacher: a window titled "cli.py — MaxOS" has to
    become an actual path before the file can be read.
    """
    needle = Path(name).name.strip()
    if not needle:
        return []
    exact: list[str] = []
    partial: list[str] = []
    seen: set[str] = set()
    lowered = needle.lower()
    for path in _walk_filename_matches(lowered, limit * 8):
        resolved = str(Path(path).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        base = Path(path).name.lower()
        if base == lowered:
            exact.append(resolved)
        elif lowered in base:
            partial.append(resolved)
        if len(exact) >= limit:
            return exact[:limit]
    return (exact + partial)[:limit]


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


_SEARCH_EXCLUDE_SEGMENTS = (
    "/node_modules/", "/.venv/", "/__pycache__/", "/.git/", "/dist/",
    # Chromium's own profile store lives inside the project and is full of
    # files with ordinary-looking names ("README", "Config") that crowd out
    # the user's real documents.
    "/browser_profile/", "/.tmp.driveupload/",
)


# Depth is bounded so a walk of Documents can't wander into a deep tree
# and stall a spoken turn; the excluded segments above prune the worst
# offenders (node_modules and friends) before that limit matters.
_WALK_MAX_DEPTH = 6


def _walk_filename_matches(query: str, limit: int) -> list[str]:
    """Filename search by walking the allowed roots directly.

    Spotlight is not a dependable index for this. Measured on this machine:
    `mdfind -onlyin <project> voice_loop` returned nothing while
    max/voice_loop.py plainly existed, and Desktop returned nothing at
    all, even though mdfind found 480 hits elsewhere in the home folder.
    Developer folders in particular often go unindexed, and mdfind reports
    that as an empty result rather than an error — so "search my files"
    answered "no files found" for files that were right there.
    """
    needle = query.lower().strip()
    if not needle:
        return []
    matches: list[str] = []
    for root in READ_ONLY_ROOTS:
        if not root.exists():
            continue
        root_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            if len(current.parts) - root_depth >= _WALK_MAX_DEPTH:
                dirnames[:] = []
                continue
            # Prune in place so os.walk never descends into them at all.
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and f"/{d}/" not in _SEARCH_EXCLUDE_SEGMENTS
                and d not in ("node_modules", "__pycache__", "dist", ".venv", "browser_profile")
            ]
            for name in filenames:
                if name.startswith("."):
                    continue
                if needle in name.lower():
                    matches.append(str(current / name))
                    if len(matches) >= limit:
                        return matches
    return matches


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

    # Always merge in a direct filename walk rather than trusting the
    # index. mdfind still earns its place — it searches CONTENT of indexed
    # documents (PDFs, Pages, Word), which a filename walk cannot — but it
    # cannot be the only source when whole folders are missing from it.
    if len(results) < max_results:
        for path in _walk_filename_matches(query, max_results - len(results)):
            if path not in seen:
                seen.add(path)
                results.append(path)

    if not results:
        return f'No files found matching "{query}".'
    return "\n".join(results[:max_results])
