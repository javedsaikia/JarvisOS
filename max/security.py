"""Local-only guards. Max must not be reachable from the LAN or WAN.

The HUD and bridge bind loopback. This module also checks the peer
address and Origin on each WebSocket handshake, hardens file modes on
secrets and memory, and refuses a remote ?bridge= URL in the HUD.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}

ALLOWED_ORIGINS = {
    "http://max.localhost:2015",
    "http://127.0.0.1:2015",
    "http://localhost:2015",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}

PRIVATE_MODE = 0o600
PRIVATE_DIR = 0o700


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    h = host.split("%", 1)[0].lower()
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    return h in LOOPBACK or h == "localhost" or h.endswith(".localhost")


def is_loopback_addr(addr) -> bool:
    if not addr:
        return False
    host = addr[0] if isinstance(addr, (tuple, list)) else str(addr)
    return is_loopback_host(str(host))


def origin_allowed(origin: str | None) -> bool:
    # No Origin: local scripts (./start_max.sh), not a browser tab.
    if not origin:
        return True
    return origin.rstrip("/") in ALLOWED_ORIGINS


def harden_file(path: Path, mode: int = PRIVATE_MODE) -> None:
    try:
        if path.exists() and path.is_file():
            os.chmod(path, mode)
    except OSError:
        pass


def harden_dir(path: Path, mode: int = PRIVATE_DIR) -> None:
    try:
        if path.exists() and path.is_dir():
            os.chmod(path, mode)
    except OSError:
        pass


def harden_home() -> None:
    """Lock down keys, memory, and live control files on this Mac."""
    root = Path(__file__).resolve().parent
    harden_file(root / ".env")
    harden_file(root / ".composio_session.json")
    harden_file(root / "config.json")
    mem = root / "memory"
    harden_dir(mem)
    for name in (
        "facts.md",
        "profile.md",
        "conversation_log.jsonl",
        "conversation_archive.jsonl",
        "memory_state.json",
    ):
        harden_file(mem / name)
    for tmp in (
        Path("/tmp/max_control.json"),
        Path("/tmp/max_bridge.log"),
        Path("/tmp/voice_loop.log"),
        Path("/tmp/max_voice_events.jsonl"),
        Path("/tmp/max_voice_loop.pid"),
    ):
        harden_file(tmp)


def websocket_process_request(connection, request):
    """Reject anything that is not this Mac talking to itself."""
    remote = getattr(connection, "remote_address", None)
    if not is_loopback_addr(remote):
        return connection.respond(403, "Forbidden")
    origin = None
    try:
        origin = request.headers.get("Origin")
    except Exception:
        origin = None
    if not origin_allowed(origin):
        return connection.respond(403, "Forbidden")
    return None


def safe_bridge_url(candidate: str | None, fallback: str) -> str:
    """HUD ?bridge= may only point at loopback. Anything else is ignored."""
    if not candidate:
        return fallback
    try:
        parsed = urlparse(candidate)
    except Exception:
        return fallback
    if parsed.scheme not in ("ws", "wss"):
        return fallback
    host = parsed.hostname or ""
    if not is_loopback_host(host):
        return fallback
    return candidate
