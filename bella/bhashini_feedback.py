"""Capture Bhashini ASR/TTS feedback for Assamese eval / contribution.

Records land in bella/memory/assamese_eval_dataset.jsonl (gitignored).
Opt-in via bhashini_feedback_enabled in config.json.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

FEEDBACK_PATH = Path(__file__).parent / "memory" / "assamese_eval_dataset.jsonl"

_lock = threading.Lock()
_last_call: dict[str, Any] | None = None
_pending: str | None = None  # "correction" after /flag

_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{8,}\d")


class BhashiniFeedbackError(Exception):
    pass


def enabled(cfg: dict | None = None) -> bool:
    if cfg is None:
        from bella.config import load_config
        cfg = load_config()
    return bool(cfg.get("bhashini_feedback_enabled", True))


def _scrub(text: str) -> str:
    if not text:
        return ""
    out = _EMAIL_RE.sub("[email]", text)
    return _PHONE_RE.sub("[phone]", out)


def _append(record: dict[str, Any]) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _lock:
        with FEEDBACK_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line)


def record_call(
    task: str,
    *,
    input_text: str,
    output_text: str,
    service_id: str = "",
    language: str = "as",
    source: str = "voice",
    cfg: dict | None = None,
    extra: dict | None = None,
) -> None:
    """Remember the latest Bhashini call (for /flag and /good)."""
    if not enabled(cfg):
        return
    global _last_call
    snapshot = {
        "ts": time.time(),
        "task": task,
        "language": language,
        "service_id": service_id or "",
        "input": _scrub((input_text or "").strip()),
        "output": _scrub((output_text or "").strip()),
        "source": source,
        "extra": extra or {},
    }
    with _lock:
        _last_call = snapshot


def record_fallback(
    task: str,
    reason: str,
    *,
    input_text: str = "",
    service_id: str = "",
    language: str = "as",
    source: str = "voice",
    cfg: dict | None = None,
) -> None:
    """Auto-log when Bhashini fails and Sarvam takes over."""
    if not enabled(cfg):
        return
    _append({
        "ts": time.time(),
        "label": "fallback",
        "task": task,
        "language": language,
        "service_id": service_id or "",
        "input": _scrub((input_text or "").strip()),
        "output": "",
        "correction": None,
        "source": source,
        "reason": _scrub(reason),
    })


def last_call() -> dict[str, Any] | None:
    with _lock:
        return dict(_last_call) if _last_call else None


def pending_correction() -> bool:
    return _pending == "correction"


def cancel_pending() -> None:
    global _pending
    _pending = None


def log_feedback(
    label: str,
    *,
    correction: str | None = None,
    reason: str | None = None,
    snapshot: dict | None = None,
) -> dict[str, Any]:
    """Write one eval row. Uses last_call() when snapshot omitted."""
    snap = snapshot or last_call()
    if not snap:
        raise BhashiniFeedbackError("nothing to log — no recent Bhashini call")
    record = {
        "ts": time.time(),
        "label": label,
        "task": snap.get("task", ""),
        "language": snap.get("language", "as"),
        "service_id": snap.get("service_id", ""),
        "input": snap.get("input", ""),
        "output": snap.get("output", ""),
        "correction": _scrub((correction or "").strip()) or None,
        "source": snap.get("source", ""),
        "reason": _scrub((reason or "").strip()) or None,
    }
    _append(record)
    return record


def begin_flag(cfg: dict | None = None) -> str:
    if not enabled(cfg):
        return "Bhashini feedback logging is off."
    if not last_call():
        return "Nothing recent from Bhashini to flag."
    global _pending
    _pending = "correction"
    snap = last_call() or {}
    task = snap.get("task", "?").upper()
    return (
        f"Got it — flagging the last Bhashini {task} call. "
        "What should the correct text have been? (/cancel to drop)"
    )


def begin_good(cfg: dict | None = None) -> str:
    if not enabled(cfg):
        return "Bhashini feedback logging is off."
    if not last_call():
        return "Nothing recent from Bhashini to mark good."
    log_feedback("good")
    return "Thanks — logged as a good Bhashini result."


def consume_correction(text: str, cfg: dict | None = None) -> str | None:
    """If waiting for a correction, log and return the ack message."""
    global _pending
    if _pending != "correction":
        return None
    if not enabled(cfg):
        _pending = None
        return None
    cleaned = (text or "").strip()
    if not cleaned:
        return "Send the corrected Assamese text, or /cancel."
    if cleaned.startswith("/"):
        return None
    log_feedback("bad", correction=cleaned)
    _pending = None
    return "Saved — thanks, that helps improve Bhashini."


def try_slash_command(text: str, cfg: dict | None = None) -> str | None:
    """Handle /flag, /good, /cancel for the feedback flow."""
    stripped = (text or "").strip()
    low = stripped.lower()
    if low.startswith("/cancel"):
        if pending_correction():
            cancel_pending()
            return "Cancelled — nothing saved."
        return "Nothing to cancel."
    if low.startswith("/flag"):
        rest = stripped[len("/flag"):].strip()
        if rest:
            if not enabled(cfg):
                return "Bhashini feedback logging is off."
            if not last_call():
                return "Nothing recent from Bhashini to flag."
            log_feedback("bad", correction=rest)
            cancel_pending()
            return "Saved — thanks, that helps improve Bhashini."
        return begin_flag(cfg)
    if low.startswith("/good"):
        return begin_good(cfg)
    if low.startswith("/export-bhashini-feedback"):
        count = export_summary()
        export_path = export_records()
        return (f"Feedback log: {count} rows in {FEEDBACK_PATH}. "
                f"Copy for sharing: {export_path}")
    return None


def export_summary() -> int:
    if not FEEDBACK_PATH.exists():
        return 0
    return sum(1 for _ in FEEDBACK_PATH.open(encoding="utf-8"))


def export_records(dest: Path | None = None) -> Path:
    """Copy the JSONL log for sharing with Bhashini / Bhasha Daan."""
    dest = dest or FEEDBACK_PATH.with_name("assamese_eval_export.jsonl")
    if not FEEDBACK_PATH.exists():
        dest.write_text("", encoding="utf-8")
        return dest
    dest.write_text(FEEDBACK_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return dest
