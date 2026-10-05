"""A calendar event waiting on a time. Asks for it; invents nothing.

"Add a meeting with 11 labs" names no time, and the old path filled the
gap by rounding up to the next full hour. Measured live at 21:35 that put
a real meeting at 10:00 PM — a time nobody had said, offered as if it
were the request. The user declined, three times in a row, and no event
was ever created.

Guessing is the wrong trade here. A meeting at a guessed hour is worse
than no meeting, because it looks scheduled: it sits in the calendar, it
alerts at the wrong moment, and the real slot stays empty. One short
question costs a few seconds and is always right.

Same shape as bella/tools/reminder.py and bella/tools/email_compose.py: the
subject is kept from the opening request, Bella asks for the time, the
answer arrives as an ordinary turn, and nothing is written until the time
is known and the whole thing is confirmed.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from bella.tools import parsing

_state: dict | None = None

_CANCEL_RE = re.compile(
    r"\b(?:cancel|never ?mind|forget it|drop it|stop|no thanks?)\b", re.IGNORECASE
)


def active() -> bool:
    return _state is not None


def start(title: str, duration: timedelta) -> dict:
    global _state
    _state = {"title": title, "duration": duration, "asked": False}
    return _state


def cancel() -> None:
    global _state
    _state = None


def snapshot() -> dict:
    return dict(_state or {})


def wants_cancel(text: str) -> bool:
    return bool(_CANCEL_RE.search(text or ""))


def note_asked() -> None:
    if _state is not None:
        _state["asked"] = True


def question() -> str:
    """Short on purpose — it is spoken, and every word is waiting time.

    Quoting the parsed title back read as confusion rather than a
    question: with a bad parse it became 'What time is "go ahead and add
    in meeting"?', which sounds like Bella has no idea what was said.
    """
    title = (_state or {}).get("title") or ""
    if title and title.lower() not in ("meeting", "event", "appointment", "call"):
        return f"What time for {title}?"
    return "What time?"


def apply_time(text: str, now: datetime | None = None) -> datetime | None:
    """The start time from a reply like "tomorrow at 1pm", or None.

    The stored title is NOT re-derived from this reply — the reply is an
    answer to a question, not a fresh request, so "1pm" must not become
    the event's name.
    """
    if _state is None or not parsing.event_has_time(text or ""):
        return None
    now = now or datetime.now()
    _, start_at, _ = parsing.extract_event(text or "", now)
    return start_at


def duration() -> timedelta:
    return (_state or {}).get("duration") or timedelta(hours=1)


def title() -> str:
    return (_state or {}).get("title") or "Meeting"
