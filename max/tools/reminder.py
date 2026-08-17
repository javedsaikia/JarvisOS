"""An in-progress reminder. Asks for what it doesn't know; invents nothing.

"Can you set up a reminder?" carries neither a subject nor a time, and the
only honest response is to ask for both. Measured live, the old behaviour
was the opposite: with no route for reminders at all the request fell
through to the chat model, which replied *"I've added a reminder for you
on the Google Calendar. Want me to make it a recurring reminder?"* — and
nothing whatsoever had been created. A confident lie about a reminder is
worse than no reminder, because the user stops thinking about the thing
they wanted remembering.

Slots are filled across turns, the same shape as max/tools/email_compose.py:
Max asks for the missing one, the answer arrives as an ordinary reply, and
nothing is written until both are known and confirmed.

Where it lands follows from whether a clock time was given, because that
is what the two Google products actually differ on:

- a time of day  -> Google Calendar event, which can alert at that moment
- a date only    -> Google Tasks, which is a to-do list with due dates

That is a real capability difference, not a preference: Tasks has no
notion of "3pm", so a reminder with a time put there would never sound.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

_state: dict | None = None

# Times of day people actually say. Kept separate from the date words
# below because their presence is what decides Calendar vs Tasks.
_TIME_RE = re.compile(
    r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)"
    r"|\bat\s+(\d{1,2})(?::(\d{2}))\b"
    r"|\b(noon|midday|midnight)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(today|tomorrow|tonight|day after tomorrow|next week|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday"]

# "remind me to <thing>" / "reminder about <thing>" / "reminder for <thing>"
_SUBJECT_RE = re.compile(
    r"\b(?:remind\s+me\s+(?:to|about|that)|reminder\s+(?:to|about|for)|"
    r"don'?t\s+let\s+me\s+forget\s+(?:to|about))\s+(.+)$",
    re.IGNORECASE,
)
_TRIGGER_ONLY_RE = re.compile(
    r"^\s*(?:can\s+you\s+|could\s+you\s+|please\s+)?"
    r"(?:set(?:\s+up)?|add|create|make|put)?\s*(?:an?\s+)?reminders?\s*"
    r"(?:for\s+me\s*)?[?.!]*\s*$",
    re.IGNORECASE,
)


def active() -> bool:
    return _state is not None


def start() -> dict:
    global _state
    _state = {"subject": "", "when": None, "has_time": False, "asked": ""}
    return _state


def cancel() -> None:
    global _state
    _state = None


def snapshot() -> dict:
    return dict(_state or {})


def is_trigger(text: str) -> bool:
    """A bare "set a reminder" with nothing else in it."""
    return bool(_TRIGGER_ONLY_RE.match(text or ""))


def extract_subject(text: str) -> str:
    """What the reminder is about, if they said it."""
    match = _SUBJECT_RE.search(text or "")
    if not match:
        return ""
    subject = match.group(1).strip()
    # Strip the time/date words back out — "call mum at 5pm tomorrow"
    # is a reminder to call mum, not to "call mum at 5pm tomorrow".
    subject = _TIME_RE.sub(" ", subject)
    subject = _DATE_RE.sub(" ", subject)
    subject = re.sub(r"\b(?:at|on|by|this|next)\b\s*$", "", subject.strip(), flags=re.I)
    subject = re.sub(r"\s{2,}", " ", subject).strip(" ,.;:-")
    return subject


def extract_when(text: str, now: datetime | None = None) -> tuple[datetime | None, bool]:
    """(when, had_a_clock_time). None when they named no day and no time.

    Deliberately conservative: a reminder placed at a guessed time is a
    reminder that fires at the wrong moment, which is worse than being
    asked one short question.
    """
    now = now or datetime.now()
    raw = text or ""
    date_hit = _DATE_RE.search(raw)
    time_hit = _TIME_RE.search(raw)
    if not date_hit and not time_hit:
        return None, False

    day = datetime(now.year, now.month, now.day)
    if date_hit:
        word = date_hit.group(1).lower()
        if word == "tomorrow":
            day += timedelta(days=1)
        elif word == "day after tomorrow":
            day += timedelta(days=2)
        elif word == "next week":
            day += timedelta(days=7)
        elif word in _WEEKDAYS:
            delta = (_WEEKDAYS.index(word) - now.weekday()) % 7 or 7
            day += timedelta(days=delta)
        # "today"/"tonight" leave `day` as it is

    hour, minute = None, None
    if time_hit:
        if time_hit.group(6):  # noon / midday / midnight
            hour = 0 if time_hit.group(6).lower() == "midnight" else 12
            minute = 0
        elif time_hit.group(1):
            hour, minute = int(time_hit.group(1)), int(time_hit.group(2) or 0)
            meridiem = (time_hit.group(3) or "").lower().replace(".", "")
            if meridiem.startswith("p") and hour != 12:
                hour += 12
            elif meridiem.startswith("a") and hour == 12:
                hour = 0
            elif not meridiem and 1 <= hour <= 7:
                # Bare "at 5" for a reminder means the afternoon far more
                # often than 5am.
                hour += 12
        elif time_hit.group(4):
            hour, minute = int(time_hit.group(4)), int(time_hit.group(5) or 0)

    if hour is None:
        # A date with no time: 9am is the ordinary "some time that day".
        when = day.replace(hour=9, minute=0)
        # "tonight" without a clock time is the evening, not the morning.
        if date_hit and date_hit.group(1).lower() == "tonight":
            when = day.replace(hour=20, minute=0)
        return when, False

    when = day.replace(hour=hour % 24, minute=minute or 0)
    # "remind me at 8" said at 10pm means tomorrow morning.
    if not date_hit and when <= now:
        when += timedelta(days=1)
    return when, True


def apply_text(text: str) -> dict:
    """Fold whatever this utterance named into the open reminder."""
    if _state is None:
        start()
    stripped = (text or "").strip()

    subject = extract_subject(stripped)
    when, has_time = extract_when(stripped)

    if not subject and _state["asked"] == "subject" and stripped:
        # They are answering "what should I remind you about?" — the whole
        # reply is the subject, minus any time they slipped in with it.
        cleaned = _TIME_RE.sub(" ", stripped)
        cleaned = _DATE_RE.sub(" ", cleaned)
        cleaned = re.sub(r"^\s*(?:to|about|that)\s+", "", cleaned.strip(), flags=re.I)
        subject = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.;:-")

    if subject:
        _state["subject"] = subject
    if when is not None:
        _state["when"] = when
        _state["has_time"] = has_time
    return snapshot()


def missing() -> str | None:
    if not _state:
        return "subject"
    if not (_state.get("subject") or "").strip():
        return "subject"
    if _state.get("when") is None:
        return "when"
    return None


def note_asked(slot: str) -> None:
    if _state is not None:
        _state["asked"] = slot


def destination() -> str:
    """"calendar" when a clock time was given, else "tasks". See the
    module docstring — Tasks cannot represent a time of day."""
    return "calendar" if (_state or {}).get("has_time") else "tasks"


def spoken_when() -> str:
    when = (_state or {}).get("when")
    if not isinstance(when, datetime):
        return ""
    if (_state or {}).get("has_time"):
        return when.strftime("%a %b %-d at %-I:%M %p")
    return when.strftime("%a %b %-d")

_CANCEL_RE = re.compile(
    r"\b(?:cancel|never ?mind|forget it|drop it|stop|don'?t bother)\b",
    re.IGNORECASE,
)


def wants_cancel(text: str) -> bool:
    return bool(_CANCEL_RE.search(text or ""))
