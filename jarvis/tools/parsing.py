"""Lightweight, dependency-free natural-language parsing for dictated
calendar/note commands. Regex-based on purpose: this stays deterministic and
fast (no extra Ollama round-trip) for the common phrasings Wispr produces.
Ambiguous phrasing falls back to sane defaults rather than failing.
"""
import re
from datetime import datetime, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_TIME_RE = re.compile(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)
_DURATION_RE = re.compile(r"\bfor\s+(\d+)\s*(hour|hr|minute|min)s?\b", re.IGNORECASE)


def _resolve_day(text: str, now: datetime) -> datetime:
    lowered = text.lower()
    day = datetime(now.year, now.month, now.day)

    if "tomorrow" in lowered:
        return day + timedelta(days=1)
    if "today" in lowered:
        return day

    for i, name in enumerate(WEEKDAYS):
        if name in lowered:
            delta = (i - now.weekday()) % 7
            if delta == 0 and "next" in lowered:
                delta = 7
            return day + timedelta(days=delta)

    return day


def _resolve_time(text: str, day: datetime, now: datetime) -> datetime:
    match = _TIME_RE.search(text)
    if not match:
        # No time mentioned — default to the next full hour.
        rounded = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return day.replace(hour=rounded.hour, minute=0)

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()

    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    elif not meridiem:
        # Ambiguous hour with no am/pm — assume daytime business hours.
        if 1 <= hour <= 7:
            hour += 12

    return day.replace(hour=hour % 24, minute=minute)


def _resolve_duration(text: str) -> timedelta:
    match = _DURATION_RE.search(text)
    if not match:
        return timedelta(hours=1)
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit in ("hour", "hr"):
        return timedelta(hours=amount)
    return timedelta(minutes=amount)


def extract_event_time(text: str, now: datetime = None) -> tuple[datetime, timedelta]:
    now = now or datetime.now()
    day = _resolve_day(text, now)
    start = _resolve_time(text, day, now)
    duration = _resolve_duration(text)
    return start, duration


_EVENT_LEADING_TRIGGER_RE = re.compile(
    r"^(?:please\s+)?(?:add|schedule|create|book|put)\s+(?:an?\s+)?(?:event\s+)?",
    re.IGNORECASE,
)
_EVENT_CALENDAR_SUFFIX_RE = re.compile(
    r"\b(?:event|meeting|appointment|call)?\s*(?:on|to)\s+(?:my\s+)?calendar\b", re.IGNORECASE
)
_DAY_WORD_RE = re.compile(r"\b(?:today|tomorrow|" + "|".join(WEEKDAYS) + r")\b", re.IGNORECASE)


def extract_event(text: str, now: datetime = None) -> tuple[str, datetime, timedelta]:
    """Return (title, start, duration) for a dictated event-creation command."""
    now = now or datetime.now()
    day = _resolve_day(text, now)
    start = _resolve_time(text, day, now)
    duration = _resolve_duration(text)

    cut_positions = [
        m.start()
        for m in (_TIME_RE.search(text), _DAY_WORD_RE.search(text), _DURATION_RE.search(text))
        if m
    ]
    cut = min(cut_positions) if cut_positions else len(text)

    title = text[:cut]
    title = _EVENT_LEADING_TRIGGER_RE.sub("", title)
    title = _EVENT_CALENDAR_SUFFIX_RE.sub("", title)
    title = re.sub(r"\b(for|at|on)\s*$", "", title, flags=re.IGNORECASE).strip(" ,.-")
    if not title:
        title = "New event"

    return title, start, duration


_NOTE_TITLE_TRIGGERS = re.compile(
    # Dictated speech naturally punctuates after "note" (a comma from a
    # pause), not just the ":"/"-" a typed command would use.
    r"^(?:take|add|create|make|write|jot down)\s+(?:a\s+)?note\b\s*(?:called|titled|named)?\s*[:,\-]?\s*",
    re.IGNORECASE,
)


def _strip_trailing_punctuation(s: str) -> str:
    return s.strip().strip(",.;").strip()


def extract_note(text: str) -> tuple[str, str]:
    """Return (title, body) for a dictated note-creation command."""
    remainder = _strip_trailing_punctuation(_NOTE_TITLE_TRIGGERS.sub("", text))

    if ":" in remainder:
        title, _, body = remainder.partition(":")
        title = _strip_trailing_punctuation(title.strip("\"'"))
        body = _strip_trailing_punctuation(body)
        if title and len(title) <= 60:
            return title, body

    if not remainder:
        remainder = "Untitled note"

    title = remainder if len(remainder) <= 40 else remainder[:37].rstrip() + "..."
    return title, remainder


# Strips a leading trigger phrase to get at the literal command. Two
# variants: one requires the word "command" (handles "run this exact shell
# command: ls -la"), the other doesn't (handles bare "Run ls -la").
_SHELL_TRIGGER_WITH_COMMAND_RE = re.compile(
    r"^(?:please\s+)?(?:run|execute)\s+(?:this\s+|the\s+|following\s+|exact\s+)*"
    r"(?:shell\s+)?command\s*[:\-,]?\s*",
    re.IGNORECASE,
)
_SHELL_TRIGGER_BARE_RE = re.compile(
    r"^(?:please\s+)?(?:run|execute)\s+(?:this\s+|the\s+|following\s+|exact\s+)*",
    re.IGNORECASE,
)
# Both trigger regexes above are anchored to the start of the string —
# seen live, dictated speech often has a lead-in before the actual "run
# ..." request ("here is run this command: sleep 30"), which the anchor
# never matches, silently falling through to treating the WHOLE phrase —
# lead-in included — as the literal command. Stripped first so the
# trigger regexes still see it starting right at "run"/"execute".
_LEADING_FILLER_RE = re.compile(
    r"^(?:so\s+|okay\s+|ok\s+|well\s+|here(?:'s|\s+is)\s+|there\s+|"
    r"i want you to\s+|can you\s+|could you\s+)+",
    re.IGNORECASE,
)


def extract_shell_command(text: str) -> str:
    """Return the literal command from a dictated/typed 'run ...' request."""
    stripped = _LEADING_FILLER_RE.sub("", text.strip())
    remainder = _SHELL_TRIGGER_WITH_COMMAND_RE.sub("", stripped)
    if remainder == stripped:
        remainder = _SHELL_TRIGGER_BARE_RE.sub("", stripped)
    return remainder.strip().rstrip(".").strip()


# Anchored to the start of the string, this never matched real dictated
# speech, which leads with something before the verb ("Hey Jarvis, can you
# play...", "I want you to play..."). The whole sentence was then handed to
# Spotify search verbatim — live, "can you play a song by Pink Floyd High
# Hopes on Spotify" returned "The Point Of No Return by Jonathan Young".
# So: find the play verb ANYWHERE and take what follows. First occurrence
# rather than last, so a title that contains the word still survives
# ("play Child's Play by Drake").
_SONG_TRIGGER_RE = re.compile(r"\b(?:play|put on|listen to)\b\s*", re.IGNORECASE)
_SONG_TRAILING_SUFFIX_RE = re.compile(r"[,\s]+(?:on|in|from|over)\s+spotify\s*$", re.IGNORECASE)
# "a song by X", "the track called Y" — drop the framing words so only the
# name reaches the search. Requires trailing whitespace, so a bare "some
# music" (a generic request, not a title) is left alone.
_SONG_FRAMING_RE = re.compile(
    r"^(?:me\s+)?(?:a|an|the|some)?\s*(?:song|track|tune)\s*(?:by|from|called|named|titled)?\s+",
    re.IGNORECASE,
)


def extract_song_query(text: str) -> str:
    """Return the song/artist search query from a dictated 'play X [on
    Spotify]' request. Deliberately loose — fed straight to Spotify's own
    fuzzy search (jarvis/spotify_client.py), not parsed into structured
    fields, so leftover filler words ("some", "that") cost nothing.
    """
    stripped = text.strip()
    match = _SONG_TRIGGER_RE.search(stripped)
    remainder = stripped[match.end():] if match else stripped
    # Trailing punctuation comes off FIRST: dictated speech ends in "?" or
    # "." and the "... on spotify$" pattern is anchored, so leaving it on
    # meant the suffix never matched and "on Spotify" stayed in the query.
    remainder = remainder.strip().strip("?.!,").strip()
    remainder = _SONG_TRAILING_SUFFIX_RE.sub("", remainder)
    remainder = _SONG_FRAMING_RE.sub("", remainder)
    return remainder.strip().strip("?.!,").strip()
