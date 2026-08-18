"""Lightweight, dependency-free natural-language parsing for dictated
calendar/note commands. Regex-based on purpose: this stays deterministic and
fast (no extra Ollama round-trip) for the common phrasings Wispr produces.
Ambiguous phrasing falls back to sane defaults rather than failing.
"""
import re
from datetime import datetime, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# "at 1pm", "at the 1 pm", "at 13:30", and a bare "1pm" with no "at" at
# all. The filler word matters: dictated speech produced "at the 1pm",
# which the older `at\s+\d` form did not match, so the time was silently
# dropped and the event landed at a default hour nobody asked for.
_TIME_RE = re.compile(
    r"\bat\s+(?:the\s+|around\s+|about\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b"
    r"|\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
    re.IGNORECASE,
)


def _time_parts(match: re.Match) -> tuple[str, str, str]:
    """(hour, minute, meridiem) from whichever alternative matched."""
    if match.group(1) is not None:
        return match.group(1), match.group(2), match.group(3)
    return match.group(4), match.group(5), match.group(6)
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

    raw_hour, raw_minute, raw_meridiem = _time_parts(match)
    hour = int(raw_hour)
    minute = int(raw_minute or 0)
    meridiem = (raw_meridiem or "").lower()

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


# Dictated requests are wrapped in politeness the typed form never had:
# "Can you add a meeting...", "I want you to add a meeting...". Stripped
# before the trigger below, which only ever anchored at position 0 — so
# live these survived into the title, and Max offered to create an event
# literally called "Can you add a ? Google Calendar".
_EVENT_POLITE_PREFIX_RE = re.compile(
    r"^\s*(?:hey|ok(?:ay)?|so)?[\s,]*"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"|i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to\s+"
    r"|i\s+want\s+"
    r"|go\s+ahead\s+(?:and\s+)?"
    r"|let'?s\s+"
    r"|please\s+)+",
    re.IGNORECASE,
)
_EVENT_LEADING_TRIGGER_RE = re.compile(
    # "add in a meeting" — dictation drops the particle in freely, and
    # leaving it made the title "in a meeting".
    r"^(?:please\s+)?(?:add|schedule|create|book|put|set\s+up)\s+(?:in\s+)?"
    r"(?:an?\s+)?(?:new\s+)?(?P<noun>event|meeting|appointment|call)?\s*"
    r"(?:(?P<named>called|titled|named)|(?P<connector>with|about|for))?\s*",
    re.IGNORECASE,
)
# A trailing destination is not part of the title: "...? Google Calendar"
# was dictated as a routing hint and became the entire event name.
_EVENT_TRAILING_DEST_RE = re.compile(
    r"\b(?:on|in|to)?\s*(?:my\s+)?(?:google\s+)?calendar\s*$", re.IGNORECASE
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

    # A time that has already passed was never the intent. "Monday at 9am"
    # said on a Monday evening means the next one, and a bare "at 9am"
    # said at night means tomorrow — creating an event in the past looks
    # exactly like the event not being created at all. "today" and
    # "tomorrow" are explicit and left alone.
    lowered = text.lower()
    if start < now and "today" not in lowered and "tomorrow" not in lowered:
        start += timedelta(days=7 if _DAY_WORD_RE.search(text) else 1)

    cut_positions = [
        m.start()
        for m in (_TIME_RE.search(text), _DAY_WORD_RE.search(text), _DURATION_RE.search(text))
        if m
    ]
    cut = min(cut_positions) if cut_positions else len(text)

    title = _EVENT_POLITE_PREFIX_RE.sub("", text[:cut])
    trigger = _EVENT_LEADING_TRIGGER_RE.match(title)
    noun = connector = ""
    if trigger:
        noun = (trigger.group("noun") or "").strip()
        connector = (trigger.group("connector") or "").strip()
        title = title[trigger.end():]
    title = _EVENT_CALENDAR_SUFFIX_RE.sub("", title)
    title = _EVENT_TRAILING_DEST_RE.sub("", title)
    title = re.sub(r"\b(for|at|on|with|to)\s*$", "", title, flags=re.IGNORECASE)
    title = title.strip(" ,.-?!")
    # "a call with Priya" -> "Call with Priya". Only when the noun is not
    # already in what the user said, so "meeting with the meeting team"
    # does not gain a second one.
    named = bool(trigger and trigger.group("named"))
    if title and noun and not named and noun.lower() not in title.lower():
        title = f"{noun.capitalize()} {connector} {title}".replace("  ", " ").strip()
    elif not title and noun:
        title = noun.capitalize()
    if not title:
        # No subject was dictated at all ("add a meeting tomorrow at 1").
        # "Meeting" is what was actually said; inventing a topic would be
        # worse than admitting there isn't one.
        title = "Meeting"

    return title, start, duration


def event_has_time(text: str) -> bool:
    """True if the request names a clock time, so none has to be invented."""
    return bool(_TIME_RE.search(text))


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
# speech, which leads with something before the verb ("Hey Max, can you
# play...", "I want you to play..."). The whole sentence was then handed to
# Spotify search verbatim — live, "can you play a song by Pink Floyd High
# Hopes on Spotify" returned "The Point Of No Return by Jonathan Young".
# So: find the play verb ANYWHERE and take what follows. First occurrence
# rather than last, so a title that contains the word still survives
# ("play Child's Play by Drake").
# "bile"/"ply" are measured STT renderings of "play" — seen live: "Bile,
# nothing else matters by Metallica" missed every play pattern, fell
# through to chat, and the model claimed it was playing while nothing was.
_SONG_TRIGGER_RE = re.compile(r"\b(?:play|put on|listen to|bile|ply)\b\s*", re.IGNORECASE)
_SONG_TRAILING_SUFFIX_RE = re.compile(r"[,\s]+(?:on|in|from|over|and)\s+spotify\s*$", re.IGNORECASE)
# "Spotify, Bohemian Rhapsody by Queen" — the app name leads the request
# as often as it trails it, and left in place it was searched as part of
# the title.
_SONG_LEADING_APP_RE = re.compile(r"^(?:on\s+|in\s+)?spotify\b[\s,:—-]*", re.IGNORECASE)
# "a song by X", "the track called Y" — drop the framing words so only the
# name reaches the search. Requires trailing whitespace, so a bare "some
# music" (a generic request, not a title) is left alone.
_SONG_FRAMING_RE = re.compile(
    r"^(?:me\s+)?(?:a|an|the|some)?\s*(?:song|track|tune)\s*(?:by|from|called|named|titled)?\s+",
    re.IGNORECASE,
)
# "play it" / "go ahead and play that" is a follow-up, not a title. Seen
# live: extract_song_query("Go ahead and play it") returned "it", Spotify
# searched the word, and a Hindi film song started.
_REFERENTIAL_SONG = {
    "",
    "it",
    "that",
    "this",
    "them",
    "again",
    "the song",
    "the track",
    "the same",
    "that one",
    "this one",
    "that song",
    "this song",
    "that track",
    "this track",
}


def is_referential_song_query(query: str) -> bool:
    q = (query or "").strip().strip("?.!,").lower()
    q = re.sub(r"\s+", " ", q)
    return q in _REFERENTIAL_SONG


# "I want to go to Guwahati right now by a car" is travel, not
# "Guwahati" by the artist "a car". Seen live: that split poisoned the
# pending Spotify query, and a later "yes" on an email draft played
# Guwahati Chariali.
_NOT_AN_ARTIST = re.compile(
    r"^(?:an?|the)\s+"
    r"(?:car|taxi|cab|bus|train|plane|flight|bike|metro|road|way)$",
    re.IGNORECASE,
)


def split_song_artist(query: str) -> tuple[str, str | None]:
    """Split 'High Hopes by Pink Floyd' into (title, artist)."""
    q = (query or "").strip()
    match = re.search(r"\s+by\s+(.+)$", q, re.IGNORECASE)
    if not match:
        return q, None
    title = q[: match.start()].strip(" \"'")
    artist = match.group(1).strip(" \"'")
    if not title or not artist:
        return q, None
    if _NOT_AN_ARTIST.match(artist) or artist.lower() in {
        "car", "taxi", "cab", "bus", "train", "plane", "bike",
    }:
        return q, None
    return title, artist


def has_song_trigger(text: str) -> bool:
    return bool(_SONG_TRIGGER_RE.search(text or ""))


def extract_song_query(text: str) -> str:
    """Return the song/artist search query from a dictated 'play X [on
    Spotify]' request. Deliberately loose — fed straight to Spotify's own
    fuzzy search (max/spotify_client.py), not parsed into structured
    fields. Referential leftovers ("it", "that") are returned as-is so
    the caller can reuse the last real query instead of searching them.
    """
    stripped = text.strip()
    match = _SONG_TRIGGER_RE.search(stripped)
    remainder = stripped[match.end():] if match else stripped
    # Trailing punctuation comes off FIRST: dictated speech ends in "?" or
    # "." and the "... on spotify$" pattern is anchored, so leaving it on
    # meant the suffix never matched and "on Spotify" stayed in the query.
    remainder = remainder.strip().strip("?.!,").strip()
    remainder = _SONG_TRAILING_SUFFIX_RE.sub("", remainder)
    remainder = _SONG_LEADING_APP_RE.sub("", remainder)
    remainder = _SONG_FRAMING_RE.sub("", remainder)
    return remainder.strip().strip("?.!,").strip()


_AFFIRMATION_RE = re.compile(
    r"^(?:yes|yeah|yep|yup|sure|ok|okay|please|go ahead|please do|do it)[.!?]?\s*$",
    re.IGNORECASE,
)


def is_affirmation(text: str) -> bool:
    return bool(_AFFIRMATION_RE.match((text or "").strip()))
