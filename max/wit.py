"""One house voice for every backend and every canned tool line.

Chat models read VOICE. Deterministic tools (weather, location, nearby)
cannot invent jokes, so they append a rotating one-liner after the fact.
The fact always comes first — TTS users should hear Jorhat before the wink.
"""


VOICE = (
    "Talk like a sharp friend, not a professor. One or two short sentences. "
    "A little dry wit is fine — a word or clause, not a second speech. "
    "Never lecture, never walk through every step, never pad. "
    "If they asked for a page, file, or piece of work: summarize, then ask "
    "what to do next. If they asked you to do something: do it and confirm "
    "in one line. Wait for them before the next action."
)


def season(fact: str, *quips: str) -> str:
    """`fact` unchanged, plus one quip picked from the fact itself so the
    same weather does not get a new punchline every time it is asked."""
    text = (fact or "").rstrip()
    if not text or not quips:
        return text
    pick = quips[sum(map(ord, text)) % len(quips)]
    if text.endswith((".", "!", "?")):
        return f"{text} {pick}"
    return f"{text}. {pick}"
