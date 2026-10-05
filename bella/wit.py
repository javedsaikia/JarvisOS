"""One house voice for every backend and every canned tool line.

Chat models read VOICE. Deterministic tools (weather, location, nearby)
cannot invent jokes, so they append a rotating one-liner after the fact.
The fact always comes first — TTS users should hear Jorhat before the wink.
"""


VOICE = (
    "Talk like a sharp adult friend, not a professor and not a nanny. "
    "One or two short sentences, unless they asked for a joke — then tell "
    "the whole joke. Dry, filthy if they asked for filthy. "
    "They are an adult. If they want an adult, dirty, or sexy joke, give a "
    "real one. Do not refuse, do not swap in a children's joke, do not "
    "moralize. Never lecture, never walk through every step, never pad. "
    "If they asked for a page, file, or piece of work: summarize, then ask "
    "what to do next. If they asked you to do something: do it and confirm "
    "in one line. Wait for them before the next action. "
    "Never involve anyone under 18 in sexual content."
)


def spoken_fail(kind: str = "") -> str:
    """What the user hears when something breaks.

    Never include exception text, error domains, or status codes.
    Those go to the process log only.
    """
    if kind in ("lookup", "maps", "nearby"):
        return season(
            "I couldn't find that. Say the name again, or give me the number.",
            "That one didn't come up.",
        )
    if kind == "call":
        return season(
            "I wasn't able to place the call. Try again in a moment.",
            "The call didn't go through.",
        )
    if kind == "location":
        return season(
            "I wasn't able to get your location just now. Try again in a moment.",
            "Give me another second and ask again.",
        )
    if kind == "weather":
        return season(
            "I wasn't able to get the weather just now. Try again in a moment.",
            "Ask me once more in a second.",
        )
    if kind == "screen":
        return season(
            "I couldn't read the screen just now. Try again.",
            "That didn't come through.",
        )
    if kind == "news":
        return season(
            "I wasn't able to get the news just now. Try again in a moment.",
            "Ask me once more in a second.",
        )
    return season(
        "I wasn't able to do that. Try saying it another way.",
        "That didn't work. One more time?",
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
