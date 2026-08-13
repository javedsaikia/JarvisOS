"""Calling via the tel: URL scheme — macOS hands this to a paired iPhone
over Continuity (Handoff), same as any tel: link clicked in Safari. No
telephony API, no dependency: `open tel:<number>` is the entire mechanism,
same `open <uri>` pattern already proven for Spotify deep links
(jarvis/tools/spotify.py). Never placed without confirmation — see
cli.handle_call, which is the only caller and always confirms first.
"""
import re
import subprocess

# A phone-number-shaped run: optional leading +, then digits/space/dash/
# parens/dots, at least 7 chars long. Loose match first, then a strict
# digit-count check on the cleaned result below — two-step so "call 555-
# 123-4567" and "dial +1 (555) 123-4567" both work without also matching
# short unrelated numbers (a house number, a year) that happen to appear
# in the same sentence.
_PHONE_RE = re.compile(r"\+?[\d\s\-().]{7,}\d")


class CallingError(Exception):
    pass


def parse_phone_number(text: str) -> str | None:
    """Extracts a phone number from dictated/typed text, cleaned to just
    +/digits (tel:-ready). Returns None if nothing plausible is found —
    callers must treat that as "couldn't tell what number to call," never
    guess or fabricate one.
    """
    match = _PHONE_RE.search(text)
    if not match:
        return None
    cleaned = re.sub(r"[^\d+]", "", match.group(0))
    if sum(c.isdigit() for c in cleaned) < 7:
        return None
    return cleaned


def call_number(number: str) -> str:
    try:
        subprocess.run(["open", f"tel:{number}"], capture_output=True, timeout=10)
    except (subprocess.TimeoutExpired, OSError) as e:
        raise CallingError(f"Could not initiate the call: {e}") from e
    return f"Calling {number} — handing off to your iPhone via Continuity."
