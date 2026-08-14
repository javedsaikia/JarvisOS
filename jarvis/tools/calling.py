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


# Continuity calling goes through FaceTime. Naming it explicitly rather
# than letting `open tel:` pick: this Mac has no default handler set for
# tel: and Microsoft Teams also registers for the scheme, so a bare
# `open tel:` could hand the call to Teams instead of the paired iPhone —
# which would silently violate the whole point of placing it from the
# phone. `open -a` removes the ambiguity.
_CALL_APP = "FaceTime"


# "call that place", "call them" and friends refer to whatever was just
# being discussed rather than naming anything to look up.
_REFERENTIAL_RE = re.compile(
    r"^(?:them|it|that|this|the)?\s*(?:place|one|restaurant|hotel|cafe|number)?$",
    re.IGNORECASE,
)
_CALL_LEAD_RE = re.compile(
    r"^\s*(?:hey\s+jarvis[,\s]*)?(?:please\s+)?(?:can|could|would)?\s*(?:you\s+)?"
    r"(?:please\s+)?(?:call|ring|dial|phone)\s+(?:up\s+)?",
    re.IGNORECASE,
)
_CALL_TRAIL_RE = re.compile(
    r"\s*(?:for me|please|now|right now|on my iphone|from my iphone)\s*$", re.IGNORECASE
)


def extract_call_target(text: str) -> str | None:
    """The thing to look up from "call <something>", or None if the phrase
    is referential ("call them") or names nothing.

    Deliberately separate from parse_phone_number: a spoken number is
    unambiguous and dialled directly, whereas a name has to be resolved to
    a number first and shown to the user before anything is dialled.
    """
    target = _CALL_LEAD_RE.sub("", (text or "").strip(), count=1)
    target = _CALL_TRAIL_RE.sub("", target).strip().strip("?.!,").strip()
    if not target or _REFERENTIAL_RE.match(target):
        return None
    return target


def is_referential_call(text: str) -> bool:
    """True for "call them" / "call that place" — a request to ring
    whatever was last talked about."""
    stripped = _CALL_LEAD_RE.sub("", (text or "").strip(), count=1)
    stripped = _CALL_TRAIL_RE.sub("", stripped).strip().strip("?.!,").strip()
    return bool(_REFERENTIAL_RE.match(stripped))


def normalize_number(raw: str) -> str:
    """Strip a looked-up number down to tel:-safe characters. Apple Maps
    returns display-formatted values like "+91 376-2301839"."""
    return re.sub(r"[^\d+]", "", raw or "")


def call_number(number: str, label: str | None = None) -> str:
    try:
        result = subprocess.run(
            ["open", "-a", _CALL_APP, f"tel:{number}"], capture_output=True, text=True, timeout=10
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        raise CallingError(f"Could not initiate the call: {e}") from e
    if result.returncode != 0:
        raise CallingError((result.stderr or "").strip() or f"Could not open {_CALL_APP}.")
    who = f"{label} on {number}" if label else number
    return f"Calling {who} — handing off to your iPhone via Continuity."
