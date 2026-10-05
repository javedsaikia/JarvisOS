"""Calling via the tel: URL scheme — macOS hands this to a paired iPhone
over Continuity (Handoff), same as any tel: link clicked in Safari. No
telephony API, no dependency: `open tel:<number>` is the entire mechanism,
same `open <uri>` pattern already proven for Spotify deep links
(bella/tools/spotify.py). Never placed without confirmation — see
cli.handle_call, which is the only caller and always confirms first.
"""
import re
import subprocess
from pathlib import Path

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


# Continuity calling used to go through FaceTime. On macOS 26 (Tahoe)
# the Phone app owns cellular calls; FaceTime Settings no longer even
# shows "Calls from iPhone". Opening FaceTime with tel: then reports
# the iPhone as unconfigured even when Phone > Settings > Calls is on.
# Naming the app explicitly: a bare `open tel:` can go to Teams.


def call_app() -> str:
    if Path("/System/Applications/Phone.app").exists():
        return "Phone"
    return "FaceTime"


# "call that place", "call them" and friends refer to whatever was just
# being discussed rather than naming anything to look up.
_REFERENTIAL_RE = re.compile(
    r"^(?:them|it|that|this|the)?\s*(?:place|one|restaurant|hotel|cafe|number)?$",
    re.IGNORECASE,
)
_CALL_VERB_RE = re.compile(
    r"\b(?:call|ring|dial|phone)\s+(?:up\s+)?(?:the\s+)?",
    re.IGNORECASE,
)
# Everything after the place name — reservation intent, a second "call
# and…", politeness. Seen live: "call Nova, call and reserve a booking?"
# was treated as the whole remainder, which then failed the lookup.
_CALL_STOP_RE = re.compile(
    r"(?:,\s*|\s+)(?:and\b|to\b|for\b|so\b|then\b|please\b|now\b|call\b|"
    r"reserve\b|book\b|reservation\b|booking\b).*$",
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
    for match in _CALL_VERB_RE.finditer(text or ""):
        rest = (text or "")[match.end():]
        rest = _CALL_STOP_RE.sub("", rest)
        rest = _CALL_TRAIL_RE.sub("", rest).strip().strip("?.!,").strip()
        if not rest or _REFERENTIAL_RE.match(rest):
            continue
        first = rest.split()[0].lower()
        if first in {"and", "or"}:
            continue
        words = rest.split()
        return " ".join(words[:5])
    return None


def normalize_number(raw: str) -> str:
    """Strip a looked-up number down to tel:-safe characters. Apple Maps
    returns display-formatted values like "+91 376-2301839"."""
    return re.sub(r"[^\d+]", "", raw or "")


def _digits(number: str) -> str:
    return re.sub(r"\D", "", number or "")


def is_own_number(number: str, user_phone: str) -> bool:
    """True when `number` is the user's own iPhone, not a restaurant."""
    mine = _digits(user_phone)
    theirs = _digits(number)
    if not mine or not theirs:
        return False
    # India: compare last 10 digits so +91 / 0 / bare local all match.
    return theirs[-10:] == mine[-10:] if len(mine) >= 10 and len(theirs) >= 10 else theirs == mine


def facetime_available() -> bool:
    """True if the app that actually places the call is installed."""
    app = call_app()
    try:
        result = subprocess.run(
            ["osascript", "-e", f'id of application "{app}"'],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def iphone_calling_ready() -> bool:
    """True when this Mac can place a cellular call via the iPhone.

    On macOS 26 the Phone app owns this. Its Settings → Calls →
    Calls from iPhone can be on while the old FaceTime
    thumperCallingCapabilitiesStates key stays empty — that empty
    key is not a signal on Tahoe. If Phone.app is present, trust it.
    Older macOS still uses FaceTime + the thumper list.
    """
    if call_app() == "Phone":
        return True
    try:
        result = subprocess.run(
            [
                "defaults",
                "read",
                "com.apple.TelephonyUtilities",
                "thumperCallingCapabilitiesStates",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if result.returncode != 0:
        return False
    inner = (result.stdout or "").replace("(", " ").replace(")", " ").strip()
    return bool(inner)


def call_number(number: str, label: str | None = None) -> str:
    app = call_app()
    try:
        result = subprocess.run(
            ["open", "-a", app, f"tel:{number}"], capture_output=True, text=True, timeout=10
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        raise CallingError(f"Could not initiate the call: {e}") from e
    if result.returncode != 0:
        raise CallingError((result.stderr or "").strip() or f"Could not open {app}.")
    who = f"{label} on {number}" if label else number
    return (
        f"Calling {who} from your iPhone."
    )

# "the contact number AT the KFC" / "number FOR Domino's". extract_call_target
# above needs a call verb ("call X"), which a request for a number does not
# have — so asking for a number found no target at all and fell back to
# "whose number do you want?" even when the place was named plainly.
_NUMBER_TARGET_RE = re.compile(
    r"\b(?:numbers?|no\.?)\s*(?:of|for|at|to)\s+(?:the\s+)?(.+)$",
    re.IGNORECASE,
)


def extract_number_target(text: str) -> str | None:
    """The business whose number was asked for, or None."""
    match = _NUMBER_TARGET_RE.search(text or "")
    if not match:
        return None
    rest = _CALL_STOP_RE.sub("", match.group(1))
    rest = _CALL_TRAIL_RE.sub("", rest).strip().strip("?.!,").strip()
    if not rest or _REFERENTIAL_RE.match(rest):
        return None
    return " ".join(rest.split()[:5])
