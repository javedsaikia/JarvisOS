"""In-progress email draft. Voice follow-ups fill fields; nothing is
sent until they say send. Isolated so a later "yes" cannot fall through
to Spotify the way a pending play query did.
"""
from __future__ import annotations

import re

_state: dict | None = None
_last_inbox: list[dict] = []

# STT of Zapier, seen live as "JAPIR".
_NAME_ALIASES = {
    "japir": "zapier",
    "zapir": "zapier",
    "zappier": "zapier",
    "japer": "zapier",
}

_TO_RE = re.compile(
    r"\b(?:to|for)\s+([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})\b",
    re.IGNORECASE,
)
_BARE_EMAIL_RE = re.compile(r"\b([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})\b", re.IGNORECASE)
_SUBJECT_RE = re.compile(
    r"\b(?:subject(?:\s+line)?|title)\s*(?:is|as|should be|:)?\s+(.+)$",
    re.IGNORECASE,
)
_ABOUT_RE = re.compile(r"\b(?:it'?s |is )?about\s+(.+)$", re.IGNORECASE)
_SEND_RE = re.compile(r"\b(send (it|the e-?mail|that)|go ahead and send|mail it)\b", re.IGNORECASE)
_CANCEL_RE = re.compile(r"\b(cancel|never mind|forget)( the)? (e-?mail|draft|that)\b", re.IGNORECASE)
_NEW_DRAFT_RE = re.compile(
    r"\b((draft|write|compose)\b.{0,24}\b(e-?mail|gmail|mail)|reply to)\b",
    re.IGNORECASE,
)
_TOPIC_CHANGE_RE = re.compile(
    r"\b(weather|forecast|hotel|traffic|volume|where am i|"
    r"i want to go|drive to|play |call )\b",
    re.IGNORECASE,
)


def active() -> bool:
    return _state is not None


def start() -> dict:
    global _state
    _state = {"to": [], "subject": "", "body": "", "name": "", "gmail_id": ""}
    return _state


def cancel() -> None:
    global _state
    _state = None


def snapshot() -> dict:
    return dict(_state or {"to": [], "subject": "", "body": ""})


def wants_send(text: str) -> bool:
    return bool(_SEND_RE.search(text or ""))


def wants_cancel(text: str) -> bool:
    return bool(_CANCEL_RE.search(text or ""))


def is_new_draft(text: str) -> bool:
    return bool(_NEW_DRAFT_RE.search(text or ""))


def remember_inbox(rows: list[dict]) -> None:
    global _last_inbox
    _last_inbox = list(rows or [])


def extract_person_name(text: str) -> str:
    stripped = (text or "").strip()
    match = re.search(
        r"\b(?:it'?s|to|for|reply to)\s+"
        r"([A-Za-z][A-Za-z'’.-]*(?:\s+[A-Za-z][A-Za-z'’.-]*){0,2}?)"
        r"(?=\s+at\b|[.,?!]|$)",
        stripped,
        re.IGNORECASE,
    )
    if not match:
        return ""
    name = match.group(1).strip(" .,?!")
    if re.search(r"\b(email|gmail|mail|inbox|subject|draft)\b", name, re.I):
        return ""
    return name


def extract_say_body(text: str) -> str:
    match = re.search(r"\b(?:and )?say(?:ing)?\s+(.+)$", text or "", re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip().strip("\"'.").strip()


def wants_reply(text: str) -> bool:
    return bool(re.search(r"\breply to\b", text or "", re.IGNORECASE))


def wants_inbox(text: str) -> bool:
    return bool(re.search(r"\b(check|read|summar).{0,20}\b(inbox|e-?mail)", text or "", re.I))


def set_gmail_id(draft_id: str) -> None:
    if _state is not None:
        _state["gmail_id"] = draft_id or ""


def resolve_name(name: str) -> list[str]:
    """Match a spoken name to last inbox or a contact."""
    raw = (name or "").strip()
    if not raw:
        return []
    key = _NAME_ALIASES.get(raw.lower(), raw)
    needle = key.lower()
    for row in _last_inbox:
        blob = f"{row.get('sender', '')} {row.get('subject', '')}".lower()
        if needle in blob or raw.lower() in blob:
            addr = _email_from_sender(row.get("sender") or "")
            return [addr] if addr else []
    from max import composio_client
    return composio_client.search_person(key)


def _email_from_sender(sender: str) -> str:
    match = _BARE_EMAIL_RE.search(sender or "")
    return match.group(1) if match else ""


def is_topic_change(text: str) -> bool:
    t = text or ""
    if _NEW_DRAFT_RE.search(t) or _SEND_RE.search(t) or _BARE_EMAIL_RE.search(t):
        return False
    return bool(_TOPIC_CHANGE_RE.search(t))


def missing() -> str | None:
    if not _state:
        return "to"
    if not _state.get("to"):
        return "to"
    if not (_state.get("subject") or "").strip():
        return "subject"
    if not (_state.get("body") or "").strip():
        return "body"
    return None


def apply_text(text: str) -> dict:
    """Merge whatever this utterance named into the open draft."""
    if _state is None:
        start()
    stripped = (text or "").strip()
    found = _TO_RE.search(stripped) or _BARE_EMAIL_RE.search(stripped)
    named = extract_person_name(stripped)
    filled_to = False
    if found:
        addr = found.group(1).strip()
        if addr.lower() not in {a.lower() for a in _state["to"]}:
            _state["to"].append(addr)
        filled_to = True
    elif named and not _state.get("to"):
        _state["name"] = named
        resolved = resolve_name(named)
        for addr in resolved:
            if addr.lower() not in {a.lower() for a in _state["to"]}:
                _state["to"].append(addr)
        filled_to = True

    said = extract_say_body(stripped)
    if said:
        _state["body"] = said

    sub = _SUBJECT_RE.search(stripped)
    if sub:
        subject = sub.group(1).strip().strip("\"'.").strip()
        subject = re.sub(r"^(?:an?|the)\s+", "", subject, flags=re.I)
        _state["subject"] = subject
    else:
        about = _ABOUT_RE.search(stripped)
        if about:
            _state["subject"] = about.group(1).strip().strip("\"'.").strip()

    # A short leftover like "GST" / "Subject High" after we already asked
    # for the subject — do not treat whole sentences as the subject.
    words = stripped.split()
    asked = missing()
    if asked == "subject" and not filled_to and 1 <= len(words) <= 4 and not found:
        if not re.search(r"\b(draft|write|compose|email|mail|send|subject)\b", stripped, re.I):
            _state["subject"] = stripped.strip("\"'.").strip()
        elif re.search(r"\b(gst|subject high)\b", stripped, re.I):
            _state["subject"] = re.sub(
                r".*\b(?:as|is|:)\s+", "", stripped, flags=re.I
            ).strip("\"'.").strip() or "GST"

    if asked == "body" and len(words) >= 4 and not wants_send(stripped):
        if not re.search(r"\b(draft|compose|subject|email address)\b", stripped, re.I):
            _state["body"] = stripped
    return snapshot()
