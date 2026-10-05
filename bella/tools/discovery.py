"""Spoken wrappers over bella/discovery_client.py.

Read-only. After listing, Bella asks which option to open. Opening the
official page is confirmed; checkout stays with the user.
"""
from __future__ import annotations

import re
import subprocess
from typing import Any

from bella import discovery_client as dc
from bella import location_client as lc
from bella.tools import location

_PENDING: dict[str, Any] | None = None

_OPEN_RE = re.compile(
    r"\b(?:open|book|take|choose|pick|go with|that one|this one|"
    r"the (?:first|second|third|1st|2nd|3rd)|"
    r"(?:number|option|choice) ?[123]|#?[123])\b",
    re.IGNORECASE,
)
_FROM_TO_RE = re.compile(
    r"\bfrom\s+([A-Za-z][A-Za-z .'-]{1,32}?)\s+to\s+([A-Za-z][A-Za-z .'-]{1,32})",
    re.IGNORECASE,
)
_ON_DATE_RE = re.compile(
    r"\bon\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}|\d{4}-\d{2}-\d{2}|"
    r"[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
    r"tomorrow|today|tonight)\b",
    re.IGNORECASE,
)
_THEATER_RE = re.compile(
    r"\b(?:at|in)\s+((?:pvr|inox|cinepolis|carnival|miraj)[^,.?!]{0,40}|"
    r"[A-Z][A-Za-z0-9 .'-]{2,40}(?:cinemas?|multiplex|theatre|theater))\b",
    re.IGNORECASE,
)
def has_pending() -> bool:
    return bool(_PENDING and (_PENDING.get("items") or _PENDING.get("listing_url")))


def clear_pending() -> None:
    global _PENDING
    _PENDING = None


def is_open_intent(text: str) -> bool:
    return bool(_OPEN_RE.search(text or ""))


def _remember(payload: dict) -> None:
    global _PENDING
    _PENDING = payload


def _city_from_text(text: str) -> str:
    named = location.extract_weather_place(text) or ""
    if named:
        return named
    try:
        loc = lc.get_location()
    except lc.LocationError:
        return ""
    return (loc.get("city") or "").strip()


def _choice_index(text: str) -> int | None:
    raw = (text or "").lower()
    if re.search(r"\b(?:first|1st|number one|option 1|#1)\b", raw) or re.search(
        r"\b(?:number|option|choice) ?1\b", raw
    ):
        return 0
    if re.search(r"\b(?:second|2nd|number two|option 2|#2)\b", raw):
        return 1
    if re.search(r"\b(?:third|3rd|number three|option 3|#3)\b", raw):
        return 2
    return None


def _speak(payload: dict) -> str:
    items = payload.get("items") or []
    listing = payload.get("listing_url") or ""
    kind = payload.get("kind") or "listing"
    disclaimer = "Live web data — times and prices can change."
    if not items:
        where = payload.get("city") or payload.get("city_or_area") or payload.get("query") or "there"
        return (
            f"{disclaimer} I couldn't read a live board just now. "
            f"The listing is {listing or where}. Say open it and I'll "
            "open the page — you finish any booking yourself."
        )
    bits = []
    for i, item in enumerate(items[:4], 1):
        title = (item.get("title") or "listing").strip()
        extra = []
        if item.get("price"):
            extra.append(str(item["price"]))
        times = item.get("times") or []
        if times:
            extra.append("/".join(times[:3]))
        tail = f" ({', '.join(extra)})" if extra else ""
        bits.append(f"{i}) {title}{tail}")
    joined = "; ".join(bits)
    return (
        f"{disclaimer} {kind}: {joined}. "
        "Which number should I open? I don't pay or check out."
    )


def get_movie_showtimes(city: str = "", theater_or_query: str = "") -> dict:
    payload = dc.get_movie_showtimes(city, theater_or_query)
    _remember(payload)
    return payload


def search_hotels(city_or_area: str, constraints: str = "") -> dict:
    payload = dc.search_hotels(city_or_area, constraints)
    _remember(payload)
    return payload


def search_flights(
    origin: str,
    dest: str,
    date: str = "",
    return_date: str = "",
    constraints: str = "",
) -> dict:
    payload = dc.search_flights(origin, dest, date, return_date, constraints)
    _remember(payload)
    return payload


def describe_movies(user_text: str) -> str:
    city = _city_from_text(user_text)
    theater = ""
    match = _THEATER_RE.search(user_text or "")
    if match:
        theater = match.group(1).strip(" .?!,")
    return _speak(get_movie_showtimes(city, theater))


def describe_hotels(user_text: str) -> str:
    city = _city_from_text(user_text)
    extra = ""
    cheap = re.search(r"\b(under|below|less than|cheap|budget)\s*([\d,]+)?", user_text or "", re.I)
    if cheap:
        extra = cheap.group(0)
    if not city:
        return "Which city or area should I search hotels in?"
    return _speak(search_hotels(city, extra))


def describe_flights(user_text: str) -> str:
    match = _FROM_TO_RE.search(user_text or "")
    if not match:
        return "From where to where, and on which date?"
    origin, dest = match.group(1).strip(" .?!,"), match.group(2).strip(" .,?!")
    dest = re.split(r"\s+on\s+", dest, maxsplit=1)[0].strip()
    date = ""
    dated = _ON_DATE_RE.search(user_text or "")
    if dated:
        date = dated.group(1).strip()
    return_date = ""
    ret = re.search(r"\breturn(?:ing)?\s+(?:on\s+)?([A-Za-z0-9 /.-]{3,20})", user_text or "", re.I)
    if ret:
        return_date = ret.group(1).strip(" .?!,")
    return _speak(search_flights(origin, dest, date, return_date))


def pick_open_target(user_text: str) -> tuple[str, str] | None:
    """(title, url) for the chosen listing, or None."""
    pending = _PENDING or {}
    items = pending.get("items") or []
    idx = _choice_index(user_text)
    if idx is not None and 0 <= idx < len(items):
        item = items[idx]
        url = item.get("url") or pending.get("listing_url") or ""
        return (item.get("title") or "that listing", url)
    listing = pending.get("listing_url") or ""
    if listing:
        return ("the listing page", listing)
    if items:
        item = items[0]
        return (item.get("title") or "that listing", item.get("url") or "")
    return None


def open_listing(url: str) -> str:
    if not dc.is_safe_listing_url(url):
        return "That's a checkout or login page. I only open listings."
    try:
        subprocess.run(["open", url], check=False, timeout=8)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"Couldn't open the page ({e}). Here's the link: {url}"
    clear_pending()
    return (
        f"Opened the listing. You finish any booking or payment yourself. {url}"
    )
