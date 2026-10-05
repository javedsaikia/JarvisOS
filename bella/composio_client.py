"""Composio session for Bella.

Official SDK pattern:

    from composio import Composio
    composio = Composio()
    session = composio.create(user_id=existing_user_id)
    tools = session.tools()

The key is bella/.env. The user id is the one Composio already stored
connections under — discovered from connected accounts, not invented.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from bella import env, llm
from bella.config import load_config, update_config

SESSION_PATH = Path(__file__).parent / ".composio_session.json"
DEFAULT_USER_ID = "bella"

# Toolkits we will attach when they have an ACTIVE connected account.
# Only these are used. Expired / failed / dummy-key accounts stay off
# the session even if they still appear in the Composio project.
KNOWN_TOOLKITS = (
    "gmail", "github", "youtube", "googledrive", "cal",
    "openweather_api", "weathermap", "google_maps", "chatfai",
    # Connected later. A toolkit that is not attached here is invisible to
    # the session — its tools cannot even be searched for, let alone run —
    # so an account being ACTIVE in Composio is necessary but not enough.
    "googlecalendar", "googletasks", "linkedin", "reddit",
)
# No connected account required — Composio serves these itself.
NO_AUTH_TOOLKITS = ("weathermap",)
# Do not attach, even if Composio lists them.
BLOCKED_TOOLKITS = ("tripadvisor", "microsoft_teams")


class ComposioError(Exception):
    pass


def configured() -> bool:
    return bool(env.load_env().get("COMPOSIO_API_KEY", "").strip())


def _api_key() -> str:
    key = env.load_env().get("COMPOSIO_API_KEY", "").strip()
    if not key:
        raise ComposioError("COMPOSIO_API_KEY not set in bella/.env")
    return key


def _client():
    os.environ["COMPOSIO_API_KEY"] = _api_key()
    from composio import Composio

    return Composio(api_key=_api_key())


# Which accounts are connected changes when the user authorises something
# in the Composio dashboard — minutes apart at the fastest, not seconds.
# It was being fetched over HTTP on every single request: get_session()
# asks twice (user_id() and active_accounts() each call list_accounts()),
# and measured live that pair cost 2.62s BEFORE any actual work started.
# Every Composio-backed answer paid it. Cached for the process, with a
# TTL so a newly connected account still appears without a restart.
_ACCOUNTS_TTL_SECONDS = 300
_accounts_cache: tuple[float, list[dict]] | None = None
_accounts_lock = threading.Lock()


def refresh_accounts() -> None:
    """Drop the cache — call after connecting an account."""
    global _accounts_cache
    with _accounts_lock:
        _accounts_cache = None


def list_accounts(force: bool = False) -> list[dict]:
    """Every connected account on this project, no user-id filter."""
    global _accounts_cache
    if not force:
        with _accounts_lock:
            cached = _accounts_cache
        if cached and (time.monotonic() - cached[0]) < _ACCOUNTS_TTL_SECONDS:
            return cached[1]
    items = _list_accounts_uncached()
    with _accounts_lock:
        _accounts_cache = (time.monotonic(), items)
    return items


def _list_accounts_uncached() -> list[dict]:
    client = _client()
    resp = client.connected_accounts.list(limit=50)
    items = []
    for a in resp.items or []:
        items.append(
            {
                "id": a.id,
                "toolkit": a.toolkit.slug,
                "status": a.status,
                "user_id": a.user_id,
            }
        )
    return items


def active_accounts() -> dict[str, str]:
    """toolkit slug -> connected account id, ACTIVE only."""
    out: dict[str, str] = {}
    for a in list_accounts():
        if a["status"] == "ACTIVE" and a["toolkit"] not in out:
            out[a["toolkit"]] = a["id"]
    return out


def user_id(cfg: dict | None = None) -> str:
    """Prefer the user id that already owns connections."""
    cfg = cfg if cfg is not None else load_config()
    configured_id = (cfg.get("composio_user_id") or "").strip()
    accounts = list_accounts()
    active_uids = [a["user_id"] for a in accounts if a["status"] == "ACTIVE" and a.get("user_id")]
    if active_uids:
        # If config still has the placeholder, lock onto the real one.
        if not configured_id or configured_id == DEFAULT_USER_ID:
            discovered = active_uids[0]
            update_config(composio_user_id=discovered)
            return discovered
        if configured_id in active_uids:
            return configured_id
        return active_uids[0]
    return configured_id or DEFAULT_USER_ID


def _load_session_record() -> dict:
    try:
        return json.loads(SESSION_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_session_record(record: dict) -> None:
    try:
        SESSION_PATH.write_text(json.dumps(record))
    except OSError:
        pass


# The live session object, not just its id. Rebuilding it from the id
# still costs a `composio.use()` round trip, and nothing about it changes
# between requests in one process.
_session_cache: tuple[dict, object] | None = None
_session_lock = threading.Lock()


def warm(cfg: dict | None = None) -> None:
    """Build the session ahead of the first request.

    Fire-and-forget at startup: the cost is real but it is the user's
    first question that should not be the one paying it.
    """
    try:
        get_session(cfg)
    except Exception as e:
        print(f"[composio] warm-up skipped ({type(e).__name__}: {e})", flush=True)
        return
    prefetch_inbox()


def get_session(cfg: dict | None = None):
    """Create or reuse a session bound to the authorized accounts."""
    uid = user_id(cfg)
    accounts = active_accounts()
    toolkits = [
        t for t in KNOWN_TOOLKITS
        if t not in BLOCKED_TOOLKITS
        and (t in accounts or t in NO_AUTH_TOOLKITS)
    ]
    fingerprint = {"user_id": uid, "toolkits": toolkits, "accounts": accounts}

    with _session_lock:
        cached = _session_cache
    if cached and cached[0] == fingerprint:
        return cached[1]

    composio = _client()
    existing = _load_session_record()
    if (
        existing.get("session_id")
        and existing.get("user_id") == uid
        and existing.get("accounts") == accounts
        and existing.get("toolkits") == toolkits
    ):
        try:
            session = composio.use(existing["session_id"])
            with _session_lock:
                globals()["_session_cache"] = (fingerprint, session)
            return session
        except Exception:
            pass

    kwargs: dict = {
        "user_id": uid,
        "manage_connections": False,
        "sandbox": {"enable": False},
    }
    if toolkits:
        kwargs["toolkits"] = toolkits
        connected = {t: accounts[t] for t in toolkits if t in accounts}
        if connected:
            kwargs["connected_accounts"] = connected
    session = composio.create(**kwargs)
    _save_session_record({**fingerprint, "session_id": session.session_id})
    with _session_lock:
        globals()["_session_cache"] = (fingerprint, session)
    return session


def tools(cfg: dict | None = None):
    return get_session(cfg).tools()


def _result_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif hasattr(result, "data") or hasattr(result, "error"):
        data = {
            "data": getattr(result, "data", None),
            "error": getattr(result, "error", None),
            "successful": getattr(result, "successful", None),
        }
    else:
        data = result
    try:
        return json.dumps(data, default=str)[:8000]
    except TypeError:
        return str(data)[:8000]


# GMAIL_FETCH_EMAILS is the slowest call Bella makes — measured 7.64s, all
# of it upstream. A short TTL is the difference between "check my email"
# being usable and being something you stop asking for. Kept deliberately
# short: mail arriving is the whole point, so this trades at most a
# minute of staleness, and a prefetch means the common case is served
# from memory anyway.
_INBOX_TTL_SECONDS = 60
_inbox_cache: tuple[float, int, list[dict]] | None = None
_inbox_lock = threading.Lock()


def prefetch_inbox() -> None:
    """Pull the inbox in the background so the first ask is not the wait."""
    try:
        fetch_inbox()
    except Exception as e:
        print(f"[composio] inbox prefetch skipped ({type(e).__name__}: {e})", flush=True)


def fetch_inbox(max_results: int = 5, force: bool = False) -> list[dict]:
    """Recent inbox rows. Deterministic — no model, so it cannot invent."""
    global _inbox_cache
    if not force:
        with _inbox_lock:
            cached = _inbox_cache
        if (cached and cached[1] >= max_results
                and (time.monotonic() - cached[0]) < _INBOX_TTL_SECONDS):
            return cached[2][:max_results]
    rows = _fetch_inbox_uncached(max_results)
    with _inbox_lock:
        _inbox_cache = (time.monotonic(), max_results, rows)
    return rows


def _fetch_inbox_uncached(max_results: int = 5) -> list[dict]:
    session = get_session()
    try:
        result = session.execute("GMAIL_FETCH_EMAILS", arguments={"max_results": max_results})
    except Exception as e:
        # The account list is cached for 5 minutes; if a call fails, that
        # cache is the first thing that could be stale (revoked token,
        # reconnected account), so drop it and let the next try re-check.
        refresh_accounts()
        raise ComposioError(f"GMAIL_FETCH_EMAILS failed: {e}") from e
    if getattr(result, "error", None):
        raise ComposioError(str(result.error))
    data = getattr(result, "data", None)
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    if not isinstance(data, dict):
        return []
    inner = data.get("data") if isinstance(data.get("data"), dict) else data
    rows = []
    for m in (inner.get("messages") or [])[:max_results]:
        if not isinstance(m, dict):
            continue
        rows.append({
            # sender is kept because email_compose.resolve_name matches a
            # spoken name against it ("reply to Zapier"), but it is
            # deliberately NOT spoken — see inbox_speech.
            "sender": (m.get("sender") or m.get("from") or "unknown").strip(),
            "subject": (m.get("subject") or "(no subject)").strip(),
            "body": clean_body(m.get("messageText") or ""),
            "id": m.get("messageId") or m.get("id") or "",
            "thread_id": m.get("threadId") or m.get("thread_id") or "",
        })
    return rows


# Marketing mail is mostly navigation furniture and tracking links. None of
# it survives to the model, so a summary can't be built out of "View this
# email in your browser".
_URL_RE = re.compile(r"https?://\S+|\bwww\.\S+")
_INVISIBLE_RE = re.compile(r"[​-‏⁠﻿\xad]")
_BOILERPLATE_RE = re.compile(
    r"(?im)^\s*(?:view (?:this|it) (?:email )?in (?:your )?browser|"
    r"click here to (?:see|view) the web version|click here to unsubscribe|"
    r"to view this email|having trouble viewing|add us to your address book|"
    r"[-=_*]{6,}|"
    r"unsubscribe|manage preferences|privacy policy|terms of use|"
    r"if you (?:no longer|do not) wish to receive|sent to you because|"
    r"copyright \S+|all rights reserved).*$"
)
_BODY_CHARS = 700


def clean_body(text: str) -> str:
    """Readable prose from a raw email body, capped for prompt cost."""
    t = _INVISIBLE_RE.sub("", text or "")
    t = _URL_RE.sub(" ", t)
    t = _BOILERPLATE_RE.sub(" ", t)
    t = re.sub(r"[ \t\r ]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()[:_BODY_CHARS]


# Marketing subjects are full of emoji, flags and decorative bars. Bulbul
# either spells them out or stalls on them; either way they are noise in
# something being read aloud.
_UNSPEAKABLE_RE = re.compile(
    "[\U0001F000-\U0001FAFF←-⯿☀-➿️‍|]+"
)


def _speakable(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _UNSPEAKABLE_RE.sub(" ", text or "")).strip(" -–—:")


_ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth")
_COUNT_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}


def inbox_speech(rows: list[dict]) -> str:
    """Subject line, then what the email is about. Spoken, so: no senders,
    no addresses, no numbering punctuation.

    The sender used to lead every item, which meant a spoken inbox was a
    minute of "HDFC MF, angle noresponse at hdfcfund dot com" before
    reaching anything the user actually wanted — and the raw address is
    both the slowest thing Bulbul has to say (measured: 104 chars of
    address-heavy text took 8.4s to synthesize, against 3.2s for 63 chars
    of prose) and the least useful. Who sent it is on screen; what it says
    is what you asked for.
    """
    if not rows:
        return "The inbox is empty."
    shown = rows[:5]
    bits = []
    for i, row in enumerate(shown):
        subject = _speakable(row.get("subject") or "") or "no subject"
        summary = _speakable(row.get("summary") or "").rstrip(".")
        # Subjects routinely end in "?" or "!" already; appending a full
        # stop regardless produced "Want more referrals?." out loud.
        bit = f"{_ORDINALS[i]}, {subject}" + ("" if subject[-1:] in ".?!" else ".")
        if summary:
            bit += f" {summary[0].upper()}{summary[1:]}."
        bits.append(bit)
    count = _COUNT_WORDS.get(len(shown), str(len(shown)))
    head = "One email." if len(shown) == 1 else f"{count} emails."
    return head + " " + " ".join(bits)


def create_reminder(subject: str, when, has_time: bool) -> str:
    """Put a reminder in Google Calendar (timed) or Google Tasks (dated).

    Slugs verified against the live session. Raises rather than returning
    a cheerful string on failure — the caller speaks the failure, and this
    is exactly the path where a made-up success does the most damage.
    """
    if not (subject or "").strip():
        raise ComposioError("A reminder needs something to be about.")
    session = get_session()
    if has_time:
        args = {
            "calendarId": "primary",
            "summary": subject,
            "start_datetime": when.strftime("%Y-%m-%dT%H:%M:%S"),
            "event_duration_hour": 0,
            "event_duration_minutes": 30,
            "timezone": _local_timezone(),
        }
        result = _execute(session, "GOOGLECALENDAR_CREATE_EVENT", args)
        return f"calendar:{result}"
    args = {
        "tasklist_id": _default_tasklist(session),
        "title": subject,
        "due": when.strftime("%Y-%m-%dT00:00:00.000Z"),
    }
    result = _execute(session, "GOOGLETASKS_INSERT_TASK", args)
    return f"tasks:{result}"


def _local_timezone() -> str:
    """This Mac's IANA zone name, e.g. "Asia/Kolkata".

    Not the abbreviation: Google rejects "IST" outright ("Timezone
    abbreviations like 'EST', 'PST', 'IST' are not valid IANA
    identifiers"), and %Z is all datetime offers on macOS. The symlink
    /etc/localtime points at the zoneinfo file, whose path IS the name.
    """
    tz = os.environ.get("TZ", "").strip()
    if "/" in tz:
        return tz
    try:
        target = os.path.realpath("/etc/localtime")
        parts = target.split("/zoneinfo/", 1)
        if len(parts) == 2 and parts[1]:
            return parts[1]
    except OSError:
        pass
    return "UTC"


def _default_tasklist(session) -> str:
    result = session.execute("GOOGLETASKS_LIST_TASK_LISTS", arguments={"maxResults": 5})
    data = getattr(result, "data", None)
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    items = (data or {}).get("items") or []
    for item in items:
        if isinstance(item, dict) and item.get("id"):
            return item["id"]
    raise ComposioError("No Google Tasks list to add to.")


def _execute(session, slug: str, args: dict) -> str:
    try:
        result = session.execute(slug, arguments=args)
    except Exception as e:
        raise ComposioError(f"{slug} failed: {e}") from e
    err = getattr(result, "error", None)
    if err:
        raise ComposioError(str(err))
    return slug


def calendar_window(user_text: str, now: datetime | None = None):
    """(start, end, label) for what the user actually asked about.

    "tomorrow" means tomorrow, not "the next 48 hours starting now" —
    that window included today's events and then announced them as
    tomorrow's, which is the kind of confidently mislabelled answer this
    project keeps having to stamp out.
    """
    now = now or datetime.now().astimezone()
    lowered = (user_text or "").lower()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if "tomorrow" in lowered:
        start = midnight + timedelta(days=1)
        return start, start + timedelta(days=1), "tomorrow"
    if "week" in lowered:
        return now, midnight + timedelta(days=8), "this week"
    if "today" in lowered or "day" in lowered:
        return now, midnight + timedelta(days=1), "today"
    return now, midnight + timedelta(days=1), "today"


def list_calendar_events(user_text: str = "") -> str:
    """Upcoming events, spoken. Google Calendar via Composio.

    Faster AND more correct than the AppleScript path it replaces:
    measured 1.04s against 5.58s, and Calendar.app was answering "No
    events found" for a day that had real entries in Google Calendar,
    because the account is synced there and not into a local calendar
    Calendar.app was scanning.
    """
    session = get_session()
    start, end, label = calendar_window(user_text)
    try:
        result = session.execute("GOOGLECALENDAR_EVENTS_LIST", arguments={
            "calendarId": "primary",
            "maxResults": 10,
            "singleEvents": True,
            "orderBy": "startTime",
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
        })
    except Exception as e:
        refresh_accounts()
        raise ComposioError(f"calendar list failed: {e}") from e
    if getattr(result, "error", None):
        raise ComposioError(str(result.error))
    data = getattr(result, "data", result) or {}
    items = data.get("items") or []
    return _events_speech(items, label)


def _events_speech(items: list[dict], window: str) -> str:
    if not items:
        return f"Nothing on your calendar {window}."
    bits = []
    for ev in items[:5]:
        title = _speakable(ev.get("summary") or "Untitled")
        start = (ev.get("start") or {})
        raw = start.get("dateTime") or start.get("date") or ""
        when = ""
        try:
            if "T" in raw:
                when = datetime.fromisoformat(raw).strftime("%a %I:%M %p").replace(" 0", " ")
            elif raw:
                when = datetime.fromisoformat(raw).strftime("%a") + ", all day"
        except ValueError:
            when = ""
        bits.append(f"{title} {when}".strip() if when else title)
    lead = f"{len(items)} on your calendar {window}." if len(items) > 1 else f"One thing {window}."
    return lead + " " + "; ".join(bits) + "."


def search_person(name: str) -> list[str]:
    """Contact emails for a spoken name. Empty if none."""
    q = (name or "").strip()
    if not q:
        return []
    session = get_session()
    try:
        result = session.execute("GMAIL_SEARCH_PEOPLE", arguments={"query": q})
    except Exception as e:
        print(f"[composio] people search failed: {e}", flush=True)
        return []
    if getattr(result, "error", None):
        return []
    data = getattr(result, "data", None)
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    if not isinstance(data, dict):
        return []
    emails: list[str] = []
    for item in data.get("results") or []:
        person = (item or {}).get("person") or item or {}
        for entry in person.get("emailAddresses") or []:
            value = (entry.get("value") or "").strip()
            if value and value.lower() not in {e.lower() for e in emails}:
                emails.append(value)
    return emails


def create_email_draft(to: list[str], subject: str, body: str) -> dict:
    """GMAIL_CREATE_EMAIL_DRAFT — Composio skill for 'draft an email'."""
    if not to:
        raise ComposioError("I need an email address to draft to.")
    session = get_session()
    try:
        result = session.execute(
            "GMAIL_CREATE_EMAIL_DRAFT",
            arguments={
                "recipient_email": to[0],
                "subject": subject or "",
                "body": body or "",
            },
        )
    except Exception as e:
        raise ComposioError(f"GMAIL_CREATE_EMAIL_DRAFT failed: {e}") from e
    if getattr(result, "error", None):
        raise ComposioError(str(result.error))
    data = getattr(result, "data", None)
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    if not isinstance(data, dict):
        return {"id": "", "url": ""}
    return {
        "id": data.get("id") or "",
        "url": data.get("display_url") or "",
    }


def send_gmail_draft(draft_id: str) -> None:
    session = get_session()
    try:
        result = session.execute("GMAIL_SEND_DRAFT", arguments={"draft_id": draft_id})
    except Exception as e:
        raise ComposioError(f"GMAIL_SEND_DRAFT failed: {e}") from e
    if getattr(result, "error", None):
        raise ComposioError(str(result.error))


READ_TOOLS = {
    "email": ("GMAIL_FETCH_EMAILS", {"max_results": 5}),
    "github": ("GITHUB_LIST_REPOSITORIES_FOR_THE_AUTHENTICATED_USER", {"per_page": 8, "sort": "updated"}),
    "youtube": ("YOUTUBE_LIST_CHANNELS", {}),
    "googledrive": ("GOOGLEDRIVE_LIST_FILES", {"pageSize": 8}),
    "cal": ("CAL_FETCH_ALL_BOOKINGS", {}),
    # Slugs and arguments verified against the live session, not guessed —
    # Composio's own COMPOSIO_SEARCH_TOOLS only lists tools for toolkits
    # already attached in KNOWN_TOOLKITS, so an invented slug fails at the
    # worst moment rather than at wiring time.
    "googlecalendar": ("GOOGLECALENDAR_EVENTS_LIST",
                       {"calendarId": "primary", "maxResults": 8,
                        "singleEvents": True, "orderBy": "startTime"}),
    "googletasks": ("GOOGLETASKS_LIST_TASK_LISTS", {"maxResults": 8}),
    "linkedin": ("LINKEDIN_GET_MY_INFO", {}),
    # Reddit needs a subreddit; run() overrides this with whichever one
    # they named, and falls back to this when they just say "reddit".
    "reddit": ("REDDIT_GET_R_TOP", {"subreddit": "technology", "limit": 6, "t": "day"}),
}

# Which subreddit "what's on reddit about X" means.
_SUBREDDIT_RE = re.compile(
    r"\b(?:r/|subreddit\s+|reddit\s+(?:about|on|for)\s+)([A-Za-z][A-Za-z0-9_]{2,20})\b",
    re.IGNORECASE,
)


def execute_and_say(user_text: str, cfg: dict, tool_slug: str, arguments: dict | None = None) -> str:
    """Run one known tool, then phrase the result. No giant tool schemas."""
    session = get_session(cfg)
    try:
        result = session.execute(tool_slug, arguments=arguments or {})
    except Exception as e:
        raise ComposioError(f"{tool_slug} failed: {e}") from e
    blob = _result_text(result)
    fallback_speech = _plain_summary(tool_slug, result)
    entry = llm.resolve(cfg, "chat")
    messages = [
        {
            "role": "system",
            "content": (
                "You are Bella. Turn this tool result into one or two short spoken "
                "sentences. No markdown, no raw JSON. If it failed, say so."
            ),
        },
        {"role": "user", "content": f"Request: {user_text}\n\nResult:\n{blob[:6000]}"},
    ]
    try:
        return llm.chat(messages, entry, cfg, options={"num_predict": 100})
    except llm.LLMError as e:
        # The assigned model is unreachable — measured live when Groq
        # decommissioned the configured Llama models and every call came
        # back 404. Try the local one before falling back to a canned
        # line: a real summary from a slower model beats no summary.
        print(f"[composio] {entry['label']} failed ({e}) — local model", flush=True)
        try:
            local = llm.resolve({**cfg, "model_roles": {}}, "chat")
            return llm.chat(messages, local, cfg, options={"num_predict": 100})
        except llm.LLMError:
            return fallback_speech


def _plain_summary(tool_slug: str, result) -> str:
    data = getattr(result, "data", None) or {}
    if not isinstance(data, dict):
        return _result_text(result)[:500]
    inner = data.get("data") if isinstance(data.get("data"), dict) else data
    if tool_slug.startswith("GMAIL_") and inner.get("messages"):
        lines = []
        for m in inner["messages"][:5]:
            sender = m.get("sender") or "unknown sender"
            subject = m.get("subject") or "(no subject)"
            lines.append(f"{sender}: {subject}")
        if not lines:
            return "The inbox is empty."
        return "Latest emails. " + " ".join(f"{i}. {line}." for i, line in enumerate(lines, 1))
    # Never the raw payload, and never the tool slug. This line is SPOKEN
    # and it is LOGGED, and both did real damage: Bella read 58 seconds of
    # Reddit JSON aloud, and the logged "REDDIT_GET_R_TOP returned. {...}"
    # then taught the chat model the format — it went on to invent
    # "LINKEDIN_CHECK_POSTS" and "TRIP_ADVISOR_SEARCH" complete with
    # fabricated posts and hotel ratings. A summariser outage must degrade
    # to saying less, never to dumping the payload.
    items = _count_items(inner)
    what = _friendly_name(tool_slug)
    if items:
        return f"I got {items} back from {what}, but couldn't summarize them just now."
    return f"{what} answered, but I couldn't put it into words just now."


def _count_items(inner: dict) -> str:
    """"3 items" for whatever list-shaped thing came back, else ""."""
    if not isinstance(inner, dict):
        return ""
    for key in ("items", "messages", "children", "results", "places", "events", "posts"):
        value = inner.get(key)
        if isinstance(value, list) and value:
            return f"{len(value)} item{'s' if len(value) != 1 else ''}"
    nested = inner.get("data")
    if isinstance(nested, dict):
        return _count_items(nested)
    return ""


# Toolkit slug -> what a person calls it. Never expose the slug itself.
_FRIENDLY = {
    "GMAIL": "Gmail", "GITHUB": "GitHub", "YOUTUBE": "YouTube",
    "GOOGLEDRIVE": "Google Drive", "GOOGLECALENDAR": "Google Calendar",
    "GOOGLETASKS": "Google Tasks", "LINKEDIN": "LinkedIn",
    "REDDIT": "Reddit", "CAL": "Cal.com", "CHATFAI": "ChatFAI",
    "GOOGLE": "Google Maps",
}


def _friendly_name(tool_slug: str) -> str:
    head = (tool_slug or "").split("_", 1)[0].upper()
    return _FRIENDLY.get(head, "that service")


def run(user_text: str, cfg: dict, domain: str = "email", action: str = "read") -> str:
    if domain == "chatfai":
        # NB: no local `import re` here. It used to be, and because Python
        # scopes a name for the whole function, it made `re` local to
        # run() — so every other branch that used the module-level `re`
        # died with UnboundLocalError the moment it was reached.
        q = re.sub(r"\bchat\s*fai\b", " ", user_text or "", flags=re.I)
        q = re.sub(r"\b(search|find|look up|lookup|on|for|about)\b", " ", q, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip(" .?!")
        if len(q) < 2:
            q = "Sherlock Holmes"
        return execute_and_say(user_text, cfg, "CHATFAI_SEARCH_CHARACTERS", {"q": q})

    if action != "write":
        if domain not in READ_TOOLS:
            raise ComposioError(f"No Composio read path for {domain}.")
        slug, args = READ_TOOLS[domain]
        if domain == "reddit":
            named = _SUBREDDIT_RE.search(user_text or "")
            if named:
                args = {**args, "subreddit": named.group(1)}
        if domain == "googlecalendar":
            # Without timeMin the API lists from the beginning of the
            # calendar — measured, it read back an electricity bill from
            # 2013 and an insurance premium from 2015 as "your events".
            # "What's on my calendar" always means from now on.
            import datetime as _dt
            now = _dt.datetime.now(_dt.timezone.utc)
            args = {**args, "timeMin": now.isoformat().replace("+00:00", "Z")}
            if re.search(r"\b(week|next few days|coming days)\b", user_text or "", re.I):
                horizon = now + _dt.timedelta(days=7)
            elif re.search(r"\btomorrow\b", user_text or "", re.I):
                horizon = now + _dt.timedelta(days=2)
            else:
                horizon = now + _dt.timedelta(days=1)
            args["timeMax"] = horizon.isoformat().replace("+00:00", "Z")
        return execute_and_say(user_text, cfg, slug, args)

    if domain != "email":
        raise ComposioError("That write goes through a confirm + specific tool — say it more plainly.")

    entry = llm.resolve(cfg, "chat")
    extract = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Extract a Gmail send. Reply with ONLY JSON: "
                    '{"to":["email"],"subject":"...","body":"..."}. '
                    "If the user did not name a recipient, set to to []."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        entry,
        cfg,
    )
    try:
        start = extract.find("{")
        end = extract.rfind("}") + 1
        payload = json.loads(extract[start:end])
    except Exception as e:
        raise ComposioError(f"Could not parse that email: {e}") from e
    to = payload.get("to") or payload.get("recipient_email") or []
    if isinstance(to, str):
        to = [to]
    if not to:
        raise ComposioError("I need an email address to send to.")
    draft = create_email_draft(
        to,
        payload.get("subject") or "",
        payload.get("body") or "",
    )
    return (
        f"Draft saved in Gmail for {', '.join(to)}"
        + (f', subject "{payload.get("subject")}"' if payload.get("subject") else "")
        + ". It is not sent. Say send it if that's right."
        + (f" {draft['url']}" if draft.get("url") else "")
    )


