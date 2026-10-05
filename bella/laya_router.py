"""Optional Laya System-1 router fallback for Bella.

Sits between regex routing (bella/router.py) and generative LLM fallback.
Disabled by default — set laya_enabled=true in config.json.

Requires: pip install laya (pulls torch ~130MB+). Use scripts/laya_poc/.venv
for POC; production install is opt-in.
"""
from __future__ import annotations

import re
from typing import Any

# Coarse categories (≤8 options — within Laya choice budget).
COARSE_CATEGORIES: dict[str, str] = {
    "calendar": "schedule, events, agenda, meetings, reminders, notes about dates",
    "comms": "email, gmail, inbox, send mail, phone calls, compose drafts",
    "media": "spotify, music, play/pause, news, movies, jokes, volume",
    "location": "weather, maps, nearby places, hotels, flights, where am I",
    "screen": "read screen, what's on screen, how to click, guide, desktop click",
    "system": "run shell command, read/list files, browser, capabilities, github, drive",
    "integrations": "youtube, linkedin, reddit, google calendar, cal.com, chatfai",
    "chat": "general conversation, greetings, chit-chat — no tool action",
}

# Fine tools per coarse category (each ≤15 options).
FINE_TOOLS: dict[str, dict[str, str]] = {
    "calendar": {
        "calendar/read": "what's on calendar, upcoming events, am I free",
        "calendar/write": "add/schedule/book meeting or event",
        "calendar_undo": "undo/remove last calendar entry",
        "notes/read": "search or list notes",
        "notes/write": "take a note, jot down",
        "reminder/auto": "remind me at a time",
        "googlecalendar/auto": "google calendar specifically",
        "googletasks/auto": "google tasks",
        "cal/auto": "cal.com booking",
    },
    "comms": {
        "email/read": "check inbox, read gmail",
        "email/write": "send, compose, reply to email",
        "compose/auto": "write draft email/letter/prompt for paste",
        "call/auto": "call someone by name",
        "call/number": "call a phone number",
    },
    "media": {
        "spotify/play": "play music generically",
        "spotify/play_track": "play specific song/artist",
        "spotify/pause": "pause or stop music",
        "spotify/next": "next track",
        "spotify/previous": "previous track",
        "spotify/open": "open spotify app",
        "spotify/read": "what's playing",
        "spotify/volume_up": "spotify volume up",
        "spotify/volume_down": "spotify volume down",
        "voice/volume_up": "bella voice volume up",
        "voice/volume_down": "bella voice volume down",
        "news/auto": "news headlines",
        "omdb/auto": "movie info",
        "joke/auto": "tell a joke",
    },
    "location": {
        "location/weather": "weather forecast",
        "location/nearby": "nearby restaurants, hotels, places",
        "location/here": "where am I, my location",
        "location/distance": "how far, drive time",
        "location/showtimes": "cinema showtimes",
        "location/open_maps": "open in maps",
        "discovery/hotels": "search hotels in a city",
        "discovery/flights": "search flights",
    },
    "screen": {
        "screen/read": "read/describe screen, what's on screen, how to click",
        "screen/region": "circle/highlight region",
        "desktop/auto": "click button or control (not teaching)",
    },
    "system": {
        "shell/auto": "run terminal command",
        "files/auto": "read/list/open project files",
        "browser/auto": "open website, browser control",
        "capabilities/read": "what can you do",
        "capabilities/model": "which model are you",
        "github/auto": "github",
        "googledrive/auto": "google drive files",
    },
    "integrations": {
        "youtube/auto": "youtube search",
        "linkedin/auto": "linkedin",
        "reddit/auto": "reddit",
        "chatfai/auto": "chatfai character",
    },
    "chat": {
        "chat/none": "no tool — general chat reply",
    },
}

DEFAULT_THRESHOLD = 0.85

_INDIC = re.compile(r"[\u0980-\u09FF\u0900-\u097F]")

_agents: dict[tuple[str, str], Any] = {}


def _needs_multilingual(text: str) -> bool:
    return bool(_INDIC.search(text))


def _coarse_questions() -> dict:
    return {
        "category": {
            "type": "choice",
            "instructions": "Which category best matches the user's request to Bella?",
            "criteria": COARSE_CATEGORIES,
        }
    }


def _fine_questions(category: str) -> dict:
    tools = FINE_TOOLS.get(category, FINE_TOOLS["chat"])
    return {
        "tool": {
            "type": "choice",
            "instructions": f"Which Bella tool should handle this {category} request?",
            "criteria": tools,
        }
    }


def _get_agent(*, device: str, multilingual: bool) -> Any:
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    key = (device, "multilingual" if multilingual else "english")
    if key in _agents:
        return _agents[key]

    import laya

    if multilingual:
        agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
    else:
        agent = laya.load("convaiinnovations/laya")
    if hasattr(agent, "to"):
        agent = agent.to(device)
    _agents[key] = agent
    return agent


def _parse_tool(label: str) -> tuple[str, str] | None:
    if not label or label in ("chat/none", "none", "chat"):
        return None
    if "/" not in label:
        return (label, "auto")
    domain, action = label.split("/", 1)
    return domain, action


def classify_hierarchical(
    text: str,
    *,
    confidence_threshold: float = DEFAULT_THRESHOLD,
    device: str = "mps",
    preload: bool = False,
) -> str | None:
    """Two-stage Laya routing. Returns 'domain/action' label or None to abstain."""
    if preload:
        _get_agent(device=device, multilingual=False)
        _get_agent(device=device, multilingual=True)

    ml = _needs_multilingual(text)
    agent = _get_agent(device=device, multilingual=ml)
    state = {"body": text}

    coarse = agent.predict(state, _coarse_questions())
    cat = coarse["answers"]["category"]["choice"]
    cat_conf = coarse["answers"]["category"]["confidence"]

    if cat == "chat" or cat_conf < confidence_threshold:
        return None

    fine = agent.predict(state, _fine_questions(cat))
    tool = fine["answers"]["tool"]["choice"]
    tool_conf = fine["answers"]["tool"]["confidence"]

    if tool_conf < confidence_threshold:
        return None
    if tool == "chat/none":
        return None
    return tool


def classify(
    text: str,
    cfg: dict | None = None,
) -> tuple[str, str] | None:
    """Entry point for process_turn(). Returns (domain, action) or None."""
    cfg = cfg or {}
    threshold = float(cfg.get("laya_confidence_threshold", DEFAULT_THRESHOLD))
    device = cfg.get("laya_device", "mps")
    label = classify_hierarchical(
        text,
        confidence_threshold=threshold,
        device=device,
        preload=bool(cfg.get("laya_preload")),
    )
    if label is None:
        return None
    return _parse_tool(label)


def guard_check(text: str, cfg: dict | None = None) -> bool:
    """Return True if input looks like a jailbreak/injection (block turn)."""
    cfg = cfg or {}
    if not cfg.get("laya_guard_enabled", False):
        return False
    try:
        import laya
    except ImportError:
        return False

    device = cfg.get("laya_device", "mps")
    agent = _get_agent(device=device, multilingual=False)
    result = agent.predict({"prompt": text}, laya.guard_questions())
    answers = result.get("answers") or {}
    threshold = float(cfg.get("laya_guard_threshold", 0.85))
    for ans in answers.values():
        if ans.get("noul", 0) >= threshold:
            return True
    return False


def head_max_len_report() -> dict:
    """Token budget check for hierarchical schemas (POC validation)."""

    def budget(criteria: dict[str, str], head_max_len: int = 256) -> dict:
        n = len(criteria)
        per = (head_max_len - 16) // max(n, 1)
        return {"options": n, "tokens_per_option": per, "ok": n <= 20 and per >= 4}

    fine_counts = {cat: budget(tools) for cat, tools in FINE_TOOLS.items()}
    worst_n = max(v["options"] for v in fine_counts.values())
    return {
        "coarse": budget(COARSE_CATEGORIES),
        "fine_max": worst_n,
        "fine_worst": {k: v for k, v in fine_counts.items() if v["options"] == worst_n},
    }
