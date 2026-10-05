"""Write text for the user, and — only ever with a yes — type it in.

Bella already reads the screen and explains it. This is the other half:
producing the words the user needs to put ON that screen. An email with
its subject line, a prompt to hand to a chatbot about the code they are
looking at, a covering letter, a set of instructions.

Two rules shape everything here.

**Nothing is typed without an explicit yes.** Insertion changes a document
the user owns — an unasked-for paste into a half-finished email is not
something an undo makes fully right. So drafting and typing are separate
steps: Bella writes, shows the draft, asks, and stops there unless the
answer is yes. Same confirm_fn contract the shell tool and the Claude Code
handoff already use, so the terminal, the voice loop and the web UI each
ask in their own way without this module knowing which is which.

**The draft is read, not recited.** A four-paragraph email read aloud is
useless — by the end you have forgotten the beginning. So the draft goes
on the instruction panel to be read, and the spoken line only says what
was written and asks whether to type it.

Insertion is clipboard + Command-V rather than synthetic keystrokes.
Typing a long body character by character takes tens of seconds, drops
characters under load, and mangles anything non-ASCII — which for this
user, who writes Assamese, is not an edge case. The previous clipboard
contents are put back afterwards.
"""
from __future__ import annotations

import re
import time

from bella import instructions, llm

# What Bella will write. Each entry is (kind, spoken name, extra instruction
# for the model). The vocabulary is closed on purpose: a request to write
# something not on this list should fall through to ordinary conversation
# rather than be silently turned into a document.
KINDS: dict[str, tuple[str, str]] = {
    "email": (
        "email",
        "Write a complete email. First line must be exactly 'Subject: <subject>', "
        "then a blank line, then the body. Include a greeting and a sign-off.",
    ),
    "prompt": (
        "prompt",
        "Write a prompt to be pasted into an AI chatbot. Keep it short. "
        "If they named a concrete task (print hello in Python, review this "
        "function), write THAT task as the prompt — a few lines, not a "
        "meta-prompt about how to write prompts. Do not invent a project, "
        "file or error that was not in the request.",
    ),
    "letter": (
        "letter",
        "Write a complete letter with a greeting, body paragraphs and a sign-off.",
    ),
    "resume": (
        "resume",
        "Write the requested resume section as clean plain text. Use short lines. "
        "Do not invent employers, dates, degrees or numbers that were not given — "
        "leave a clearly marked blank like [dates] instead.",
    ),
    "instructions": (
        "instructions",
        "Write clear numbered steps. One action per step. No preamble.",
    ),
    "message": ("message", "Write a short message, three sentences at most."),
    "notes": ("notes", "Write concise notes as short bullet lines starting with '- '."),
}

_KIND_RE = re.compile(
    r"\b(?:write|draft|compose|create|prepare|make)\b[^.?!]{0,40}?\b("
    r"e-?mail|prompt|letter|cover letter|covering letter|resume|résumé|cv|"
    r"instructions?|message|reply|notes?|essay|paragraph|summary|bio|"
    r"description|post|caption"
    r")\b",
    re.IGNORECASE,
)

_ALIAS = {
    "e-mail": "email", "email": "email",
    "prompt": "prompt",
    "letter": "letter", "cover letter": "letter", "covering letter": "letter",
    "resume": "resume", "résumé": "resume", "cv": "resume",
    "instruction": "instructions", "instructions": "instructions",
    "message": "message", "reply": "message", "post": "message",
    "caption": "message", "bio": "message", "description": "message",
    "note": "notes", "notes": "notes",
    "essay": "letter", "paragraph": "message", "summary": "notes",
}

# "draft an email to bob@example.com" is the Gmail flow's job (it resolves
# the address, saves a real draft and can send it). This module is for
# composing text into whatever is on screen, so it stands aside whenever a
# recipient is named.
_HAS_RECIPIENT_RE = re.compile(
    r"\b(?:to|for)\s+(?:[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", re.IGNORECASE
)


# Scaffolding that is not a subject. "can you write a prompt?" has none.
# "write a prompt in python to print hello" still has python/print/hello.
_SUBJECT_STRIP_RE = re.compile(
    r"\b(?:hey|hi|ok|okay|bella|please|just|sorry|can you|could you|"
    r"would you|go ahead(?: and)?|yes or no|yes|yeah|yep|no|"
    r"write|draft|compose|create|prepare|make|"
    r"a|an|the|me|this|that|it)\b",
    re.IGNORECASE,
)
_SCREEN_REF_RE = re.compile(
    r"\b(?:this|that|the)\s+(?:code|page|screen|file|error|app|window|project)\b",
    re.IGNORECASE,
)
_WAITING_SUBJECT: dict | None = None


def has_subject(text: str) -> bool:
    """True when the request names what to write, not just the kind."""
    rest = _KIND_RE.sub(" ", text or "")
    rest = _SUBJECT_STRIP_RE.sub(" ", rest)
    rest = re.sub(r"[\s?!.:,]+", " ", rest).strip()
    return bool(re.search(r"[A-Za-z0-9]{3,}", rest))


def refers_to_screen(text: str) -> bool:
    return bool(_SCREEN_REF_RE.search(text or ""))


def waiting_for_subject() -> bool:
    return bool(_WAITING_SUBJECT) and (time.monotonic() - _WAITING_SUBJECT["at"]) < 180


def wait_for_subject(kind: str) -> None:
    global _WAITING_SUBJECT
    _WAITING_SUBJECT = {"kind": kind, "at": time.monotonic()}


def take_waiting_kind() -> str | None:
    global _WAITING_SUBJECT
    if not waiting_for_subject():
        _WAITING_SUBJECT = None
        return None
    kind = _WAITING_SUBJECT["kind"]
    _WAITING_SUBJECT = None
    return kind


def clear_waiting_subject() -> None:
    global _WAITING_SUBJECT
    _WAITING_SUBJECT = None


def detect(text: str) -> str | None:
    """The kind of thing to write, or None if this isn't a writing request."""
    match = _KIND_RE.search(text or "")
    if not match:
        return None
    kind = _ALIAS.get(match.group(1).lower())
    if kind == "email" and _HAS_RECIPIENT_RE.search(text or ""):
        return None  # belongs to the Gmail flow
    return kind


def draft(request: str, kind: str, cfg: dict, screen_text: str = "", app: str = "") -> str:
    """Write the thing. Returns plain text, ready to paste."""
    _name, rule = KINDS.get(kind, ("text", ""))
    context = ""
    # Only attach OCR when they pointed at the screen. Seen live: "write a
    # prompt" in ChatGPT pulled Bella window titles into a 200-word
    # meta-prompt instead of asking what the prompt should say.
    if screen_text.strip() and refers_to_screen(request):
        context = (
            "\n\nThis is what is currently on their screen"
            f"{f' in {app}' if app else ''} — it is the subject matter:\n---\n"
            f"{screen_text[:3000]}\n---\n"
        )
    prompt = (
        f"You are writing on behalf of {cfg.get('user_name', 'the user')}, who "
        "will paste your output straight into a document. "
        f"{rule}\n\n"
        "Output ONLY the finished text. No preamble, no explanation, no "
        "markdown fences, no commentary about what you wrote, no placeholders "
        "except where a fact was genuinely not supplied. Never invent names, "
        "figures, dates or quotes that were not given to you."
        f"{context}\n\nThe request: {request}"
    )
    entry = llm.resolve(cfg, "chat")
    try:
        text = llm.chat(
            [{"role": "user", "content": prompt}], entry, cfg,
            options={"num_predict": cfg.get("compose_max_tokens", 700)},
        )
    except llm.LLMError as e:
        raise ComposeError(f"could not reach {entry['label']}: {e}") from e
    return _tidy(text)


class ComposeError(Exception):
    pass


_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
_PREAMBLE_RE = re.compile(
    r"^\s*(?:here'?s?|here is|sure|certainly|of course|below is|i'?ve written)\b[^\n]*\n+",
    re.IGNORECASE,
)


def _tidy(text: str) -> str:
    """Strip the wrapper models put around a document they were asked for."""
    out = _FENCE_RE.sub("", (text or "").strip())
    out = _PREAMBLE_RE.sub("", out)
    return out.strip()


def subject_of(text: str) -> str:
    """The Subject: line of a drafted email, if there is one."""
    match = re.match(r"\s*subject\s*:\s*(.+)", text or "", re.IGNORECASE)
    return match.group(1).strip() if match else ""


def preview(text: str, limit: int = 900) -> str:
    """The draft as it goes on the panel — whole if it fits, else trimmed."""
    body = (text or "").strip()
    if len(body) <= limit:
        return body
    cut = body.rfind("\n", 0, limit)
    if cut < limit // 2:
        cut = body.rfind(" ", 0, limit)
    return body[: cut if cut > 0 else limit].rstrip() + "\n…"


def spoken_summary(text: str, kind: str) -> str:
    """What Bella says out loud about a draft it is not going to recite."""
    name = KINDS.get(kind, ("draft", ""))[0]
    subject = subject_of(text)
    words = len(re.findall(r"\S+", text))
    if subject:
        return f"I've drafted the {name}, subject “{subject}”, about {words} words."
    return f"I've drafted the {name}, about {words} words."


# --- a draft waiting to be typed -----------------------------------------
# Saying no to "type this in?" must not throw the draft away. People say
# no because they want to read it first, move the cursor, or open the
# right window — and then "go on, type it in" has to mean something. The
# draft stays here until it is used, replaced, or the session moves on.
_PENDING: dict | None = None

_INSERT_RE = re.compile(
    r"\b(?:type (?:it|that|this)(?: in| out)?|paste (?:it|that|this)?|"
    r"insert (?:it|that|this)?|put (?:it|that) in|go ahead|do it|"
    r"write it in|add it)\b",
    re.IGNORECASE,
)


def remember(text: str, kind: str, app: str) -> None:
    global _PENDING
    _PENDING = {"text": text, "kind": kind, "app": app, "at": time.monotonic()}


def has_pending(max_age: float = 600.0) -> bool:
    return bool(_PENDING) and (time.monotonic() - _PENDING["at"]) < max_age


def pending() -> dict | None:
    return dict(_PENDING) if _PENDING else None


def clear_pending() -> None:
    global _PENDING
    _PENDING = None


def wants_insert(text: str) -> bool:
    """"Type it in" / "go ahead" — the yes that arrives a turn later."""
    return bool(_INSERT_RE.search(text or ""))


# --- insertion ------------------------------------------------------------

def insert(text: str) -> None:
    """Paste `text` into whatever has keyboard focus.

    Clipboard + Command-V, not synthetic per-character keystrokes: a long
    body typed character by character takes tens of seconds, loses
    characters when the CPU is busy, and cannot reliably produce non-ASCII
    at all — which matters here, where the user writes Assamese. The old
    clipboard contents are restored afterwards so this does not quietly
    eat something they had copied.
    """
    if not (text or "").strip():
        return
    import AppKit
    import Quartz

    board = AppKit.NSPasteboard.generalPasteboard()
    previous = board.stringForType_(AppKit.NSPasteboardTypeString)

    board.clearContents()
    board.setString_forType_(text, AppKit.NSPasteboardTypeString)
    # A beat for the pasteboard to settle before the paste; without it the
    # keystroke can arrive while the old contents are still current.
    time.sleep(0.15)

    source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    v_keycode = 9  # 'v'
    for down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(source, v_keycode, down)
        Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(0.03)

    if previous is not None:
        # Long enough for the target app to have taken the paste.
        time.sleep(0.6)
        board.clearContents()
        board.setString_forType_(previous, AppKit.NSPasteboardTypeString)


def show_draft(text: str, kind: str, app: str = "") -> None:
    """Put the draft on the instruction panel to be read and checked."""
    name = KINDS.get(kind, ("draft", ""))[0]
    instructions.show(
        preview(text),
        app=app,
        title=f"DRAFT — {name.upper()}",
        # Long enough to actually read and edit against, and it still
        # closes when they switch app or say they are done.
        seconds=300.0,
        raw=True,
    )
