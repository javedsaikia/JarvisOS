"""Put what Max just said on screen, as text, next to the work.

The writer half of the instruction panel; max/instruction_panel.py is the
viewer. Split the same way max/control.py splits from the voice loop: two
processes, a small JSON file between them, and neither one caring whether
the other is running. Calling show() when no panel process exists starts
one; calling it when the panel has died is harmless.

Everything here is best-effort on purpose. A panel that fails to appear
must never take down the spoken answer it was accompanying — the voice is
the primary channel, the text is the reinforcement.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from max.instruction_panel import PID_PATH, STATE_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# How long a single instruction stays up if nothing else ends it. The panel
# also closes itself when the user switches app, which is usually what
# actually ends a lesson; this is the backstop for "they wandered off".
DEFAULT_SECONDS = 120.0


def _panel_running() -> bool:
    try:
        pid = int(PID_PATH.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _ensure_panel() -> None:
    if _panel_running():
        return
    try:
        subprocess.Popen(
            [sys.executable, "-m", "max.instruction_panel"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as e:
        print(f"[panel] could not start the instruction panel: {e}", flush=True)


def _write(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, STATE_PATH)
        os.chmod(STATE_PATH, 0o600)
    except OSError as e:
        print(f"[panel] could not write panel state: {e}", flush=True)


# This panel is a reference card, not a transcript. Anything that is part
# of a conversation rather than part of an instruction is removed here,
# whatever produced it. The card is meant to be glanced at while working —
# "Want me to show you?" sitting under a shortcut is noise, and "Next
# step?" is worse than noise because it implies the card is waiting on
# something.
_CONVERSATIONAL_RE = re.compile(
    r"(?:"
    r"\b(?:want me to|shall i|should i|would you like|do you want|let me know|"
    r"tell me if|just say|say the word|feel free)\b[^.!?]*[.!?]?|"
    r"\b(?:i'?ll|i can|i have|i'?ve|i'?m|let me)\b[^.!?]*[.!?]?|"
    r"\bwhat (?:do you want|would you like|else|now|next)\b[^.!?]*[.!?]?|"
    r"\b(?:next step|anything else|that'?s it|hope (?:that|this) helps|"
    r"you'?re welcome|no problem|sure thing|of course)\b[^.!?]*[.!?]?|"
    # Talk ABOUT the exchange rather than about the task. The teacher
    # prompt hands the model "The student said: …", and it echoes that
    # back — "The student asked you to summarize this page." is a true
    # sentence and still has no business on a reference card.
    r"\b(?:the (?:student|user)|they|you)\s+(?:asked|said|wanted?|requested|"
    r"would like)\b[^.!?]*[.!?]?|"
    r"\bclick it whenever you'?re ready\b[^.!?]*[.!?]?"
    r")",
    re.IGNORECASE,
)
# A line that is only a question is conversation by definition — a
# reference card states, it does not ask.
_QUESTION_LINE_RE = re.compile(r"^[^.!?]*\?\s*$")

MAX_LINES = 6
MAX_LINE_CHARS = 72


def for_screen(text: str) -> str:
    """Strip conversational scaffolding out of a line bound for the panel."""
    cleaned = _CONVERSATIONAL_RE.sub(" ", text or "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -–—,;")
    return "" if _QUESTION_LINE_RE.match(cleaned) else cleaned


def build_card(steps, fallback: str = "") -> str:
    """Turn instruction lines into the panel's body.

    `steps` is what the model wrote for the screen; `fallback` is the
    spoken line, used only when the model ignored the format. Either way
    every line goes through for_screen, so a model that slips back into
    conversation cannot put conversation on the card.
    """
    lines: list[str] = []
    for step in steps or []:
        line = for_screen(str(step))
        if not line:
            continue
        if len(line) > MAX_LINE_CHARS:
            line = line[: MAX_LINE_CHARS - 1].rstrip() + "…"
        if line not in lines:
            lines.append(line)
        if len(lines) >= MAX_LINES:
            break
    if lines:
        return "\n".join(lines)
    # No usable steps. Salvage the spoken line by sentence, dropping any
    # sentence that is conversational — better a one-line card than none.
    salvaged = [
        s for s in (for_screen(part) for part in re.split(r"(?<=[.!?])\s+", fallback or ""))
        if s
    ]
    return "\n".join(salvaged[:MAX_LINES])


def show(
    text: str,
    app: str = "",
    title: str = "",
    seconds: float = DEFAULT_SECONDS,
    raw: bool = False,
) -> None:
    """Display `text` in the panel, replacing whatever was there.

    `app` is the application the instruction is about; the panel closes
    itself when the user moves to a different one, because at that point
    the instruction on screen is about something they are no longer doing.

    `raw` skips the conversational filter, and exists for one case: a
    DRAFT is the user's own document, not Max talking. An email that
    genuinely says "I'll send you the report on Friday" would have that
    sentence deleted by the filter, which is right for a reference card
    and catastrophic for a draft.
    """
    body = (text or "").strip() if raw else for_screen(text)
    if not body:
        return
    _ensure_panel()
    _write({
        "visible": True,
        "text": body,
        "title": (title or "MAX").upper()[:24],
        "app": app or "",
        "seconds": float(seconds),
        # Any change to the text restarts the panel's own timer and
        # rewrites it in place, rather than opening a second window.
        "stamp": f"{time.time():.3f}",
    })


def hide() -> None:
    """Take the panel down now — the lesson is over."""
    if not STATE_PATH.exists():
        return
    _write({"visible": False, "text": "", "stamp": f"{time.time():.3f}"})
