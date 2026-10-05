"""Screen understanding: a real screenshot, read on this machine.

Native capture and OCR live in bella/screen_capture.py (why they are
native, and how the HUD keeps out of its own screenshot, is explained
there). This module is the tool layer: it orchestrates hide -> capture ->
restore, turns the capture into an answer, and reports failures in
language a person can act on.

The answer comes from whichever path can actually read the screen:

1. **OCR + the local text model** — the default, and the only one that
   works for the question people actually ask. A screen is mostly text,
   macOS reads it exactly, and the local model then answers about real
   strings instead of guessing from a thumbnail.
2. **The local vision model** — for screens with little text (a photo, a
   video, a design), where OCR has nothing to work with.
3. **A cloud vision model** — off unless configured *and* confirmed,
   because it is a paid call and this project's rule is that paid calls
   are opt-in and confirmed, never a silent fallback.

1 and 2 are local and free; no image leaves the machine unless you turn
on 3 and say yes to the prompt.
"""
import base64
import json
import re
import time
import urllib.error
import urllib.request

from bella import env, instructions, llm, screen_capture, voice_events
from bella.tools import files

DEFAULT_QUESTION = "In one or two short sentences, what is on the screen? Then ask what they want next."

# Below this much recognized text, the screen is treated as visual rather
# than textual and goes to the vision model instead. A desktop with only
# a menu bar and a Dock lands around 200 characters, an actual working
# window is thousands.
MIN_OCR_CHARS = 220


class VisionError(Exception):
    pass



def _log(message: str) -> None:
    """One prefixed line per stage, so the capture -> read -> answer flow
    can be followed in /tmp/voice_loop.log or the bridge log without a
    debugger. This path touches the screen and (optionally) the network,
    which is exactly the kind of thing that should never happen silently."""
    print(f"[screen] {message}", flush=True)


def _ask_vision_model(jpeg_bytes: bytes, prompt: str, cfg: dict) -> str:
    """Local Ollama vision model (moondream by default) on the image."""
    model = cfg.get("vision_ollama_model", "moondream")
    host = cfg["ollama_host"]
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt,
             "images": [base64.b64encode(jpeg_bytes).decode("ascii")]}
        ],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise VisionError(
            f"Could not reach Ollama at {host} ({e}). Is `ollama serve` running?"
        ) from e

    content = data.get("message", {}).get("content", "").strip()
    if not content:
        raise VisionError(
            f"The vision model ({model}) returned nothing — is it pulled? "
            f"Run `ollama pull {model}`."
        )
    return content


# macOS reports the process name, which is not always what the app is
# called on screen. Only the ones that actually differ are listed; anything
# missing is used as-is, which is right far more often than not.
APP_NAMES = {
    "Code": "Visual Studio Code",
    "Google Chrome": "Chrome",
    "com.apple.finder": "Finder",
    "Electron": "an Electron app",
}

BROWSERS = {
    "Safari", "Google Chrome", "Chrome", "Firefox", "Arc",
    "Brave Browser", "Microsoft Edge", "Orion", "Dia",
}
PICTURE_APPS = {
    "Preview", "Photos", "Photo Booth", "QuickTime Player",
    "Image Capture", "Screenshot",
}

# "What's on my screen" with nothing else in it. These get a deterministic
# answer built from the window list and the OCR — see _summarize_screen.
_LEAD = r"^(hey[, ]+)?((bella)[, ]+)?(can you |could you |please )?(tell me )?"
# Spoken questions trail off into time words. "What's on my screen RIGHT
# NOW" is the same generic question as "what's on my screen", but the old
# `$` anchor rejected it — so it went to the model, which answered with a
# four-line inventory of the Word ribbon instead of "This is a Microsoft
# Word window. What do you want help with?". Measured from a real turn.
_TRAIL = r"(?:[, ]+(?:right now|at the moment|just now|now|today|currently|bella|please))*$"
_MINE = r"( my| the| this| your)?"
GENERIC_SCREEN_QUESTIONS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Deliberately anchored end-to-end with only filler words allowed
        # in between: "what's on my screen" is generic, but "what's the
        # error message on screen" carries a subject and must reach the
        # model, so nothing arbitrary may match in the middle.
        # Same manglings the router now accepts (see SCREEN_PATTERNS): if
        # they reach the screen tool but not this list, they take the slow
        # model path instead of the instant window-facts answer.
        _LEAD + r"(what'?s|what is|wat'?s|watch|what son|was)( on| in)?" + _MINE + r" screen" + _TRAIL,
        _LEAD + r"i watch( on| in)?" + _MINE + r" screen" + _TRAIL,
        _LEAD + r"what son" + _MINE + r" screen" + _TRAIL,
        _LEAD + r"(describe|read|check|look at|analyse|analyze)" + _MINE + r" screen" + _TRAIL,
        _LEAD + r"what (can |do )?you (see|tell).{0,40}screen" + _TRAIL,
        _LEAD + r"what am i (looking at|seeing)" + _TRAIL,
        _LEAD + r"(can|do) you see" + _MINE + r" screen" + _TRAIL,
        _LEAD + r"what do you see" + _TRAIL,
        _LEAD + r"(take|grab|get) a screenshot" + _TRAIL,
        # Assamese "what's on my screen" — same look-then-ask as English.
        r".{0,40}(?:স্ক্ৰীণ|স্ক্ৰেইন|স্ক্ৰিন|স্ক্ৰীন|স্ক্রিন|স্ক্রীন|পৰ্দা).{0,40}$",
        r".{0,24}কি দেখিছা.{0,24}$",
    )
]


_TEACHER_RE = re.compile(
    r"\b(teach|tutor|explain|walk me through|help me understand|"
    r"what file|this file|this code|my desktop|the desktop|whole desktop|"
    r"read (this |that |the |my )?(file|code|desktop))\b|"
    r"বুজাই দিয়|ব্যাখ্যা কৰ|এই প্ৰশ্নটো কি",
    re.IGNORECASE,
)

# Window titles and OCR both leak filenames. VS Code: "cli.py — Bella".
# Finder/Preview: "README.md". Ignore obvious non-files.
_FILENAME_RE = re.compile(
    r"\b([\w.-]+\.(?:py|ts|tsx|js|jsx|md|json|txt|css|html|htm|sh|zsh|rs|go|"
    r"swift|pdf|csv|yml|yaml|toml|ini|env|ipynb|rb|java|c|h|cpp|hpp))\b",
    re.IGNORECASE,
)
_SKIP_TITLES = {"bella hud", "finder", "untitled", "desktop"}


def _is_teacher(question: str) -> bool:
    return bool(_TEACHER_RE.search(question or ""))


# Must name a control, a tab, or an explicit "show me where to click".
# Bare "where is" / "go to" / "show me" swallowed ordinary chat
# ("where is Paris", "show me a good recipe") and then pointed at
# whatever OCR happened to match. "click the Save button" is a
# desktop do-it, not a guide — only "click … tab" points.
_GUIDE_RE = re.compile(
    r"(?:"
    r"\b(?:show|point|guide)\s+(?:me\s+)?(?:where\s+|to\s+)?"
    r"(?:the\s+|this\s+|that\s+)?.{0,40}"
    r"\b(?:button|tab|link|icon|menu|option|setting|toggle|"
    r"checkbox|cursor|pointer|mouse)s?\b|"
    r"\bwhere\s+(?:is|are)\s+(?:the\s+|this\s+|that\s+)?.{0,40}"
    r"\b(?:button|tab|link|icon|menu|option|setting|toggle|checkbox)s?\b|"
    r"\bwhere\s+do\s+i\s+(?:click|press|tap|go)\b|"
    r"\b(?:can'?t|cannot|not able to)\s+see\b|"
    r"\bpoint\s+.{0,16}\b(?:cursor|pointer|mouse)\b|"
    r"\b(?:go to|take me to|navigate to|open)\b.{0,40}\btabs?\b|"
    r"\b(?:click|press|tap)\b.{0,40}\btabs?\b|"
    r"\bhelp me\s+(?:find|click|go to|open|get to)\b|"
    r"\bguide me\b|"
    r"\bshow me where\b|"
    r"\bhow\s+(?:do i|can i|should i|to)\s+(?:click|press|tap)\b|"
    r"\bthe\s+.{0,24}\s+tab\b|"
    r"\bthat application\b"
    r")",
    re.IGNORECASE,
)
# "Show me" / "point" / "guide me" are instructions to MOVE the ring.
# Bare "Help me" is not — that is "I need help; ask me what".
_GUIDE_FOLLOWUP_RE = re.compile(
    r"^(?:hey[, ]+)?(?:bella[, ]+)?(?:yes[, ]+)?(?:please )?"
    r"(?:(?:can|could|will|would) you )?(?:just )?"
    r"(?:guide|show|point)(?: me)?"
    r"(?: (?:to|at|where|here))?"
    r"(?: (?:it|that|this|here))?"
    r"\s*[.?!]*$",
    re.IGNORECASE,
)
_HELP_ASK_RE = re.compile(
    r"^(?:hey[, ]+)?(?:bella[, ]+)?"
    r"(?:(?:can|could|will|would) you )?(?:just )?"
    r"help(?: me)?(?: (?:please|out|here))?"
    r"\s*[.?!]*$",
    re.IGNORECASE,
)
_NEXT_ASK = "What do you want me to do with this?"
_EXPLAIN_ASK = (
    "What kind of help do you want next — the answer, more explanation, "
    "or for me to point at something?"
)
_BARE_SHOW_ME_RE = re.compile(
    r"^(?:hey[, ]+)?(?:bella[, ]+)?(?:yes[, ]+)?(?:can you )?"
    r"show me\s*(?:it|that|this|where)?\s*[.?!]*$",
    re.IGNORECASE,
)


# "Show me the bold button" is short and imperative. A paragraph is
# someone talking, and any long stretch of speech will contain "show" or
# "guide" or "view" eventually. Seen live during a demo: a 100-word
# spoken introduction ("...it can view the screen and it can guide the
# user...") tripped this, and the reply opened with three invented
# pointer cues — 'This is "View", the top left. This is "it Bella" then the
# command", the bottom right.'
_GUIDE_MAX_WORDS = 25


def _is_guide(question: str) -> bool:
    text = question or ""
    if len(text.split()) > _GUIDE_MAX_WORDS:
        return False
    stripped = text.strip()
    return bool(
        _GUIDE_RE.search(text)
        or _BARE_SHOW_ME_RE.match(stripped)
        or _GUIDE_FOLLOWUP_RE.match(stripped)
    )


def is_guide(text: str) -> bool:
    """They asked to be shown a control, not just told about the page."""
    return _is_guide(text)


def is_help_ask(text: str) -> bool:
    """Bare 'help me' — read the page and ASK, do not point yet."""
    return bool(_HELP_ASK_RE.match((text or "").strip()))


def _with_next_ask(answer: str, ask: str = _NEXT_ASK) -> str:
    """Keep the look, then one ask. The code used to strip any '?'."""
    text = re.sub(r"\s+", " ", answer or "").strip()
    if not text:
        return ask
    tail = text[-90:].lower()
    if "?" in tail or "what do you want" in tail:
        return text
    return f"{text} {ask}"


def matches_visible_control(text: str) -> bool:
    """True when this turn names a label already read off the screen.

    After a Composio dashboard look, "playground" / "API Keys" should
    move the pointer, even without "show me the tab".
    """
    if not session_active() or not _LAST_LINES:
        return False
    words = [w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
             if w not in _GUIDE_SKIP]
    if not words:
        return False
    return _best_ocr_match(_LAST_LINES, " ".join(words)) is not None


# Last OCR + labels we already named, so "show me" / "point at that"
# can walk the same controls instead of pointing at the middle of Safari.
_LAST_LINES: list[dict] = []
_LAST_HINTS: list[str] = []

# --- The teaching session -------------------------------------------------
# "What's on my screen" opens a session; follow-ups ("how do I bold this?",
# "yes, show me") belong to whatever app was named, without the user having
# to say its name again. Router patterns are stateless by design, so this
# state is consulted from cli.process_turn the same way an open email draft
# and a pending Spotify query already are.
SESSION_SECONDS = 600.0

_SESSION: dict = {"at": 0.0, "app": "", "raw_app": ""}
# Set when Bella has offered to point at a specific control, cleared once
# the offer is taken or the subject changes. Holding the control's name is
# what lets a bare "yes" mean something concrete.
_PENDING_SHOW: dict | None = None


def _open_session(meta: dict) -> None:
    raw = meta.get("frontmost", "") or ""
    _SESSION.update({"at": time.monotonic(), "raw_app": raw,
                     "app": APP_NAMES.get(raw, raw),
                     "width": int(meta.get("width") or 0),
                     "height": int(meta.get("height") or 0)})


def session_active() -> bool:
    """True while a screen conversation is still live."""
    return bool(_SESSION["app"]) and (time.monotonic() - _SESSION["at"]) < SESSION_SECONDS


def session_app() -> str:
    return _SESSION["app"] if session_active() else ""


def set_reply_language(code: str | None) -> None:
    """as-IN: write the screen reply in Assamese in one model call."""
    _SESSION["language"] = (code or "").strip()


def _reply_language() -> str:
    return (_SESSION.get("language") or "").strip()


def has_pending_show() -> bool:
    return _PENDING_SHOW is not None


def clear_pending_show() -> None:
    global _PENDING_SHOW
    _PENDING_SHOW = None


def dismiss_panel() -> None:
    """Take the on-screen instruction down now."""
    instructions.hide()


_HOWTO_RE = re.compile(
    r"\b(?:"
    # "how do I" and "how can I" — plus the word order people actually
    # speak. Measured from a real turn: "can you help me HOW I CAN bold
    # the text on the Microsoft Word?" matched nothing, so it fell through
    # to plain chat, which meant no instruction card and no pointer — the
    # two things that were supposed to appear.
    r"how\s+(?:do|can|could|would|should)\s+i|"
    r"how\s+i\s+(?:can|could|would|should)|"
    r"how\s+(?:to|does\s+one)|"
    r"which\s+(?:button|key|tab|menu|option|shortcut)|"
    r"what'?s?\s+(?:is\s+)?the\s+(?:shortcut|key|keys)|"
    r"where\s+(?:is|do i find|can i find)\s+the\s+.{0,30}(?:button|tab|menu|option|setting)|"
    # "can you help me …" / "show me how to …". Broad on purpose: this is
    # only consulted while a screen session is open AND nothing else
    # matched, so it cannot steal a calendar or email request.
    r"(?:can|could|will|would)\s+you\s+(?:help|show|teach)\s+me|"
    r"help\s+me\s+(?:to\s+)?\w+|"
    r"show\s+me\s+how"
    r")",
    re.IGNORECASE,
)


def is_howto(text: str) -> bool:
    """A "how do I do X in this app" question, for the open session."""
    return bool(_HOWTO_RE.search(text or ""))


_TASK_LEAD_RE = re.compile(
    # Repeated on purpose: spoken requests stack these up — "can you help
    # me how I can bold the text" is three leads before the actual task.
    # Matching only one left "how I can bold" as the card's heading.
    r"^\s*(?:"
    r"hey\s+bella[,\s]*|please\s+|(?:can|could|will|would)\s+you\s+|"
    r"help\s+me\s+(?:to\s+)?|show\s+me\s+how\s+(?:to\s+)?|teach\s+me\s+(?:to\s+)?|"
    r"how\s+(?:do|can|could|would|should)\s+i\s+|how\s+i\s+(?:can|could|would|should)\s+|"
    r"how\s+(?:to|does\s+one)\s+|"
    r"what'?s?\s+(?:is\s+)?the\s+(?:shortcut|key|keys)\s+(?:for|to)\s+|"
    r"which\s+(?:button|key|tab|menu|option)\s+(?:do\s+i\s+(?:press|click|use)\s+)?(?:for|to)?\s*"
    r")+",
    re.IGNORECASE,
)


_SUMMARIZE_RE = re.compile(
    r"\b(summari[sz]e|sum up|tldr|give me the gist|the gist of)\b", re.IGNORECASE
)


def _is_summarize(question: str) -> bool:
    return bool(_SUMMARIZE_RE.search(question or ""))


def _panel_title(question: str) -> str:
    """Heading for the reference card: "HOW TO — BOLD TEXT"."""
    task = _task_title(question)
    return f"HOW TO — {task}" if task and task != "BELLA" else "INSTRUCTIONS"


def _task_title(question: str) -> str:
    """"how do I bold text" -> "BOLD TEXT", for the panel's heading.

    A heading naming the task is what makes the panel scannable when the
    user looks back at it thirty seconds later, having forgotten which of
    the three things they asked about it is answering.
    """
    task = _TASK_LEAD_RE.sub("", question or "").strip(" ?.!,")
    task = re.sub(r"\b(in|on|for) (this|the|my) (app|application|screen|window|document)\b",
                  "", task, flags=re.IGNORECASE).strip()
    task = re.sub(r"\s+", " ", task)
    return task[:24].upper() if task else "BELLA"


def _remember_screen(lines: list[dict], extra: list[str] | None = None) -> None:
    global _LAST_LINES, _LAST_HINTS
    if lines:
        _LAST_LINES = list(lines)
    hints: list[str] = []
    for raw in extra or []:
        t = re.sub(r"\s+", " ", raw).strip()
        if 2 <= len(t) <= 40:
            hints.append(t)
    for line in (lines or [])[:40]:
        t = re.sub(r"\s+", " ", (line.get("text") or "")).strip()
        if 2 <= len(t) <= 32 and t.lower() not in _GUIDE_SKIP:
            hints.append(t)
    seen: set[str] = set()
    kept: list[str] = []
    for h in hints:
        key = h.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(h)
    if kept:
        _LAST_HINTS = kept[:16]


def _hints_from_answer(text: str) -> list[str]:
    found = re.findall(r"[“\"]([^”\"]{2,40})[”\"]", text or "")
    found += re.findall(r"\b((?:Google |Microsoft |Apple )?[A-Z][A-Za-z0-9+]{2,}(?:\s+[A-Z][A-Za-z0-9+]{2,})*)\b", text or "")
    return found


def _filenames_on_screen(meta: dict, lines: list[dict]) -> list[str]:
    """Filenames visible in window titles or OCR, in frontmost-first order."""
    found: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        base = name.strip()
        if not base or base.lower() in seen:
            return
        seen.add(base.lower())
        found.append(base)

    for window in meta.get("windows") or []:
        title = (window.get("title") or "").strip()
        if not title or title.lower() in _SKIP_TITLES:
            continue
        # "cli.py — Bella — …" / "cli.py - Bella"
        head = re.split(r"\s+[—–\-•|]\s+", title, maxsplit=1)[0].strip()
        if _FILENAME_RE.search(head):
            add(head.split("/")[-1])
        for match in _FILENAME_RE.finditer(title):
            add(match.group(1))

    for line in lines or []:
        for match in _FILENAME_RE.finditer(line.get("text") or ""):
            add(match.group(1))
        if len(found) >= 8:
            break
    return found[:6]


# How much of the files on screen goes to the model. Three files at 4,000
# characters was 12,000 characters — roughly 3,000 tokens, dwarfing the
# OCR it was meant to accompany, and the single biggest reason the screen
# model kept hitting its 6,000 tokens-per-minute ceiling. Two files at
# 1,200 is enough to say what a file IS, which is all this answer claims
# to do; reading a file properly is what "teach me this file" is for.
FILE_PEEK_CHARS = 1200
FILE_PEEK_COUNT = 2


def _read_visible_files(names: list[str]) -> list[tuple[str, str]]:
    """Resolve on-screen names to real files and read the top of each."""
    loaded: list[tuple[str, str]] = []
    for name in names:
        paths = files.find_named(name, limit=1)
        if not paths:
            continue
        try:
            content = files.read_file(paths[0], max_chars=FILE_PEEK_CHARS)
        except files.FileToolError:
            continue
        loaded.append((paths[0], content))
        if len(loaded) >= FILE_PEEK_COUNT:
            break
    return loaded


# Screen questions carry the whole OCR of a desktop, which is the biggest
# payload Bella ever sends a model — and the assigned screen model sits on a
# 6,000 tokens-per-minute ceiling. Measured live: a single question used
# ~4,100 tokens, so the second one inside a minute came back
# "HTTP 429: Rate limit reached ... Limit 6000, Used 3905, Requested 4133"
# and the turn was lost outright with "I couldn't read the screen."
#
# Two answers, both needed. The payload caps below are far tighter than
# they were, and any failure of the assigned model falls back to the local
# one rather than dropping the turn — the same bargain cli.handle_ollama
# already makes, and the reason a rate limit should cost a little quality
# rather than the answer.
OCR_FOR_MODEL_CHARS = 2600


def _screen_token_budget(cfg: dict | None) -> int:
    """Enough tokens to finish an exam answer. 120 cut them off mid-stem."""
    try:
        n = int((cfg or {}).get("screen_max_tokens") or 400)
    except (TypeError, ValueError):
        n = 400
    return max(n, 320)


def _screen_chat(prompt: str, cfg: dict, max_tokens: int) -> str:
    """Ask the screen model, falling back to the local one on any failure."""
    entry = llm.resolve(cfg, "screen")
    pinned = cfg.get("screen_text_model")
    if pinned:
        entry = {**entry, "provider": "ollama", "model": pinned, "label": f"Bella · {pinned}"}
    try:
        return llm.chat([{"role": "user", "content": prompt}], entry, cfg,
                        options={"num_predict": max_tokens}).strip()
    except llm.LLMError as e:
        if entry["provider"] == "ollama":
            raise VisionError(f"Could not reach {entry['label']}: {e}") from e
        _log(f"{entry['label']} failed ({e}) — falling back to the local model")
        local = llm.resolve({**cfg, "model_roles": {}}, "screen")
        try:
            return llm.chat([{"role": "user", "content": prompt}], local, cfg,
                            options={"num_predict": max_tokens}).strip()
        except llm.LLMError as local_error:
            raise VisionError(f"Could not read the screen: {local_error}") from local_error


def _teach_from_screen(
    question: str,
    meta: dict,
    lines: list[dict],
    visible: list[str],
    loaded: list[tuple[str, str]],
    cfg: dict,
) -> str:
    """Teacher answer: desktop context + real file contents."""
    pinned = cfg.get("screen_text_model")
    entry = llm.resolve(cfg, "screen")
    if pinned:
        entry = {**entry, "provider": "ollama", "model": pinned, "label": f"Bella · {pinned}"}

    ocr = screen_capture.ocr_text(lines)[:OCR_FOR_MODEL_CHARS]
    windows = "; ".join(
        f"{w['app']}: {w['title']}" for w in meta.get("windows", [])[:10]
    )
    file_block = ""
    for path, content in loaded:
        file_block += f"\n--- {path} ---\n{content}\n"
    if not file_block:
        file_block = "(no matching files could be opened from the folders Bella can read)"

    names = ", ".join(visible) if visible else "none identified"
    prompt = (
        "You are Bella, talking to them at their Mac — not a professor.\n"
        "Teach from what is actually there. Do not invent files, errors, or code.\n"
        "Two to four short spoken sentences. Explain the content they asked about.\n"
        "Do not walk through every step unless they asked for that next action.\n"
        "Do not move or claim to point at anything.\n\n"
        f"Frontmost app: {meta.get('frontmost', 'unknown')}"
        f"{' — ' + meta['frontmost_title'] if meta.get('frontmost_title') else ''}\n"
        f"Windows: {windows}\n"
        f"Files identified on screen: {names}\n\n"
        f"File contents Bella opened:\n{file_block}\n\n"
        f"Text read off the screen:\n---\n{ocr}\n---\n\n"
        f"The student said: {question}\n\n"
        "Name the file and app, say what it is doing. No markdown, no lists, "
        "no lecture. If you could not open a file, say the name you saw. "
        "After the explanation, ask what they want next."
    )
    return _screen_chat(prompt, cfg, _screen_token_budget(cfg))


def _is_generic(question: str) -> bool:
    """True for "what's on my screen" and its phrasings, false for
    "what's the error on screen" — anything with a subject of its own
    needs the model, because only the model can find that subject in the
    text."""
    raw = question or ""
    # ASCII-only strip used to wipe Assamese, so "মোৰ স্ক্ৰিনত কি দেখিছা"
    # never counted as a look and never asked what they wanted next.
    if re.search(
        r"স্ক্ৰীণ|স্ক্ৰেইন|স্ক্ৰিন|স্ক্ৰীন|স্ক্রিন|স্ক্রীন|পৰ্দা|কি দেখিছা",
        raw,
    ):
        return True
    normalized = re.sub(r"[^a-z' ]", " ", raw.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return any(p.match(normalized) for p in GENERIC_SCREEN_QUESTIONS)


def _front_kind(app: str) -> str:
    raw = app or ""
    pretty = APP_NAMES.get(raw, raw)
    if raw in BROWSERS or pretty in BROWSERS:
        return "browser"
    if raw in PICTURE_APPS:
        return "picture"
    return "app"


_GUIDE_SKIP = {
    "what", "this", "that", "these", "those", "with", "from", "have",
    "screen", "please", "check", "look", "tell", "about", "show", "find",
    "where", "guide", "point", "want", "able", "just", "here", "there",
    "button", "link", "tab", "icon", "menu", "item", "bella",
    "can", "you", "the", "how", "all", "okay", "please",
    "cursor", "pointer", "mouse", "application", "app", "still",
    "struggling", "exactly", "dont", "don't", "see",
    "click", "press", "tap", "open", "take", "navigate", "towards",
    "help", "need", "could", "would", "will",
}

# macOS / browser chrome. Seen live: "go to the playground tab" on a
# medical PDF pointed at Safari / File / Edit because those are short
# labels at the top of the capture and Playground was not on that page.
_CHROME_LABELS = {
    "safari", "chrome", "firefox", "file", "edit", "view", "history",
    "bookmarks", "bookmark", "window", "windows", "help", "apple",
    "code", "finder", "display", "develop", "format", "insert",
    "share", "tools", "window", "minimize", "zoom",
}


def _guide_terms(question: str, use_hints: bool = True) -> list[str]:
    raw = (question or "").lower()
    terms: list[str] = []
    for phrase in (
        "pull requests", "pull request", "google sheets", "google docs",
        "show me how it works", "new issue", "code review",
    ):
        if phrase in raw:
            terms.append(phrase)
    words = [w for w in re.findall(r"[a-z0-9]{3,}", raw) if w not in _GUIDE_SKIP]
    for w in words:
        if w not in terms and w not in _CHROME_LABELS:
            terms.append(w)
    # Bare "show me" / "Help me" / "that application" — walk labels from
    # the last look (Playground, API Keys) instead of matching nothing.
    vague = bool(re.search(r"\bthat (app|application|one|button|thing)\b", raw))
    if use_hints and (
        vague
        or _GUIDE_FOLLOWUP_RE.match((question or "").strip())
        or not terms
        or set(terms) <= {"yes", "yeah", "please"}
    ):
        for hint in _LAST_HINTS:
            if hint.lower() not in terms and hint.lower() not in _CHROME_LABELS:
                terms.append(hint.lower())
    return terms


def _is_chrome_label(text: str) -> bool:
    t = re.sub(r"\s+", " ", (text or "").strip()).lower()
    return t in _CHROME_LABELS or t in _GUIDE_SKIP


def _is_menubar_line(line: dict, meta: dict, width: int, height: int) -> bool:
    """True for the system menu bar — not a tab on the page."""
    point = screen_capture.ocr_line_point(line, width, height)
    if point is None:
        y_mid = float(line.get("y") or 0) + float(line.get("height") or 0) / 2
        return y_mid > 0.97
    return point[1] < 32.0


def _best_ocr_match(lines: list[dict], question: str) -> dict | None:
    terms = _guide_terms(question)
    if not terms:
        return None
    best, score = None, 0.0
    for line in lines or []:
        text = re.sub(r"\s+", " ", (line.get("text") or "").lower()).strip()
        if not text or _is_chrome_label(text):
            continue
        s = 0.0
        for term in terms:
            if term in text:
                s += 3.0 if " " in term else 1.0
                if text == term or text.startswith(term):
                    s += 2.0
        if s > score:
            best, score = line, s
    return best if score >= 1.0 else None


def _region_phrase(line: dict) -> str:
    """Turn an OCR box (bottom-left, 0–1) into 'top left of the screen'."""
    x = line.get("x", 0) + line.get("width", 0) / 2
    y_from_bottom = line.get("y", 0) + line.get("height", 0) / 2
    y = 1 - y_from_bottom
    across = "left" if x < 0.33 else "right" if x > 0.67 else "the middle"
    down = "top" if y < 0.33 else "bottom" if y > 0.67 else "the middle"
    if across == "the middle" and down == "the middle":
        return "the centre of the screen"
    if across == "the middle":
        return f"the {down} of the screen"
    if down == "the middle":
        return f"the {across} side"
    return f"the {down} {across}"


def _small_capture(cfg: dict, fallback: bytes, meta: dict | None = None,
                   lines: list[dict] | None = None) -> bytes:
    """A cropped, downscaled recapture for a vision model.

    The OCR copy is the full desktop because OCR wants every pixel. The
    model must not get that 3456x2234 file — it times out, and it is not
    what they asked about. Crop the page body first, then cap width.
    """
    if meta is not None:
        focused = _focus_jpeg(meta, lines or [], fallback)
        if focused:
            return focused
    try:
        image, _ = screen_capture.capture(
            hud_marker=cfg.get("screen_hud_window_marker", "Bella HUD"), max_width=1024)
        return image or fallback
    except Exception as e:
        _log(f"could not re-capture small ({e}) — sending the full image")
        return fallback


def _mark_described_window(meta: dict, lines: list[dict] | None = None) -> None:
    """Put Bella's pointer on the page body it is talking about.

    Fire-and-forget, and deliberately the overlay rather than the real
    cursor: this runs on every screen question, and moving someone's mouse
    while they are working is not a feature. The marker sits on the
    content cluster, not the menu bar.
    """
    app = (meta or {}).get("frontmost") or ""
    try:
        rect = _content_rect(lines or [], meta)
        if rect:
            x, y, w, h = rect
            screen_capture.show_marker(
                x + w / 2, y + h / 2,
                label=APP_NAMES.get(app, app), seconds=20.0)
            return
        bounds = screen_capture.frontmost_window_bounds(app)
        if not bounds:
            return
        x, y, w, h = bounds
        screen_capture.show_marker(x + w / 2, y + h / 2,
                                   label=APP_NAMES.get(app, app), seconds=20.0)
    except Exception as e:
        _log(f"could not place the pointer: {e}")


def _tab_like_lines(lines: list[dict], meta: dict | None = None) -> list[dict]:
    """Short labels in the page chrome — tabs, nav — never the menu bar."""
    width = int((meta or {}).get("width") or _SESSION.get("width") or 0)
    height = int((meta or {}).get("height") or _SESSION.get("height") or 0)
    found: list[dict] = []
    seen: set[str] = set()
    for line in lines or []:
        text = re.sub(r"\s+", " ", (line.get("text") or "")).strip()
        if not text or len(text) > 28:
            continue
        words = text.split()
        if not (1 <= len(words) <= 4):
            continue
        if _is_chrome_label(text):
            continue
        if width and height and _is_menubar_line(line, meta or {}, width, height):
            continue
        y_mid = float(line.get("y") or 0) + float(line.get("height") or 0) / 2
        if y_mid < 0.62:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(line)
    return found


def _point_at_front(meta: dict, lines: list[dict], question: str) -> str:
    """Glide the pointer onto each matching control. Returns a spoken cue.

    If they named a tab that is not on this page, do not invent a click
    target. Seen live on a medical SBA PDF: "go to the playground tab"
    moved the ring onto Safari / File / Edit — chrome with no control
    that was asked for.
    """
    pool = _page_lines(lines or _LAST_LINES, meta)
    width, height = int(meta.get("width") or 0), int(meta.get("height") or 0)
    if not width:
        width = int(_SESSION.get("width") or 0)
        height = int(_SESSION.get("height") or 0)
    named = _guide_terms(question, use_hints=False)
    terms = _guide_terms(question)
    matches: list[dict] = []
    seen: set[str] = set()
    for term in terms[:8]:
        hit = _best_ocr_match(pool, term)
        if not hit:
            continue
        key = (hit.get("text") or "").strip().lower()
        if not key or key in seen or _is_chrome_label(key):
            continue
        seen.add(key)
        matches.append(hit)
    if not matches:
        hit = _best_ocr_match(pool, question)
        if hit and not _is_chrome_label(hit.get("text") or ""):
            matches.append(hit)
    vague = bool(_GUIDE_FOLLOWUP_RE.match((question or "").strip()))
    if not matches and vague:
        for line in _tab_like_lines(pool, meta):
            key = (line.get("text") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                matches.append(line)
            if len(matches) >= 3:
                break
    if matches and width:
        bits: list[str] = []
        for hit in matches[:4]:
            label = re.sub(r"\s+", " ", (hit.get("text") or "").strip())
            point = screen_capture.point_at_ocr_line(
                hit, width, height, label=label, seconds=20.0)
            _log(f"pointing at {label!r} -> {point}")
            bits.append(f"Click “{label}”, {_region_phrase(hit)}.")
            time.sleep(0.45)
        return " ".join(bits)
    if named:
        want = named[0]
        title = (meta.get("frontmost_title") or "").strip()
        page = f" “{title}”" if title else " this page"
        return f"I can't see “{want}” on{page}. Nothing to click there."
    return "I couldn't find that on the screen to point at."


def _product_phrase(meta: dict, lines: list[dict]) -> str:
    """Name the thing they are looking at, the way a person would.

    "This is a Microsoft Word window." — the product, first, in one
    sentence. That is what someone asking "what's on my screen" wants
    identified before anything else, and it is the opening move of
    teaching: establish what we are both looking at. Everything else the
    old answer volunteered (which other apps are open, the largest text,
    filenames) is detail nobody asked for, and read aloud it buried the
    one fact that mattered.
    """
    raw_app = meta.get("frontmost", "") or ""
    app = APP_NAMES.get(raw_app, raw_app)
    title = (meta.get("frontmost_title") or "").strip()
    kind = _front_kind(raw_app)

    if not app:
        return "I can see the screen, but not which app is in front."
    if kind == "browser":
        # Naming the browser is useless on its own — measured from a real
        # turn, the answer came back "This is Chrome, on “an untitled
        # tab”." and told the user nothing about the page they were
        # looking at. In a browser the *page* is the product, so say what
        # the page is. The heading is the largest prose on screen, read
        # straight off the OCR: no model call, no invention, and it is
        # what a person would name.
        # The window title first: a browser sets it to the page title, so
        # it is the page's own name rather than whatever happened to be
        # rendered largest. Falling back to the biggest prose on screen
        # picked "History [edit 1" off a scrolled Wikipedia article — a
        # real section heading, and not what the page is.
        tab = title if title and title.lower() not in ("window", "untitled") else ""
        if tab:
            return f"This is a page in {app}: “{tab}”."
        headings = screen_capture.prominent_lines(lines, count=1)
        if headings:
            heading = re.sub(r"\s*\[\s*edit[^\]]*\]?\s*\d*", " ", headings[0])
            heading = re.sub(r"\s+", " ", heading).strip()[:90]
            if heading:
                return f"This is a page in {app}: “{heading}”."
        return f"This is {app}, on a page with no readable heading."
    if kind == "picture":
        return f"This is a picture open in {app}."
    if kind == "app" and len(screen_capture.ocr_text(lines)) < 80:
        return f"This is {app}, showing a picture."
    return f"This is a {app} window."


def _must_read_pixels(prompt: str, lines: list[dict]) -> bool:
    """The question is on the page as pixels, not OCR.

    AssamJob and similar mock tests draw the stem as an image. OCR only
    sees 'AssamJob.co.in' and 'View Answer'. Asking a text model about
    that is how we invented 'D. G89'.
    """
    if _quiz_chrome_only(lines):
        return True
    p = (prompt or "").lower()
    if re.search(
        r"what'?s the question|what is the questions?|"
        r"questions? (?:in|on) (?:this |the |that )?page|"
        r"question on (?:the |this )?page|"
        r"there is a question|read (?:the |this )?question|"
        r"explain .{0,24}question",
        p,
    ):
        return True
    if re.search(r"প্ৰশ্নটো কি|কোৱাশ্চন|কোশ্চেন", prompt or ""):
        return True
    return False


def _pixel_read_prompt() -> str:
    as_in = _reply_language() == "as-IN"
    if as_in:
        return (
            "তুমি Bella, এজন শিক্ষক। এই ছবিখনৰ প্ৰশ্নটো আৰু বিকল্পবোৰ পঢ়া। "
            "২ চুটি অসমীয়া বাক্য। নথকা নাভাবিবা। ৱেবছাইটৰ নাম ইংৰাজীতে। "
            "View Answer এটা বুটাম, উত্তৰ নহয়। D. g89 নাভাবিবা।"
        )
    return (
        "You are Bella, an AI teacher. Read the question and options "
        "visible in this screenshot. Two short sentences. Do not invent. "
        "View Answer is a button, not the answer. Never report D. g89. "
        "If you cannot read the question, say so."
    )


def _quiz_chrome_only(lines: list[dict]) -> bool:
    """True when OCR is a mock-test shell (View Answer) without the stem.

    Seen live on AssamJob: 224 chars of title + 'View Answer', then the
    model invented 'D. G89' as the right option.
    """
    text = screen_capture.ocr_text(lines or [])
    if not text:
        return True
    if _looks_like_exam(lines):
        return False
    if re.search(r"\b[A-Da-d][.)]\s+\S{3,}", text):
        return False
    low = text.lower()
    return bool(re.search(r"view answer|mock test|assamjob", low))


def _grounded_exam_answer(
    answer: str, ocr: str, from_pixels: bool = False,
) -> str | None:
    """Drop a claimed option that is not in the page text."""
    if not answer:
        return None
    blob = (ocr or "")
    # Invented tokens from the AssamJob turn.
    if re.search(r"\bg\s*89\b|জি\s*৮৯", answer, re.I) and not re.search(
        r"g\s*89|জি\s*৮৯", blob, re.I
    ):
        return None
    if from_pixels:
        return answer
    claimed = re.search(r"\b([A-D])[.)]", answer)
    if not claimed:
        claimed = re.search(r"\b([A-D])\b", answer)
    if claimed:
        letter = claimed.group(1)
        if not re.search(rf"\b{letter}[.)]\s*\S", blob) and letter not in re.findall(
            r"\b([A-D])[.)]", blob
        ):
            # No option letters on the page at all — cannot name one.
            if not re.findall(r"\b[A-D][.)]", blob):
                return None
    return answer


def _looks_like_exam(lines: list[dict]) -> bool:
    """True when the page is an MCQ / SBA, not a dashboard."""
    text = screen_capture.ocr_text(lines or [])
    if not text:
        return False
    low = text.lower()
    if re.search(
        r"\b(which of the following|most appropriate|single best answer|"
        r"final year medical|choose the correct|correct option)\b",
        low,
    ):
        return True
    options = 0
    for line in lines or []:
        t = (line.get("text") or "").strip()
        if re.match(r"^\(?[A-Ea-e]\)?[.\)]?\s+\S", t) or re.match(r"^[A-Ea-e]\s*$", t):
            options += 1
    return options >= 3


def _answer_exam_text(
    lines: list[dict], meta: dict, question: str, cfg: dict,
) -> str:
    """Answer the on-screen exam item from OCR. No second full-screen send."""
    ocr = screen_capture.ocr_text(lines)[:OCR_FOR_MODEL_CHARS]
    if _quiz_chrome_only(lines):
        raise VisionError("the options are not in the page text")
    app = (meta or {}).get("frontmost") or session_app() or "this app"
    title = (meta or {}).get("frontmost_title") or ""
    as_in = _reply_language() == "as-IN"
    if as_in:
        teacher = (
            "তুমি Bella, এজন শিক্ষক। স্ক্ৰিনৰ প্ৰশ্নটোৰ সঠিক অপচন আখৰেৰে "
            "আৰু এটা চুটি কাৰণেৰে কোৱা। লিখাত নথকা অপচন নাভাবিবা। "
            "১–২ চুটি অসমীয়া বাক্য। ৱেবছাইটৰ নাম ইংৰাজীতে। "
            f"এপ: {app}{(' — ' + title) if title else ''}\n"
            f"তেওঁ সুধিলে: {question or 'সঠিক উত্তৰটো কি?'}\n"
            f"\nস্ক্ৰিনৰ লিখা:\n---\n{ocr}\n---\n"
        )
    else:
        teacher = (
        "You are Bella, an AI teacher sitting with a student at a Mac.\n"
        "The screen has an exam or textbook question. "
        "Give the correct option first (letter and a short name), then "
        "one short reason. Do not restate the whole stem. "
        "Do not invent options that are not in the text. "
        "One or two short spoken sentences. No markdown, no lecture.\n\n"
        f"App: {app}"
        f"{' — ' + title if title else ''}\n"
        f"They asked: {question or 'what is the answer?'}\n"
        f"\nText on screen:\n---\n{ocr}\n---\n"
    )
    if len(re.sub(r"\s+", "", ocr)) < 40:
        raise VisionError("the question on screen is not readable as text")
    try:
        answer = llm.chat(
            [{"role": "user", "content": teacher}],
            llm.resolve(cfg, "chat"),
            cfg,
            options={"num_predict": 120 if as_in else _screen_token_budget(cfg)},
        ).strip()
    except llm.LLMError as e:
        raise VisionError(f"Could not work the question out: {e}") from e
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer:
        raise VisionError("no answer came back")
    grounded = _grounded_exam_answer(answer, ocr)
    if grounded is None:
        if as_in:
            return "পৃষ্ঠাত বিকল্পবোৰ পঢ়া নগ'ল। মই উত্তৰটো অনুমান নকৰোঁ।"
        return (
            "I can see the quiz chrome, not the options. "
            "I won't guess. Open the answer on the page or ask me to look again."
        )
    return grounded


def _summarize_screen(meta: dict, lines: list[dict], cfg: dict | None = None) -> str:
    """"What's on my screen?" — name it, then ASK. Do not act yet.

    The look is step one. They then say what to do (explain, answer,
    point at a tab). The ring does not move on this turn. The old
    prompt forbade questions and the code then stripped any trailing
    '?', so she never waited for instructions.
    """
    where = _product_phrase(meta, lines)
    ocr = screen_capture.ocr_text(lines)[:OCR_FOR_MODEL_CHARS]
    as_in = _reply_language() == "as-IN"
    ask = "ইয়াৰ পিছত কি কৰিব খোজা?" if as_in else _NEXT_ASK
    if cfg is None or len(re.sub(r"\s+", "", ocr)) < 40:
        return _with_next_ask(where, ask)
    if as_in:
        prompt = (
            "তুমি Bella। ১টা চুটি অসমীয়া বাক্যত এই পৃষ্ঠা কি কোৱা। "
            "ৱেবছাইটৰ নাম ইংৰাজীতে ৰাখিবা, আখৰে আখৰে নকবা। "
            "উত্তৰ নিদিবা, অপচন নাভাবিবা। শেষত সুধিবা নাই — কোডে সুধিব।\n"
            f"সত্য: {where}\n"
            f"লিখা:\n---\n{ocr}\n---\n"
        )
    else:
        prompt = (
        "You are Bella, looking at their Mac with them.\n"
        f"Opening fact (true, do not contradict it): {where}\n"
        "Text read off the screen:\n---\n"
        f"{ocr}\n---\n"
        "In one or two short spoken sentences, say what this page or "
        "window is. If there is a question or problem visible, name the "
        "topic in one clause — do NOT give the answer, do NOT list "
        "options, do NOT tell them where to click. Use only the text "
        "above. Do not invent. Do not say you cannot see the screen. "
        "No markdown. Do not ask anything yet; a follow-up ask is added "
        "in code."
    )
    try:
        answer = llm.chat(
            [{"role": "user", "content": prompt}],
            llm.resolve(cfg, "chat"),
            cfg,
            options={"num_predict": 80 if as_in else _screen_token_budget(cfg)},
        ).strip()
    except llm.LLMError as e:
        _log(f"look failed ({e}) — naming the page only")
        return _with_next_ask(where, ask)
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer:
        return _with_next_ask(where, ask)
    return _with_next_ask(answer, ask)


def _ask_about_text(lines: list[dict], meta: dict, question: str, cfg: dict) -> str:
    """Answer a *specific* question from the recognized screen text.

    The model's job here is narrow on purpose: find the thing the user
    asked about in text that is already known to be accurate, and say it.
    Asking the same small model to summarize the whole desktop produced
    inventions every time it was tried (a URL that wasn't on screen, an
    integration that doesn't exist), which is why the generic question is
    answered by _summarize_screen instead and never reaches this.
    """
    # screen_text_model still pins a specific local model when set;
    # otherwise the dashboard's "screen" role decides.
    pinned = cfg.get("screen_text_model")
    entry = llm.resolve(cfg, "screen")
    if pinned:
        entry = {**entry, "provider": "ollama", "model": pinned, "label": f"Bella · {pinned}"}
    # OCR of a dense screen can run long; the local model's context is the
    # limit, and the top of the reading order is where the meaningful
    # content is.
    text = screen_capture.ocr_text(lines)[:OCR_FOR_MODEL_CHARS]

    windows = "; ".join(
        f"{w['app']}: {w['title']}" for w in meta.get("windows", [])[:8]
    )
    prompt = (
        "Here is what is on the user's Mac screen right now.\n\n"
        f"Active window: {meta.get('frontmost', 'unknown')}"
        f"{' — ' + meta['frontmost_title'] if meta.get('frontmost_title') else ''}\n"
        f"Also open: {windows}\n\n"
        f"Text read off the screen:\n---\n{text}\n---\n\n"
        f"The user asked: {question}\n\n"
        "Answer in one or two short spoken sentences, using only what is above. "
        "If they asked you to show a control, name it and where it is — do not "
        "claim you clicked. If it is not on the screen, say so. No lists."
    )
    return _screen_chat(prompt, cfg, _screen_token_budget(cfg))


def describe_region(question: str, cfg: dict, interactive: bool = False,
                    confirm_fn=None) -> str:
    """Let them circle something, then answer about only that.

    The whole point is the narrowing: asked about a 3456x2234 desktop the
    models describe the desktop, and any one thing on it gets averaged
    away. Given a 400x300 crop of a diagram they describe the diagram.

    Route inside the crop is the same judgement describe_screen makes for
    the whole screen — text if there is text, the vision model if there
    is not — except the threshold is far lower, because a circled region
    is small by definition and a caption of a dozen words is still text
    worth answering from.
    """
    if not cfg.get("vision_enabled", True):
        return "Screen understanding is disabled in config."

    region = screen_capture.select_region()
    if not region:
        return "Nothing circled — ask me again and draw a ring around it."

    try:
        jpeg = screen_capture.capture_region(
            region["x"], region["y"], region["w"], region["h"]
        )
    except screen_capture.ScreenCaptureError as e:
        raise VisionError(f"Could not capture that area: {e}") from e
    _log(f"region {region['w']:.0f}x{region['h']:.0f} at "
         f"({region['x']:.0f},{region['y']:.0f}), {len(jpeg)} bytes")

    lines = screen_capture.ocr(jpeg) if cfg.get("screen_ocr_enabled", True) else []
    text = screen_capture.ocr_text(lines)
    asked = (question or "").strip() or "What is this?"

    # 40 characters, not describe_screen's 220: this is one circled thing,
    # so a title or a label is the whole content and there is no desktop
    # furniture inflating the count.
    if len(text) >= 40:
        entry = llm.resolve(cfg, "chat")
        prompt = (
            "The user circled one area of their screen and asked about it. "
            "Everything below was read from inside that area — nothing else "
            "on their screen is included.\n\n"
            f"---\n{text[:3000]}\n---\n\n"
            f"They asked: {asked}\n\n"
            "Answer in one or two short spoken sentences, from the text above "
            "only. If it does not answer the question, say what the area "
            "contains instead. No lists, no markdown."
        )
        try:
            answer = llm.chat([{"role": "user", "content": prompt}], entry, cfg,
                              options={"num_predict": _screen_token_budget(cfg)}).strip()
            _log("answered from region text")
            _panel_region(answer, asked)
            return answer
        except llm.LLMError as e:
            raise VisionError(f"Could not reach {entry['label']}: {e}") from e

    # Little or no text — a photo, a diagram, an anatomy figure. This is
    # what the vision model is for, and a crop is where it performs best:
    # a small image of one subject, rather than a downscaled desktop.
    vision_prompt = (
        f"{asked}\n\nThis image is a region the user circled on their screen. "
        "Say what it shows in one or two short sentences."
    )
    try:
        answer = _ask_vision_model(jpeg, vision_prompt, cfg)
        _log("answered from the vision model on the region")
        _panel_region(answer, asked)
        return answer
    except VisionError:
        if not (cfg.get("screen_cloud_fallback", False) and interactive and confirm_fn):
            raise
        if not confirm_fn("The local vision model couldn't read that area. "
                          "Send just that cropped area to the cloud model (a paid call)?"):
            return "Kept it local. The local vision model couldn't read that area."
        answer = _ask_cloud_vision(jpeg, vision_prompt, cfg)
        _log("answered from the cloud vision model on the region")
        _panel_region(answer, asked)
        return answer


def _panel_region(answer: str, question: str) -> str:
    card = instructions.build_card([], fallback=answer or "")
    if card:
        instructions.show(card, app=_SESSION["raw_app"], title="SELECTED AREA")
    return answer


def screen_context(max_age: float = 20.0) -> tuple[str, str]:
    """(text on screen, frontmost app) for something that needs to know
    what the user is looking at — "write a prompt to review this code".

    Reuses the last capture when it is fresh, because this is called on
    the way to a model call that already costs a second or two and the
    screen rarely changes in the meantime. Returns empty strings rather
    than raising: the caller can still do its job without context.
    """
    fresh = _LAST_LINES and (time.monotonic() - _SESSION["at"]) < max_age
    if not fresh:
        try:
            jpeg, meta = screen_capture.capture()
        except screen_capture.ScreenCaptureError as e:
            _log(f"no screen context: {e}")
            return "", _SESSION["raw_app"]
        _remember_screen(screen_capture.ocr(jpeg))
        _open_session(meta)
    return screen_capture.ocr_text(_LAST_LINES), _SESSION["raw_app"]


def _lines_in_front_window(
    lines: list[dict], meta: dict, width: int, height: int
) -> list[dict]:
    """OCR lines whose centre falls inside the frontmost window.

    screen_capture.ocr_line_point returns the same global, top-left,
    logical-point space that frontmost_bounds is reported in, so the two
    compare directly with no scaling maths here.
    """
    bounds = meta.get("frontmost_bounds") or {}
    bw, bh = bounds.get("w") or 0, bounds.get("h") or 0
    if not (bw and bh and width and height):
        return list(lines or [])
    bx, by = bounds.get("x") or 0, bounds.get("y") or 0
    inside: list[dict] = []
    for line in lines or []:
        point = screen_capture.ocr_line_point(line, width, height)
        if point is None:
            continue
        x, y = point
        if bx <= x <= bx + bw and by <= y <= by + bh:
            inside.append(line)
    return inside


def _page_lines(lines: list[dict], meta: dict) -> list[dict]:
    """OCR of the front page body — not the menu bar, not other windows.

    The full desktop capture is for finding this region. The model only
    gets these lines, so a medical question is not drowned in Safari
    chrome and a 3456px screenshot never leaves the machine as a payload.
    """
    width = int((meta or {}).get("width") or 0)
    height = int((meta or {}).get("height") or 0)
    front = _lines_in_front_window(lines, meta, width, height)
    source = front or list(lines or [])
    kept: list[dict] = []
    for line in source:
        text = (line.get("text") or "").strip()
        if not text:
            continue
        if _is_chrome_label(text):
            continue
        if width and height and _is_menubar_line(line, meta or {}, width, height):
            continue
        kept.append(line)
    return kept or front or list(lines or [])


def _content_rect(lines: list[dict], meta: dict) -> tuple[float, float, float, float] | None:
    """Padded global CG box around the page body."""
    width = int((meta or {}).get("width") or 0)
    height = int((meta or {}).get("height") or 0)
    page = _page_lines(lines, meta)
    xs: list[float] = []
    ys: list[float] = []
    rights: list[float] = []
    bottoms: list[float] = []
    for line in page:
        box = screen_capture.ocr_line_rect(line, width, height) if width and height else None
        if not box:
            continue
        x, y, w, h = box
        xs.append(x)
        ys.append(y)
        rights.append(x + w)
        bottoms.append(y + h)
    bounds = (meta or {}).get("frontmost_bounds") or {}
    if xs:
        pad = 24.0
        x0 = max(0.0, min(xs) - pad)
        y0 = max(0.0, min(ys) - pad)
        x1 = max(rights) + pad
        y1 = max(bottoms) + pad
        bw, bh = bounds.get("w") or 0, bounds.get("h") or 0
        if bw and bh:
            bx, by = bounds.get("x") or 0, bounds.get("y") or 0
            # Keep the crop inside the window, below the title/tab strip.
            x0 = max(x0, bx)
            y0 = max(y0, by + 64.0)
            x1 = min(x1, bx + bw)
            y1 = min(y1, by + bh)
        w, h = x1 - x0, y1 - y0
        if w >= 80 and h >= 80:
            return x0, y0, w, h
    bw, bh = bounds.get("w") or 0, bounds.get("h") or 0
    if bw >= 80 and bh >= 80:
        return (
            bounds.get("x") or 0,
            (bounds.get("y") or 0) + 72.0,
            bw,
            max(80.0, bh - 72.0),
        )
    return None


def _focus_jpeg(meta: dict, lines: list[dict], fallback: bytes | None) -> bytes:
    """Crop of the page body, capped at 1024px wide. Empty bytes if it fails."""
    rect = _content_rect(lines, meta)
    if not rect:
        bounds = (meta or {}).get("frontmost_bounds") or {}
        if bounds.get("w"):
            try:
                cropped = screen_capture.capture_region(*bounds, max_width=1024)
                if cropped:
                    _log(f"window crop {bounds.get('w')}x{bounds.get('h')}")
                    return cropped
            except Exception as e:
                _log(f"window crop failed ({e})")
        return fallback or b""
    x, y, w, h = rect
    try:
        cropped = screen_capture.capture_region(x, y, w, h, max_width=1024)
        if cropped:
            _log(f"page crop {w:.0f}x{h:.0f} at ({x:.0f},{y:.0f}), {len(cropped)} bytes")
            return cropped
    except Exception as e:
        _log(f"page crop failed ({e})")
    return fallback or b""


def _refresh_for_point(cfg: dict) -> dict:
    """Fresh capture so the ring lands on the control as it is now.

    Seen live: how-to skipped the screenshot, the model then returned
    broken JSON, and the pointer never moved. Coordinates have to come
    from this look, not from a session that is minutes old.
    """
    hud = (cfg or {}).get("screen_hud_window_marker", "Bella HUD")
    jpeg, meta = screen_capture.capture(hud_marker=hud)
    lines = []
    if (cfg or {}).get("screen_ocr_enabled", True):
        lines = screen_capture.ocr(jpeg)
        _log(f"guide recapture OCR {len(lines)} lines")
    _remember_screen(lines)
    _open_session(meta)
    return meta


def explain_howto(question: str, cfg: dict) -> str:
    """Teach one task in the app they are already in, and POINT at it.

    The spoken answer can still come from knowledge of the app. The
    pointer cannot: it needs a live OCR box. Recapture first, move the
    ring, then talk. If the model returns nothing, the ring still moved.
    """
    pointed = ""
    meta: dict = {}
    try:
        meta = _refresh_for_point(cfg)
        pointed = _point_at_front(meta, _LAST_LINES, question)
        if pointed:
            _log(pointed)
    except screen_capture.ScreenCaptureError as e:
        _log(f"guide recapture failed ({e}) — teaching without a pointer")
    except Exception as e:
        _log(f"could not point during howto: {e}")

    app = session_app() or "the app they are using"
    # The CHAT model, not the screen one. Every other path here hands the
    # model text that is already known to be true and asks it to find
    # something in it; this one asks what a particular application's
    # interface looks like, which is world knowledge, and small models
    # fabricate it confidently. Measured on "how do I bold text" in VS
    # Code: Groq 8B answered "the Format panel under the Text tab" — no
    # such thing — while 70B correctly returned no control at all and gave
    # the shortcut. Teaching someone a button that does not exist is worse
    # than any latency this saves.
    entry = llm.resolve(cfg, "chat")

    # Two separate outputs on purpose. "say" is conversation — it is heard
    # once and gone. "steps" is reference — it stays on screen while the
    # user works, so it must read like a manual entry, not like half a
    # dialogue. Generating one from the other never worked: stripping
    # pleasantries out of a spoken sentence leaves a spoken sentence.
    prompt = (
        f"You are Bella, teaching someone sitting at a Mac using {app}.\n"
        f"They asked: {question}\n\n"
        "Answer as JSON only, no prose around it:\n"
        '{"control": "<exact on-screen name of the button or menu item, or ""'
        ' if it is keyboard-only>",\n'
        ' "say": "<two short spoken sentences>",\n'
        ' "steps": ["<short instruction line>", "..."]}\n\n'
        '"say" is spoken aloud: the macOS keyboard shortcut first, then where '
        "the control is.\n\n"
        '"steps" is printed on screen as a reference card while they work. '
        "Write it like a manual, never like conversation:\n"
        "- 2 to 4 lines, each under 60 characters.\n"
        '- Start with the shortcut line, e.g. "Shortcut:  Command + B".\n'
        '- Then where the control is, e.g. "Ribbon:  Home tab, Font group".\n'
        "- Then any precondition, e.g. \"Select the text first\".\n"
        "- Imperative or label-style only. NO greetings, NO questions, NO "
        '"you can", NO "I", NO "let me", NO offers to help, NO commentary.\n\n'
        f"- Use the real shortcut for {app} on macOS. Never invent one, and "
        "never invent a button that does not exist — if you are not sure a "
        'control exists, give the shortcut and set "control" to "".\n'
        "- No markdown. Do not claim to have done it or to be clicking anything."
    )
    try:
        raw = llm.chat(
            [{"role": "user", "content": prompt}], entry, cfg,
            options={"num_predict": _screen_token_budget(cfg)},
        ).strip()
    except llm.LLMError as e:
        if pointed:
            return pointed
        raise VisionError(f"Could not reach {entry['label']} to work that out: {e}") from e

    control, say = "", ""
    steps: list[str] = []
    start = raw.find("{")
    if start != -1:
        try:
            # raw_decode, not a first-brace-to-last-brace slice: the 8B
            # returned two JSON objects back to back, and spanning them
            # produced text that parsed as neither.
            data, _ = json.JSONDecoder().raw_decode(raw[start:])
            if isinstance(data, dict):
                control = str(data.get("control") or "").strip()
                say = re.sub(r"\s+", " ", str(data.get("say") or "")).strip()
                raw_steps = data.get("steps")
                if isinstance(raw_steps, list):
                    steps = [
                        re.sub(r"\s+", " ", str(s)).strip(" -•*")
                        for s in raw_steps if str(s).strip()
                    ]
        except json.JSONDecodeError:
            pass
    if not say:
        # The model ignored the format. Its prose is still the answer —
        # but only if it IS prose. Seen live, a truncated reply left this
        # holding '{"control' and Bella read that out loud. Half a JSON
        # object is not an answer, and speaking it is worse than saying
        # nothing.
        candidate = re.sub(r"\s+", " ", raw).strip()
        say = "" if candidate.lstrip().startswith(("{", "[", '"control')) else candidate
    if not say:
        if pointed:
            return pointed
        raise VisionError("no answer came back")

    title = _panel_title(question)
    card = instructions.build_card(steps, fallback=say)

    global _PENDING_SHOW
    _PENDING_SHOW = None
    # Only offer to point at something we can actually see. The control
    # name is the one part of this answer that is a claim about THIS
    # screen rather than general knowledge, so it gets checked against the
    # text really read off it. Without this, an invented control produced
    # an invented offer, and "yes" then led to "I couldn't find it" — Bella
    # wasting the user's turn on its own mistake.
    already = (pointed or "").lower()
    if control and _best_ocr_match(_LAST_LINES, control):
        hit = _best_ocr_match(_LAST_LINES, control)
        w, h = int(_SESSION.get("width") or 0), int(_SESSION.get("height") or 0)
        if hit and w and h:
            label = re.sub(r"\s+", " ", (hit.get("text") or control)).strip()
            if label.lower() not in already:
                screen_capture.point_at_ocr_line(
                    hit, w, h, label=control, seconds=20.0)
            say = f"{say} I'm pointing at “{control}”."
        elif not pointed:
            _PENDING_SHOW = {"control": control, "question": question,
                             "card": card, "title": title}
            say = f"{say} Want me to show you where it is?"
    elif pointed:
        if pointed not in say:
            say = f"{pointed} {say}".strip()
    elif control:
        _log(f"not offering to point: {control!r} is not in the screen text")
    _SESSION["at"] = time.monotonic()
    if card:
        instructions.show(card, app=_SESSION["raw_app"], title=title)
    return say


_ADVICE_RE = re.compile(
    r"\b(?:"
    r"what(?:'s| is| should)?\s+(?:my|the|your)?\s*(?:next|best)\s+move|"
    r"what should i (?:do|play|move)|what do i do (?:now|next)|"
    r"which (?:move|one) should i|"
    # "can you help me with the next move?" — the phrasing actually used,
    # which matched nothing here and fell through to the how-to path,
    # where it asked a text model about a chess board it could not see.
    # Every one of these is a phrasing that was actually said to Bella and
    # matched nothing, falling through to the how-to path where a text
    # model was asked about a board it could not see. The verb varies far
    # more than the noun does, so the noun is what anchors this.
    r"(?:help me|helping me)\s+(?:to\s+)?(?:\w+\s+){0,3}?(?:next |best |good )?move|"
    r"(?:identify|find|pick|choose|work out|figure out|tell me)\s+"
    r"(?:\w+\s+){0,3}?(?:next |best |good )?move|"
    r"(?:suggest|recommend|give me) (?:a |the |my )?(?:next |best )?move|"
    r"my (?:next|best) move|"
    r"help me (?:with )?(?:this|here|out)|"
    r"any (?:advice|suggestion|idea)|what would you (?:do|play)|"
    r"analy[sz]e (?:this|it|the (?:board|position|screen))|"
    r"how (?:am i|are we) doing|am i winning"
    r")\b",
    re.IGNORECASE,
)


def is_advice_request(text: str) -> bool:
    """Asking for a judgement about what is on screen, not for a fact."""
    return bool(_ADVICE_RE.search(text or ""))


# Seen live: Safari was on "320 Single Best Answer Questions for Final
# Year Medical Students". "can you help me to identify the right answer?"
# matched is_howto ("can you help me"), asked how to use Safari, and
# failed. That is a teacher question about the page, not a UI lesson.
_QUESTION_HELP_RE = re.compile(
    r"\b(?:"
    r"(?:right|correct|best)\s+answer|"
    r"which\s+(?:option|choice|answer|one)|"
    r"what(?:'s| is)\s+the\s+answer|"
    r"answer\s+(?:this|the|these)\s+(?:question|one)|"
    r"help\s+me\s+(?:to\s+)?(?:identify|find|pick|choose|answer)|"
    r"identify\s+the\s+(?:right|correct|best)|"
    r"solve\s+(?:this|it|the\s+(?:question|problem))|"
    r"(?:explain|read)\s+(?:this|the|)\s*question|"
    r"explain\s+(?:it|this|that)\s+to\s+me"
    r")\b|"
    r"সঠিক উত্তৰ|উত্তৰটো কি|কোনটো (?:অপচন|বিকল্প|ঠিক)|"
    r"এই প্ৰশ্নটোৰ উত্তৰ",
    re.IGNORECASE,
)


def is_question_help(text: str) -> bool:
    """Asking Bella to read an on-screen exam question and answer it."""
    return bool(_QUESTION_HELP_RE.search(text or ""))


def advise_screen(question: str, cfg: dict) -> str:
    """Look at the screen and give an opinion about it.

    Every other screen path reasons over OCR text. That is right for a
    menu or a document and useless for anything positional: asked what to
    play next, a text model got 48 lines of OCR from Chess.app that say
    nothing about where the pieces are, and answered "control the centre
    and develop your pieces" — true of every chess position ever played.

    This one sends the pixels. 1024px wide because that is what the local
    vision model can actually read — measured, at 768 it invented a line
    of C++ that was not on screen, and at full 3456px it returned a
    fragment of a URN.
    """
    image, facts = screen_capture.capture(
        hud_marker=cfg.get("screen_hud_window_marker", "Bella HUD"), max_width=1024)
    app = facts.get("frontmost") or session_app() or "this app"
    # Prefer just the window they are working in. The desktop around it is
    # noise the model will otherwise narrate back at them.
    bounds = screen_capture.frontmost_window_bounds(app)
    if bounds:
        try:
            cropped = screen_capture.capture_region(*bounds, max_width=1024)
            if cropped:
                image = cropped
                _log(f"advising on the {app} window, not the whole desktop")
        except Exception as e:
            _log(f"window crop failed ({e}) — using the full screen")
    prompt = (
        f"You are Bella, looking over the shoulder of someone using {app} on a Mac.\n"
        f"They asked: {question}\n\n"
        "Look at the screenshot and answer from what you can actually see.\n"
        "- Be specific about what is on screen. Name real things you can see.\n"
        "- If it is a game, say what you would play and why, in one line. Name "
        "the move the way a person says it out loud — \"knight to f3\", not "
        "\"Nf3\" and not \"the third square on the right\".\n"
        "- Two or three short sentences, spoken aloud, no markdown.\n"
        "- If you genuinely cannot make out the detail you would need, say "
        "so plainly instead of guessing."
    )
    try:
        answer = llm.look(prompt, image, cfg).strip()
    except llm.LLMError as e:
        raise VisionError(f"I couldn't get a look at that: {e}") from e
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer:
        raise VisionError("nothing came back from the screenshot")

    _SESSION["at"] = time.monotonic()
    card = instructions.build_card([], fallback=answer)
    if card:
        instructions.show(card, app=_SESSION.get("raw_app") or app, title="ON SCREEN")
    return answer


def answer_on_screen(question: str, cfg: dict) -> str:
    """Read the exam question on screen and answer it as a teacher.

    Prefer OCR of the front window plus the chat model. A scanned book
    page has almost no OCR — seen live on the medical SBA Safari tab —
    so fall back to looking at the pixels instead of saying we cannot
    read the screen.
    """
    hud_marker = cfg.get("screen_hud_window_marker", "Bella HUD")
    try:
        jpeg, meta = screen_capture.capture(hud_marker=hud_marker)
    except screen_capture.ScreenCaptureError as e:
        raise VisionError(f"Could not look at the screen: {e}") from e
    app = meta.get("frontmost") or session_app() or "this app"
    lines = screen_capture.ocr(jpeg) if cfg.get("screen_ocr_enabled", True) else []
    page = _page_lines(lines, meta)
    focus = _focus_jpeg(meta, page, b"")
    if focus:
        jpeg = focus
        try:
            crop_lines = screen_capture.ocr(jpeg)
            if len(screen_capture.ocr_text(crop_lines)) >= 40:
                page = crop_lines
        except Exception as e:
            _log(f"crop OCR failed ({e}) — using page lines")
    else:
        bounds = screen_capture.frontmost_window_bounds(app)
        if bounds:
            try:
                cropped = screen_capture.capture_region(*bounds, max_width=1024)
                if cropped:
                    jpeg = cropped
            except Exception as e:
                _log(f"window crop failed ({e}) — using the full screen")
    lines = page
    _remember_screen(lines)
    _open_session(meta)
    ocr = screen_capture.ocr_text(lines)[:OCR_FOR_MODEL_CHARS]
    from_pixels = False
    usable = len(re.sub(r"\s+", "", ocr)) >= 40 and not _quiz_chrome_only(lines)
    if usable:
        try:
            answer = _answer_exam_text(lines, meta, question, cfg)
        except VisionError:
            usable = False
            answer = ""
    if not usable:
        _log("exam OCR too thin — looking at the cropped page")
        from_pixels = True
        as_in = _reply_language() == "as-IN"
        teacher = (
            "তুমি Bella। স্ক্ৰিনশ্বটৰ প্ৰশ্নটোৰ সঠিক অপচন আখৰেৰে কোৱা, "
            "এটা চুটি কাৰণ। লিখাত নথকা নাভাবিবা। ১–২ অসমীয়া বাক্য।"
            if as_in else
            "You are Bella, an AI teacher. This screenshot is an exam "
            "question. Give the correct option first, then one reason. "
            "Do not invent. One or two short sentences."
        )
        try:
            answer = llm.look(teacher, jpeg, cfg).strip()
        except llm.LLMError as e:
            raise VisionError(f"I couldn't get a look at that: {e}") from e
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer:
        raise VisionError("no answer came back")
    grounded = _grounded_exam_answer(answer, ocr, from_pixels=from_pixels)
    if grounded is None:
        as_in = _reply_language() == "as-IN"
        answer = (
            "পৃষ্ঠাত বিকল্পবোৰ পঢ়া নগ'ল। মই উত্তৰটো অনুমান নকৰোঁ।"
            if as_in else
            "I can see the quiz chrome, not the options. I won't guess."
        )
    else:
        answer = grounded
    _SESSION["at"] = time.monotonic()
    card = instructions.build_card([], fallback=answer)
    if card:
        instructions.show(
            card, app=_SESSION.get("raw_app") or app, title="ANSWER"
        )
    return answer


def show_pending(cfg: dict) -> str:
    """Point Bella's marker at the control that was just offered.

    Takes a fresh capture on purpose: this is the one step where being
    wrong by a few centimetres is worse than being a second slower, and
    toolbars move when a window is resized or a tab is switched.
    """
    pending = _PENDING_SHOW
    if not pending:
        return "There's nothing I offered to point at just now."
    control = pending.get("control") or ""

    hud_marker = cfg.get("screen_hud_window_marker", "Bella HUD")
    try:
        jpeg, meta = screen_capture.capture(hud_marker=hud_marker)
    except screen_capture.ScreenCaptureError as e:
        raise VisionError(f"Could not look at the screen: {e}") from e
    lines = screen_capture.ocr(jpeg) if cfg.get("screen_ocr_enabled", True) else []
    _remember_screen(lines)
    _open_session(meta)

    width, height = int(meta.get("width") or 0), int(meta.get("height") or 0)
    # Match inside the app being taught, not the whole desktop. Seen while
    # testing: "Home" matched a Safari tab sitting behind the window in
    # question, and the ring landed on a browser instead of the toolbar
    # being explained. Falls back to the full screen, because a control in
    # the wrong place still beats no answer.
    front = _lines_in_front_window(lines, meta, width, height)
    hit = None
    if control:
        hit = _best_ocr_match(front, control) or _best_ocr_match(lines, control)
    if hit and width:
        label = re.sub(r"\s+", " ", (hit.get("text") or control)).strip()
        screen_capture.point_at_ocr_line(hit, width, height, label=label)
        clear_pending_show()
        _log(f"pointed at {label!r} for {control!r}")
        spoken = (
            f"There — my ring is on “{label}”, {_region_phrase(hit)}. "
            "Click it whenever you're ready."
        )
        # The card keeps the instruction it already had and gains the
        # location as one more reference line — it does not get replaced
        # by a sentence about what just happened, which would be
        # commentary rather than instruction.
        card = pending.get("card") or ""
        location = f"On screen:  {label} — {_region_phrase(hit)}"
        instructions.show(
            f"{card}\n{location}" if card else location,
            app=_SESSION["raw_app"],
            title=pending.get("title") or "INSTRUCTIONS",
        )
        return spoken

    clear_pending_show()
    _log(f"could not find {control!r} on screen to point at")
    missed = (
        f"I couldn't find “{control}” on screen just now — it may be under a "
        "menu or a collapsed toolbar. The keyboard shortcut still works."
    )
    # The instruction card stays exactly as it was: the shortcut on it is
    # still correct, and "I couldn't find it" is a remark about Bella, not
    # an instruction for the user.
    return missed


def _ask_cloud_vision(jpeg_bytes: bytes, prompt: str, cfg: dict) -> str:
    """Optional paid fallback (Gemini). Never reached without both
    `screen_cloud_fallback: true` and an explicit confirmation — same rule
    the Claude Code handoff follows, for the same reason."""
    api_key = env.load_env().get("GEMINI_API_KEY", "")
    if not api_key:
        raise VisionError("No GEMINI_API_KEY in bella/.env, so there's no cloud fallback to use.")
    model = cfg.get("screen_cloud_model", "gemini-2.0-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(jpeg_bytes).decode("ascii")}},
            ]
        }]
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise VisionError(f"Cloud vision call failed: {e}") from e
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise VisionError("The cloud vision model returned no usable answer.")


def describe_screen(
    question: str | None,
    cfg: dict,
    interactive: bool = False,
    confirm_fn=None,
) -> str:
    """Look at the screen and answer `question` about it.

    The HUD is kept out of the capture rather than being asked to move:
    see screen_capture.capture(). The hide/restore round trip below only
    runs when that native path is unavailable, since hiding the window the
    user is talking to is a cost worth paying only when it buys something.
    """
    prompt = question.strip() if question and question.strip() else DEFAULT_QUESTION
    hud_marker = cfg.get("screen_hud_window_marker", "Bella HUD")
    native = screen_capture.available()
    hide_ui = cfg.get("screen_hide_ui", True) and not native

    if hide_ui:
        # Only the fallback path needs this: screencapture photographs the
        # whole display, HUD included.
        voice_events.screen_capture_phase("start")
        time.sleep(cfg.get("screen_hide_delay_ms", 400) / 1000)

    try:
        jpeg, meta = screen_capture.capture(hud_marker=hud_marker)
    except screen_capture.ScreenPermissionError as e:
        raise VisionError(str(e)) from e
    except screen_capture.ScreenCaptureError as e:
        raise VisionError(f"Could not capture the screen: {e}") from e
    finally:
        if hide_ui:
            voice_events.screen_capture_phase("end")

    excluded = f", excluded {meta['excluded']}" if meta.get("excluded") else ""
    _log(
        f"captured {meta.get('width', '?')}x{meta.get('height', '?')} via "
        f"{meta['method']} in {meta.get('seconds')}s{excluded}; frontmost "
        f"{meta.get('frontmost') or 'unknown'}"
    )

    lines: list[dict] = []
    char_count = 0
    if cfg.get("screen_ocr_enabled", True):
        started = time.monotonic()
        lines = screen_capture.ocr(jpeg)
        char_count = len(screen_capture.ocr_text(lines))
        _log(
            f"OCR {len(lines)} lines / {char_count} chars in "
            f"{time.monotonic() - started:.2f}s"
        )
        page = _page_lines(lines, meta)
        if page is not lines:
            _log(f"page body {len(page)} lines / "
                 f"{len(screen_capture.ocr_text(page))} chars "
                 f"(dropped {len(lines) - len(page)} chrome/other-window)")
            lines = page
            char_count = len(screen_capture.ocr_text(lines))
        _remember_screen(lines)

    # A browser tab often mentions README.md; that is not a local file
    # they asked us to teach. Opening the project README then drowned
    # the actual GitHub page they were looking at.
    front_kind = _front_kind(meta.get("frontmost") or "")
    visible = [] if front_kind == "browser" else _filenames_on_screen(meta, lines)
    # Not for a summary: "summarise this page" is about what is
    # displayed, and opening local files that happen to share a name
    # with something on screen answers a question nobody asked —
    # seen live, a Wikipedia article summary went via index.html.
    loaded = ([] if (_is_generic(prompt) or _is_summarize(prompt))
              else _read_visible_files(visible) if visible else [])
    if visible:
        _log("files on screen: " + ", ".join(visible))
    if loaded:
        _log("opened " + ", ".join(p for p, _ in loaded))

    _open_session(meta)
    # Point ONLY when they asked to be shown something. This used to run on
    # every single screen question, so merely asking "what's on my screen"
    # sent the pointer skating across the desktop onto whatever matched
    # best — startling, and it moved the mouse out from under the user's
    # hand mid-task. Being shown a control is now something Bella offers and
    # they accept (see explain_howto / show_pending).
    pointed = _point_at_front(meta, lines, prompt) if _is_guide(prompt) else ""
    if pointed:
        _log(pointed)

    def spoken(answer: str) -> str:
        _remember_screen(lines, _hints_from_answer(answer or "") + _hints_from_answer(pointed or ""))
        # Every return path in describe_screen goes through here, which is
        # why the card is raised from this one place: hooking the
        # summarize branch alone missed it whenever a file on screen sent
        # the question down the teacher path instead.
        # The panel comes up for EVERY screen answer, not just summaries.
        # "What's on my screen" is answered from window facts with no model
        # call, and that branch never raised a card — so the one question
        # people actually ask was the one question that left the screen
        # blank. Title still says what kind of answer it is.
        card = instructions.build_card([], fallback=answer or "")
        if card:
            instructions.show(
                card,
                app=_SESSION.get("raw_app") or meta.get("frontmost", ""),
                title="SUMMARY" if _is_summarize(prompt) else "ON SCREEN",
            )
        # The ring moves only after they asked to be shown a control.
        # A generic look used to mark the window (or the menu bar) every
        # time, which is movement they did not instruct.
        if pointed and "can't see" not in pointed.lower() and pointed not in (answer or ""):
            return f"{pointed} {answer}".strip()
        return answer

    if _must_read_pixels(prompt, lines) and not _is_guide(prompt):
        _log("question not in OCR — reading the page pixels")
        try:
            payload = _small_capture(cfg, jpeg, meta, lines) or jpeg
            seen = llm.look(_pixel_read_prompt(), payload, cfg)
        except llm.LLMError as e:
            _log(f"pixel read failed ({e})")
            seen = ""
        if seen:
            ocr = screen_capture.ocr_text(lines)
            if _grounded_exam_answer(seen, ocr) is None and re.search(
                r"g\s*89|view answer", seen, re.I
            ):
                as_in = _reply_language() == "as-IN"
                seen = (
                    "প্ৰশ্নটো ছবি হিচাপে আছে, মই স্পষ্টকৈ পঢ়িব পৰা নাই। "
                    "View Answer টো উত্তৰ নহয়।"
                    if as_in else
                    "The question is on the page as an image. I couldn't "
                    "read it clearly. View Answer is a button, not the answer."
                )
            as_in = _reply_language() == "as-IN"
            ask = "ইয়াৰ পিছত কি কৰিব খোজা?" if as_in else _NEXT_ASK
            if _is_generic(prompt) or is_help_ask(prompt):
                return spoken(_with_next_ask(seen, ask))
            return spoken(seen.strip())

    if _is_generic(prompt) or is_help_ask(prompt):
        # Look first. They say what to do next. Do not answer the exam
        # and do not move the ring on this turn.
        _log("look — describe the page, then ask")
        return spoken(_summarize_screen(meta, lines, cfg))

    if _is_teacher(prompt) or loaded and not _is_generic(prompt):
        started = time.monotonic()
        answer = _teach_from_screen(prompt, meta, lines, visible, loaded, cfg)
        if _is_teacher(prompt) and not _is_guide(prompt):
            answer = _with_next_ask(answer, _EXPLAIN_ASK)
        _log(f"taught from screen + {len(loaded)} file(s) in {time.monotonic() - started:.2f}s")
        return spoken(answer)

    if char_count >= MIN_OCR_CHARS:
        started = time.monotonic()
        if loaded:
            answer = _teach_from_screen(prompt, meta, lines, visible, loaded, cfg)
        else:
            answer = _ask_about_text(lines, meta, prompt, cfg)
        _log(f"answered from screen text in {time.monotonic() - started:.2f}s")
        return spoken(answer)

    if loaded or _is_teacher(prompt):
        started = time.monotonic()
        answer = _teach_from_screen(prompt, meta, lines, visible, loaded, cfg)
        _log(f"taught from identified files in {time.monotonic() - started:.2f}s")
        return spoken(answer)

    # Little or no text: a photo, a video, a design. That is what the
    # vision model is actually good at.
    # llm.look picks the cloud vision model when a key is configured and
    # the local one only when it is not. That ordering matters for more
    # than quality: the local model saturates this machine's CPU, and the
    # speech recogniser shares it. Measured in one conversation while
    # moondream was running, STT went from 0.73s to 28.55s and the voice
    # loop printed "something else is using the CPU" itself.
    _log("little text on screen — looking at the image instead of the text")
    try:
        started = time.monotonic()
        # The OCR copy is full resolution (3456x2234 here) because OCR
        # wants every pixel. A vision model does not, and sending it that
        # made the request time out at 45s — after which the local
        # fallback ran for 51s and took speech-to-text with it. Same
        # 1024px the advice path uses, for the same measured reason.
        payload = _small_capture(cfg, jpeg, meta, lines) or _small_capture(cfg, jpeg)
        answer = llm.look(prompt, payload, cfg)
        _log(f"answered from the vision model in {time.monotonic() - started:.2f}s")
        return spoken(answer)
    except llm.LLMError as e:
        raise VisionError(str(e)) from e
    except VisionError:
        if not cfg.get("screen_cloud_fallback", False):
            raise
        if not interactive or confirm_fn is None:
            raise
        if not confirm_fn("The local vision model couldn't read the screen. Send this "
                          "screenshot to the cloud vision model (a paid call)?"):
            return "Left it local, then. The local vision model couldn't read that screen."
        _log("local vision failed; sending the capture to the cloud model (confirmed)")
        answer = _ask_cloud_vision(jpeg, prompt, cfg)
        _log("answered from the cloud vision model")
        return spoken(answer)
