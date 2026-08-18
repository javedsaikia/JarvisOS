"""Screen understanding: a real screenshot, read on this machine.

Native capture and OCR live in max/screen_capture.py (why they are
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

from max import env, instructions, llm, screen_capture, voice_events
from max.tools import files

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
_LEAD = r"^(hey[, ]+)?((max|jarvis|orin)[, ]+)?(can you |could you |please )?(tell me )?"
# Spoken questions trail off into time words. "What's on my screen RIGHT
# NOW" is the same generic question as "what's on my screen", but the old
# `$` anchor rejected it — so it went to the model, which answered with a
# four-line inventory of the Word ribbon instead of "This is a Microsoft
# Word window. What do you want help with?". Measured from a real turn.
_TRAIL = r"(?:[, ]+(?:right now|at the moment|just now|now|today|currently|max|please))*$"
_MINE = r"( my| the| this| your)?"
GENERIC_SCREEN_QUESTIONS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Deliberately anchored end-to-end with only filler words allowed
        # in between: "what's on my screen" is generic, but "what's the
        # error message on screen" carries a subject and must reach the
        # model, so nothing arbitrary may match in the middle.
        _LEAD + r"(what'?s|what is)( on| in)?" + _MINE + r" screen" + _TRAIL,
        _LEAD + r"(describe|read|check|look at|analyse|analyze)" + _MINE + r" screen" + _TRAIL,
        _LEAD + r"what (can |do )?you (see|tell).{0,40}screen" + _TRAIL,
        _LEAD + r"what am i (looking at|seeing)" + _TRAIL,
        _LEAD + r"(can|do) you see" + _MINE + r" screen" + _TRAIL,
        _LEAD + r"what do you see" + _TRAIL,
        _LEAD + r"(take|grab|get) a screenshot" + _TRAIL,
    )
]


_TEACHER_RE = re.compile(
    r"\b(teach|tutor|explain|walk me through|help me understand|"
    r"what file|this file|this code|my desktop|the desktop|whole desktop|"
    r"read (this |that |the |my )?(file|code|desktop))\b",
    re.IGNORECASE,
)

# Window titles and OCR both leak filenames. VS Code: "cli.py — MaxOS".
# Finder/Preview: "README.md". Ignore obvious non-files.
_FILENAME_RE = re.compile(
    r"\b([\w.-]+\.(?:py|ts|tsx|js|jsx|md|json|txt|css|html|htm|sh|zsh|rs|go|"
    r"swift|pdf|csv|yml|yaml|toml|ini|env|ipynb|rb|java|c|h|cpp|hpp))\b",
    re.IGNORECASE,
)
_SKIP_TITLES = {"max hud", "finder", "untitled", "desktop"}


def _is_teacher(question: str) -> bool:
    return bool(_TEACHER_RE.search(question or ""))


_GUIDE_RE = re.compile(
    r"\b(show|point|guide|where is|where are|can'?t see|cannot see|"
    r"not able to see|find me|this .{0,24}(button|tab|link|icon)|"
    r"point .{0,16}(cursor|pointer|mouse)|that application|show me)\b",
    re.IGNORECASE,
)


def _is_guide(question: str) -> bool:
    return bool(_GUIDE_RE.search(question or ""))


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
# Set when Max has offered to point at a specific control, cleared once
# the offer is taken or the subject changes. Holding the control's name is
# what lets a bare "yes" mean something concrete.
_PENDING_SHOW: dict | None = None


def _open_session(meta: dict) -> None:
    raw = meta.get("frontmost", "") or ""
    _SESSION.update({"at": time.monotonic(), "raw_app": raw,
                     "app": APP_NAMES.get(raw, raw)})


def session_active() -> bool:
    """True while a screen conversation is still live."""
    return bool(_SESSION["app"]) and (time.monotonic() - _SESSION["at"]) < SESSION_SECONDS


def session_app() -> str:
    return _SESSION["app"] if session_active() else ""


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
    r"hey\s+max[,\s]*|please\s+|(?:can|could|will|would)\s+you\s+|"
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
    return f"HOW TO — {task}" if task and task != "MAX" else "INSTRUCTIONS"


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
    return task[:24].upper() if task else "MAX"


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
        # "cli.py — MaxOS — …" / "cli.py - MaxOS"
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
# payload Max ever sends a model — and the assigned screen model sits on a
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


def _screen_chat(prompt: str, cfg: dict, max_tokens: int) -> str:
    """Ask the screen model, falling back to the local one on any failure."""
    entry = llm.resolve(cfg, "screen")
    pinned = cfg.get("screen_text_model")
    if pinned:
        entry = {**entry, "provider": "ollama", "model": pinned, "label": f"Max · {pinned}"}
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
        entry = {**entry, "provider": "ollama", "model": pinned, "label": f"Max · {pinned}"}

    ocr = screen_capture.ocr_text(lines)[:OCR_FOR_MODEL_CHARS]
    windows = "; ".join(
        f"{w['app']}: {w['title']}" for w in meta.get("windows", [])[:10]
    )
    file_block = ""
    for path, content in loaded:
        file_block += f"\n--- {path} ---\n{content}\n"
    if not file_block:
        file_block = "(no matching files could be opened from the folders Max can read)"

    names = ", ".join(visible) if visible else "none identified"
    prompt = (
        "You are Max, talking to them at their Mac — not a professor.\n"
        "Teach from what is actually there. Do not invent files, errors, or code.\n"
        "One or two short spoken sentences. Summarize. Ask what they want next.\n"
        "Do not walk through every step unless they asked for that next action.\n\n"
        f"Frontmost app: {meta.get('frontmost', 'unknown')}"
        f"{' — ' + meta['frontmost_title'] if meta.get('frontmost_title') else ''}\n"
        f"Windows: {windows}\n"
        f"Files identified on screen: {names}\n\n"
        f"File contents Max opened:\n{file_block}\n\n"
        f"Text read off the screen:\n---\n{ocr}\n---\n\n"
        f"The student said: {question}\n\n"
        "Name the file and app, say what it is doing in one line, ask the next action. "
        "No markdown, no lists, no lecture. If you could not open a file, say the name "
        "you saw."
    )
    return _screen_chat(prompt, cfg, cfg.get("screen_max_tokens", 120))


def _is_generic(question: str) -> bool:
    """True for "what's on my screen" and its phrasings, false for
    "what's the error on screen" — anything with a subject of its own
    needs the model, because only the model can find that subject in the
    text."""
    normalized = re.sub(r"[^a-z' ]", " ", question.lower())
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
    "button", "link", "tab", "icon", "menu", "item", "max", "jarvis",
    "can", "you", "the", "how", "all", "okay", "please",
    "cursor", "pointer", "mouse", "application", "app", "still",
    "struggling", "exactly", "dont", "don't", "see",
}


def _guide_terms(question: str) -> list[str]:
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
        if w not in terms:
            terms.append(w)
    # Bare "show me" / "that application" — use labels from the last look.
    vague = bool(re.search(r"\bthat (app|application|one|button|thing)\b", raw))
    if vague or not terms or set(terms) <= {"yes", "yeah", "please"}:
        for hint in _LAST_HINTS:
            if hint.lower() not in terms:
                terms.append(hint.lower())
    return terms


def _best_ocr_match(lines: list[dict], question: str) -> dict | None:
    terms = _guide_terms(question)
    if not terms:
        return None
    best, score = None, 0.0
    for line in lines or []:
        text = re.sub(r"\s+", " ", (line.get("text") or "").lower()).strip()
        if not text:
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


def _point_at_front(meta: dict, lines: list[dict], question: str) -> str:
    """Glide the pointer onto each matching control. Returns a spoken cue."""
    pool = lines or _LAST_LINES
    width, height = int(meta.get("width") or 0), int(meta.get("height") or 0)
    terms = _guide_terms(question)
    matches: list[dict] = []
    seen: set[str] = set()
    for term in terms[:8]:
        hit = _best_ocr_match(pool, term)
        if not hit:
            continue
        key = (hit.get("text") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        matches.append(hit)
    if not matches:
        hit = _best_ocr_match(pool, question)
        if hit:
            matches.append(hit)
    if matches and width:
        bits: list[str] = []
        for hit in matches[:4]:
            screen_capture.point_at_ocr_line(hit, width, height)
            label = re.sub(r"\s+", " ", (hit.get("text") or "").strip())
            bits.append(f"This is “{label}”, {_region_phrase(hit)}.")
            time.sleep(0.45)
        return " ".join(bits)
    bounds = meta.get("frontmost_bounds") or {}
    if bounds.get("w"):
        screen_capture.point_at_window(bounds)
        app = meta.get("frontmost") or "the front window"
        return f"I couldn't find that label. I'm pointing at the middle of {app}."
    windows = meta.get("windows") or []
    if windows:
        screen_capture.point_at_window(windows[0])
        return "I couldn't find that label. I'm pointing at the front window."
    return "I couldn't find that on the screen to point at."


def _product_phrase(meta: dict, lines: list[dict]) -> str:
    """Name the thing they are looking at, the way a person would.

    "This is a Microsoft Word window." — the product, first, in one
    sentence. That is what someone asking "what's on my screen" wants
    identified before anything else, and it is the opening move of
    teaching: establish what we are both looking at, then ask what they
    need. Everything else the old answer volunteered (which other apps are
    open, the largest text, filenames) is detail nobody asked for, and read
    aloud it buried the one fact that mattered.
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


def _summarize_screen(meta: dict, lines: list[dict]) -> str:
    """Answer "what's on my screen?" — the product, then the handover.

    Two sentences, always the same shape: what this is, and what do you
    want. See _product_phrase for why the detail was cut.
    """
    return f"{_product_phrase(meta, lines)} What do you want help with?"


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
        entry = {**entry, "provider": "ollama", "model": pinned, "label": f"Max · {pinned}"}
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
    return _screen_chat(prompt, cfg, cfg.get("screen_max_tokens", 220))


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
                              options={"num_predict": cfg.get("screen_max_tokens", 220)}).strip()
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


def explain_howto(question: str, cfg: dict) -> str:
    """Teach one task in the app they are already in.

    Deliberately does NOT take a screenshot. The answer needed here —
    "Command B, or the bold B in the Home tab" — is ordinary knowledge
    about a well-known application, and the app was already established
    when the session opened. Capturing and OCRing again would add a second
    or two to every follow-up to learn something we already know. The
    screenshot comes back only if they take up the offer to be shown,
    where the exact position genuinely matters.
    """
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
        f"You are Max, teaching someone sitting at a Mac using {app}.\n"
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
            options={"num_predict": cfg.get("screen_max_tokens", 220)},
        ).strip()
    except llm.LLMError as e:
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
        # The model ignored the format. Its prose is still the answer.
        say = re.sub(r"\s+", " ", raw).strip()
    if not say:
        raise VisionError("no answer came back")

    title = _panel_title(question)
    card = instructions.build_card(steps, fallback=say)

    global _PENDING_SHOW
    _PENDING_SHOW = None
    # Only offer to point at something we can actually see. The control
    # name is the one part of this answer that is a claim about THIS
    # screen rather than general knowledge, so it gets checked against the
    # text really read off it. Without this, an invented control produced
    # an invented offer, and "yes" then led to "I couldn't find it" — Max
    # wasting the user's turn on its own mistake.
    if control and _best_ocr_match(_LAST_LINES, control):
        _PENDING_SHOW = {"control": control, "question": question,
                         "card": card, "title": title}
        # Appended to the SPOKEN line only. The offer is a question, and
        # a question has no place on a reference card.
        say = f"{say} Want me to show you where it is?"
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
        hud_marker=cfg.get("screen_hud_window_marker", "Max HUD"), max_width=1024)
    app = facts.get("frontmost") or session_app() or "this app"
    # Prefer just the window they are working in. The desktop around it is
    # noise the model will otherwise narrate back at them.
    bounds = screen_capture.frontmost_window_bounds(app)
    if bounds:
        try:
            cropped = screen_capture.capture_region(*bounds)
            if cropped:
                image = cropped
                _log(f"advising on the {app} window, not the whole desktop")
        except Exception as e:
            _log(f"window crop failed ({e}) — using the full screen")
    prompt = (
        f"You are Max, looking over the shoulder of someone using {app} on a Mac.\n"
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


def show_pending(cfg: dict) -> str:
    """Point Max's marker at the control that was just offered.

    Takes a fresh capture on purpose: this is the one step where being
    wrong by a few centimetres is worse than being a second slower, and
    toolbars move when a window is resized or a tab is switched.
    """
    pending = _PENDING_SHOW
    if not pending:
        return "There's nothing I offered to point at just now."
    control = pending.get("control") or ""

    hud_marker = cfg.get("screen_hud_window_marker", "Max HUD")
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
    # still correct, and "I couldn't find it" is a remark about Max, not
    # an instruction for the user.
    return missed


def _ask_cloud_vision(jpeg_bytes: bytes, prompt: str, cfg: dict) -> str:
    """Optional paid fallback (Gemini). Never reached without both
    `screen_cloud_fallback: true` and an explicit confirmation — same rule
    the Claude Code handoff follows, for the same reason."""
    api_key = env.load_env().get("GEMINI_API_KEY", "")
    if not api_key:
        raise VisionError("No GEMINI_API_KEY in max/.env, so there's no cloud fallback to use.")
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
    hud_marker = cfg.get("screen_hud_window_marker", "Max HUD")
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
    # hand mid-task. Being shown a control is now something Max offers and
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
        if _is_summarize(prompt):
            card = instructions.build_card([], fallback=answer or "")
            if card:
                instructions.show(card, app=_SESSION["raw_app"], title="SUMMARY")
        if pointed and ( _is_guide(prompt) or "pointing" in pointed.lower() or pointed.startswith("This is") ) and pointed not in (answer or ""):
            return f"{pointed} {answer}".strip()
        return answer

    if _is_teacher(prompt) or loaded and not _is_generic(prompt):
        started = time.monotonic()
        answer = _teach_from_screen(prompt, meta, lines, visible, loaded, cfg)
        _log(f"taught from screen + {len(loaded)} file(s) in {time.monotonic() - started:.2f}s")
        return spoken(answer)

    if char_count >= MIN_OCR_CHARS:
        if _is_generic(prompt):
            # No inference at all for the plain question — the answer is
            # assembled from the window list and the OCR itself.
            _log("generic question — answering from window facts, no model call")
            return spoken(_summarize_screen(meta, lines))
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
    _log("little text on screen — using the local vision model instead")
    try:
        started = time.monotonic()
        answer = _ask_vision_model(jpeg, prompt, cfg)
        _log(f"answered from the vision model in {time.monotonic() - started:.2f}s")
        return spoken(answer)
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
