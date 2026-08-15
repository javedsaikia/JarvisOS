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

from max import env, llm, screen_capture, voice_events
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
_TRAIL = r"([, ]+(max|please))?$"
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


def _read_visible_files(names: list[str]) -> list[tuple[str, str]]:
    """Resolve on-screen names to real files and read them."""
    loaded: list[tuple[str, str]] = []
    for name in names:
        paths = files.find_named(name, limit=1)
        if not paths:
            continue
        try:
            content = files.read_file(paths[0], max_chars=4000)
        except files.FileToolError:
            continue
        loaded.append((paths[0], content))
        if len(loaded) >= 3:
            break
    return loaded


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

    ocr = screen_capture.ocr_text(lines)[:4000]
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
    try:
        return llm.chat(
            [{"role": "user", "content": prompt}],
            entry,
            cfg,
            options={"num_predict": cfg.get("screen_max_tokens", 120)},
        ).strip()
    except llm.LLMError as e:
        raise VisionError(f"Could not reach {entry['label']} to teach from the screen: {e}") from e


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


def _summarize_screen(meta: dict, lines: list[dict]) -> str:
    """Answer "what's on my screen?" from facts, with no model involved.

    Measured, repeatedly: handed a screen's worth of OCR and asked to
    summarize it, the small local models invent — a URL that isn't there,
    an integration that doesn't exist, contents for a window that is only
    a title in a list. The window server already knows exactly which app
    is in front and what else is open, and the OCR already knows which
    line is drawn largest. Stating those is faster (no inference at all),
    always true, and answers the question people are actually asking.

    Specific questions still go to the model, where its job is finding
    something in supplied text rather than summarizing a whole desktop.
    """
    raw_app = meta.get("frontmost", "") or ""
    app = APP_NAMES.get(raw_app, raw_app)
    title = (meta.get("frontmost_title") or "").strip()
    kind = _front_kind(raw_app)
    parts: list[str] = []

    if kind == "browser":
        tab = title or "an untitled tab"
        parts.append(f"This is a browser page in {app}.")
        parts.append(f"This tab is “{tab}”.")
    elif kind == "picture":
        parts.append("This is a picture.")
        if title and title.lower() not in app.lower():
            parts.append(f"It's open in {app}, titled “{title}”.")
        else:
            parts.append(f"It's open in {app}.")
    elif not app:
        parts.append("I can see the screen.")
    elif title and title.lower() not in app.lower():
        parts.append(f"You're in {app}, on “{title}”.")
    else:
        parts.append(f"You're in {app}.")

    if kind == "app" and len(screen_capture.ocr_text(lines)) < 80:
        parts = ["This is a picture."] + ([f"It's in {app}."] if app else [])

    others = []
    for window in meta.get("windows", [])[1:]:
        name = APP_NAMES.get(window["app"], window["app"])
        if name and name != app and name not in others:
            others.append(name)
    others = others[:3]
    if others:
        if len(others) == 1:
            parts.append(f"{others[0]} is also open.")
        else:
            parts.append(f"{', '.join(others[:-1])} and {others[-1]} are also open.")

    headline = screen_capture.prominent_lines(lines, count=1)
    if headline and kind != "picture":
        parts.append(f"The largest text on screen reads “{headline[0]}”.")

    visible = _filenames_on_screen(meta, lines)
    if visible:
        if len(visible) == 1:
            parts.append(f"The file on screen is {visible[0]}.")
        else:
            parts.append(f"Files on screen: {', '.join(visible)}.")
        parts.append("Ask me to teach you this and I'll read the file and walk you through it.")

    return " ".join(parts)


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
    text = screen_capture.ocr_text(lines)[:6000]

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
    try:
        return llm.chat(
            [{"role": "user", "content": prompt}],
            entry,
            cfg,
            options={"num_predict": cfg.get("screen_max_tokens", 220)},
        ).strip()
    except llm.LLMError as e:
        raise VisionError(f"Could not reach {entry['label']} to describe the screen: {e}") from e


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
    loaded = _read_visible_files(visible) if visible and not _is_generic(prompt) else []
    if visible:
        _log("files on screen: " + ", ".join(visible))
    if loaded:
        _log("opened " + ", ".join(p for p, _ in loaded))

    pointed = _point_at_front(meta, lines, prompt)
    if pointed:
        _log(pointed)

    def spoken(answer: str) -> str:
        _remember_screen(lines, _hints_from_answer(answer or "") + _hints_from_answer(pointed or ""))
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
    except VisionError as local_error:
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
