"""Screen understanding: a real screenshot, read on this machine.

Native capture and OCR live in jarvis/screen_capture.py (why they are
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

from jarvis import env, ollama_client, screen_capture, voice_events

DEFAULT_QUESTION = "Describe what's on the screen in 2-4 clear, concise sentences."

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

# "What's on my screen" with nothing else in it. These get a deterministic
# answer built from the window list and the OCR — see _summarize_screen.
_LEAD = r"^(hey )?(jarvis )?(can you |could you |please )?(tell me )?"
_MINE = r"( my| the| this| your)?"
GENERIC_SCREEN_QUESTIONS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Deliberately anchored end-to-end with only filler words allowed
        # in between: "what's on my screen" is generic, but "what's the
        # error message on screen" carries a subject and must reach the
        # model, so nothing arbitrary may match in the middle.
        _LEAD + r"(what'?s|what is)( on| in)?" + _MINE + r" screen$",
        _LEAD + r"(describe|read|check|look at|analyse|analyze)" + _MINE + r" screen$",
        _LEAD + r"what am i (looking at|seeing)$",
        _LEAD + r"(can|do) you see" + _MINE + r" screen$",
        _LEAD + r"what do you see$",
        _LEAD + r"(take|grab|get) a screenshot$",
    )
]


def _is_generic(question: str) -> bool:
    """True for "what's on my screen" and its phrasings, false for
    "what's the error on screen" — anything with a subject of its own
    needs the model, because only the model can find that subject in the
    text."""
    normalized = re.sub(r"[^a-z' ]", " ", question.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return any(p.match(normalized) for p in GENERIC_SCREEN_QUESTIONS)


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
    app = APP_NAMES.get(meta.get("frontmost", ""), meta.get("frontmost", ""))
    title = (meta.get("frontmost_title") or "").strip()

    if not app:
        opening = "I can see the screen"
    elif title and title.lower() not in app.lower():
        opening = f"You're in {app}, on “{title}”"
    else:
        opening = f"You're in {app}"

    others = []
    for window in meta.get("windows", [])[1:]:
        name = APP_NAMES.get(window["app"], window["app"])
        if name and name != app and name not in others:
            others.append(name)
    others = others[:3]

    parts = [opening + "."]
    if others:
        if len(others) == 1:
            parts.append(f"{others[0]} is also open.")
        else:
            parts.append(f"{', '.join(others[:-1])} and {others[-1]} are also open.")

    headline = screen_capture.prominent_lines(lines, count=1)
    if headline:
        parts.append(f"The largest text on screen reads “{headline[0]}”.")

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
    model = cfg.get("screen_text_model") or cfg.get("voice_ollama_model") or cfg["ollama_model"]
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
        "Answer that question in 2 or 3 plain spoken sentences, using only what is "
        "above. If the answer isn't on the screen, say so plainly instead of "
        "guessing. No lists, no bullet points, no code blocks."
    )
    try:
        return ollama_client.chat(
            [{"role": "user", "content": prompt}],
            model,
            cfg["ollama_host"],
            keep_alive=cfg.get("ollama_keep_alive"),
            options={"num_predict": cfg.get("screen_max_tokens", 220)},
        ).strip()
    except ollama_client.OllamaError as e:
        raise VisionError(f"Could not reach the local model to describe the screen: {e}") from e


def _ask_cloud_vision(jpeg_bytes: bytes, prompt: str, cfg: dict) -> str:
    """Optional paid fallback (Gemini). Never reached without both
    `screen_cloud_fallback: true` and an explicit confirmation — same rule
    the Claude Code handoff follows, for the same reason."""
    api_key = env.load_env().get("GEMINI_API_KEY", "")
    if not api_key:
        raise VisionError("No GEMINI_API_KEY in jarvis/.env, so there's no cloud fallback to use.")
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
    hud_marker = cfg.get("screen_hud_window_marker", "J.A.R.V.I.S.")
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

    if char_count >= MIN_OCR_CHARS:
        if _is_generic(prompt):
            # No inference at all for the plain question — the answer is
            # assembled from the window list and the OCR itself.
            _log("generic question — answering from window facts, no model call")
            return _summarize_screen(meta, lines)
        started = time.monotonic()
        answer = _ask_about_text(lines, meta, prompt, cfg)
        _log(f"answered from screen text in {time.monotonic() - started:.2f}s")
        return answer

    # Little or no text: a photo, a video, a design. That is what the
    # vision model is actually good at.
    _log("little text on screen — using the local vision model instead")
    try:
        started = time.monotonic()
        answer = _ask_vision_model(jpeg, prompt, cfg)
        _log(f"answered from the vision model in {time.monotonic() - started:.2f}s")
        return answer
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
        return answer
