#!/usr/bin/env python3
"""Pre-flight checks for Bella. Run before testing, and before committing.

    bella/.venv/bin/python3 checkpoints.py          # static + wiring (fast)
    bella/.venv/bin/python3 checkpoints.py --live   # also hits the network

This exists because the failures that actually hurt in this project are
not crashes — they are silent disconnections. A router destination with no
handler. A bridge message the HUD never listens for. A queue that can be
waited on forever. Each one looks fine until the moment it doesn't, and
each one is trivially checkable from the source itself.

Nothing here needs the voice loop running. --live adds the checks that
cost a network round trip (model reachability, TTS latency), so the fast
path stays usable as a pre-commit gate.
"""
from __future__ import annotations

import argparse
import ast
import collections
import pathlib
import re
import statistics
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent
PY_FILES = [
    p for p in (ROOT / "bella").rglob("*.py")
    if ".venv" not in str(p) and "browser_profile" not in str(p)
]

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    def wrap(fn):
        def run():
            try:
                detail = fn() or ""
                _results.append((name, True, detail))
            except AssertionError as e:
                _results.append((name, False, str(e)))
            except Exception as e:  # a broken check is a failed check
                _results.append((name, False, f"{type(e).__name__}: {e}"))
        run.__name__ = fn.__name__
        return run
    return wrap


# --- static ---------------------------------------------------------------

@check("compiles")
def c_compile():
    r = subprocess.run([sys.executable, "-m", "compileall", "-q", str(ROOT / "bella")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return f"{len(PY_FILES)} files"


@check("imports")
def c_imports():
    mods = ["bella.cli", "bella.router", "bella.bridge", "bella.voice_loop", "bella.llm",
            "bella.compose", "bella.instructions", "bella.tools.vision", "bella.screen_capture",
            "bella.tools.desktop",
            "bella.sarvam_client", "bella.reflection", "bella.memory", "bella.persona"]
    r = subprocess.run([sys.executable, "-c", "import " + ", ".join(mods)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr.strip()[-400:]
    return f"{len(mods)} modules"


@check("no unused names (pyflakes)")
def c_pyflakes():
    r = subprocess.run([sys.executable, "-m", "pyflakes", *map(str, PY_FILES)],
                       capture_output=True, text=True)
    out = r.stdout.strip()
    assert not out, out
    return "clean"


@check("no unreferenced functions")
def c_dead():
    src = {p: p.read_text() for p in PY_FILES}
    # Definitions come from bella/ only, but a caller may legitimately live
    # outside it — this file drives test-only entry points that would
    # otherwise look dead.
    scan = dict(src)
    scan[pathlib.Path(__file__)] = pathlib.Path(__file__).read_text()
    used = collections.Counter()
    for p, s in scan.items():
        for n in ast.walk(ast.parse(s)):
            if isinstance(n, ast.Name):
                used[n.id] += 1
            elif isinstance(n, ast.Attribute):
                used[n.attr] += 1
        for m in re.finditer(r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']", s):
            used[m.group(1)] += 1
    dead = []
    for p, s in src.items():
        for n in ast.parse(s).body:
            if (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name != "main" and not n.name.startswith("__")
                    and used[n.name] == 0):
                dead.append(f"{p.relative_to(ROOT)}:{n.lineno} {n.name}")
    assert not dead, ", ".join(dead)
    return "none"


@check("no TODO/FIXME/debug leftovers")
def c_leftovers():
    hits = []
    for p in PY_FILES:
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"\b(TODO|FIXME|XXX|HACK)\b", line):
                hits.append(f"{p.relative_to(ROOT)}:{i}")
    assert not hits, ", ".join(hits)
    return "none"


# --- wiring ---------------------------------------------------------------

@check("every router destination has a handler")
def c_router_handlers():
    from bella import router
    src = (ROOT / "bella" / "cli.py").read_text()
    tree = ast.parse(src)
    body = ast.get_source_segment(
        src, next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "handle_tool"))
    handled = set(re.findall(r'domain == "(\w+)"', body))
    for m in re.finditer(r"domain in \(([^)]*)\)", body):
        handled |= set(re.findall(r'"(\w+)"', m.group(1)))
    routed = sorted({(d, a) for d, a, _ in router._TOOL_GROUPS})
    proc = ast.get_source_segment(
        src, next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "process_turn"))
    state = sorted(set(re.findall(r'tool_match = \("(\w+)", "(\w+)"\)', proc)))
    missing = [f"{d}/{a}" for d, a in routed + state if d not in handled]
    assert not missing, ", ".join(missing)
    # and nothing handled that nobody can route to
    orphan = handled - {d for d, _ in routed} - {d for d, _ in state}
    assert not orphan, f"unreachable handler branches: {sorted(orphan)}"
    from bella import discovery_client as disco
    assert disco.is_safe_listing_url("https://in.bookmyshow.com/explore/home/goa")
    assert not disco.is_safe_listing_url("https://pay.example/checkout")
    assert not disco.is_safe_listing_url("https://site.example/login?next=/pay")
    return f"{len(routed)} routed + {len(state)} state-driven"


@check("bridge <-> HUD protocol is symmetric")
def c_protocol():
    bridge = (ROOT / "bella" / "bridge.py").read_text()
    socket = (ROOT / "frontend" / "src" / "socket.ts").read_text()
    main = (ROOT / "frontend" / "src" / "main.ts").read_text()
    sends = set(re.findall(r'"type":\s*"(\w+)"', bridge))
    cases = set(re.findall(r'case "(\w+)"', main))
    declared = set(re.findall(r'type:\s*"(\w+)"', socket))
    browser_sends = set(re.findall(r'this\.send\(\{\s*type:\s*"(\w+)"', socket))
    handled = set(re.findall(r'mtype == "(\w+)"', bridge))
    problems = []
    for t in sorted(sends - cases - {"confirm_response"}):
        problems.append(f"server sends {t!r}, HUD has no case")
    for t in sorted(sends - declared):
        problems.append(f"server sends {t!r}, not in socket.ts types")
    for t in sorted(browser_sends - handled):
        problems.append(f"HUD sends {t!r}, bridge ignores it")
    for t in sorted(handled - browser_sends):
        problems.append(f"bridge handles {t!r}, HUD never sends it")
    assert not problems, "; ".join(problems)
    return f"{len(sends)} down, {len(browser_sends)} up"


@check("config keys referenced all have defaults")
def c_config():
    from bella.config import DEFAULTS
    import json
    saved = json.loads((ROOT / "bella" / "config.json").read_text())
    known = set(DEFAULTS) | set(saved)
    referenced = set()
    for p in PY_FILES:
        s = p.read_text()
        referenced |= set(re.findall(r'cfg\.get\(\s*"(\w+)"', s))
        referenced |= set(re.findall(r'cfg\[\s*"(\w+)"\s*\]', s))
    missing = sorted(referenced - known)
    assert not missing, f"read but never defaulted: {missing}"
    return f"{len(referenced)} keys"


# --- behaviour ------------------------------------------------------------

ROUTES = [
    ("what's on my screen", "screen/read"), ("summarize this page", "screen/read"),
    ("পেলা ৱাটছ অট মাই স্ক্ৰীণ?", "screen/read"),
    ("ৱাটছ অ' মাই স্ক্ৰেইন?", "screen/read"),
    ("মোৰ স্ক্ৰিনত কি দেখিছা তুমি?", "screen/read"),
    ("হাই তুমি মোৰ স্ক্ৰিনত কি দেখিছা?", "screen/read"),
    ("ৱাটছ দা কোৱাশ্চন?", "screen/read"),
    ("ৱাট ইজ দা কোশ্চেনছ ইন দিছ পেজ ৰাইট নাউ?", "screen/read"),
    ("দেৰ ইজ এ কোশ্চেন অন দা পেজ।", "screen/read"),
    ("ষ্টপ দ মিউজিক।", "spotify/pause"),
    ("what's the question", "screen/read"),
    ("what is this picture", "screen/region"), ("let me circle something", "screen/region"),
    ("write an email about pricing", "compose/auto"),
    ("write me a prompt to review this code", "compose/auto"),
    ("draft an email to sabina@gmail.com", "email/write"), ("check my email", "email/read"),
    ("what's the weather", "location/weather"), ("where is the nearest ATM", "location/nearby"),
    ("hotels near me", "location/nearby"),
    ("where am I", "location/here"), ("call Nova", "call/auto"),
    ("play High Hopes on Spotify", "spotify/play_track"), ("pause the music", "spotify/pause"),
    ("run ls -la", "shell/auto"), ("what can you do", "capabilities/read"),
    ("tell me about the movie Inception", "omdb/auto"), ("tell me a joke", "joke/auto"),
    ("what's on my calendar tomorrow", "calendar/read"),
    ("what movies are showing at the cinema", "location/showtimes"),
    ("hotels in Goa", "discovery/hotels"),
    ("flights from Guwahati to Delhi on Friday", "discovery/flights"),
    ("click the Save button", "desktop/auto"),
    ("click the Save button in Notes", "desktop/auto"),
    ("how do I click the Save button", "screen/read"),
    ("show me the Save button", "screen/read"),
    ("go to the playground tab", "screen/read"),
    ("click the Playground tab", "screen/read"),
    ("where do I click the API Keys tab", "screen/read"),
    ("guide me to settings", "screen/read"),
    ("Help me", "screen/read"),
    ("can you help me?", "screen/read"),
    ("can you go ahead and check their numbers? I need the numbers of Novotel Guwahati",
     "call/number"),
    ("Please share the number if you have", "call/number"),
    ("I wanted to have some crispy fried chicken", "location/nearby"),
    # must NOT be captured by any tool
    ("where is Paris", None), ("show me a good recipe for pasta", None),
    ("how do I show gratitude", None), ("point out the obvious", None),
]


@check("router behaviour matrix")
def c_routes():
    from bella import router
    bad = []
    for text, want in ROUTES:
        got = router.detect_tool(text)
        got = f"{got[0]}/{got[1]}" if got else None
        if got != want:
            bad.append(f"{text!r} -> {got} (want {want})")
    assert not bad, "; ".join(bad)
    return f"{len(ROUTES)} cases"


@check("speech chunking is lossless and fast to first word")
def c_speech():
    from bella.voice_loop import _split_for_speech, _clean_for_speech, _FIRST_MAX_CHARS
    texts = [
        "Yes.",
        'A man asked, "Books on Pavlov?" The librarian replied, "It rings a bell." Ha.',
        "This is a Microsoft Word window. The Home tab is open.",
        "It is 31 degrees in Jorhat. No rain expected today.",
        "Near you in Jorhat: Hotel Ligiri, about 700 m; Mising Kitchen, about 1.2 km.",
    ]
    for t in texts:
        chunks = _split_for_speech(_clean_for_speech(t))
        joined = " ".join(" ".join(chunks).split())
        assert joined == " ".join(t.split()), f"text changed: {t[:40]!r}"
        assert len(chunks[0]) <= _FIRST_MAX_CHARS, \
            f"opening chunk {len(chunks[0])} chars > {_FIRST_MAX_CHARS}: {t[:40]!r}"
    return f"{len(texts)} replies"


@check("TTS producer cannot hang the voice loop")
def c_tts_hang():
    from bella import voice_loop as vl, sarvam_client, voice_events, cli
    import collections as _c

    class Stub:
        _last_ttfa = None
        _barge_frames = _c.deque(maxlen=12)
        _barge_frames_lock = threading.Lock()
        # Pinned to the Sarvam path so the patched text_to_speech below is
        # what actually raises. Without a cfg the producer died on an
        # AttributeError inside speech.synthesize instead, which still
        # exercised the sentinel but never the exceptions named here.
        cfg = {"tts_provider_english": "sarvam", "tts_voice": "ritu",
               "tts_voice_indic": "ritu"}

    original_speak, original_play = voice_events.speaking, cli.tts.play
    original_tts = sarvam_client.text_to_speech
    voice_events.speaking = lambda b: None
    cli.tts.play = lambda b, stop_event=None: True
    # The failures are deliberate, so their tracebacks are noise here.
    # KeyboardInterrupt is in the list on purpose: it is a BaseException,
    # so `except Exception` does NOT catch it — only the `finally` that
    # posts the queue sentinel saves the loop, which is the thing being
    # checked.
    devnull = open("/dev/null", "w")
    real_stderr, sys.stderr = sys.stderr, devnull
    # The thread excepthook can fire after this function restores stderr,
    # printing a deliberate traceback into a closed file. Silence it at
    # the source rather than racing it.
    real_hook, threading.excepthook = threading.excepthook, lambda a: None
    try:
        for exc in (TimeoutError, ValueError, sarvam_client.SarvamError, KeyboardInterrupt):
            sarvam_client.text_to_speech = (
                lambda *a, _e=exc, **k: (_ for _ in ()).throw(_e("x")))
            t = threading.Thread(
                target=lambda: vl.VoiceLoop._speak_streaming(
                    Stub(), "One. Two three four. Five.", "en-IN",
                    threading.Event(), threading.Event()),
                daemon=True)
            t.start(); t.join(timeout=5)
            assert not t.is_alive(), f"hung on {exc.__name__}"
    finally:
        threading.excepthook = real_hook
        sys.stderr = real_stderr
        devnull.close()
        voice_events.speaking, cli.tts.play = original_speak, original_play
        sarvam_client.text_to_speech = original_tts
    return "4 failure modes, incl. BaseException"


@check("Assamese answers survive the lecture filter")
def c_as_lecture():
    from bella import cli
    # Verbatim shape of the live miss: a long answer that happens to
    # mention হয় was replaced with "কওক, কি লাগে?"
    real = (
        "প্ৰগ্ৰেমিংটো ঠিক আছে। মাজে মাজে লোড বেছি হ'লে "
        "'হয়' বুলি নকবা — ক'ডখন চাই কওঁ।"
    )
    assert not cli._assamese_is_lecture(real), "a real answer was treated as a lecture"
    assert cli._assamese_is_lecture("তুমি আকৌ একো নিলিখিলা।")
    assert cli._assamese_is_lecture("বাৰে বাৰে একে কথা।")
    assert "কওক, কি লাগে?" not in real
    vloop = (ROOT / "bella" / "voice_loop.py").read_text()
    assert "_awaiting_rest_of_line" in vloop, \
        "saying Bella then the command is ignored as room noise again"
    return "long reply kept, short lecture dropped, name-then-command"


@check("Assamese TTS does not hang probing as-IN")
def c_as_in_tts():
    src = (ROOT / "bella" / "sarvam_client.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "text_to_speech")
    body = ast.get_source_segment(src, fn)
    assert "_tts_rest_request" in body, "as-IN first probe is back on the stream path"
    assert "cpu_threads=2" in (ROOT / "bella" / "voice_loop.py").read_text(), \
        "whisper is back on all cores"

    from bella import sarvam_client as sc
    calls = []

    def rest(*a, **k):
        calls.append("rest")
        raise sc.SarvamError("beta", status=400)

    def stream(*a, **k):
        calls.append("stream")
        raise AssertionError("as-IN must not use the stream endpoint")

    def request(text, language_code, model, speaker, timeout):
        calls.append(language_code)
        return b"RIFF"

    orig = (sc._tts_rest_request, sc._tts_stream_request, sc._tts_request,
            sc._as_in_tts_ok, dict(sc._tts_route))
    sc._tts_rest_request = rest
    sc._tts_stream_request = stream
    sc._tts_request = request
    sc._as_in_tts_ok = None
    sc._tts_route.clear()
    sc._tts_wav_cache.clear()
    try:
        assert sc.text_to_speech("কওক।", language_code="as-IN") == b"RIFF"
        assert calls[0] == "rest" and "stream" not in calls
        assert sc._as_in_tts_ok is False
        sc.text_to_speech("এবাৰ আকৌ।", language_code="as-IN")
        assert calls[-1] == "bn-IN"
    finally:
        (sc._tts_rest_request, sc._tts_stream_request, sc._tts_request,
         sc._as_in_tts_ok, route) = orig
        sc._tts_route.clear()
        sc._tts_route.update(route)
        sc._tts_wav_cache.clear()
    return "REST probe, latch bn-IN, no stream"


@check("network clients wrap bare timeouts")
def c_timeouts():
    import urllib.request
    from bella import sarvam_client as sc, llm
    original = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(TimeoutError("read timed out"))
    try:
        for label, fn, err in (
            ("sarvam tts", lambda: sc._tts_request("x", "en-IN", "bulbul:v3", "shubh", 5), sc.SarvamError),
            ("sarvam chat", lambda: sc._post_json(sc.API_URL, {}, 5), sc.SarvamError),
            ("sarvam stt", lambda: sc.speech_to_text(b"", "as-IN"), sc.SarvamError),
            ("llm post", lambda: llm._post("https://x", {}, {}, 5), llm.LLMError),
        ):
            try:
                fn()
                raise AssertionError(f"{label} did not raise")
            except err:
                pass
            except AssertionError:
                raise
            except Exception as e:
                raise AssertionError(f"{label} leaked {type(e).__name__}") from None
    finally:
        urllib.request.urlopen = original
        # The forced TTS failure above trips the stream->REST cooldown.
        # Clear it, or the latency checkpoint would measure the fallback
        # path and report a number that is not what the user experiences.
        sc._STREAM_DISABLED_UNTIL = 0.0
    return "4 clients"


@check("instruction panel never shows conversation")
def c_panel():
    from bella import instructions as ins
    must_vanish = [
        "Want me to show you where it is?", "I'll point at it.", "Next step?",
        "What do you want help with?", "Hope that helps!", "You're welcome.",
        "The student asked you to summarize this page.",
    ]
    for t in must_vanish:
        assert ins.for_screen(t) == "", f"leaked: {t!r}"
    for t in ["Shortcut: Command + B", "Ribbon: Home tab, Font group", "Select the text first"]:
        assert ins.for_screen(t) == t, f"instruction mangled: {t!r}"
    # a DRAFT is the user's own words and must survive verbatim
    card = ins.build_card(["Shortcut: Command + B", "Want me to show you?"])
    assert "Want me" not in card, "card leaked conversation"
    return f"{len(must_vanish)} filtered"


@check("calendar events survive dictation")
def c_calendar():
    from datetime import datetime, timedelta
    from bella.voice_loop import _strip_confirm_echo, CONFIRM_YES_RE, CONFIRM_NO_RE
    from bella.tools import parsing, event_draft

    # A confirmation must never be decided by Bella's own prompt coming back
    # through the mic. Live, "yes or no? yes." (the prompt's tail plus a
    # real YES) was cancelled because CONFIRM_NO_RE matched the echo.
    prompt = 'Add "Meeting with 11 labs" to your calendar for Tue, 1:00 PM? Say yes or no.'
    for heard, want in (("yes or no? yes.", True), ("yes", True),
                        ("no", False), ("no, not that one", False),
                        ("yes or no.", None), ("say yes or no no", False)):
        left = _strip_confirm_echo(heard, prompt)
        if not left:
            got = None
        elif CONFIRM_NO_RE.search(left):
            got = False
        elif CONFIRM_YES_RE.search(left):
            got = True
        else:
            got = None
        assert got is want, f"{heard!r} -> {left!r} read as {got}, expected {want}"

    now = datetime(2026, 8, 17, 21, 34)  # a Monday evening
    for text, title, when in (
        ("I want you to add a meeting tomorrow at the 1pm", "Meeting", "Aug 18 13:00"),
        ("add a meeting with 11 labs tomorrow at 1pm", "Meeting with 11 labs", "Aug 18 13:00"),
        ("schedule a call with Priya on Friday at 3:30 pm", "Call with Priya", "Aug 21 15:30"),
        ("add an event called sprint review at 11", "sprint review", "Aug 18 11:00"),
    ):
        got_title, start, _ = parsing.extract_event(text, now)
        assert got_title == title, f"{text!r} titled {got_title!r}, expected {title!r}"
        assert f"{start:%b %d %H:%M}" == when, f"{text!r} -> {start:%b %d %H:%M}, expected {when}"
        assert start >= now, f"{text!r} resolved into the past"

    # No time said means asking, never guessing an hour.
    assert not parsing.event_has_time("add a meeting with 11 labs")
    event_draft.cancel()
    event_draft.start("Meeting with 11 labs", timedelta(hours=1))
    assert event_draft.apply_time("uh, sometime") is None, "accepted a reply with no time in it"
    filled = event_draft.apply_time("tomorrow at 1pm", now)
    assert filled and f"{filled:%b %d %H:%M}" == "Aug 18 13:00", f"slot filled to {filled}"
    assert event_draft.title() == "Meeting with 11 labs", "the time reply overwrote the title"
    event_draft.cancel()

    # Dictation fillers must not become the title. "can you go ahead and
    # add in meeting?" titled itself "go ahead and add in meeting" and
    # then asked 'What time is "go ahead and add in meeting"?', which is
    # what "it is literally clueless" sounded like from the outside.
    for text in ("can you go ahead and add in meeting?", "go ahead and add a meeting",
                 "can you add in a meeting"):
        got, _, _ = parsing.extract_event(text, now)
        assert got == "Meeting", f"{text!r} titled {got!r}"

    # Adding must not cost a yes/no round trip: the write is ~1s and the
    # confirm was several times that. The undo is what makes that safe,
    # so it has to exist and has to be reachable.
    from bella import cli, router
    src = (ROOT / "bella" / "cli.py").read_text()
    body = ast.get_source_segment(
        src, next(n for n in ast.parse(src).body
                  if isinstance(n, ast.FunctionDef) and n.name == "handle_tool"))
    assert "confirm_calendar_writes" in body, "calendar adds went back to confirming by default"
    assert not cli.load_config().get("confirm_calendar_writes", False), \
        "confirm_calendar_writes is on, so adding costs an extra turn"
    for text in ("remove that meeting", "delete that", "undo that"):
        assert router.detect_tool(text) == ("calendar_undo", "auto"), \
            f"no undo route for {text!r}: {router.detect_tool(text)}"
    # The question is spoken, so it stays short.
    event_draft.start("Meeting", timedelta(hours=1))
    assert event_draft.question() == "What time?", event_draft.question()
    event_draft.cancel()
    return "6 confirms, 7 events, undo, no-confirm add"


@check("no answers invented in place of a tool")
def c_no_fabrication():
    from bella import cli
    # Verbatim from the live log: three replies the chat model produced
    # with no tool behind any of them.
    for user, reply in (
        ("in the next half and all",
         "Around 6 PM the sky should clear a bit, but it's still 70% chance of rain."),
        ("can you find them?",
         "Here's a list of cafés within a mile: Café Lune, Brewed Awakenings, the Corner Bistro."),
        ("No I want to know how the person look like",
         "She's a woman with long dark hair, wearing a blue jacket and a wide-smiling grin."),
    ):
        out = cli._rewrite_unverifiable_answer(reply, user)
        assert out != reply, f"invented answer survived: {user!r}"
    # Ordinary conversation must pass through untouched.
    for user, reply in (
        ("how are you?", "Doing good, thanks Javed. What's up?"),
        ("tell me a joke", "Why did the scarecrow win an award? He was outstanding in his field."),
        ("what time is my meeting", "Your meeting is at 1 PM tomorrow."),
        ("thanks", "Anytime, Javed."),
    ):
        out = cli._rewrite_unverifiable_answer(reply, user)
        assert out == reply, f"rewrote an honest reply: {user!r} -> {out!r}"
    # Screen-content questions must reach the screen, not a language model.
    for text in ("No I want to know how the person look like", "what does it say",
                 "who is in the picture", "describe it", "read the screen",
                 "can you just let me know more about Zubin Gark"):
        assert cli._SCREEN_CONTENT_QUESTION_RE.search(text), f"not claimed: {text!r}"
    for text in ("how are you", "how do I bold this", "what about dinner"):
        assert not cli._SCREEN_CONTENT_QUESTION_RE.search(text), f"over-claimed: {text!r}"
    # Live: "share the number" missed the tool; Groq promised a lookup.
    pulled = cli._rewrite_false_action_claim(
        "Sure, let me pull that up for you. One moment.",
        "Please share the number if you have",
    )
    assert pulled != "Sure, let me pull that up for you. One moment.", pulled
    assert "haven't" in pulled.lower()
    # Live: invented fried-chicken shops with distances, no nearby lookup.
    food = cli._rewrite_unverifiable_answer(
        'Check out Jorhat’s “Chicken Curry & Wings” (≈ 850 m, S. S. Nagar).',
        "I wanted to have some crispy fried chicken",
    )
    assert food != 'Check out Jorhat’s “Chicken Curry & Wings” (≈ 850 m, S. S. Nagar).', food
    return "3 fabrications, 4 pass-through, 8 screen, 2 live lies"


@check("a named city beats GPS")
def c_named_city():
    from bella import cli
    from bella.tools import location
    # Verbatim from the live log. Answered "Near you in Jorhat" with
    # Jorhat hotels, ~300 km from the city named twice in the sentence.
    asked = ("Guwahati next week and I want you to identify some hotels there. "
             "Can you check with TrivatripAdvisor and let me know which are the "
             "best hotels in Guwahati")
    origin, remote = location.search_origin(asked)
    assert remote and origin, "the named city was ignored"
    assert "guwahati" in (origin.get("city") or "").lower(), f"resolved to {origin!r}"
    assert 25.5 < origin["lat"] < 26.5 and 91.0 < origin["lon"] < 92.5, \
        f"Guwahati resolved to {origin['lat']},{origin['lon']}"
    # "near me" must still mean GPS, or this fix would break the common case.
    for text in ("find me some restaurants close to my location", "hotels near me",
                 "what's the weather"):
        _, is_remote = location.search_origin(text)
        assert not is_remote, f"{text!r} was treated as a remote city"
    # A source Bella cannot reach is named rather than silently substituted.
    assert cli._OTHER_PLACES_SOURCE_RE.search(asked), "TripAdvisor mention not detected"
    assert not cli._OTHER_PLACES_SOURCE_RE.search("hotels near me")

    # Every path that takes a location, not just the one that was
    # reported. handle_nearby was fixed first and the same bug was still
    # sitting in showtimes and the phone lookup — this is the sweep.
    for text in ("which cinemas are showing in Guwahati",
                 "give me the contact number of Novotel in Guwahati",
                 "best hotels in Guwahati"):
        origin, is_remote = location.search_origin(text)
        assert is_remote and origin, f"named city dropped by: {text!r}"
        assert "guwahati" in (origin.get("city") or "").lower(), \
            f"{text!r} resolved to {origin.get('city')!r}"
    # A venue name wrapped around the city must not swallow it.
    origin, _ = location.search_origin("the number of Hotel Novotel in Guwahati")
    assert origin and "guwahati" in (origin.get("city") or "").lower(), \
        "the city was lost inside a venue phrase"

    src = (ROOT / "bella" / "cli.py").read_text()
    for fn in ("handle_nearby", "handle_showtimes", "handle_phone_lookup"):
        body = ast.get_source_segment(
            src, next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == fn))
        assert "search_origin" in body, f"{fn} still assumes the current GPS position"
    return "Guwahati via 4 phrasings, 3 handlers, 3 local kept"


@check("screen answers raise the panel and the pointer")
def c_screen_surfaces():
    src = (ROOT / "bella" / "tools" / "vision.py").read_text()
    body = ast.get_source_segment(
        src, next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "spoken"))
    # The card used to be raised only for summaries, so "what's on my
    # screen" — the one question people actually ask, answered from window
    # facts with no model call — left the screen blank.
    assert "instructions.show(" in body, "screen answers no longer raise the panel"
    assert "_is_summarize(prompt)" not in body.split("instructions.show(")[0].split("card =")[-1], \
        "the panel is gated on summaries again"
    # The overlay helper still exists, but a generic look must not call
    # it. Movement is only after they asked to be shown a control.
    assert "_mark_described_window" in src
    assert "_mark_described_window" not in body, \
        "a generic look still moves the overlay before they instruct"
    marker = ast.get_source_segment(
        src, next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "_mark_described_window"))
    assert "show_marker" in marker, "the pointer is not the overlay marker"
    assert "glide_cursor" not in marker and "point_cursor" not in marker, \
        "this would move the user's real mouse on every screen question"
    from bella import screen_capture as sc
    assert callable(sc.point_at_window)
    assert callable(sc.ocr_line_rect)

    # Images going to a vision model must be downscaled. Sending the full
    # 3456x2234 capture timed the request out at 45s, after which the
    # local fallback ran for 51s and took speech-to-text with it.
    for call in ("capture_region(*bounds, max_width=1024)", "_small_capture(cfg, jpeg)"):
        assert call in src, f"a vision call is sending a full-size image: {call}"
    lsrc = (ROOT / "bella" / "llm.py").read_text()
    look = ast.get_source_segment(
        lsrc, next(n for n in ast.parse(lsrc).body
                   if isinstance(n, ast.FunctionDef) and n.name == "look"))
    assert "_look_local" not in look.split("except LLMError as e:")[-1], \
        "a failed cloud call falls back to the CPU-saturating local model again"
    assert "retrying once" in look, "a single stalled request loses the turn again"

    # Long speech is not a request to point at something. A 76-word demo
    # introduction produced three invented pointer cues.
    from bella.tools import vision as v
    assert not v._is_guide(" ".join(["it can guide the user and show the view"] * 12))
    assert v._is_guide("show me the bold button")
    assert v.is_guide("go to the playground tab")
    assert v.is_guide("click the Playground tab")
    assert v.is_guide("where do I click API Keys")
    assert v.is_guide("help me find the settings button")
    assert v.is_guide("guide me to the settings button")
    assert not v.is_guide("Help me")
    assert not v.is_guide("can you help me?")
    assert v.is_help_ask("Help me")
    assert v.is_help_ask("can you help me?")
    assert v.is_guide("can you show me that?")
    assert v.is_guide("show me where")
    assert v._with_next_ask("This is a medical SBA page.").endswith("?")
    assert "do with this" in v._with_next_ask("This is Safari.").lower()
    assert not v.is_guide("where is Paris")
    assert not v.is_guide("show me a good recipe for pasta")
    assert not v.is_guide("click the Save button")
    assert v._is_chrome_label("File") and v._is_chrome_label("Edit")
    assert v._is_chrome_label("Safari")
    chrome_tabs = v._tab_like_lines([
        {"text": "File", "x": 0.05, "y": 0.98, "width": 0.04, "height": 0.02},
        {"text": "Edit", "x": 0.10, "y": 0.98, "width": 0.04, "height": 0.02},
        {"text": "Playground", "x": 0.20, "y": 0.90, "width": 0.10, "height": 0.03},
    ], {"width": 1000, "height": 1000})
    labels = {(ln.get("text") or "") for ln in chrome_tabs}
    assert "File" not in labels and "Edit" not in labels, labels
    assert v._guide_terms("go to the playground tab", use_hints=False) == ["playground"]
    assert v._looks_like_exam([
        {"text": "Which of the following is the most appropriate treatment?"},
        {"text": "A Fluid restriction"},
        {"text": "B Hypertonic saline"},
        {"text": "C Demerol"},
        {"text": "D Normal saline"},
    ])
    assert not v._looks_like_exam([{"text": "Playground"}, {"text": "API Keys"}])
    from bella.config import DEFAULTS
    assert int(DEFAULTS.get("screen_max_tokens", 0)) >= 320, "screen answers die mid-sentence again"

    # A rate limit says how long to wait ("Please retry in 14.198461649s")
    # and the free tier allows 20 requests a MINUTE. A whole turn was
    # thrown away over a fourteen-second wait, with the generic "I
    # couldn't read the screen just now" — which is not what happened and
    # not what to do about it.
    from bella import cli, llm
    assert "429" in look and "retry_after" in look, \
        "a rate limit no longer waits out a short window"
    err = llm.LLMError("HTTP 429: quota. Please retry in 14.19s", status=429, retry_after=14.19)
    assert err.retry_after == 14.19 and err.status == 429, "retry-after is not parsed"
    for message, expected in (
        ("HTTP 429: You exceeded your current quota", "rate limit"),
        ("no vision model available", "vision model"),
        ("the API timed out", "in time"),
        ("HTTP 400: API key not valid", "key"),
    ):
        reply = cli._screen_failure_reply(v.VisionError(message))
        assert expected in reply, f"{message!r} answered {reply!r}"
    # The panel is tagged with the app it is about and closes when the
    # user leaves that app. Bella answers about the FRONTMOST app and then
    # speaks for 5-10s, during which the user looks at the HUD — so
    # without a grace period the card appeared and vanished at once.
    psrc = (ROOT / "bella" / "instruction_panel.py").read_text()
    assert "MIN_VISIBLE_SECONDS" in psrc, "the panel can vanish the instant focus moves"
    grace = float(re.search(r"MIN_VISIBLE_SECONDS = ([\d.]+)", psrc).group(1))
    assert grace >= 15, f"grace period {grace}s is shorter than Bella takes to speak"
    assert "_app_has_window" in psrc, "the card no longer leaves when that app's window is gone"
    vloop = (ROOT / "bella" / "voice_loop.py").read_text()
    assert "_dismiss_screen_help" in vloop, "stop Bella no longer takes the card down"
    # Same reasoning for the pointer: it is placed before the answer is
    # spoken, so it has to outlast the speaking.
    mark = ast.get_source_segment(
        src, next(n for n in ast.parse(src).body
                  if isinstance(n, ast.FunctionDef) and n.name == "_mark_described_window"))
    seconds = float(re.search(r"seconds=([\d.]+)", mark).group(1))
    assert seconds >= 15, f"the pointer disappears after {seconds}s, before the reply is heard"
    # Half a JSON object is not an answer. Bella read '{"control' aloud.
    assert 'startswith(("{", "[",' in src, "a truncated JSON reply can be spoken again"
    return "panel + pointer, downscaled, retried, 4 named failures, grace"


@check("nothing competes with the microphone")
def c_cpu_contention():
    from bella import control, reflection
    from bella.tools import spotify, vision

    # A local 7B reflection starting mid-conversation took the next
    # transcription from 0.73s to 28.55s. The timer arms on turn END, so
    # activity is marked at the WAKE, and reflection re-checks it.
    control.note_activity()
    assert control.seconds_since_activity() < 2, "activity marker did not record"
    src = (ROOT / "bella" / "reflection.py").read_text()
    assert "seconds_since_activity" in src, \
        "reflection no longer checks whether a conversation is in progress"
    assert "control.note_activity()" in (ROOT / "bella" / "voice_loop.py").read_text(), \
        "the voice loop no longer marks activity at the wake"
    # The local vision model is the other CPU hog: prefer the cloud one.
    vsrc = (ROOT / "bella" / "tools" / "vision.py").read_text()
    assert "_ask_vision_model(jpeg, prompt, cfg)" not in vsrc, \
        "the low-text screen path is back on the local vision model"

    # Asking for help with a move must reach the screen, not the how-to
    # path — which asked a text model about a board it could not see.
    for text in ("can you help me with the next move?", "what should my next move be",
                 "suggest a move", "my next move"):
        assert vision.is_advice_request(text), f"not routed to advice: {text!r}"
    for text in ("how do I bold this", "what is on my screen", "how are you"):
        assert not vision.is_advice_request(text), f"over-claimed as advice: {text!r}"
    asked = "can you help me to identify the right answer?"
    assert vision.is_question_help(asked), "exam help not claimed"
    assert vision.is_howto(asked), "howto still matches help-me; answer must win in process_turn"
    assert vision.is_question_help("can explain the question to me what exactly")

    # A sentence about yourself is not a song title, however recently
    # music played. "and I am using the white one" got "I didn't catch
    # the song."
    spotify.note_music_turn()
    for text in ("and I am using the white one", "I am playing white", "it's my turn now"):
        assert not spotify.looks_like_followup(text), f"claimed as a song: {text!r}"
    for text in ("by Metallica", "Anuradha", "Bohemian Rhapsody"):
        assert spotify.looks_like_followup(text), f"real follow-up lost: {text!r}"
    from bella import voice_loop
    # Wake already heard the English command — do not wait on a second STT.
    got = voice_loop._command_from_wake_transcript("Bella, what's on my screen?", "bella")
    assert got and "screen" in got.lower(), got
    assert voice_loop._command_from_wake_transcript("Bella, Matoumak Bhaal Pao.", "bella") is None
    return "reflection gated, 4 advice, 6 song, wake stt reuse"


@check("compose guards")
def c_compose():
    from bella import compose
    assert compose.detect("draft an email to bob@example.com") is None, \
        "compose stole an addressed email from the Gmail flow"
    assert compose.detect("write an email about pricing") == "email"
    assert compose.detect("what's the weather") is None
    assert compose.wants_insert("go ahead and type it in")
    assert not compose.wants_insert("what's the weather")
    assert compose.subject_of("Subject: Hello\n\nBody") == "Hello"
    assert "```" not in compose._tidy("```\ntext\n```")
    # Live: "can you write a prompt?" in ChatGPT invented a 200-word
    # Bella meta-prompt. No subject → ask. A real task stays a draft.
    assert not compose.has_subject("can you write a prompt?")
    assert not compose.has_subject("Yes go ahead and draft prompt")
    assert compose.has_subject("write a prompt in python to print hello")
    assert compose.has_subject("write me a prompt to review this code")
    assert not compose.refers_to_screen("can you write a prompt?")
    assert compose.refers_to_screen("write a prompt to review this code")
    return "13 assertions"


@check("overlay helper processes are importable")
def c_overlays():
    for mod in ("bella.cursor_overlay", "bella.instruction_panel", "bella.region_select"):
        r = subprocess.run([sys.executable, "-c", f"import {mod}"],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f"{mod}: {r.stderr.strip()[-200:]}"
    return "3 processes"


# --- live -----------------------------------------------------------------

@check("services are up")
def c_services():
    down = []
    for label, port in (("HUD", 2015), ("bridge", 8765)):
        r = subprocess.run(["lsof", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True)
        if r.returncode != 0:
            down.append(f"{label}:{port}")
    assert not down, f"not listening: {', '.join(down)}"
    voice = subprocess.run(["pgrep", "-f", "bella.voice_loop"], capture_output=True)
    return "HUD + bridge up" + (", voice up" if voice.returncode == 0 else ", voice DOWN")


@check("assigned models actually answer")
def c_models():
    from bella.config import load_config
    from bella import llm
    cfg = load_config()
    out, broken = [], []
    for role in llm.ROLES:
        entry = llm.resolve(cfg, role)
        if not entry["available"]:
            broken.append(f"{role} -> {entry['label']} has no key")
            continue
        # Actually CALL it. Checking availability only proved a key was
        # present, which is why nobody noticed Groq decommissioning both
        # configured Llama models: every real call returned "HTTP 404: the
        # model does not exist or you do not have access to it", chat,
        # voice and screen all fell back to the local model, and one turn
        # took 32 seconds. A provider can retire a model at any time and
        # the config will look perfectly fine.
        #
        # num_predict is deliberately generous: reasoning models spend the
        # budget thinking first and return an empty reply if starved,
        # which llm.chat now raises on.
        try:
            reply = llm.chat(
                [{"role": "user", "content": "Reply with the single word: ok"}],
                entry, cfg, options={"num_predict": 200}, timeout=30,
            )
        except llm.LLMError as e:
            broken.append(f"{role} -> {entry['label']}: {str(e)[:90]}")
            continue
        if not reply.strip():
            broken.append(f"{role} -> {entry['label']} returned nothing")
            continue
        out.append(f"{role}={entry['label']}")
    assert not broken, "; ".join(broken)
    return "; ".join(out)


@check("time to first spoken word")
def c_ttfw():
    from bella import sarvam_client as sc
    from bella.voice_loop import _split_for_speech, _clean_for_speech
    reply = "This is a Microsoft Word window. The Home tab is open."
    first = _split_for_speech(_clean_for_speech(reply))[0]
    sc._tts_wav_cache.clear()
    started = time.perf_counter()
    sc.text_to_speech(first, language_code="en-IN")
    elapsed = time.perf_counter() - started
    assert elapsed < 4.0, f"{elapsed:.2f}s to first word (was ~1s)"
    return f"{elapsed:.2f}s for a {len(first)}-char opener"


@check("hot-path tool latency")
def c_latency():
    from bella.config import load_config
    from bella.memory import MemoryStore
    from bella import cli
    cfg, mem = load_config(), MemoryStore()
    # Budgets are generous versus what these measure at, so this catches a
    # regression (a slow provider creeping back to the front of a chain)
    # rather than ordinary jitter. Weather is here because it silently
    # cost 8.9s until the provider order was measured — see
    # location_client.get_weather.
    budgets = {"what's the weather": 5.0, "tell me a joke": 4.0,
               "what can you do": 1.0}
    out, slow = [], []
    for text, budget in budgets.items():
        # Median of three, after a warm-up. A single sample is not a
        # measurement here: Open-Meteo alone ranges 1.7-6.3s (measured over
        # 8 calls, median 2.1s), so one unlucky draw failed a 5s budget
        # while nothing was wrong. The regression this guards against — a
        # slow provider back at the front of a chain — moves the median,
        # which is what gets asserted.
        cli.process_turn(text, cfg, mem, interactive=False, source="text")
        samples = []
        for _ in range(3):
            started = time.perf_counter()
            cli.process_turn(text, cfg, mem, interactive=False, source="text")
            samples.append(time.perf_counter() - started)
        elapsed = statistics.median(samples)
        out.append(f"{text.split()[-1]}={elapsed:.1f}s")
        if elapsed > budget:
            slow.append(f"{text!r} took {elapsed:.1f}s (budget {budget}s)")
    assert not slow, "; ".join(slow)
    return ", ".join(out)


@check("configured voice works")
def c_voice():
    from bella.config import load_config
    from bella import speech, elevenlabs_client
    cfg = load_config()
    provider = (cfg.get("tts_provider_english") or "elevenlabs").lower()
    assert provider in ("elevenlabs", "macos", "sarvam"), \
        f"unknown tts_provider_english {provider!r}"
    if provider == "elevenlabs":
        assert (cfg.get("tts_voice_id") or "").strip(), "tts_provider_english is elevenlabs but tts_voice_id is empty"
    fallback = (cfg.get("tts_english_fallback") or "sarvam").lower()
    assert fallback in ("sarvam", "macos"), \
        f"unknown tts_english_fallback {fallback!r}"
    # tts_voice is read by whichever engine actually ends up speaking
    # English, which is the fallback when the provider is ElevenLabs.
    # Validate it against that engine — a Bulbul speaker name is not a
    # macOS voice and vice versa, and `say` silently substitutes the
    # system default for an unknown name rather than failing.
    engine = fallback if provider == "elevenlabs" else provider
    voice = cfg.get("tts_voice") or ""
    if engine == "sarvam":
        assert voice in speech.SARVAM_SPEAKERS, \
            f"English voice {voice!r} is not a bulbul:v3 speaker"
    else:
        installed = speech.installed_macos_voices()
        assert voice in installed, f"English voice {voice!r} is not installed (`say -v '?'`)"
    assert cfg.get("tts_voice_indic") in speech.SARVAM_SPEAKERS, \
        f"tts_voice_indic {cfg.get('tts_voice_indic')!r} is not a bulbul:v3 speaker"
    # Fresh denial state, so this measures the configured path rather
    # than inheriting a latched failure from an earlier checkpoint.
    speech._reset_elevenlabs_denial()
    wav = speech.synthesize("Checkpoint.", "en-IN", cfg)
    assert speech.playable(wav), "the English voice produced audio the pipeline cannot play"
    note = "" if elevenlabs_client.configured() else "  [ELEVENLABS_API_KEY unset]"
    return speech.describe(cfg) + note


@check("desktop harness wrapper stays closed")
def c_desktop():
    src = (ROOT / "bella" / "tools" / "desktop.py").read_text()
    assert "import macos_harness" not in src, "desktop.py imported the 3.12 package into Bella"
    assert "MAX_DESKTOP_ARGS" in src, "arguments must travel as JSON, not Python source"
    assert "f\"\"\"" not in src and "f'''" not in src, "a program template became an f-string"
    from bella import cli, router
    from bella.config import DEFAULTS
    assert DEFAULTS.get("desktop_enabled") is False, "desktop_enabled must default off"
    assert not cli.load_config().get("desktop_enabled", False), \
        "desktop_enabled is on in config.json — leave it off until a live check"
    assert router.detect_tool("click the Save button") == ("desktop", "auto")
    assert router.detect_tool("how do I click the Save button") == ("screen", "read")
    assert router.detect_tool("show me the Save button")[0] == "screen"
    assert router.detect_tool("play Bohemian Rhapsody")[0] == "spotify"
    assert router.detect_tool("press play on Spotify")[0] == "spotify"
    assert router.detect_tool("click the Submit button") == ("desktop", "auto")
    body = ast.get_source_segment(
        (ROOT / "bella" / "cli.py").read_text(),
        next(n for n in ast.parse((ROOT / "bella" / "cli.py").read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == "execute_desktop_tool"),
    )
    assert "confirm_fn" in body, "desktop writes lost their confirm gate"
    assert "Click" in body and "Type" in body and "Press" in body
    # Flag off: a false "I clicked" still teaches. Flag on: do not deny clicking.
    off = cli._rewrite_false_action_claim(
        "I'm clicking the Save button now.", "click save", {"desktop_enabled": False})
    assert "don't click" in off.lower(), off
    on = cli._rewrite_false_action_claim(
        "I'm clicking the Save button now.", "click save", {"desktop_enabled": True})
    assert "don't click" not in on.lower(), on
    return "wrapper closed, teach wins, flag off"


@check("laya router schema (optional)")
def c_laya_router():
    from bella import laya_router
    from bella.config import DEFAULTS
    assert DEFAULTS.get("laya_enabled") is False, "laya_enabled must default off"
    report = laya_router.head_max_len_report()
    assert report["coarse"]["ok"], report["coarse"]
    assert report["fine_max"] <= 15, f"fine stage has {report['fine_max']} options"
    assert len(laya_router.COARSE_CATEGORIES) <= 8
    return f"coarse={report['coarse']['options']} fine_max={report['fine_max']}"


STATIC = [c_compile, c_imports, c_pyflakes, c_dead, c_leftovers,
          c_router_handlers, c_protocol, c_config, c_routes, c_laya_router, c_desktop, c_speech,
          c_tts_hang, c_as_lecture, c_as_in_tts, c_timeouts, c_panel, c_calendar, c_no_fabrication, c_named_city, c_screen_surfaces, c_cpu_contention,
          c_compose, c_overlays, c_voice]
LIVE = [c_services, c_models, c_ttfw, c_latency]


def main() -> int:
    parser = argparse.ArgumentParser(description="Bella pre-flight checks")
    parser.add_argument("--live", action="store_true",
                        help="also run checks that need the network / running services")
    args = parser.parse_args()

    for fn in STATIC + (LIVE if args.live else []):
        fn()

    width = max(len(n) for n, _, _ in _results)
    failed = 0
    for name, ok, detail in _results:
        mark = "PASS" if ok else "FAIL"
        failed += not ok
        print(f"  [{mark}] {name:<{width}}  {detail}")
    print()
    if failed:
        print(f"{failed} of {len(_results)} checkpoints FAILED")
        return 1
    print(f"all {len(_results)} checkpoints passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
