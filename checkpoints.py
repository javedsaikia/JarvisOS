#!/usr/bin/env python3
"""Pre-flight checks for Max. Run before testing, and before committing.

    max/.venv/bin/python3 checkpoints.py          # static + wiring (fast)
    max/.venv/bin/python3 checkpoints.py --live   # also hits the network

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
    p for p in (ROOT / "max").rglob("*.py")
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
    r = subprocess.run([sys.executable, "-m", "compileall", "-q", str(ROOT / "max")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return f"{len(PY_FILES)} files"


@check("imports")
def c_imports():
    mods = ["max.cli", "max.router", "max.bridge", "max.voice_loop", "max.llm",
            "max.compose", "max.instructions", "max.tools.vision", "max.screen_capture",
            "max.sarvam_client", "max.reflection", "max.memory", "max.persona"]
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
    # Definitions come from max/ only, but a caller may legitimately live
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
    from max import router
    src = (ROOT / "max" / "cli.py").read_text()
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
    return f"{len(routed)} routed + {len(state)} state-driven"


@check("bridge <-> HUD protocol is symmetric")
def c_protocol():
    bridge = (ROOT / "max" / "bridge.py").read_text()
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
    from max.config import DEFAULTS
    import json
    saved = json.loads((ROOT / "max" / "config.json").read_text())
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
    ("what is this picture", "screen/region"), ("let me circle something", "screen/region"),
    ("write an email about pricing", "compose/auto"),
    ("write me a prompt to review this code", "compose/auto"),
    ("draft an email to sabina@gmail.com", "email/write"), ("check my email", "email/read"),
    ("what's the weather", "location/weather"), ("where is the nearest ATM", "location/nearby"),
    ("where am I", "location/here"), ("call Nova", "call/auto"),
    ("play High Hopes on Spotify", "spotify/play_track"), ("pause the music", "spotify/pause"),
    ("run ls -la", "shell/auto"), ("what can you do", "capabilities/read"),
    ("tell me about the movie Inception", "omdb/auto"), ("tell me a joke", "joke/auto"),
    ("what's on my calendar tomorrow", "calendar/read"),
    ("what movies are showing at the cinema", "location/showtimes"),
    # must NOT be captured by any tool
    ("where is Paris", None), ("show me a good recipe for pasta", None),
    ("how do I show gratitude", None), ("point out the obvious", None),
]


@check("router behaviour matrix")
def c_routes():
    from max import router
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
    from max.voice_loop import _split_for_speech, _clean_for_speech, _FIRST_MAX_CHARS
    texts = [
        "Yes.",
        'A man asked, "Books on Pavlov?" The librarian replied, "It rings a bell." Ha.',
        "This is a Microsoft Word window. What do you want help with?",
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
    from max import voice_loop as vl, sarvam_client, voice_events, cli
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


@check("network clients wrap bare timeouts")
def c_timeouts():
    import urllib.request
    from max import sarvam_client as sc, llm
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
    from max import instructions as ins
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
    from max.voice_loop import _strip_confirm_echo, CONFIRM_YES_RE, CONFIRM_NO_RE
    from max.tools import parsing, event_draft

    # A confirmation must never be decided by Max's own prompt coming back
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
    from max import cli, router
    src = (ROOT / "max" / "cli.py").read_text()
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
    from max import cli
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
                 "who is in the picture", "describe it", "read the screen"):
        assert cli._SCREEN_CONTENT_QUESTION_RE.search(text), f"not claimed: {text!r}"
    for text in ("how are you", "how do I bold this"):
        assert not cli._SCREEN_CONTENT_QUESTION_RE.search(text), f"over-claimed: {text!r}"
    return "3 fabrications, 4 pass-through, 7 screen"


@check("a named city beats GPS")
def c_named_city():
    from max import cli
    from max.tools import location
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
    # A source Max cannot reach is named rather than silently substituted.
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

    src = (ROOT / "max" / "cli.py").read_text()
    for fn in ("handle_nearby", "handle_showtimes", "handle_phone_lookup"):
        body = ast.get_source_segment(
            src, next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == fn))
        assert "search_origin" in body, f"{fn} still assumes the current GPS position"
    return "Guwahati via 4 phrasings, 3 handlers, 3 local kept"


@check("nothing competes with the microphone")
def c_cpu_contention():
    from max import control, reflection
    from max.tools import spotify, vision

    # A local 7B reflection starting mid-conversation took the next
    # transcription from 0.73s to 28.55s. The timer arms on turn END, so
    # activity is marked at the WAKE, and reflection re-checks it.
    control.note_activity()
    assert control.seconds_since_activity() < 2, "activity marker did not record"
    src = (ROOT / "max" / "reflection.py").read_text()
    assert "seconds_since_activity" in src, \
        "reflection no longer checks whether a conversation is in progress"
    assert "control.note_activity()" in (ROOT / "max" / "voice_loop.py").read_text(), \
        "the voice loop no longer marks activity at the wake"
    # The local vision model is the other CPU hog: prefer the cloud one.
    vsrc = (ROOT / "max" / "tools" / "vision.py").read_text()
    assert "_ask_vision_model(jpeg, prompt, cfg)" not in vsrc, \
        "the low-text screen path is back on the local vision model"

    # Asking for help with a move must reach the screen, not the how-to
    # path — which asked a text model about a board it could not see.
    for text in ("can you help me with the next move?", "what should my next move be",
                 "suggest a move", "my next move"):
        assert vision.is_advice_request(text), f"not routed to advice: {text!r}"
    for text in ("how do I bold this", "what is on my screen", "how are you"):
        assert not vision.is_advice_request(text), f"over-claimed as advice: {text!r}"

    # A sentence about yourself is not a song title, however recently
    # music played. "and I am using the white one" got "I didn't catch
    # the song."
    spotify.note_music_turn()
    for text in ("and I am using the white one", "I am playing white", "it's my turn now"):
        assert not spotify.looks_like_followup(text), f"claimed as a song: {text!r}"
    for text in ("by Metallica", "Anuradha", "Bohemian Rhapsody"):
        assert spotify.looks_like_followup(text), f"real follow-up lost: {text!r}"
    return "reflection gated, 4 advice, 6 song"


@check("compose guards")
def c_compose():
    from max import compose
    assert compose.detect("draft an email to bob@example.com") is None, \
        "compose stole an addressed email from the Gmail flow"
    assert compose.detect("write an email about pricing") == "email"
    assert compose.detect("what's the weather") is None
    assert compose.wants_insert("go ahead and type it in")
    assert not compose.wants_insert("what's the weather")
    assert compose.subject_of("Subject: Hello\n\nBody") == "Hello"
    assert "```" not in compose._tidy("```\ntext\n```")
    return "7 assertions"


@check("overlay helper processes are importable")
def c_overlays():
    for mod in ("max.cursor_overlay", "max.instruction_panel", "max.region_select"):
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
    voice = subprocess.run(["pgrep", "-f", "max.voice_loop"], capture_output=True)
    return "HUD + bridge up" + (", voice up" if voice.returncode == 0 else ", voice DOWN")


@check("assigned models actually answer")
def c_models():
    from max.config import load_config
    from max import llm
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
    from max import sarvam_client as sc
    from max.voice_loop import _split_for_speech, _clean_for_speech
    reply = "This is a Microsoft Word window. What do you want help with?"
    first = _split_for_speech(_clean_for_speech(reply))[0]
    sc._tts_wav_cache.clear()
    started = time.perf_counter()
    sc.text_to_speech(first, language_code="en-IN")
    elapsed = time.perf_counter() - started
    assert elapsed < 4.0, f"{elapsed:.2f}s to first word (was ~1s)"
    return f"{elapsed:.2f}s for a {len(first)}-char opener"


@check("hot-path tool latency")
def c_latency():
    from max.config import load_config
    from max.memory import MemoryStore
    from max import cli
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
    from max.config import load_config
    from max import speech, elevenlabs_client
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


STATIC = [c_compile, c_imports, c_pyflakes, c_dead, c_leftovers,
          c_router_handlers, c_protocol, c_config, c_routes, c_speech,
          c_tts_hang, c_timeouts, c_panel, c_calendar, c_no_fabrication, c_named_city, c_cpu_contention,
          c_compose, c_overlays, c_voice]
LIVE = [c_services, c_models, c_ttfw, c_latency]


def main() -> int:
    parser = argparse.ArgumentParser(description="Max pre-flight checks")
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
