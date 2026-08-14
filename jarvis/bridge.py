"""WebSocket bridge for the JARVIS web UI.

Pure presentation-layer glue: imports and calls the SAME functions the CLI
uses (process_turn, label, load_config, MemoryStore). Zero routing, cost,
or confirmation logic lives here — that's all in jarvis/router.py and
jarvis/cli.py, completely untouched by this file. If this file disappeared,
`python3 -m jarvis.cli` and `python3 -m jarvis.voice_loop` would be
unaffected; it's a third front end on the same brain, same pattern as the
voice loop being a second one.

v1 scope: single global session (cfg + MemoryStore shared across all
connections, matching the personal single-user nature of the whole
project), one turn in flight at a time per connection (the frontend
disables input while awaiting a reply/confirmation — the same "one turn at
a time" model the CLI and voice loop already have, just enforced in the UI
instead of by blocking on input()).

Run with: jarvis/.venv/bin/python3 -m jarvis.bridge
"""
import asyncio
import base64
import json
import queue
import signal
import subprocess
import threading
import sys
import time
from pathlib import Path

import websockets
from websockets.exceptions import ConnectionClosed

from jarvis import cli, tts, voice_events
from jarvis.config import load_config
from jarvis.memory import LOG_PATH, MemoryStore
from jarvis.tools import spotify, vision

HOST = "localhost"
PORT = 8765
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Same path the standalone `python -m jarvis.voice_loop` CLI process has
# always logged to — bridge-spawned turns show up in the same place so
# there's one log to watch regardless of which front end started the loop.
VOICE_LOG_PATH = Path("/tmp/voice_loop.log")

cfg = load_config()
mem = MemoryStore()
_cfg_lock = threading.Lock()
_voice_lock = threading.RLock()
_voice_proc: subprocess.Popen | None = None

# Connected web UI clients, for the Spotify Now Playing poll broadcast
# below — v1 scope is a single global session (see module docstring), but
# this stays a set rather than a single slot so a second tab doesn't break
# anything, it just also gets the broadcast.
_connected_clients: set = set()
_spotify_poll_task: asyncio.Task | None = None
_transcript_poll_task: asyncio.Task | None = None
_voice_events_poll_task: asyncio.Task | None = None
_screen_feed_task: asyncio.Task | None = None

# Live screen card. Set by default so the card shows something the moment
# the UI opens, but the UI can switch it off (and the config can disable
# it outright) — this takes real screenshots of the user's desktop, so it
# needs an obvious off switch, not just a hidden config key.
_screen_feed_on = threading.Event()
_screen_feed_on.set()


class _WebStreamingSpeech:
    """Sentence-boundary streaming TTS for the web bridge.

    Mirrors voice_loop.StreamingSpeech's split (a dedicated worker thread so
    Piper's blocking subprocess call never stalls the thread reading the
    Ollama stream) but sends each finished clip over the websocket instead
    of playing it locally via afplay — playback happens in the browser.
    Synthesizing and sending per sentence, as soon as each one is ready,
    is what lets the browser start playing before the full reply is done
    generating, instead of waiting for one giant clip at the end.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, ws, voice_model: str):
        self.loop = loop
        self.ws = ws
        self.voice_model = voice_model
        self.queue: queue.Queue[str | None] = queue.Queue()
        self.buffer = ""
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self) -> None:
        while True:
            text = self.queue.get()
            if text is None:
                return
            if self.stopped.is_set():
                continue
            try:
                wav_bytes = tts.synthesize(text, self.voice_model)
            except tts.TTSError:
                continue
            if not wav_bytes or self.stopped.is_set():
                continue
            payload = json.dumps({
                "type": "audio",
                "data": base64.b64encode(wav_bytes).decode("ascii"),
            })
            try:
                asyncio.run_coroutine_threadsafe(self.ws.send(payload), self.loop).result()
            except ConnectionClosed:
                self.stopped.set()

    def feed(self, chunk: str) -> None:
        if not chunk or self.stopped.is_set():
            return
        with self.lock:
            self.buffer += chunk
            sentences, self.buffer = tts.split_ready_sentences(self.buffer)
        for sentence in sentences:
            self.queue.put(sentence)

    def finish(self) -> None:
        with self.lock:
            tail = self.buffer.strip()
            self.buffer = ""
        if tail:
            self.queue.put(tail)
        self.queue.put(None)
        self.worker.join()


class WSConfirm:
    """confirm_fn for process_turn(): sends a confirm_request to the browser
    and blocks the calling (worker) thread until a confirm_response arrives.
    Same synchronous confirm_fn contract cli.text_confirm and
    voice_loop.VoiceLoop.voice_confirm already implement — this is just a
    third implementation of it, not a new mechanism.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, ws):
        self.loop = loop
        self.ws = ws
        self.answers: queue.Queue = queue.Queue()

    def __call__(self, prompt: str) -> bool:
        asyncio.run_coroutine_threadsafe(
            self.ws.send(json.dumps({"type": "confirm_request", "prompt": prompt})),
            self.loop,
        )
        return self.answers.get()


def _voice_running() -> bool:
    global _voice_proc
    return _voice_proc is not None and _voice_proc.poll() is None


def _watch_voice_process(proc: subprocess.Popen) -> None:
    """Record how a spawned voice loop ended, in the voice log itself.

    Seen live: the loop was spawned, died leaving a zero-byte log, and
    nothing anywhere said why — no traceback (the process hadn't reached
    voice_loop.main()'s try yet; module imports alone take seconds and
    print nothing), and no macOS crash report. Without this, that failure
    mode is indistinguishable from "never spawned at all".

    The exit status separates the cases that were previously guesswork: a
    negative code means a signal killed it (-15 SIGTERM points at another
    process, e.g. a stray pkill; -9 SIGKILL at a force-kill or the OOM
    killer), while a positive code means Python itself exited early and
    the traceback, if any, is above this line.
    """
    started = time.monotonic()
    # Polled rather than proc.wait(): _voice_stop() waits on the same
    # Popen from the asyncio thread, and a blocking wait here holds the
    # internal waitpid lock long enough to make that one spuriously time
    # out and escalate to SIGKILL. poll() never holds it.
    while (code := proc.poll()) is None:
        time.sleep(0.25)
    elapsed = time.monotonic() - started

    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = "unknown signal"
        how = f"killed by signal {-code} ({name})"
    elif code == 0:
        how = "exited cleanly (code 0)"
    else:
        how = f"exited with code {code}"

    try:
        with open(VOICE_LOG_PATH, "a") as f:
            f.write(f"\n[bridge] voice loop (pid {proc.pid}) {how} "
                    f"after {elapsed:.1f}s.\n")
    except OSError:
        pass


def _voice_start() -> bool:
    global _voice_proc
    with _voice_lock:
        if _voice_running():
            return True

        log_file = open(VOICE_LOG_PATH, "a")
        try:
            _voice_proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "jarvis.voice_loop"],
                cwd=str(PROJECT_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            # Popen dup()s the fd for the child; our handle isn't needed
            # once the subprocess has it.
            log_file.close()
        threading.Thread(
            target=_watch_voice_process, args=(_voice_proc,), daemon=True
        ).start()
        return True


def _voice_stop() -> bool:
    global _voice_proc
    with _voice_lock:
        if not _voice_running():
            _voice_proc = None
            return True

        assert _voice_proc is not None
        _voice_proc.terminate()
        try:
            _voice_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _voice_proc.kill()
            _voice_proc.wait(timeout=5)
        finally:
            _voice_proc = None
        return True


def _voice_wake() -> bool:
    global _voice_proc
    with _voice_lock:
        started = False
        if not _voice_running():
            _voice_start()
            started = True
        assert _voice_proc is not None
        proc = _voice_proc

        def send_wake() -> None:
            if proc.poll() is None:
                proc.send_signal(signal.SIGUSR1)

        # Give a cold-started Python process time to install its signal
        # handler; model loading can continue after that point.
        if started:
            threading.Timer(0.5, send_wake).start()
        else:
            send_wake()
        return True


def _voice_interrupt() -> bool:
    """Tell a running voice loop to stop talking (SIGUSR2 -> the same path
    push-to-talk uses). Silent no-op when nothing is running, which is
    also what the web UI's Stop button means in that state."""
    with _voice_lock:
        if not _voice_running():
            return False
        assert _voice_proc is not None
        try:
            _voice_proc.send_signal(signal.SIGUSR2)
        except (ProcessLookupError, OSError):
            return False
        return True


def _voice_state_message() -> str:
    return "running" if _voice_running() else "stopped"


async def _handle_message(ws, loop: asyncio.AbstractEventLoop, confirm: WSConfirm, text: str) -> None:
    with _cfg_lock:
        tts_enabled = cfg.get("tts_enabled")
        tts_backend = cfg.get("tts_backend")
        voice_model = cfg.get("tts_voice_model")
        chat_model = cfg.get("bridge_ollama_model", cfg["ollama_model"])
        chat_max_tokens = cfg.get("bridge_ollama_max_tokens")
    # Muted whenever voice_loop is running: it and the web UI share this
    # Mac's physical speakers/mic, so anything the web UI speaks aloud can
    # bleed straight into voice_loop's microphone — seen live, a web-UI
    # shell-command reply got picked up and transcribed as if the user had
    # said it, and Ollama then hallucinated a fake "I've executed it"
    # confirmation in response to its own overheard echo. The web UI still
    # shows the text reply either way; this only suppresses the audio.
    stream_tts = tts_enabled and tts_backend == "piper" and not _voice_running()
    chat_options = {"num_predict": chat_max_tokens} if chat_max_tokens else None

    # Only the plain-Ollama backend ever calls stream_callback (see
    # cli.handle_ollama) — tool/Claude Code turns resolve to a complete
    # response in one shot regardless, so for those this closure simply
    # never fires and `streamed` stays False, leaving their behavior (one
    # "reply" + one "audio" message) exactly as before.
    streamed = False
    speech: _WebStreamingSpeech | None = None
    closed = False

    def send_now(payload: dict) -> None:
        nonlocal closed
        if closed:
            return
        try:
            asyncio.run_coroutine_threadsafe(ws.send(json.dumps(payload)), loop).result()
        except ConnectionClosed:
            closed = True

    def on_chunk(piece: str) -> None:
        nonlocal streamed, speech
        if closed:
            return
        if not streamed:
            streamed = True
            send_now({"type": "reply_start", "backend": "ollama", "label": cli.label("ollama")})
            if stream_tts:
                speech = _WebStreamingSpeech(loop, ws, voice_model)
        send_now({"type": "reply_chunk", "text": piece})
        if speech:
            speech.feed(piece)

    backend, response = await loop.run_in_executor(
        None,
        lambda: cli.process_turn(
            text,
            cfg,
            mem,
            True,
            confirm,
            model_override=chat_model,
            ollama_options=chat_options,
            stream_callback=on_chunk,
        ),
    )

    if streamed:
        if speech:
            await loop.run_in_executor(None, speech.finish)
        if not closed:
            await ws.send(json.dumps({"type": "reply_end", "backend": backend, "label": cli.label(backend)}))
        return

    try:
        await ws.send(json.dumps({
            "type": "reply",
            "backend": backend,
            "label": cli.label(backend),
            "text": response,
        }))
    except ConnectionClosed:
        return

    if stream_tts:
        try:
            wav_bytes = await loop.run_in_executor(None, tts.synthesize, response, voice_model)
            if wav_bytes:
                await ws.send(json.dumps({
                    "type": "audio",
                    "data": base64.b64encode(wav_bytes).decode("ascii"),
                }))
        except tts.TTSError as e:
            try:
                await ws.send(json.dumps({"type": "error", "message": f"TTS unavailable: {e}"}))
            except ConnectionClosed:
                pass
        except ConnectionClosed:
            pass


async def _send_voice_state(ws) -> None:
    await ws.send(json.dumps({
        "type": "voice_state",
        "running": _voice_running(),
        "state": _voice_state_message(),
    }))


async def _send_spotify_state(ws) -> None:
    loop = asyncio.get_running_loop()
    state = await loop.run_in_executor(None, spotify.now_playing_state)
    await ws.send(json.dumps({"type": "spotify_state", **state}))


async def _spotify_poll_loop() -> None:
    """Broadcasts Spotify's now-playing state to every connected web UI
    client every 2s — cheap enough (one AppleScript round-trip) to just
    poll rather than build an event-driven notifier around Spotify.app,
    which has none for scripting clients anyway. Only does the AppleScript
    call at all when a client is actually connected, and only broadcasts
    when the state actually changed, so an idle tab with Spotify closed
    costs nothing beyond one no-op process-list check every 2s.
    """
    loop = asyncio.get_running_loop()
    last_state = None
    while True:
        await asyncio.sleep(2)
        if not _connected_clients:
            continue
        try:
            state = await loop.run_in_executor(None, spotify.now_playing_state)
        except Exception:
            state = {"active": False}
        if state == last_state:
            continue
        last_state = state
        payload = json.dumps({"type": "spotify_state", **state})
        for client in list(_connected_clients):
            try:
                await client.send(payload)
            except ConnectionClosed:
                _connected_clients.discard(client)


async def _broadcast(payload: str) -> None:
    for client in list(_connected_clients):
        try:
            await client.send(payload)
        except ConnectionClosed:
            _connected_clients.discard(client)


def _read_new_voice_events(offset: int) -> tuple[int, list[dict]]:
    """Reads whatever voice_loop appended to the events file since `offset`.

    Returns the new offset alongside the events. A file smaller than the
    offset means voice_events truncated it (it caps its own size), so the
    read restarts from the top rather than seeking past the end forever.
    """
    path = voice_events.EVENTS_PATH
    if not path.exists():
        return 0, []
    size = path.stat().st_size
    if size < offset:
        offset = 0
    if size == offset:
        return offset, []
    with path.open("r") as f:
        f.seek(offset)
        chunk = f.read()
        new_offset = f.tell()
    events = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return new_offset, events


async def _voice_events_poll_loop() -> None:
    """Relays the voice loop's real speech telemetry to the web UI.

    voice_loop.py plays its audio on the Mac's speakers, so the browser
    never hears a spoken turn and its analyser has nothing to show. This
    forwards the measured amplitude envelope of the clip actually being
    played (see jarvis/voice_events.py) so the orb moves with real speech
    instead of a fabricated animation.

    Polled faster than the transcript loop because this one is used for
    animation: the envelope carries its own timeline, but its arrival is
    what marks "JARVIS started talking now".
    """
    loop = asyncio.get_running_loop()
    # Start from the end of the file: events from before this bridge
    # existed describe speech that finished long ago.
    offset = 0
    if voice_events.EVENTS_PATH.exists():
        offset = voice_events.EVENTS_PATH.stat().st_size

    while True:
        await asyncio.sleep(0.1)
        try:
            offset, events = await loop.run_in_executor(
                None, _read_new_voice_events, offset
            )
        except OSError:
            continue
        if not events or not _connected_clients:
            continue
        for event in events:
            if event.get("type") == "speaking":
                await _broadcast(json.dumps({
                    "type": "voice_audio",
                    "envelope": event.get("envelope", []),
                    "hz": event.get("hz", voice_events.ENVELOPE_HZ),
                    "duration": event.get("duration", 0),
                }))
            elif event.get("type") == "speaking_end":
                await _broadcast(json.dumps({"type": "voice_audio_end"}))


async def _screen_feed_loop() -> None:
    """Pushes a real, downscaled screenshot to the UI's screen card.

    The card used to loop a canned video clip labelled "Live Clip", which
    is exactly the kind of thing this project doesn't do — so it now shows
    the actual screen, via the same macOS `screencapture` the screen tool
    already uses. Capture only happens when a client is watching AND the
    feed is switched on, so nothing is screenshotted while the tab is
    closed. A permission failure stops the loop entirely rather than
    retrying every 2s forever: it can't fix itself without the user
    granting Screen Recording and restarting.
    """
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(2)
        if not _connected_clients or not _screen_feed_on.is_set():
            continue
        with _cfg_lock:
            if not cfg.get("screen_feed_enabled", True):
                continue
        try:
            jpeg = await loop.run_in_executor(None, vision.capture_screen_jpeg)
        except vision.VisionError as e:
            await _broadcast(json.dumps({"type": "screen_frame", "error": str(e)}))
            return
        except Exception:
            continue
        await _broadcast(json.dumps({
            "type": "screen_frame",
            "data": base64.b64encode(jpeg).decode("ascii"),
        }))


def _count_log_lines() -> int:
    if not LOG_PATH.exists():
        return 0
    return len(LOG_PATH.read_text().strip().splitlines())


def _read_new_log_entries(since_count: int) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text().strip().splitlines()
    entries = []
    for line in lines[since_count:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


async def _transcript_poll_loop() -> None:
    """Relays voice_loop.py's conversation into every connected web UI
    client's Comms Log. voice_loop runs as a wholly separate OS process
    (spawned by _voice_start) with no shared in-memory state — the on-disk
    conversation log is the only thing connecting it to this bridge
    process, so this polls that file the same way _spotify_poll_loop polls
    AppleScript. Only source=="voice" entries are forwarded — turns
    dispatched through the web UI itself (source=="text", the default —
    see MemoryStore.log_turn) are already broadcast live by
    _handle_message, so relaying them here too would show every web-typed
    turn twice.
    """
    loop = asyncio.get_running_loop()
    last_count = await loop.run_in_executor(None, _count_log_lines)
    while True:
        await asyncio.sleep(1)
        if not _connected_clients:
            # Don't accumulate a backlog while nobody's watching — a client
            # that connects later should only see NEW turns from that point
            # on, not a dump of everything said while the tab was closed.
            last_count = await loop.run_in_executor(None, _count_log_lines)
            continue
        entries = await loop.run_in_executor(None, _read_new_log_entries, last_count)
        if not entries:
            continue
        last_count += len(entries)
        for entry in entries:
            if entry.get("source") != "voice":
                continue
            backend = entry.get("backend") or ""
            payload = json.dumps({
                "type": "transcript_turn",
                "role": entry.get("role"),
                "content": entry.get("content"),
                "label": cli.label(backend),
            })
            for client in list(_connected_clients):
                try:
                    await client.send(payload)
                except ConnectionClosed:
                    _connected_clients.discard(client)


async def handle_connection(ws) -> None:
    loop = asyncio.get_running_loop()
    confirm = WSConfirm(loop, ws)
    _connected_clients.add(ws)

    try:
        await _send_voice_state(ws)
        await _send_spotify_state(ws)
        with _cfg_lock:
            feed_allowed = cfg.get("screen_feed_enabled", True)
        await ws.send(json.dumps({
            "type": "screen_feed_state",
            "enabled": feed_allowed and _screen_feed_on.is_set(),
        }))
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")

            if mtype == "confirm_response":
                confirm.answers.put(bool(msg.get("answer")))

            elif mtype == "voice_start":
                _voice_start()
                await _send_voice_state(ws)

            elif mtype == "voice_stop":
                _voice_stop()
                await _send_voice_state(ws)

            elif mtype == "voice_wake":
                _voice_wake()
                await _send_voice_state(ws)

            elif mtype == "voice_interrupt":
                _voice_interrupt()

            elif mtype == "mute":
                with _cfg_lock:
                    cfg["tts_enabled"] = False
                await ws.send(json.dumps({"type": "muted", "enabled": False}))

            elif mtype == "unmute":
                with _cfg_lock:
                    cfg["tts_enabled"] = True
                await ws.send(json.dumps({"type": "muted", "enabled": True}))

            elif mtype == "screen_feed":
                enabled = bool(msg.get("enabled"))
                if enabled:
                    _screen_feed_on.set()
                else:
                    _screen_feed_on.clear()
                await _broadcast(json.dumps({
                    "type": "screen_feed_state", "enabled": enabled,
                }))

            elif mtype == "message":
                text = (msg.get("text") or "").strip()
                if text:
                    # Dispatched as a task, NOT awaited directly — process_turn
                    # can block on confirm.answers.get() waiting for a
                    # confirm_response, which can only arrive if this receive
                    # loop stays free to keep reading incoming messages.
                    # Awaiting it here would deadlock the moment a
                    # confirmation is needed.
                    asyncio.create_task(_handle_message(ws, loop, confirm, text))
    except ConnectionClosed:
        pass
    finally:
        _connected_clients.discard(ws)


async def main() -> None:
    print(f"JARVIS bridge listening on ws://{HOST}:{PORT}")
    # Reference kept deliberately — asyncio only holds a weak reference to
    # tasks, so an unreferenced create_task() result is eligible for
    # garbage collection mid-run, silently killing the poll loop.
    global _spotify_poll_task, _transcript_poll_task
    global _voice_events_poll_task, _screen_feed_task
    _spotify_poll_task = asyncio.create_task(_spotify_poll_loop())
    _transcript_poll_task = asyncio.create_task(_transcript_poll_loop())
    _voice_events_poll_task = asyncio.create_task(_voice_events_poll_loop())
    _screen_feed_task = asyncio.create_task(_screen_feed_loop())
    async with websockets.serve(handle_connection, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
