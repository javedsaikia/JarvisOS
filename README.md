# JARVIS

A cost-optimized personal AI assistant for macOS. Everything defaults to a
local model (Ollama) and stays local — memory, routing, Calendar/Notes,
file/shell tools, speech in, speech out. Nothing costs a token unless a
request is genuinely complex enough to need Claude Code, and even then it
always asks first.

## Quickstart

Three front ends, same brain underneath (`jarvis/cli.py`'s router → backend
pipeline):

```bash
# Text loop — type, or dictate with Wispr Flow into the terminal prompt
python3 -m jarvis.cli

# Always-on voice loop — say "Hey Jarvis", no typing at all
jarvis/.venv/bin/python3 -m jarvis.voice_loop

# Web UI — audio-reactive orb + transcript, needs the bridge + dev server (two terminals):
jarvis/.venv/bin/python3 -m jarvis.bridge
cd frontend && npm install && npm run dev   # → http://localhost:5173
```

First run creates `jarvis/config.json` — set `user_name` there. Everything
else has a sensible default; see [Config](#config) below for what's tunable.

**How a request gets handled**, cheapest first — each tier only escalates to
the next if it doesn't match:

1. **Router keyword match** (`jarvis/router.py`, zero cost, no model call) →
   Calendar/Notes (AppleScript), Files (Ollama + tools), or Shell
   (regex-extracted, deterministic — see [File & shell tools](#file--shell-tools))
2. **Ollama** (`qwen2.5-coder`, local, free) — the default for everything else
3. **Claude Code** — only for requests that are clearly multi-file
   coding/agentic work, and only after you confirm the handoff

Every reply is labeled with what answered it: `[Ollama]`, `[Calendar]`,
`[Notes]`, `[Files]`, `[Shell]`, or `[Claude Code]`.

**Project layout:**

| Path | What it is |
|---|---|
| `jarvis/cli.py` | Text loop, router dispatch, confirmation logic |
| `jarvis/voice_loop.py` | Always-on mic loop (wake word → STT → same pipeline → TTS) |
| `jarvis/router.py` | Rule-based backend/tool routing, zero token cost |
| `jarvis/ollama_client.py` | stdlib-only client for local Ollama |
| `jarvis/claude_handoff.py` | Shells out to the `claude` CLI |
| `jarvis/memory.py` | Facts + conversation log persistence |
| `jarvis/persona.py` | JARVIS system prompt — personality, addresses you by name |
| `jarvis/tts.py` | Piper text-to-speech (synthesize/play split for the web UI — see [Web UI](#web-ui)) |
| `jarvis/tools/` | Calendar, Notes, files, shell, and the file/shell tool-call registry |
| `jarvis/bridge.py` | WebSocket bridge for the web UI — presentation-layer glue only, zero routing/cost logic of its own |
| `jarvis/.venv/` | Isolated Python env for Piper/openWakeWord/faster-whisper/websockets (never touches system Python) |
| `jarvis/config.json` | All settings — see [Config](#config) |
| `frontend/` | Vite + TypeScript + Three.js web UI — talks only to `jarvis/bridge.py`, never to a model directly |

Sections below go deep on each piece; read them when you need to tune or
debug something specific.

## Voice input (Wispr)

Wispr Flow's dictation works at the OS level — it types into whatever text
field has focus. There's no separate wiring needed: run the interactive loop
above, click into the terminal, and dictate straight into the `You:` prompt.
It's transcribed, submitted on your pause, and goes through the normal
router → backend → confirmation path exactly like typed input.

`--once "<text>"` also exists for scripted/batch use, but it skips the Claude
Code confirmation prompt (there's no terminal to confirm in) — use the
interactive loop for anything where you want that safety check to apply.

```bash
python3 -m jarvis.cli --once "what's on my calendar today"
```

## Config

`jarvis/config.json` (created on first run, editable):

- `user_name` — how JARVIS addresses you
- `default_backend` — `"ollama"` (keep this — never set to a paid backend as default)
- `ollama_model` / `ollama_host` — local model to use
- `claude_code_enabled` — set `false` to fully disable the escalation path
- `claude_code_command` — the CLI command to invoke (default `"claude"`)
- `claude_code_confirm` — ask before handing off to Claude Code (recommended, since it can take real actions)
- `tools_confirm_writes` — ask before creating a calendar event or note (recommended, since misheard dictation could create the wrong thing)
- `tts_enabled` — speak every reply out loud via Piper (default `true`)
- `tts_backend` — `"piper"` (the only backend implemented)
- `tts_voice_model` — path to the Piper `.onnx` voice model, relative to the project root
- `voice_wake_word` — openWakeWord pretrained model for the always-on voice loop (default `"hey_jarvis"`)
- `voice_stt_model` — faster-whisper model size for transcribing voice commands (default `"small.en"`)
- `voice_context_turns` — how many recent turns the voice loop includes in the prompt; lower is faster
- `voice_ollama_model` — optional smaller model for spoken turns only; defaults to the main `ollama_model`
- `voice_ollama_max_tokens` — cap for voice replies; lower values reduce response latency
- `voice_silence_seconds` — how long the voice loop waits after you stop speaking before it hands off to STT
- `voice_silence_rms_threshold` — mic energy level below which the voice loop considers you done talking; raise it in a noisy room, lower it if it cuts you off early

## Personality

`jarvis/persona.py` builds the system prompt: calm, competent, slightly
formal British-butler tone with dry wit — classic JARVIS. Addresses you by
`user_name` from config (not every line — just occasionally). Concise by
default, more detail on request. This prompt is only used on the Ollama
path (`handle_ollama` in `cli.py`) — Calendar/Notes/Files tool replies are
generated directly from tool output, and Claude Code has its own voice.

## Switching models

Edit `ollama_model` in `config.json` to any model you've pulled (`ollama list`).
No code changes needed — `ollama_client.py` is a thin stdlib wrapper around the
local `/api/chat` endpoint.

## Forcing a backend

- `/ollama <message>` — force local model for this turn
- `/claude <task>` — force Claude Code handoff for this turn
- Otherwise, `jarvis/router.py` decides via keyword rules (zero token cost,
  no model call). Coding/build/refactor/debug/deploy-shaped requests escalate;
  everything else stays local.

## Memory

- `jarvis/memory/facts.md` — durable facts/preferences, human-editable, fed
  into the system prompt every turn. Add with `/remember <fact>`.
- `jarvis/memory/conversation_log.jsonl` — recent turn history for context.

## Calendar & Notes (AppleScript, no Ollama/Claude call needed)

`jarvis/router.py` also recognizes Calendar/Notes-shaped requests and routes
them straight to `jarvis/tools/calendar.py` / `jarvis/tools/notes.py` — pure
AppleScript via `osascript`, zero model calls, zero token cost. This is
checked *before* the Ollama/Claude Code decision, since it's the cheapest
possible path.

Reads (no confirmation):
- "what's on my calendar today" / "...this week"
- "search my notes for dentist" / "list my notes"

Writes (asks to confirm first, unless `tools_confirm_writes` is `false`):
- "schedule a call with Bob tomorrow at 3pm"
- "take a note: buy milk and eggs"

Limitations:
- Date/time parsing (`jarvis/tools/parsing.py`) is regex-based, not a full
  NLP date parser. It handles "tomorrow", weekday names, "at 3pm", "for 30
  minutes". An hour with no am/pm is guessed (1–7 → PM, else AM). Unusual
  phrasing may need a retry with more explicit wording.
- Wide calendar date ranges (`days` > ~7) can take Calendar.app 15–30s to
  answer — this is `osascript`/Calendar.app's own `whose`-filter slowness,
  not a bug; `list_events` uses a 60s timeout to accommodate it.
- Notes are created in your default Notes account (`iCloud` here).

## File & shell tools

Files and shell commands are handled by two **separately routed** paths —
this split exists because an early version routed both through Ollama
tool-calling, and that turned out to be unsafe for shell specifically (see
below). No Claude Code involved in either path.

### Files (Ollama tool-calling)

File paths and content are too free-form for regex extraction, so
`jarvis/cli.py` gives Ollama the tool schemas from `jarvis/tools/registry.py`
and lets *it* decide which tool to call and with what arguments (`read_file`,
`write_file`, `list_dir`, defined in `jarvis/tools/files.py`). Still 100%
local/free — it's Ollama, just with tools attached.

- "read the file README.md", "list the files in jarvis/tools" — read-only, no confirmation
- "write ... to a file called notes.txt" — asks to confirm first (`tools_confirm_writes`)

Safety/accuracy notes:
- `read_file`/`write_file`/`list_dir` are hard-scoped to the project
  directory — `jarvis/tools/files.py` resolves and rejects any path that
  escapes it (path traversal via `../` is blocked, verified).
- Ollama's model template doesn't populate the native `tool_calls` field for
  `qwen2.5-coder` — it emits the call as a JSON object in plain `content`
  instead (verified empirically), sometimes wrapped in a markdown code
  fence, sometimes with conversational text before it. `jarvis/tools/registry.py`
  scans for a JSON object anywhere in the reply (not just at the start) to
  handle this, and also reads the native `tool_calls` field if a future
  model populates it correctly.
- **The model doesn't always call a tool even when told to** — observed
  live: asked to run a shell command, it sometimes just fabricated a
  plausible-looking answer directly instead of emitting the tool call. When
  that happens the fabricated text was being shown as the answer with the
  real tool never invoked. This is *why* shell has its own non-LLM-mediated
  path below rather than living in this one.
- When `read_file` relays a long file back through the phrasing pass, the
  model can subtly paraphrase rather than quote verbatim — observed one
  fabricated sentence when asked to relay this README back. Don't treat a
  `[Files]` read-back of file content as guaranteed verbatim; check the
  actual file for anything that matters.
### Shell (deterministic — never asks Ollama whether to run something)

Shell commands are **not** LLM-mediated. `jarvis/router.py` detects an
explicit "run"/"execute" request (`SHELL_PATTERNS`, checked before the
`files` patterns so it always wins), `jarvis/tools/parsing.py`'s
`extract_shell_command()` regex-extracts the literal command from the
phrasing, and `jarvis/cli.py`'s `handle_shell()` confirms and runs it
directly via `jarvis/tools/shell.py` — no model ever decides *whether* to
run it, and the model never sees (or can misrepresent) the output, since
there's no phrasing pass afterward; you get the raw command output back.

- "run ls -la", "Run this exact shell command: ls -la", "execute git status" → `[Shell]`
- **Always** asks to confirm first — `[Router] Run this shell command? \`ls -la\``
  — unconditionally, regardless of `tools_confirm_writes`. This gate cannot
  be disabled from config; `handle_shell()` doesn't check that setting at all.
- `run_shell` is **not** sandboxed beyond running with cwd set to the
  project root — safety is the confirmation prompt showing you the exact
  command before it runs, not a technical restriction on what it can do.
- "run" is treated as literal and wins over Claude Code even when the
  command overlaps a `CLAUDE_CODE_PATTERNS` keyword (e.g. "run git status"
  → local shell tool, not Claude Code) — the distinction is that "run X"
  names an exact command, while "commit this" / "install the package"
  describe an outcome Claude Code still needs to figure out the specifics of.

Both this and the file tool-calling path were tried as one combined
LLM-mediated domain first; that's what "Ollama sometimes doesn't call the
tool it's told to" above is describing, and why shell was split out into
its own deterministic path once that was caught live.

## Voice output (Piper)

Every reply is spoken aloud via [Piper](https://github.com/rhasspy/piper), a
local neural TTS engine — no cloud calls, no cost. It's installed in an
isolated project-local virtualenv (`jarvis/.venv`) so it never touches system
Python; delete that directory to fully remove it.

Setup (already done in this repo, for reference / reinstalling elsewhere):

```bash
python3 -m venv jarvis/.venv
jarvis/.venv/bin/pip install piper-tts
# voice model (British male, fits the JARVIS persona):
mkdir -p jarvis/voices
curl -sL -o jarvis/voices/en_GB-alan-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx"
curl -sL -o jarvis/voices/en_GB-alan-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json"
```

Controls:
- `/mute`, `/unmute` — toggle speech for the current session
- `--no-speak` — disable TTS for a single run regardless of config
- `tts_enabled: false` in config — disable permanently

To use a different voice, download another `.onnx`/`.onnx.json` pair from the
[Piper voices list](https://huggingface.co/rhasspy/piper-voices/tree/main/en)
and point `tts_voice_model` at it.

## Always-on voice loop

```bash
jarvis/.venv/bin/python3 -m jarvis.voice_loop
```

Continuous mic capture — no terminal typing, no per-turn dictation. Say
"Hey Jarvis", then your request; it transcribes, routes, and speaks the
reply, then goes back to listening. This is a genuinely different pipeline
from the text loop's Wispr integration, because Wispr Flow doesn't do
ambient background listening — it's a dictation tool bound to a focused
text field. So this mode uses its own fully local stack instead:

- **Wake word**: [openWakeWord](https://github.com/dscripka/openWakeWord)'s
  pretrained `hey_jarvis` model (ONNX, runs continuously on CPU, no account/API key needed)
- **Transcription**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (`small.en` by default), only run on the few seconds of audio after the
  wake word fires — not continuously
- **Utterance boundary**: simple RMS-energy silence detection (`jarvis/voice_loop.py`)
  — not full WebRTC VAD, kept minimal on purpose. The threshold is
  **auto-calibrated at startup** (`VoiceLoop.calibrate()`): it samples ~2.5s
  of real ambient noise, takes the *median* (not max — a single spike like a
  click or cough isn't representative and will throw off a max-based
  estimate, confirmed by hitting this exact bug live), and sets the working
  threshold to `median × 2.2`, floored at `voice_silence_rms_threshold` in
  config. Stay quiet during the "Calibrating mic..." message at startup.
  If it's still over/under-triggering for your room, raise your system mic
  input volume (System Settings → Sound → Input) for more headroom before
  tuning the config value — low input gain was the actual root cause the
  first time this got tested live, not the threshold math.

Confirmations (Claude Code handoff, calendar/note writes, shell commands)
work the same way here as in the text loop, just spoken instead of typed:
JARVIS speaks the question aloud, then listens for a short yes/no reply.
`jarvis/cli.py`'s `process_turn`/`handle_tool`/etc. all take a swappable
`confirm_fn` for exactly this — `text_confirm` (types) for the terminal
loop, `VoiceLoop.voice_confirm` (speaks + listens) here. Same core pipeline,
two front ends.

Setup (already done in this repo, for reference / reinstalling elsewhere):

```bash
jarvis/.venv/bin/pip install sounddevice numpy openwakeword faster-whisper
jarvis/.venv/bin/python3 -c "from openwakeword.utils import download_models; download_models(['hey_jarvis'])"
# faster-whisper downloads its model automatically on first run
```

Tested twice: first end-to-end with synthesized "Hey Jarvis, ..." phrases
(Piper) played through the speakers into the real mic, then live with
Javed's actual voice. The live run surfaced a real calibration bug the
synthetic test couldn't have caught (see below) — fixed and re-verified live
afterward: wake word fired correctly on real speech, "What's the weather
like today?" transcribed and answered correctly, and a note-creation
request went through the full voice-confirm (spoke the question, correctly
parsed a spoken "yes") end to end.

Limitations:
- Requires `jarvis/.venv`'s Python specifically (`sounddevice`/`openwakeword`/
  `faster-whisper` aren't in system Python) — always launch with
  `jarvis/.venv/bin/python3 -m jarvis.voice_loop`, not plain `python3`.
- Only `hey_jarvis` has been tested; openWakeWord also ships `alexa`,
  `hey_mycroft`, `hey_rhasspy` pretrained models if you'd rather use one of those.
- No barge-in — JARVIS can't be interrupted mid-response; it finishes
  speaking before listening for the next wake word.
- RMS silence detection is a simple energy threshold, not true voice
  activity detection — a loud enough room can extend recording past when
  you've actually stopped talking (observed: one utterance ran the full 15s
  cap without ever finding 1.2s of continuous quiet, despite transcribing fine).
- `small.en` sometimes mishears short/quiet utterances (observed one
  transcribed as "watch my location" that was likely something else) —
  normal for a small local STT model, not a pipeline bug.
- Calibration is one-shot at startup, not continuous — if room noise changes
  significantly mid-session (AC kicks on, etc.), restart the loop to recalibrate.

## Web UI

An audio-reactive orb + transcript panel — a visual front end, nothing
more. It does not route, does not decide anything, does not talk to any
model directly. It talks to `jarvis/bridge.py` over a WebSocket, and
`jarvis/bridge.py` calls the *exact same* `process_turn()` / `label()` /
`load_config()` / `MemoryStore` that `cli.py` and `voice_loop.py` already
use. If `bridge.py` were deleted, `python3 -m jarvis.cli` and
`python3 -m jarvis.voice_loop` would be completely unaffected — this is a
third front end bolted onto the existing brain, not a rewrite of any part
of it.

### Run it

Two processes, two terminals:

```bash
# Terminal 1 — the bridge (backend)
jarvis/.venv/bin/python3 -m jarvis.bridge      # ws://localhost:8765

# Terminal 2 — the UI (frontend)
cd frontend
npm install
npm run dev                                     # http://localhost:5173
```

Open the printed `localhost:5173` URL. If the bridge isn't running yet (or
drops), the UI shows a "Disconnected — reconnecting…" banner and retries
with backoff — it doesn't fall back to doing anything itself.

### How it connects (and what it deliberately can't do)

`jarvis/bridge.py` is presentation-layer glue, not a second implementation
of anything:

- **Sending a message**: UI sends `{"type": "message", "text": "..."}` →
  bridge calls `cli.process_turn(text, cfg, mem, interactive=True, confirm_fn=...)`
  in a worker thread (it's synchronous — blocking network/subprocess calls,
  same as the CLI) → sends back `{"type": "reply", "label": cli.label(backend), "text": response}`.
  The label shown is *always* `cli.label()`'s actual output — Ollama,
  Calendar, Notes, Files, Shell, Claude Code — never reimplemented or
  guessed at in the frontend.
- **Confirmations are not bypassed**: when `process_turn` needs to confirm
  something (Claude Code handoff, a calendar/note write, any shell
  command), its `confirm_fn` sends `{"type": "confirm_request", "prompt": "..."}`
  to the browser and **blocks the worker thread** until the browser answers
  with `{"type": "confirm_response", "answer": true|false}`. This is the
  same synchronous `confirm_fn` contract `cli.text_confirm` (types) and
  `voice_loop.VoiceLoop.voice_confirm` (speaks + listens) already
  implement — the UI is a third implementation of the same interface, not
  a new bypass path. The input box disables itself while a confirmation
  (or a reply) is pending, so you can't fire a second request mid-flow.
- **"Force Local" / "Ask Claude Code" buttons**: these just prepend
  `/ollama ` or `/claude ` to whatever's in the text box before sending —
  reusing `process_turn`'s *existing* prefix handling verbatim. Zero new
  server-side logic. `/claude` still goes through the exact same
  `claude_code_confirm` gate as typing it in the terminal would.
- **TTS plays in the browser, not on the server**: `jarvis/tts.py` was
  split into `synthesize()` (Piper → WAV bytes, no playback) and `play()`
  (afplay, unchanged) — `speak()` still calls both in sequence, so
  `cli.py` and `voice_loop.py` behave identically to before. The bridge
  calls `synthesize()` only, base64-encodes the WAV, and sends it to the
  browser, which decodes it through the Web Audio API, plays it, and feeds
  a live `AnalyserNode` reading to the orb every frame — the orb's glow
  during speech is driven by the actual audio energy, not a fake pulse.
- **Mute/unmute**: `{"type": "mute"}` / `{"type": "unmute"}` toggle
  `cfg["tts_enabled"]` for the bridge's session — same in-memory-only
  toggle `cli.py`'s `/mute`/`/unmute` already does, just triggered by a
  button instead of a slash command.

### Visual design

Electric-blue holographic HUD modelled on the classic JARVIS interface
look: a compact glowing core inside a layered SVG reticle (tick ring,
segmented arcs, dashed ring, node dots, corner brackets — each rotating at
a different rate), telemetry cards down the left, comms log right, and a
routing-tier icon strip along the bottom.

**Every readout is bound to real data** — the reference material for this
design is a marketing poster with mocked-up panels (lights, temperature,
security), and none of that was reproduced, because we have no such data:

| Element | Source |
|---|---|
| System Status oscilloscope | live `AnalyserNode` time-domain data (real TTS audio) |
| Output Spectrum bars | live `AnalyserNode` frequency data |
| Level History | rolling history of measured output amplitude |
| Audio Out meter + ring level arc | same real amplitude that drives the orb |
| Status dot / label | real pipeline state (idle/thinking/speaking/error) |
| Comms Log label + backend chip | `cli.label()` output, verbatim |
| Routing-tier strip highlight | lights the tier matching the real backend that answered |
| Session / Messages | real elapsed time and real sent-message count |

When nothing is playing, the visualizers render a **flat "no signal"
baseline** rather than idle noise — they never imply audio that isn't
there. The only purely decorative elements are the background grid,
vignette, scanline sweep, and the two scrolling hex columns at the screen
edges; all are `aria-hidden` and none are presented as data.

### Scope / limitations (v1)

- **Single global session**: one `cfg` + one `MemoryStore`, shared across
  all connections — matches the personal, single-user nature of the whole
  project (same memory the CLI and voice loop read/write).
- **One turn in flight at a time per connection**: the frontend disables
  input while awaiting a reply or confirmation. This mirrors how the CLI
  and voice loop already work (both process one turn at a time, blocking on
  `input()` or on mic capture) — not a new restriction, just the same model
  enforced in the UI instead of by blocking a terminal.
- **Text input only** — the UI sends typed text, not audio, to the bridge.
  Voice input still means either dictating into the CLI's terminal (Wispr)
  or running the separate always-on voice loop; the two aren't merged yet.
- User mic-level reactivity (visualizing *your* voice, not just JARVIS's
  replies) wasn't built this pass — worth adding later, but out of scope
  for a pure presentation layer with no voice-input wiring yet.

## Not yet built

Nothing from the original milestone list is missing anymore. The web UI
(Phase 1: presentation layer only) is built on top of it, unchanged.
