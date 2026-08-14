# Orin

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

# Always-on voice loop — say "Hey Orin", no typing at all
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

Every reply is labeled with what answered it: `[Orin]` (the local model),
`[Calendar]`, `[Notes]`, `[Files]`, `[Shell]`, or `[Claude Code]`. The local
model is labeled by what it is to you rather than by the runtime serving it —
naming Ollama in the transcript read like a second assistant had replied.

**Project layout:**

| Path | What it is |
|---|---|
| `jarvis/cli.py` | Text loop, router dispatch, confirmation logic |
| `jarvis/voice_loop.py` | Always-on mic loop (wake word → STT → same pipeline → TTS) |
| `jarvis/router.py` | Rule-based backend/tool routing, zero token cost |
| `jarvis/ollama_client.py` | stdlib-only client for local Ollama |
| `jarvis/claude_handoff.py` | Shells out to the `claude` CLI |
| `jarvis/memory.py` | Facts + conversation log persistence |
| `jarvis/persona.py` | Orin system prompt — personality, addresses you by name |
| `jarvis/tts.py` | Piper text-to-speech (synthesize/play split for the web UI — see [Web UI](#web-ui)) |
| `jarvis/tools/` | Calendar, Notes, files, shell, and the file/shell tool-call registry |
| `jarvis/screen_capture.py` | Native screen capture (excludes the Orin window) + on-device OCR |
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

- `user_name` — how Orin addresses you
- `default_backend` — `"ollama"` (keep this — never set to a paid backend as default)
- `ollama_model` / `ollama_host` — local model to use
- `claude_code_enabled` — set `false` to fully disable the escalation path
- `claude_code_command` — the CLI command to invoke (default `"claude"`)
- `claude_code_confirm` — ask before handing off to Claude Code (recommended, since it can take real actions)
- `tools_confirm_writes` — ask before creating a calendar event or note (recommended, since misheard dictation could create the wrong thing)
- `tts_enabled` — speak every reply out loud via Piper (default `true`)
- `tts_backend` — `"piper"` (the only backend implemented)
- `tts_voice_model` — path to the Piper `.onnx` voice model, relative to the project root
- `wake_mode` — `"phrase"` (default; spot any phrase with VAD + Whisper) or `"model"` (an openWakeWord detector). See [Wake phrase](#wake-phrase)
- `wake_phrases` — what to say to wake it (default `["hey orin"]`); the last word is treated as the name and matched allowing for how speech-to-text mangles it
- `wake_stt_model` — empty (default) shares the command model; set a size to load a separate, faster one for spotting
- `voice_wake_word` — only used when `wake_mode` is `"model"`: which openWakeWord pretrained model to load (default `"hey_jarvis"`)
- `voice_stt_model` — faster-whisper model size for transcribing voice commands (default `"small.en"`)
- `voice_context_turns` — how many recent turns the voice loop includes in the prompt; lower is faster
- `voice_ollama_model` — optional smaller model for spoken turns only; defaults to the main `ollama_model`
- `voice_ollama_max_tokens` — cap for voice replies; lower values reduce response latency
- `voice_silence_seconds` — how long the voice loop waits after you stop speaking before it hands off to STT
- `voice_silence_rms_threshold` — mic energy level below which the voice loop considers you done talking; raise it in a noisy room, lower it if it cuts you off early
- `vision_enabled` — set `false` to fully disable screen understanding (default `true`)
- `vision_ollama_model` — local Ollama vision model, used only for screens with almost no text (default `"moondream"`, `ollama pull moondream` first)
- `screen_ocr_enabled` — read the screen with macOS on-device OCR (default `true`; the primary path — see [Screen understanding](#screen-understanding-read-only-never-confirmed))
- `screen_text_model` — local model that answers *specific* questions about screen text (default `"qwen2.5:1.5b"`; a bigger model is more accurate and slower)
- `screen_hud_window_marker` — window title treated as Orin's own and left out of the capture (default `"Orin HUD"`, matching the web UI's `<title>`; it is deliberately two words, since a bare "Orin" is a substring of ordinary window titles like "Monitoring")
- `screen_hide_ui` / `screen_hide_delay_ms` — blank the web UI before a fallback screenshot, and how long to wait first (only used when native capture is unavailable)
- `screen_cloud_fallback` / `screen_cloud_model` — paid cloud vision fallback, off by default; needs `GEMINI_API_KEY` in `jarvis/.env` and still asks before sending anything
- `screen_feed_enabled` — the web UI's live Screen Feed card, which screenshots this Mac every 2s while a tab is open; set `false` to disable it regardless of the UI's own toggle (default `true`)
- `browser_enabled` — set `false` to fully disable browser control (default `true`)
- `location_enabled` — set `false` to fully disable weather/nearby-places/Maps (default `true`)
- `location_default_radius_m` — search radius for nearby-places lookups, in meters (default `3000`)
- `calling_enabled` — set `false` to fully disable the calling feature (default `true`)

## Personality

`jarvis/persona.py` builds the system prompt: calm, competent, slightly
formal British-butler tone with dry wit — classic Orin. Addresses you by
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

## Screen understanding & browser control

Two more tool domains, same "cheapest-capable-first, confirm before
anything changes state" architecture as everything else above. Both can
be fully disabled via config (`vision_enabled` / `browser_enabled`).

**One-time setup** (not needed for anything else in this project):
```bash
jarvis/.venv/bin/pip install pyobjc-framework-Vision   # on-device OCR, ~1MB
ollama pull moondream                          # local vision model, ~1.6GB (text-free screens only)
jarvis/.venv/bin/pip install playwright
jarvis/.venv/bin/playwright install chromium   # ~150-300MB
```
`pyobjc-framework-Quartz` is already a dependency (Maps/CoreLocation use
it), and Vision comes from the same family — together they give native
capture and OCR with nothing else added. Without them the feature still
works, via `screencapture` and the vision model, just less well.

Screenshots require macOS's "Screen Recording" permission — grant it once
to whatever app hosts this process (Terminal, etc.) in
System Settings > Privacy & Security > Screen Recording, then restart the
voice loop/bridge. Orin detects a missing grant *before* capturing (other
apps' window titles are privileged the same way pixels are) and says
exactly what to do, rather than silently describing a blank desktop.

### Screen understanding (read-only, never confirmed)

"What's on my screen?", "describe what I'm looking at", "what do you see?",
"what does the screen say about X?", "read the error on my screen" →
`[Screen]`. Deterministic router match (`router.SCREEN_PATTERNS`), not
LLM-mediated. Same tier as calendar/notes/email reads: automatic, local,
free, no confirmation.

**The capture is native, and Orin is not in it.** The HUD is a web page,
so nothing it can capture is more than its own tab — capture therefore
happens in Python, through macOS's window server
(`jarvis/screen_capture.py`). More importantly, with the HUD open the
honest answer to "what's on my screen" was "a glowing blue orb". Rather
than minimising the window and hoping, the screen is composited from the
window list **with the Orin window left out**
(`CGWindowListCreateImageFromArray`): everything else is captured exactly
as it is, including other windows of the same browser, with no hiding, no
delay and no flicker. Runs in ~0.1s.

**The screen is read with OCR, not guessed at from a thumbnail.** Measured
on a real capture: `moondream` returned *"a computer screen with a black
background and white text"*, because a small vision model downsamples a
Retina screenshot until every label is mush. macOS's Vision framework OCRs
the same image on-device in about a second, for free. So the pipeline is:

| Question | Path | Typical |
|---|---|---|
| "what's on my screen?" | window list + OCR, assembled directly — **no model call at all** | ~2s |
| "what does the screen say about X?" | OCR text + the local text model | ~8s |
| a screen with almost no text | the local vision model on the image | ~3s (warm) |
| local vision fails, cloud fallback on | Gemini — **only after you confirm** | opt-in |

The generic question is answered without inference on purpose. Handed a
desktop's worth of OCR and asked to summarize, the small local models
invented every time — a URL that wasn't on screen, an integration that
doesn't exist. The window server already knows exactly which app is in
front and what else is open, and the OCR already knows which line is drawn
largest, so Orin states those instead: *"You're in Visual Studio Code, on
'JarvisOS'. Safari, Terminal and Notes are also open. The largest text on
screen reads ..."*. Always true, and faster than a model call.

**Permissions.** This needs **Screen Recording**, granted to whatever app
starts Orin (your Terminal, if you use `./start_orin.sh`) in System
Settings > Privacy & Security > Screen Recording. Without it Orin says
so in plain language instead of describing a blank desktop — window titles
are privileged the same way pixels are, so a missing grant is detected
before anything is captured (`screen_capture.has_permission()`).

**Turning it off.** `vision_enabled: false` disables everything that looks
at the screen. Finer control: `screen_ocr_enabled` (OCR path),
`screen_feed_enabled` (the live card in the UI), `screen_cloud_fallback`
(paid cloud vision, off by default and confirmed even when on).

**Hide/restore.** If the native path is unavailable (no pyobjc), Orin
falls back to `screencapture`, which photographs everything including the
HUD. In that case only, it publishes a `screen_capture` phase event; the
bridge relays it and the web UI blanks itself for the shot, then comes
back. A browser page cannot minimise its own window, so blanking is the
most a web front end can do — which is exactly why the native exclusion is
the primary path.

**Limitations.**
- One display: the composite covers the main desktop, not a second monitor.
- Reading order is approximate. OCR returns lines with positions, not
  document structure, so a dense multi-column screen can read jumbled.
- Specific questions inherit the small local model's limits — it will
  occasionally answer confidently about something adjacent to what you
  asked. Point `screen_text_model` at a bigger local model if that matters
  more than latency.
- It reads; it does not click. Anything that acts on the screen would go
  through the existing confirmation gates, and nothing here does that.

### Browser control (Playwright, confirmed before anything happens)

"Click the login button", "fill out the form", "go to example.com",
"navigate to ...", "type my email into the field" → `[Browser]`.
LLM-mediated (same pattern as the file tools above) — click targets,
typed text, and URLs are too free-form for regex extraction. Ollama picks
one of five tools from `jarvis/tools/browser.py`, run against **one
persistent, Orin-controlled Chromium instance** (`jarvis/browser_profile/`,
gitignored — never your actual daily browser), launched lazily on first
use so logins/cookies persist across turns instead of a fresh incognito
window spawning every command.

| Tool | Confirmed? | Notes |
|---|---|---|
| `open_url` | **Yes** | `"Open this URL in the browser: {url}?"` |
| `click` | **Yes** | `"Click \"{target}\" on the current page?"` |
| `type_text` | **Yes** | `"Type \"{text}\" into \"{target}\" on the current page?"` |
| `get_page_text` | No | Read-only |
| `get_current_url` | No | Read-only |

The confirmation prompt always names the exact URL/target/text about to
be acted on — never a generic "run this action?" — so you see precisely
what's about to happen before it does. This goes through the same
`confirm_fn` mechanism as everything else in this project: a real spoken
yes/no over voice, a `y/N` prompt in the terminal, or a Yes/No button in
the web UI.

`click`/`type_text` resolve the plain-language `target` via Playwright
locators (accessible role/name, label, placeholder, text — in that
order), matching exactly one element; zero or multiple matches raise a
clear error rather than guessing which one was meant.

Safety/accuracy notes:
- Verified live: a bare "go to example.com" (no confirmation, no action)
  used to fall through to plain Ollama chat and get a **hallucinated**
  "you're now on the page" reply with nothing actually navigated —
  `router.BROWSER_PATTERNS` was missing a bare "go to \<url\>" case at
  first. Fixed; this is exactly the failure mode confirmation exists to
  prevent, so if a browser phrase ever answers instead of acting, treat
  it as a router gap to fix, not a quirk to route around.
- The tool-selection call in `handle_browser()` runs at low temperature
  (`0.1`) — verified live, default sampling occasionally skipped emitting
  the tool-call JSON for even an unambiguous instruction, answering
  conversationally from background knowledge instead of acting. It never
  fabricated a false "done" confirmation this way (the unsafe failure
  mode), but it also didn't do what was asked; the lower temperature
  made tool-call emission consistent across repeated identical requests.

## Location-aware local search & Calling

Two more tool domains, same "cheapest-capable-first, confirm before
anything changes state" architecture as everything else in this project.
Both fully disableable via config (`location_enabled` / `calling_enabled`).
**No new dependencies** — `jarvis/location_client.py` is stdlib `urllib`
only, same pattern as `spotify_client.py`/`sarvam_client.py`, and calling
is a single `open tel:...` subprocess call.

Unlike the browser tools above, **none of this is LLM-mediated** — place
category (restaurant/hotel/gas station/cafe/pharmacy) is a plain keyword
lookup over a small fixed vocabulary
(`jarvis/tools/location.py:detect_category`), and a phone number is
regex-extracted (`jarvis/tools/calling.py:parse_phone_number`), the same
non-LLM-mediated treatment `router.py` already gives shell commands and
for the same reason: a real-world action (ringing a phone, opening a
real app) can't depend on a model reliably choosing to call a tool.

### Weather & nearby places (read-only, never confirmed)

"What's the weather?", "is it going to rain?", "find nearby restaurants",
"where's the nearest gas station?" → `[Location]`. Same tier as
calendar/notes/email reads: automatic, no confirmation.

- **Location**: prefers `CoreLocationCLI` (real GPS) if it's installed
  (`shutil.which` — this project never installs it for you); otherwise
  falls back to free IP-based geolocation (`ipwho.is`, HTTPS, no key).

  **Strongly recommended**, because the IP fallback resolves to your
  ISP's gateway city — measured ~300km off here, reporting Guwahati for a
  user in Jorhat, which silently poisons weather and nearby-places too:

  ```sh
  brew install --cask corelocationcli
  CoreLocationCLI --format "%latitude|%longitude|%locality"   # triggers the permission prompt
  ```

  Then enable it under **System Settings → Privacy & Security → Location
  Services** (both the global toggle and the CoreLocationCLI entry).
  Until that is granted the binary just errors and Orin stays on IP.
  When it does fall back, spoken answers say "near \<city\>, going by your
  network location" rather than stating the place as fact.
- **Weather**: [Open-Meteo](https://open-meteo.com/) — free, no API key.
- **Business phone numbers** (for "call \<restaurant\>"): Apple Maps via
  MapKit — free, no API key, no account.

  ```sh
  jarvis/.venv/bin/pip install pyobjc-framework-MapKit
  ```

  Used instead of OpenStreetMap because OSM rarely carries phone numbers:
  measured 1 of 32 places within 5km, versus a number for every nearby
  restaurant MapKit returned. Note `MKLocalSearch` treats its region as a
  ranking hint, not a filter — searching a name with no local match
  happily returns one hundreds of km away — so `jarvis/maps_client.py`
  discards anything beyond 25km and every confirmation shows the distance.
- **Nearby places**: OpenStreetMap's Overpass API — free, no API key.
  Tries three known public mirrors in sequence (the main instance
  returned a real 504 under load during development — a known
  characteristic of the free community-run service, not a bug here).
- Categories supported out of the box: restaurants, hotels, gas stations,
  cafes, pharmacies. Add more by adding one line each to
  `location_client.CATEGORY_TAGS` (an OSM tag) and
  `location.py`'s `_CATEGORY_KEYWORDS` (the trigger words).

### Open in Apple Maps (confirmed)

"Open this in Maps", "show that on the map" → `[Location]`, but always
asks first: `Open Apple Maps for "<query>"?` — names the exact search
query before anything opens.

### Calling (confirmed, always)

"Call 555-123-4567", "dial this number" → `[Call]`. Always asks first:
`Call <number>?` — showing the exact parsed number — before ever touching
`tel:`. On confirmation, `open tel:<number>` hands the call to Continuity/
Handoff (a paired iPhone), the same way clicking a `tel:` link in Safari
would. **This phase deliberately does not**: dial automatically without
confirmation, resolve a contact name to a number, or use any paid
telephony API — "call this number" with no digits anywhere in the
utterance gets a clear "I couldn't tell what number to call" rather than
guessing.

## Voice output (Piper)

Every reply is spoken aloud via [Piper](https://github.com/rhasspy/piper), a
local neural TTS engine — no cloud calls, no cost. It's installed in an
isolated project-local virtualenv (`jarvis/.venv`) so it never touches system
Python; delete that directory to fully remove it.

Setup (already done in this repo, for reference / reinstalling elsewhere):

```bash
python3 -m venv jarvis/.venv
jarvis/.venv/bin/pip install piper-tts
# voice model (British male, fits the Orin persona):
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
"Hey Orin", then your request; it transcribes, routes, and speaks the
reply, then goes back to listening.

### Wake phrase

**Say "Hey Orin."** The phrase is whatever you put in config, and nothing
in it is named after another product:

```json
"wake_mode": "phrase",
"wake_phrases": ["hey orin"],
```

This is not a pretrained wake-word model, because there is no pretrained
"hey orin" to use. openWakeWord ships exactly four — `hey_jarvis`,
`alexa`, `hey_mycroft`, `hey_rhasspy` — so a product built on the
pretrained set is permanently woken by somebody else's brand, which is
fine for a hobby and not fine for anything you intend to ship.

Instead (`jarvis/wake_phrase.py`): Silero VAD marks each stretch of
speech, the loop's own Whisper transcribes that stretch, and the phrase is
matched in the text. Nothing to train, nothing downloaded that carries a
name, and the phrase is a config line rather than a model file.

**Why the matching is fuzzy.** Whisper is not a keyword spotter — it
writes what it hears through an English lexicon, so an invented name comes
back as the nearest real word, and a *different* one each time. Measured
across models on one synthesized "Hey Orin": `Oren`, `Orrin`, `Arin`,
`Arryn`, `Orange`, `Auring`, `or in`. So the name is matched by
similarity, with the wilder renderings listed explicitly, and two rules
keep that from firing on ordinary speech:

- the loose spellings are only accepted straight after a lead word ("hey",
  "ok", "hi"), so *"the orange juice is in the fridge"* stays quiet while
  *"hey Orange"* wakes it;
- a bare name at the start of an utterance ("Orin, check my calendar") is
  accepted only for the close spellings.

Verified end to end against synthesized speech — 13/13, including *"I was
reading about the origin of the universe"* and *"play some music on
Spotify"* correctly ignored.

**What it costs.** The detector shares the command Whisper, so there is no
second model and no extra memory. VAD runs per frame as it already did;
transcription runs only when someone actually speaks, at ~0.65s per
utterance. Setting `wake_stt_model` to e.g. `"tiny.en"` loads a separate
faster model (~0.1s) — measured, it misses real wakes often enough that
sharing the bigger model is the better default.

**One real trade-off.** A wake-word model fires *mid-phrase*; this fires
when you stop speaking. In exchange the whole utterance is already
captured, so "hey Orin, what's the weather" needs no second recording —
the same audio is reused, which is quicker overall than waking and then
listening again. Saying just "Hey Orin" still chimes and waits, as before.

**Going back to a trained model** is a config flip, not a code change:

```json
"wake_mode": "model",
"voice_wake_word": "hey_jarvis"
```

and if you ever want a *trained* "hey orin", openWakeWord's custom
training produces a model file that drops straight into that same setting.

- **Wake phrase**: VAD-segmented speech transcribed by the same local
  Whisper and matched against `wake_phrases` (`jarvis/wake_phrase.py`) — see
  [Wake phrase](#wake-phrase) above. `wake_mode: "model"` switches to
  [openWakeWord](https://github.com/dscripka/openWakeWord) instead.
- **Transcription**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (`small.en` by default), run on each stretch of speech rather than
  continuously
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
Orin speaks the question aloud, then listens for a short yes/no reply.
`jarvis/cli.py`'s `process_turn`/`handle_tool`/etc. all take a swappable
`confirm_fn` for exactly this — `text_confirm` (types) for the terminal
loop, `VoiceLoop.voice_confirm` (speaks + listens) here. Same core pipeline,
two front ends.

Setup (already done in this repo, for reference / reinstalling elsewhere):

```bash
jarvis/.venv/bin/pip install sounddevice numpy openwakeword faster-whisper
# faster-whisper downloads its model automatically on first run.
# openwakeword is only needed for wake_mode: "model"; the default phrase
# mode uses the Whisper above and the vendored Silero VAD, nothing else.
```

Tested twice: first end-to-end with synthesized "Hey Orin, ..." phrases
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
- The wake phrase is matched from a transcript, so it fires when you stop
  speaking rather than mid-phrase, and an unusual name will be misheard in
  ways the shipped spelling list may not cover — add what you actually see
  in the log to `NAME_MISHEARINGS` (`jarvis/wake_phrase.py`).
- Barge-in is phrase-gated: say the wake phrase (or press push-to-talk)
  to interrupt a reply. It is deliberately not volume-gated — this machine
  has no echo cancellation, so a loudness threshold cuts Orin off on the
  sound of its own voice.
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
  The label shown is *always* `cli.label()`'s actual output — Orin,
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

Electric-blue holographic HUD modelled on the classic Orin interface
look: a compact glowing core inside a layered SVG reticle (tick ring,
segmented arcs, dashed ring, node dots, corner brackets — each rotating at
a different rate), a large command rail down the left, comms log right,
and a routing-tier icon strip along the bottom.

**Controls live in the left rail**, not in a strip under the input: these
are reached mid-conversation, often from across the room, so they get real
size and a fixed position — a primary voice-loop toggle with its own state
LED, then Wake / Mute / Stop / Screen, then the two routing overrides. The
bottom bar keeps only the text input and Send.

**The core shows what state Orin is in.** `ui.setStatus()` stamps the
state on `<html>`, and the whole HUD answers to it at once: the core
caption (LISTENING / PROCESSING / SPEAKING / LINK LOST), the orb colour,
the level arc, the corner brackets, and the reticle's rotation speed. The
orb itself deforms with a ripple whose rate follows the audio, so a
speaking orb reads as speaking rather than merely brighter.

**Every readout is bound to real data** — the reference material for this
design is a marketing poster with mocked-up panels (lights, temperature,
security), and none of that was reproduced, because we have no such data:

| Element | Source |
|---|---|
| System Status oscilloscope | live `AnalyserNode` time-domain data (real TTS audio) |
| Output Spectrum bars | live `AnalyserNode` frequency data, or measured level-over-time while the voice loop is speaking (the caption says which) |
| Circular waveform around the core | the last ~1.5s of measured amplitude, one bar per frame |
| Audio Out meter + ring level arc | same real amplitude that drives the orb |
| Status dot / label / core caption | real pipeline state (idle/thinking/speaking/error) |
| Comms Log label + backend chip | `cli.label()` output, verbatim |
| Routing-tier strip highlight | lights the tier matching the real backend that answered |
| Session / Messages | real elapsed time and real sent-message count |
| Screen Feed | a real `screencapture` of this Mac, every 2s (see below) |

When nothing is playing, the visualizers render a **flat "no signal"
baseline** rather than idle noise — they never imply audio that isn't
there. The only purely decorative elements are the background grid,
vignette, scanline sweep, and the two scrolling hex columns at the screen
edges; all are `aria-hidden` and none are presented as data.

A "Vision Feed" card used to loop a canned video clip labelled *Live
Clip*. It was the one element on screen pretending to be telemetry, and it
is gone: the card now shows a real screenshot of this Mac.

**The orb moves with spoken replies too.** Voice-loop audio is played by
`afplay` on the Mac's speakers and never reaches the browser, so the
`AnalyserNode` has nothing to read and the orb used to sit still through
every spoken turn — the main way this thing is used. `jarvis/voice_events.py`
measures the RMS envelope of the exact WAV about to be played and
publishes it as JSONL; the bridge tails that file and forwards it, and
`envelope-player.ts` replays it against the browser's clock. The motion is
therefore the real speech, just measured in another process. Gain is set
from measurements of real Piper output (mean 0.46 / p90 0.76, nothing
clipped) rather than picked by eye.

**Screen Feed** is `screencapture` + `sips` (macOS's own tools, same
choice as the screen tool) downscaled to a ~40KB JPEG and pushed every 2s.
It only captures while a browser tab is actually connected *and* the feed
is switched on, the card's button toggles it live, and
`screen_feed_enabled: false` in config disables it outright. Without
Screen Recording permission the card says so and the loop stops rather
than retrying forever.

**Stop stops both halves.** The Stop button used to call `audio.stop()`,
which silences WebAudio playback in the tab and does exactly nothing to a
spoken reply coming out of the speakers. It now also sends
`{"type": "voice_interrupt"}`, which the bridge turns into `SIGUSR2` to
the voice loop — the same interrupt path push-to-talk uses.

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
- User mic-level reactivity (visualizing *your* voice, not just Orin's
  replies) wasn't built this pass — worth adding later, but out of scope
  for a pure presentation layer with no voice-input wiring yet.

## Not yet built

Nothing from the original milestone list is missing anymore. The web UI
(Phase 1: presentation layer only) is built on top of it, unchanged.
