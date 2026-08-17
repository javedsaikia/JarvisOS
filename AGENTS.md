# Max / JarvisOS

Personal local-first macOS assistant for **Javed**. Short replies. Ask before the next action. Do not lecture.

## Load this project — do not start from scratch

On open, Max is already built. Do **not** `npm install`, `npm run build`, recreate `max/.venv`, or reinstall launch agents unless something is actually missing.

1. `./start_max.sh status` — Caddy, bridge, voice, login agents, HUD build.
2. HUD: http://max.localhost:2015 (static files in `frontend/dist`).
3. If the bridge or Caddy is down: `./deploy/install_local.sh` (skips rebuild when `frontend/dist` exists; pass `--rebuild` only when `frontend/src` changed).
4. Voice needs **this Mac's Terminal** for Input Monitoring / Screen Recording: `./start_max.sh`. Do not put the voice loop in launchd.

Closing the editor or this chat does **not** stop Max. Caddy + bridge are login agents (`com.max.caddy`, `com.max.bridge`) with KeepAlive. Personal state is already on disk and gitignored:

| Path | What persists |
|---|---|
| `max/config.json` | Name, models, roles, phone, flags |
| `max/memory/facts.md` | Hand-written facts (`/remember`) |
| `max/memory/profile.md` | Auto-distilled profile |
| `max/memory/conversation_log.jsonl` | Full chat history |
| `max/memory/session.md` | Where we left off |
| `frontend/dist/` | Built HUD — never delete to "clean" |

## Architecture (one brain, three fronts)

- Brain: `max/cli.py` `process_turn()` — every front end calls this.
- Router: `max/router.py` — zero-cost rules, then tools, then the model, then Claude Code (confirmed).
- Voice: `max/voice_loop.py`. HUD glue: `max/bridge.py` on `127.0.0.1:8765`.
- Tools: Calendar, Notes, Email/GitHub/Drive/YouTube/Cal.com (Composio), Spotify, files, shell, screen, location, calls, news, OMDb. Browser is **off**.

### Teaching on screen

`max/tools/vision.py` identifies the app, explains how to do things in it,
and points at the control. Three helper **processes** draw on the desktop —
they are separate processes because AppKit needs the main thread's run
loop and the voice loop never gives its own back:

| Process | Draws | Talks to Max via |
|---|---|---|
| `max/cursor_overlay.py` | the ring that points at a control | argv, exits on its own |
| `max/instruction_panel.py` | the instruction card (writer: `max/instructions.py`) | `/tmp/max_instruction.json` |
| `max/region_select.py` | the crayon, for circling an area | `/tmp/max_region.json` |

`max/compose.py` writes emails/prompts/letters and pastes them **only**
after a yes. The panel is instructions-only — `instructions.for_screen()`
strips anything conversational; drafts use `raw=True` to bypass it.

### Before you commit

`max/.venv/bin/python3 checkpoints.py` (add `--live` for network/service
checks). It verifies every router destination has a handler, the bridge
and HUD agree on the protocol, the TTS producer cannot hang the voice
loop, and nothing is dead or unreferenced.

## Live roles (see `max/config.json`)

Chat: Groq 70B. Voice / screen: Groq 8B. Memory reflection: local.

## Before closing a coding session

Leave the agents running. Do not wipe `frontend/dist`, `max/.venv`, or `max/memory`. If work is mid-flight, update `max/memory/session.md` so the next load continues from there.
