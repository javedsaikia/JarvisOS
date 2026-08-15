"""Orin text conversation loop. Ollama by default; escalates to Claude Code when
the router flags genuine agentic/coding work. Ready to accept Wispr transcripts
via --once for voice-pipeline integration.
"""
import argparse
import json
import re
import subprocess
from typing import Callable

from jarvis import claude_handoff, location_client, maps_client, ollama_client, router, sarvam_client, tts
from jarvis.config import load_config
from jarvis.memory import MemoryStore
from jarvis import llm
from jarvis.persona import build_system_prompt
from jarvis.tools import browser, calendar, calling, files, location, notes, parsing, registry, shell, spotify, vision

HELP_TEXT = """Commands:
  /remember <fact>   Save a durable fact/preference to memory
  /claude <task>      Force this turn to Claude Code
  /ollama <message>   Force this turn to local Ollama
  /mute, /unmute      Toggle spoken replies for this session
  /help               Show this help
  /exit, /quit        Leave

Calendar/Notes/Email are handled automatically (no slash command needed) when
your phrasing matches, e.g. "what's on my calendar today", "schedule a call
with Bob tomorrow at 3pm", "take a note: buy milk", "search my notes for
dentist", "check my email" (email reads hand off to Claude Code, which has
the real Gmail connection — replies take a few seconds longer).

File/shell requests are also automatic, e.g. "read the file README.md",
"list the files in jarvis/tools", "run this command: ls -la" — scoped to the
project directory; writes and shell commands always ask to confirm first.
"""


def text_confirm(prompt: str) -> bool:
    """Default confirmation: ask on the terminal. Swappable — the voice loop
    passes a confirm_fn that speaks the prompt and listens for yes/no instead.
    """
    answer = input(f"[Router] {prompt} [y/N] ")
    return answer.strip().lower() == "y"


def speak_response(text: str, cfg: dict, stop_event=None) -> bool:
    if not cfg.get("tts_enabled"):
        return True
    if cfg.get("tts_backend") != "piper":
        return True
    try:
        return tts.speak(text, cfg["tts_voice_model"], stop_event=stop_event)
    except tts.TTSError as e:
        print(f"(TTS unavailable: {e})")
        return True


def handle_ollama(
    user_text: str,
    cfg: dict,
    mem: MemoryStore,
    recent_turns_limit: int = 12,
    model_override: str | None = None,
    ollama_options: dict | None = None,
    stream_callback: Callable[[str], None] | None = None,
    model_role: str = "chat",
    used_entry: dict | None = None,
) -> str:
    system_prompt = build_system_prompt(cfg["user_name"], mem.load_facts())
    messages = [{"role": "system", "content": system_prompt}]
    # Excludes Sarvam turns — seen live: Ollama's small English-tuned model
    # produced garbled Hindi-ish gibberish (not just imitation, actual
    # nonsense text) after seeing real Devanagari output nearby in context.
    for turn in mem.recent_turns(recent_turns_limit, exclude_backend_prefixes=("sarvam:",)):
        role = "assistant" if turn["role"] == "assistant" else "user"
        messages.append({"role": role, "content": turn["content"]})
    messages.append({"role": "user", "content": user_text})

    # Which model answers is a per-role assignment the user makes in the
    # dashboard (see jarvis/llm.py). model_override still wins, so
    # /ollama and the voice loop's own smaller-model setting behave as
    # they always did.
    entry = llm.resolve(cfg, model_role)
    if model_override:
        entry = {**entry, "provider": "ollama", "model": model_override, "local": True}

    # Reports which model *actually* answered, which is not always the one
    # selected: a cloud model can fail and fall back below. Labelling the
    # reply with the selected model in that case would credit an answer to
    # something that never ran.
    if used_entry is not None:
        used_entry.clear()
        used_entry.update(entry)

    try:
        return llm.chat(
            messages, entry, cfg, options=ollama_options, stream_callback=stream_callback
        )
    except llm.LLMError as e:
        if entry["provider"] == "ollama":
            return f"(Ollama unavailable: {e})"
        # A cloud model that fails should not end the turn — fall back to
        # the local one that is always there, and say so, rather than
        # handing back an error where an answer was expected.
        print(f"({entry['label']} failed: {e} — falling back to the local model.)", flush=True)
        local = llm.resolve({**cfg, "model_roles": {}}, model_role)
        if used_entry is not None:
            used_entry.clear()
            used_entry.update(local)
        try:
            return llm.chat(
                messages, local, cfg, options=ollama_options, stream_callback=stream_callback
            )
        except llm.LLMError as local_error:
            return f"({entry['label']} failed: {e}; local fallback also failed: {local_error})"


def handle_sarvam(
    user_text: str,
    cfg: dict,
    mem: MemoryStore,
    language_code: str,
    recent_turns_limit: int = 12,
) -> str:
    """Sarvam-105B for Hindi/Assamese — the local Ollama model isn't
    reliable here (seen live: hallucinated movie titles/directors, drifted
    into broken Hindi-ish text mid-conversation). No streaming support:
    Sarvam's reasoning-model responses (see sarvam_client.chat) don't
    arrive incrementally the way Ollama's do, so callers get the full
    response at once, same as the tool/Claude Code backends."""
    # Verified live: without an explicit instruction, Sarvam-105B replied
    # in English to a romanized-Hindi message even though this path was
    # explicitly chosen for Hindi/Assamese — the model won't infer the
    # target language just from being routed here.
    language_name = {"hi-IN": "Hindi", "as-IN": "Assamese"}.get(language_code, language_code)
    system_prompt = build_system_prompt(cfg["user_name"], mem.load_facts())
    system_prompt += f"\n\nRespond in {language_name}, regardless of what language the user writes in."
    messages = [{"role": "system", "content": system_prompt}]
    for turn in mem.recent_turns(recent_turns_limit):
        role = "assistant" if turn["role"] == "assistant" else "user"
        messages.append({"role": role, "content": turn["content"]})
    messages.append({"role": "user", "content": user_text})

    try:
        return sarvam_client.chat(messages)
    except sarvam_client.SarvamError as e:
        return f"(Sarvam unavailable: {e})"


def handle_claude_code(user_text: str, cfg: dict, interactive: bool, confirm_fn=text_confirm) -> str:
    if not cfg["claude_code_enabled"]:
        return "Claude Code handoff is disabled in config."

    if interactive and cfg.get("claude_code_confirm", True):
        if not confirm_fn("This looks like agentic/coding work — hand off to Claude Code?"):
            return "(Cancelled — staying local.)"

    try:
        return claude_handoff.invoke(user_text, cfg["claude_code_command"])
    except claude_handoff.Cancelled:
        # Propagate — the voice loop catches this specifically to skip
        # speaking a stale/error response and go straight back to
        # listening, rather than reporting it as a failure.
        raise
    except claude_handoff.ClaudeCodeError as e:
        return f"(Claude Code handoff failed: {e})"


def handle_email(user_text: str, cfg: dict) -> str:
    """Escalates to Claude Code, which already has a real, first-party
    Gmail connection — Orin's own backend has no mailbox access of its
    own and isn't an MCP host, so this can't be answered locally the way
    calendar/notes reads are. Read-only, so unlike handle_claude_code this
    skips the "hand off to Claude Code?" confirm gate — that prompt is
    worded for agentic/coding work and would be a confusing thing to ask
    before a plain inbox check; calendar/notes reads don't confirm either.
    """
    if not cfg["claude_code_enabled"]:
        return "Claude Code handoff is disabled in config, so I can't check email right now."
    task = (
        "Check the user's Gmail inbox using the connected Gmail tool. For each relevant "
        "email, give exactly one short spoken sentence summarizing it (sender or subject "
        "plus the gist) — never quote or reproduce an email's full body text, only a "
        "brief summary line per email. No markdown, no headers, no bullet points, just "
        f"short spoken sentences one after another. Answer this request: {user_text}"
    )
    try:
        return claude_handoff.invoke(task, cfg["claude_code_command"])
    except claude_handoff.Cancelled:
        raise
    except claude_handoff.ClaudeCodeError as e:
        return f"(Email check failed: {e})"


def handle_spotify(user_text: str, cfg: dict) -> str:
    """Escalates to Claude Code, which has a first-party Spotify connection
    with a get_currently_playing tool. Deliberately narrow — search,
    add_to_library, etc. are explicitly widget-only per their own tool
    descriptions and don't work from a headless handoff like this one, so
    router.SPOTIFY_READ_PATTERNS only ever sends "what's playing"-style
    requests here in the first place.
    """
    if not cfg["claude_code_enabled"]:
        return "Claude Code handoff is disabled in config, so I can't check Spotify right now."
    task = (
        "Use the connected Spotify tool to check what's currently playing and answer this "
        f"request in 1-2 short spoken sentences, no markdown, no lists: {user_text}"
    )
    try:
        return claude_handoff.invoke(task, cfg["claude_code_command"])
    except claude_handoff.Cancelled:
        raise
    except claude_handoff.ClaudeCodeError as e:
        return f"(Spotify check failed: {e})"


def handle_spotify_control(action: str, user_text: str) -> str:
    """Local AppleScript playback control (jarvis/tools/spotify.py) — no
    Claude Code handoff. Distinct from handle_spotify's account-wide
    status read above. "play_track" is the only action that needs the raw
    text (parsing.extract_song_query pulls the search query out of it);
    the Web API search it triggers is the only network call in this
    function — everything else is purely local. Any error raised here is
    caught by handle_tool's outer try/except, same as every other domain.
    """
    if action == "open":
        return spotify.open_app()
    if action == "play":
        return spotify.play()
    if action == "play_track":
        query = parsing.extract_song_query(user_text)
        return spotify.play_track(query) if query else spotify.play()
    if action == "pause":
        return spotify.pause()
    if action == "mute":
        return spotify.mute()
    if action == "unmute":
        return spotify.unmute()
    if action == "volume_up":
        return spotify.volume_up()
    if action == "volume_down":
        return spotify.volume_down()
    if action == "next":
        return spotify.next_track()
    if action == "previous":
        return spotify.previous_track()
    return "(Unrecognized Spotify action.)"


def _confirm_write(interactive: bool, cfg: dict, prompt: str, confirm_fn=text_confirm) -> bool:
    if not interactive or not cfg.get("tools_confirm_writes", True):
        return True
    return confirm_fn(prompt)


def execute_file_tool(name: str, arguments: dict, cfg: dict, interactive: bool, confirm_fn=text_confirm) -> str:
    try:
        if name == "read_file":
            return files.read_file(arguments.get("path", ""))
        if name == "list_dir":
            return files.list_dir(arguments.get("path", "."))
        if name == "search_files":
            return files.search_files(arguments.get("query", ""))
        if name == "write_file":
            path = arguments.get("path", "")
            content = arguments.get("content", "")
            if not _confirm_write(interactive, cfg, f'Write to file "{path}"?', confirm_fn):
                return "Cancelled by user — file not written."
            return files.write_file(path, content)
        if name == "run_shell":
            command = arguments.get("command", "")
            # Always confirms, regardless of tools_confirm_writes — shell is
            # categorically higher-risk than a scoped file write ('shell
            # (carefully)' per the mission spec).
            if interactive and not confirm_fn(f"Run shell command: `{command}` ?"):
                return "Cancelled by user — command not run."
            return shell.run(command)
    except shell.Cancelled:
        raise  # voice loop catches this specifically, see handle_claude_code
    except (files.FileToolError, shell.ShellError) as e:
        return f"Tool error: {e}"
    return f"Unknown tool: {name}"


def handle_shell(user_text: str, cfg: dict, interactive: bool, confirm_fn=text_confirm) -> str:
    """Deterministic shell dispatch — NOT LLM-mediated. The command is
    regex-extracted (parsing.extract_shell_command) and confirmation is
    unconditional whenever interactive, independent of tools_confirm_writes
    and independent of any model's willingness to call a tool correctly.
    Returns the real command output directly (no LLM phrasing pass), so
    there's no chance of the response misrepresenting what actually ran.
    """
    command = parsing.extract_shell_command(user_text)
    if not command:
        return "I couldn't tell what command to run — try phrasing it as \"run <command>\"."

    if interactive and not confirm_fn(f"Run this shell command? `{command}`"):
        return "Cancelled — command not run."

    try:
        result = shell.run(command)
    except shell.Cancelled:
        raise
    except shell.ShellError as e:
        return f"Shell error: {e}"
    return f"Ran `{command}`:\n{result}"


def handle_files_shell(user_text: str, cfg: dict, mem: MemoryStore, interactive: bool, confirm_fn=text_confirm) -> str:
    system_prompt = build_system_prompt(cfg["user_name"], mem.load_facts()) + "\n\n" + registry.build_tool_prompt()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]

    try:
        message = ollama_client.chat_message(
            messages,
            cfg["ollama_model"],
            cfg["ollama_host"],
            tools=registry.TOOL_SCHEMAS,
            # Same reasoning as handle_browser: tool SELECTION should be as
            # near deterministic as this model allows. At default sampling
            # this path asked "did you mean filenames or contents?" instead
            # of just searching, and refused a Desktop listing outright.
            options={"temperature": 0.1},
            keep_alive=cfg.get("ollama_keep_alive"),
        )
    except ollama_client.OllamaError as e:
        return f"(Ollama unavailable: {e})"

    call = None
    native_calls = message.get("tool_calls")
    if native_calls:
        fn = native_calls[0]["function"]
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        call = {"name": fn["name"], "arguments": args}
    else:
        call = registry.parse_tool_call(message.get("content", ""))

    if not call:
        return message.get("content", "")

    result = execute_file_tool(call["name"], call["arguments"], cfg, interactive, confirm_fn)

    followup = messages + [
        {"role": "assistant", "content": json.dumps(call)},
        {
            "role": "user",
            "content": (
                f"Tool result for {call['name']}: {result}\n\n"
                "Now answer the original question in plain natural language, in your "
                "Orin persona. Do not output JSON."
            ),
        },
    ]
    try:
        return ollama_client.chat(
            followup, cfg["ollama_model"], cfg["ollama_host"], keep_alive=cfg.get("ollama_keep_alive")
        )
    except ollama_client.OllamaError as e:
        return f"({call['name']} result: {result}) (Ollama unavailable for final phrasing: {e})"


def handle_screen(user_text: str, cfg: dict, interactive: bool = False,
                  confirm_fn=text_confirm) -> str:
    """Read-only — same tier as calendar/notes/email reads, no confirm.
    Deterministic router match (router.SCREEN_PATTERNS), not LLM-mediated,
    since there's nothing to extract from the phrasing.

    Looking at the screen stays unconfirmed because it is local, free and
    read-only. interactive/confirm_fn are passed through only for the
    optional paid cloud fallback, which is off by default and confirms
    before any image leaves the machine (see vision.describe_screen).
    """
    if not cfg.get("vision_enabled", True):
        return "Screen understanding is disabled in config."
    try:
        return vision.describe_screen(user_text, cfg, interactive=interactive,
                                      confirm_fn=confirm_fn)
    except vision.VisionError as e:
        return f"(Screen check failed: {e})"


def execute_browser_tool(name: str, arguments: dict, cfg: dict, interactive: bool, confirm_fn=text_confirm) -> str:
    """Mirrors execute_file_tool. get_page_text/get_current_url run
    immediately (read-only); open_url/click/type_text always confirm
    first, with a specific prompt naming exactly what will happen — the
    literal "I will click the button labeled 'Submit'... Proceed?"
    requirement, not a generic "run this action?".
    """
    try:
        if name == "get_page_text":
            return browser.get_page_text()
        if name == "get_current_url":
            return browser.get_current_url()
        if name == "open_url":
            url = arguments.get("url", "")
            if interactive and not confirm_fn(f"Open this URL in the browser: {url}?"):
                return "Cancelled by user — nothing opened."
            return browser.open_url(url)
        if name == "click":
            target = arguments.get("target", "")
            if interactive and not confirm_fn(f'Click "{target}" on the current page?'):
                return "Cancelled by user — nothing clicked."
            return browser.click(target)
        if name == "type_text":
            target = arguments.get("target", "")
            text = arguments.get("text", "")
            if interactive and not confirm_fn(f'Type "{text}" into "{target}" on the current page?'):
                return "Cancelled by user — nothing typed."
            return browser.type_text(target, text)
    except browser.BrowserError as e:
        return f"Browser error: {e}"
    return f"Unknown tool: {name}"


def handle_browser(user_text: str, cfg: dict, mem: MemoryStore, interactive: bool, confirm_fn=text_confirm) -> str:
    """Mirrors handle_files_shell exactly, but against BROWSER_TOOL_SCHEMAS
    instead of FILE_TOOL_SCHEMAS — a browser-domain turn should only ever
    see browser tools, not file tools (see registry.py's module docstring
    for why the two schema lists are kept separate).
    """
    if not cfg.get("browser_enabled", True):
        return "Browser control is disabled in config."

    system_prompt = (
        build_system_prompt(cfg["user_name"], mem.load_facts())
        + "\n\n"
        + registry.build_tool_prompt(registry.BROWSER_TOOL_SCHEMAS)
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]

    try:
        message = ollama_client.chat_message(
            messages,
            cfg["ollama_model"],
            cfg["ollama_host"],
            tools=registry.BROWSER_TOOL_SCHEMAS,
            # Low temperature specifically for this call — verified live,
            # the default sampling temperature made the model inconsistently
            # skip emitting the tool-call JSON for even a precise, literal
            # instruction ("go to example.com"), instead chatting about the
            # site from background knowledge without navigating. It never
            # hallucinated a false "done" claim (the safe failure mode), but
            # it also didn't do what was asked. Tool SELECTION should be as
            # close to deterministic as this model allows; the final
            # natural-language phrasing pass below stays at default
            # temperature since some variation there is fine.
            options={"temperature": 0.1},
            keep_alive=cfg.get("ollama_keep_alive"),
        )
    except ollama_client.OllamaError as e:
        return f"(Ollama unavailable: {e})"

    call = None
    native_calls = message.get("tool_calls")
    if native_calls:
        fn = native_calls[0]["function"]
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        call = {"name": fn["name"], "arguments": args}
    else:
        call = registry.parse_tool_call(message.get("content", ""), registry.tool_names(registry.BROWSER_TOOL_SCHEMAS))

    if not call:
        return message.get("content", "")

    result = execute_browser_tool(call["name"], call["arguments"], cfg, interactive, confirm_fn)

    followup = messages + [
        {"role": "assistant", "content": json.dumps(call)},
        {
            "role": "user",
            "content": (
                f"Tool result for {call['name']}: {result}\n\n"
                "Now answer the original question in plain natural language, in your "
                "Orin persona. Do not output JSON."
            ),
        },
    ]
    try:
        return ollama_client.chat(
            followup, cfg["ollama_model"], cfg["ollama_host"], keep_alive=cfg.get("ollama_keep_alive")
        )
    except ollama_client.OllamaError as e:
        return f"({call['name']} result: {result}) (Ollama unavailable for final phrasing: {e})"


def handle_weather(cfg: dict) -> str:
    """Read-only informational — same tier as calendar/notes/email reads,
    no confirm. Deterministic router match, nothing to extract from the
    phrasing.
    """
    if not cfg.get("location_enabled", True):
        return "Location features are disabled in config."
    try:
        return location.describe_weather()
    except location_client.LocationError as e:
        return f"(Weather check failed: {e})"


def handle_nearby(user_text: str, cfg: dict) -> str:
    """Read-only informational, no confirm. Category is a plain keyword
    lookup (location.detect_category) over a small fixed vocabulary, not
    LLM extraction — see router.py's NEARBY_PATTERNS comment for why.
    """
    if not cfg.get("location_enabled", True):
        return "Location features are disabled in config."
    category = location.detect_category(user_text)
    if not category:
        known = ", ".join(sorted(location_client.CATEGORY_TAGS))
        return f"I couldn't tell what you're looking for — I can search for: {known}."
    radius = cfg.get("location_default_radius_m", 3000)
    try:
        return location.describe_nearby(category, radius_m=radius)
    except location_client.LocationError as e:
        return f"(Nearby search failed: {e})"


def handle_open_maps(user_text: str, cfg: dict, interactive: bool, confirm_fn=text_confirm) -> str:
    """Confirmed — a real app opens with a real URL. Mirrors
    execute_browser_tool's open_url: the confirmation prompt always names
    the exact thing that will happen, never a generic "run this action?".
    """
    if not cfg.get("location_enabled", True):
        return "Location features are disabled in config."

    # MAPS_PATTERNS is deliberately referential ("open this/that/it in
    # maps", bare "open maps") — the trigger phrase itself is never a real
    # place name. Strip it so it doesn't get sent to Apple Maps as a
    # literal (and nonsensical) search query — seen live: "open this in
    # maps" was searching for the text "open this in maps".
    query = user_text.strip()
    for pattern in router.MAPS_PATTERNS:
        query = re.sub(pattern, "", query, flags=re.IGNORECASE).strip()

    lat = lon = None
    try:
        loc = location_client.get_location()
        lat, lon = loc["lat"], loc["lon"]
    except location_client.LocationError:
        pass  # Maps still works with just the query text, no coordinates.

    url = location.maps_url(query, lat, lon)
    prompt = f'Open Apple Maps for "{query}"?' if query else "Open Apple Maps near your current location?"
    if interactive and not confirm_fn(prompt):
        return "Cancelled by user — Maps not opened."

    try:
        subprocess.run(["open", url], capture_output=True, timeout=10)
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"(Could not open Maps: {e})"
    return f'Opened Apple Maps for "{query}".' if query else "Opened Apple Maps near your current location."


def handle_call(user_text: str, cfg: dict, interactive: bool, confirm_fn=text_confirm) -> str:
    """Deterministic dispatch — NOT LLM-mediated, same reasoning and
    structure as handle_shell above: a real-world action (ringing a real
    phone) must never depend on a model reliably choosing to call a tool.
    The number is regex-extracted (calling.parse_phone_number) and
    confirmation is unconditional whenever interactive, independent of
    tools_confirm_writes.
    """
    if not cfg.get("calling_enabled", True):
        return "Calling is disabled in config."

    number = calling.parse_phone_number(user_text)
    label = None
    detail = ""

    if not number:
        # No digits spoken, so this names a place ("call Hotel Heritage")
        # or refers back to one ("call them"). Resolving a name to a number
        # is a lookup, never a guess — if nothing is found nothing is
        # dialled, and the resolved number is always shown before the call.
        target = calling.extract_call_target(user_text)
        if target is None:
            remembered = location.last_places()
            if len(remembered) == 1:
                target = remembered[0]
            elif remembered:
                options = ", ".join(remembered[:4])
                return f"Which one would you like me to call — {options}?"
            else:
                return "I'm not sure who you mean — say the name or the number."

        try:
            here = location_client.get_location()
            place = maps_client.find_phone(target, here["lat"], here["lon"])
        except (location_client.LocationError, maps_client.MapsError) as e:
            return f"(Couldn't look that up: {e})"

        if not place:
            return (
                f'I couldn\'t find a phone number for "{target}" near you. '
                "Tell me the number and I'll dial it."
            )
        number = calling.normalize_number(place["phone"])
        label = place["name"]
        km = place["distance_m"] / 1000
        detail = f" ({km:.1f}km away)" if km >= 0.1 else " (nearby)"

    prompt = f"Call {label} on {number}{detail}?" if label else f"Call {number}?"
    if interactive and not confirm_fn(prompt):
        return "Cancelled — call not placed."

    try:
        return calling.call_number(number, label)
    except calling.CallingError as e:
        return f"(Call failed: {e})"


def handle_tool(
    domain: str, action: str, user_text: str, cfg: dict, mem: MemoryStore, interactive: bool, confirm_fn=text_confirm
) -> str:
    try:
        if domain == "calendar":
            if action == "read":
                lowered = user_text.lower()
                days = 7 if "week" in lowered else (2 if "tomorrow" in lowered else 1)
                return calendar.list_events(days=days)

            title, start, duration = parsing.extract_event(user_text)
            end = start + duration
            when = start.strftime("%a %b %d, %I:%M %p")
            if not _confirm_write(interactive, cfg, f'Add "{title}" to your calendar for {when}?', confirm_fn):
                return "(Cancelled — nothing added.)"
            return calendar.add_event(title, start, end)

        if domain == "notes":
            if action == "read":
                query = user_text.lower()
                for trigger in ("search my notes for", "search notes for", "find note", "find my note", "read my note", "my notes"):
                    if trigger in query:
                        idx = query.index(trigger) + len(trigger)
                        remainder = user_text[idx:].strip(" :for")
                        if remainder:
                            return notes.search_notes(remainder)
                return notes.list_notes()

            title, body = parsing.extract_note(user_text)
            if not _confirm_write(interactive, cfg, f'Create note "{title}"?', confirm_fn):
                return "(Cancelled — nothing created.)"
            return notes.create_note(title, body)

        if domain == "email":
            return handle_email(user_text, cfg)

        if domain == "spotify":
            if action == "read":
                return handle_spotify(user_text, cfg)
            return handle_spotify_control(action, user_text)

        if domain == "shell":
            return handle_shell(user_text, cfg, interactive, confirm_fn)

        if domain == "files":
            return handle_files_shell(user_text, cfg, mem, interactive, confirm_fn)

        if domain == "screen":
            return handle_screen(user_text, cfg, interactive, confirm_fn)

        if domain == "browser":
            return handle_browser(user_text, cfg, mem, interactive, confirm_fn)

        if domain == "location":
            if action == "weather":
                return handle_weather(cfg)
            if action == "nearby":
                return handle_nearby(user_text, cfg)
            if action == "open_maps":
                return handle_open_maps(user_text, cfg, interactive, confirm_fn)

        if domain == "call":
            return handle_call(user_text, cfg, interactive, confirm_fn)
    except (claude_handoff.Cancelled, shell.Cancelled):
        # Must reach voice_loop.py's own handler around cli.process_turn()
        # uncaught — that's what decides to skip TTS and go straight back
        # to listening instead of speaking a "tool failed" error for a
        # cancellation the user asked for on purpose.
        raise
    except Exception as e:
        return f"({domain.capitalize()} tool failed: {e})"

    return "(Unrecognized tool request.)"


def process_turn(
    user_text: str,
    cfg: dict,
    mem: MemoryStore,
    interactive: bool = True,
    confirm_fn=text_confirm,
    recent_turns_limit: int = 12,
    model_override: str | None = None,
    model_role: str = "chat",
    ollama_options: dict | None = None,
    stream_callback: Callable[[str], None] | None = None,
    source: str = "text",
) -> tuple[str, str]:
    stripped = user_text.strip()

    if stripped.startswith("/remember"):
        fact = stripped[len("/remember"):].strip()
        if fact:
            mem.remember(fact)
            return "ollama", f"Noted, {cfg['user_name']}. I'll remember that."
        return "ollama", "Usage: /remember <fact>"

    # Explicit-only for now — no auto-detection of Hindi/Assamese from
    # plain text, since the router's job is routing *English* phrasing to
    # tools/backends, and language detection is a different problem. A
    # forced prefix keeps this predictable while it's new.
    sarvam_language = None
    for prefix, lang_code in (("/hindi", "hi-IN"), ("/assamese", "as-IN")):
        if stripped.startswith(prefix):
            sarvam_language = lang_code
            break

    forced = stripped.startswith("/claude") or stripped.startswith("/ollama") or sarvam_language is not None
    clean_text = stripped
    for prefix in ("/claude", "/ollama", "/hindi", "/assamese"):
        if clean_text.startswith(prefix):
            clean_text = clean_text[len(prefix):].strip()

    tool_match = None if forced else router.detect_tool(clean_text)

    if tool_match:
        domain, action = tool_match
        backend = f"tools:{domain}"
        response = handle_tool(domain, action, clean_text, cfg, mem, interactive, confirm_fn)
    elif sarvam_language:
        backend = f"sarvam:{sarvam_language}"
        response = handle_sarvam(clean_text, cfg, mem, sarvam_language, recent_turns_limit=recent_turns_limit)
    else:
        backend = router.decide_backend(stripped, cfg["default_backend"])
        if backend == "claude_code":
            response = handle_claude_code(clean_text, cfg, interactive, confirm_fn)
        else:
            # The backend recorded for this turn names the model that
            # actually answered, so the transcript label and the memory
            # log both say Groq/Gemini/Orin rather than always "ollama".
            used: dict = {}
            response = handle_ollama(
                clean_text,
                cfg,
                mem,
                recent_turns_limit=recent_turns_limit,
                model_override=model_override,
                ollama_options=ollama_options,
                stream_callback=stream_callback,
                model_role=model_role,
                used_entry=used,
            )
            if used and not model_override:
                backend = llm.backend_name(used)

    mem.log_turn("user", clean_text, backend, source=source)
    mem.log_turn("assistant", response, backend, source=source)
    return backend, response


def label(backend: str) -> str:
    if backend == "claude_code":
        return "[Claude Code]"
    if backend == "tools:calendar":
        return "[Calendar]"
    if backend == "tools:notes":
        return "[Notes]"
    if backend == "tools:email":
        return "[Email]"
    if backend == "tools:spotify":
        return "[Spotify]"
    if backend == "tools:files":
        return "[Files]"
    if backend == "tools:shell":
        return "[Shell]"
    if backend == "tools:screen":
        return "[Screen]"
    if backend == "tools:browser":
        return "[Browser]"
    if backend == "tools:location":
        return "[Location]"
    if backend == "tools:call":
        return "[Call]"
    if backend == "sarvam:hi-IN":
        return "[Sarvam Hindi]"
    if backend == "sarvam:as-IN":
        return "[Sarvam Assamese]"
    # Cloud models answer as themselves: if a reply came from Groq or
    # Gemini, the transcript says so. Which model is answering is a
    # setting the user changes, so it has to be visible in the reply.
    if backend.startswith("groq:"):
        return "[Groq]"
    if backend.startswith("gemini:"):
        return "[Gemini]"
    # The local model is Orin itself as far as anyone using this is
    # concerned — "Ollama" is the runtime it happens to be served by, and
    # naming the plumbing in the transcript reads like a different
    # assistant answered. Every other label here names the capability, not
    # the vendor behind it.
    return "[Orin]"


def main():
    parser = argparse.ArgumentParser(description="Orin text conversation loop")
    parser.add_argument("--once", help="Process a single transcript (e.g. from Wispr) and exit")
    parser.add_argument("--no-speak", action="store_true", help="Disable TTS for this run, regardless of config")
    args = parser.parse_args()

    cfg = load_config()
    if args.no_speak:
        cfg["tts_enabled"] = False
    mem = MemoryStore()

    if args.once:
        backend, response = process_turn(args.once, cfg, mem, interactive=False)
        print(f"{label(backend)} {response}")
        speak_response(response, cfg)
        return

    print(f"Orin online. Good day, {cfg['user_name']}. Type /help for commands.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_text:
            continue
        if user_text in ("/exit", "/quit"):
            print("Goodbye.")
            break
        if user_text == "/help":
            print(HELP_TEXT)
            continue
        if user_text == "/mute":
            cfg["tts_enabled"] = False
            print("(Muted for this session.)")
            continue
        if user_text == "/unmute":
            cfg["tts_enabled"] = True
            print("(Unmuted.)")
            continue

        backend, response = process_turn(user_text, cfg, mem, interactive=True)
        print(f"{label(backend)} {response}\n")
        speak_response(response, cfg)


if __name__ == "__main__":
    main()
