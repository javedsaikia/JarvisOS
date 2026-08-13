"""JARVIS text conversation loop. Ollama by default; escalates to Claude Code when
the router flags genuine agentic/coding work. Ready to accept Wispr transcripts
via --once for voice-pipeline integration.
"""
import argparse
import json
import sys
from typing import Callable

from jarvis import claude_handoff, ollama_client, router, sarvam_client, tts
from jarvis.config import load_config
from jarvis.memory import MemoryStore
from jarvis.persona import build_system_prompt
from jarvis.tools import calendar, files, notes, parsing, registry, shell, spotify

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

    try:
        model = model_override or cfg["ollama_model"]
        keep_alive = cfg.get("ollama_keep_alive")
        if stream_callback is None:
            return ollama_client.chat(
                messages, model, cfg["ollama_host"], options=ollama_options, keep_alive=keep_alive
            )

        chunks: list[str] = []
        for chunk in ollama_client.chat_stream(
            messages, model, cfg["ollama_host"], options=ollama_options, keep_alive=keep_alive
        ):
            piece = ""
            message = chunk.get("message")
            if isinstance(message, dict):
                piece = message.get("content") or ""
            if not piece:
                piece = chunk.get("response") or ""
            if piece:
                chunks.append(piece)
                stream_callback(piece)
        return "".join(chunks).strip()
    except ollama_client.OllamaError as e:
        return f"(Ollama unavailable: {e})"


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
    except claude_handoff.ClaudeCodeError as e:
        return f"(Claude Code handoff failed: {e})"


def handle_email(user_text: str, cfg: dict) -> str:
    """Escalates to Claude Code, which already has a real, first-party
    Gmail connection — JARVIS's own backend has no mailbox access of its
    own and isn't an MCP host, so this can't be answered locally the way
    calendar/notes reads are. Read-only, so unlike handle_claude_code this
    skips the "hand off to Claude Code?" confirm gate — that prompt is
    worded for agentic/coding work and would be a confusing thing to ask
    before a plain inbox check; calendar/notes reads don't confirm either.
    """
    if not cfg["claude_code_enabled"]:
        return "Claude Code handoff is disabled in config, so I can't check email right now."
    task = (
        "Check the user's Gmail inbox using the connected Gmail tool and answer this "
        f"request in 2-4 short spoken sentences, no markdown, no lists: {user_text}"
    )
    try:
        return claude_handoff.invoke(task, cfg["claude_code_command"])
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
    if action == "play":
        return spotify.play()
    if action == "play_track":
        query = parsing.extract_song_query(user_text)
        return spotify.play_track(query) if query else spotify.play()
    if action == "pause":
        return spotify.pause()
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
                "JARVIS persona. Do not output JSON."
            ),
        },
    ]
    try:
        return ollama_client.chat(
            followup, cfg["ollama_model"], cfg["ollama_host"], keep_alive=cfg.get("ollama_keep_alive")
        )
    except ollama_client.OllamaError as e:
        return f"({call['name']} result: {result}) (Ollama unavailable for final phrasing: {e})"


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
            response = handle_ollama(
                clean_text,
                cfg,
                mem,
                recent_turns_limit=recent_turns_limit,
                model_override=model_override,
                ollama_options=ollama_options,
                stream_callback=stream_callback,
            )

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
    if backend == "sarvam:hi-IN":
        return "[Sarvam Hindi]"
    if backend == "sarvam:as-IN":
        return "[Sarvam Assamese]"
    return "[Ollama]"


def main():
    parser = argparse.ArgumentParser(description="JARVIS text conversation loop")
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

    print(f"JARVIS online. Good day, {cfg['user_name']}. Type /help for commands.\n")
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
