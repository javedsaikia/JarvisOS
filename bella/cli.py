"""Bella text conversation loop. Ollama by default; escalates to Claude Code when
the router flags genuine agentic/coding work. Ready to accept Wispr transcripts
via --once for voice-pipeline integration.
"""
import argparse
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Callable

from bella import bhashini_feedback, claude_handoff, composio_client, compose, location_client, maps_client, news_client, ollama_client, router, sarvam_client, tts
from bella.config import load_config
from bella.memory import MemoryStore
from bella import llm, reflection, wake_phrase
from bella.persona import build_system_prompt, build_voice_prompt
from bella.wit import VOICE, spoken_fail
from bella.tools import browser, calendar, calling, desktop, discovery, email_compose, event_draft, files, location, news, notes, omdb, parsing, registry, reminder, shell, spotify, vision

HELP_TEXT = """Commands:
  /remember <fact>   Save a durable fact/preference to memory
  /claude <task>      Force this turn to Claude Code
  /ollama <message>   Force this turn to local Ollama
  /assamese [msg]     Speak Assamese (Sarvam). Sticks until /english
  /hindi [msg]        Speak Hindi (Sarvam). Sticks until /english
  /english            Switch back to English
  /flag [correction]  Flag the last Bhashini ASR/TTS result (Assamese eval)
  /good               Mark the last Bhashini result as good
  /export-bhashini-feedback  Feedback log row count + export copy path
  /mute, /unmute      Toggle spoken replies for this session
  /help               Show this help
  /exit, /quit        Leave

Calendar/Notes/Email are handled automatically (no slash command needed) when
your phrasing matches, e.g. "what's on my calendar today", "schedule a call
with Bob tomorrow at 3pm", "take a note: buy milk", "search my notes for
dentist", "check my email". Gmail, GitHub, YouTube, Google Drive and Cal.com
go through Composio when those accounts are connected.

File/shell requests are also automatic, e.g. "read the file README.md",
"list the files in bella/tools", "run this command: ls -la" — scoped to the
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


def _misheard_names(cfg: dict) -> tuple[str, ...]:
    """Spellings the speech-to-text produces for Bella's own wake name.

    Comes from the same table the wake detector matches on, so there is
    one list to maintain: anything Bella will answer to is also something
    the model is told never to mistake for the user's name.
    """
    names: list[str] = []
    for phrase in cfg.get("wake_phrases") or ["hey bella", "bella"]:
        name = phrase.strip().split()[-1].lower() if phrase.strip() else ""
        if not name:
            continue
        names.append(name)
        names.extend(wake_phrase.NAME_MISHEARINGS.get(name, []))
    # Two-word renderings ("or in") read as noise in a prompt; the single
    # words are what a model would actually copy.
    return tuple(dict.fromkeys(n for n in names if " " not in n))


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
    # Spoken turns get the compact prompt. The full one is ~2,213 tokens
    # of chat guidance on a path where latency is the whole point, and
    # most of it is explanation the model does not need to behave.
    builder = build_voice_prompt if model_role == "voice" else build_system_prompt
    system_prompt = builder(cfg["user_name"], mem.context_block(), _misheard_names(cfg), ability_sentence(cfg))
    messages = [{"role": "system", "content": system_prompt}]
    # Excludes Sarvam turns — seen live: Ollama's small English-tuned model
    # produced garbled Hindi-ish gibberish (not just imitation, actual
    # nonsense text) after seeing real Devanagari output nearby in context.
    for turn in mem.recent_turns(recent_turns_limit, exclude_backend_prefixes=("sarvam:",)):
        # Seen live: Groq kept a fake "I clicked Pull requests / PR #12"
        # thread going across later turns. Those claims never happened —
        # feeding them back in makes the next reply continue the fiction.
        if turn.get("role") == "assistant" and _ACTION_CLAIM_RE.search(turn.get("content") or ""):
            continue
        role = "assistant" if turn["role"] == "assistant" else "user"
        messages.append({"role": role, "content": turn["content"]})
    messages.append({"role": "user", "content": user_text})

    # Which model answers is a per-role assignment the user makes in the
    # dashboard (see bella/llm.py). model_override still wins, so
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

    options = dict(ollama_options or {})
    if "num_predict" not in options:
        options["num_predict"] = int(cfg.get("chat_max_tokens", 120))

    try:
        return llm.chat(
            messages, entry, cfg, options=options, stream_callback=stream_callback
        )
    except llm.LLMError as e:
        if entry["provider"] == "ollama":
            print(f"[chat] local model failed: {e}", flush=True)
            return spoken_fail()
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
                messages, local, cfg, options=options, stream_callback=stream_callback
            )
        except llm.LLMError as local_error:
            print(f"[chat] fallback failed: {local_error}", flush=True)
            return spoken_fail()


_ASSAMESE_STYLE_PATH = Path(__file__).parent / "memory" / "assamese_style.md"
_ASSAMESE_STYLE_MAX = 800
_ASSAMESE_JOKES_PATH = Path(__file__).parent / "memory" / "assamese_jokes.md"
_last_assamese_joke: str | None = None


def _assamese_style_block() -> str:
    """Voice/Do/Don't only. Example pairs and "How I sound" were treated
    as live facts (seen: 'এতিয়া ব্যস্ত আছোঁ' → 'Javed is in a meeting').
    """
    try:
        text = _ASSAMESE_STYLE_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
    keep: list[str] = []
    wanted = False
    for ln in text.splitlines():
        if ln.startswith("## "):
            heading = ln.strip().lower()
            wanted = heading in ("## voice", "## do", "## don't", "## dont")
            continue
        if not wanted or ln.startswith("#"):
            continue
        keep.append(ln)
    return "\n".join(keep).strip()[:_ASSAMESE_STYLE_MAX]


_AS_GREET_RE = re.compile(
    r"গু\S*\s*ম\S*|শুভ\s*(?:ৰাতিপুৱা|সন্ধিয়া)|নমস্কাৰ|"
    r"good\s*mornin[g']?|good\s*evening|hello",
    re.I,
)
_AS_HOW_ARE_YOU_RE = re.compile(r"কি\s*খবৰ", re.I)
_AS_VAGUE = {
    "এনেকুৱা", "এনেকুৱা।", "উম", "হুম", "অ", "আ", "hmm", "huh",
}
_AS_ACK_RE = re.compile(
    r"^(?:হয়|হৈ|হঁ|হাঁ|yes|yeah|yep|ok|okay|ঠিক|আচ্ছা)"
    r"(?:[\s,।.]+(?:হয়|হৈ|হঁ|হাঁ|yes|yeah|yep|ok|okay|ঠিক|আচ্ছা))*$",
    re.I,
)
# Only the actual lecture, and only as a short reply. Quoted হয় used
# to match any answer that mentioned yes, so a 68-token reply about
# programming was thrown away and replaced with "কওক, কি লাগে?" —
# live, 22 Aug, after "এইখিনি গোটেইটো প্ৰগ্ৰেমিংখিনি…".
_AS_LECTURE_RE = re.compile(
    r"নিলিখিলা|বাৰে বাৰে|হয় বুলি",
)
_AS_LECTURE_MAX_CHARS = 80


def _assamese_is_greeting(text: str) -> bool:
    """True only when the whole line is a greeting.

    Seen live: search() on 'কি খবৰ' ate 'কি খবৰ? মই ভালে, আপুনি ভালে আছে'
    and 'ভাত খালে?' and replied 'শুভ ৰাতিপুৱা' every time.
    """
    t = _assamese_core(text)
    return bool(t) and bool(_AS_GREET_RE.fullmatch(t))


def _assamese_is_how_are_you(text: str) -> bool:
    t = _assamese_core(text)
    return bool(t) and bool(_AS_HOW_ARE_YOU_RE.fullmatch(t))


def _assamese_core(text: str) -> str:
    t = sarvam_client.odia_to_assamese(text or "")
    t = re.sub(r"[\s।.!,?]+", " ", t).strip()
    t = re.sub(
        r"^(?:হে(?:ই)?|হায়)?\s*(?:বেলা|বেল্লা|বেলাহ|bella)\s*",
        "",
        t,
        flags=re.I,
    ).strip()
    t = re.sub(r"\s*বেলা$", "", t, flags=re.I).strip()
    return t


def _assamese_is_vague(text: str) -> bool:
    return _assamese_core(text) in _AS_VAGUE


def _assamese_is_ack(text: str) -> bool:
    t = _assamese_core(text)
    return bool(t) and bool(_AS_ACK_RE.fullmatch(t))


def _assamese_is_lecture(reply: str) -> bool:
    """True for the short 'you didn't write / say yes' loop, not a real answer."""
    t = (reply or "").strip()
    if not t or len(t) > _AS_LECTURE_MAX_CHARS:
        return False
    return bool(_AS_LECTURE_RE.search(t))


def _sarvam_skip_history(content: str) -> bool:
    t = (content or "").strip()
    if not t or t in "।./":
        return True
    if t.startswith("(Sarvam unavailable"):
        return True
    if re.search(r"\b(gud|somefin|ur ai|2 b ur)\b", t, re.I):
        return True
    if re.fullmatch(r"(?:হে(?:ই)?\s*)?(?:বেলা|বেল্লা|বেলাহ|bella)[।.!?]*", t, re.I):
        return True
    if _AS_LECTURE_RE.search(t):
        return True
    if _assamese_is_ack(t) or _assamese_is_vague(t):
        return True
    if re.search(r"\bJaved\b.+(?:মিটিং|ফ্ৰী|ব্যস্ত)", t):
        return True
    return False


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
    # "Can you speak in Hindi?" strips down to "can you?" — Sarvam then
    # answered in English about what "that" is. If nothing real is left,
    # acknowledge in the target language and wait.
    user_text = sarvam_client.odia_to_assamese(user_text or "")
    if not (user_text or "").strip():
        if language_code == "hi-IN":
            return "Haan, Hindi mein hi. Boliye — main sun raha hoon, aur thoda mazak bhi karunga."
        if language_code == "as-IN":
            return "হয়, অসমীয়াতেই। কওক — মই শুনি আছোঁ, আৰু অলপ ধেমালিও কৰিম।"
        return f"Yes — I'll answer in {language_name}, and I'll try to be good company. What do you need?"
    if language_code == "as-IN":
        name = cfg["user_name"]
        # Seen live: "গুড মাৰ্ণিং" became "Javed is in a meeting" because
        # the prompt said "Javedৰ সহায়ক" and the style file had
        # "এতিয়া ব্যস্ত আছোঁ". Talk TO him. Do not invent a calendar.
        if _assamese_is_greeting(user_text):
            return "শুভ ৰাতিপুৱা। কি খবৰ?"
        if _assamese_is_how_are_you(user_text):
            return "ভালেই আছোঁ। তোমাৰ দিনটো কেনে?"
        if _assamese_is_ack(user_text):
            return "কওক।"
        if _assamese_is_vague(user_text):
            return "কওক, কি লাগে?"
        system_prompt = (
            f"তুমি Bella। এতিয়া {name} তোমাৰ লগত কথা পাতিছে — তেওঁ ব্যৱহাৰকাৰী। "
            f"{name}ক তুমি বুলি সম্বোধন কৰা। তেওঁ আন ঠাইত আছে বা মিটিংত আছে বুলি নাভাবিবা, "
            "তেওঁৰ বিষয়ে তৃতীয় পুৰুষত ৰিপ’ৰ্ট নকৰিবা।\n"
            "কেৱল অসমীয়াত (অসমীয়া লিপিত) উত্তৰ দিবা। হিন্দী, বাংলা, বা ইংৰাজী SMS নাই।\n"
            "কেৱল এইবাৰৰ কথাটোৰ উত্তৰ দিবা। পুৰণা টপিক, মিটিং, খোৱা-বোৱা, বা ইমেইল "
            "তেওঁ এতিয়া নুসুধিলে আনিব নালাগে। "
            # Seen live: "মৃণাল তালুকদাৰ কোন হয়" and "মুখ্যমন্ত্ৰী কোন"
            # got "মোৰ ওচৰত তথ্য নাই" — ordinary people-knowledge, not a
            # live lookup. Answer those. Phone numbers still stay off.
            "পৰিচিত মানুহ (মুখ্যমন্ত্ৰী, লেখক, সাংবাদিক, শিল্পী)ৰ সাধাৰণ পৰিচয় দিবা — "
            "'মোৰ ওচৰত তথ্য নাই' নকবা। ৰাজ্য নক'লে মুখ্যমন্ত্ৰী মানে অসমৰ। "
            "চিকিৎসা, পদাৰ্থ বিজ্ঞান, আৰু পৰীক্ষাৰ প্ৰশ্নৰ উত্তৰ দিবা। "
            "ফোন নম্বৰ বা এই মুহূৰ্তৰ খবৰ মিছা নকবা। "
            "স্ক্ৰিনখন দেখা পোৱা নাই বুলি নকবা — স্ক্ৰিনৰ কথা আহিলে "
            "সেই কাম স্ক্ৰিন টুলে কৰে।\n"
            "বুজি নাপালে স্বীকাৰ কৰি আকৌ ক’বলৈ ক’বা। "
            "২–৪ বাক্য। জোক নিজে লিখিব নালাগে।"
        )
        style = _assamese_style_block()
        if style:
            system_prompt += (
                "\n\nকথা কোৱাৰ ঢং (এতিয়াৰ তথ্য নহয়, কেৱল ষ্টাইল):\n" + style
            )
    else:
        system_prompt = build_system_prompt(cfg["user_name"], mem.context_block(), _misheard_names(cfg), ability_sentence(cfg))
        system_prompt += (
            f"\n\nRespond in {language_name}, regardless of what language "
            "the user writes or speaks in. Two to four short sentences. "
            "Answer what they asked. Same dry, funny voice. No lectures. "
            "No English SMS slang."
        )
    messages = [{"role": "system", "content": system_prompt}]
    kept = 0
    history_cap = 4 if language_code == "as-IN" else 6
    for turn in reversed(mem.recent_turns(recent_turns_limit)):
        content = sarvam_client.odia_to_assamese((turn.get("content") or "")).strip()
        if _sarvam_skip_history(content):
            continue
        role = "assistant" if turn["role"] == "assistant" else "user"
        messages.insert(1, {"role": role, "content": content})
        kept += 1
        if kept >= history_cap:
            break
    messages.append({"role": "user", "content": user_text.strip()})

    try:
        reply = sarvam_client.chat(
            messages,
            temperature=0.6,
            max_tokens=120,
            timeout=20,
            reasoning=False,
        )
    except sarvam_client.SarvamError as e:
        print(f"[sarvam] failed: {e}", flush=True)
        if language_code == "as-IN":
            if _assamese_is_greeting(user_text):
                return "হাই। কওক, কি লাগে?"
            return "এই মুহূৰ্তত অসমীয়া চাৰ্ভাৰটো ব্যস্ত। এবাৰ আকৌ কৈ চাবা।"
        return spoken_fail()
    if language_code == "as-IN" and _assamese_is_lecture(reply):
        print(f"[sarvam] dropped short lecture: {reply!r}", flush=True)
        return "কওক, কি লাগে?"
    return reply


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
        print(f"[claude] failed: {e}", flush=True)
        return spoken_fail()


def _body_gist(body: str, limit: int = 90) -> str:
    """First real sentence of an email body — the deterministic fallback
    when the summarizer can't run. Never invents: this is the email's own
    words, just shortened."""
    text = re.sub(r"\s+", " ", body or "").strip()
    if not text:
        return ""
    match = re.search(r"^(.{20,%d}?[.!?])(?:\s|$)" % limit, text)
    gist = match.group(1) if match else text[:limit].rstrip()
    return gist.strip().rstrip(".,;:-")


def _summarize_inbox(rows: list[dict], cfg: dict) -> None:
    """Fill each row's "summary" with one short line about what it says.

    Same contract as handle_news: the model sees ONLY the fetched text and
    is told to summarize it, never to add to it — an inbox summary that
    invents a detail is worse than no summary. One request for the whole
    inbox, not one per email, because this sits directly in front of a
    spoken reply. Every row gets a deterministic fallback first, so a model
    that fails or returns junk degrades to the email's own opening line
    rather than to silence.
    """
    for row in rows:
        row["summary"] = _body_gist(row.get("body") or "")
    if not rows:
        return

    listing = "\n\n".join(
        f"[{i}] SUBJECT: {r.get('subject') or '(no subject)'}\nBODY: {r.get('body') or '(empty)'}"
        for i, r in enumerate(rows, 1)
    )
    prompt = (
        "For each email below, write one short line saying what it is about.\n"
        "- The subject is already read out, so your line must add something "
        "the subject does not say. If the body adds nothing, return \"\" for "
        "that number rather than restating the subject.\n"
        "- Use ONLY what is in that email's text. Do not add facts, do not "
        "guess, do not mention the sender or any email address.\n"
        "- At most 12 words. Plain spoken English, no markdown, no emoji.\n"
        'Return ONLY a JSON object mapping the number to the line, like '
        '{"1": "...", "2": ""}.\n\n' + listing
    )
    try:
        reply = llm.chat(
            [
                {"role": "system", "content": "You summarize emails from supplied text only."},
                {"role": "user", "content": prompt},
            ],
            llm.resolve(cfg, "chat"),
            cfg,
            # Five short lines is not much, but a reasoning model bills
            # its thinking to the same budget — at 220 the JSON came back
            # truncated for a full inbox and every row silently fell back
            # to its raw first sentence.
            options={"num_predict": 500},
        )
    except llm.LLMError as e:
        print(f"[email] inbox summary skipped: {e}", flush=True)
        return

    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end == -1:
        return
    try:
        data = json.loads(reply[start:end + 1])
    except json.JSONDecodeError:
        return
    if not isinstance(data, dict):
        return
    for i, row in enumerate(rows, 1):
        line = data.get(str(i), data.get(i))
        # An explicit "" means "the body adds nothing to the subject" — a
        # real answer, so honour it and say only the subject. The
        # deterministic gist stays only when the model skipped the row.
        if isinstance(line, str):
            row["summary"] = re.sub(r"\s+", " ", line).strip().strip('"').rstrip(".")


def _inbox_line(cfg: dict) -> str:
    try:
        rows = composio_client.fetch_inbox(5)
    except (composio_client.ComposioError, llm.LLMError) as e:
        print(f"[email] inbox failed: {e}", flush=True)
        return spoken_fail()
    email_compose.remember_inbox(rows)
    _summarize_inbox(rows, cfg)
    return composio_client.inbox_speech(rows)


def _save_gmail_draft(draft: dict) -> str:
    try:
        saved = composio_client.create_email_draft(
            draft.get("to") or [],
            draft.get("subject") or "",
            draft.get("body") or "",
        )
    except (composio_client.ComposioError, llm.LLMError) as e:
        print(f"[email] draft failed: {e}", flush=True)
        return spoken_fail()
    email_compose.set_gmail_id(saved.get("id") or "")
    who = ", ".join(draft.get("to") or [])
    subj = draft.get("subject") or "(no subject)"
    return f"Draft in Gmail for {who}, subject {subj}. Not sent. Say send it if that's right."


def handle_email(user_text: str, cfg: dict, action: str = "read",
                 interactive: bool = True, confirm_fn=text_confirm) -> str:
    """Gmail via Composio. Drafts use GMAIL_CREATE_EMAIL_DRAFT.

    Inbox reads are a deterministic list — Groq must not rewrite them.
    A spoken name is resolved with GMAIL_SEARCH_PEOPLE (Sabina Saikia
    live-resolved to sabinafarmin@gmail.com). Nothing is sent until
    they confirm; then GMAIL_SEND_DRAFT.
    """
    if action != "write" and not email_compose.active():
        return _inbox_line(cfg)

    if email_compose.wants_cancel(user_text):
        email_compose.cancel()
        return "Dropped the draft."

    prefix = ""
    if email_compose.wants_inbox(user_text):
        prefix = _inbox_line(cfg) + " "

    if not email_compose.active() or email_compose.is_new_draft(user_text):
        email_compose.start()
    draft = email_compose.apply_text(user_text)

    if email_compose.wants_reply(user_text) and not draft.get("subject"):
        name = draft.get("name") or email_compose.extract_person_name(user_text)
        draft["subject"] = f"Re: {name}" if name else "Re: your email"
        email_compose.apply_text(f"subject {draft['subject']}")
        draft = email_compose.snapshot()

    if email_compose.wants_send(user_text) or (
        email_compose.missing() is None and re.search(r"\bsend\b", user_text, re.I)
    ):
        if email_compose.missing():
            need = email_compose.missing()
            return prefix + f"Not yet — I still need the {need}."
        if interactive and not confirm_fn(
            f'Send to {", ".join(draft["to"])} subject "{draft["subject"]}"?'
        ):
            if not draft.get("gmail_id"):
                return prefix + _save_gmail_draft(draft)
            return prefix + "Held in Gmail drafts. Say send it when you want it gone."
        try:
            if not draft.get("gmail_id"):
                saved = composio_client.create_email_draft(
                    draft["to"], draft["subject"], draft["body"]
                )
                email_compose.set_gmail_id(saved.get("id") or "")
                draft = email_compose.snapshot()
            if draft.get("gmail_id"):
                composio_client.send_gmail_draft(draft["gmail_id"])
            else:
                return prefix + spoken_fail()
        except (composio_client.ComposioError, llm.LLMError) as e:
            print(f"[email] send failed: {e}", flush=True)
            return prefix + spoken_fail()
        finally:
            email_compose.cancel()
        return prefix + f'Sent to {", ".join(draft["to"])}.'

    need = email_compose.missing()
    if need == "to":
        named = draft.get("name")
        if named:
            return prefix + f"I don't have an address for {named}. Spell the email."
        return prefix + "Who is it to? Name them or give the address."
    if need == "subject":
        who = f' to {", ".join(draft["to"])}' if draft["to"] else ""
        return prefix + f"Subject{who}?"
    if need == "body":
        return prefix + f'Subject is "{draft["subject"]}". What should it say?'
    return prefix + _save_gmail_draft(draft)


def handle_composio(user_text: str, cfg: dict, domain: str = "github", action: str = "read") -> str:
    """Any authorized Composio toolkit — Gmail, GitHub, YouTube, Drive, Cal."""
    if not composio_client.configured():
        return "Composio is not configured — add COMPOSIO_API_KEY to bella/.env."
    try:
        return composio_client.run(user_text, cfg, domain=domain, action=action)
    except (composio_client.ComposioError, llm.LLMError) as e:
        print(f"[composio] failed: {e}", flush=True)
        return spoken_fail()


def handle_voice_volume(action: str) -> str:
    """Bella's spoken-reply gain, not Spotify and not the Mac output volume."""
    before = tts.get_volume()
    if action == "volume_down":
        level = tts.adjust_volume(-tts.VOLUME_STEP)
        if level == before:
            return "That's as quiet as I go."
        return f"Speaking at {level} percent."
    if action == "volume_up":
        level = tts.adjust_volume(tts.VOLUME_STEP)
        if level == before:
            return "That's as loud as I go."
        return f"Speaking at {level} percent."
    if action == "ping":
        return "Yes. I'm here."
    return "(Unrecognized voice action.)"


def handle_spotify(user_text: str, cfg: dict) -> str:
    """What's playing on THIS Mac's Spotify.app — local AppleScript, no cloud."""
    state = spotify.now_playing_state()
    if not state.get("active"):
        return "Nothing is playing on this Mac."
    verb = "Playing" if state.get("playing") else "Paused on"
    track = state.get("track") or "a track"
    artist = state.get("artist") or "an unknown artist"
    return f"{verb} {track} by {artist}."


def handle_spotify_control(action: str, user_text: str) -> str:
    """Local AppleScript playback control (bella/tools/spotify.py).
    Distinct from handle_spotify's local now-playing read. "play_track"
    is the only action that needs the raw text
    (parsing.extract_song_query pulls the search query out of it);
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
        if parsing.is_referential_song_query(query):
            query = spotify.last_query() or ""
        if query and not parsing.has_song_trigger(user_text) and not parsing.split_song_artist(query)[1]:
            # Play-track routed without a play verb — refuse travel/email
            # leftovers that Spotify would happily search.
            if re.search(
                r"\b(i want to go|by a (car|taxi|cab|bus|train)|e-?mail|draft|volume)\b",
                query,
                re.I,
            ):
                # Leftovers from an unrelated request, not a title. Ask
                # like a person rather than reciting the syntax.
                return "That didn't sound like a song. What do you want on?"
        if query:
            return spotify.play_track(query)
        # Do not fall through to generic play() — that resumes whatever
        # happens to be paused, which is how "play it" started a random
        # track. Ask for a name instead.
        return "What do you want me to play?"
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
    system_prompt = build_system_prompt(cfg["user_name"], mem.context_block(), _misheard_names(cfg), ability_sentence(cfg)) + "\n\n" + registry.build_tool_prompt()
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
        print(f"[files] model failed: {e}", flush=True)
        return spoken_fail()

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
                "Bella persona. Do not output JSON."
            ),
        },
    ]
    try:
        return ollama_client.chat(
            followup, cfg["ollama_model"], cfg["ollama_host"], keep_alive=cfg.get("ollama_keep_alive")
        )
    except ollama_client.OllamaError as e:
        print(f"[files] phrasing failed: {e}", flush=True)
        return result or spoken_fail()


# What Bella can actually do, and the plainest phrasing that triggers each.
# Kept next to the router rather than in the persona because it is a fact
# about the code, not a matter of tone — and because a model asked the
# same question twice should not give two different answers.
CAPABILITIES = [
    ("files", "read and search files on this Mac", "find my notes about X"),
    ("shell", "run shell commands", "run ls in the terminal"),
    ("calendar", "read and add calendar events", "what's on my calendar tomorrow"),
    ("notes", "read and write Notes", "make a note about X"),
    ("email", "check and send Gmail", "check my email"),
    ("github", "GitHub issues, repos and PRs", "what's on my GitHub"),
    ("youtube", "YouTube", "what's on my YouTube"),
    ("googledrive", "Google Drive", "find a file on Google Drive"),
    ("cal", "Cal.com bookings", "what's on cal.com"),
    ("googlecalendar", "Google Calendar", "what's on my Google Calendar"),
    ("googletasks", "Google Tasks", "check my task list"),
    ("linkedin", "LinkedIn", "what's on my LinkedIn"),
    ("reddit", "Reddit", "what's top on r/technology"),
    ("chatfai", "ChatFAI fictional characters", "search ChatFAI for Sherlock Holmes"),
    ("spotify", "control Spotify on this Mac", "play some music"),
    ("vision", "see what app you're in, teach you how to do things in it, and point at the button", "what's on my screen"),
    ("desktop", "click and type in an app that is already open", "click the Save button in Notes"),
    ("browser", "control a browser", "open example.com"),
    ("location", "say the town you're in, weather, and nearby places", "where am I / find restaurants near me"),
    ("calling", "place a phone call", "call this number"),
    ("news", "today's AI, tech, politics, finance or world news", "what's the latest AI news"),
    ("omdb", "movie and series facts from OMDb", "tell me about the movie Inception"),
    ("discovery", "look up movie showtimes, hotel options, and flight listings (no booking)", "what's showing tonight / hotels in Goa / flights from Guwahati to Delhi"),
]

# Which config flag, if any, switches each one off.
_CAPABILITY_FLAGS = {
    "vision": "vision_enabled",
    "desktop": "desktop_enabled",
    "browser": "browser_enabled",
    "location": "location_enabled",
    "calling": "calling_enabled",
    "news": "news_enabled",
    "omdb": "omdb_enabled",
    "discovery": "discovery_enabled",
}


def ability_sentence(cfg: dict) -> str:
    """The tool list as a phrase for the system prompt, config flags applied.

    Same source as handle_capabilities below, so what the model is told it
    can do and what it says it can do cannot drift apart — and neither can
    claim a capability that is switched off.
    """
    available = [
        what for key, what, _ in CAPABILITIES
        if cfg.get(_CAPABILITY_FLAGS.get(key, ""), True)
    ]
    if not available:
        return "nothing right now — every tool is switched off in config"
    if len(available) == 1:
        return available[0]
    return ", ".join(available[:-1]) + f", and {available[-1]}"


def handle_model_question(cfg: dict) -> str:
    """Which model is answering — read from the live role assignments."""
    roles = {r: llm.resolve(cfg, r) for r in llm.ROLES}
    chat = roles["chat"]
    others = ", ".join(
        f"{role} on {entry['label']}" for role, entry in roles.items() if role != "chat"
    )
    where = "on this Mac, nothing leaves the machine" if chat["local"] else "in the cloud"
    return (
        f"Right now I'm answering you with {chat['label']}, running {where}. "
        f"The rest: {others}. Change any of them in Front on the HUD if you tire of this one."
    )


def handle_capabilities(cfg: dict) -> str:
    """Answer "what can you do?" from the tool list, with no model call."""
    available = [
        (what, example) for key, what, example in CAPABILITIES
        if cfg.get(_CAPABILITY_FLAGS.get(key, ""), True)
    ]
    # Written to be *spoken*: this answer goes through TTS, and a
    # ten-bullet list read aloud takes half a minute. One sentence naming
    # what exists, then an offer of detail.
    if not available:
        return "Every tool is switched off in config at the moment, so I can only talk."
    what = [w for w, _ in available]
    listed = ", ".join(what[:-1]) + f", and {what[-1]}" if len(what) > 1 else what[0]
    return f"I can {listed}. Say what you need."


def handle_reminder(user_text: str, cfg: dict, interactive: bool, confirm_fn=text_confirm) -> str:
    """Set a reminder, asking for whatever was not said.

    "Can you set up a reminder?" names neither a subject nor a time, and
    the only honest reply is a question. Measured live, with no route at
    all this fell through to the chat model, which announced "I've added a
    reminder for you on the Google Calendar" — nothing existed. Slots are
    filled over as many turns as it takes (bella/tools/reminder.py); nothing
    is written until both are known and confirmed.
    """
    if not reminder.active() or reminder.is_trigger(user_text):
        reminder.start()
    draft = reminder.apply_text(user_text)

    need = reminder.missing()
    if need == "subject":
        reminder.note_asked("subject")
        return "What should I remind you about?"
    if need == "when":
        reminder.note_asked("when")
        return f"When do you want reminding about {draft['subject']}?"

    where = reminder.destination()
    when_text = reminder.spoken_when()
    place = "Google Calendar" if where == "calendar" else "Google Tasks"
    if interactive and not confirm_fn(
        f'Add "{draft["subject"]}" to {place} for {when_text}?'
    ):
        reminder.cancel()
        return "Dropped it."

    try:
        composio_client.create_reminder(draft["subject"], draft["when"], draft["has_time"])
    except (composio_client.ComposioError, llm.LLMError) as e:
        print(f"[reminder] failed: {e}", flush=True)
        reminder.cancel()
        return spoken_fail()
    reminder.cancel()
    return f"Done — {draft['subject']}, {when_text}, in {place}."


def handle_compose(user_text: str, cfg: dict, interactive: bool, confirm_fn=text_confirm) -> str:
    """Write something, show it, and type it in only if they say yes.

    The confirmation is unconditional whenever interactive — not gated on
    tools_confirm_writes, same as the shell tool and for the same reason.
    Pasting into a document the user is in the middle of writing is a
    real-world change to their work, and an undo does not fully take it
    back. Drafting is free and reversible; typing is neither, so they are
    two separate steps with an explicit yes in between.
    """
    kind = compose.detect(user_text)
    if not kind and compose.waiting_for_subject():
        kind = compose.take_waiting_kind()
        user_text = f"write a {kind}: {user_text}"
    if not kind:
        return "Tell me what to write — an email, a prompt, a letter, some notes."

    if compose.detect(user_text) and not compose.has_subject(user_text):
        compose.wait_for_subject(kind)
        name = {"prompt": "prompt", "email": "email", "letter": "letter"}.get(kind, "draft")
        return f"What should the {name} say?"

    compose.clear_waiting_subject()

    # What is on screen is the subject matter for "write a prompt to review
    # this code". Best-effort: if the screen can't be read, the request
    # still gets written from its own words.
    screen_text, app = "", ""
    if cfg.get("vision_enabled", True):
        try:
            screen_text, app = vision.screen_context()
        except Exception as e:
            print(f"[compose] no screen context: {type(e).__name__}: {e}", flush=True)

    try:
        text = compose.draft(user_text, kind, cfg, screen_text=screen_text, app=app)
    except compose.ComposeError as e:
        print(f"[compose] draft failed: {e}", flush=True)
        return spoken_fail()
    if not text:
        return spoken_fail()

    compose.show_draft(text, kind, app=app)
    compose.remember(text, kind, app)
    summary = compose.spoken_summary(text, kind)

    where = f" into {app}" if app else ""
    if interactive and not confirm_fn(f"Type this draft{where}?"):
        # Kept, not discarded: "no" here usually means "not yet" — they
        # want to read it, or put the cursor somewhere first. See
        # handle_compose_insert.
        return f"{summary} It's on screen — say type it in when you're ready."
    if not interactive:
        # No one to ask (a --once run or a scripted turn): never type.
        return f"{summary} It's on screen; say type it in when you want it."
    return _insert_draft(text, summary)


def _insert_draft(text: str, summary: str) -> str:
    try:
        compose.insert(text)
    except Exception as e:
        print(f"[compose] insert failed: {type(e).__name__}: {e}", flush=True)
        return f"{summary} I couldn't type it in — it's on screen to copy."
    compose.clear_pending()
    return f"{summary} Typed it in.".strip()


def handle_compose_insert(cfg: dict) -> str:
    """"Type it in" for a draft that is already written and on screen.

    The instruction itself is the consent, so this does not ask again —
    asking twice for the same yes is how a confirmation stops meaning
    anything.
    """
    draft = compose.pending()
    if not draft:
        return "I don't have a draft waiting. Tell me what to write."
    return _insert_draft(draft["text"], "")


def handle_screen_region(user_text: str, cfg: dict, interactive: bool = False,
                         confirm_fn=text_confirm) -> str:
    """Circle an area, then answer about only that."""
    if not cfg.get("vision_enabled", True):
        return "Screen understanding is disabled in config."
    try:
        return vision.describe_region(user_text, cfg, interactive=interactive,
                                      confirm_fn=confirm_fn)
    except vision.VisionError as e:
        print(f"[screen] region failed: {e}", flush=True)
        return spoken_fail("screen")


def handle_screen_howto(user_text: str, cfg: dict) -> str:
    """"How do I bold this?" — teach the task in the app already on screen."""
    if not cfg.get("vision_enabled", True):
        return "Screen understanding is disabled in config."
    try:
        return vision.explain_howto(user_text, cfg)
    except vision.VisionError as e:
        print(f"[screen] howto failed: {e} — pointing from a fresh look", flush=True)
        try:
            return vision.describe_screen(user_text, cfg)
        except vision.VisionError:
            return spoken_fail("screen")


def handle_screen_show(cfg: dict) -> str:
    """"Yes, show me" — put Bella's pointer on the control just offered."""
    if not cfg.get("vision_enabled", True):
        return "Screen understanding is disabled in config."
    try:
        return vision.show_pending(cfg)
    except vision.VisionError as e:
        print(f"[screen] show failed: {e}", flush=True)
        return spoken_fail("screen")


def handle_screen_answer(user_text: str, cfg: dict) -> str:
    """Read the exam question on screen and say the answer."""
    if not cfg.get("vision_enabled", True):
        return "Screen understanding is disabled in config."
    try:
        return vision.answer_on_screen(user_text, cfg)
    except vision.VisionError as e:
        print(f"[screen] answer failed: {e}", flush=True)
        return _screen_failure_reply(e)


def handle_screen_advise(user_text: str, cfg: dict) -> str:
    """"What should my next move be?" — an opinion about what is on screen.

    Separate from handle_screen_howto because the question is different in
    kind. "How do I bold this" is world knowledge about an application;
    "what should I play here" is a judgement about THIS board, and can
    only be answered by looking at it.
    """
    if not cfg.get("vision_enabled", True):
        return "Screen understanding is disabled in config."
    try:
        return vision.advise_screen(user_text, cfg)
    except vision.VisionError as e:
        print(f"[screen] advise failed: {e}", flush=True)
        return _screen_failure_reply(e)


def _screen_failure_reply(e: Exception) -> str:
    """Say what actually went wrong, when it is something the user can act on.

    "I couldn't read the screen just now. Try again." was said after a
    rate-limit error that reported exactly how long to wait, and after a
    missing-key error that no amount of trying again would fix. A generic
    apology for a specific, fixable problem wastes the user's next turn.
    """
    text = str(e)
    if "vision model" in text:
        return ("I can't look at the screen — there's no vision model "
                "available. Set GEMINI_API_KEY in bella/.env, or run "
                "ollama pull moondream.")
    if "429" in text or "quota" in text.lower():
        return ("I've hit Gemini's rate limit — it's twenty requests a "
                "minute on the free tier. Give it a moment and ask again.")
    if "API key not valid" in text or "401" in text:
        return "Gemini isn't accepting the key in bella/.env. It needs replacing."
    if "timed out" in text:
        return "Gemini didn't answer in time. Ask me again."
    return spoken_fail("screen")


def _screen_reply_in_session_language(
    reply: str, user_text: str, cfg: dict, mem: MemoryStore | None,
) -> str:
    """Screen answers are English from Groq. If they asked in Assamese,
    say the same facts in Assamese. Seen live: 'মোৰ স্ক্ৰিনত কি দেখিছা
    তুমি?' never reached the screen tool; once it does, English is
    still the wrong mouth.
    """
    if not (reply or "").strip():
        return reply
    as_in = bool(mem and get_conversation_language(mem) == "as-IN")
    if not as_in and not _ASSAMESE_SCRIPT_RE.search(user_text or ""):
        return reply
    if len(re.findall(r"[\u0980-\u09FF]", reply or "")) >= 12:
        return reply
    messages = [
        {
            "role": "system",
            "content": (
                "তুমি Bella। তলৰ কথাখিনি কেৱল অসমীয়া লিপিত ১–২ চুটি বাক্যত কোৱা। "
                "নতুন তথ্য, নতুন অপচন, বা নতুন নম্বৰ নিদিবা। "
                "ৱেবছাইটৰ নাম ইংৰাজীতে ৰাখিবা, আখৰে আখৰে নকবা। "
                "যদি ইতিমধ্যে সুধিছে কি কৰিব, আকৌ নুসুধিবা।"
            ),
        },
        {"role": "user", "content": reply},
    ]
    try:
        out = sarvam_client.chat(
            messages, temperature=0.2, max_tokens=80, timeout=12, reasoning=False,
        )
    except sarvam_client.SarvamError as e:
        print(f"[screen] Assamese wrap failed: {e}", flush=True)
        return reply
    out = (out or "").strip()
    return out or reply


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
        print(f"[screen] failed: {e}", flush=True)
        return _screen_failure_reply(e)


def execute_desktop_tool(name: str, arguments: dict, cfg: dict, interactive: bool, confirm_fn=text_confirm) -> str:
    """Mirrors execute_browser_tool. see_app/find_control run immediately;
    click/type_text/press_key always confirm first with a prompt that names
    the app and the control.
    """
    app = arguments.get("app") or ""
    try:
        if name == "see_app":
            return desktop.see_app(app)
        if name == "find_control":
            return desktop.find_control(app, arguments.get("target", ""))
        if name == "click":
            target = arguments.get("target", "")
            where = app or "the front app"
            if interactive and not confirm_fn(f'Click "{target}" in {where}?'):
                return "Cancelled by user — nothing clicked."
            return desktop.click(app, target)
        if name == "type_text":
            target = arguments.get("target", "")
            text = arguments.get("text", "")
            where = app or "the front app"
            into = f' into "{target}"' if target else ""
            if interactive and not confirm_fn(f'Type{into} in {where}?'):
                return "Cancelled by user — nothing typed."
            return desktop.type_text(app, text, target)
        if name == "press_key":
            key = arguments.get("key", "")
            where = app or "the front app"
            if interactive and not confirm_fn(f"Press {key} in {where}?"):
                return "Cancelled by user — nothing pressed."
            return desktop.press_key(app, key)
    except desktop.DesktopError as e:
        return str(e)
    return f"Unknown tool: {name}"


def handle_desktop(user_text: str, cfg: dict, mem: MemoryStore, interactive: bool, confirm_fn=text_confirm) -> str:
    """LLM-mediated native-app control. Off until desktop_enabled is on,
    same gate as browser_enabled. While off, keep the old teaching line so
    a click request is not answered with a config-flag error.
    """
    if not cfg.get("desktop_enabled", False):
        return "I can point at it, but I don't click. Want me to show you where it is?"

    try:
        info = desktop.doctor()
    except desktop.DesktopError as e:
        return str(e)
    perms = info.get("permissions") or {}
    if perms.get("accessibility") is False or perms.get("post_events") is False:
        return "This Mac has not given macos-harness the permissions it needs."

    system_prompt = (
        build_system_prompt(cfg["user_name"], mem.context_block(), _misheard_names(cfg), ability_sentence(cfg))
        + "\n\n"
        + registry.build_tool_prompt(registry.DESKTOP_TOOL_SCHEMAS)
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]

    try:
        message = ollama_client.chat_message(
            messages,
            cfg["ollama_model"],
            cfg["ollama_host"],
            tools=registry.DESKTOP_TOOL_SCHEMAS,
            options={"temperature": 0.1},
            keep_alive=cfg.get("ollama_keep_alive"),
        )
    except ollama_client.OllamaError as e:
        print(f"[desktop] model failed: {e}", flush=True)
        return spoken_fail()

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
        call = registry.parse_tool_call(
            message.get("content", ""),
            registry.tool_names(registry.DESKTOP_TOOL_SCHEMAS),
        )

    if not call:
        return message.get("content", "")

    result = execute_desktop_tool(call["name"], call["arguments"], cfg, interactive, confirm_fn)

    followup = messages + [
        {"role": "assistant", "content": json.dumps(call)},
        {
            "role": "user",
            "content": (
                f"Tool result for {call['name']}: {result}\n\n"
                "Now answer the original question in plain natural language, in your "
                "Bella persona. Do not output JSON."
            ),
        },
    ]
    try:
        return ollama_client.chat(
            followup, cfg["ollama_model"], cfg["ollama_host"], keep_alive=cfg.get("ollama_keep_alive")
        )
    except ollama_client.OllamaError as e:
        print(f"[desktop] phrasing failed: {e}", flush=True)
        return result or spoken_fail()


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
        build_system_prompt(cfg["user_name"], mem.context_block(), _misheard_names(cfg), ability_sentence(cfg))
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
        print(f"[browser] model failed: {e}", flush=True)
        return spoken_fail()

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
                "Bella persona. Do not output JSON."
            ),
        },
    ]
    try:
        return ollama_client.chat(
            followup, cfg["ollama_model"], cfg["ollama_host"], keep_alive=cfg.get("ollama_keep_alive")
        )
    except ollama_client.OllamaError as e:
        print(f"[browser] phrasing failed: {e}", flush=True)
        return result or spoken_fail()


def _load_assamese_jokes() -> list[str]:
    try:
        raw = _ASSAMESE_JOKES_PATH.read_text(encoding="utf-8")
    except OSError:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    for ln in raw.splitlines():
        if ln.startswith("#"):
            continue
        if ln.strip() == "---":
            text = "\n".join(buf).strip()
            if text:
                chunks.append(text)
            buf = []
            continue
        buf.append(ln)
    tail = "\n".join(buf).strip()
    if tail:
        chunks.append(tail)
    return chunks


def _pick_assamese_joke() -> str | None:
    global _last_assamese_joke
    jokes = _load_assamese_jokes()
    if not jokes:
        return None
    if len(jokes) == 1:
        _last_assamese_joke = jokes[0]
        return jokes[0]
    choices = [j for j in jokes if j != _last_assamese_joke] or jokes
    picked = random.choice(choices)
    _last_assamese_joke = picked
    return picked


def _wants_assamese_joke(user_text: str, mem: MemoryStore | None) -> bool:
    if mem and get_conversation_language(mem) == "as-IN":
        return True
    if re.search(r"জোক|ধেমালি", user_text or ""):
        return True
    return False


def handle_joke(user_text: str, cfg: dict, mem: MemoryStore | None = None) -> str:
    """One joke per ask. Assamese comes from the local pool, not the model."""
    if _wants_assamese_joke(user_text, mem):
        joke = _pick_assamese_joke()
        if joke:
            return joke
        return "এতিয়া হাতত জোক নাই। bella/memory/assamese_jokes.md ত এটা যোগ কৰা।"
    adult = bool(re.search(r"\b(adult|dirty|sexy|filthy|rude|nsfw)\b", user_text or "", re.I))
    system = (
        "You are Bella. The user is an adult named Javed. "
        "Tell one joke now, in clear English. "
        + (
            "They asked for an adult/dirty/sexy joke: make it actually dirty, "
            "not a schoolyard dodge, not a refusal, not a lecture. "
            if adult
            else "A proper joke with a punchline. Dry delivery. "
        )
        + "Two to six short spoken sentences. Then stop. "
        "Never involve anyone under 18. Do not use SMS shorthand."
    )
    entry = llm.resolve(cfg, "chat")
    try:
        return llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            entry,
            cfg,
            options={"num_predict": 180},
        )
    except llm.LLMError as e:
        print(f"[joke] failed: {e}", flush=True)
        return spoken_fail()


def handle_omdb(user_text: str, cfg: dict) -> str:
    if not cfg.get("omdb_enabled", True):
        return "Movie lookup is disabled in config."
    try:
        return omdb.describe(user_text)
    except Exception as e:
        print(f"[omdb] failed: {type(e).__name__}: {e}", flush=True)
        return spoken_fail()


def handle_news(user_text: str, cfg: dict) -> str:
    """Today's headlines from public RSS, then a spoken summary.

    The model only sees fetched items. If it cannot run, we speak titles.
    """
    if not cfg.get("news_enabled", True):
        return "News briefings are disabled in config."
    topics = news.detect_topics(user_text)
    try:
        block, items = news.briefing_block(topics)
    except news_client.NewsError as e:
        print(f"[news] failed: {e}", flush=True)
        return spoken_fail("news")
    if not items:
        return "I couldn't find any headlines just now."

    labels = ", ".join(news.TOPICS[t][0] for t in topics if t in news.TOPICS) or "the news"
    entry = llm.resolve(cfg, "chat")
    prompt = (
        f"Summarize ONLY the headlines below as a spoken briefing on {labels} for today. "
        "Two short sentences. The gist only. Do not add anything that "
        "is not in the list. No markdown, no URLs, no bullet points.\n\n"
        f"{block}"
    )
    try:
        return llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are Bella. Brief the user from supplied headlines only. "
                        + VOICE
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            entry,
            cfg,
            options={"num_predict": 100},
        )
    except llm.LLMError:
        return news.speak_titles(topics, items)


def handle_here(cfg: dict) -> str:
    """Read-only — live GPS town name, never a model guess."""
    if not cfg.get("location_enabled", True):
        return "Location features are disabled in config."
    try:
        return location.describe_here()
    except location_client.LocationError as e:
        print(f"[location] here failed: {e}", flush=True)
        return spoken_fail("location")


def handle_weather(cfg: dict, user_text: str = "", mem: MemoryStore | None = None) -> str:
    """Read-only informational — same tier as calendar/notes/email reads,
    no confirm. Deterministic router match, nothing to extract from the
    phrasing.
    """
    if not cfg.get("location_enabled", True):
        return "Location features are disabled in config."
    lang = get_conversation_language(mem) if mem else None
    try:
        return location.describe_weather(
            include_rain=True, user_text=user_text, language=lang
        )
    except location_client.LocationError as e:
        print(f"[location] weather failed: {e}", flush=True)
        return spoken_fail("weather")


def handle_showtimes(user_text: str, cfg: dict, mem: MemoryStore | None = None) -> str:
    """Live listing search, then Maps halls as a fallback.

    Discovery is read-only. If the board cannot be read, point at
    BookMyShow rather than inventing films.
    """
    if cfg.get("discovery_enabled", True):
        try:
            reply = discovery.describe_movies(user_text)
            if reply:
                return reply
        except Exception as e:
            print(f"[showtimes] discovery failed: {type(e).__name__}: {e}", flush=True)

    city = ""
    remote_city = False
    halls = ""
    if cfg.get("location_enabled", True):
        try:
            origin, remote_city = location.search_origin(user_text)
            loc = origin or location_client.get_location()
            city = (loc.get("city") or "").strip()
            results = location_client.search_nearby(
                loc["lat"], loc["lon"], "movie_theater", radius_m=12000
            )
            names = [r.get("name", "").strip() for r in results if r.get("name")]
            if names:
                halls = ", ".join(names[:3])
                location.remember_places(names)
        except Exception as e:
            print(f"[showtimes] nearby failed: {type(e).__name__}: {e}", flush=True)

    as_in = bool(mem and get_conversation_language(mem) == "as-IN")
    where = city or "your town"
    if as_in:
        where_as = location._as_city(city) if city else "তোমাৰ চহৰ"
        if halls:
            return (
                f"ওচৰত {halls} আছে। কোনখন ছবি কেতিয়া চলিছে সেই ব’ৰ্ড "
                f"BookMyShow-ত চাবা — {where_as} বাছনি কৰি আজিৰ শ্ব’ খুলিবা।"
            )
        return (
            f"চিনেমাৰ শ্ব’ টাইম ইয়াত লাইভ নাহে। BookMyShow খুলি {where_as} "
            "বাছনি কৰা — আজি কোনখন ছবি চলিছে তাতেই পাম।"
        )
    if halls:
        lead = f"In {city}, there's {halls}" if remote_city else f"Nearest are {halls}"
        return (
            f"{lead}. Today's board isn't on Maps — "
            f"open BookMyShow, pick {where}, and you'll see what's on and when."
        )
    return (
        f"Showtimes aren't on the Maps feed. Open BookMyShow, choose {where}, "
        "and you'll get today's films and times."
    )


def handle_discovery(action: str, user_text: str, cfg: dict, interactive: bool,
                     confirm_fn=text_confirm) -> str:
    """Movies / hotels / flights — list, then maybe open. Never pay."""
    if not cfg.get("discovery_enabled", True):
        return "Listing search is disabled in config."
    try:
        if action == "hotels":
            return discovery.describe_hotels(user_text)
        if action == "flights":
            return discovery.describe_flights(user_text)
        if action == "showtimes":
            return discovery.describe_movies(user_text)
        if action == "open":
            target = discovery.pick_open_target(user_text)
            if not target:
                return "There's no listing waiting. Ask for movies, hotels, or flights first."
            title, url = target
            if not url:
                return "I don't have a safe link for that one."
            prompt = (
                f'Open “{title}”? You finish any booking or payment yourself.'
            )
            if interactive and not confirm_fn(prompt):
                return "Cancelled — nothing opened."
            return discovery.open_listing(url)
    except Exception as e:
        print(f"[discovery] failed: {type(e).__name__}: {e}", flush=True)
        return "I couldn't read that listing just now. Try again in a moment."
    return "I can look up movie showtimes, hotels, or flights. I don't book them."


# Review sites Bella is asked for by name but cannot query. STT renders
# TripAdvisor freely ("TrivatripAdvisor" came through live).
_OTHER_PLACES_SOURCE_RE = re.compile(
    r"\b(?:trip\s?advisor|tripadvisor|trivatrip\w*|yelp|zomato|booking\.com|makemytrip)\b",
    re.IGNORECASE,
)


def handle_nearby(user_text: str, cfg: dict, mem: MemoryStore | None = None) -> str:
    """Read-only informational, no confirm. Category is a plain keyword
    lookup (location.detect_category) over a small fixed vocabulary, not
    LLM extraction — see router.py's NEARBY_PATTERNS comment for why.
    """
    if not cfg.get("location_enabled", True):
        return "Location features are disabled in config."
    category = location.detect_category(user_text)
    named = location.extract_nearby_name(user_text)
    radius = cfg.get("location_default_radius_m", 3000)
    lang = get_conversation_language(mem) if mem else None
    try:
        # A city named in the request wins over GPS — "hotels in Guwahati"
        # asked from Jorhat must not answer with Jorhat hotels.
        origin, remote = location.search_origin(user_text)
        # Naming a source Bella cannot reach deserves an answer, not
        # silence: TripAdvisor is blocked (see composio_client's
        # BLOCKED_TOOLKITS — the model invented TRIP_ADVISOR_SEARCH
        # results once), so say where the list actually came from.
        source_note = ""
        if _OTHER_PLACES_SOURCE_RE.search(user_text):
            source_note = " This is from Google Maps — I'm not connected to TripAdvisor."
        if named and not category:
            return location.describe_named_nearby(named, language=lang,
                                                  origin=origin, remote=remote) + source_note
        if category:
            return location.describe_nearby(category, radius_m=radius, language=lang,
                                            origin=origin, remote=remote) + source_note
        return location.describe_text_search(user_text, language=lang,
                                             origin=origin, remote=remote) + source_note
    except (location_client.LocationError, maps_client.MapsError) as e:
        print(f"[location] nearby failed: {e}", flush=True)
        return spoken_fail("nearby")


def handle_distance(user_text: str, cfg: dict) -> str:
    if not cfg.get("location_enabled", True):
        return "Location features are disabled in config."
    dest = location.extract_distance_dest(user_text)
    if not dest:
        return "Where to? Name the place and I'll check Google Maps."
    try:
        return location.describe_distance(dest)
    except location_client.LocationError as e:
        print(f"[location] distance failed: {e}", flush=True)
        return spoken_fail("nearby")


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
        print(f"[location] open maps failed: {e}", flush=True)
        return spoken_fail("maps")
    return f'Opened Apple Maps for "{query}".' if query else "Opened Apple Maps near your current location."


def handle_phone_lookup(user_text: str, cfg: dict) -> str:
    """Say a business's phone number. Looks it up; never dials, never guesses.

    Split from handle_call because asking for a number is not asking to
    ring it, and because the answer must come from Maps rather than from
    a model — see router.PHONE_LOOKUP_PATTERNS for the invented number
    this replaces.
    """
    if not cfg.get("location_enabled", True):
        return "Location features are disabled in config."
    target = (calling.extract_number_target(user_text)
              or calling.extract_call_target(user_text)
              or location.extract_nearby_name(user_text))
    if not target:
        target = location.focused_place()
    if not target:
        remembered = location.last_places()
        if len(remembered) == 1:
            target = remembered[0]
        elif remembered:
            return f"Which one — {', '.join(remembered[:4])}?"
        else:
            return "Whose number do you want?"
    matched = location.match_remembered_place(target)
    if matched:
        target = matched
    try:
        # "the number for Novotel in Guwahati" must search Guwahati. The
        # distance cap inside find_phone is measured from whichever
        # origin it is given, so a named city moves the whole search
        # rather than widening a Jorhat one.
        origin, remote = location.search_origin(user_text)
        here = origin or location_client.get_location()
        place = location_client.find_phone(
            target, here["lat"], here["lon"], here.get("city") or ""
        )
        if not place:
            place = maps_client.find_phone(target, here["lat"], here["lon"])
    except (location_client.LocationError, maps_client.MapsError) as e:
        print(f"[phone] lookup failed: {e}", flush=True)
        return spoken_fail("lookup")
    if not place:
        # Saying so is the whole point. The alternative is what happened
        # before: a confident, invented number.
        where_not = f"in {here.get('city')}" if remote else "near you"
        return f"I couldn't find a listed number for {target} {where_not}."
    if remote:
        # Distance from a city centre the user is not standing in is not
        # information they can use.
        return (f"{place['name']} in {here.get('city')}: {place['phone']}. "
                "Say call them if you want me to dial it.")
    km = place["distance_m"] / 1000
    where = f"{km:.1f} km away" if km >= 0.1 else "just nearby"
    return f"{place['name']}, {where}: {place['phone']}. Say call them if you want me to dial it."


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

    own = (cfg.get("user_phone") or "").strip()
    number = calling.parse_phone_number(user_text)
    label = None
    detail = ""

    if number and calling.is_own_number(number, own):
        return (
            "That's your iPhone. Name the restaurant and I'll look up "
            "its number and ring it from your phone."
        )

    if not number:
        # No digits spoken, so this names a place ("call Hotel Heritage")
        # or refers back to one ("call them"). Resolving a name to a number
        # is a lookup, never a guess — if nothing is found nothing is
        # dialled, and the resolved number is always shown before the call.
        target = calling.extract_call_target(user_text)
        if target is None:
            target = location.focused_place()
        if target is None:
            remembered = location.last_places()
            if len(remembered) == 1:
                target = remembered[0]
            elif remembered:
                options = ", ".join(remembered[:4])
                return f"Which one would you like me to call — {options}?"
            else:
                return "I'm not sure who you mean — say the name or the number."
        else:
            # "Nova" after a nearby-restaurants list should ring the
            # closest Nova, not start a nationwide Maps search.
            matched = location.match_remembered_place(target)
            if matched:
                target = matched

        try:
            here = location_client.get_location()
            # Google first — it is the same provider that produced the
            # nearby list this name almost certainly came from, so the
            # place Bella dials is the place it named. Apple MapKit is the
            # fallback for when Google has no number on file.
            place = location_client.find_phone(
                target, here["lat"], here["lon"], here.get("city") or "")
            if not place:
                fallback = maps_client.find_phone(target, here["lat"], here["lon"])
                # Same guard as the Google path: MapKit will also hand
                # back a real business that is not the one asked for.
                if fallback and location_client.name_matches(target, fallback.get("name", "")):
                    place = fallback
        except (location_client.LocationError, maps_client.MapsError) as e:
            print(f"[call] lookup failed: {e}", flush=True)
            return spoken_fail("lookup")

        if not place:
            return (
                f'I couldn\'t find a phone number for "{target}" near you. '
                "Tell me the number and I'll dial it."
            )
        number = calling.normalize_number(place["phone"])
        label = place["name"]
        km = place["distance_m"] / 1000
        detail = f" ({km:.1f}km away)" if km >= 0.1 else " (nearby)"
        if own and calling.is_own_number(number, own):
            return (
                f'Maps gave your own iPhone number for "{label}", so I will not dial it. '
                "Tell me a different place."
            )

    who = f"{label} on {number}{detail}" if label else number
    if not calling.facetime_available():
        app = calling.call_app()
        return (
            f"{app} is not available on this Mac, so I cannot hand the "
            "call to your iPhone. Open the Phone app once, then ask again."
        )
    if not calling.iphone_calling_ready():
        return (
            f"I have {who}, but this Mac is not set up to call from your iPhone. "
            "On this Mac: open Phone → Settings → Calls → turn on Calls from iPhone. "
            "Unlock the iPhone, keep it nearby, then ask again."
        )

    prompt = f"Call {who} from your iPhone?"
    if interactive and not confirm_fn(prompt):
        return "Cancelled — call not placed."

    try:
        return calling.call_number(number, label)
    except calling.CallingError as e:
        print(f"[call] failed: {e}", flush=True)
        return spoken_fail("call")


def handle_tool(
    domain: str, action: str, user_text: str, cfg: dict, mem: MemoryStore, interactive: bool, confirm_fn=text_confirm
) -> str:
    try:
        if domain == "calendar":
            if action == "read":
                lowered = user_text.lower()
                days = 7 if "week" in lowered else (2 if "tomorrow" in lowered else 1)
                # Google Calendar first: measured 1.04s against 5.58s for
                # the AppleScript path, and Calendar.app was reporting "No
                # events found" for days that had real entries — the
                # account syncs to Google, not into a local calendar.
                # AppleScript stays as the offline fallback.
                if composio_client.configured():
                    try:
                        return composio_client.list_calendar_events(user_text)
                    except composio_client.ComposioError as e:
                        print(f"[calendar] Composio read failed ({e}) — using Calendar.app",
                              flush=True)
                return calendar.list_events(days=days)

            # A time was never said. Ask for it instead of rounding up to
            # the next hour and offering that as though it were the
            # request — see bella/tools/event_draft.py for what that cost.
            if event_draft.active():
                if event_draft.wants_cancel(user_text):
                    event_draft.cancel()
                    return "Dropped it."
                start = event_draft.apply_time(user_text)
                if start is None:
                    return event_draft.question()
                title, duration = event_draft.title(), event_draft.duration()
                event_draft.cancel()
            else:
                title, start, duration = parsing.extract_event(user_text)
                if not parsing.event_has_time(user_text):
                    event_draft.start(title, duration)
                    event_draft.note_asked()
                    return event_draft.question()

            end = start + duration
            when = start.strftime("%a %b %d, %I:%M %p")
            # Adding is 1.14s of AppleScript; the yes/no round trip cost
            # far more than the write it was guarding. A calendar entry is
            # also the most reversible thing Bella writes — it is visible,
            # it is one line to undo, and Bella says exactly what it added.
            # That is not true of a sent email or a shell command, which
            # still confirm. Set confirm_calendar_writes to turn it back on.
            if cfg.get("confirm_calendar_writes", False):
                if not _confirm_write(interactive, cfg,
                                      f'Add "{title}" to your calendar for {when}?', confirm_fn):
                    return "(Cancelled — nothing added.)"
            added = calendar.add_event(title, start, end)
            calendar.remember_last_added(title, start)
            return added

        if domain == "calendar_undo":
            # The escape hatch that makes adding without a confirmation
            # safe: a misheard event is one sentence to remove.
            return calendar.remove_last_added()

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
            return handle_email(user_text, cfg, action, interactive, confirm_fn)

        if domain in ("github", "youtube", "googledrive", "cal", "chatfai",
                      "googlecalendar", "googletasks", "linkedin", "reddit"):
            return handle_composio(user_text, cfg, domain=domain, action=action)

        if domain == "voice":
            return handle_voice_volume(action)

        if domain == "spotify":
            if action == "read":
                return handle_spotify(user_text, cfg)
            answer = handle_spotify_control(action, user_text)
            # Music is now the subject, so the next bare fragment is a
            # song rather than something for the chat model.
            spotify.note_music_turn()
            return answer

        if domain == "shell":
            return handle_shell(user_text, cfg, interactive, confirm_fn)

        if domain == "files":
            return handle_files_shell(user_text, cfg, mem, interactive, confirm_fn)

        if domain == "capabilities":
            if action == "model":
                return handle_model_question(cfg)
            return handle_capabilities(cfg)

        if domain == "reminder":
            return handle_reminder(user_text, cfg, interactive, confirm_fn)

        if domain == "compose":
            if action == "insert":
                return handle_compose_insert(cfg)
            return handle_compose(user_text, cfg, interactive, confirm_fn)

        if domain == "screen":
            lang = get_conversation_language(mem) if mem else None
            if _ASSAMESE_SCRIPT_RE.search(user_text or ""):
                lang = "as-IN"
            vision.set_reply_language(lang)
            try:
                if action == "region":
                    reply = handle_screen_region(user_text, cfg, interactive, confirm_fn)
                elif action == "howto":
                    reply = handle_screen_howto(user_text, cfg)
                elif action == "show":
                    reply = handle_screen_show(cfg)
                elif action == "advise":
                    reply = handle_screen_advise(user_text, cfg)
                elif action == "answer":
                    reply = handle_screen_answer(user_text, cfg)
                else:
                    reply = handle_screen(user_text, cfg, interactive, confirm_fn)
            finally:
                vision.set_reply_language(None)
            return _screen_reply_in_session_language(reply, user_text, cfg, mem)

        if domain == "desktop":
            return handle_desktop(user_text, cfg, mem, interactive, confirm_fn)

        if domain == "browser":
            return handle_browser(user_text, cfg, mem, interactive, confirm_fn)

        if domain == "location":
            if action == "here":
                return handle_here(cfg)
            if action == "weather":
                return handle_weather(cfg, user_text, mem)
            if action == "showtimes":
                return handle_showtimes(user_text, cfg, mem)
            if action == "nearby":
                return handle_nearby(user_text, cfg, mem)
            if action == "distance":
                return handle_distance(user_text, cfg)
            if action == "open_maps":
                return handle_open_maps(user_text, cfg, interactive, confirm_fn)

        if domain == "discovery":
            return handle_discovery(action, user_text, cfg, interactive, confirm_fn)

        if domain == "news":
            return handle_news(user_text, cfg)

        if domain == "omdb":
            return handle_omdb(user_text, cfg)

        if domain == "joke":
            return handle_joke(user_text, cfg, mem)

        if domain == "call":
            if action == "number":
                return handle_phone_lookup(user_text, cfg)
            return handle_call(user_text, cfg, interactive, confirm_fn)
    except (claude_handoff.Cancelled, shell.Cancelled):
        # Must reach voice_loop.py's own handler around cli.process_turn()
        # uncaught — that's what decides to skip TTS and go straight back
        # to listening instead of speaking a "tool failed" error for a
        # cancellation the user asked for on purpose.
        raise
    except Exception as e:
        print(f"[{domain}] tool failed: {type(e).__name__}: {e}", flush=True)
        return spoken_fail()

    return "(Unrecognized tool request.)"


# The model is told not to claim it acted. Small hosted models ignore that
# and say "Playing X" / "I'll start the call" anyway — seen live on Groq
# 8B after every router miss this morning. A false "done" is worse than a
# missed route, so rewrite those replies before they are spoken.
_ACTION_CLAIM_RE = re.compile(
    r"(?:"
    r"\bi(?:'m| am) (?:now )?(?:playing|calling|diall?ing|emailing|sending|opening|booking|clicking)\b|"
    # "added"/"created"/"scheduled"/"set" were missing, and that is
    # exactly the shape the worst observed lie took: asked for a reminder
    # with no route to one, the model replied "I've added a reminder for
    # you on the Google Calendar" and nothing existed. A claim to have
    # CREATED something is the most damaging kind, because the user stops
    # tracking the thing themselves.
    r"\bi(?:'ve| have) (?:just )?(?:played|called|dialled|emailed|sent|opened|clicked|"
    r"added|created|scheduled|set|saved|booked|made|put)\b|"
    r"\bi(?:'ve| have) (?:just )?(?:set up|added .{0,20}reminder)\b|"
    r"\bi(?:'ll| will)\b(?!.{0,20}\bhow\b).{0,20}\b(?:play|call|dial|email|send|open|book|ring|click)\b|"
    r"\bas i click\b|"
    r"\bhere,? i(?:'m| am) going to click\b|"
    r"\bi(?:'m| am) going to click\b|"
    # Seen live: "Please share the number" missed the lookup, Groq said
    # "Sure, let me pull that up for you. One moment." and never did.
    r"\blet me (?:pull|look|check|find|get|fetch)\b|"
    r"\bone moment\b|"
    r"\bgive me a (?:second|moment|minute)\b|"
    r"\bplaying\b|"
    r"\bcalling (?:them|[A-Z]|\d)|"
    r"\bi(?:'m| am) en route\b|"
    r"\bon your way\b|"
    r"\bsent to\b|"
    r"\bfrom your gmail\b|"
    r"\bemail with subject\b|"
    # A model announcing a tool ran. It learned this format by seeing raw
    # "REDDIT_GET_R_TOP returned. {json}" in its own history, then invented
    # LINKEDIN_CHECK_POSTS and TRIP_ADVISOR_SEARCH — neither exists — and
    # produced fabricated posts and hotel ratings to match. Bella never
    # names a tool slug out loud, so any reply that does is the model
    # play-acting.
    r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){1,}\b\s*(?:returned|responded|says|result)"
    r")",
    re.IGNORECASE,
)


# Questions whose honest answer requires a tool that did NOT run. Reached
# here, the reply is the chat model's alone, so any specific detail in it
# was invented. Measured live, all three in one conversation:
#   "in the next half an hour"  -> "Around 6 PM ... still 70% chance of rain"
#   "can you find them?"        -> "Café Lune, Brewed Awakenings, the Corner Bistro"
#   "how does the person look"  -> "long dark hair, wearing a blue jacket"
# No forecast was fetched, no place was searched, no screen was read. This
# is worse than a missed route: the user acts on it. A wrong café sends
# them across town for somewhere that does not exist.
_NEEDS_TOOL_RE = re.compile(
    r"\b(?:"
    r"find (?:me |them|one|some|a )|near(?:by| me| here)|around here|"
    r"cafe|café|coffee|restaurant|hotel|shop|petrol|pharmacy|atm|"
    r"fried chicken|hungry|craving|"
    r"forecast|rain|weather|temperature|humidity|"
    r"on (?:my|the) screen|this (?:picture|photo|image)|look like"
    r")\b",
    re.IGNORECASE,
)
# The shapes an invented answer takes: a named place, a clock time with a
# condition, a percentage, or a physical description.
_INVENTED_DETAIL_RE = re.compile(
    r"\b\d{1,3}\s*%|"
    r"\b\d{1,2}\s*(?:am|pm)\b|"
    r"\b\d{1,2}\s*°|"
    r"\b(?:wearing|dark hair|blonde|brunette|smiling|jacket|shirt|glasses)\b|"
    r"\b(?:café|cafe|bistro|restaurant|coffee shop)\b\s*[A-Z]|"
    r"[A-Z][a-z]+\s+(?:Café|Cafe|Bistro|Coffee|Roasters|Kitchen)\b|"
    r"\b\d+(?:\.\d+)?\s*(?:km|m)\b",
)


def _rewrite_unverifiable_answer(response: str, user_text: str) -> str:
    """Replace an invented answer with an offer to actually go and look."""
    if not response or not _INVENTED_DETAIL_RE.search(response):
        return response
    # The subject can live in either half. "In the next half an hour"
    # names no domain at all, yet the reply that came back was a forecast
    # — so the reply is checked too, not just the question.
    if not (_NEEDS_TOOL_RE.search(user_text or "") or _NEEDS_TOOL_RE.search(response)):
        return response
    both = f"{user_text or ''} {response}".lower()
    if any(w in both for w in ("screen", "picture", "photo", "image", "look like", "wearing")):
        return "I'd be guessing — I haven't looked. Want me to read your screen?"
    if any(w in both for w in ("weather", "rain", "forecast", "temperature", "humidity")):
        return "I haven't checked the forecast, so I won't guess at it. Want me to pull it up?"
    return "I don't want to invent places. Tell me what you're after and I'll search nearby."


def _rewrite_false_action_claim(response: str, user_text: str, cfg: dict | None = None) -> str:
    if not response or not _ACTION_CLAIM_RE.search(response):
        return response
    lowered = (user_text or "").lower()
    claimed = response.lower()
    # Written as something a person would say. The job here is only to
    # stop Bella claiming it did something it did not — it is NOT to explain
    # the routing. The old wording ("the request never reached the tool.
    # Say: play <song> on Spotify.") described Bella's own plumbing to
    # someone who just wanted a song, and reads like an error message
    # because that is what it was.
    if any(w in lowered for w in ("play", "song", "spotify", "track", "bile", "ply")) or "playing" in claimed:
        return "I didn't catch the song. What do you want me to play?"
    if any(w in lowered for w in ("call", "dial", "ring", "phone")) or "calling" in claimed:
        return "I didn't catch who to call. Say the name or the number."
    if any(w in lowered for w in ("email", "inbox", "gmail", "mail")):
        return "I haven't sent anything. Who's it for, and what should it say?"
    if any(w in lowered for w in ("number", "phone", "call", "dial", "novotel", "hotel")) or "pull" in claimed:
        return "I haven't looked it up. Name the place and I'll find the number."
    if any(w in lowered for w in ("click", "button", "tab", "pull request", "screen")) or "click" in claimed:
        if cfg and cfg.get("desktop_enabled"):
            return "I haven't clicked anything. Want me to?"
        return "I can point at it, but I don't click. Want me to show you where it is?"
    return "I haven't done that yet. Say it once more?"


# Questions about what is actually ON the screen, asked while a screen
# session is open. Seen live: after Bella correctly said "This is a picture
# open in Photos", the follow-up "I want to know how the person look like"
# matched no pattern, fell through to the chat model, and came back "She's
# a woman with long dark hair, wearing a blue jacket" — a confident
# description of a photo nothing had looked at. Anything asking about the
# CONTENT of the screen has to reach the screen, not a language model
# working from the previous sentence.
_SCREEN_CONTENT_QUESTION_RE = re.compile(
    r"\b(?:"
    r"how (?:does|do|did) (?:it|this|that|the \w+|he|she|they) look|"
    r"how (?:the )?\w+ looks?(?: like)?|"
    r"what does (?:it|this|that|the \w+) say|"
    r"who(?:'s| is| are)? (?:in )?(?:it|this|that|the (?:picture|photo|image|screen))|"
    # Seen live: Safari was on "zubeen garg", they asked "more about
    # Zubin Gark", chat invented a YouTuber. The page is the source.
    r"(?:tell me |let me know )(?:more )?about\b|"
    r"\bmore about\b|"
    r"\bwho is \w|"
    r"describe (?:it|this|that|the (?:picture|photo|image|screen))|"
    r"read (?:it|this|that|the (?:screen|page|text))|"
    r"what(?:'s| is)? (?:in|on) (?:it|this|that|the (?:picture|photo|image|screen))|"
    r"what (?:can you |do you )?see"
    r")\b",
    re.IGNORECASE,
)

# Assamese follow-ups while a screen session is open. STT writes
# "পচন টু" for "option two". Without this they hit Sarvam, which has
# no screenshot and lies.
_AS_SCREEN_FOLLOWUP_RE = re.compile(
    r"প্ৰশ্ন|প্রশ্ন|উত্তৰ|অপচন|অপশন|পচন|বিকল্প|"
    r"কি হয়|কি হয়|সঠিক|কোনটো|বুজাই|পঢ়|"
    r"কোৱাশ্চন|কোশ্চেন|কোৱেচন|কুৱেশ্চন|পেজ"
)

# A bare "what is this?" — only meaningful while a screen session is open.
_BARE_WHAT_IS_THIS_RE = re.compile(
    r"^\s*(?:hey\s+bella[,\s]*)?(?:so\s+)?what(?:'s| is| are)?\s+(?:this|that|these)"
    r"(?:\s+(?:one|thing|here|about))?\s*[?.!]*\s*$",
    re.IGNORECASE,
)

# "I'm finished with that" — closes the on-screen instruction panel.
# Anchored to whole words so "no" inside "notes" doesn't dismiss it.
_DONE_WITH_IT_RE = re.compile(
    r"\b(?:no|nope|not now|i see it|found it|got it|that'?s all|that'?s it|"
    r"all good|done|thanks|thank you|close (?:it|that|the panel))\b",
    re.IGNORECASE,
)

CONVERSATION_LANGUAGE_KEY = "conversation_language"
_ASSAMESE_SCRIPT_RE = re.compile(r"[\u0980-\u09FF]")

# Spoken slash-commands. Voice STT (especially Saaras in Assamese mode)
# will not produce "/english"; it writes "speak in English" or the
# Assamese-script rendering actually seen live: "স্পিক ইংলিছ".
_HINDI_TRIGGER_RE = re.compile(
    r"\b(?:in\s+hindi|hindi\s+mein|speak\s+(?:in\s+)?hindi|"
    r"talk\s+(?:to\s+me\s+)?(?:in\s+)?hindi|switch\s+to\s+hindi)\b",
    re.IGNORECASE,
)
_ASSAMESE_TRIGGER_RE = re.compile(
    r"\b(?:in\s+assamese|assamese\s+mein|"
    r"speak(?:ing)?\s+(?:in\s+)?(?:assamese|asamese|asamiya|axomiya|oxomiya|asmiz|asmus|achomiah|asomiya|asme|asmes|asthma)|"
    r"talk\s+(?:to\s+me\s+)?(?:in\s+)?(?:assamese|asamese|asamiya|axomiya|oxomiya)|"
    r"switch\s+to\s+(?:assamese|asamese|asamiya|axomiya|oxomiya)|"
    r"(?:asme'?s?|asmes|asthma)\s+language|"
    r"axomiya|oxomiya|asamiya|achomiah|asomiya|asme'?s?|asthma)\b|অসমীয়া|"
    r"\bassamese\b|"
    r"^(?:assamese|asamese)[?.!]*$",
    re.IGNORECASE,
)
_ENGLISH_TRIGGER_RE = re.compile(
    r"\b(?:in\s+english|speak\s+(?:in\s+)?(?:english|inglish|ingraji|angrezi)|"
    r"talk\s+(?:to\s+me\s+)?(?:in\s+)?(?:english|inglish|ingraji|angrezi)|"
    r"switch\s+to\s+(?:english|inglish)|"
    r"(?:english|inglish)\s+(?:now|please|right\s+now))\b|"
    r"স্পিক\s*(?:ইন\s*)?ইং(?:লিছ|লিশ|ৰাজী|রাজী|রেজি)|"
    r"ইং(?:লিছ|লিশ|ৰাজী|রাজী|রেজি)(?:ত)?\s*(?:ক(?:ওক|বা|'ব|ব)|ভাষা|কথা)|"
    r"ইংৰাজীত",
    re.IGNORECASE,
)
_LANG_LEFTOVER_RE = re.compile(
    r"^(?:hey\s+)?(?:\w+[,\s]+)*(?:please\s+)?(?:can|could|would)\s+you\s*(?:please)?\s*[?.!]*$"
    r"|^(?:please|now|right\s+now|okay|ok|i want you to|i want to)[?.!]*$",
    re.IGNORECASE,
)
_LANG_FILLER_RE = re.compile(
    r"(?:hey|hi|ok|okay|please|now|right\s+now|bella|oren|orange|orion|"
    r"language|ভাষা|"
    r"ওকে|হাই|হে|অৰিন|অরিন|অৰিয়ন|অৰিণ্ড|"
    r"আৰ\s*ইন|ৰাইট\s*নাউ|নাউ)",
    re.IGNORECASE,
)


def _is_language_leftover(rest: str) -> bool:
    if not rest or _LANG_LEFTOVER_RE.match(rest):
        return True
    stripped = _LANG_FILLER_RE.sub(" ", rest)
    stripped = re.sub(r"[\s,?.!।]+", " ", stripped).strip()
    return not stripped or bool(_LANG_LEFTOVER_RE.match(stripped))


def apply_language_trigger(text: str) -> str:
    """Map spoken 'speak in English / Assamese / Hindi' onto /english etc."""
    if not text:
        return text
    # English first: once Saaras is in Assamese it writes "স্পিক ইংলিছ",
    # and that must win over staying in Assamese.
    if _ENGLISH_TRIGGER_RE.search(text):
        rest = _ENGLISH_TRIGGER_RE.sub("", text).strip(" ,.?!")
        if _is_language_leftover(rest):
            return "/english"
        return "/english " + rest
    if _HINDI_TRIGGER_RE.search(text):
        rest = _HINDI_TRIGGER_RE.sub("", text).strip(" ,.?!")
        if _is_language_leftover(rest):
            return "/hindi"
        return "/hindi " + rest
    if _ASSAMESE_TRIGGER_RE.search(text):
        rest = _ASSAMESE_TRIGGER_RE.sub("", text).strip(" ,.?!")
        if _is_language_leftover(rest):
            return "/assamese"
        return "/assamese " + rest
    return text


def get_conversation_language(mem: MemoryStore) -> str | None:
    lang = mem.state().get(CONVERSATION_LANGUAGE_KEY)
    if lang in ("as-IN", "hi-IN"):
        return lang
    return None


def set_conversation_language(mem: MemoryStore, language_code: str | None) -> None:
    mem.update_state(**{CONVERSATION_LANGUAGE_KEY: language_code})


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
    stripped = apply_language_trigger(user_text.strip())

    feedback_cmd = bhashini_feedback.try_slash_command(stripped, cfg)
    if feedback_cmd is not None:
        return "ollama", feedback_cmd

    correction_ack = bhashini_feedback.consume_correction(stripped, cfg)
    if correction_ack:
        return "ollama", correction_ack

    if stripped.startswith("/remember"):
        fact = stripped[len("/remember"):].strip()
        if fact:
            mem.remember(fact)
            return "ollama", f"Noted, {cfg['user_name']}. I'll remember that."
        return "ollama", "Usage: /remember <fact>"

    # Slash prefixes are explicit. After /assamese or /hindi the language
    # sticks in memory_state so follow-up turns (voice or typed) stay in
    # that language without repeating the command. /english clears it.
    # Persisted language is NOT "forced": tools still run (play, call,
    # news) and only the leftover chat goes to Sarvam.
    explicit_sarvam = None
    for prefix, lang_code in (("/hindi", "hi-IN"), ("/assamese", "as-IN")):
        if stripped.startswith(prefix):
            explicit_sarvam = lang_code
            set_conversation_language(mem, lang_code)
            break

    english_switch = stripped.startswith("/english")
    if english_switch:
        set_conversation_language(mem, None)
        stripped = stripped[len("/english"):].strip()
        if not stripped:
            return "ollama", "Alright — I'll speak English again. What do you need?"

    forced = (
        stripped.startswith("/claude")
        or stripped.startswith("/ollama")
        or explicit_sarvam is not None
    )
    clean_text = stripped
    for prefix in ("/claude", "/ollama", "/hindi", "/assamese"):
        if clean_text.startswith(prefix):
            clean_text = clean_text[len(prefix):].strip()

    # Script alone must not undo an explicit English switch. Seen live:
    # Saaras wrote "স্পিক ইংলিছ" and this line glued Assamese back on.
    if not forced and not english_switch and _ASSAMESE_SCRIPT_RE.search(clean_text):
        set_conversation_language(mem, "as-IN")

    sarvam_language = explicit_sarvam
    if sarvam_language is None and not stripped.startswith("/ollama") and not stripped.startswith("/claude"):
        sarvam_language = get_conversation_language(mem)

    # Saaras as-IN writes spoken English in Assamese letters. Fold those
    # to English before routing or they never hit the screen tool.
    routable = router.fold_screen_phonetic(clean_text)
    if routable != clean_text:
        print(f"[stt] folded {clean_text!r} -> {routable!r}", flush=True)
    tool_match = None if forced else router.detect_tool(routable)

    # "Novotel" after a hotel list is the one they mean, even if this
    # turn is "share the number" / "call them" and names nothing new.
    if not forced:
        location.note_mentioned_place(clean_text)

    # An open email draft owns follow-ups ("GST", "yes", "headlines") so
    # they cannot fall through to news or a leftover Spotify play.
    if not forced and email_compose.active():
        if email_compose.is_topic_change(clean_text):
            email_compose.cancel()
        elif tool_match is None or tool_match[0] == "news":
            tool_match = ("email", "write")
        elif tool_match[0] == "spotify" and not parsing.has_song_trigger(clean_text):
            tool_match = ("email", "write")
        elif tool_match[0] == "email":
            # A draft is open — "D-Read Gmail.com" must not flip to inbox read.
            tool_match = ("email", "write")
        else:
            email_compose.cancel()

    # An open reminder owns the next answers. Bella asked "what should I
    # remind you about?" — the reply IS the subject, and it will match no
    # router pattern at all. Without this the answers fell through to the
    # chat model, which then claimed the reminder was set. Same shape as
    # the open email draft below.
    if not forced and reminder.active():
        if reminder.wants_cancel(clean_text):
            reminder.cancel()
            return "ollama", "Dropped it."
        if tool_match is None or tool_match[0] in ("calendar", "googlecalendar", "googletasks"):
            tool_match = ("reminder", "auto")

    # An event waiting on its time owns the next answer. "tomorrow at 1pm"
    # is a bare time phrase that matches no router pattern, so without
    # this it fell through to the chat model and the event was lost. Same
    # shape as the open reminder above.
    if not forced and event_draft.active():
        if event_draft.wants_cancel(clean_text):
            event_draft.cancel()
            return "ollama", "Dropped it."
        if tool_match is None or tool_match[0] in ("calendar", "googlecalendar"):
            tool_match = ("calendar", "write")

    # A written-but-not-typed draft owns "type it in" / "go ahead". Same
    # shape as the pending Spotify play and the pending screen pointer
    # below: the router is stateless, so the fact that a draft is waiting
    # lives here.
    # They said "write a prompt" with no topic — the next line is the topic
    # ("in python to print hello"), unless another tool already claimed it.
    if not forced and compose.waiting_for_subject() and tool_match is None:
        # Live: "ষ্টপ দ মিউজিক" and "হাই" were swallowed as the prompt
        # subject. A waiting draft is not every other intent.
        if (
            _assamese_is_greeting(clean_text)
            or _assamese_is_how_are_you(clean_text)
            or re.search(r"ষ্টপ|মিউজিক|স্ক্ৰিন|প্ৰশ্ন", clean_text)
            or re.search(r"\b(stop|music|screen|question)\b", routable, re.I)
        ):
            compose.clear_waiting_subject()
        else:
            tool_match = ("compose", "auto")

    # A waiting draft owns "type it in" even if desktop also matched that
    # phrasing — the draft is the thing they just approved.
    if not forced and compose.has_pending() and compose.wants_insert(clean_text):
        if tool_match is None or tool_match[0] == "desktop":
            tool_match = ("compose", "insert")

    # After a listing, "open the second one" / "book the first" is that
    # listing. Confirmed open of the official page only — never checkout.
    if (
        not forced
        and discovery.has_pending()
        and discovery.is_open_intent(clean_text)
        and (tool_match is None or tool_match[0] == "discovery")
    ):
        tool_match = ("discovery", "open")

    # An open teaching session owns follow-ups. Router patterns are
    # stateless, so "how do I bold this?" and a bare "yes" carry no clue
    # that they are about the Word window Bella just named — that context
    # lives here, the same way an open email draft and a pending Spotify
    # query already claim their own follow-ups above and below.
    if not forced and vision.session_active():
        if vision.has_pending_show() and parsing.is_affirmation(clean_text):
            # "Yes" right after "want me to show you where it is?"
            tool_match = ("screen", "show")
        else:
            # Done with it — take the instruction off their screen rather
            # than leaving it there until a timer notices. Deliberately
            # NOT gated on there being a pending offer: the panel is up
            # whenever Bella has taught something, whether or not it also
            # offered to point, and "got it" means the same either way.
            # This only dismisses; the turn itself still routes normally.
            if _DONE_WITH_IT_RE.search(clean_text):
                vision.clear_pending_show()
                vision.dismiss_panel()
            if tool_match is None and vision.is_advice_request(clean_text):
                # "What should my next move be?" — a judgement about THIS
                # screen. OCR cannot answer it: asked over a chess board it
                # returns text that says nothing about where the pieces
                # are, and the model fills the gap with generalities.
                tool_match = ("screen", "advise")
            elif tool_match is None and vision.is_question_help(routable):
                # "identify the right answer" on a medical SBA page is a
                # teacher question, not "how do I use Safari".
                tool_match = ("screen", "answer")
            elif tool_match is None and _AS_SCREEN_FOLLOWUP_RE.search(clean_text):
                # Live: "প্ৰশ্নটো কি?" / "পচন টু কি হয়?" during an open
                # teaching session went to Sarvam, which said it could
                # not see the screen and then invented "option two is a
                # soil bacterium". The page is the subject.
                if vision.is_question_help(clean_text) or re.search(
                    r"উত্তৰ|সঠিক|ঠিক|অপচন|অপশন|পচন|option",
                    clean_text,
                ):
                    tool_match = ("screen", "answer")
                else:
                    tool_match = ("screen", "read")
            elif tool_match is None and _SCREEN_CONTENT_QUESTION_RE.search(clean_text):
                # "How does the person look?" — about the screen, and only
                # the screen can answer it. See the pattern for what the
                # chat model said when this fell through to it.
                tool_match = ("screen", "read")
            elif tool_match is None and _BARE_WHAT_IS_THIS_RE.match(clean_text):
                # "What is this?" with a screen conversation already open
                # means the thing on screen. Seen live: it fell through to
                # chat, which answered from the previous reply still in
                # context rather than from the screen — a guess dressed as
                # an observation. Needs the session, so a cold "what is
                # this?" is still ordinary conversation.
                tool_match = ("screen", "read")
            elif tool_match is None and vision.is_help_ask(clean_text):
                # Bare "Help me" / "can you help me?" — look and ASK.
                # Not howto (that path failed with empty JSON) and not
                # a guide (that moved the ring before they said where).
                tool_match = ("screen", "read")
            elif tool_match is None and vision.is_guide(clean_text):
                # "show me the Playground tab" / "where do I click" —
                # recapture and move the pointer onto the control.
                tool_match = ("screen", "read")
            elif tool_match is None and vision.matches_visible_control(clean_text):
                # They named a label already on the card ("Playground",
                # "API Keys") without saying "show me".
                tool_match = ("screen", "read")
            elif tool_match is None and vision.is_howto(clean_text):
                # "How do I bold this?" with no app named — it's the app
                # on screen. Only claimed when nothing else matched, so
                # "how do I play music" still reaches Spotify.
                tool_match = ("screen", "howto")
            elif tool_match is None and re.fullmatch(
                r"(?:hey[, ]+)?(?:bella[, ]+)?(?:can|could) you(?: just)?\s*[.?!]*",
                clean_text,
                re.IGNORECASE,
            ):
                # Seen live after the medical SBA page: "can you just"
                # (STT cut off) fell through to chat. The open screen
                # session is the subject — keep teaching.
                tool_match = ("screen", "answer")

    # Exam help without an open teaching session: they pointed the
    # question at the screen and asked for the answer.
    if not forced and tool_match is None and vision.is_question_help(clean_text):
        tool_match = ("screen", "answer")

    # While music is the live subject, a bare fragment is a song. STT
    # loses titles far more than artists, so the follow-up is regularly
    # just "by Metallica" — which matched nothing, reached the chat model,
    # and came back "I can't play Spotify here. Say 'play <song> by
    # <artist> on Spotify'". A refusal, a recitation of syntax, and untrue.
    if not forced and tool_match is None and spotify.looks_like_followup(clean_text):
        tool_match = ("spotify", "play_track")

    # Remember a named song even when this turn doesn't play it, so a
    # follow-up "yes" / "play it" can. And if the user just said "yes"
    # after such a miss, send it to Spotify instead of letting the model
    # claim it is playing. Only an explicit play verb may arm that.
    if not forced:
        if (
            tool_match is None
            and parsing.is_affirmation(clean_text)
            and spotify.has_pending_play()
        ):
            tool_match = ("spotify", "play_track")
            clean_text = spotify.last_query() or clean_text
        elif parsing.has_song_trigger(clean_text):
            maybe_song = parsing.extract_song_query(clean_text)
            if maybe_song and not parsing.is_referential_song_query(maybe_song):
                spotify.remember_query(maybe_song)
        elif not (tool_match and tool_match[0] == "spotify"):
            spotify.clear_pending_play()

    # Regex miss → optional Laya System-1 fallback before Sarvam/chat LLM.
    # Disabled by default; see scripts/laya_poc/FINETUNE_RECOMMENDATION.md.
    if tool_match is None and cfg.get("laya_enabled"):
        try:
            from bella import laya_router
            if cfg.get("laya_guard_enabled") and laya_router.guard_check(routable, cfg):
                return "ollama", "I can't process that request."
            tool_match = laya_router.classify(routable, cfg)
            if tool_match:
                print(f"[laya] routed {routable!r} -> {tool_match[0]}/{tool_match[1]}", flush=True)
        except ImportError:
            print("[laya] pip install laya to enable laya_enabled", flush=True)
        except Exception as e:
            print(f"[laya] classify failed: {e}", flush=True)

    if tool_match:
        domain, action = tool_match
        backend = f"tools:{domain}"
        tool_text = routable if domain == "screen" else clean_text
        response = handle_tool(domain, action, tool_text, cfg, mem, interactive, confirm_fn)
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
            # log both say Groq/Gemini/Bella rather than always "ollama".
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
            response = _rewrite_false_action_claim(response, clean_text, cfg)
            response = _rewrite_unverifiable_answer(response, clean_text)

    # Empty user text is a language switch leftover ("speak Assamese"
    # with nothing after). Logging it as "" poisons the next Sarvam call.
    shown = wake_phrase.strip_address_name(clean_text or "")
    if shown.strip():
        mem.log_turn("user", shown, backend, source=source)
    mem.log_turn("assistant", response, backend, source=source)
    # Learn from what just happened. Returns immediately — the distilling
    # runs on a background thread, after the reply is already on its way,
    # so it can never cost the user latency. See bella/reflection.py.
    reflection.maybe_reflect_async(mem, cfg)
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
    if backend == "tools:github":
        return "[GitHub]"
    if backend == "tools:youtube":
        return "[YouTube]"
    if backend == "tools:googledrive":
        return "[Drive]"
    if backend == "tools:cal":
        return "[Cal]"
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
    if backend == "tools:news":
        return "[News]"
    # These five were missing, so a deterministic tool answer was labelled
    # "[Bella]" — crediting the chat model for a line it never wrote.
    if backend == "tools:omdb":
        return "[Movies]"
    if backend == "tools:joke":
        return "[Joke]"
    if backend == "tools:voice":
        return "[Voice]"
    if backend == "tools:capabilities":
        return "[Bella]"
    if backend == "tools:chatfai":
        return "[ChatFAI]"
    if backend == "tools:compose":
        return "[Draft]"
    if backend == "tools:reminder":
        return "[Reminder]"
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
    # The local model is Bella itself as far as anyone using this is
    # concerned — "Ollama" is the runtime it happens to be served by, and
    # naming the plumbing in the transcript reads like a different
    # assistant answered. Every other label here names the capability, not
    # the vendor behind it.
    return "[Bella]"


def main():
    parser = argparse.ArgumentParser(description="Bella text conversation loop")
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

    print(f"Bella online. Good day, {cfg['user_name']}. Type /help for commands.\n")
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
