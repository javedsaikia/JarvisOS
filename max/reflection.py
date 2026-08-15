"""Turning conversations into a profile, without being asked to.

`/remember` only ever captured what someone thought to say out loud. The
things that actually make an assistant feel like it knows you — that you
want short answers, that you ask about the same three projects, that you
speak in a particular register — are never volunteered; they have to be
noticed. So this reads the turns since it last ran and folds what it
finds into memory/profile.md.

Design decisions that matter more than the prompt:

- **It never blocks a turn.** Reflection runs on a background thread after
  a reply has already been delivered, and only one runs at a time.
  Latency is the thing people actually notice; nothing here is allowed to
  cost the user a second.
- **It defaults to the local model, and the profile is only ever
  rewritten by a model you chose for the job.** Cost rules are unchanged:
  the "memory" role starts local and free, and stays that way until you
  assign it something else in the dashboard.
- **It appends and merges, never truncates.** The distiller proposes; the
  merge decides. New items are checked against what is already there so
  the same observation doesn't accumulate ten slightly different ways,
  and each section is capped so the profile can't grow into a context
  problem of its own.
- **It cannot touch facts.md.** Anything you wrote by hand stays exactly
  as you wrote it. If the automatic profile ever goes wrong, deleting
  profile.md loses nothing that was yours.
"""
import json
import re
import threading
import time

from max import llm

# Reflect once there are this many new turns, or this long after the last
# reflection with at least a couple of turns to look at. Both are needed:
# a busy hour should distil more than once, and a quiet day that produced
# one real exchange shouldn't wait forever to be noticed.
DEFAULT_EVERY_TURNS = 20
DEFAULT_MIN_SECONDS = 20 * 60
MIN_TURNS = 4

# Per-section ceiling in profile.md. A profile is a summary; past a few
# dozen bullets it stops being one, and it is injected into every prompt.
MAX_ITEMS_PER_SECTION = 24

_running = threading.Lock()

EXTRACTION_PROMPT = """You are maintaining a profile of a person, for an assistant that talks to them daily.

Below is a recent slice of their conversation with the assistant, and the profile so far.

Return ONLY a JSON object, no prose. Every entry is an object with a
"text" and a "quote", where the quote is the user's EXACT words that
prove it:

{{
  "identity": [{{"text": "Prefers to be called Javed", "quote": "just call me Javed"}}],
  "speaking_style": [],
  "preferences": [],
  "recurring_topics": []
}}

- identity: durable facts about who they are — name, work, location, people, tools
- speaking_style: how they talk and how they want to be answered — length, tone, formality, language
- preferences: what they like, want, avoid, or ask for repeatedly
- recurring_topics: subjects they keep returning to

Rules:
- The quote must be words the USER actually typed or said in the conversation below. Copy them exactly.
- If you cannot quote it, leave it out. Never infer personality, feelings, or circumstances.
- Never invent a city or home. Location only if they said it themselves.
- Each "text" is one short sentence about the user in third person ("Prefers ...", "Works on ...").
- Do NOT repeat anything already in the profile, unless you are making it more specific.
- Ignore small talk, and ignore anything the assistant said about itself.
- Return an empty list for a key you learned nothing new for.
- Never include passwords, API keys, card numbers or anything credential-shaped.

Profile so far:
{profile}

Conversation:
{conversation}
"""

KEY_TO_SECTION = {
    "identity": "Identity",
    "speaking_style": "Speaking style",
    "preferences": "Preferences",
    "recurring_topics": "Recurring topics",
}

# Anything matching this never enters the profile, whatever the model
# proposes. The profile is injected into every prompt, including ones sent
# to hosted providers — a credential that lands here would be sent to them
# on every single turn thereafter.
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|gsk_[A-Za-z0-9]{8,}|AIza[A-Za-z0-9_\-]{10,}"
    r"|\b\d{13,19}\b|password\s*[:=]|api[_ -]?key\s*[:=])",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split().__str__()


def _too_similar(candidate: str, existing: list[str]) -> bool:
    """Word-overlap duplicate check.

    Deliberately not exact-match: models restate the same observation
    endlessly ("Prefers short answers" / "Likes concise replies"), and a
    profile full of near-duplicates is both useless to read and wasteful
    to inject.
    """
    words = set(re.sub(r"[^a-z0-9 ]", " ", candidate.lower()).split())
    if not words:
        return True
    for item in existing:
        other = set(re.sub(r"[^a-z0-9 ]", " ", item.lower()).split())
        if not other:
            continue
        overlap = len(words & other) / max(1, min(len(words), len(other)))
        if overlap >= 0.6:
            return True
    return False


def _format_conversation(turns: list[dict], user_name: str) -> str:
    lines = []
    for turn in turns:
        who = user_name if turn.get("role") == "user" else "Max"
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        # Long tool dumps (an inbox summary, a file listing) say nothing
        # about the person and would crowd out the turns that do.
        if len(content) > 600:
            content = content[:600] + " …"
        lines.append(f"{who}: {content}")
    return "\n".join(lines)


def _parse(reply: str) -> dict:
    """Pull the JSON object out of a model reply, fenced or not."""
    text = reply.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


# Items whose whole content is a shrug. A small model asked for a list
# will fill it rather than return it empty, and "Tools: Unknown" as
# permanent context is worse than nothing.
_EMPTY_RE = re.compile(
    r"^\s*(?:[a-z /]+:\s*)?(?:unknown|none|n/?a|not (?:specified|mentioned|stated|provided|employed)"
    r"|unclear|no (?:information|data|preference))\b",
    re.IGNORECASE,
)


def _evidenced(quote: str, conversation: str) -> bool:
    """Is this quote really in the conversation?

    The single most important check here. A profile is permanent context
    injected into every future prompt, so an invented line does not just
    produce one bad answer, it produces bad answers forever. Measured on
    real history, the small local model confidently produced "Known for
    being reflective and deeply loving towards others" — from a
    conversation containing nothing of the sort.

    Requiring the model to quote its evidence, and then checking that
    quote actually appears, turns "please don't hallucinate" into
    something mechanically enforced. The match is fuzzy on word overlap
    rather than exact, because models paraphrase quotes slightly even when
    the underlying evidence is real.
    """
    normalize = lambda t: " ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split())
    quote_norm = normalize(quote)
    if len(quote_norm.split()) < 3:
        return False

    # Strongest evidence: the quote appears verbatim. Checked first
    # because a real short quote ("i live in bangalore") carries few
    # long words and would fail the overlap test below on length alone —
    # which rejected a correctly evidenced fact in testing.
    if quote_norm in normalize(conversation):
        return True

    # Otherwise fall back to word overlap, for quotes the model
    # paraphrased slightly. Short words are dropped here since "i", "in"
    # and "the" match everything and prove nothing.
    words = [w for w in quote_norm.split() if len(w) > 2]
    if len(words) < 3:
        return False
    haystack = set(normalize(conversation).split())
    return sum(1 for w in words if w in haystack) / len(words) >= 0.8


def merge(
    sections: dict[str, list[str]], proposed: dict, conversation: str = ""
) -> tuple[dict[str, list[str]], int]:
    """Fold proposed items into the existing profile. Returns (sections, added).

    `conversation` is the text the proposal was drawn from; when given,
    every item must quote it to be accepted.
    """
    added = 0
    for key, section in KEY_TO_SECTION.items():
        items = proposed.get(key) or []
        if not isinstance(items, list):
            continue
        existing = sections.setdefault(section, [])
        for raw in items:
            if isinstance(raw, dict):
                item = str(raw.get("text") or "").strip()
                quote = str(raw.get("quote") or "")
            else:
                item, quote = str(raw).strip(), ""
            item = item.rstrip(".")
            if not item or len(item) > 200:
                continue
            if _SECRET_RE.search(item) or _SECRET_RE.search(quote):
                continue
            if _EMPTY_RE.match(item):
                continue
            if conversation and not _evidenced(quote, conversation):
                continue
            if _too_similar(item, existing):
                continue
            existing.append(item)
            added += 1
        # Oldest first out: a profile should drift toward who someone is
        # now, not who they were when the file was created.
        if len(existing) > MAX_ITEMS_PER_SECTION:
            del existing[: len(existing) - MAX_ITEMS_PER_SECTION]
    return sections, added


def due(mem, cfg: dict) -> bool:
    if not cfg.get("memory_reflection_enabled", True):
        return False
    state = mem.state()
    last_ts = state.get("last_reflection_ts", 0)
    new_turns = len(mem.turns_since(last_ts))
    if new_turns < MIN_TURNS:
        return False
    if new_turns >= cfg.get("memory_reflect_every_turns", DEFAULT_EVERY_TURNS):
        return True
    return (time.time() - last_ts) >= cfg.get("memory_reflect_min_seconds", DEFAULT_MIN_SECONDS)


def reflect(mem, cfg: dict, verbose: bool = False) -> int:
    """Distil new turns into the profile. Returns how many items were added."""
    # Retention housekeeping rides along here rather than running at
    # startup: this is already a background thread that only one caller
    # can be inside, so the log rewrite can't race a second pruner or
    # delay a turn.
    moved = mem.prune(cfg.get("memory_retention_days", 21))
    if moved:
        print(f"[memory] archived {moved} turn(s) past the retention window", flush=True)

    state = mem.state()
    last_ts = state.get("last_reflection_ts", 0)
    turns = mem.turns_since(last_ts)
    if len(turns) < MIN_TURNS:
        return 0

    # Cap what one pass looks at: enough to see a pattern, small enough to
    # fit a small local model's context comfortably.
    turns = turns[-cfg.get("memory_reflect_max_turns", 60):]
    sections = mem.load_profile_sections()
    conversation = _format_conversation(turns, cfg.get("user_name", "the user"))
    # Only the user's own words count as evidence — otherwise the model
    # can "quote" something the assistant said and pass its own inference
    # off as proof.
    user_said = "\n".join(
        (t.get("content") or "") for t in turns if t.get("role") == "user"
    )
    prompt = EXTRACTION_PROMPT.format(
        profile=mem.load_profile() or "(empty)", conversation=conversation
    )

    entry = llm.resolve(cfg, "memory")
    started = time.monotonic()
    try:
        reply = llm.chat(
            [{"role": "user", "content": prompt}], entry, cfg,
            options={"num_predict": cfg.get("memory_reflect_max_tokens", 500)},
        )
    except llm.LLMError as e:
        if verbose:
            print(f"[memory] reflection skipped: {e}", flush=True)
        return 0

    proposed = _parse(reply)
    if not proposed:
        if verbose:
            print("[memory] reflection produced no usable JSON", flush=True)
        # Still mark the attempt, so a model that reliably returns prose
        # doesn't make every subsequent turn retry the same window.
        mem.update_state(last_reflection_ts=turns[-1].get("ts", time.time()))
        return 0

    sections, added = merge(sections, proposed, conversation=user_said)
    if added:
        mem.save_profile_sections(sections)
    mem.update_state(
        last_reflection_ts=turns[-1].get("ts", time.time()),
        last_reflection_model=entry["label"],
        last_reflection_added=added,
    )
    print(
        f"[memory] reflected over {len(turns)} turns with {entry['label']} "
        f"in {time.monotonic() - started:.1f}s — {added} new item(s)",
        flush=True,
    )
    return added


def maybe_reflect_async(mem, cfg: dict) -> None:
    """Called after a turn. Returns immediately; work happens on a thread.

    The lock is the whole concurrency story: if a reflection is already
    running (or the voice loop and the web UI both finish a turn at once),
    the second caller simply does nothing rather than distilling the same
    window twice and doubling the profile's entries.
    """
    if not due(mem, cfg):
        return
    if not _running.acquire(blocking=False):
        return

    def run():
        try:
            reflect(mem, cfg)
        except Exception as e:  # never take a turn down with us
            print(f"[memory] reflection failed: {type(e).__name__}: {e}", flush=True)
        finally:
            _running.release()

    threading.Thread(target=run, daemon=True).start()


def main():
    """`python3 -m max.reflection [--now|--show|--prune]` — inspect and
    drive the memory system by hand."""
    import sys

    from max.config import load_config
    from max.memory import MemoryStore

    cfg = load_config()
    mem = MemoryStore()
    args = sys.argv[1:]

    if "--show" in args or not args:
        state = mem.state()
        turns = mem.all_turns()
        print(f"Conversation log : {len(turns)} turns")
        if turns:
            span = (turns[-1]["ts"] - turns[0]["ts"]) / 86400
            print(f"                   spanning {span:.1f} days")
        print(f"Facts (yours)    : {len(mem.load_facts().splitlines())} lines")
        print(f"Reflection model : {llm.resolve(cfg, 'memory')['label']}")
        if state.get("last_reflection_ts"):
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(state["last_reflection_ts"]))
            print(f"Last reflection  : {when} ({state.get('last_reflection_added', 0)} added, "
                  f"{state.get('last_reflection_model', '?')})")
        else:
            print("Last reflection  : never")
        print(f"Pending turns    : {len(mem.turns_since(state.get('last_reflection_ts', 0)))}"
              f" (due: {due(mem, cfg)})")
        print("\n" + (mem.load_profile() or "(no profile yet)"))

    if "--prune" in args:
        moved = mem.prune(cfg.get("memory_retention_days", 21))
        print(f"Archived {moved} turn(s) older than "
              f"{cfg.get('memory_retention_days', 21)} days.")

    if "--now" in args:
        print("Reflecting...")
        added = reflect(mem, cfg, verbose=True)
        print(f"{added} item(s) added.\n")
        print(mem.load_profile() or "(no profile yet)")


if __name__ == "__main__":
    main()
