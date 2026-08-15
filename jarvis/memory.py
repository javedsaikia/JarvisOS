"""What Orin knows about you, kept locally and injected into every model.

The model is the reasoning engine, not the memory. Everything personal
lives in this directory as plain text you can read and edit, and is fed
into whichever model is answering — so switching from the local model to
Groq or Gemini changes who is thinking, never who Orin thinks you are.

Three files, with deliberately different owners:

- `facts.md` — **yours**. Written by hand and by `/remember`. Nothing
  automatic ever edits it. If Orin's self-written notes go wrong, this is
  the copy that is still exactly what you said.
- `profile.md` — **Orin's**. Distilled automatically from conversations
  (see jarvis/reflection.py): how you speak, what you prefer, what you
  keep coming back to. Rewritten as it learns, which is precisely why it
  is kept apart from facts.md.
- `conversation_log.jsonl` — the raw turns, append-only, one JSON object
  per line. Entries older than the retention window move to
  `conversation_archive.jsonl` rather than being deleted; nothing said is
  ever thrown away, it just stops being loaded on every turn.

`memory_state.json` holds the bookkeeping (when reflection last ran, how
many turns it had seen), so restarts don't re-distill the same history.
"""
import json
import os
import time
from pathlib import Path

MEMORY_DIR = Path(__file__).parent / "memory"
FACTS_PATH = MEMORY_DIR / "facts.md"
PROFILE_PATH = MEMORY_DIR / "profile.md"
LOG_PATH = MEMORY_DIR / "conversation_log.jsonl"
ARCHIVE_PATH = MEMORY_DIR / "conversation_archive.jsonl"
STATE_PATH = MEMORY_DIR / "memory_state.json"

# Sections of profile.md, in the order they are written. Fixed set, so the
# file stays predictable enough to read at a glance and to merge into
# without re-parsing free-form prose.
PROFILE_SECTIONS = ("Identity", "Speaking style", "Preferences", "Recurring topics")

EMPTY_PLACEHOLDER = "(nothing yet)"

PROFILE_HEADER = """# Orin's profile of you

Written automatically from conversations (jarvis/reflection.py). Safe to
edit or delete — it will be rebuilt over time. Hand-written facts belong
in facts.md, which nothing automatic ever touches.
"""


class MemoryStore:
    def __init__(self):
        MEMORY_DIR.mkdir(exist_ok=True)
        if not FACTS_PATH.exists():
            FACTS_PATH.write_text("# Orin Memory\n\n## Facts & Preferences\n")

    # --- The two durable stores -----------------------------------------

    def load_facts(self) -> str:
        return FACTS_PATH.read_text().strip() if FACTS_PATH.exists() else ""

    def remember(self, fact: str) -> None:
        with FACTS_PATH.open("a") as f:
            f.write(f"- {fact.strip()}\n")

    def load_profile(self) -> str:
        return PROFILE_PATH.read_text().strip() if PROFILE_PATH.exists() else ""

    def load_profile_sections(self) -> dict[str, list[str]]:
        """profile.md parsed back into {section: [bullet, ...]}.

        Parsed rather than kept in a database so the file itself stays the
        source of truth: editing or deleting a line in a text editor is a
        supported way to correct what Orin believes.
        """
        sections: dict[str, list[str]] = {name: [] for name in PROFILE_SECTIONS}
        current = None
        for line in self.load_profile().splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                current = stripped[3:].strip()
                sections.setdefault(current, [])
            elif stripped.startswith("- ") and current:
                item = stripped[2:].strip()
                # The empty-section placeholder is presentation, not
                # content: reading it back as an item would let it block
                # real items as a near-duplicate and, worse, get injected
                # into prompts as something Orin "knows".
                if item and item != EMPTY_PLACEHOLDER:
                    sections[current].append(item)
        return sections

    def save_profile_sections(self, sections: dict[str, list[str]]) -> None:
        parts = [PROFILE_HEADER]
        for name in PROFILE_SECTIONS:
            items = sections.get(name) or []
            parts.append(f"\n## {name}\n")
            if items:
                parts.extend(f"- {item}\n" for item in items)
            else:
                parts.append(f"- {EMPTY_PLACEHOLDER}\n")
        _atomic_write(PROFILE_PATH, "".join(parts))

    # --- Raw history ----------------------------------------------------

    def log_turn(self, role: str, content: str, backend: str = None, source: str = "text") -> None:
        # source distinguishes which front end produced this turn — "voice"
        # (voice_loop.py, a separate OS process with no shared in-memory
        # state) vs "text" (cli.py's terminal loop / bridge.py's web UI).
        # bridge.py's transcript relay uses this to forward voice_loop's
        # conversation into the web UI without also re-broadcasting the web
        # UI's own turns, which it already streams live as they happen.
        entry = {"ts": time.time(), "role": role, "content": content, "backend": backend, "source": source}
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def all_turns(self) -> list[dict]:
        return _read_jsonl(LOG_PATH)

    def turns_since(self, ts: float) -> list[dict]:
        return [t for t in self.all_turns() if t.get("ts", 0) > ts]

    def recent_turns(self, n: int = 12, exclude_backend_prefixes: tuple[str, ...] = ()) -> list[dict]:
        """exclude_backend_prefixes filters out turns whose logged backend
        starts with any of these — e.g. excluding "sarvam:" so Ollama's
        small English-tuned model never sees a nearby Hindi/Assamese
        exchange as context. Verified live: seeing Sarvam's Devanagari
        output in recent history made Ollama produce garbled Hindi-ish
        gibberish on an unrelated English turn, not just imitate the
        language — an actual quality failure, not a stylistic one.
        Filtering happens before slicing to n, so excluded turns don't
        shrink the effective context window below what was asked for.
        """
        entries = self.all_turns()
        if exclude_backend_prefixes:
            entries = [e for e in entries if not (e.get("backend") or "").startswith(exclude_backend_prefixes)]
        return entries[-n:]

    def prune(self, retention_days: int = 21) -> int:
        """Move turns older than the window into the archive. Returns how
        many moved.

        Archived rather than deleted: the requirement is that recent
        history is always available, not that old history is destroyed,
        and a distiller that later wants a longer view can still read it.
        """
        entries = self.all_turns()
        if not entries:
            return 0
        cutoff = time.time() - retention_days * 86400
        keep = [e for e in entries if e.get("ts", 0) >= cutoff]
        old = [e for e in entries if e.get("ts", 0) < cutoff]
        if not old:
            return 0
        with ARCHIVE_PATH.open("a") as f:
            for entry in old:
                f.write(json.dumps(entry) + "\n")
        _atomic_write(LOG_PATH, "".join(json.dumps(e) + "\n" for e in keep))
        return len(old)

    # --- Bookkeeping ----------------------------------------------------

    def state(self) -> dict:
        if not STATE_PATH.exists():
            return {}
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            return {}

    def update_state(self, **changes) -> None:
        current = self.state()
        current.update(changes)
        _atomic_write(STATE_PATH, json.dumps(current, indent=2))

    # --- What every model is told ---------------------------------------

    def context_block(self) -> str:
        """The personal context injected into every backend.

        One place builds this, and every provider path goes through it, so
        "what Orin knows about you" cannot differ between models — that is
        the whole point of keeping memory out of the model.
        """
        parts = []
        facts = self.load_facts()
        if facts:
            parts.append(facts)
        # Rebuilt from the parsed sections rather than pasted in whole:
        # the file's header is documentation for a human, and its
        # "(nothing yet)" placeholders are presentation. Neither belongs
        # in a prompt that is sent on every single turn.
        sections = self.load_profile_sections()
        learned = []
        for name in PROFILE_SECTIONS:
            items = [i for i in sections.get(name, []) if i]
            if items:
                learned.append(f"{name}:")
                learned.extend(f"- {item}" for item in items)
        if learned:
            parts.append("Learned from your conversations so far:\n" + "\n".join(learned))
        return "\n\n".join(parts).strip()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file and rename.

    The voice loop and the bridge are separate processes appending to the
    same log; a plain truncate-and-write would let one of them read a
    half-written file. os.replace is atomic, so a reader sees either the
    old file or the new one.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
