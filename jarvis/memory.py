"""Durable memory: facts.md (human-editable preferences/facts) + conversation_log.jsonl (recent context)."""
import json
import time
from pathlib import Path

MEMORY_DIR = Path(__file__).parent / "memory"
FACTS_PATH = MEMORY_DIR / "facts.md"
LOG_PATH = MEMORY_DIR / "conversation_log.jsonl"


class MemoryStore:
    def __init__(self):
        MEMORY_DIR.mkdir(exist_ok=True)
        if not FACTS_PATH.exists():
            FACTS_PATH.write_text("# JARVIS Memory\n\n## Facts & Preferences\n")

    def load_facts(self) -> str:
        return FACTS_PATH.read_text().strip()

    def remember(self, fact: str) -> None:
        with FACTS_PATH.open("a") as f:
            f.write(f"- {fact.strip()}\n")

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
        if not LOG_PATH.exists():
            return []
        lines = LOG_PATH.read_text().strip().splitlines()
        entries = [json.loads(line) for line in lines]
        if exclude_backend_prefixes:
            entries = [e for e in entries if not (e.get("backend") or "").startswith(exclude_backend_prefixes)]
        return entries[-n:]
