#!/usr/bin/env python3
"""Benchmark zero-shot Laya hierarchical routing vs Bella regex router.

Uses conversation_log.jsonl (real misses) + checkpoints ROUTES matrix +
paraphrased miss cases to reach ~100 labeled utterances.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bella import router as bella_router  # noqa: E402

# Import hierarchical schema from integration module once it exists.
from bella.laya_router import classify_hierarchical, COARSE_CATEGORIES  # noqa: E402

LOG = ROOT / "bella" / "memory" / "conversation_log.jsonl"

# Ground truth from checkpoints.py ROUTES (subset expanded with paraphrases).
ROUTES_MATRIX: list[tuple[str, str | None]] = [
    ("what's on my calendar tomorrow", "calendar/read"),
    ("show me my schedule for next week", "calendar/read"),
    ("am I free at 3pm", "calendar/read"),
    ("add a meeting at 1pm today with the insurance agents", "calendar/write"),
    ("schedule a call with Bob tomorrow at 3pm", "calendar/write"),
    ("book an appointment with the dentist", "calendar/write"),
    ("take a note: buy milk", "notes/write"),
    ("search my notes for dentist", "notes/read"),
    ("check my email", "email/read"),
    ("read my inbox", "email/read"),
    ("send an email to john@example.com", "email/write"),
    ("draft an email to sabina@gmail.com", "email/write"),
    ("play High Hopes on Spotify", "spotify/play_track"),
    ("play Bohemian Rhapsody", "spotify/play"),
    ("pause the music", "spotify/pause"),
    ("stop the music", "spotify/pause"),
    ("what's the weather", "location/weather"),
    ("what's the weather in Guwahati", "location/weather"),
    ("where am I", "location/here"),
    ("hotels near me", "location/nearby"),
    ("hotels in Goa", "discovery/hotels"),
    ("flights from Guwahati to Delhi on Friday", "discovery/flights"),
    ("what's on my screen", "screen/read"),
    ("read my screen", "screen/read"),
    ("what's the question on this page", "screen/read"),
    ("how do I click the Save button", "screen/read"),
    ("click the Save button", "desktop/auto"),
    ("run ls -la", "shell/auto"),
    ("list files in bella/tools", "files/auto"),
    ("call Nova", "call/auto"),
    ("tell me a joke", "joke/auto"),
    ("tell me about the movie Inception", "omdb/auto"),
    ("what can you do", "capabilities/read"),
    ("what's in the news about AI", "news/auto"),
    ("write an email about pricing", "compose/auto"),
    ("remind me to call mom at 5pm", "reminder/auto"),
    ("open google calendar", "googlecalendar/auto"),
    ("check my github", "github/auto"),
    ("search youtube for tutorials", "youtube/auto"),
    # Negatives — should stay chat (no tool)
    ("how are you", None),
    ("I love you", None),
    ("where is Paris", None),
    ("show me a good recipe for pasta", None),
    ("how do I show gratitude", None),
    ("explain quantum computing", None),
    ("what is the meaning of life", None),
    ("thanks", None),
    ("hello", None),
    ("good morning", None),
]

# Paraphrases that historically miss regex (from router comments / live logs).
MISS_PARAPHRASES: list[tuple[str, str | None]] = [
    ("Can you add a meeting at 1pm today with the insurance agents?", "calendar/write"),
    ("What's going on the calendar", "calendar/read"),
    ("Do I have anything scheduled today", "calendar/read"),
    ("Put a dentist visit on my calendar next Tuesday", "calendar/write"),
    ("Any new mail in my gmail", "email/read"),
    ("Go through my inbox and summarize", "email/read"),
    ("Reply to Zapier and say thank you", "email/write"),
    ("Compose a mail about the invoice", "email/write"),
    ("Don't play the music", "spotify/pause"),
    ("Turn the volume down on Spotify", "spotify/volume_down"),
    ("Next song please", "spotify/next"),
    ("What's the temperature outside", "location/weather"),
    ("Find restaurants close to me", "location/nearby"),
    ("Open that in Apple Maps", "location/open_maps"),
    ("What movies are playing at the cinema", "location/showtimes"),
    ("Circle this area on screen", "screen/region"),
    ("Guide me to the settings tab", "screen/read"),
    ("Execute git status in the project", "shell/auto"),
    ("Read the README file", "files/auto"),
    ("Dial Sabina's number", "call/auto"),
    ("Set a reminder for tomorrow morning", "reminder/auto"),
    ("Write me a cover letter", "compose/auto"),
    ("What's trending in tech news", "news/auto"),
    ("Who directed Inception", "omdb/auto"),
    ("Which model are you using", "capabilities/model"),
    ("Refactor this service using dependency injection", None),  # claude_code, not tool
    ("Build me a Python script", None),
    ("Implement user authentication", None),
]

# Hindi samples (multilingual routing test).
HINDI_SAMPLES: list[tuple[str, str | None]] = [
    ("मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।", "email/read"),  # billing-ish → email
    ("आज मौसम कैसा है", "location/weather"),
    ("मेरा इनबॉक्स चेक करो", "email/read"),
    ("कल मेरी कैलेंडर में क्या है", "calendar/read"),
    ("स्पotify पर गाना बजाओ", "spotify/play"),
    ("नमस्ते कैसे हो", None),
]


def load_log_misses() -> list[tuple[str, str | None]]:
    """Utterances from conversation_log that regex missed (went to sarvam/chat)."""
    if not LOG.exists():
        return []
    out: list[tuple[str, str | None]] = []
    for line in LOG.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("role") != "user":
            continue
        text = (row.get("content") or "").strip()
        if not text or len(text) < 3:
            continue
        backend = row.get("backend") or ""
        folded = bella_router.fold_screen_phonetic(text)
        detected = bella_router.detect_tool(folded)
        if detected:
            continue  # regex hit — not a miss
        # Label from backend when it was a tool anyway (state override path).
        if backend.startswith("tools:"):
            label = backend.replace("tools:", "", 1)
            out.append((text, label))
        elif backend.startswith("sarvam:"):
            out.append((text, None))  # chat in indic
        else:
            out.append((text, None))
    return out


def build_dataset() -> list[dict]:
    seen: set[str] = set()
    cases: list[dict] = []

    def add(text: str, want: str | None, source: str) -> None:
        key = text.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        folded = bella_router.fold_screen_phonetic(text)
        regex = bella_router.detect_tool(folded)
        regex_label = f"{regex[0]}/{regex[1]}" if regex else None
        cases.append({
            "text": text,
            "want": want,
            "regex": regex_label,
            "source": source,
            "is_miss": regex_label is None and want is not None,
        })

    for text, want in ROUTES_MATRIX:
        add(text, want, "routes_matrix")
    for text, want in MISS_PARAPHRASES:
        add(text, want, "miss_paraphrase")
    for text, want in HINDI_SAMPLES:
        add(text, want, "hindi")
    for text, want in load_log_misses():
        add(text, want, "conversation_log")

    return cases


def main() -> int:
    cases = build_dataset()
    print(f"Dataset: {len(cases)} utterances "
          f"({sum(1 for c in cases if c['is_miss'])} regex-miss with tool label)")

    misses = [c for c in cases if c["regex"] is None]
    print(f"Regex miss rate on dataset: {len(misses)}/{len(cases)} "
          f"({100 * len(misses) / len(cases):.1f}%)")

    results = []
    correct = 0
    correct_miss = 0
    miss_total = 0
    abstain = 0

    for i, case in enumerate(cases):
        got = classify_hierarchical(case["text"], device="mps", preload=(i == 0))
        want = case["want"]
        hit = got == want
        if got is None:
            abstain += 1
        if hit:
            correct += 1
        if case["regex"] is None:
            miss_total += 1
            if hit:
                correct_miss += 1
        results.append({**case, "laya": got, "correct": hit})

    acc = correct / len(cases) if cases else 0
    miss_acc = correct_miss / miss_total if miss_total else 0

    print(f"\nZero-shot hierarchical Laya accuracy:")
    print(f"  Overall:     {correct}/{len(cases)} = {100 * acc:.1f}%")
    print(f"  Regex misses only: {correct_miss}/{miss_total} = {100 * miss_acc:.1f}%")
    print(f"  Abstained (low confidence): {abstain}")

    out = Path(__file__).parent / "accuracy_results.json"
    summary = {
        "n_cases": len(cases),
        "overall_accuracy": acc,
        "miss_accuracy": miss_acc,
        "abstain_count": abstain,
        "finetune_recommended": acc < 0.70 or miss_acc < 0.70,
        "cases": results,
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
