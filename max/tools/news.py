"""Spoken news briefing over max/news_client.py."""
from max import news_client

TOPICS = {
    "ai": ("AI", ("ai", "artificial intelligence", "machine learning", "llm", "chatgpt", "openai")),
    "tech": ("technology", ("tech", "technology", "gadget", "software", "silicon")),
    "politics": ("politics", ("politic", "election", "government", "parliament", "minister")),
    "finance": ("finance", ("finance", "financial", "market", "stock", "economy", "business")),
    "world": ("the world", ("world", "globe", "global", "international", "across the globe")),
}


def detect_topics(text: str) -> list[str]:
    """Which briefings to fetch. Empty request → world + tech + finance."""
    lowered = (text or "").lower()
    hit = [key for key, (_, words) in TOPICS.items() if any(w in lowered for w in words)]
    if hit:
        return hit
    if any(w in lowered for w in ("news", "headline", "what's happening", "whats happening")):
        return ["world", "tech", "finance"]
    return ["world"]


def detect_topic(text: str) -> str:
    topics = detect_topics(text)
    return topics[0] if topics else "world"


def briefing_block(topics: list[str], per_topic: int = 6) -> tuple[str, list[dict]]:
    """Plain text the model (or a fallback speaker) can use. Raises NewsError."""
    chunks = []
    all_items: list[dict] = []
    for topic in topics:
        label = TOPICS.get(topic, (topic, ()))[0]
        items = news_client.fetch(topic, limit=per_topic)
        all_items.extend(items)
        lines = [f"{label}:"]
        for i, item in enumerate(items, 1):
            src = f" ({item['source']})" if item.get("source") else ""
            extra = f" — {item['summary']}" if item.get("summary") else ""
            lines.append(f"{i}. {item['title']}{src}{extra}")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks), all_items


def speak_titles(topics: list[str], items: list[dict]) -> str:
    """If the model cannot run, speak the first few titles."""
    if not items:
        return "I couldn't pull any headlines just now."
    labels = [TOPICS[t][0] for t in topics if t in TOPICS] or ["the news"]
    heads = [it["title"] for it in items[:5]]
    spoken = "; ".join(heads)
    return f"Here is the latest on {', '.join(labels)}. {spoken}."
