"""Latest headlines via public RSS. Stdlib only.

The model must never invent the news. This module fetches real items;
callers pass those strings into the model (or speak the titles if the
model is unavailable).
"""
from __future__ import annotations

import html
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

_USER_AGENT = "Max/1.0 (personal assistant; +https://github.com/javedsaikia/JarvisOS)"

# Google News for "today"; BBC as a second source if Google is empty.
_FEEDS = {
    "ai": [
        "https://news.google.com/rss/search?q=artificial+intelligence+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
    ],
    "tech": [
        "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en",
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
    ],
    "politics": [
        "https://news.google.com/rss/headlines/section/topic/NATION?hl=en-IN&gl=IN&ceid=IN:en",
        "https://feeds.bbci.co.uk/news/politics/rss.xml",
    ],
    "finance": [
        "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
    ],
    "world": [
        "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
    ],
}


class NewsError(Exception):
    pass


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_xml(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError) as e:
        raise NewsError(f"Could not reach news feed ({e})") from e


def _parse_items(xml_text: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise NewsError(f"News feed was not valid XML ({e})") from e
    items = []
    for node in root.iter():
        tag = node.tag.split("}")[-1].lower()
        if tag not in {"item", "entry"}:
            continue
        def child(name: str) -> str:
            for c in node:
                if c.tag.split("}")[-1].lower() == name:
                    return (c.text or "").strip()
            return ""
        title = _strip_html(child("title"))
        if not title:
            continue
        summary = _strip_html(child("description") or child("summary"))
        # Google News puts source after " - " in the title.
        source = child("source")
        if not source and " - " in title:
            title, _, source = title.rpartition(" - ")
            title, source = title.strip(), source.strip()
        items.append({
            "title": title,
            "summary": summary[:280],
            "source": source,
            "published": child("pubDate") or child("updated") or child("published"),
        })
    return items


def fetch(topic: str, limit: int = 8) -> list[dict]:
    """Return up to `limit` headlines for a topic key in _FEEDS."""
    urls = _FEEDS.get(topic) or _FEEDS["world"]
    last_error: Exception | None = None
    seen: set[str] = set()
    out: list[dict] = []
    for url in urls:
        try:
            items = _parse_items(_fetch_xml(url))
        except NewsError as e:
            last_error = e
            continue
        for item in items:
            key = item["title"].lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= limit:
                return out
    if not out and last_error:
        raise last_error
    return out
