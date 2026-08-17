"""Spoken movie/series facts over max/omdb_client.py."""
from __future__ import annotations

import re

from max import omdb_client

_TITLE_RES = [
    re.compile(r"\b(?:plot|synopsis|summary) of\s+(.+)$", re.I),
    re.compile(r"\bwho directed\s+(.+)$", re.I),
    re.compile(r"\bwho (?:stars|starred|acts|acted) in\s+(.+)$", re.I),
    re.compile(r"\b(?:imdb )?(?:rating|score) (?:of|for)\s+(.+)$", re.I),
    re.compile(r"\b(?:tell me about|look up|lookup|search(?: for)?)\s+(?:the )?(?:movie|film|series|show|tv show)\s+(.+)$", re.I),
    re.compile(r"\b(?:the )?(?:movie|film|series|show|tv show)\s+(.+)$", re.I),
    re.compile(r"\b(?:when (?:was|did)|what year (?:was|did)|release date (?:of|for))\s+(.+?)(?:\s+(?:come out|released|come out)\b)?$", re.I),
    re.compile(r"\b(?:is|was)\s+(.+?)\s+(?:a |an )?(?:good )?(?:movie|film|series|show)\b", re.I),
    re.compile(r"\b(?:what is|what's)\s+(.+?)\s+about\b", re.I),
]

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_STRIP_TRAIL = re.compile(
    r"\s+(?:movie|film|series|show|tv show|please|right now|\?)\s*$",
    re.I,
)


def extract_title(text: str) -> tuple[str, str | None]:
    stripped = (text or "").strip().strip("?.!")
    year = None
    ymatch = _YEAR_RE.search(stripped)
    if ymatch:
        year = ymatch.group(1)
        stripped = (stripped[: ymatch.start()] + stripped[ymatch.end() :]).strip()
    title = ""
    for pattern in _TITLE_RES:
        match = pattern.search(stripped)
        if match:
            title = match.group(1).strip(" \"'.")
            break
    if not title:
        title = re.sub(
            r"^(?:hey max[,!.]?\s*)?(?:can you |please )?(?:tell me |look up |search )?",
            "",
            stripped,
            flags=re.I,
        ).strip()
    title = _STRIP_TRAIL.sub("", title).strip(" \"'.")
    title = re.sub(r"^(?:the|a|an)\s+", "", title, flags=re.I).strip()
    return title, year


def describe(user_text: str) -> str:
    if not omdb_client.configured():
        return "OMDb is not configured — add OMDB_API_KEY to max/.env."
    title, year = extract_title(user_text)
    if not title or len(title) < 2:
        return "Which movie or series?"
    info = None
    try:
        hits = omdb_client.search(title)
    except omdb_client.OmdbError:
        hits = []
    if hits:
        lowered = title.lower()
        exact = [h for h in hits if h["title"].lower() == lowered]
        starts = [h for h in hits if h["title"].lower().startswith(lowered)]
        contains = [
            h for h in hits
            if lowered in h["title"].lower() or h["title"].lower() in lowered
        ]
        pool = starts or exact or contains or hits
        if year:
            yeared = [h for h in pool if year in (h.get("year") or "")]
            if yeared:
                pool = yeared
        prefer_movie = bool(re.search(r"\b(movie|film|directed)\b", user_text, re.I))
        def _rank(h: dict) -> tuple:
            ymatch = re.search(r"(?:19|20)\d{2}", h.get("year") or "")
            y = int(ymatch.group(0)) if ymatch else 0
            movie = 1 if (h.get("type") or "") == "movie" else 0
            return (movie if prefer_movie else 0, y)
        pool = sorted(pool, key=_rank, reverse=True)
        try:
            info = omdb_client.by_id(pool[0]["imdb_id"]) if pool[0].get("imdb_id") else None
        except omdb_client.OmdbError:
            info = None
    if info is None:
        try:
            info = omdb_client.by_title(title, year)
        except omdb_client.OmdbError as e:
            print(f"[omdb] failed: {e}", flush=True)
            return f'OMDb has nothing for "{title}".'

    kind = info.get("type") or "title"
    year_s = f" ({info['year']})" if info.get("year") else ""
    bits = [f"{info['title']}{year_s} is a {info['genre'] or kind}."]
    if info.get("director") and info["director"].lower() != "n/a":
        bits.append(f"Directed by {info['director']}.")
    if info.get("actors"):
        bits.append(f"Stars {info['actors']}.")
    if info.get("imdb_rating"):
        bits.append(f"IMDb {info['imdb_rating']}/10.")
    if info.get("runtime"):
        bits.append(f"{info['runtime']}.")
    if info.get("plot"):
        bits.append(info["plot"])
    if info.get("total_seasons"):
        bits.append(f"{info['total_seasons']} seasons.")
    return " ".join(bits)
