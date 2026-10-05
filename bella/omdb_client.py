"""OMDb — movie and series facts. Key in bella/.env (OMDB_API_KEY)."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from bella import env

OMDB_URL = "https://www.omdbapi.com/"
_USER_AGENT = "Bella/1.0 (local personal assistant)"


class OmdbError(Exception):
    pass


def configured() -> bool:
    return bool(env.load_env().get("OMDB_API_KEY", "").strip())


def _key() -> str:
    key = env.load_env().get("OMDB_API_KEY", "").strip()
    if not key:
        raise OmdbError("OMDB_API_KEY is not set in bella/.env")
    return key


def _get(params: dict) -> dict:
    query = dict(params)
    query["apikey"] = _key()
    query["r"] = "json"
    url = OMDB_URL + "?" + urllib.parse.urlencode(query)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise OmdbError(f"OMDb lookup failed ({e})") from e
    if not isinstance(data, dict):
        raise OmdbError("OMDb returned nothing usable.")
    if str(data.get("Response", "")).lower() == "false":
        raise OmdbError(data.get("Error") or "Not found.")
    return data


def by_title(title: str, year: str | None = None) -> dict:
    params: dict = {"t": title.strip(), "plot": "short"}
    if year:
        params["y"] = year
    return _normalize(_get(params))


def by_id(imdb_id: str) -> dict:
    return _normalize(_get({"i": imdb_id.strip(), "plot": "short"}))


def search(title: str) -> list[dict]:
    data = _get({"s": title.strip()})
    rows = []
    for item in data.get("Search") or []:
        if not isinstance(item, dict):
            continue
        name = (item.get("Title") or "").strip()
        if not name:
            continue
        rows.append({
            "title": name,
            "year": (item.get("Year") or "").strip(),
            "type": (item.get("Type") or "").strip(),
            "imdb_id": (item.get("imdbID") or "").strip(),
        })
    return rows


def _na(value: str | None) -> str:
    text = (value or "").strip()
    return "" if not text or text.upper() == "N/A" else text


def _normalize(data: dict) -> dict:
    ratings = {}
    for item in data.get("Ratings") or []:
        if isinstance(item, dict) and item.get("Source") and item.get("Value"):
            ratings[item["Source"]] = item["Value"]
    return {
        "title": _na(data.get("Title")),
        "year": _na(data.get("Year")),
        "rated": _na(data.get("Rated")),
        "released": _na(data.get("Released")),
        "runtime": _na(data.get("Runtime")),
        "genre": _na(data.get("Genre")),
        "director": _na(data.get("Director")),
        "writer": _na(data.get("Writer")),
        "actors": _na(data.get("Actors")),
        "plot": _na(data.get("Plot")),
        "language": _na(data.get("Language")),
        "country": _na(data.get("Country")),
        "awards": _na(data.get("Awards")),
        "imdb_rating": _na(data.get("imdbRating")),
        "imdb_votes": _na(data.get("imdbVotes")),
        "imdb_id": _na(data.get("imdbID")),
        "type": _na(data.get("Type")),
        "total_seasons": _na(data.get("totalSeasons")),
        "ratings": ratings,
        "box_office": _na(data.get("BoxOffice")),
    }
