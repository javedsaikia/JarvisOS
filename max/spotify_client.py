"""Minimal Spotify Web API client — Client Credentials flow, search only.
No user login/OAuth needed since this never touches personal data, just
the public catalog: resolving a name ("Metallica", "Bohemian Rhapsody") to
a spotify:track:... URI. Stdlib only, same pattern as sarvam_client.py.
Playback itself happens locally via max/tools/spotify.py's AppleScript
control — this module only does the name -> URI lookup that Spotify's
AppleScript dictionary has no equivalent for.
"""
import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from max.env import load_env

TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"


class SpotifyClientError(Exception):
    pass


# Client Credentials tokens last ~1hr; cached at module scope (with a 60s
# safety margin) so a run of searches in one process doesn't re-auth every
# time — same tradeoff as sarvam_client's per-call simplicity, just with a
# cache since this token is reused across many search calls, not one
# request per token the way Sarvam's per-call auth header is.
_token_cache: dict = {"access_token": None, "expires_at": 0.0}


def _credentials() -> tuple[str, str]:
    env = load_env()
    client_id = env.get("SPOTIFY_CLIENT_ID", "")
    client_secret = env.get("SPOTIFY_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise SpotifyClientError("SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET not set in max/.env")
    return client_id, client_secret


def _get_access_token() -> str:
    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["access_token"]

    client_id, client_secret = _credentials()
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    payload = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
    req = urllib.request.Request(
        TOKEN_URL,
        data=payload,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SpotifyClientError(f"Spotify auth error {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise SpotifyClientError(f"Could not reach Spotify auth ({e})") from e

    token = data["access_token"]
    _token_cache["access_token"] = token
    _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)
    return token


# Covers and karaoke routinely outrank the original when we take
# Spotify's first hit. Seen live: "High Hopes by Pink Floyd" started a
# karaoke version; "High Hopes" without an artist started Panic! At The
# Disco even after the user had just named Pink Floyd.
_JUNK_TITLE = re.compile(
    r"\b(karaoke|tribute|originally performed|made famous|sound-?alike|"
    r"cover version|instrumental tribute)\b",
    re.IGNORECASE,
)


def _split_artist(query: str) -> tuple[str, str | None]:
    q = (query or "").strip()
    match = re.search(r"\s+by\s+(.+)$", q, re.IGNORECASE)
    if not match:
        return q, None
    title = q[: match.start()].strip(" \"'")
    artist = match.group(1).strip(" \"'")
    if not title or not artist:
        return q, None
    return title, artist


def _search_items(query: str, token: str, limit: int = 10) -> list[dict]:
    params = urllib.parse.urlencode({"q": query, "type": "track", "limit": limit})
    req = urllib.request.Request(
        f"{SEARCH_URL}?{params}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SpotifyClientError(f"Spotify search error {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise SpotifyClientError(f"Could not reach Spotify search ({e})") from e
    return data.get("tracks", {}).get("items", []) or []


def _artists(track: dict) -> list[str]:
    return [a["name"] for a in track.get("artists", []) if a.get("name")]


def _score_track(track: dict, title: str, artist: str | None, raw_query: str) -> int:
    name = track.get("name") or ""
    artists = _artists(track)
    blob = f"{name} {' '.join(artists)}".lower()
    score = int(track.get("popularity") or 0)
    if artist and any(artist.lower() in a.lower() or a.lower() in artist.lower() for a in artists):
        score += 80
    elif any(len(a) > 2 and a.lower() in raw_query.lower() for a in artists):
        score += 50
    title_l = title.lower()
    name_l = name.lower()
    if title_l and (title_l in name_l or name_l in title_l):
        score += 20
    if _JUNK_TITLE.search(blob):
        score -= 100
    return score


def search_track(query: str) -> dict | None:
    """Returns {"uri", "name", "artist"} for the best match, or None.

    Takes the top handful of hits and prefers the named artist and the
    official recording over whatever Spotify ranked first.
    """
    query = (query or "").strip()
    if not query:
        return None
    token = _get_access_token()
    title, artist = _split_artist(query)
    searches = []
    if artist:
        searches.append(f'track:"{title}" artist:"{artist}"')
        searches.append(f"{title} {artist}")
    searches.append(query)

    items: list[dict] = []
    seen: set[str] = set()
    for q in searches:
        for track in _search_items(q, token):
            uri = track.get("uri")
            if not uri or uri in seen:
                continue
            seen.add(uri)
            items.append(track)
        if items:
            break

    if not items:
        return None
    track = max(items, key=lambda t: _score_track(t, title, artist, query))
    artist_name = ", ".join(_artists(track))
    return {"uri": track["uri"], "name": track["name"], "artist": artist_name}
