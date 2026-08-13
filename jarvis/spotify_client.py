"""Minimal Spotify Web API client — Client Credentials flow, search only.
No user login/OAuth needed since this never touches personal data, just
the public catalog: resolving a name ("Metallica", "Bohemian Rhapsody") to
a spotify:track:... URI. Stdlib only, same pattern as sarvam_client.py.
Playback itself happens locally via jarvis/tools/spotify.py's AppleScript
control — this module only does the name -> URI lookup that Spotify's
AppleScript dictionary has no equivalent for.
"""
import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from jarvis.env import load_env

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
        raise SpotifyClientError("SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET not set in jarvis/.env")
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


def search_track(query: str) -> dict | None:
    """Returns {"uri", "name", "artist"} for the best match, or None."""
    token = _get_access_token()
    params = urllib.parse.urlencode({"q": query, "type": "track", "limit": 1})
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

    items = data.get("tracks", {}).get("items", [])
    if not items:
        return None
    track = items[0]
    artist = ", ".join(a["name"] for a in track.get("artists", []))
    return {"uri": track["uri"], "name": track["name"], "artist": artist}
