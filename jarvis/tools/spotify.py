"""Local Spotify playback control via AppleScript (macOS Spotify.app).
Pure local automation, no credentials, no network calls of our own — same
pattern as calendar.py/notes.py. This is playback CONTROL only
(play/pause/skip); "what's playing" status still goes through
cli.handle_spotify's Claude Code handoff, since that reads the account's
active session across ALL devices (phone, speaker, etc.), while
AppleScript's `current track` only ever reflects this Mac's local app.
"""
import subprocess
import time

from jarvis import spotify_client
from jarvis.applescript import run

# Spotify's official "Today's Top Hits" playlist — fallback starting point
# for a bare "play something" when nothing is loaded yet (e.g. right after
# install/login, player state is "stopped" and there's no current track to
# resume). Verified live: AppleScript's own `open location` inside `tell
# application "Spotify"` silently no-ops on this Mac; the macOS `open`
# command (LaunchServices) is what actually hands the spotify: URI off to
# the app and gets it playing.
_DEFAULT_PLAYLIST_URI = "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M"


def _player_state() -> str:
    return run('tell application "Spotify" to player state', app_name="Spotify")


def _now_playing_line() -> str:
    script = """
tell application "Spotify"
    return (name of current track) & " by " & (artist of current track)
end tell
"""
    try:
        return run(script, app_name="Spotify")
    except Exception:
        return ""


def play() -> str:
    state = _player_state()
    if state == "playing":
        line = _now_playing_line()
        return f"Already playing {line}." if line else "Already playing."

    if state == "paused":
        run('tell application "Spotify" to play', app_name="Spotify")
    else:
        # "stopped" with no current track — nothing for a bare `play` to
        # resume, so seed a default playlist instead.
        subprocess.run(["open", _DEFAULT_PLAYLIST_URI], capture_output=True, timeout=10)
        for _ in range(8):
            time.sleep(1)
            if _player_state() == "playing":
                break

    line = _now_playing_line()
    return f"Playing {line}." if line else "Playing."


def play_track(query: str) -> str:
    """Resolves a name (e.g. "Metallica", "Nothing Else Matters") to a
    real track via the Spotify Web API (jarvis/spotify_client.py — the
    AppleScript dictionary has no search of its own), then hands that URI
    to the same `open` mechanism play() uses for the default playlist.
    """
    try:
        match = spotify_client.search_track(query)
    except spotify_client.SpotifyClientError as e:
        return f"(Spotify search unavailable: {e})"
    if not match:
        return f'Couldn\'t find "{query}" on Spotify.'

    subprocess.run(["open", match["uri"]], capture_output=True, timeout=10)
    for _ in range(8):
        time.sleep(1)
        if _player_state() == "playing":
            break
    return f'Playing {match["name"]} by {match["artist"]}.'


def _is_running() -> bool:
    script = 'tell application "System Events" to (name of processes) contains "Spotify"'
    try:
        return run(script, timeout=5) == "true"
    except Exception:
        return False


_STATE_SCRIPT = """
tell application "Spotify"
    if player state is playing or player state is paused then
        set trackName to name of current track
        set trackArtist to artist of current track
        set trackArt to artwork url of current track
        set trackDuration to duration of current track
        set trackPosition to player position
        return (player state as string) & "|||" & trackName & "|||" & trackArtist & "|||" & trackArt & "|||" & (trackDuration as string) & "|||" & (trackPosition as string)
    end if
    return "stopped"
end tell
"""


def now_playing_state() -> dict:
    """Combined state read for the web UI's Now Playing widget — one
    AppleScript round-trip bundling everything the frontend needs, polled
    periodically by bridge.py. Deliberately checks _is_running() first:
    the `run()` helper's app_name auto-launch would otherwise launch
    Spotify just because a browser tab is open and polling, which is not
    something a passive status widget should ever cause.
    """
    if not _is_running():
        return {"active": False}
    try:
        result = run(_STATE_SCRIPT, timeout=10)
    except Exception:
        return {"active": False}

    if result == "stopped" or not result:
        return {"active": False}

    parts = result.split("|||")
    if len(parts) != 6:
        return {"active": False}
    state, name, artist, artwork, duration_ms, position_s = parts
    try:
        duration = float(duration_ms) / 1000.0
        position = float(position_s)
    except ValueError:
        return {"active": False}

    return {
        "active": True,
        "playing": state == "playing",
        "track": name,
        "artist": artist,
        "artwork_url": artwork,
        "duration": duration,
        "position": position,
    }


def pause() -> str:
    run('tell application "Spotify" to pause', app_name="Spotify")
    return "Paused."


def next_track() -> str:
    run('tell application "Spotify" to next track', app_name="Spotify")
    line = _now_playing_line()
    return f"Skipped to {line}." if line else "Skipped to the next track."


def previous_track() -> str:
    run('tell application "Spotify" to previous track', app_name="Spotify")
    line = _now_playing_line()
    return f"Back to {line}." if line else "Back to the previous track."
