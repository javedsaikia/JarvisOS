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
from jarvis.config import load_config

# Remembers the volume level from before a mute, so unmute() can restore
# it rather than guessing a fixed value. Module-level since VoiceLoop
# creates no persistent Spotify object of its own — matches the rest of
# this module's stateless-function style otherwise.
_volume_before_mute: int | None = None


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
        _apply_volume_cap()
        run('tell application "Spotify" to play', app_name="Spotify")
        line = _now_playing_line()
        return f"Playing {line}." if line else "Playing."

    # "stopped" with no current track — nothing for a bare `play` to
    # resume. This used to seed a default playlist here, but that's
    # guessing at content the user never asked for. Explicitly requested:
    # a generic Spotify command should just open the app and wait for an
    # actual song name (play_track), not start playing something random.
    subprocess.run(["open", "-a", "Spotify"], capture_output=True, timeout=10)
    return "Spotify's open — tell me what song to play."


def open_app() -> str:
    """Launch Spotify without touching playback.

    Explicitly required: naming Spotify must never start music by itself —
    the user says what to play. Before this existed, "open spotify" matched
    no pattern at all and fell through to the LLM, which answered from
    imagination (observed live: it announced "Another Brick in the Wall by
    Pink Floyd" was playing when nothing was).
    """
    subprocess.run(["open", "-a", "Spotify"], capture_output=True, timeout=10)
    line = _now_playing_line()
    if line and _player_state() == "paused":
        return f"Spotify's open — {line} is paused. Tell me what to play."
    return "Spotify's open — tell me what song to play."


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

    _apply_volume_cap()
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


# Remembers whether the voice loop itself paused Spotify for a
# conversation turn, so resume_after_conversation() only restarts playback
# it actually stopped — same "only restore what we changed" pattern as
# _volume_before_mute above. Silent (no spoken confirmation) since this
# runs automatically on every turn, not on a user command.
_was_playing_before_conversation = False


def pause_for_conversation() -> None:
    """Called right before the mic starts recording a turn — so background
    music doesn't bleed into the recording or talk over Orin's reply.
    Deliberately checks _is_running() first (same reasoning as
    now_playing_state()): most turns have nothing to do with Spotify at
    all, and the run() helper's app_name would otherwise auto-launch
    Spotify just to answer an unrelated question.
    """
    global _was_playing_before_conversation
    _was_playing_before_conversation = False
    if not _is_running():
        return
    try:
        if _player_state() == "playing":
            run('tell application "Spotify" to pause', app_name="Spotify")
            _was_playing_before_conversation = True
    except Exception:
        pass  # a Spotify hiccup here should never break the voice loop


def resume_after_conversation() -> None:
    """Called once the turn is fully wrapped up (response spoken, or the
    turn ended early with nothing to say) — resumes playback only if this
    module was the one that paused it.
    """
    global _was_playing_before_conversation
    if not _was_playing_before_conversation:
        return
    _was_playing_before_conversation = False
    # No _is_running() re-check here (unlike pause_for_conversation()) —
    # the flag being True already proves Spotify was running moments ago,
    # in this same turn, when we paused it. Seen live: re-checking here
    # made resume silently no-op under real CPU load (whisper/TTS running
    # concurrently) — _is_running()'s own 5s AppleScript timeout occasionally
    # got exceeded, caught internally, and returned False, skipping the
    # `play` call below entirely and leaving Spotify stuck paused for the
    # rest of the conversation.
    try:
        run('tell application "Spotify" to play', app_name="Spotify")
    except Exception:
        pass


def next_track() -> str:
    run('tell application "Spotify" to next track', app_name="Spotify")
    line = _now_playing_line()
    return f"Skipped to {line}." if line else "Skipped to the next track."


def previous_track() -> str:
    run('tell application "Spotify" to previous track', app_name="Spotify")
    line = _now_playing_line()
    return f"Back to {line}." if line else "Back to the previous track."


_VOLUME_STEP = 20


def _get_volume() -> int:
    result = run('tell application "Spotify" to sound volume', app_name="Spotify")
    try:
        return int(float(result))
    except ValueError:
        return 100


def _set_volume(level: int) -> None:
    level = max(0, min(100, level))
    run(f'tell application "Spotify" to set sound volume to {level}', app_name="Spotify")


_DEFAULT_MAX_AUTO_VOLUME = 75


def _apply_volume_cap() -> None:
    """Caps the volume when Orin itself starts or resumes playback — seen
    live: at volume 100, "Hey Jarvis" stopped registering at all (the mic
    has no echo/noise cancellation, so loud music at full volume drowns out
    the wake word before it ever reaches the pipeline). Only applies when
    Orin starts playback; explicit volume_up/volume_down/mute commands
    are the user's own deliberate choice and stay untouched.
    """
    cap = load_config().get("spotify_max_volume", _DEFAULT_MAX_AUTO_VOLUME)
    if _get_volume() > cap:
        _set_volume(cap)


def mute() -> str:
    """Lowers Spotify to silent so the mic can hear a spoken command
    clearly without music underneath it — distinct from pause(), which
    stops playback outright. Remembers the pre-mute level so unmute()
    restores it rather than guessing a fixed volume back."""
    global _volume_before_mute
    current = _get_volume()
    if current > 0:
        _volume_before_mute = current
    _set_volume(0)
    return "Muted Spotify."


def unmute() -> str:
    restore = _volume_before_mute if _volume_before_mute else 70
    _set_volume(restore)
    return "Unmuted Spotify."


def volume_up() -> str:
    new_level = _get_volume() + _VOLUME_STEP
    _set_volume(new_level)
    return f"Volume up to {min(100, new_level)} percent."


def volume_down() -> str:
    new_level = _get_volume() - _VOLUME_STEP
    _set_volume(new_level)
    return f"Volume down to {max(0, new_level)} percent."
