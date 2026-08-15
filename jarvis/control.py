"""Live switches shared between the front ends and the voice loop.

The voice loop is a separate OS process from the bridge, so a button in
the web UI cannot simply set an attribute on it. Signals already carry the
two momentary commands (SIGUSR1 wake, SIGUSR2 interrupt), but a *state* —
"the microphone is off until further notice" — needs somewhere to live
that survives a restart of either side and can be read by anything.

So: one small JSON file, written atomically, polled cheaply by mtime. It
is deliberately plain text at a fixed path, because the thing it controls
is the microphone, and "is it actually off?" should be answerable with
`cat` rather than by trusting a UI.

    {"mic_muted": true, "changed_at": 1786...}
"""
import json
import os
import time
from pathlib import Path

CONTROL_PATH = Path("/tmp/orin_control.json")

DEFAULTS = {"mic_muted": False}


def read() -> dict:
    state = dict(DEFAULTS)
    try:
        state.update(json.loads(CONTROL_PATH.read_text()))
    except (OSError, json.JSONDecodeError):
        pass
    return state


def write(**changes) -> dict:
    state = read()
    state.update(changes)
    state["changed_at"] = time.time()
    tmp = CONTROL_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, CONTROL_PATH)
    except OSError:
        pass
    return state


def mic_muted() -> bool:
    return bool(read().get("mic_muted"))


def set_mic_muted(muted: bool) -> dict:
    return write(mic_muted=bool(muted))


class Watcher:
    """Polls the control file, cheaply.

    Called once per audio frame (~80ms), so it stats the file rather than
    parsing it, and only re-reads when the mtime actually changes.
    """

    def __init__(self):
        self._mtime = -1.0
        self._state = dict(DEFAULTS)
        self.refresh()

    def refresh(self) -> dict:
        try:
            mtime = CONTROL_PATH.stat().st_mtime
        except OSError:
            self._state = dict(DEFAULTS)
            self._mtime = -1.0
            return self._state
        if mtime != self._mtime:
            self._mtime = mtime
            self._state = read()
        return self._state

    @property
    def mic_muted(self) -> bool:
        return bool(self.refresh().get("mic_muted"))
