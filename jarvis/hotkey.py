"""Push-to-talk: hold a modifier combo to talk, release to send.

The idea is borrowed from Clicky (https://www.heyclicky.com,
github.com/farzaa/clicky), which invokes with a hotkey rather than a wake
word. That choice solves a real problem here rather than being a nicety.
Everything a wake word depends on is probabilistic and was observed
failing repeatedly: the wake detector has to hear the wake phrase over room
noise and music, the VAD then has to agree speech happened (measured
scoring 0.03 on audio peaking at RMS 3553), and the listen window has to
still be open when the user actually starts talking. A key being held is
none of those things — it is a fact, so the recording boundaries are
exact and nothing can be missed, mis-scored, or timed out.

It sits alongside the wake word rather than replacing it: hands-free still
works, and this is the reliable path when it matters.

Requires macOS Input Monitoring permission (System Settings > Privacy &
Security > Input Monitoring). Without it pynput's listener starts happily
and simply never receives an event, so `saw_any_event` is exposed for
callers to detect and report that rather than looking silently broken —
the same class of failure as the mic returning digital silence.
"""
import threading

from pynput import keyboard

# Control+Option, matching Clicky's default. Both are modifiers, so
# holding them cannot type anything into whatever app has focus.
DEFAULT_COMBO = ("ctrl", "alt")

_KEY_ALIASES = {
    "ctrl": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
    "alt": {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr},
    "shift": {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r},
    "cmd": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
}


class PushToTalk:
    """Tracks whether the configured modifier combo is currently held."""

    def __init__(self, combo=DEFAULT_COMBO, on_press=None):
        self.combo = tuple(combo)
        self._targets = [_KEY_ALIASES.get(name, set()) for name in self.combo]
        self._down: set = set()
        self._on_press = on_press
        self._listener: keyboard.Listener | None = None
        self.held = threading.Event()
        self.saw_any_event = False

    def _combo_held(self) -> bool:
        return all(any(k in self._down for k in target) for target in self._targets if target)

    def _handle_press(self, key) -> None:
        self.saw_any_event = True
        self._down.add(key)
        if self._combo_held() and not self.held.is_set():
            self.held.set()
            if self._on_press:
                # Never let a callback exception kill the listener thread and
                # silently disable the hotkey for the rest of the session.
                try:
                    self._on_press()
                except Exception:
                    pass

    def _handle_release(self, key) -> None:
        self.saw_any_event = True
        self._down.discard(key)
        if not self._combo_held():
            self.held.clear()

    def start(self) -> bool:
        try:
            self._listener = keyboard.Listener(
                on_press=self._handle_press, on_release=self._handle_release
            )
            self._listener.daemon = True
            self._listener.start()
            return True
        except Exception as e:
            print(f"  (push-to-talk unavailable: {type(e).__name__}: {e})")
            self._listener = None
            return False

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def describe(self) -> str:
        return "+".join(self.combo)
