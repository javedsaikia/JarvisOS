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
import time

from pynput import keyboard

# Warms pyobjc's lazy symbol cache for HIServices.AXIsProcessTrusted
# *before* any listener thread can touch it. pynput's darwin backend calls
# that function from inside each Listener's own background thread the
# moment it starts, and pyobjc resolves it lazily on first use with a
# plain check-then-pop on a shared dict (objc/_lazyimport.py) — not
# thread-safe. With one listener (the original push-to-talk-only setup)
# that race never had a second thread to lose to. Adding the wake hotkey
# listener gave it one: both start within microseconds of each other in
# VoiceLoop.__init__, occasionally both hit the lazy resolution at once,
# and the loser's `funcmap.pop(name)` raises KeyError on an already-popped
# key — printed as "Exception in thread Thread-1" with no indication
# whatsoever that a hotkey was the cause, and that thread's listener is
# simply dead from then on. Calling the function once here, single-
# threaded, before either Listener starts, means both threads find it
# already resolved and never touch the race.
try:
    import HIServices as _HIServices

    _HIServices.AXIsProcessTrusted()
except Exception:
    pass

# Control+Option, matching Clicky's default. Both are modifiers, so
# holding them cannot type anything into whatever app has focus.
DEFAULT_COMBO = ("ctrl", "alt")

_KEY_ALIASES = {
    "ctrl": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
    "alt": {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr},
    "shift": {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r},
    "cmd": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
    # Sided variants, so a shortcut can use the right-hand modifier that
    # nothing else on macOS claims.
    "ctrl_l": {keyboard.Key.ctrl_l}, "ctrl_r": {keyboard.Key.ctrl_r},
    "alt_l": {keyboard.Key.alt_l}, "alt_r": {keyboard.Key.alt_r},
    "cmd_l": {keyboard.Key.cmd_l}, "cmd_r": {keyboard.Key.cmd_r},
    "shift_l": {keyboard.Key.shift_l}, "shift_r": {keyboard.Key.shift_r},
    "space": {keyboard.Key.space},
    "esc": {keyboard.Key.esc},
    "tab": {keyboard.Key.tab},
    "enter": {keyboard.Key.enter},
}
# Function keys and plain letters/digits, so a combo can name any key.
for _n in range(1, 21):
    _KEY_ALIASES[f"f{_n}"] = {getattr(keyboard.Key, f"f{_n}")}


def _resolve(name: str) -> set:
    """Keys matching a config name. Falls back to a literal character, so
    "o" in a combo means the O key."""
    name = name.strip().lower()
    if name in _KEY_ALIASES:
        return _KEY_ALIASES[name]
    if len(name) == 1:
        return {keyboard.KeyCode.from_char(name)}
    return set()


def _key_matches(key, targets: set) -> bool:
    """Compare a pressed key against a target set.

    Character keys need care: with a modifier held, macOS reports the
    *modified* character (Option+O arrives as "ø"), and pynput exposes the
    unmodified one as `key.vk`-adjacent info rather than `.char`. Matching
    on the raw char alone silently breaks every letter-based shortcut, so
    both the char and the key object are checked.
    """
    if key in targets:
        return True
    char = getattr(key, "char", None)
    if char:
        for target in targets:
            if getattr(target, "char", None) == char:
                return True
    return False


class PushToTalk:
    """Tracks whether the configured modifier combo is currently held."""

    def __init__(self, combo=DEFAULT_COMBO, on_press=None):
        self.combo = tuple(combo)
        self._targets = [_resolve(name) for name in self.combo]
        self._down: set = set()
        self._on_press = on_press
        self._listener: keyboard.Listener | None = None
        self.held = threading.Event()
        self.saw_any_event = False

    def _combo_held(self) -> bool:
        return all(
            any(_key_matches(k, target) for k in self._down)
            for target in self._targets if target
        )

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


# Double-tapping a modifier is how macOS itself starts dictation, and it
# is the one gesture that cannot collide with an application shortcut: a
# bare modifier press does nothing in any app, so nothing is stolen from
# whatever has focus. Right Command by default — the least-used modifier
# on an Apple keyboard, and not part of push-to-talk's Control+Option.
DEFAULT_WAKE_KEY = "cmd_r"
DEFAULT_WAKE_COMBO = ("ctrl", "alt", "space")
DOUBLE_TAP_SECONDS = 0.45


class WakeHotkey:
    """Tap a shortcut to wake Orin, instead of holding one to talk.

    Two modes, because they suit different hands:

    - "double_tap" (default): tap a modifier twice, like macOS dictation.
      Collides with nothing, since a lone modifier means nothing to apps.
    - "combo": a chord such as Control+Option+Space. Note that pynput
      observes keystrokes without consuming them, so a chord also reaches
      whatever app is focused — which is why the default isn't one.

    Runs its own listener rather than sharing PushToTalk's. Keeping them
    independent means a broken or unpermitted wake hotkey can never take
    push-to-talk down with it.
    """

    def __init__(self, mode: str = "double_tap", key: str = DEFAULT_WAKE_KEY,
                 combo=DEFAULT_WAKE_COMBO, on_wake=None):
        self.mode = mode
        self.key_name = key
        self.combo = tuple(combo)
        self._targets = _resolve(key)
        self._combo_targets = [_resolve(name) for name in self.combo]
        self._on_wake = on_wake
        self._listener: keyboard.Listener | None = None
        self._down: set = set()
        self._last_tap = 0.0
        self._tap_armed = False
        self._combo_fired = False
        self.saw_any_event = False

    # --- double tap ---

    def _is_target(self, key) -> bool:
        return _key_matches(key, self._targets)

    def _handle_double_tap_press(self, key) -> None:
        if self._is_target(key):
            self._tap_armed = True
            return
        # Any other key during the gesture cancels it: a modifier held as
        # part of a real shortcut (Cmd+S) must not count as a tap.
        self._tap_armed = False
        self._last_tap = 0.0

    def _handle_double_tap_release(self, key) -> None:
        if not self._is_target(key) or not self._tap_armed:
            return
        self._tap_armed = False
        now = time.monotonic()
        if now - self._last_tap <= DOUBLE_TAP_SECONDS:
            self._last_tap = 0.0
            self._fire()
        else:
            self._last_tap = now

    # --- chord ---

    def _combo_held(self) -> bool:
        return all(
            any(_key_matches(k, target) for k in self._down)
            for target in self._combo_targets if target
        )

    def _handle_combo_press(self, key) -> None:
        self._down.add(key)
        if self._combo_held() and not self._combo_fired:
            self._combo_fired = True
            self._fire()

    def _handle_combo_release(self, key) -> None:
        self._down.discard(key)
        if not self._combo_held():
            self._combo_fired = False

    # --- shared ---

    def _fire(self) -> None:
        if not self._on_wake:
            return
        try:
            self._on_wake()
        except Exception:
            # Same reasoning as PushToTalk: a callback that raises must
            # not kill the listener thread and silently disable the key
            # for the rest of the session.
            pass

    def _on_press(self, key) -> None:
        self.saw_any_event = True
        if self.mode == "combo":
            self._handle_combo_press(key)
        else:
            self._handle_double_tap_press(key)

    def _on_release(self, key) -> None:
        self.saw_any_event = True
        if self.mode == "combo":
            self._handle_combo_release(key)
        else:
            self._handle_double_tap_release(key)

    def start(self) -> bool:
        try:
            self._listener = keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release
            )
            self._listener.daemon = True
            self._listener.start()
            return True
        except Exception as e:
            print(f"  (wake hotkey unavailable: {type(e).__name__}: {e})")
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
        if self.mode == "combo":
            return "+".join(self.combo)
        pretty = {
            "cmd_r": "Right Command", "cmd_l": "Left Command", "cmd": "Command",
            "ctrl_r": "Right Control", "ctrl_l": "Left Control", "ctrl": "Control",
            "alt_r": "Right Option", "alt_l": "Left Option", "alt": "Option",
            "shift_r": "Right Shift", "shift_l": "Left Shift", "shift": "Shift",
        }.get(self.key_name, self.key_name)
        return f"double-tap {pretty}"
