"""Push-to-talk and wake hotkeys.

Hold Command+Option to talk, or double-tap Right Command to wake. Both
are modifier-only gestures: they type nothing into the focused app.

macOS does not deliver those as KeyDown/KeyUp. They arrive as
kCGEventFlagsChanged, and the reliable signal is the flag mask (Control
bit, Option bit, Command bit) — not a key identity. pynput's listener
starts fine and then misses the real keys; injected KeyDown events still
work, which is how this looked tested and then did nothing on the
keyboard. One Quartz tap serves both shortcuts so they cannot race.
"""
import threading
import time

from Quartz import (
    CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource,
    CFRunLoopGetCurrent,
    CFRunLoopRunInMode,
    CFRunLoopStop,
    CGEventGetFlags,
    CGEventGetIntegerValueField,
    CGEventMaskBit,
    CGEventTapCreate,
    CGEventTapEnable,
    kCFRunLoopDefaultMode,
    kCFRunLoopRunTimedOut,
    kCGEventFlagMaskAlternate,
    kCGEventFlagMaskCommand,
    kCGEventFlagMaskControl,
    kCGEventFlagMaskShift,
    kCGEventFlagsChanged,
    kCGEventKeyDown,
    kCGEventKeyUp,
    kCGEventTapOptionListenOnly,
    kCGHeadInsertEventTap,
    kCGKeyboardEventKeycode,
    kCGSessionEventTap,
)

try:
    from Quartz import CGPreflightListenEventAccess, CGRequestListenEventAccess
except ImportError:  # older macOS
    CGPreflightListenEventAccess = None
    CGRequestListenEventAccess = None

try:
    from Quartz import kCGEventSystemDefined
except ImportError:
    kCGEventSystemDefined = 14

try:
    from AppKit import NSEvent
except ImportError:
    NSEvent = None

# Hardware media keys (HIToolbox) and NX_KEYTYPE_* for NSSystemDefined.
_VK_VOLUME_UP = 0x48
_VK_VOLUME_DOWN = 0x49
_NX_KEYTYPE_SOUND_UP = 0
_NX_KEYTYPE_SOUND_DOWN = 1
_NX_SUBTYPE_AUX = 8
_NX_KEY_DOWN = 0x0A

DEFAULT_COMBO = ("cmd", "alt")
DEFAULT_WAKE_KEY = "cmd_r"
DEFAULT_WAKE_COMBO = ("ctrl", "alt", "space")
DOUBLE_TAP_SECONDS = 0.45

# Hardware keycodes (Carbon / HIToolbox). FlagsChanged events carry these
# even when pynput would report the key as None or KeyCode(0).
_VK = {
    "ctrl": {0x3B, 0x3E},
    "ctrl_l": {0x3B},
    "ctrl_r": {0x3E},
    "alt": {0x3A, 0x3D},
    "alt_l": {0x3A},
    "alt_r": {0x3D},
    "cmd": {0x37, 0x36},
    "cmd_l": {0x37},
    "cmd_r": {0x36},
    "shift": {0x38, 0x3C},
    "shift_l": {0x38},
    "shift_r": {0x3C},
    "space": {0x31},
    "esc": {0x35},
    "tab": {0x30},
    "enter": {0x24},
}

_FLAG = {
    "ctrl": kCGEventFlagMaskControl,
    "ctrl_l": kCGEventFlagMaskControl,
    "ctrl_r": kCGEventFlagMaskControl,
    "alt": kCGEventFlagMaskAlternate,
    "alt_l": kCGEventFlagMaskAlternate,
    "alt_r": kCGEventFlagMaskAlternate,
    "cmd": kCGEventFlagMaskCommand,
    "cmd_l": kCGEventFlagMaskCommand,
    "cmd_r": kCGEventFlagMaskCommand,
    "shift": kCGEventFlagMaskShift,
    "shift_l": kCGEventFlagMaskShift,
    "shift_r": kCGEventFlagMaskShift,
}

# Bits we care about when deciding "only this modifier changed".
_MOD_BITS = (
    kCGEventFlagMaskControl
    | kCGEventFlagMaskAlternate
    | kCGEventFlagMaskCommand
    | kCGEventFlagMaskShift
)


def _flag_bits(names) -> int:
    bits = 0
    for name in names:
        bits |= _FLAG.get(name.strip().lower(), 0)
    return bits


class _FlagTap:
    """One listen-only session tap. Shared by PushToTalk and WakeHotkey."""

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: list = []
        self._thread: threading.Thread | None = None
        self._loop = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self.saw_any_event = False
        self.flags = 0
        self._logged_first = False

    def subscribe(self, callback) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._ready.clear()
                self._thread = threading.Thread(target=self._run, name="max-hotkey", daemon=True)
                self._thread.start()
        if not self._ready.wait(timeout=2.0):
            print("  (hotkey tap did not come up — Input Monitoring may be missing)")

    def unsubscribe(self, callback) -> None:
        with self._lock:
            self._subscribers = [c for c in self._subscribers if c is not callback]
            empty = not self._subscribers
        if empty:
            self._stop.set()
            loop = self._loop
            if loop is not None:
                try:
                    CFRunLoopStop(loop)
                except Exception:
                    pass

    def _run(self) -> None:
        mask = (
            CGEventMaskBit(kCGEventFlagsChanged)
            | CGEventMaskBit(kCGEventKeyDown)
            | CGEventMaskBit(kCGEventKeyUp)
            | CGEventMaskBit(int(kCGEventSystemDefined))
        )
        if CGPreflightListenEventAccess is not None:
            allowed = bool(CGPreflightListenEventAccess())
            print(f"  (Input Monitoring preflight: {'granted' if allowed else 'denied'})",
                  flush=True)
            if not allowed and CGRequestListenEventAccess is not None:
                CGRequestListenEventAccess()
        tap = CGEventTapCreate(
            kCGSessionEventTap,
            kCGHeadInsertEventTap,
            kCGEventTapOptionListenOnly,
            mask,
            self._handler,
            None,
        )
        if tap is None:
            print("  (hotkey tap could not be created — grant Input Monitoring "
                  "to Terminal, then run ./start_max.sh from that Terminal)")
            self._ready.set()
            return
        source = CFMachPortCreateRunLoopSource(None, tap, 0)
        self._loop = CFRunLoopGetCurrent()
        CFRunLoopAddSource(self._loop, source, kCFRunLoopDefaultMode)
        CGEventTapEnable(tap, True)
        self._ready.set()
        try:
            while not self._stop.is_set():
                result = CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.25, False)
                try:
                    if result != kCFRunLoopRunTimedOut:
                        # Handled an event; keep going unless we were asked to stop.
                        continue
                except Exception:
                    break
        finally:
            self._loop = None

    def _handler(self, proxy, event_type, event, refcon):
        try:
            flags = int(CGEventGetFlags(event))
            vk = int(CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode))
            etype = int(event_type)
            aux = _sound_key(event, etype, vk)
            self.flags = flags
            self.saw_any_event = True
            if not self._logged_first:
                self._logged_first = True
                print(f"  (hotkey tap: first event type={etype} vk={vk} "
                      f"flags=0x{flags & _MOD_BITS:x})", flush=True)
            with self._lock:
                subs = list(self._subscribers)
            for cb in subs:
                try:
                    cb(etype, vk, flags, aux)
                except TypeError:
                    try:
                        cb(etype, vk, flags)
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass
        return event


def _sound_key(event, etype, vk) -> str | None:
    """'up' / 'down' for the Mac or EarPods volume keys, else None."""
    if vk == _VK_VOLUME_UP:
        return "up"
    if vk == _VK_VOLUME_DOWN:
        return "down"
    if NSEvent is None or etype != int(kCGEventSystemDefined):
        return None
    try:
        nse = NSEvent.eventWithCGEvent_(event)
        if nse is None or int(nse.subtype()) != _NX_SUBTYPE_AUX:
            return None
        data1 = int(nse.data1())
        key = (data1 & 0xFFFF0000) >> 16
        state = (data1 & 0x0000FF00) >> 8
        if state != _NX_KEY_DOWN:
            return None
        if key == _NX_KEYTYPE_SOUND_UP:
            return "up"
        if key == _NX_KEYTYPE_SOUND_DOWN:
            return "down"
    except Exception:
        return None
    return None


_TAP = _FlagTap()


class PushToTalk:
    """Tracks whether a configured modifier combo is currently held.

    Command+Option is the default (how it is actually said). Control+Option
    still counts so the older chord keeps working.
    """

    def __init__(self, combo=DEFAULT_COMBO, on_press=None, extra_combos=None):
        self.combo = tuple(combo)
        extras = list(extra_combos or ())
        if self.combo == ("cmd", "alt") and ("ctrl", "alt") not in extras:
            extras.append(("ctrl", "alt"))
        self._masks = [_flag_bits(self.combo)]
        for extra in extras:
            bits = _flag_bits(extra)
            if bits and bits not in self._masks:
                self._masks.append(bits)
        self._on_press = on_press
        self.held = threading.Event()
        self.saw_any_event = False
        self._started = False

    def _on_event(self, etype, vk, flags, aux=None) -> None:
        self.saw_any_event = True
        _TAP.saw_any_event = True
        if not self._masks:
            return
        down = any((flags & needed) == needed for needed in self._masks)
        if down and not self.held.is_set():
            self.held.set()
            if self._on_press:
                try:
                    self._on_press()
                except Exception:
                    pass
        elif not down and self.held.is_set():
            self.held.clear()

    def start(self) -> bool:
        if not self._masks or not any(self._masks):
            print("  (push-to-talk unavailable: empty combo)")
            return False
        _TAP.subscribe(self._on_event)
        self._started = True
        return True

    def stop(self) -> None:
        if self._started:
            _TAP.unsubscribe(self._on_event)
            self._started = False
        self.held.clear()

    def describe(self) -> str:
        pretty = {
            "cmd": "Command", "alt": "Option", "ctrl": "Control", "shift": "Shift",
        }
        return "+".join(pretty.get(k, k) for k in self.combo)


class WakeHotkey:
    """Tap a shortcut to wake Max, instead of holding one to talk.

    - "double_tap" (default): tap a modifier twice, like macOS dictation.
    - "combo": a chord such as Control+Option+Space.
    """

    def __init__(self, mode: str = "double_tap", key: str = DEFAULT_WAKE_KEY,
                 combo=DEFAULT_WAKE_COMBO, on_wake=None):
        self.mode = mode
        self.key_name = key
        self.combo = tuple(combo)
        self._on_wake = on_wake
        self._vks = _VK.get(key.strip().lower(), set())
        self._key_flag = _FLAG.get(key.strip().lower(), 0)
        self._combo_flags = _flag_bits(n for n in self.combo if n in _FLAG)
        self._combo_vks = set()
        for n in self.combo:
            if n not in _FLAG:
                self._combo_vks |= _VK.get(n.strip().lower(), set())
        self._last_tap = 0.0
        self._tap_armed = False
        self._combo_fired = False
        self._prev_flags = 0
        self.saw_any_event = False
        self._started = False

    def _is_target_modifier(self, vk, flags, prev) -> bool:
        """True when the configured wake modifier just went down.

        Prefer the hardware keycode (so Right Command is not Left Command).
        If the OS omitted the keycode, accept a lone Command/Control/Option
        bit flipping on and nothing else.
        """
        if vk in self._vks:
            return bool(flags & self._key_flag) and not (prev & self._key_flag)
        if vk == 0 and self._key_flag:
            changed = (flags ^ prev) & _MOD_BITS
            return changed == self._key_flag and bool(flags & self._key_flag)
        return False

    def _is_target_release(self, vk, flags, prev) -> bool:
        if vk in self._vks:
            return (not (flags & self._key_flag)) and bool(prev & self._key_flag)
        if vk == 0 and self._key_flag:
            changed = (flags ^ prev) & _MOD_BITS
            return changed == self._key_flag and not (flags & self._key_flag)
        return False

    def _on_event(self, etype, vk, flags, aux=None) -> None:
        self.saw_any_event = True
        prev = self._prev_flags
        self._prev_flags = flags
        if self.mode == "combo":
            self._handle_combo(etype, vk, flags, prev)
        else:
            self._handle_double_tap(etype, vk, flags, prev)

    def _handle_double_tap(self, etype, vk, flags, prev) -> None:
        if etype not in (int(kCGEventFlagsChanged), kCGEventFlagsChanged):
            # Any real character key cancels the gesture, same as before.
            if etype in (int(kCGEventKeyDown), kCGEventKeyDown):
                self._tap_armed = False
                self._last_tap = 0.0
            return
        if self._is_target_modifier(vk, flags, prev):
            self._tap_armed = True
            return
        if self._is_target_release(vk, flags, prev) and self._tap_armed:
            self._tap_armed = False
            now = time.monotonic()
            if now - self._last_tap <= DOUBLE_TAP_SECONDS:
                self._last_tap = 0.0
                self._fire()
            else:
                self._last_tap = now
            return
        # Some other modifier moved — cancel.
        if ((flags ^ prev) & _MOD_BITS) and not self._is_target_modifier(vk, flags, prev):
            if not self._is_target_release(vk, flags, prev):
                self._tap_armed = False
                self._last_tap = 0.0

    def _handle_combo(self, etype, vk, flags, prev) -> None:
        flags_ok = (not self._combo_flags) or ((flags & self._combo_flags) == self._combo_flags)
        if self._combo_vks:
            key_ok = etype in (int(kCGEventKeyDown), kCGEventKeyDown) and vk in self._combo_vks
            down = flags_ok and key_ok
        else:
            was = (prev & self._combo_flags) == self._combo_flags
            down = flags_ok and not was
        if down and not self._combo_fired:
            self._combo_fired = True
            self._fire()
        elif not flags_ok:
            self._combo_fired = False

    def _fire(self) -> None:
        if not self._on_wake:
            return
        try:
            self._on_wake()
        except Exception:
            pass

    def start(self) -> bool:
        _TAP.subscribe(self._on_event)
        self._started = True
        return True

    def stop(self) -> None:
        if self._started:
            _TAP.unsubscribe(self._on_event)
            self._started = False

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


class VolumeKeys:
    """Mac / EarPods volume keys adjust Max's spoken-voice gain.

    Listen-only: the OS still changes system volume. We also step the
    digital afplay level, because that is what actually makes Piper
    tolerable on earphones.
    """

    def __init__(self, on_change=None):
        self._on_change = on_change
        self._started = False
        self._last = 0.0

    def _on_event(self, etype, vk, flags, aux=None) -> None:
        if aux not in ("up", "down"):
            return
        if etype not in (int(kCGEventKeyDown), int(kCGEventSystemDefined)):
            return
        now = time.monotonic()
        if now - self._last < 0.14:
            return
        self._last = now
        if self._on_change:
            try:
                self._on_change(aux)
            except Exception:
                pass

    def start(self) -> bool:
        _TAP.subscribe(self._on_event)
        self._started = True
        return True

    def stop(self) -> None:
        if self._started:
            _TAP.unsubscribe(self._on_event)
            self._started = False
