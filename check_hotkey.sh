#!/bin/bash
# Verifies the keyboard shortcuts can actually see your keyboard.
#
# Run this from YOUR OWN Terminal, not from an agent — macOS grants Input
# Monitoring to the "responsible" app that started the process, so a
# process launched by something else does not inherit your Terminal's
# permission even though the code is identical.
cd "$(dirname "$0")" || exit 1
max/.venv/bin/python3 - <<'PYEOF'
import time

from max import hotkey
from max.config import load_config

cfg = load_config()

ptt = hotkey.PushToTalk(tuple(cfg.get("push_to_talk_combo", hotkey.DEFAULT_COMBO)))
wake = hotkey.WakeHotkey(
    mode=cfg.get("wake_hotkey_mode", "double_tap"),
    key=cfg.get("wake_hotkey_key", hotkey.DEFAULT_WAKE_KEY),
    combo=tuple(cfg.get("wake_hotkey_combo", hotkey.DEFAULT_WAKE_COMBO)),
)
woke = []
wake._on_wake = lambda: woke.append(time.time())

print("push-to-talk listener:", ptt.start(), "|", ptt.describe())
print("wake hotkey listener :", wake.start(), "|", wake.describe())

print(f"\n>>> 1. HOLD {ptt.describe()} for ~3 seconds...")
held = 0.0
end = time.time() + 8
while time.time() < end:
    time.sleep(0.1)
    if ptt.held.is_set():
        held += 0.1

print(f"\n>>> 2. Now trigger the wake hotkey: {wake.describe()}...")
time.sleep(8)

ptt.stop()
wake.stop()

saw_keys = ptt.saw_any_event or wake.saw_any_event
print(f"\nkey events received : {saw_keys}")
print(f"push-to-talk held   : {held:.1f}s")
print(f"wake hotkey fired   : {len(woke)} time(s)")

if not saw_keys:
    print("\nRESULT: BLOCKED — no key events at all.")
    print("        System Settings > Privacy & Security > Input Monitoring")
    print("        and enable the terminal app you are running this from,")
    print("        then QUIT AND REOPEN that terminal (the permission only")
    print("        takes effect on a fresh launch).")
elif held > 0 and woke:
    print("\nRESULT: WORKING — both shortcuts are good to go.")
else:
    print("\nRESULT: keyboard is visible, but not every shortcut registered.")
    if held == 0:
        print(f"        Push-to-talk: hold {ptt.describe()} together, not in sequence.")
    if not woke:
        if wake.mode == "double_tap":
            print(f"        Wake: two quick taps of {wake.describe().replace('double-tap ', '')},")
            print("        within about half a second, with no other key in between.")
        else:
            print(f"        Wake: press {wake.describe()} together.")
PYEOF
