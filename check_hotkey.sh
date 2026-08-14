#!/bin/bash
# Verifies push-to-talk can actually see your keyboard.
#
# Run this from YOUR OWN Terminal, not from an agent — macOS grants Input
# Monitoring to the "responsible" app that started the process, so a
# process launched by something else does not inherit your Terminal's
# permission even though the code is identical.
cd "$(dirname "$0")" || exit 1
jarvis/.venv/bin/python3 - <<'PYEOF'
import time
from jarvis import hotkey

ptt = hotkey.PushToTalk()
print("listener started:", ptt.start(), "| combo:", ptt.describe())
print("\n>>> PRESS AND HOLD Control+Option for ~3 seconds now...\n")
held = 0.0
end = time.time() + 10
while time.time() < end:
    time.sleep(0.1)
    if ptt.held.is_set():
        held += 0.1
ptt.stop()

print(f"key events received : {ptt.saw_any_event}")
print(f"combo held for      : {held:.1f}s")
if held > 0:
    print("\nRESULT: WORKING — push-to-talk is good to go.")
elif ptt.saw_any_event:
    print("\nRESULT: keyboard is visible, but Control+Option was not detected.")
    print("        Try again and hold both keys together.")
else:
    print("\nRESULT: BLOCKED — no key events at all.")
    print("        System Settings > Privacy & Security > Input Monitoring")
    print("        and enable the terminal app you are running this from,")
    print("        then QUIT AND REOPEN that terminal (the permission only")
    print("        takes effect on a fresh launch).")
PYEOF
