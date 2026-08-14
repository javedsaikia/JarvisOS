#!/bin/bash
# Start JARVIS (bridge + voice loop).
#
# Run this from YOUR OWN Terminal. macOS grants Input Monitoring — which
# push-to-talk needs to see the Control+Option keypress — to the
# "responsible" app that started the process. A JARVIS launched by some
# other tool inherits that tool's permissions instead of your Terminal's,
# and push-to-talk then silently receives nothing.
#
#   ./start_jarvis.sh          start it
#   ./start_jarvis.sh stop     stop it
#   ./start_jarvis.sh log      follow the voice log
set -u
cd "$(dirname "$0")" || exit 1

BRIDGE_LOG=/tmp/jarvisos_bridge.log
VOICE_LOG=/tmp/voice_loop.log
PY=jarvis/.venv/bin/python3

stop_all() {
    pkill -f "jarvis.bridge" 2>/dev/null
    pkill -f "jarvis.voice_loop" 2>/dev/null
    sleep 2
}

case "${1:-start}" in
  stop)
    stop_all
    echo "JARVIS stopped."
    exit 0
    ;;
  log)
    exec tail -f "$VOICE_LOG"
    ;;
esac

echo "Stopping any running instance..."
stop_all

: > "$VOICE_LOG"
echo "Starting bridge..."
nohup "$PY" -m jarvis.bridge > "$BRIDGE_LOG" 2>&1 &
sleep 2

echo "Starting voice loop..."
"$PY" - <<'PYEOF'
import asyncio, json, sys
try:
    import websockets
except ImportError:
    sys.exit("websockets not installed in the venv")

async def main():
    try:
        async with websockets.connect("ws://localhost:8765", open_timeout=8) as ws:
            await asyncio.wait_for(ws.recv(), timeout=3)
            try:
                await asyncio.wait_for(ws.recv(), timeout=2)
            except asyncio.TimeoutError:
                pass
            await ws.send(json.dumps({"type": "voice_start"}))
            print("  ", await asyncio.wait_for(ws.recv(), timeout=10))
    except Exception as e:
        sys.exit(f"could not reach the bridge: {e}")

asyncio.run(main())
PYEOF

echo "Waiting for the voice loop to come up (models load on first start)..."
online=0
seen_proc=0
for _ in $(seq 1 90); do
    if grep -q "voice loop online" "$VOICE_LOG" 2>/dev/null; then
        online=1
        break
    fi
    if pgrep -f "jarvis.voice_loop" > /dev/null 2>&1; then
        seen_proc=1
    elif [ "$seen_proc" = 1 ]; then
        # It was alive and now isn't: it died during startup. No point
        # sitting out the rest of the wait.
        break
    fi
    sleep 2
done

echo
if [ "$online" != 1 ]; then
    echo "JARVIS did NOT come up: the voice loop never reported itself online."
    echo
    echo "--- $VOICE_LOG"
    if [ -s "$VOICE_LOG" ]; then
        cat "$VOICE_LOG"
    else
        echo "(empty — the voice loop produced no output at all)"
    fi
    echo
    echo "--- $BRIDGE_LOG (last 20 lines)"
    if [ -s "$BRIDGE_LOG" ]; then
        tail -n 20 "$BRIDGE_LOG"
    else
        echo "(empty)"
    fi
    echo
    echo "Try again, or run the voice loop in the foreground to see it fail:"
    echo "  $PY -u -m jarvis.voice_loop"
    exit 1
fi

cat "$VOICE_LOG"
echo

# Ask the config itself, rather than inferring push-to-talk from whatever
# happens to be in the log — an empty log used to read as "disabled".
combo=$("$PY" - <<'PYEOF'
from jarvis import config, hotkey
cfg = config.load_config()
if cfg.get("push_to_talk_enabled", True):
    keys = tuple(cfg.get("push_to_talk_combo", hotkey.DEFAULT_COMBO))
    print(hotkey.PushToTalk(keys).describe())
else:
    print("disabled")
PYEOF
)

case "$combo" in
    # Empty means the check itself failed, which is not the same thing as
    # push-to-talk being off — say so instead of guessing.
    "")         echo "JARVIS is running (could not read push-to-talk config, see above)." ;;
    disabled)   echo "JARVIS is running (push-to-talk disabled in config)." ;;
    *)          echo "JARVIS is running. Hold $combo and speak, or say \"Hey Jarvis\"." ;;
esac
echo "Follow the log with:  ./start_jarvis.sh log"
