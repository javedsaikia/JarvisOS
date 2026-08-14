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
for _ in $(seq 1 90); do
    grep -q "voice loop online" "$VOICE_LOG" 2>/dev/null && break
    sleep 2
done

echo
cat "$VOICE_LOG"
echo
if grep -q "push-to-talk" "$VOICE_LOG"; then
    echo "JARVIS is running. Hold Control+Option and speak, or say \"Hey Jarvis\"."
else
    echo "JARVIS is running (push-to-talk disabled in config)."
fi
echo "Follow the log with:  ./start_jarvis.sh log"
