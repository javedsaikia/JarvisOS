#!/bin/bash
# Start Bella (bridge + voice loop).
#
# Run this from YOUR OWN Terminal. macOS grants Input Monitoring — which
# Command+Option push-to-talk needs — to the "responsible" app that
# started the process. Voice must be a child of Terminal, not of the
# login-agent bridge, or the keys arrive as silence.
#
#   ./start_bella.sh          start voice (and the bridge if it is not up)
#   ./start_bella.sh stop     stop the voice loop
#   ./start_bella.sh log      follow the voice log
#   ./start_bella.sh status   what's already running — no rebuild, no restart
#
# The HUD is local Caddy at http://bella.localhost:2015 — see Caddyfile and
# deploy/install_local.sh. Voice still starts from Terminal so macOS
# Input Monitoring and Screen Recording attach here.
set -u
cd "$(dirname "$0")" || exit 1

BRIDGE_LOG=/tmp/bella_bridge.log
VOICE_LOG=/tmp/voice_loop.log
PY=bella/.venv/bin/python3

stop_all() {
    # The rename left a Max voice loop running. It still grabs the mic,
    # writes to max/memory (gone), and kills the turn. Kill both names.
    pkill -f "bella.voice_loop" 2>/dev/null
    pkill -f "max.voice_loop" 2>/dev/null
    sleep 2
}

case "${1:-start}" in
  stop)
    stop_all
    echo "Bella stopped."
    exit 0
    ;;
  log)
    exec tail -f "$VOICE_LOG"
    ;;
  status)
    echo "HUD:    $(lsof -iTCP:2015 -sTCP:LISTEN >/dev/null 2>&1 && echo up || echo down)  http://bella.localhost:2015"
    echo "Bridge: $(lsof -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && echo up || echo down)  ws://127.0.0.1:8765"
    leftover=$(pgrep -f 'max.voice_loop' >/dev/null 2>&1 && echo " leftover max.voice_loop" || echo "")
    echo "Voice:  $(pgrep -f 'bella.voice_loop' >/dev/null 2>&1 && echo up || echo down)$leftover"
    echo "Agents: $(launchctl print "gui/$(id -u)/com.bella.bridge" >/dev/null 2>&1 && echo installed || echo missing)"
    echo "HUD build: $([ -f frontend/dist/index.html ] && echo ready || echo missing — run deploy/install_local.sh --rebuild)"
    exit 0
    ;;
esac

echo "Stopping any running voice loop..."
stop_all

: > "$VOICE_LOG"
if ! lsof -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Starting bridge..."
    nohup "$PY" -m bella.bridge > "$BRIDGE_LOG" 2>&1 &
    sleep 2
else
    echo "Bridge already listening on 8765."
fi

echo "Starting voice loop from this Terminal (so hotkeys can see the keyboard)..."
# Child of this shell — not of the launchd bridge — so Input Monitoring
# attaches to Terminal. nohup only ignores hangup; it does not reparent.
nohup "$PY" -u -m bella.voice_loop >> "$VOICE_LOG" 2>&1 &

echo "Waiting for the voice loop to come up (models load on first start)..."
online=0
seen_proc=0
for _ in $(seq 1 90); do
    if grep -qE "voice loop online|Microphone OFF" "$VOICE_LOG" 2>/dev/null; then
        online=1
        break
    fi
    if pgrep -f "bella.voice_loop" > /dev/null 2>&1; then
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
    echo "Bella did NOT come up: the voice loop never reported itself online."
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
    echo "  $PY -u -m bella.voice_loop"
    exit 1
fi

cat "$VOICE_LOG"
echo

# Ask the config itself, rather than inferring push-to-talk from whatever
# happens to be in the log — an empty log used to read as "disabled".
combo=$("$PY" - <<'PYEOF'
from bella import config, hotkey
cfg = config.load_config()
if cfg.get("push_to_talk_enabled", True):
    keys = tuple(cfg.get("push_to_talk_combo", hotkey.DEFAULT_COMBO))
    print(hotkey.PushToTalk(keys).describe())
else:
    print("disabled")
PYEOF
)

# Same reason as the combo above: read the phrase from config, so this can
# never tell you to say something the loop isn't listening for.
wake=$("$PY" -c 'from bella import config; p = config.load_config().get("wake_phrases") or ["hey bella"]; print(" ".join(w.capitalize() for w in p[0].split()))')

case "$combo" in
    # Empty means the check itself failed, which is not the same thing as
    # push-to-talk being off — say so instead of guessing.
    "")         echo "Bella is running (could not read push-to-talk config, see above)." ;;
    disabled)   echo "Bella is running (push-to-talk disabled in config)." ;;
    *)          echo "Bella is running. Hold $combo and speak, or say \"$wake\"." ;;
esac
echo "HUD:  http://bella.localhost:2015"
echo "Follow the log with:  ./start_bella.sh log"
