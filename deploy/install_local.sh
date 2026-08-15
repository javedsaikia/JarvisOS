#!/bin/bash
# Install the local Caddy hostname and login agents for the HUD + bridge.
# Voice still starts from Terminal: ./start_max.sh
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

# *.localhost resolves on this Mac with no /etc/hosts edit.
echo "HUD hostname: max.localhost (no hosts file change)"

echo "Building the HUD..."
(cd frontend && npm run build)

mkdir -p "$HOME/Library/LaunchAgents"
# Templates use __ROOT__; fill in this machine's path so the repo never
# has to carry a home directory.
for name in com.max.caddy com.max.bridge; do
  sed "s|__ROOT__|$ROOT|g" "$ROOT/deploy/${name}.plist" \
    > "$HOME/Library/LaunchAgents/${name}.plist"
done

launchctl bootout "gui/$(id -u)/com.max.caddy" 2>/dev/null || true
launchctl bootout "gui/$(id -u)/com.max.bridge" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.max.caddy.plist"
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.max.bridge.plist"
launchctl enable "gui/$(id -u)/com.max.caddy"
launchctl enable "gui/$(id -u)/com.max.bridge"
launchctl kickstart -k "gui/$(id -u)/com.max.caddy"
launchctl kickstart -k "gui/$(id -u)/com.max.bridge"

echo
echo "Caddy + bridge will start at login."
echo "HUD:   http://max.localhost:2015"
echo "Voice: ./start_max.sh   (from Terminal, for mic/hotkey permission)"
