#!/bin/bash
# Install the local Caddy hostname and login agents for the HUD + bridge.
# Voice still starts from Terminal: ./start_max.sh
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

# *.localhost resolves on this Mac with no /etc/hosts edit.
echo "HUD hostname: max.localhost (no hosts file change)"

# The HUD is already on disk in frontend/dist. Rebuilding on every
# install is what made "open the project" feel like starting from
# scratch. Only build when asked, or when there is nothing to serve.
if [ "${1:-}" = "--rebuild" ] || [ ! -f frontend/dist/index.html ]; then
  echo "Building the HUD..."
  (cd frontend && npm run build)
else
  echo "HUD already built (frontend/dist). Pass --rebuild to rebuild."
fi

mkdir -p "$HOME/Library/LaunchAgents"
# Templates use __ROOT__; fill in this machine's path so the repo never
# has to carry a home directory.
for name in com.max.caddy com.max.bridge; do
  sed "s|__ROOT__|$ROOT|g" "$ROOT/deploy/${name}.plist" \
    > "$HOME/Library/LaunchAgents/${name}.plist"
done

launchctl bootout "gui/$(id -u)/com.max.caddy" 2>/dev/null || true
launchctl bootout "gui/$(id -u)/com.max.bridge" 2>/dev/null || true
# Pre-rename leftover. Two Caddys on :2015 if this stays loaded.
launchctl bootout "gui/$(id -u)/com.orin.caddy" 2>/dev/null || true
launchctl bootout "gui/$(id -u)/com.orin.bridge" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/com.orin.caddy.plist" \
      "$HOME/Library/LaunchAgents/com.orin.bridge.plist"

# Ad-hoc Caddy/bridge (started by hand or start_max.sh) still hold the
# ports. Stop those so the login agents can own them instead of
# crash-looping on "address already in use". Voice is left alone.
pkill -f "caddy run --config ${ROOT}/Caddyfile" 2>/dev/null || true
# -- so macOS pkill does not treat "-m" as a flag.
pkill -f -- "max.bridge" 2>/dev/null || true
sleep 1

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
