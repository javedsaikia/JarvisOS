#!/bin/bash
# Install the local Caddy hostname and login agents for the HUD + bridge.
# Voice still starts from Terminal: ./start_bella.sh
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

# *.localhost resolves on this Mac with no /etc/hosts edit.
echo "HUD hostname: bella.localhost (no hosts file change)"

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
for name in com.bella.caddy com.bella.bridge; do
  sed "s|__ROOT__|$ROOT|g" "$ROOT/deploy/${name}.plist" \
    > "$HOME/Library/LaunchAgents/${name}.plist"
done

launchctl bootout "gui/$(id -u)/com.bella.caddy" 2>/dev/null || true
launchctl bootout "gui/$(id -u)/com.bella.bridge" 2>/dev/null || true
# Pre-rename leftovers. Two Caddys on :2015 if these stay loaded.
for old in com.max.caddy com.max.bridge com.orin.caddy com.orin.bridge; do
  launchctl bootout "gui/$(id -u)/$old" 2>/dev/null || true
  rm -f "$HOME/Library/LaunchAgents/${old}.plist"
done

# Ad-hoc Caddy/bridge (started by hand or start_bella.sh) still hold the
# ports. Stop those so the login agents can own them instead of
# crash-looping on "address already in use". Voice is left alone.
pkill -f "caddy run --config ${ROOT}/Caddyfile" 2>/dev/null || true
# -- so macOS pkill does not treat "-m" as a flag.
pkill -f -- "bella.bridge" 2>/dev/null || true
pkill -f -- "max.bridge" 2>/dev/null || true
sleep 1

launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.bella.caddy.plist"
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.bella.bridge.plist"
launchctl enable "gui/$(id -u)/com.bella.caddy"
launchctl enable "gui/$(id -u)/com.bella.bridge"
launchctl kickstart -k "gui/$(id -u)/com.bella.caddy"
launchctl kickstart -k "gui/$(id -u)/com.bella.bridge"

echo
echo "Caddy + bridge will start at login."
echo "HUD:   http://bella.localhost:2015"
echo "Voice: ./start_bella.sh   (from Terminal, for mic/hotkey permission)"
