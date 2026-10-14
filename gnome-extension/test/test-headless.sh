#!/bin/bash
# Start an isolated headless GNOME Shell (own HOME, runtime dir, dconf, session bus)
# with the extension enabled. Does not touch the live session. Usage:
#   test-headless.sh            -> prints the bus address to /tmp/cwtest/bus, runs until killed
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
UUID=classwatch@bogdanstamenovic.github.io
T=/tmp/cwtest
rm -rf "$T"; mkdir -p "$T/home" "$T/run"; chmod 700 "$T/run"
export HOME="$T/home" XDG_RUNTIME_DIR="$T/run"
export XDG_CONFIG_HOME="$HOME/.config" XDG_DATA_HOME="$HOME/.local/share" XDG_CACHE_HOME="$HOME/.cache"
unset WAYLAND_DISPLAY DISPLAY DBUS_SESSION_BUS_ADDRESS XDG_SESSION_ID
mkdir -p "$XDG_DATA_HOME/gnome-shell/extensions"
cp -r "$SRC" "$XDG_DATA_HOME/gnome-shell/extensions/$UUID"
exec dbus-run-session -- bash -c '
  echo "$DBUS_SESSION_BUS_ADDRESS" > '"$T"'/bus
  gsettings set org.gnome.shell enabled-extensions "[\"'"$UUID"'\"]"
  gsettings set org.gnome.shell welcome-dialog-last-shown-version "999"
  exec gnome-shell --headless --virtual-monitor 1920x1080 --wayland --no-x11 --force-animations
' > "$T/shell.log" 2>&1
