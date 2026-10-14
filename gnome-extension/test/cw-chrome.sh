#!/bin/bash
export XDG_RUNTIME_DIR=/tmp/cwtest/run HOME=/tmp/cwtest/home
export DBUS_SESSION_BUS_ADDRESS=$(cat /tmp/cwtest/bus)
export WAYLAND_DISPLAY=$(cd /tmp/cwtest/run && ls wayland-* | grep -v lock | head -1)
unset DISPLAY
setsid nohup google-chrome-stable --ozone-platform=wayland --class=classwatch-meet --user-data-dir=/tmp/cwtest/chrome \
  --password-store=basic --no-first-run --no-default-browser-check --disable-gpu --app=file:///tmp/cw-fake-meet.html \
  > /tmp/cwtest/chrome.log 2>&1 < /dev/null &
