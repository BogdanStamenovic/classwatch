#!/bin/bash
export XDG_RUNTIME_DIR=/tmp/cwtest/run HOME=/tmp/cwtest/home
export DBUS_SESSION_BUS_ADDRESS=$(cat /tmp/cwtest/bus)
export WAYLAND_DISPLAY=wayland-0 GDK_BACKEND=wayland
unset DISPLAY
setsid nohup gnome-text-editor --new-window > /tmp/cwtest/gte.log 2>&1 < /dev/null &
