#!/bin/bash
# Kill every process belonging to the isolated test session (identified by its
# XDG_RUNTIME_DIR) and unmount the FUSE mounts its gvfs/portal daemons leave behind.
T=/tmp/cwtest
for p in $(pgrep -u "$(id -u)"); do
  [ "$p" = "$$" ] && continue
  grep -qz "XDG_RUNTIME_DIR=$T/run" /proc/$p/environ 2>/dev/null && kill "$p" 2>/dev/null
done
sleep 2
for p in $(pgrep -u "$(id -u)"); do
  [ "$p" = "$$" ] && continue
  grep -qz "XDG_RUNTIME_DIR=$T/run" /proc/$p/environ 2>/dev/null && kill -9 "$p" 2>/dev/null
done
for m in "$T/run/gvfs" "$T/run/doc"; do
  mountpoint -q "$m" && fusermount3 -u "$m" 2>/dev/null || fusermount -u "$m" 2>/dev/null || true
done
exit 0
