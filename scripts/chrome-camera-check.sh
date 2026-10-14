#!/bin/bash
# Does Chrome actually receive the fake camera, in the right colours? For each pixel format
# (FORMATS="YUY2 I420 ..."), start a test camera node "cwtest" from a scratch copy of this repo,
# open it with getUserMedia in a throwaway headless Chrome, and print frame size + mean RGB.
# Compare the mean with the clip's (see README). Run on the laptop, in a logged-in session.
# This is how the blue-skin (BGR) and black-frame (I420) bugs were found.
HERE=$(cd "$(dirname "$0")/.." && pwd)
export XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
PY="$HERE/.venv/bin/python"
rm -rf /tmp/cwfix && cp -r "$HERE" /tmp/cwfix && cd /tmp/cwfix
sed -i 's/^CAM_NAME = "classwatch"/CAM_NAME = "cwtest"/; s/node.description=classwatch/node.description=cwtest/' src/classwatch/camera.py
for FMT in ${FORMATS:-YUY2 I420 NV12 RGB BGR}; do
  cp src/classwatch/camera.py /tmp/cam.base
  sed -i -E "s#\"video/x-raw,format=[A-Z0-9]+\"#\"video/x-raw,format=$FMT\"#" src/classwatch/camera.py
  mkdir -p /tmp/cwtest-state
  XDG_STATE_HOME=/tmp/cwtest-state CLASSWATCH_CAM_CLIP=$HOME/.local/state/classwatch/clip.mp4 PYTHONPATH=src $PY -m classwatch.cli camera up > /tmp/cwtest.log 2>&1 &
  CAM=$!; sleep 4
  rm -rf /tmp/cwchrome
  timeout 25 google-chrome-stable --headless=new --user-data-dir=/tmp/cwchrome --no-first-run \
     --use-fake-ui-for-media-stream --enable-features=WebRtcPipeWireCamera \
     --enable-logging=stderr --v=0 "file://$HERE/scripts/chrome-camera-check.html" 2>&1 | grep -o "CWTEST.*" | sed 's/", source.*//' > /tmp/cwres.txt &
  CH=$!
  for i in $(seq 1 22); do grep -q RESULT /tmp/cwres.txt 2>/dev/null && break; sleep 1; done
  pkill -f "[u]ser-data-dir=/tmp/cwchrome"; wait $CH 2>/dev/null
  echo "== $FMT: $(grep RESULT /tmp/cwres.txt | head -1)"; [ "$FMT" = "${FORMATS%% *}" ] && grep devices /tmp/cwres.txt | head -1
  kill -TERM $CAM; for i in $(seq 1 10); do kill -0 $CAM 2>/dev/null || break; sleep 1; done; kill -9 $CAM 2>/dev/null
  cp /tmp/cam.base src/classwatch/camera.py; rm -rf /tmp/cwtest-state
done
rm -rf /tmp/cwfix /tmp/cwchrome /tmp/cam.base /tmp/cwres.txt /tmp/cwtest.log
