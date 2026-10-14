#!/bin/bash
SHOT="python3 /tmp/cw-shot.py"; K="python3 /tmp/cw-key.py"
rm -f ~/data/classwatch/gnome-extension/shots/*.png
/tmp/cw-restart.sh
/tmp/cw-chrome.sh; sleep 7
$K 0xff1b; sleep 1
/tmp/cw-gte.sh; sleep 4
export DBUS_SESSION_BUS_ADDRESS=$(cat /tmp/cwtest/bus)
gdbus monitor --session -d io.github.bogdanstamenovic.Classwatch -o /io/github/bogdanstamenovic/Classwatch > /tmp/cwtest/monitor.log 2>&1 &
MON=$!
sigs() { echo "TakeoverPressed signals so far: $(grep -c TakeoverPressed /tmp/cwtest/monitor.log)"; }
echo "== 1 idle: type a t"; $K 0x61 0x74; $SHOT 30-idle-typed-at
/tmp/cw.sh SetContent 'Koja je razlika izmedju TCP i UDP?' $'ANSWER:\nTCP je konekciono orijentisan, pouzdan, garantuje redosled.\nUDP je bez konekcije, brzi, bez garancija.\n\nPOINTERS:\n- three-way handshake\n- DNS i streaming koriste UDP'
echo "== 2 alert on"; /tmp/cw.sh SetAlert true; sleep 1; $SHOT 31-alert-on
echo "== 3 right-click inside left glow strip over desktop"; $K rclick:15,600; sleep 0.5; $SHOT 32-glow-clickthrough cursor; $K 0xff1b; sleep 0.3; $K click:960,600
echo "== 4 alert off, type t"; /tmp/cw.sh SetAlert false; sleep 0.3; $K 0x74; $SHOT 33-alert-off-t-typed; sigs
echo "== 5 alert on, type b then t"; /tmp/cw.sh SetAlert true; sleep 0.5; $K 0x62; sleep 0.3; $SHOT 34a-alert-b-typed; $K 0x74; sleep 1.5; $SHOT 34-takeover-by-t; sigs
echo "== 6 long content"; /tmp/cw.sh SetContent 'Live update question (streamed)?' "$(for i in $(seq 1 40); do echo "line $i: the answer keeps streaming in while the panel is visible, wrapping long text across the panel width to check that wrapping works properly and that the panel caps its height"; done)"; sleep 0.5; $SHOT 35-long-content
echo "== 7 streaming 50 updates @10Hz"; for i in $(seq 1 50); do /tmp/cw.sh SetContent "Streaming question $i?" "$(printf 'QUESTION: q%s\nANSWER: partial answer token count %s\nPOINTERS:\n- one\n- two' $i $i)" >/dev/null; sleep 0.1; done; $SHOT 36-after-stream
echo "== 8 click panel"; $K click:960,60; sleep 0.8; $SHOT 37-after-panel-click
echo "== 9 alert over fullscreen"; /tmp/cw.sh SetAlert true; sleep 1.2; $SHOT 38-alert-over-fullscreen
echo "== 10 dismiss then t (must not trigger)"; /tmp/cw.sh Dismiss; sleep 0.3; $K 0x74; sleep 0.5; sigs; $SHOT 39-dismissed
echo "== 11 Takeover method"; /tmp/cw.sh Takeover; sleep 1; $SHOT 40-takeover-method; sigs; /tmp/cw.sh Dismiss
echo "== 12 disable/enable cycle"; /tmp/cw.sh SetAlert true; gnome-extensions disable classwatch@bogdanstamenovic.github.io; sleep 1; $SHOT 41-disabled; /tmp/cw.sh Ping 2>&1 | head -1; $K 0x74; sigs
gnome-extensions enable classwatch@bogdanstamenovic.github.io; sleep 1; /tmp/cw.sh Ping; gnome-extensions info classwatch@bogdanstamenovic.github.io | grep -i state
kill $MON
echo "== shell log (filtered)"; grep -iE "classwatch|JS ERROR|TypeError|ReferenceError|Gjs-CRITICAL|Gjs-WARNING" /tmp/cwtest/shell.log
