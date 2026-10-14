#!/usr/bin/env python3
# Inject input into the isolated test shell via Mutter's RemoteDesktop API.
# usage: cw-key.py 0x74 click:X,Y rclick:X,Y dclick:X,Y ...
import sys, time
from gi.repository import Gio, GLib
addr = open('/tmp/cwtest/bus').read().strip()
conn = Gio.DBusConnection.new_for_address_sync(addr,
    Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION, None, None)
S = 'org.gnome.Mutter.RemoteDesktop.Session'
def call(path, iface, method, args=None):
    return conn.call_sync('org.gnome.Mutter.RemoteDesktop', path, iface, method, args, None, 0, -1, None).unpack()
def rel(dx, dy):
    # Unpaced bursts of relative motion were partly dropped; 10ms pacing keeps them 1:1.
    call(sp, S, 'NotifyPointerMotionRelative', GLib.Variant('(dd)', (dx, dy)))
    time.sleep(0.01)
(sp,) = call('/org/gnome/Mutter/RemoteDesktop', 'org.gnome.Mutter.RemoteDesktop', 'CreateSession')
call(sp, S, 'Start')
time.sleep(0.3)
for a in sys.argv[1:]:
    kind = a.split(':')[0]
    if kind in ('click', 'rclick', 'dclick', 'move'):
        x, y = map(float, a.split(':')[1].split(','))
        # Park bottom-right (no hot corner there), then walk to the target.
        for _ in range(25): rel(100.0, 0.0)
        for _ in range(15): rel(0.0, 100.0)
        dx, dy = x - 1919, y - 1079
        n = int(max(abs(dx), abs(dy)) // 10) + 1
        for _ in range(n): rel(dx / n, dy / n)
        time.sleep(0.2)
        if kind == 'move':
            continue
        btn = 0x111 if kind == 'rclick' else 0x110
        for _ in range(2 if kind == 'dclick' else 1):
            for down in (True, False):
                call(sp, S, 'NotifyPointerButton', GLib.Variant('(ib)', (btn, down)))
                time.sleep(0.05)
    else:
        ks = int(a, 16)
        for down in (True, False):
            call(sp, S, 'NotifyKeyboardKeysym', GLib.Variant('(ub)', (ks, down)))
            time.sleep(0.05)
    time.sleep(0.4)
call(sp, S, 'Stop')
print('sent', sys.argv[1:])
