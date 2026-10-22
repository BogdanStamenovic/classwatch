#!/usr/bin/env python3
# Ctrl+D / Ctrl+P camera keys in the isolated test shell: signals fire only with Meet
# focused, Ctrl+D still reaches Meet, Ctrl+P does not, other windows get both keys.
# Needs test-headless.sh running, the fake Meet (cw-keys-meet.html) and gnome-text-editor open.
import os, sys, time, subprocess
from gi.repository import Gio, GLib
addr = open('/tmp/cwtest/bus').read().strip()
conn = Gio.DBusConnection.new_for_address_sync(addr,
    Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION, None, None)
conn.call_sync('org.freedesktop.DBus', '/org/freedesktop/DBus', 'org.freedesktop.DBus', 'RequestName',
               GLib.Variant('(su)', ('org.gnome.SettingsDaemon.MediaKeys', 4)), None, 0, -1, None)
def call(dest, path, iface, method, args=None):
    return conn.call_sync(dest, path, iface, method, args, None, 0, -1, None).unpack()
RD = 'org.gnome.Mutter.RemoteDesktop'; S = RD + '.Session'
CW = ('io.github.bogdanstamenovic.Classwatch', '/io/github/bogdanstamenovic/Classwatch', 'io.github.bogdanstamenovic.Classwatch')
(sp,) = call(RD, '/org/gnome/Mutter/RemoteDesktop', RD, 'CreateSession')
call(RD, sp, S, 'Start'); time.sleep(0.3)
signals = []
for name in ('MicOpened', 'MicClosed', 'RecordPressed'):
    conn.signal_subscribe(None, CW[2], name, None, None, 0, lambda *a, n=name: signals.append(n))
def pump(sec):
    ctx = GLib.MainContext.default(); end = time.time() + sec
    while time.time() < end:
        ctx.iteration(False); time.sleep(0.01)
def ks(sym, down):
    call(RD, sp, S, 'NotifyKeyboardKeysym', GLib.Variant('(ub)', (sym, down))); time.sleep(0.03)
CTRL, D, P = 0xffe3, 0x64, 0x70
def chord(letter, release_ctrl_first=False):
    ks(CTRL, True); ks(letter, True)
    if release_ctrl_first:  # Ctrl let go before the extension resends
        ks(CTRL, False); ks(letter, False)
    else:
        ks(letter, False); ks(CTRL, False)
    pump(0.5)
def rel(dx, dy):
    call(RD, sp, S, 'NotifyPointerMotionRelative', GLib.Variant('(dd)', (dx, dy))); time.sleep(0.01)
def click(x, y):
    for _ in range(25): rel(100.0, 0.0)
    for _ in range(15): rel(0.0, 100.0)
    dx, dy = x - 1919, y - 1079; n = int(max(abs(dx), abs(dy)) // 10) + 1
    for _ in range(n): rel(dx / n, dy / n)
    for down in (True, False):
        call(RD, sp, S, 'NotifyPointerButton', GLib.Variant('(ib)', (0x110, down))); time.sleep(0.05)
    pump(0.5)
def shot(name):
    out = os.path.expanduser(f'~/data/classwatch/gnome-extension/shots/{name}.png')
    call('org.gnome.Shell.Screenshot', '/org/gnome/Shell/Screenshot', 'org.gnome.Shell.Screenshot',
         'Screenshot', GLib.Variant('(bbs)', (False, False, out)))
results = []
def check(label, want):
    pump(0.3)
    got = list(signals); signals.clear()
    ok = got == want
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {label}: signals {got} (want {want})")
print('Ping:', call(*CW, 'Ping'))
meet_xy = tuple(map(int, sys.argv[1].split(',')))  # a point on Meet not covered by the editor

call(*CW, 'SetCameraKeys', GLib.Variant('(b)', (True,)))
click(*meet_xy)
chord(D); check('Ctrl+D in Meet opens mic', ['MicOpened'])
chord(D); check('Ctrl+D in Meet again closes mic', ['MicClosed'])
chord(D, release_ctrl_first=True); check('Ctrl+D with Ctrl released early', ['MicOpened'])
chord(D); check('back to closed', ['MicClosed'])
chord(P); check('Ctrl+P in Meet records', ['RecordPressed'])
for _ in range(3): chord(D)  # fast repeats: each one must still reach Meet and flip the mic
check('three quick Ctrl+D', ['MicOpened', 'MicClosed', 'MicOpened'])
chord(D); check('closed again', ['MicClosed'])
shot('80-meet-keys')  # page should list 8 x ctrl+d, no ctrl+p
ks(0xffe9, True); ks(0xff09, True); ks(0xff09, False); ks(0xffe9, False); pump(1.0)  # Alt+Tab to the editor
chord(D); chord(P); check('Ctrl+D / Ctrl+P in another window: no signals', [])
pump(1.0); shot('81-other-window')  # editor got its own Ctrl+P (print dialog)
subprocess.run(['true'])
call(*CW, 'SetCameraKeys', GLib.Variant('(b)', (False,)))
click(*meet_xy)
chord(D); check('keys off: Ctrl+D in Meet, no signal', [])
shot('82-keys-off')  # page should list a 9th ctrl+d
call(RD, sp, S, 'Stop')
print('ALL PASS' if all(results) else 'SOME FAILED')
