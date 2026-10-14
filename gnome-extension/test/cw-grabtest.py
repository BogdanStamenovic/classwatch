#!/usr/bin/env python3
# Single-process grab test: one RemoteDesktop session for all input (separate
# short-lived sessions dropped keys after the second one), D-Bus calls to the
# extension, and screenshots, all on the isolated test bus.
import os, sys, time, subprocess
from gi.repository import Gio, GLib
addr = open('/tmp/cwtest/bus').read().strip()
def mkconn():
    return Gio.DBusConnection.new_for_address_sync(addr,
        Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION, None, None)
conn = mkconn()
conn.call_sync('org.freedesktop.DBus', '/org/freedesktop/DBus', 'org.freedesktop.DBus', 'RequestName',
               GLib.Variant('(su)', ('org.gnome.SettingsDaemon.MediaKeys', 4)), None, 0, -1, None)
def call(dest, path, iface, method, args=None):
    return conn.call_sync(dest, path, iface, method, args, None, 0, -1, None).unpack()
RD = 'org.gnome.Mutter.RemoteDesktop'; S = RD + '.Session'
(sp,) = call(RD, '/org/gnome/Mutter/RemoteDesktop', RD, 'CreateSession')
call(RD, sp, S, 'Start'); time.sleep(0.3)
signals = []
conn.signal_subscribe(None, 'io.github.bogdanstamenovic.Classwatch', 'TakeoverPressed', None, None, 0,
                      lambda *a: signals.append(time.time()))
def pump(sec):
    ctx = GLib.MainContext.default(); end = time.time() + sec
    while time.time() < end:
        ctx.iteration(False); time.sleep(0.01)
def key(ks):
    for down in (True, False):
        call(RD, sp, S, 'NotifyKeyboardKeysym', GLib.Variant('(ub)', (ks, down))); time.sleep(0.05)
    pump(0.4)
def cw(method, args=None):
    r = call('io.github.bogdanstamenovic.Classwatch', '/io/github/bogdanstamenovic/Classwatch',
             'io.github.bogdanstamenovic.Classwatch', method, args); pump(0.4); return r
def shot(name):
    out = os.path.expanduser(f'~/data/classwatch/gnome-extension/shots/{name}.png')
    call('org.gnome.Shell.Screenshot', '/org/gnome/Shell/Screenshot', 'org.gnome.Shell.Screenshot',
         'Screenshot', GLib.Variant('(bbs)', (False, False, out)))
def ext(cmd):
    env = dict(os.environ, DBUS_SESSION_BUS_ADDRESS=addr)
    subprocess.run(['gnome-extensions', cmd, 'classwatch@bogdanstamenovic.github.io'], env=env, check=True); pump(1)
T, ONE, B, C, TWO = 0x74, 0x31, 0x62, 0x63, 0x32
print('ping', cw('Ping'))
key(ONE)                                   # idle: '1' expected
key(T)                                     # idle: 't' must reach the editor
cw('SetContent', GLib.Variant('(ss)', ('Q?', 'A')))
cw('SetAlert', GLib.Variant('(b)', (True,)))
key(B)                                     # alerting: 'b' must pass through
shot('70-alert-on-after-b')
print('signals before t while alerting:', len(signals))
key(T)                                     # alerting: 't' grabbed -> takeover, NOT typed
print('signals after t while alerting:', len(signals))
shot('71-takeover-panel')
cw('Dismiss')
key(T)                                     # after dismiss: 't' must be typed
cw('SetAlert', GLib.Variant('(b)', (True,)))
cw('SetAlert', GLib.Variant('(b)', (False,)))
key(C); key(T)                             # after alert off: 'ct' typed
cw('SetAlert', GLib.Variant('(b)', (True,)))
ext('disable')
key(TWO); key(T)                           # extension disabled while alerting: '2t' typed
ext('enable')
print('ping after re-enable', cw('Ping'))
print('signals total (expect 1):', len(signals))
shot('72-editor-final')
print('expected editor text: 1tbtct2t')
call(RD, sp, S, 'Stop')
