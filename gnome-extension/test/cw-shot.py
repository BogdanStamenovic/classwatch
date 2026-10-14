#!/usr/bin/env python3
# Screenshot the isolated test shell. Shell only allows allow-listed bus names to
# call Screenshot (unsafe mode off), so this connection owns org.gnome.SettingsDaemon.MediaKeys
# on the private test bus first.
import os, sys
from gi.repository import Gio, GLib
addr = open('/tmp/cwtest/bus').read().strip()
conn = Gio.DBusConnection.new_for_address_sync(addr,
    Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION, None, None)
conn.call_sync('org.freedesktop.DBus', '/org/freedesktop/DBus', 'org.freedesktop.DBus', 'RequestName',
               GLib.Variant('(su)', ('org.gnome.SettingsDaemon.MediaKeys', 4)), None, 0, -1, None)
out = os.path.expanduser(f'~/data/classwatch/gnome-extension/shots/{sys.argv[1]}.png')
r = conn.call_sync('org.gnome.Shell.Screenshot', '/org/gnome/Shell/Screenshot', 'org.gnome.Shell.Screenshot',
                   'Screenshot', GLib.Variant('(bbs)', (len(sys.argv) > 2, False, out)), None, 0, -1, None)
print(r.unpack())
