// Classwatch: edge glow + takeover panel, driven by the classwatch daemon over D-Bus.
import Clutter from 'gi://Clutter';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Meta from 'gi://Meta';
import Pango from 'gi://Pango';
import Shell from 'gi://Shell';
import St from 'gi://St';

import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';

const VERSION = 'classwatch-gnome-extension 3';
const BUS_NAME = 'io.github.bogdanstamenovic.Classwatch';
const OBJ_PATH = '/io/github/bogdanstamenovic/Classwatch';
const GLOW_PX = 40;
const PANEL_MAX_FRACTION = 0.45;

const TAKEOVER_KEY = 't';
const MIC_KEY = '<Control>d';  // Meet's mic toggle
const RECORD_KEY = '<Control>p';
// evdev codes, so the resent key is the same physical key whatever layout is active
// (a keyval like 'd' does not exist in the Serbian Cyrillic layout).
const EVDEV = {[MIC_KEY]: 32, [RECORD_KEY]: 25, ctrl: 29};
// Measured in the headless test shell: a copy sent from inside the accelerator handler,
// or 20-50 ms after it, is dropped (one in eight still was at 50 ms); 100 ms after it,
// with the keys regrabbed 400 ms after it, every copy arrived. A second Ctrl+D inside
// that 400 ms reaches Meet without moving the camera.
const FORWARD_DELAY_MS = 100;
const REGRAB_MS = 400;

const IFACE_XML = `
<node>
  <interface name="io.github.bogdanstamenovic.Classwatch">
    <method name="SetAlert"><arg type="b" name="on" direction="in"/></method>
    <method name="SetContent">
      <arg type="s" name="question" direction="in"/>
      <arg type="s" name="answer" direction="in"/>
    </method>
    <method name="Takeover"/>
    <method name="Release"/>
    <method name="Dismiss"/>
    <method name="SetCameraKeys"><arg type="b" name="on" direction="in"/></method>
    <method name="Ping"><arg type="s" name="version" direction="out"/></method>
    <signal name="TakeoverPressed"/>
    <signal name="ReleasePressed"/>
    <signal name="MicOpened"/>
    <signal name="MicClosed"/>
    <signal name="RecordPressed"/>
  </interface>
</node>`;

export default class ClasswatchExtension extends Extension {
    enable() {
        this._question = '';
        this._answer = '';
        this._alertOn = false;
        this._takenOver = false;
        this._unredirectOff = false;
        this._keys = new Map();  // accelerator -> {action, name}
        this._cameraKeys = false;
        this._micOpen = false;  // Meet's mic as toggled by Ctrl+D; assumed muted at start
        this._regrabId = 0;
        this._forwardId = 0;
        this._vkbd = null;
        this._pulseId = 0;

        this._buildGlow();
        this._buildPanel();

        this._acceleratorId = global.display.connect('accelerator-activated',
            (_display, action) => {
                const accel = [...this._keys].find(([, k]) => k.action === action)?.[0];
                if (accel === TAKEOVER_KEY) {
                    // T toggles: take over the call, then T again hands it back.
                    if (this._takenOver)
                        this.Release();
                    else
                        this.Takeover();
                } else if (accel === MIC_KEY || accel === RECORD_KEY) {
                    this._onCameraKey(accel);
                }
            });
        this._monitorsId = Main.layoutManager.connect('monitors-changed',
            () => this._relayout());

        this._dbus = Gio.DBusExportedObject.wrapJSObject(IFACE_XML, this);
        this._dbus.export(Gio.DBus.session, OBJ_PATH);
        this._ownerId = Gio.bus_own_name_on_connection(Gio.DBus.session, BUS_NAME,
            Gio.BusNameOwnerFlags.REPLACE, null,
            () => console.warn(`classwatch: lost bus name ${BUS_NAME}`));
    }

    disable() {
        if (this._ownerId) {
            Gio.bus_unown_name(this._ownerId);
            this._ownerId = 0;
        }
        this._dbus?.unexport();
        this._dbus = null;

        this._cameraKeys = false;
        for (const id of [this._regrabId, this._forwardId]) {
            if (id)
                GLib.source_remove(id);
        }
        this._regrabId = this._forwardId = 0;
        for (const accel of [...this._keys.keys()])
            this._setKey(accel, false);
        this._vkbd = null;
        this._stopPulse();
        this._setUnredirect(true);
        if (this._acceleratorId) {
            global.display.disconnect(this._acceleratorId);
            this._acceleratorId = 0;
        }
        if (this._monitorsId) {
            Main.layoutManager.disconnect(this._monitorsId);
            this._monitorsId = 0;
        }
        for (const actor of [this._glow, this._panel]) {
            if (actor) {
                Main.layoutManager.removeChrome(actor);
                actor.destroy();
            }
        }
        this._glow = null;
        this._panel = null;
        this._scroll = null;
        this._qLabel = null;
        this._aLabel = null;
    }

    // ---- D-Bus methods ----

    SetAlert(on) {
        this._alertOn = on;
        if (on) {
            this._showGlow();
            this._setKey(TAKEOVER_KEY, true);
        } else {
            this._hideGlow();
            // The alert timing out must not take T away in the middle of a takeover.
            if (!this._takenOver)
                this._setKey(TAKEOVER_KEY, false);
        }
    }

    SetContent(question, answer) {
        this._question = question;
        this._answer = answer;
        this._qLabel.text = question;
        this._aLabel.text = answer;
        if (this._panel.visible)
            this._relayout();
    }

    Takeover() {
        this._alertOn = false;
        this._takenOver = true;
        this._hideGlow();
        this._setKey(TAKEOVER_KEY, true);  // stays grabbed so the next T hands the call back
        try {
            this._focusMeet();
        } catch (e) {
            console.error(`classwatch: focusing meet window failed: ${e}`);
        }
        this._showPanel();
        this._dbus?.emit_signal('TakeoverPressed', null);
    }

    Release() {
        this._takenOver = false;
        this._alertOn = false;
        this._hideGlow();
        this._panel.hide();
        this._setKey(TAKEOVER_KEY, false);
        try {
            this._minimizeMeet();
        } catch (e) {
            console.error(`classwatch: minimizing meet window failed: ${e}`);
        }
        this._syncUnredirect();
        this._dbus?.emit_signal('ReleasePressed', null);
    }

    Dismiss() {
        // Hides the panel only; during a takeover T keeps working to hand the call back.
        this._alertOn = false;
        this._hideGlow();
        if (!this._takenOver)
            this._setKey(TAKEOVER_KEY, false);
        this._panel.hide();
        this._syncUnredirect();
    }

    SetCameraKeys(on) {
        // `classwatch run --camera` turns these on, so Ctrl+D and Ctrl+P go untouched
        // when the camera is not in use.
        this._cameraKeys = on;
        this._micOpen = false;
        this._setCameraGrabs(on);
    }

    Ping() {
        return VERSION;
    }

    // ---- camera keys ----

    _onCameraKey(accel) {
        const inMeet = this._isMeet(global.display.focus_window);
        if (accel === MIC_KEY) {
            // Meet must still get Ctrl+D, or the mic would stop toggling.
            if (inMeet) {
                this._micOpen = !this._micOpen;
                this._dbus?.emit_signal(this._micOpen ? 'MicOpened' : 'MicClosed', null);
            }
            this._forward(MIC_KEY);
        } else if (inMeet) {
            // Swallowed in Meet: Chrome's print dialog would cover the call.
            this._dbus?.emit_signal('RecordPressed', null);
        } else {
            this._forward(RECORD_KEY);
        }
    }

    _setCameraGrabs(on) {
        for (const accel of [MIC_KEY, RECORD_KEY])
            this._setKey(accel, on);
    }

    // A grabbed accelerator never reaches the focused window, and there is no
    // pass-through flag. Watching keys without grabbing is not possible either: Mutter
    // hands them to the Wayland client before any extension event filter runs, and
    // reading /dev/input needs the input group. So: ungrab, resend the key through a
    // virtual keyboard a moment later, regrab once it has gone through.
    _forward(accel) {
        this._setCameraGrabs(false);
        this._forwardId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, FORWARD_DELAY_MS, () => {
            this._forwardId = 0;
            this._sendKey(accel);
            return GLib.SOURCE_REMOVE;
        });
    }

    _sendKey(accel) {
        try {
            this._vkbd ??= global.stage.context.get_backend().get_default_seat()
                .create_virtual_device(Clutter.InputDeviceType.KEYBOARD_DEVICE);
            // Ctrl is always sent with the letter: by now the user may have let go of it.
            // If they still hold it, Mutter drops the virtual Ctrl press and release as
            // duplicates of a key already down (seat-wide key count), so their Ctrl stays.
            const t = GLib.get_monotonic_time();
            const {PRESSED, RELEASED} = Clutter.KeyState;
            this._vkbd.notify_key(t, EVDEV.ctrl, PRESSED);
            this._vkbd.notify_key(t, EVDEV[accel], PRESSED);
            this._vkbd.notify_key(t, EVDEV[accel], RELEASED);
            this._vkbd.notify_key(t, EVDEV.ctrl, RELEASED);
        } catch (e) {
            console.error(`classwatch: resending ${accel} failed: ${e}`);
        }
        if (this._regrabId)
            GLib.source_remove(this._regrabId);
        this._regrabId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, REGRAB_MS, () => {
            this._regrabId = 0;
            if (this._cameraKeys)
                this._setCameraGrabs(true);
            return GLib.SOURCE_REMOVE;
        });
    }

    // ---- glow ----

    _buildGlow() {
        // Four gradient strips inside a monitor-sized, non-reactive container.
        // Shell 51 dropped the affectsInputRegion chrome param (Wayland-only: input
        // goes to whatever reactive actor is picked), so reactive:false on every
        // actor here is what keeps the glow click-through. trackFullscreen stays
        // off so it remains visible over fullscreen windows.
        this._glow = new St.Widget({reactive: false, visible: false, name: 'classwatchGlow'});
        this._edges = {};
        for (const side of ['top', 'bottom', 'left', 'right']) {
            const edge = new St.Widget({reactive: false, style_class: `classwatch-edge-${side}`});
            this._edges[side] = edge;
            this._glow.add_child(edge);
        }
        Main.layoutManager.addTopChrome(this._glow, {
            affectsStruts: false,
            trackFullscreen: false,
        });
        this._relayout();
    }

    _showGlow() {
        this._relayout();
        this._glow.opacity = 255;
        this._glow.show();
        Main.layoutManager.uiGroup.set_child_above_sibling(this._glow, null);
        this._startPulse();
        this._syncUnredirect();
    }

    _hideGlow() {
        this._stopPulse();
        this._glow.hide();
        this._syncUnredirect();
    }

    // A fullscreen window (Meet after takeover) gets scanned out directly, bypassing the
    // compositor, and then nothing drawn by the shell shows on top of it: the panel
    // vanished the moment Meet went fullscreen. Keep compositing on while we draw.
    _syncUnredirect() {
        const need = Boolean(this._glow?.visible || this._panel?.visible);
        this._setUnredirect(!need);
    }

    _setUnredirect(allowed) {
        if (allowed === !this._unredirectOff)
            return;
        try {
            if (allowed)
                global.compositor.enable_unredirect();
            else
                global.compositor.disable_unredirect();
            this._unredirectOff = !allowed;
        } catch (e) {
            console.error(`classwatch: toggling unredirect failed: ${e}`);
        }
    }

    _startPulse() {
        if (this._pulseId)
            return;
        let up = false;
        const step = () => {
            this._glow.ease({
                opacity: up ? 255 : 160,
                duration: 900,
                mode: Clutter.AnimationMode.EASE_IN_OUT_SINE,
            });
            up = !up;
            return GLib.SOURCE_CONTINUE;
        };
        step();
        this._pulseId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 950, step);
    }

    _stopPulse() {
        if (this._pulseId) {
            GLib.source_remove(this._pulseId);
            this._pulseId = 0;
        }
        this._glow?.remove_all_transitions();
    }

    // ---- panel ----

    _buildPanel() {
        this._panel = new St.BoxLayout({
            style_class: 'classwatch-panel',
            orientation: Clutter.Orientation.VERTICAL,
            reactive: true,
            visible: false,
            name: 'classwatchPanel',
        });
        this._qLabel = new St.Label({style_class: 'classwatch-question', text: ''});
        this._aLabel = new St.Label({style_class: 'classwatch-answer', text: ''});
        for (const label of [this._qLabel, this._aLabel]) {
            label.clutter_text.line_wrap = true;
            label.clutter_text.line_wrap_mode = Pango.WrapMode.WORD_CHAR;
            label.clutter_text.ellipsize = Pango.EllipsizeMode.NONE;
        }
        const inner = new St.BoxLayout({orientation: Clutter.Orientation.VERTICAL});
        inner.add_child(this._qLabel);
        inner.add_child(this._aLabel);

        this._scroll = new St.ScrollView({
            hscrollbar_policy: St.PolicyType.NEVER,
            vscrollbar_policy: St.PolicyType.AUTOMATIC,
            overlay_scrollbars: true,
            x_expand: true,
            y_expand: true,
        });
        this._scroll.child = inner;
        this._panel.add_child(this._scroll);
        this._panel.add_child(new St.Label({
            style_class: 'classwatch-hint',
            text: 'click to dismiss',
        }));

        // Click anywhere on the panel dismisses; scroll events still reach the ScrollView.
        this._panel.connect('button-release-event', () => {
            this.Dismiss();
            return Clutter.EVENT_STOP;
        });
        this._panel.connect('touch-event', (_a, event) => {
            if (event.type() === Clutter.EventType.TOUCH_END)
                this.Dismiss();
            return Clutter.EVENT_PROPAGATE;
        });

        Main.layoutManager.addTopChrome(this._panel, {
            affectsStruts: false,
            trackFullscreen: false,
        });
    }

    _showPanel() {
        this._relayout();
        this._panel.show();
        Main.layoutManager.uiGroup.set_child_above_sibling(this._panel, null);
        this._scroll.vadjustment.value = 0;
        this._syncUnredirect();
    }

    _relayout() {
        const m = Main.layoutManager.primaryMonitor;
        if (!m || !this._glow)
            return;
        this._glow.set_position(m.x, m.y);
        this._glow.set_size(m.width, m.height);
        const g = GLOW_PX;
        this._edges.top.set_position(0, 0);
        this._edges.top.set_size(m.width, g);
        this._edges.bottom.set_position(0, m.height - g);
        this._edges.bottom.set_size(m.width, g);
        this._edges.left.set_position(0, 0);
        this._edges.left.set_size(g, m.height);
        this._edges.right.set_position(m.width - g, 0);
        this._edges.right.set_size(g, m.height);

        if (this._panel) {
            const width = Math.round(m.width * 0.9);
            this._panel.set_width(width);
            this._panel.set_position(m.x + Math.round((m.width - width) / 2), m.y);
            // Cap the scroll area, not the panel: the panel then sizes to its content
            // (short answers stay compact) and never exceeds ~45% of the monitor.
            // 70px leaves room for the panel padding and the hint line.
            const maxH = Math.round(m.height * PANEL_MAX_FRACTION) - 70;
            this._scroll.style = `max-height: ${maxH}px;`;
        }
    }

    // ---- key grab ----

    _setKey(accel, on) {
        const k = this._keys.get(accel);
        if (on && !k) {
            // The camera keys fire on release: Mutter drops a virtual key press while the
            // same physical key is still down (seat-wide key count), so the copy _forward
            // sends would never arrive if it went out on the press.
            const flags = accel === TAKEOVER_KEY ? Meta.KeyBindingFlags.NONE
                : Meta.KeyBindingFlags.TRIGGER_RELEASE | Meta.KeyBindingFlags.IGNORE_AUTOREPEAT;
            const action = global.display.grab_accelerator(accel, flags);
            if (action === Meta.KeyBindingAction.NONE) {
                console.error(`classwatch: grab_accelerator("${accel}") failed`);
                return;
            }
            const name = Meta.external_binding_name_for_action(action);
            Main.wm.allowKeybinding(name, Shell.ActionMode.ALL);
            this._keys.set(accel, {action, name});
        } else if (!on && k) {
            global.display.ungrab_accelerator(k.action);
            Main.wm.allowKeybinding(k.name, Shell.ActionMode.NONE);
            this._keys.delete(accel);
        }
    }

    // ---- meet window ----

    _isMeetById(w) {
        return [w.get_wm_class(), w.get_wm_class_instance(),
            w.get_gtk_application_id(), w.get_sandboxed_app_id()]
            .filter(Boolean).some(s => s.toLowerCase().includes('classwatch-meet'));
    }

    _isMeetByTitle(w) {
        const t = w.get_title() ?? '';
        return t.includes('Meet') || t.includes('meet.google.com');
    }

    _isMeet(w) {
        return Boolean(w) && (this._isMeetById(w) || this._isMeetByTitle(w));
    }

    _findMeet() {
        const windows = global.get_window_actors()
            .map(a => a.meta_window)
            .filter(w => w && w.get_window_type() === Meta.WindowType.NORMAL);
        return windows.find(w => this._isMeetById(w)) ??
            windows.find(w => this._isMeetByTitle(w)) ?? null;
    }

    _minimizeMeet() {
        const win = this._findMeet();
        if (!win)
            return;
        if (win.is_fullscreen())
            win.unmake_fullscreen();
        win.minimize();
    }

    _focusMeet() {
        const win = this._findMeet();
        if (!win) {
            console.log('classwatch: no Meet window found');
            return;
        }
        if (win.minimized)
            win.unminimize();
        Main.activateWindow(win);
        if (!win.is_fullscreen())
            win.make_fullscreen();
    }
}
