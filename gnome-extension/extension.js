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

const VERSION = 'classwatch-gnome-extension 1';
const BUS_NAME = 'io.github.bogdanstamenovic.Classwatch';
const OBJ_PATH = '/io/github/bogdanstamenovic/Classwatch';
const GLOW_PX = 40;
const PANEL_MAX_FRACTION = 0.45;

const IFACE_XML = `
<node>
  <interface name="io.github.bogdanstamenovic.Classwatch">
    <method name="SetAlert"><arg type="b" name="on" direction="in"/></method>
    <method name="SetContent">
      <arg type="s" name="question" direction="in"/>
      <arg type="s" name="answer" direction="in"/>
    </method>
    <method name="Takeover"/>
    <method name="Dismiss"/>
    <method name="Ping"><arg type="s" name="version" direction="out"/></method>
    <signal name="TakeoverPressed"/>
  </interface>
</node>`;

export default class ClasswatchExtension extends Extension {
    enable() {
        this._question = '';
        this._answer = '';
        this._alertOn = false;
        this._grabAction = 0;
        this._pulseId = 0;

        this._buildGlow();
        this._buildPanel();

        this._acceleratorId = global.display.connect('accelerator-activated',
            (_display, action) => {
                if (this._grabAction && action === this._grabAction)
                    this.Takeover();
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

        this._setGrab(false);
        this._stopPulse();
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
            this._setGrab(true);
        } else {
            this._hideGlow();
            this._setGrab(false);
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
        this._hideGlow();
        this._setGrab(false);
        try {
            this._focusMeet();
        } catch (e) {
            console.error(`classwatch: focusing meet window failed: ${e}`);
        }
        this._showPanel();
        this._dbus?.emit_signal('TakeoverPressed', null);
    }

    Dismiss() {
        this._alertOn = false;
        this._hideGlow();
        this._setGrab(false);
        this._panel.hide();
    }

    Ping() {
        return VERSION;
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
    }

    _hideGlow() {
        this._stopPulse();
        this._glow.hide();
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

    _setGrab(on) {
        if (on && !this._grabAction) {
            const action = global.display.grab_accelerator('t', Meta.KeyBindingFlags.NONE);
            if (action === Meta.KeyBindingAction.NONE) {
                console.error('classwatch: grab_accelerator("t") failed');
                return;
            }
            this._grabAction = action;
            this._grabName = Meta.external_binding_name_for_action(action);
            Main.wm.allowKeybinding(this._grabName, Shell.ActionMode.ALL);
        } else if (!on && this._grabAction) {
            global.display.ungrab_accelerator(this._grabAction);
            Main.wm.allowKeybinding(this._grabName, Shell.ActionMode.NONE);
            this._grabAction = 0;
            this._grabName = null;
        }
    }

    // ---- meet window ----

    _focusMeet() {
        const windows = global.get_window_actors()
            .map(a => a.meta_window)
            .filter(w => w && w.get_window_type() === Meta.WindowType.NORMAL);
        const ids = w => [w.get_wm_class(), w.get_wm_class_instance(),
            w.get_gtk_application_id(), w.get_sandboxed_app_id()]
            .filter(Boolean).map(s => s.toLowerCase());
        let win = windows.find(w => ids(w).some(s => s.includes('classwatch-meet')));
        if (!win) {
            win = windows.find(w => {
                const t = w.get_title() ?? '';
                return t.includes('Meet') || t.includes('meet.google.com');
            });
        }
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
