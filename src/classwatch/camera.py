"""Fake webcam: a PipeWire virtual camera that plays a prerecorded clip of Bogdan.

When Bogdan takes over (presses T), the fake link collapses: frames turn blocky,
the picture updates a few times a second and freezes now and then, and while it
looks broken the webcam silently replaces the clip; the link then recovers
slowly. Pressing T again does the reverse and goes back to the clip. The link
also dips briefly at every loop seam, so the jump cut reads as a network hiccup.

Lag is faked by showing fewer distinct frames, never by delaying them: sources
run in real time and only their newest frame is kept, so nothing plays in slow
motion (sleeping in the frame loop did exactly that to the webcam).

Architecture (no kernel module, no sudo):
  ClipSource (PyAV, real-time) or WebcamSource (ffmpeg v4l2)  -> newest frame
      ── CameraStream: steady 30 Hz, Link quality q decides fps/blocks/freezes
      ── rawvideo pipe (fd 0) ──▶ gst-launch-1.0
      ──▶ pipewiresink mode=provide ──▶ PipeWire node "classwatch"

The node advertises media.class=Video/Source, media.role=Camera, so it shows up
in `wpctl status` under Video > Sources next to the real webcam. Chrome only
sees PipeWire cameras with --enable-features=WebRtcPipeWireCamera (added in
`classwatch meet`), and only lists the cameras present when Chrome starts.

That flag routes Chrome through the xdg camera portal. GNOME refuses to show
the portal's permission dialog for Chrome ("Only the focused app is allowed to
show a system access dialog"), so without a stored grant Chrome waits forever
and Meet's Settings and Join buttons silently do nothing. `grant_portal_camera`
stores that grant up front.
"""
from __future__ import annotations

import fcntl
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

import av
import numpy as np

STATE_DIR = pathlib.Path(os.environ.get("XDG_STATE_HOME", pathlib.Path.home() / ".local/state")) / "classwatch"
CLIP_FILE = pathlib.Path(os.environ.get("CLASSWATCH_CAM_CLIP", STATE_DIR / "clip.mp4"))
RES = (640, 480)
FPS = 30
RECOVER_SECS = 20.0  # takeover: webcam quality climbs back to clean over this long
LAG_SECS = 6.0  # takeover: how long the bad-connection phase lasts before the swap
RESTORE_LAG_SECS = 3.0  # back to the clip: bad link this long before the swap
RESTORE_RECOVER_SECS = 6.0
SEAM_LEAD_SECS = 0.5  # loop seam: the link starts dipping this long before the clip wraps
SEAM_RECOVER_SECS = 1.6
CAM_NAME = "classwatch"
STREAM_PROPS = "props,media.class=Video/Source,media.role=Camera,node.description=classwatch"
CONTROL_FILE = STATE_DIR / "cam-control"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / "camera.log", "a") as f:
        f.write(line + "\n")


def gst_available() -> bool:
    return subprocess.run(["gst-launch-1.0", "--version"], capture_output=True).returncode == 0


def node_in_wpctl(name: str = CAM_NAME) -> bool:
    """True if a PipeWire node with this name is currently registered."""
    r = subprocess.run(["wpctl", "status"], capture_output=True, text=True)
    return name in r.stdout


# The portal derives a host app's id from its systemd scope (app-com.google.Chrome-N.scope);
# an empty id, which older portals used for unsandboxed apps, is not consulted.
PORTAL_APP_ID = "com.google.Chrome"
_PERM_STORE = ["gdbus", "call", "--session", "--dest", "org.freedesktop.impl.portal.PermissionStore",
               "--object-path", "/org/freedesktop/impl/portal/PermissionStore", "--method"]
_PERM_IFACE = "org.freedesktop.impl.portal.PermissionStore"


def portal_grant_present(lookup_output: str, app_id: str = PORTAL_APP_ID) -> bool:
    """Parse `PermissionStore.Lookup devices camera` output for a 'yes' grant to app_id."""
    return f"'{app_id}': ['yes']" in lookup_output


def grant_portal_camera(app_id: str = PORTAL_APP_ID) -> str:
    """Make sure the camera portal will not need a dialog for Chrome. Returns what happened."""
    r = subprocess.run(_PERM_STORE + [f"{_PERM_IFACE}.Lookup", "devices", "camera"],
                       capture_output=True, text=True)
    if r.returncode == 0 and portal_grant_present(r.stdout, app_id):
        return "already granted"
    r = subprocess.run(_PERM_STORE + [f"{_PERM_IFACE}.SetPermission", "devices", "true", "camera",
                                      app_id, "['yes']"], capture_output=True, text=True)
    if r.returncode != 0:
        return f"could not grant ({r.stderr.strip() or r.returncode})"
    return "granted"


class Link:
    """Fake network quality q in [0, 1] (1 = clean), made of overlapping dips.

    A dip holds q at 1 - depth for `hold` seconds, then recovers linearly over
    `recover` seconds. The lowest active dip wins, so a takeover dip is not
    shortened by a loop-seam dip that happens to start inside it.
    """

    def __init__(self) -> None:
        self._dips: list[tuple[float, float, float, float]] = []
        self._was_dipped = False

    def dip(self, now: float, depth: float, hold: float, recover: float) -> None:
        self._dips.append((now, depth, hold, recover))

    def q(self, now: float) -> float:
        q = 1.0
        live = []
        for start, depth, hold, recover in self._dips:
            t = now - start
            if t < 0:
                live.append((start, depth, hold, recover))
                continue
            if t < hold:
                v = 1.0 - depth
            elif t < hold + recover:
                v = 1.0 - depth * (1.0 - (t - hold) / recover)
            else:
                continue
            live.append((start, depth, hold, recover))
            q = min(q, v)
        self._dips = live
        return q

    def recovered(self, now: float) -> bool:
        """True exactly once each time the link returns to clean after a dip."""
        q = self.q(now)
        if q < 1.0:
            self._was_dipped = True
            return False
        if self._was_dipped:
            self._was_dipped = False
            return True
        return False


def out_fps(q: float) -> float:
    """Distinct frames per second shown at quality q: 4 when awful, 30 when clean."""
    return 4.0 + (FPS - 4.0) * max(0.0, min(1.0, q))


def render(img: np.ndarray, q: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Make a frame look like it went through a link of quality q (blocks + noise)."""
    if q >= 0.97:
        return img
    rng = rng or np.random.default_rng()
    block = round(1 + (1.0 - q) * 9)
    h, w = img.shape[:2]
    small = img[::block, ::block, :].astype(np.float32)
    out = np.repeat(np.repeat(small, block, axis=0), block, axis=1)[:h, :w]
    noise = (rng.random((h, w, 1), dtype=np.float32) - 0.5) * (24.0 * (1.0 - q))
    return np.clip(out + noise, 0, 255).astype(np.uint8)


class _Source:
    """A frame producer running in real time on its own thread; keeps only the newest frame.

    Keeping only the newest frame is the point: the old design slept inside the
    frame loop to fake lag, so the webcam's frames queued up and played back in
    slow motion. Here a slow consumer just skips frames.
    """

    def __init__(self) -> None:
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> _Source:
        self._thread.start()
        return self

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def _put(self, img: np.ndarray) -> None:
        with self._lock:
            self._latest = img

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        raise NotImplementedError


class ClipSource(_Source):
    """Loops a clip at its real speed. Calls on_seam(lead) shortly before each wrap."""

    def __init__(self, path: pathlib.Path, on_seam=None) -> None:
        super().__init__()
        self.path = path
        self.on_seam = on_seam

    def _loop(self) -> None:
        frame_secs = 1.0 / FPS
        while not self._stop.is_set():
            # Open, decode and close all on this thread: closing a container while
            # another thread decodes from it is what segfaulted before.
            with av.open(str(self.path)) as c:
                vs = c.streams.video[0]
                dur = float(vs.duration * vs.time_base) if vs.duration else (
                    float(c.duration) / 1e6 if c.duration else 0.0)
                seam_fired = False
                next_t = time.monotonic()
                for frame in c.decode(video=0):
                    if self._stop.is_set():
                        return
                    img = frame.to_ndarray(format="bgr24")
                    if img.shape[1] != RES[0] or img.shape[0] != RES[1]:
                        img = frame.reformat(width=RES[0], height=RES[1]).to_ndarray(format="bgr24")
                    self._put(img)
                    t = float(frame.time) if frame.time is not None else 0.0
                    if self.on_seam and not seam_fired and dur and t >= dur - SEAM_LEAD_SECS:
                        seam_fired = True
                        self.on_seam()
                    next_t += frame_secs
                    sleep_for = next_t - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    else:
                        next_t = time.monotonic()
                if self.on_seam and not seam_fired:
                    self.on_seam()  # duration unknown: dip at the wrap itself


class WebcamSource(_Source):
    """Reads the real webcam through ffmpeg as fast as it delivers."""

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        self._proc: subprocess.Popen | None = None

    def _loop(self) -> None:
        self._proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "v4l2", "-framerate", str(FPS), "-input_format", "mjpeg",
             "-video_size", f"{RES[0]}x{RES[1]}", "-i", self.device,
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        n = RES[0] * RES[1] * 3
        assert self._proc.stdout is not None
        while not self._stop.is_set():
            data = self._proc.stdout.read(n)
            if not data or len(data) < n:
                break
            self._put(np.frombuffer(data, np.uint8).reshape(RES[1], RES[0], 3))

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.kill()  # unblocks the read and frees the webcam
        super().stop()


class CameraStream:
    """Feeds the PipeWire camera node at a steady 30 Hz from the current source.

    Modes: clip (the prerecorded loop), degrade (takeover: link collapses, then
    the webcam), real (the webcam), restore (link collapses, then back to the clip).
    """

    def __init__(self, real_device: str = "/dev/video0"):
        self.real_device = real_device
        self.mode = "clip"
        self.clip_path: pathlib.Path | None = None
        self.link = Link()
        self._source: _Source | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._timers: list[threading.Timer] = []
        self._pipe_fd = -1
        self._feeder: subprocess.Popen | None = None
        self._rng = np.random.default_rng()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---- control ----------------------------------------------------------

    def start(self, clip: str | None = None) -> None:
        """Start playing the prerecorded clip (or the real cam if no clip file)."""
        if not gst_available():
            raise RuntimeError("gst-launch-1.0 not found (install gstreamer1.0-plugins-base)")
        clip_path = pathlib.Path(clip) if clip else CLIP_FILE
        if clip_path.exists():
            self.clip_path = clip_path
            self._use(self._clip_source())
            self.mode = "clip"
            log(f"camera: playing clip {clip_path.name}")
        else:
            self._use(WebcamSource(self.real_device))
            self.mode = "real"
            log(f"camera: no clip at {clip_path}, using real webcam {self.real_device}")

    def degrade(self) -> None:
        """Takeover: the link collapses for LAG_SECS, the webcam replaces the clip, then recovers."""
        if self.mode in ("degrade", "real"):
            return
        now = time.monotonic()
        self.link.dip(now, depth=0.9, hold=LAG_SECS + 1.0, recover=RECOVER_SECS)
        self.mode = "degrade"
        log("camera: DEGRADE (link collapses, swapping to the real cam)")
        self._later(LAG_SECS, self._swap_to_real)

    def restore(self) -> None:
        """Back to the clip: the link collapses again so the cut is hidden, then recovers."""
        if self.mode in ("clip", "restore") or not self.clip_path:
            return
        now = time.monotonic()
        self.link.dip(now, depth=0.85, hold=RESTORE_LAG_SECS + 0.8, recover=RESTORE_RECOVER_SECS)
        self.mode = "restore"
        log("camera: RESTORE (link collapses, swapping back to the clip)")
        self._later(RESTORE_LAG_SECS, self._swap_to_clip)

    def stop(self) -> None:
        self._stop.set()
        for t in self._timers:
            t.cancel()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        with self._lock:
            if self._source:
                self._source.stop()
                self._source = None
        if self._pipe_fd >= 0:
            try:
                os.close(self._pipe_fd)
            except OSError:
                pass
            self._pipe_fd = -1
        if self._feeder:
            try:
                self._feeder.wait(timeout=2)
            except Exception:
                self._feeder.kill()
            self._feeder = None
        log("camera: stopped")

    # ---- sources ------------------------------------------------------------

    def _clip_source(self) -> ClipSource:
        assert self.clip_path is not None
        return ClipSource(self.clip_path, on_seam=self._on_seam)

    def _on_seam(self) -> None:
        if self.mode == "clip":
            self.link.dip(time.monotonic(), depth=0.6,
                          hold=SEAM_LEAD_SECS + 0.4, recover=SEAM_RECOVER_SECS)

    def _use(self, src: _Source) -> None:
        src.start()
        with self._lock:
            old, self._source = self._source, src
        if old:
            old.stop()

    def _later(self, secs: float, fn) -> None:
        t = threading.Timer(secs, fn)
        t.daemon = True
        self._timers.append(t)
        t.start()

    def _swap_to_real(self) -> None:
        if self.mode != "degrade" or self._stop.is_set():
            return
        self._use(WebcamSource(self.real_device))
        self.mode = "real"
        log("camera: switched to REAL webcam, link recovering")

    def _swap_to_clip(self) -> None:
        if self.mode != "restore" or self._stop.is_set():
            return
        self._use(self._clip_source())
        self.mode = "clip"
        log("camera: switched back to the CLIP, link recovering")

    # ---- output -------------------------------------------------------------

    def _run(self) -> None:
        """Steady 30 Hz to gst; the link decides how often the shown frame changes."""
        tick = 1.0 / FPS
        next_t = time.monotonic()
        shown: np.ndarray | None = None
        next_change = 0.0
        frozen_until = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            q = self.link.q(now)
            if self.link.recovered(now):
                log("camera: link fully recovered")
            if q < 0.3 and now >= frozen_until and self._rng.random() < 0.01:
                frozen_until = now + float(self._rng.uniform(0.3, 1.2))  # a lag spike
            # Half a tick of slack: deadlines computed from a jittery `now` otherwise miss
            # every other 30 Hz tick, and a clean clip judders at ~18 fps.
            if now + tick / 2 >= next_change and now >= frozen_until:
                with self._lock:
                    src = self._source
                img = src.latest() if src else None
                if img is not None:
                    shown = render(img, q, self._rng)
                    next_change = max(next_change + 1.0 / out_fps(q), now - tick)
            if shown is not None:
                self._write_frame(shown.tobytes())
            next_t += tick
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.monotonic()

    def _write_frame(self, frame: bytes) -> None:
        """Push one BGR frame into the gst feeder's stdin."""
        if self._pipe_fd < 0:
            if self._stop.is_set():
                return
            self._spawn_feeder()
        try:
            os.write(self._pipe_fd, frame)
        except (BrokenPipeError, OSError) as e:
            if self._stop.is_set():
                return
            log(f"camera: pipe write failed ({e!r}); restarting feeder")
            self._spawn_feeder()

    def _spawn_feeder(self) -> None:
        """Long-lived gst-launch: raw BGR on stdin -> PipeWire camera node."""
        if self._feeder:
            try:
                self._feeder.kill()
            except Exception:
                pass
        r, pipe_w = os.pipe()
        self._pipe_fd = pipe_w
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        errlog = open(STATE_DIR / "cam-feeder.err", "ab")
        width, height = RES
        self._feeder = subprocess.Popen(
            ["gst-launch-1.0", "-q",
             "fdsrc", "fd=0", f"blocksize={width*height*3}",
             "!",
             "rawvideoparse", f"width={width}", f"height={height}", "format=bgr", f"framerate={FPS}/1",
             "!",
             "videoconvert",
             "!",
             # Offer I420, not the BGR we feed in: Chrome read the node's packed 24-bit
             # BGR as RGB, so red and blue swapped and skin came out blue in Meet.
             "video/x-raw,format=I420",
             "!",
             "pipewiresink", "mode=provide", f"client-name={CAM_NAME}-cam",
             f"stream-properties={STREAM_PROPS}"],
            stdin=r, stdout=subprocess.DEVNULL, stderr=errlog)
        os.close(r)
        log("camera: gst feeder started (PipeWire node 'classwatch')")


def cmd_camera(action: str, clip: str | None = None, real: str = "/dev/video0") -> int:
    """Manage the fake camera: up / down / degrade / restore."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if action == "up":
        # One camera at a time: two would publish two "classwatch" nodes and race for
        # the control file.
        lock = open(STATE_DIR / "camera.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("classwatch: a `camera up` is already running (stop it with `classwatch camera down`)",
                  file=sys.stderr)
            return 1
        # Ctrl-C and systemctl stop (SIGTERM) take the same clean path. SIGINT is set
        # explicitly because a shell starting us in the background leaves it ignored.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        cam = CameraStream(real_device=real)
        try:
            cam.start(clip)
        except Exception as e:
            log(f"camera: {e}")
            cam.stop()
            return 1
        # `classwatch run` asks for degrade (takeover) and restore (back to the clip) here.
        try:
            while True:
                try:
                    cmd = CONTROL_FILE.read_text().strip()
                    CONTROL_FILE.unlink()
                except FileNotFoundError:
                    cmd = ""
                if cmd:
                    if cmd == "degrade":
                        cam.degrade()
                    elif cmd == "restore":
                        cam.restore()
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            cam.stop()
        return 0
    if action == "down":
        # Stop `camera up` first (SIGINT = its clean Ctrl-C path); killing only the
        # gst feeder just makes `up` respawn it.
        subprocess.run(["pkill", "-TERM", "-f", "classwatch(.cli)? camera up"], capture_output=True)
        time.sleep(2.5)
        subprocess.run(["pkill", "-f", f"client-name={CAM_NAME}-cam"], capture_output=True)
        log("camera: stopped PipeWire camera node")
        return 0
    if action in ("degrade", "restore"):
        CONTROL_FILE.write_text(action)
        log(f"camera: {action} requested")
        return 0
    return 1
