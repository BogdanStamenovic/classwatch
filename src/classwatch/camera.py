"""Fake webcam: a PipeWire virtual camera that plays a prerecorded clip of Bogdan.

Switching sources (T, Ctrl+D, Ctrl+P) plays a Meet connection drop replayed from a
measured trace (classwatch.degrade): the picture freezes, the source swaps inside
that freeze (0.25-0.6 s after the request), then the picture comes back soft and
smeared through a real VP9 encoder and climbs back over 10-25 s. Loop seams get a
short freeze over the wrap and a few seconds of mild smear. By default the camera
always runs at a mediocre baseline (a little soft, an occasional stutter), so a
drop never stands out against a perfect picture.

`record` (Ctrl+P in Meet) swaps to the webcam the same way, records RECORD_SECS of
it as the new clip while the call keeps showing the live webcam, then swaps back
to the clip. The recorder taps the frames the live feed already reads: only one
process can stream /dev/video0, so a second ffmpeg would fight the call for it.

Lag is shown by holding frames, never by delaying them: sources run in real time
and only their newest frame is kept, so nothing plays in slow motion (sleeping in
the frame loop did exactly that to the webcam).

Architecture (no kernel module, no sudo):
  ClipSource (PyAV, real-time) or WebcamSource (ffmpeg v4l2)  -> newest frame
      ── CameraStream: steady 30 Hz; Degrader (VP9 at the drop's or baseline's quality)
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

from .degrade import CLEAN, Baseline, Degrader, MeetDrop

STATE_DIR = pathlib.Path(os.environ.get("XDG_STATE_HOME", pathlib.Path.home() / ".local/state")) / "classwatch"
CLIP_FILE = pathlib.Path(os.environ.get("CLASSWATCH_CAM_CLIP", STATE_DIR / "clip.mp4"))
RES = (640, 480)
FPS = 30
SEAM_LEAD_SECS = 0.3  # loop seam: the freeze starts this long before the clip wraps
SEAM_FREEZE_SECS = 0.7  # ...and lasts this long, so it covers the wrap
BASELINE = os.environ.get("CLASSWATCH_CAM_BASELINE", "1") != "0"
RECORD_SECS = 60.0
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

    Modes: clip (the prerecorded loop), degrade (a drop has started, the webcam
    swaps in during its freeze), real (the webcam), restore (a drop, then back to
    the clip). A recording runs alongside "real" and ends with a restore.
    """

    def __init__(self, real_device: str = "/dev/video0", baseline: bool = BASELINE):
        self.real_device = real_device
        self.mode = "clip"
        self.clip_path: pathlib.Path | None = None
        self._drop: MeetDrop | None = None
        self._drop_t0 = 0.0
        self._source: _Source | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._timers: list[threading.Timer] = []
        self._pipe_fd = -1
        self._feeder: subprocess.Popen | None = None
        self._rng = np.random.default_rng()
        self._baseline = Baseline(self._rng) if baseline else None
        self._degrader = Degrader(*RES, rng=self._rng)
        self._record_pending: float | None = None  # seconds to record once the webcam is live
        self._record_cancel: threading.Event | None = None  # set while a recording runs
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
        """A drop starts; the webcam replaces the clip inside its first freeze."""
        if self.mode in ("degrade", "real"):
            return
        drop = self._start_drop(MeetDrop(self._rng))
        self.mode = "degrade"
        log(f"camera: DEGRADE ({drop.name}), webcam swaps in at {drop.swap_at:.2f}s, "
            f"back to normal in {drop.dur:.0f}s")
        self._later(drop.swap_at, self._swap_to_real)

    def restore(self) -> None:
        """Back to the clip: a drop starts and the clip swaps in inside its first freeze.

        Cancels a recording in progress (the old clip stays): going back to the clip
        stops the webcam, so the recording would only freeze on its last frame.
        """
        self._record_pending = None
        if self._record_cancel and not self._record_cancel.is_set():
            self._record_cancel.set()
            log("camera: recording cancelled, keeping the old clip")
        if self.mode in ("clip", "restore") or not self.clip_path:
            return
        drop = self._start_drop(MeetDrop(self._rng))
        self.mode = "restore"
        log(f"camera: RESTORE ({drop.name}), clip swaps in at {drop.swap_at:.2f}s")
        self._later(drop.swap_at, self._swap_to_clip)

    def record(self, secs: float = RECORD_SECS) -> None:
        """Record `secs` of webcam as the new clip, then go back to the clip.

        From the clip this first degrades like a takeover; the call shows the live
        webcam for the whole recording.
        """
        if self._record_cancel and not self._record_cancel.is_set():
            log("camera: already recording, ignoring")
            return
        if self.mode == "restore":
            # Still on the webcam (the swap back has not happened yet): stay there.
            self.mode = "real"
            log("camera: swap back to the clip called off for the recording")
        if self.mode == "real":
            self._start_recording(secs)
            return
        self._record_pending = secs
        self.degrade()  # no-op if already degrading; _swap_to_real starts the recording

    def stop(self) -> None:
        self._stop.set()
        if self._record_cancel:
            self._record_cancel.set()
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
            # Replaces a drop still recovering: the freeze over the wrap matters more.
            self._start_drop(MeetDrop(self._rng, "loss-burst", severity=0.45, max_secs=4.0,
                                      lead_freeze=SEAM_FREEZE_SECS))

    def _start_drop(self, drop: MeetDrop) -> MeetDrop:
        with self._lock:
            self._drop, self._drop_t0 = drop, time.monotonic()
        return drop

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
        log("camera: switched to REAL webcam")
        if self._record_pending is not None:
            self._start_recording(self._record_pending)

    def _swap_to_clip(self) -> None:
        if self.mode != "restore" or self._stop.is_set():
            return
        self._use(self._clip_source())
        self.mode = "clip"
        log("camera: switched back to the CLIP")

    # ---- recording ------------------------------------------------------------

    def _start_recording(self, secs: float) -> None:
        self._record_pending = None
        cancel = threading.Event()
        self._record_cancel = cancel
        threading.Thread(target=self._record, args=(secs, cancel), daemon=True).start()

    def _record(self, secs: float, cancel: threading.Event) -> None:
        """Sample the live source at a steady FPS into a new clip, swap it in, restore.

        Sampled on a clock rather than written per webcam frame: ClipSource paces
        playback by frame count, so a webcam that delivered 24 fps would otherwise
        give a clip that plays fast.
        """
        target = self.clip_path or CLIP_FILE
        tmp = target.with_name(target.stem + ".new.mp4")
        n = round(secs * FPS)
        log(f"camera: RECORDING {secs:.0f}s of webcam into {target.name}")
        written = 0
        try:
            with av.open(str(tmp), "w") as out:
                st = out.add_stream("libx264", rate=FPS)
                st.width, st.height = RES
                st.pix_fmt = "yuv420p"
                st.options = {"preset": "veryfast", "crf": "20"}
                tick = 1.0 / FPS
                next_t = time.monotonic()
                while written < n and not cancel.is_set() and not self._stop.is_set():
                    with self._lock:
                        src = self._source
                    img = src.latest() if src else None
                    if img is not None:  # the webcam's first frame can take a moment
                        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                        frame.pts = written
                        for p in st.encode(frame):
                            out.mux(p)
                        written += 1
                        if written % (FPS * 15) == 0:
                            log(f"camera: recording {written // FPS}s / {secs:.0f}s")
                    next_t += tick
                    sleep_for = next_t - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    else:
                        next_t = time.monotonic()
                for p in st.encode(None):
                    out.mux(p)
        except Exception as e:
            log(f"camera: recording failed ({e!r}), keeping the old clip")
            tmp.unlink(missing_ok=True)
            cancel.set()
            return
        if written < n:
            tmp.unlink(missing_ok=True)
            return
        if target.exists():
            os.replace(target, target.with_name(target.stem + ".prev.mp4"))
        os.replace(tmp, target)
        self.clip_path = target
        cancel.set()
        log(f"camera: recorded {written} frames; new clip saved (old one kept as {target.stem}.prev.mp4)")
        self.restore()

    # ---- output -------------------------------------------------------------

    def _run(self) -> None:
        """Steady 30 Hz to gst. The degrader decides what each tick shows: the newest source
        frame through the codec at the current drop's (or the baseline's) quality, or a held
        frame while the picture is frozen or between the drop's sparser frames."""
        tick = 1.0 / FPS
        next_t = time.monotonic()
        shown: np.ndarray | None = None
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                src, drop, t0 = self._source, self._drop, self._drop_t0
            img = src.latest() if src else None
            if img is not None:
                rest = self._baseline.at(now) if self._baseline else CLEAN
                if drop is not None and now - t0 >= drop.dur:
                    with self._lock:
                        if self._drop is drop:
                            self._drop = None
                    log("camera: drop over")
                    drop = None
                state = drop.at(now - t0, rest) if drop is not None else rest
                try:
                    shown = self._degrader.frame(img, now, state)
                except Exception as e:  # a codec hiccup must not take the camera down
                    log(f"camera: degrader failed ({e!r}); showing the clean frame")
                    self._degrader.reset()
                    shown = img
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
             # Offer YUY2, not the BGR we feed in. Measured against Chrome's PipeWire capture
             # (headless Chrome, getUserMedia on a test node): BGR came out red/blue swapped
             # (blue skin in Meet), I420 and NV12 gave 2x2 black frames, RGB no frames;
             # YUY2 gave 640x480 with the clip's colours.
             "video/x-raw,format=YUY2",
             "!",
             "pipewiresink", "mode=provide", f"client-name={CAM_NAME}-cam",
             f"stream-properties={STREAM_PROPS}"],
            stdin=r, stdout=subprocess.DEVNULL, stderr=errlog)
        os.close(r)
        log("camera: gst feeder started (PipeWire node 'classwatch')")


def cmd_camera(action: str, clip: str | None = None, real: str = "/dev/video0",
               baseline: bool = BASELINE) -> int:
    """Manage the fake camera: up / down / degrade / restore / record."""
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
        cam = CameraStream(real_device=real, baseline=baseline)
        try:
            cam.start(clip)
        except Exception as e:
            log(f"camera: {e}")
            cam.stop()
            return 1
        # `classwatch run` asks for degrade (takeover), restore (back to the clip) and record here.
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
                    elif cmd == "record":
                        cam.record()
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
    if action in ("degrade", "restore", "record"):
        CONTROL_FILE.write_text(action)
        log(f"camera: {action} requested")
        return 0
    return 1
