"""Fake webcam: a PipeWire virtual camera that plays a prerecorded clip of Bogdan.

When classwatch detects that Bogdan is being called on, the camera "degrades":
frames become blocky/noisy (fake low bitrate) and arrive late (visible lag in
Meet), then it silently swaps to the real webcam and the stream recovers.

Architecture (no kernel module, no sudo):
  frame source (clip via PyAV, or real /dev/videoN via ffmpeg)
      ── BGR frames ──▶ CameraStream (pacing + degrade/recovery)
      ── rawvideo pipe (fd 0) ──▶ gst-launch-1.0
      ──▶ pipewiresink mode=provide ──▶ PipeWire node "classwatch"

The node advertises media.class=Video/Source, media.role=Camera, so it shows up
in `wpctl status` under Video > Sources next to the real webcam. Chrome only
sees PipeWire cameras with --enable-features=WebRtcPipeWireCamera (added in
`classwatch meet`).
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import threading
import time
import types

import av
import numpy as np

STATE_DIR = pathlib.Path(os.environ.get("XDG_STATE_HOME", pathlib.Path.home() / ".local/state")) / "classwatch"
CLIP_FILE = pathlib.Path(os.environ.get("CLASSWATCH_CAM_CLIP", STATE_DIR / "clip.mp4"))
RES = (640, 480)
FPS = 30
GOOD_KBPS = 2500
BAD_KBPS = 300
RECOVER_SECS = 20.0
LAG_SECS = 6.0  # how long the bad-connection phase lasts before the swap
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


class CameraStream:
    """Feeds frames into the PipeWire camera node.

    Modes:
      clip     - plays the prerecorded clip, looping, clean bitrate
      degrade  - same clip but blocky/noisy (fake low bitrate) with injected frame delay
      real     - the actual webcam, low bitrate at first, recovering over RECOVER_SECS
    """

    def __init__(self, real_device: str = "/dev/video0"):
        self.real_device = real_device
        self.mode = "clip"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._reader: av.container.InputContainer | None = None
        self._gen: "types.Generator[av.VideoFrame, None, None] | None" = None
        self._degraded_at = 0.0
        self._recovered_logged = False
        self._bitrate = GOOD_KBPS * 1000
        self._delay = 0.0
        self._pipe_fd = -1
        self._feeder: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---- control ----------------------------------------------------------

    def start(self, clip: str | None = None) -> None:
        """Start playing the prerecorded clip (or the real cam if no clip file)."""
        if not gst_available():
            raise RuntimeError("gst-launch-1.0 not found (install gstreamer1.0-plugins-base)")
        clip_path = pathlib.Path(clip) if clip else CLIP_FILE
        if clip_path.exists():
            self._set_source(clip_path, real=False)
            self.mode = "clip"
            log(f"camera: playing clip {clip_path.name}")
        else:
            self._set_source(self.real_device, real=True)
            self.mode = "real"
            log(f"camera: no clip at {clip_path}, using real webcam {self.real_device}")

    def degrade(self) -> None:
        """Called when Bogdan is called on: drop bitrate, add lag, then swap to real cam."""
        with self._lock:
            self.mode = "degrade"
            self._degraded_at = time.time()
            self._bitrate = BAD_KBPS * 1000
            self._delay = 0.35
            log("camera: DEGRADE (low bitrate + lag, switching to real cam)")
        threading.Timer(LAG_SECS, self._switch_to_real).start()

    def _switch_to_real(self) -> None:
        with self._lock:
            if self.mode != "degrade":
                return
            self.mode = "real"
            self._delay = 0.15
            self._degraded_at = time.time()  # recovery clock starts at the swap
            self._recovered_logged = False
            log("camera: switched to REAL webcam, bitrate recovering")
        self._set_source(self.real_device, real=True)

    def stop(self) -> None:
        self._stop.set()
        self._stop_source()
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

    # ---- source plumbing ----------------------------------------------------

    def _set_source(self, source: str | pathlib.Path, real: bool) -> None:
        with self._lock:
            self._stop_source()
            if real:
                self._proc = subprocess.Popen(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error",
                     "-f", "v4l2", "-framerate", str(FPS), "-input_format", "mjpeg",
                     "-video_size", f"{RES[0]}x{RES[1]}", "-i", str(source),
                     "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                self._reader = None
            else:
                self._reader = av.open(str(source))
                self._gen = self._reader.decode(video=0)
                self._proc = None
            log(f"camera: source -> {'real cam' if real else 'clip'} {source}")

    def _stop_source(self) -> None:
        if self._proc:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None
        self._gen = None
        if self._reader:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None

    # ---- frame pump ---------------------------------------------------------
    def _recovery_step(self, now: float) -> float:
        """Update bitrate/delay for the current mode; return the frame delay.

        In "real" mode the bitrate recovers BAD -> GOOD over RECOVER_SECS
        after the swap; the recovery-complete log line is emitted exactly once.
        """
        if self.mode != "real":
            return self._delay
        elapsed = now - self._degraded_at
        frac = min(1.0, elapsed / RECOVER_SECS)
        self._bitrate = int(BAD_KBPS * 1000 + (GOOD_KBPS - BAD_KBPS) * 1000 * frac)
        self._delay = 0.15 * (1.0 - frac)
        if frac >= 1.0 and not self._recovered_logged:
            self._recovered_logged = True
            log("camera: bitrate fully recovered")
        return self._delay


    def _run(self) -> None:
        """Pace frames to FPS (real time) and push them into the gst feeder."""
        frame_secs = 1.0 / FPS
        next_t = time.monotonic()
        while not self._stop.is_set():
            frame = self._next_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            with self._lock:
                delay = self._recovery_step(time.time())
            if delay > 0:
                time.sleep(delay)
            self._write_frame(frame)
            # Pace to wall clock so the clip plays at its real speed.
            next_t += frame_secs
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.monotonic()  # fell behind; resync instead of bursting

    def _next_frame(self) -> bytes | None:
        with self._lock:
            mode, reader, proc = self.mode, self._reader, self._proc
            if mode in ("clip", "degrade") and reader is not None:
                gen = self._gen
                if gen is None:
                    return None
                try:
                    frame = next(gen)
                except StopIteration:
                    # clip exhausted: close and reopen to loop
                    try:
                        path = reader.name
                        reader.close()
                        reader = av.open(path)
                        self._reader = reader
                        self._gen = reader.decode(video=0)
                        frame = next(self._gen)
                    except Exception:
                        return None
                except av.error.EOFError:
                    # decoder flush: treat as end of clip, loop
                    return None
                except Exception:
                    return None
                img = frame.to_ndarray(format="bgr24")
                if mode == "degrade":
                    img = self._degrade_image(img)
                return img.tobytes()
            if mode == "real" and proc is not None:
                n = RES[0] * RES[1] * 3
                data = proc.stdout.read(n)
                if data and len(data) == n:
                    return data
        return None

    @staticmethod
    def _degrade_image(img: np.ndarray) -> np.ndarray:
        """Cheap fake of a low-bitrate stream: blocky downscale-upscale + noise."""
        h, w = img.shape[:2]
        small = img[::4, ::4, :].astype(np.float32)
        blocky = np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)[:h, :w]
        noise = (np.random.rand(h, w, 1).astype(np.float32) - 0.5) * 18
        return np.clip(blocky + noise, 0, 255).astype(np.uint8)

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
             "pipewiresink", "mode=provide", f"client-name={CAM_NAME}-cam",
             f"stream-properties={STREAM_PROPS}"],
            stdin=r, stdout=subprocess.DEVNULL, stderr=errlog)
        os.close(r)
        log("camera: gst feeder started (PipeWire node 'classwatch')")


def cmd_camera(action: str, clip: str | None = None, real: str = "/dev/video0") -> int:
    """Manage the fake camera: up / down / degrade."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if action == "up":
        cam = CameraStream(real_device=real)
        try:
            cam.start(clip)
        except Exception as e:
            log(f"camera: {e}")
            return 1
        # Watch for the degrade signal from the main classwatch run.
        try:
            while True:
                if CONTROL_FILE.exists() and "degrade" in CONTROL_FILE.read_text():
                    CONTROL_FILE.unlink()
                    cam.degrade()
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            cam.stop()
        return 0
    if action == "down":
        subprocess.run(["pkill", "-f", f"client-name={CAM_NAME}-cam"], capture_output=True)
        time.sleep(0.5)
        log("camera: stopped PipeWire camera node")
        return 0
    if action == "degrade":
        CONTROL_FILE.write_text("degrade")
        log("camera: degrade requested")
        return 0
    return 1
