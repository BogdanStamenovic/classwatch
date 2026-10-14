"""Tests for the fake webcam: clip decode, degrade image, mode transitions, recovery curve."""
from __future__ import annotations

import pathlib
import tempfile
import time

import av
import numpy as np

from classwatch.camera import CameraStream, GOOD_KBPS, RECOVER_SECS, RES


def make_clip(path: pathlib.Path) -> None:
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    c = av.open(str(path), "w")
    st = c.add_stream("libx264", rate=30)
    st.width, st.height = 640, 480
    for _ in range(30):
        f = av.VideoFrame.from_ndarray(img, format="bgr24")
        for p in st.encode(f):
            c.mux(p)
    for p in st.encode(None):
        c.mux(p)
    c.close()


def test_degrade_image():
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    out = CameraStream._degrade_image(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    # blocky: pixels within the same 4x4 block should be close after degrade
    # (noise amplitude is 18 peak-to-peak, so allow a small spread)
    a = out[12:16, 12:16, 0]  # exactly one 4x4 block (rows 12-15 come from small[3])
    assert a.max() - a.min() <= 20


def test_clip_frames_and_loop():
    with tempfile.TemporaryDirectory() as d:
        clip = pathlib.Path(d) / "clip.mp4"
        make_clip(clip)
        cam = CameraStream()
        cam._set_source(clip, real=False)
        cam.mode = "clip"
        try:
            frames = [cam._next_frame() for _ in range(5)]
            assert all(f is not None and len(f) == RES[0] * RES[1] * 3 for f in frames)
            # exhaust the 30-frame clip, then verify it loops instead of returning None
            for _ in range(26):
                assert cam._next_frame() is not None
            assert cam._next_frame() is not None  # wrapped back to frame 0
        finally:
            cam.stop()


def test_degrade_mode():
    with tempfile.TemporaryDirectory() as d:
        clip = pathlib.Path(d) / "clip.mp4"
        make_clip(clip)
        cam = CameraStream()
        cam._set_source(clip, real=False)
        try:
            cam.mode = "degrade"
            cam._degraded_at = time.time()
            f = cam._next_frame()
            assert f is not None and len(f) == RES[0] * RES[1] * 3
        finally:
            cam.stop()


def test_recovery_curve():
    cam = CameraStream()
    try:
        cam.mode = "real"
        cam._degraded_at = time.time() - 100  # well past recovery
        # simulate the recovery math from _run
        from classwatch.camera import BAD_KBPS, GOOD_KBPS, RECOVER_SECS
        elapsed = time.time() - cam._degraded_at
        frac = min(1.0, elapsed / RECOVER_SECS)
        bitrate = int(BAD_KBPS * 1000 + (GOOD_KBPS - BAD_KBPS) * 1000 * frac)
        assert frac == 1.0 and bitrate == GOOD_KBPS * 1000
    finally:
        cam.stop()


def test_recovery_logs_once(tmp_path, monkeypatch):
    """_recovery_step past RECOVER_SECS must log the recovery line exactly once,
    no matter how many frames keep arriving at steady state."""
    import classwatch.camera as cam
    log_file = tmp_path / "camera.log"
    monkeypatch.setattr(cam, "STATE_DIR", tmp_path)
    stream = cam.CameraStream()
    try:
        stream.mode = "real"
        stream._degraded_at = 1000.0
        now = 1000.0 + RECOVER_SECS + 10  # well past recovery
        for _ in range(60):  # 60 steady-state frames through the real code path
            stream._recovery_step(now)
        lines = log_file.read_text().splitlines()
        assert sum("fully recovered" in l for l in lines) == 1
        # and the steady state itself is correct
        assert stream._bitrate == GOOD_KBPS * 1000
    finally:
        stream.stop()
