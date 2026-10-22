"""Tests for the fake webcam: link quality, rendering, real-time sources, mode switching."""
from __future__ import annotations

import pathlib
import time

import av
import numpy as np
import pytest

import classwatch.camera as cam
from classwatch.camera import FPS, RES, ClipSource, Link, _Source, out_fps, render


def make_clip(path: pathlib.Path, frames: int = 30) -> None:
    """A clip whose frame i is filled with value i, so tests can tell frames apart."""
    c = av.open(str(path), "w")
    st = c.add_stream("libx264", rate=FPS)
    st.width, st.height = RES
    st.pix_fmt = "yuv420p"
    st.options = {"qp": "0"}
    for i in range(frames):
        img = np.full((RES[1], RES[0], 3), i * 8 % 256, np.uint8)
        for p in st.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
            c.mux(p)
    for p in st.encode(None):
        c.mux(p)
    c.close()


@pytest.fixture(autouse=True)
def _state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cam, "STATE_DIR", tmp_path)


# ---- link quality -------------------------------------------------------------

def test_link_dip_holds_then_recovers_linearly():
    link = Link()
    link.dip(100.0, depth=0.8, hold=2.0, recover=4.0)
    assert link.q(99.0) == 1.0
    assert link.q(101.0) == pytest.approx(0.2)
    assert link.q(104.0) == pytest.approx(0.6)  # halfway through recovery
    assert link.q(106.5) == 1.0


def test_link_deepest_dip_wins():
    link = Link()
    link.dip(0.0, depth=0.9, hold=10.0, recover=5.0)  # takeover
    link.dip(1.0, depth=0.3, hold=0.5, recover=0.5)  # a seam inside it
    assert link.q(1.2) == pytest.approx(0.1)


def test_link_reports_recovery_once_per_dip():
    link = Link()
    link.dip(0.0, depth=0.5, hold=1.0, recover=1.0)
    hits = [link.recovered(t / 10) for t in range(60)]
    assert sum(hits) == 1
    link.dip(10.0, depth=0.5, hold=1.0, recover=1.0)
    assert sum(link.recovered(10.0 + t / 10) for t in range(60)) == 1


def test_out_fps_range():
    assert out_fps(1.0) == FPS
    assert out_fps(0.0) == 4.0
    assert out_fps(0.0) < out_fps(0.5) < out_fps(1.0)


def test_render_clean_is_untouched_and_bad_is_blocky():
    img = (np.random.rand(RES[1], RES[0], 3) * 255).astype(np.uint8)
    assert render(img, 1.0) is img
    out = render(img, 0.1, np.random.default_rng(0))
    assert out.shape == img.shape and out.dtype == np.uint8
    block = round(1 + 0.9 * 9)
    # inside one block every pixel came from the same source pixel (plus noise of +-~11)
    a = out[:block, :block, 0].astype(int)
    assert a.max() - a.min() <= 24
    assert np.abs(out.astype(int) - img.astype(int)).mean() > 20


# ---- sources ------------------------------------------------------------------

def test_clip_source_plays_in_real_time_and_signals_seam_before_wrap(tmp_path):
    clip = tmp_path / "clip.mp4"
    make_clip(clip, frames=30)  # 1 s
    seams: list[float] = []
    src = ClipSource(clip, on_seam=lambda: seams.append(time.monotonic())).start()
    t0 = time.monotonic()
    try:
        values = []
        while time.monotonic() - t0 < 2.4:
            img = src.latest()
            if img is not None:
                values.append(int(img[0, 0, 0]))
            time.sleep(0.05)
    finally:
        src.stop()
    # Real time: about 2.4 loops' worth of a 1 s clip went by, so it wrapped twice.
    assert len(seams) == 2
    # The seam fires ~SEAM_LEAD_SECS before the end of each 1 s loop, not after it.
    first = seams[0] - t0
    assert 1.0 - cam.SEAM_LEAD_SECS - 0.25 <= first <= 1.0
    # The last frames (value 232) were reached, i.e. the whole clip played, not a slowed part of it.
    assert max(values) > 200


def test_slow_consumer_gets_newest_frame_not_a_backlog():
    """The slow-motion bug: frames must not queue up behind a slow consumer."""

    class Counter(_Source):
        def _loop(self):
            i = 0
            while not self._stop.is_set():
                self._put(np.array([i]))
                i += 1
                time.sleep(0.001)

    src = Counter().start()
    try:
        time.sleep(0.2)
        a = int(src.latest()[0])
        time.sleep(0.3)  # a consumer that stalled for 0.3 s
        b = int(src.latest()[0])
    finally:
        src.stop()
    assert b - a > 50  # it jumped ahead to the newest frame


# ---- camera stream ------------------------------------------------------------

def test_degrade_then_restore_round_trip(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    make_clip(clip)
    monkeypatch.setattr(cam, "LAG_SECS", 0.3)
    monkeypatch.setattr(cam, "RESTORE_LAG_SECS", 0.3)
    monkeypatch.setattr(cam, "WebcamSource", lambda dev: ClipSource(clip))  # no webcam in tests
    monkeypatch.setattr(cam, "gst_available", lambda: True)
    monkeypatch.setattr(cam.CameraStream, "_write_frame", lambda self, frame: None)
    stream = cam.CameraStream()
    try:
        stream.start(str(clip))
        assert stream.mode == "clip"
        stream.degrade()
        assert stream.mode == "degrade"
        assert stream.link.q(time.monotonic()) < 0.2
        time.sleep(0.6)
        assert stream.mode == "real"
        stream.restore()
        assert stream.mode == "restore"
        time.sleep(0.6)
        assert stream.mode == "clip"
    finally:
        stream.stop()
    log = (tmp_path / "camera.log").read_text()
    assert "switched to REAL" in log and "switched back to the CLIP" in log


def test_stop_waits_for_pump_thread():
    """stop() must not close sources while the pump thread can still use them
    (closing a decoder under a running thread segfaulted on Ctrl-C)."""
    stream = cam.CameraStream()
    stream.stop()
    assert not stream._thread.is_alive()


def test_portal_grant_present():
    from classwatch.camera import portal_grant_present
    assert portal_grant_present("({'com.google.Chrome': ['yes'], 'org.gnome.Snapshot': ['yes']}, <byte 0x00>)")
    assert not portal_grant_present("({'org.gnome.Snapshot': ['yes']}, <byte 0x00>)")
    assert not portal_grant_present("({'com.google.Chrome': ['no']}, <byte 0x00>)")


# ---- recording ------------------------------------------------------------------

def _fast_stream(tmp_path, monkeypatch, webcam_clip):
    monkeypatch.setattr(cam, "LAG_SECS", 0.3)
    monkeypatch.setattr(cam, "RESTORE_LAG_SECS", 0.3)
    monkeypatch.setattr(cam, "WebcamSource", lambda dev: ClipSource(webcam_clip))
    monkeypatch.setattr(cam, "gst_available", lambda: True)
    monkeypatch.setattr(cam.CameraStream, "_write_frame", lambda self, frame: None)
    return cam.CameraStream()


def _frames_and_secs(path):
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        n = sum(1 for _ in c.decode(video=0))
        return n, float(vs.duration * vs.time_base)


def test_record_from_clip_swaps_records_and_comes_back(tmp_path, monkeypatch):
    clip, webcam = tmp_path / "clip.mp4", tmp_path / "webcam.mp4"
    make_clip(clip, frames=30)
    make_clip(webcam, frames=90)
    old_bytes = clip.read_bytes()
    stream = _fast_stream(tmp_path, monkeypatch, webcam)
    try:
        stream.start(str(clip))
        stream.record(secs=1.0)
        assert stream.mode == "degrade"  # the call sees the link collapse first, as on T
        deadline = time.monotonic() + 5
        while stream.mode != "clip" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert stream.mode == "clip"
    finally:
        stream.stop()
    # One second at a steady FPS: the player paces by frame count, so the count is what matters.
    n, secs = _frames_and_secs(clip)
    assert n == FPS and secs == pytest.approx(1.0, abs=0.05)
    assert (tmp_path / "clip.prev.mp4").read_bytes() == old_bytes
    assert not (tmp_path / "clip.new.mp4").exists()
    log = (tmp_path / "camera.log").read_text()
    assert "RECORDING" in log and "new clip saved" in log and "switched back to the CLIP" in log


def test_record_while_real_starts_at_once(tmp_path, monkeypatch):
    clip, webcam = tmp_path / "clip.mp4", tmp_path / "webcam.mp4"
    make_clip(clip)
    make_clip(webcam, frames=90)
    stream = _fast_stream(tmp_path, monkeypatch, webcam)
    try:
        stream.start(str(clip))
        stream.degrade()
        time.sleep(0.6)
        assert stream.mode == "real"
        stream.record(secs=0.5)
        assert stream.mode == "real"
        assert stream._record_cancel is not None and not stream._record_cancel.is_set()
        stream.record(secs=0.5)  # a second Ctrl+P while recording is ignored
        assert "already recording" in (tmp_path / "camera.log").read_text()
    finally:
        stream.stop()


def test_restore_during_recording_cancels_and_keeps_old_clip(tmp_path, monkeypatch):
    clip, webcam = tmp_path / "clip.mp4", tmp_path / "webcam.mp4"
    make_clip(clip)
    make_clip(webcam, frames=90)
    old_bytes = clip.read_bytes()
    stream = _fast_stream(tmp_path, monkeypatch, webcam)
    try:
        stream.start(str(clip))
        stream.record(secs=5.0)
        time.sleep(0.8)  # swapped to the webcam, recording
        stream.restore()  # Ctrl+D closing the mic
        time.sleep(0.8)
        assert stream.mode == "clip"
    finally:
        stream.stop()
    assert clip.read_bytes() == old_bytes
    assert not (tmp_path / "clip.new.mp4").exists()
    assert not (tmp_path / "clip.prev.mp4").exists()
    assert "recording cancelled" in (tmp_path / "camera.log").read_text()


def test_record_during_restore_lag_stays_on_webcam(tmp_path, monkeypatch):
    clip, webcam = tmp_path / "clip.mp4", tmp_path / "webcam.mp4"
    make_clip(clip)
    make_clip(webcam, frames=90)
    stream = _fast_stream(tmp_path, monkeypatch, webcam)
    try:
        stream.start(str(clip))
        stream.degrade()
        time.sleep(0.6)
        stream.restore()
        stream.record(secs=2.0)
        time.sleep(0.6)  # past the restore swap, which must have been called off
        assert stream.mode == "real"
    finally:
        stream.stop()
