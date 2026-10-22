"""Tests for the fake webcam: link quality, rendering, real-time sources, mode switching."""
from __future__ import annotations

import pathlib
import time

import av
import numpy as np
import pytest

import classwatch.camera as cam
from classwatch.camera import FPS, RES, ClipSource, _Source


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
    monkeypatch.setattr(cam, "WebcamSource", lambda dev: ClipSource(clip))  # no webcam in tests
    monkeypatch.setattr(cam, "gst_available", lambda: True)
    monkeypatch.setattr(cam.CameraStream, "_write_frame", lambda self, frame: None)
    stream = cam.CameraStream()
    try:
        stream.start(str(clip))
        assert stream.mode == "clip"
        stream.degrade()
        assert stream.mode == "degrade"
        time.sleep(1.8)  # the swap lands inside the intro's second spike, at most ~1.6 s in
        assert stream.mode == "real"
        stream.restore()
        assert stream.mode == "restore"
        time.sleep(1.8)
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
        time.sleep(1.8)
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
        time.sleep(2.0)  # swapped to the webcam, recording
        stream.restore()  # Ctrl+D closing the mic
        time.sleep(1.8)
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
        time.sleep(1.8)
        stream.restore()
        stream.record(secs=2.0)
        time.sleep(1.8)  # past the restore swap, which must have been called off
        assert stream.mode == "real"
    finally:
        stream.stop()


# ---- measured drops ------------------------------------------------------------

def test_every_switch_swaps_early_inside_a_freeze():
    from classwatch.degrade import SWITCH_SCENARIOS, MeetDrop
    for seed in range(20):
        for name in SWITCH_SCENARIOS:
            d = MeetDrop(np.random.default_rng(seed), name)
            assert d.swap_at <= 1.6, (name, seed, d.swap_at)
            assert d.at(d.swap_at)[0], f"{name} seed {seed}: swap not hidden by a freeze"


def test_switch_intro_spike_blocks_spike_then_blocks_sharpen():
    from classwatch.degrade import MeetDrop
    for seed in range(10):
        d = MeetDrop(np.random.default_rng(seed))
        assert d.at(d.spike1_end / 2)[0]                                   # lag spike
        frozen, h, _, _ = d.at((d.spike1_end + d.px_end) / 2)
        assert not frozen and h < 0.1                                      # sudden macroblocks
        assert d.px_end < d.swap_at < d.intro_end and d.at(d.swap_at)[0]  # swap inside spike 2
        hs = [d.at(t)[1] for t in np.arange(d.intro_end, d.intro_end + 5, 0.1) if not d.at(t)[0]]
        assert hs[0] < 0.1 and max(hs) > 0.5                               # arrives blocky, sharpens


def test_drop_never_beats_the_rest_state_and_returns_to_it():
    from classwatch.degrade import Baseline, MeetDrop
    b = Baseline(np.random.default_rng(0))
    for name in ("sudden-drop", "squeeze", "congestion"):
        d = MeetDrop(np.random.default_rng(1), name)
        for k in range(0, int(d.dur * 10)):
            rest = b.at(k / 10)
            frozen, h, fps, q = d.at(k / 10, rest)
            if not frozen:
                assert h <= rest[1] + 1e-9 and fps <= rest[2] + 1e-9 and q >= rest[3] - 1e-9
        assert d.at(d.dur + 0.1, rest) == rest


def test_seam_freeze_covers_the_wrap_even_when_mild():
    from classwatch.degrade import MeetDrop
    d = MeetDrop(np.random.default_rng(3), "loss-burst", severity=0.45, max_secs=4.0,
                 lead_freeze=cam.SEAM_FREEZE_SECS)
    assert all(d.at(t)[0] for t in np.arange(0, cam.SEAM_LEAD_SECS + 0.2, 0.05))
    assert not any(d.at(t)[0] for t in np.arange(cam.SEAM_FREEZE_SECS + 0.05, d.dur, 0.05))
    assert d.dur <= 5


def test_degrader_holds_while_frozen_and_softens_otherwise():
    from classwatch.degrade import CLEAN, Degrader
    rng = np.random.default_rng(0)
    img = (rng.random((RES[1], RES[0], 3)) * 255).astype(np.uint8)
    deg = Degrader(*RES, rng=rng)
    assert deg.frame(img, 0.0, CLEAN) is img  # clean passes straight through, no codec
    bad = deg.frame(img, 0.1, (False, 0.75, 30.0, 210))
    assert bad.shape == img.shape and bad.dtype == np.uint8
    # Noise is the worst case for a starved codec: most of it must be gone.
    assert np.abs(np.diff(bad.astype(int), axis=1)).mean() < 0.5 * np.abs(np.diff(img.astype(int), axis=1)).mean()
    other = (rng.random(img.shape) * 255).astype(np.uint8)
    assert deg.frame(other, 0.2, (True, 0.75, 0.0, 210)) is bad  # frozen: the last frame stays
