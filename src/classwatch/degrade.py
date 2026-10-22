"""Meet-style connection drops, replayed from measurements and rendered by a real codec.

meet_traces.json holds what a Meet receiver actually decoded while the sender's upload was
throttled (2026-10-22): per 200 ms, the decoded height, frame rate, VP9 quantizer (q-index,
0-255), and freezes. A drop picks one of those scenarios, stretches it and shifts its quality
a little so no two drops look alike, and pushes every shown frame through libvpx-vp9 at that
quantizer and resolution. Measured, the real thing is not blocky squares or a low frame rate:
it freezes when the link breaks, then stays fairly fluid but soft, smeared and washed out,
and climbs back over 10-25 s. Drawn blocks never looked like that; the codec does.
"""
from __future__ import annotations

import fractions
import json
import pathlib

import av
import numpy as np

TRACES: dict[str, list] = {k: v for k, v in json.loads(
    (pathlib.Path(__file__).with_name("meet_traces.json")).read_text()).items() if not k.startswith("_")}
CLEAN_Q = 100  # q-index of the clean stream in the same measurement
CLEAN_FPS = 30.0
TAIL_SECS = 3.0  # the last stretch of a trace eases back to clean so the drop actually ends


def vpx_q(q_index: float) -> int:
    """VP9 q-index (0-255, what WebRTC stats report) to libvpx's 0-63 quantizer scale."""
    return int(np.clip(round(q_index / 4), 4, 63))


class MeetDrop:
    """One drop: a measured scenario, stretched in time and shifted in quality at random."""

    def __init__(self, rng: np.random.Generator, scenario: str | None = None, severity: float = 1.0):
        self.name = scenario or str(rng.choice(sorted(TRACES)))
        raw = TRACES[self.name]
        self.stretch = float(rng.uniform(0.8, 1.2))
        self.q_shift = float(rng.normal(0, 8))
        self.severity = severity  # < 1 pulls every value toward clean (loop seams use a light one)
        self.t = np.array([s[0] for s in raw]) * self.stretch
        self.h = np.array([s[1] / 360 for s in raw])
        self.fps = np.array([s[2] for s in raw])
        self.q = np.array([s[3] if s[3] is not None else np.nan for s in raw])
        self.dur = float(self.t[-1])
        frozen = self.fps < 0.5
        if frozen.any():
            # Swap the source inside the first freeze: the picture stalls and comes back as the other one.
            k = int(np.argmax(frozen))
            end = k
            while end < len(frozen) and frozen[end]:
                end += 1
            self.swap_at = float((self.t[k] + self.t[min(end, len(self.t) - 1)]) / 2)
        else:
            self.swap_at = float(self.t[int(np.nanargmax(self.q))])

    def at(self, t: float) -> tuple[bool, float, float, float]:
        """(frozen, height ratio, fps, q-index) at t seconds into the drop."""
        if t >= self.dur:
            return False, 1.0, CLEAN_FPS, CLEAN_Q
        k = min(int(np.searchsorted(self.t, t)), len(self.t) - 1)
        if self.fps[k] < 0.5 and self.severity >= 0.5:
            return True, self.h[k], 0.0, CLEAN_Q
        q = self.q[k] if not np.isnan(self.q[k]) else CLEAN_Q
        q, h, fps = q + self.q_shift, self.h[k], max(self.fps[k], 4.0)
        ease = min(1.0, max(0.0, (self.dur - t) / TAIL_SECS)) * self.severity
        return (False, 1 - (1 - h) * ease, CLEAN_FPS - (CLEAN_FPS - fps) * ease,
                CLEAN_Q + (q - CLEAN_Q) * ease)


class Degrader:
    """Turns clean frames into what the drop would look like on the other end."""

    def __init__(self, width: int, height: int, rng: np.random.Generator | None = None):
        self.w, self.h = width, height
        self.rng = rng or np.random.default_rng()
        self._key: tuple[int, int, int] | None = None
        self._enc = self._dec = None
        self._pts = 0
        self._held: np.ndarray | None = None
        self._next_t = 0.0
        self._was_frozen = False

    def reset(self) -> None:
        self._key = self._enc = self._dec = None
        self._held = None
        self._next_t = 0.0

    def frame(self, img: np.ndarray, t: float, drop: MeetDrop) -> np.ndarray:
        frozen, hr, fps, q = drop.at(t)
        if frozen:
            self._was_frozen = True
            return self._held if self._held is not None else img
        if self._held is not None and t < self._next_t:
            return self._held
        # Uneven frame gaps, as decoded frames arrive with network jitter.
        self._next_t = t + float(self.rng.uniform(0.7, 1.3)) / fps
        ew = max(64, int(self.w * hr) // 2 * 2)
        eh = max(48, int(self.h * hr) // 2 * 2)
        vq = vpx_q(q)
        if self._key is None or self._was_frozen or (ew, eh) != self._key[:2] or abs(vq - self._key[2]) >= 3:
            # A new encoder starts on a keyframe. Real Meet also sends one after a freeze, and
            # keeping the encoder between small quality changes keeps the inter-frame smear.
            self._open(ew, eh, vq)
            self._was_frozen = False
        assert self._enc is not None and self._dec is not None
        f = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(width=ew, height=eh, format="yuv420p")
        f.pts = self._pts
        self._pts += 1
        for pkt in self._enc.encode(f):
            for d in self._dec.decode(pkt):
                self._held = d.reformat(width=self.w, height=self.h, format="bgr24",
                                        interpolation="BILINEAR").to_ndarray()
        return self._held if self._held is not None else img

    def _open(self, w: int, h: int, vq: int) -> None:
        enc = av.CodecContext.create("libvpx-vp9", "w")
        enc.width, enc.height, enc.pix_fmt = w, h, "yuv420p"
        enc.time_base = fractions.Fraction(1, 30)
        enc.bit_rate = 2_000_000  # quality is pinned by qmin = qmax; libvpx only needs a non-zero rate
        enc.options = {"deadline": "realtime", "cpu-used": "8", "lag-in-frames": "0",
                       "qmin": str(vq), "qmax": str(vq), "row-mt": "1"}
        self._enc, self._dec = enc, av.CodecContext.create("vp9", "r")
        self._key = (w, h, vq)
        self._pts = 0
