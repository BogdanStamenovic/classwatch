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
# Scenarios that freeze within ~0.5 s: the source swap hides inside that freeze, so a switch
# happens almost as soon as it is asked for. Squeeze and congestion never freeze and would
# show the jump on a moving picture, so they are not used for switching.
SWITCH_SCENARIOS = ("sudden-drop", "loss-burst", "lag-spike", "stall-2s", "stall-5s")
SWAP_INTO_FREEZE = 0.25  # how far into the first freeze the source swaps


State = tuple[bool, float, float, float]  # (frozen, height ratio, fps, q-index)
CLEAN: State = (False, 1.0, CLEAN_FPS, CLEAN_Q)


class Baseline:
    """A permanently mediocre link: what the measured stream settled at between drops (480x270
    of 640x360, q ~110), with the quality wandering slowly and a short stutter now and then.
    Against it, a drop for a switch does not stand out from a perfect picture."""

    def __init__(self, rng: np.random.Generator, height: float = 0.75, q: float = 112.0,
                 fps: float = 25.0, stutter_every: float = 35.0):
        self.rng, self.hr, self.q0, self.fps = rng, height, q, fps
        self.stutter_every = stutter_every
        self._x = 0.0
        self._last = None
        self._stutter_until = 0.0
        self._next_stutter = None

    def at(self, now: float) -> State:
        if self._last is None:
            self._last = now
            self._next_stutter = now + self.rng.exponential(self.stutter_every)
        dt, self._last = max(0.0, now - self._last), now
        # Ornstein-Uhlenbeck wander of the quantizer: drifts, never runs away.
        self._x += -self._x * dt / 6.0 + self.rng.normal(0, 4.0 * np.sqrt(dt))
        if now >= self._next_stutter:
            self._stutter_until = now + self.rng.uniform(0.15, 0.4)
            self._next_stutter = now + self.rng.exponential(self.stutter_every)
        return (now < self._stutter_until, self.hr, self.fps, float(np.clip(self.q0 + self._x, CLEAN_Q, 150)))


def vpx_q(q_index: float) -> int:
    """VP9 q-index (0-255, what WebRTC stats report) to libvpx's 0-63 quantizer scale."""
    return int(np.clip(round(q_index / 4), 4, 63))


class MeetDrop:
    """One drop: a measured scenario, stretched in time and shifted in quality at random."""

    def __init__(self, rng: np.random.Generator, scenario: str | None = None, severity: float = 1.0,
                 max_secs: float | None = None, lead_freeze: float = 0.0):
        self.name = scenario or str(rng.choice(SWITCH_SCENARIOS))
        raw = TRACES[self.name]
        if lead_freeze:
            # A freeze up front (loop seams), then the scenario from its first decoded frame on.
            first = next(k for k, s in enumerate(raw) if s[2] >= 0.5)
            t1 = raw[first][0]
            raw = [[0.0, raw[first][1], 0.0, None], [lead_freeze, raw[first][1], 0.0, None]] + \
                  [[s[0] - t1 + lead_freeze + 0.2, *s[1:]] for s in raw[first:]]
        if max_secs:
            raw = [s for s in raw if s[0] <= max_secs]
        self.stretch = float(rng.uniform(0.8, 1.2))
        self.q_shift = float(rng.normal(0, 8))
        self.severity = severity  # < 1 pulls every value toward clean (loop seams use a light one)
        self.lead_freeze = lead_freeze
        self.t = np.array([s[0] for s in raw]) * self.stretch
        self.h = np.array([s[1] / 360 for s in raw])
        self.fps = np.array([s[2] for s in raw])
        self.q = np.array([s[3] if s[3] is not None else np.nan for s in raw])
        self.dur = float(self.t[-1])
        frozen = self.fps < 0.5
        if frozen.any():
            # Swap the source just inside the first freeze: the picture stalls and comes back as the
            # other one. at() reports a freeze from sample k's time on, so t[k-1] is where it starts.
            k = int(np.argmax(frozen))
            start = float(self.t[k - 1]) if k else 0.0
            end = k
            while end < len(frozen) and frozen[end]:
                end += 1
            # The freeze runs up to t[end-1]; short ones get the swap at their midpoint.
            self.swap_at = min(start + SWAP_INTO_FREEZE, (start + float(self.t[end - 1])) / 2)
        else:
            self.swap_at = float(self.t[int(np.nanargmax(self.q))])

    def at(self, t: float, rest: State = CLEAN) -> State:
        """State at t seconds into the drop; it eases back to `rest` (clean or the baseline).
        A drop is never better than the rest state: it only ever makes things worse."""
        _, rh, rfps, rq = rest
        if t >= self.dur:
            return rest
        if t < self.lead_freeze:
            return True, rh, 0.0, rq  # the seam freeze happens whatever the severity
        k = min(int(np.searchsorted(self.t, t)), len(self.t) - 1)
        if self.fps[k] < 0.5 and self.severity >= 0.5:
            return True, rh, 0.0, rq
        q = self.q[k] if not np.isnan(self.q[k]) else CLEAN_Q
        q, h, fps = max(q + self.q_shift, rq), min(self.h[k], rh), min(max(self.fps[k], 4.0), rfps)
        ease = min(1.0, max(0.0, (self.dur - t) / TAIL_SECS)) * self.severity
        return (False, rh - (rh - h) * ease, rfps - (rfps - fps) * ease, rq + (q - rq) * ease)


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

    def frame(self, img: np.ndarray, t: float, state: State) -> np.ndarray:
        """img as it would arrive on the other end in `state`; t is any steady clock in seconds."""
        if state == CLEAN:
            self.reset()  # nothing to do, and the next drop starts on a fresh keyframe
            return img
        frozen, hr, fps, q = state
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
