"""Render candidate "connection drop" styles for the fake camera, one mp4 each plus a grid.

Each demo: clean "before" source, a sudden drop during which the source swaps to "after",
then clean again. Usage:
    degrade_demo.py BEFORE.mp4 AFTER.mp4 OUTDIR
"""
from __future__ import annotations

import fractions
import pathlib
import subprocess
import sys

import av
import numpy as np

FPS = 30
W, H = 640, 480
LEAD = 2.0  # clean seconds before the drop
TAIL = 2.5  # clean seconds after recovery


def load(path: str, start: float, secs: float) -> list[np.ndarray]:
    out = []
    with av.open(path) as c:
        for f in c.decode(video=0):
            if f.time is None or f.time < start:
                continue
            out.append(f.reformat(width=W, height=H).to_ndarray(format="bgr24"))
            if len(out) >= secs * FPS:
                break
    return out


def blocks(img: np.ndarray, b: int) -> np.ndarray:
    """Flat b x b blocks (each the block's mean colour), like a starved codec's DC-only frame."""
    if b <= 1:
        return img
    h, w = img.shape[:2]
    hp, wp = -(-h // b) * b, -(-w // b) * b
    pad = np.pad(img, ((0, hp - h), (0, wp - w), (0, 0)), mode="edge").astype(np.float32)
    m = pad.reshape(hp // b, b, wp // b, b, 3).mean(axis=(1, 3))
    return np.repeat(np.repeat(m, b, 0), b, 1)[:h, :w].astype(np.uint8)


class Sources:
    """Frame at output index i: 'before' until swap_at, 'after' from then on (both real time)."""

    def __init__(self, before, after, drop_frame: int, swap_after: float):
        self.before, self.after = before, after
        self.swap = drop_frame + round(swap_after * FPS)

    def __call__(self, i: int) -> np.ndarray:
        if i < self.swap:
            return self.before[i % len(self.before)]
        return self.after[(i - self.swap) % len(self.after)]


# Each style: (name, description, drop length in s, swap time after the drop, frame fn).
# frame fn(t, i, src, st) -> image; t = seconds since the drop, st = per-style scratch dict.

def style_freeze_blocks(t, i, src, st):
    """Freeze, then a few seconds of huge blocks at a few fps, then straight back."""
    if t < 0.35:
        return st.setdefault("frozen", src(i - 1))
    if t < 1.15:
        return held(st, i, 4, lambda: blocks(src(i), 40))
    if t < 1.5:
        return held(st, i, 12, lambda: blocks(src(i), 16))
    return src(i)


def style_bitrate_ladder(t, i, src, st):
    """Freeze, then steps up a quality ladder like an adaptive-bitrate stream re-negotiating."""
    if t < 0.3:
        return st.setdefault("frozen", src(i - 1))
    for until, b, fps in ((0.9, 48, 5), (1.3, 24, 10), (1.6, 12, 20)):
        if t < until:
            return held(st, i, fps, lambda b=b: blocks(src(i), b))
    return src(i)


def style_smear(t, i, src, st):
    """Lost reference frames: only some 16 px macroblocks refresh, the rest keep stale pixels,
    so the old picture smears into the new one (the classic Meet/Zoom packet-loss look)."""
    canvas = st.get("canvas")
    if canvas is None or t < 0.25:
        st["canvas"] = canvas = src(i - 1).copy()
        return canvas
    if t >= 1.5:
        return src(i)
    rng = st.setdefault("rng", np.random.default_rng(3))
    frac = 0.04 + 0.9 * ((t - 0.25) / 1.25) ** 2
    b = 16
    cur = src(i)
    coarse = blocks(cur, 32) if t < 0.9 else cur
    mask = rng.random((H // b, W // b)) < frac
    m = np.repeat(np.repeat(mask, b, 0), b, 1)[:, :, None]
    canvas[:] = np.where(m, coarse, canvas)
    return canvas


def style_x264(t, i, src, st):
    """Real codec artifacts: frames really go through x264 at a starved bitrate and are decoded."""
    if t < 0.3:
        return st.setdefault("frozen", src(i - 1))
    if t >= 1.7:
        return src(i)
    for until, scale, crf, fps in ((1.0, 6, 51, 6), (1.4, 3, 45, 12), (1.7, 2, 38, 24)):
        if t < until:
            return held(st, i, fps, lambda: codec_pass(st, src(i), scale, crf))
    return src(i)


def style_hiccup(t, i, src, st):
    """The shortest: a hard freeze, one chunky frame, a moment of medium blocks, clean."""
    if t < 0.45:
        return st.setdefault("frozen", src(i - 1))
    if t < 0.62:
        return held(st, i, 30, lambda: blocks(src(i), 56))
    if t < 0.95:
        return held(st, i, 10, lambda: blocks(src(i), 20))
    return src(i)


def held(st, i, fps, make):
    """Show a new frame only `fps` times a second; repeat the last one in between (lag)."""
    step = max(1, round(FPS / fps))
    if i % step == 0 or "held" not in st:
        st["held"] = make()
    return st["held"]


def codec_pass(st, img, scale, crf):
    w, h = W // scale // 2 * 2, H // scale // 2 * 2
    key = (scale, crf)
    if st.get("codec_key") != key:
        enc = av.CodecContext.create("libx264", "w")
        enc.width, enc.height, enc.pix_fmt = w, h, "yuv420p"
        enc.time_base = fractions.Fraction(1, FPS)
        enc.options = {"crf": str(crf), "preset": "ultrafast", "tune": "zerolatency"}
        st.update(codec_key=key, enc=enc, dec=av.CodecContext.create("h264", "r"), pts=0)
    f = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(width=w, height=h, format="yuv420p")
    f.pts = st["pts"]
    st["pts"] += 1
    out = st.get("last_dec", img)
    for pkt in st["enc"].encode(f):
        for d in st["dec"].decode(pkt):
            small = d.to_ndarray(format="bgr24")
            out = np.repeat(np.repeat(small, scale, 0), scale, 1)[:H, :W]
            out = np.pad(out, ((0, H - out.shape[0]), (0, W - out.shape[1]), (0, 0)), mode="edge")
    st["last_dec"] = out
    return out


# (start s, end s, block px at start, block px at end, fps at start, fps at end)
LADDER2 = [
    (0.0, 1.2, 1, 1, 22, 9),     # lag: frames drop out, picture still sharp
    (1.2, 2.2, 16, 48, 8, 4),    # bitrate collapses; the swap lands at the worst point
    (2.2, 4.7, 48, 1, 4, 30),    # gradual recovery
]
STUTTERS = (0.35, 0.9)  # short freezes during the lag phase (s since the drop starts)


def _ladder2(t, i, src, st, delay_max):
    for t0, t1, b0, b1, f0, f1 in LADDER2:
        if t0 <= t < t1:
            k = (t - t0) / (t1 - t0)
            b = round(b0 + (b1 - b0) * k)
            fps = f0 + (f1 - f0) * k
            break
    else:
        return src(i)
    if any(s <= t < s + 0.22 for s in STUTTERS):
        return st.setdefault("held", src(i))
    # B3: playback falls behind real time during the lag, then the swap resets it.
    lag = 0
    if delay_max and t < 1.6:
        lag = round(min(1.0, t / 1.2) * delay_max * FPS)
    rng = st.setdefault("rng", np.random.default_rng(7))
    if "next" not in st or i >= st["next"]:
        st["held"] = blocks(src(i - lag), b)
        # Uneven gaps between new frames read as network jitter, not a lower frame rate.
        st["next"] = i + max(1, round(FPS / fps * rng.uniform(0.6, 1.5)))
    return st["held"]


def style_ladder2(t, i, src, st):
    """B, slower: a stuttering lead-up, then the bitrate collapses, then a gradual recovery."""
    return _ladder2(t, i, src, st, 0.0)


def style_ladder3(t, i, src, st):
    """B2, and during the lead-up the picture also falls ~0.4 s behind real time."""
    return _ladder2(t, i, src, st, 0.4)


class RandomDrop:
    """A drop with no visible pattern: link quality q wanders (Ornstein-Uhlenbeck noise around an
    envelope that sags then recovers, plus relapses), and everything is drawn from q per frame:
    block size, the gap until the next new frame, freezes of random length, and frames that only
    half arrive. The swap to the new source is hidden inside a freeze at the worst point."""

    SIZES = (8, 12, 16, 24, 32, 40, 48, 64)

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.dur = float(rng.uniform(3.8, 5.5))
        self.t_low = float(rng.uniform(1.4, 2.2))
        n = round(self.dur * FPS)
        t = np.arange(n) / FPS
        down = np.clip(t / self.t_low, 0, 1) ** 1.6
        up = np.clip((t - self.t_low) / (self.dur - self.t_low), 0, 1) ** 0.8
        env = np.where(t < self.t_low, 1 - 0.92 * down, 0.08 + 0.92 * up)
        noise, x = np.zeros(n), 0.0
        for k in range(n):
            x += -x * 0.15 + rng.normal(0, 0.09)
            noise[k] = x
        q = env + noise
        for _ in range(rng.integers(1, 3)):  # relapses during recovery
            s = rng.uniform(self.t_low + 0.4, self.dur - 0.5)
            q -= 0.45 * ((t > s) & (t < s + rng.uniform(0.2, 0.5)))
        q[t < 0.25] = np.maximum(q[t < 0.25], 0.85)  # the very start is only a hiccup, never blocks
        self.q = np.clip(q, 0, 1)
        # The swap freeze: the picture stalls at the bottom and comes back as the new source.
        self.swap_freeze = (self.t_low - rng.uniform(0.1, 0.35), rng.uniform(0.35, 0.8))
        self.swap_after = self.swap_freeze[0] + self.swap_freeze[1] * 0.6
        self.freezes = [self.swap_freeze]
        tt = 0.15
        while tt < self.dur - 0.4:
            qq = self.q[min(n - 1, round(tt * FPS))]
            if rng.random() < 0.18 * (1.3 - qq):
                self.freezes.append((tt, float(rng.uniform(0.12, 0.6))))
                tt += 0.5
            tt += 1 / 6

    def __call__(self, t, i, src, st):
        if t >= self.dur:
            return src(i)
        if "held" not in st:
            st["held"] = src(i - 1).copy()
        if any(s <= t < s + d for s, d in self.freezes):
            return st["held"]
        q = self.q[min(len(self.q) - 1, round(t * FPS))]
        if i < st.get("next", 0):
            return st["held"]
        rng = self.rng
        if q > 0.8:
            img = src(i)
        else:
            k = (1 - q / 0.8) * (len(self.SIZES) - 1) + rng.normal(0, 1.2)
            img = blocks(src(i), self.SIZES[int(np.clip(round(k), 0, len(self.SIZES) - 1))])
        if q < 0.55 and rng.random() < 0.3:
            # Only a band of the frame made it; the rest keeps the stale picture.
            b = 32
            rows = H // b
            h = int(rng.integers(rows // 4, rows * 2 // 3))
            r0 = int(rng.integers(0, rows - h + 1))
            out = st["held"].copy()
            out[r0 * b:(r0 + h) * b] = img[r0 * b:(r0 + h) * b]
            img = out
        st["held"] = img
        fps = 3 + 27 * q ** 1.5
        st["next"] = i + max(1, round(rng.exponential(FPS / fps)))
        return img


def random_style(seed: int):
    d = RandomDrop(seed)
    return (f"R{seed}-random-drop", f"random drop, {d.dur:.1f} s, swap at {d.swap_after:.1f} s",
            d.dur, d.swap_after, d)


def meet_style(scenario: str, seed: int):
    """The codec-based Meet replay (classwatch.degrade) for one measured scenario."""
    from classwatch.degrade import Degrader, MeetDrop
    drop = MeetDrop(np.random.default_rng(seed), scenario)
    deg = Degrader(W, H, np.random.default_rng(seed + 1))

    def fn(t, i, src, st):
        return deg.frame(src(i), t, drop) if t < drop.dur else src(i)
    return (f"M-{scenario}", f"real Meet trace '{scenario}', {drop.dur:.1f} s, swap at {drop.swap_at:.1f} s",
            drop.dur, drop.swap_at, fn)


STYLES = [
    ("A-freeze-blocks", "freeze 0.35 s, huge blocks at 4 fps, back in 1.5 s", 1.5, 0.6, style_freeze_blocks),
    ("B-bitrate-ladder", "freeze, then 48 > 24 > 12 px steps, back in 1.6 s", 1.6, 0.5, style_bitrate_ladder),
    ("C-smear", "packet loss: old picture smears into new, 1.5 s", 1.5, 0.35, style_smear),
    ("D-real-x264", "frames really crushed by x264, back in 1.7 s", 1.7, 0.6, style_x264),
    ("E-hiccup", "freeze + one chunky burst, back in 0.95 s", 0.95, 0.5, style_hiccup),
    ("B2-lag-ladder", "stutter 1.2 s, blocks grow, swap, recover over 2.5 s", 4.7, 1.9, style_ladder2),
    ("B3-lag-behind-ladder", "B2 + falls 0.4 s behind before the drop", 4.7, 1.9, style_ladder3),
    *(random_style(seed) for seed in (1, 2, 3)),
    *(meet_style(name, 10 + k) for k, name in enumerate(
        ["sudden-drop", "loss-burst", "squeeze", "lag-spike", "stall-2s", "stall-5s", "congestion"])),
]


def render(name, fn, drop_secs, swap_after, before, after, outdir: pathlib.Path) -> pathlib.Path:
    drop_frame = round(LEAD * FPS)
    total = drop_frame + round((drop_secs + TAIL) * FPS)
    src = Sources(before, after, drop_frame, swap_after)
    path = outdir / f"{name}.mp4"
    with av.open(str(path), "w") as out:
        st = out.add_stream("libx264", rate=FPS)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.options = {"crf": "16", "preset": "medium"}
        scratch: dict = {}
        for i in range(total):
            t = (i - drop_frame) / FPS
            img = src(i) if t < 0 else fn(t, i, src, scratch)
            for p in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="bgr24")):
                out.mux(p)
        for p in st.encode(None):
            out.mux(p)
    return path


def grid(paths: list[pathlib.Path], labels: list[str], out: pathlib.Path) -> None:
    """3x2 grid with labels, every tile scaled to half size, for side-by-side comparison."""
    tiles = paths + [paths[0]]  # pad the sixth tile; overwritten by a black box below
    inputs = sum((["-i", str(p)] for p in tiles), [])
    font = subprocess.run(["fc-match", "-f", "%{file}", "sans:bold"], capture_output=True, text=True).stdout
    parts = []
    for k in range(6):
        if k < len(paths):
            txt = labels[k].replace(":", "\\:").replace("'", "")
            parts.append(f"[{k}:v]scale=320:240,drawtext=fontfile={font}:text='{txt}':x=6:y=6:"
                         f"fontsize=15:fontcolor=white:box=1:boxcolor=black@0.6[v{k}]")
        else:
            parts.append(f"[{k}:v]scale=320:240,drawbox=c=black:t=fill[v{k}]")
    parts.append("".join(f"[v{k}]" for k in range(6)) +
                 "xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0:shortest=0[g]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(parts),
                    "-map", "[g]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)],
                   check=True)


def main() -> None:
    before_path, after_path, outdir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    before = load(before_path, 10.0, 5.0)
    after = load(after_path, 20.0, 6.0)
    paths = []
    only = sys.argv[4:]
    for name, desc, drop_secs, swap_after, fn in STYLES:
        if only and name not in only:
            continue
        paths.append(render(name, fn, drop_secs, swap_after, before, after, outdir))
        print(f"{paths[-1].name}: {desc}")
    if not only:
        grid(paths, [f"{n}" for n, *_ in STYLES], outdir / "all-side-by-side.mp4")
        print("all-side-by-side.mp4")


if __name__ == "__main__":
    main()
