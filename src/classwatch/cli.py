"""classwatch: listen to an online class, notice when Bogdan is called on, have an answer ready."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave

import numpy as np

RATE = 16000
SINK = "classwatch"
STATE_DIR = pathlib.Path(os.environ.get("XDG_STATE_HOME", pathlib.Path.home() / ".local/state")) / "classwatch"
KEY_FILE = pathlib.Path.home() / "keys/openrouter"  # Jev only
CLAUDE_KEY_FILE = pathlib.Path.home() / "keys/claude.txt"
JEV_MODEL = "typesafe/jev-1.13-20260917"
FAST_MODEL = "claude-haiku-5-5"
GOOD_MODEL = "claude-sonnet-5-5"
# Thinking off on both: it is latency he would be sitting through. Haiku 5.5 takes
# "disabled" (at effort high or below); Sonnet 5.5 rejects "disabled" and calls it
# "between_tools", which no other model accepts.
ANSWER_PARAMS = {
    FAST_MODEL: {"thinking": {"type": "disabled"}},
    GOOD_MODEL: {"thinking": {"type": "between_tools"},
                 # On a safety decline the API reruns the request on another model in the same call.
                 "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"},
}
DBUS = ["--dest", "io.github.bogdanstamenovic.Classwatch",
        "--object-path", "/io/github/bogdanstamenovic/Classwatch"]
IFACE = "io.github.bogdanstamenovic.Classwatch"
EXT_VERSION = 3  # Ctrl+D / Ctrl+P camera keys need extension 3
SIGNALS = ("TakeoverPressed", "ReleasePressed", "MicOpened", "MicClosed", "RecordPressed")
ASR_URL = os.environ.get("CLASSWATCH_ASR", "http://100.72.2.62:8791")
ASR_TOKEN_FILE = pathlib.Path.home() / ".config/classwatch/asr-token"

JEV_QUESTION = {"c": {"type": "choice",
    "instructions": (
        "This is the last ~40 seconds of a live online class transcript (Serbian and English mixed, "
        "produced by noisy speech recognition, so names may be misspelled or split, e.g. 'Bog dane', "
        "'Bogdane', 'bok dan', 'Bogdan', or the surname 'Stamenović' / 'Stamenoviću'). Is the teacher, "
        "in the most recent lines, calling on or directly addressing the student Bogdan (by first name "
        "or by surname) right now and expecting him to respond?"),
    "criteria": {
        "called": "the teacher is addressing Bogdan now, asking him a question, to answer, to speak, or checking whether he is there",
        "not_called": "Bogdan is not being addressed now: someone else is, he is only mentioned, or the name is a misheard ordinary word"}}}

ANSWER_PROMPT = """Live class transcript (speech recognition, noisy, Serbian/English mixed). The teacher just called on the student Bogdan; the newest lines are at the bottom.

EARLIER CONTEXT:
{context}

MOST RECENT:
{recent}

Reply in exactly this format, in the language the teacher used for the question:
QUESTION: <the question he must answer, cleaned up, one line; if no question is asked yet, say what the teacher wants from him>
ANSWER: <what to say out loud, 1-3 short sentences>
POINTERS: <2-3 short bullet keywords he can expand on>"""


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / "session.log", "a") as f:
        f.write(line + "\n")


def api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY") or KEY_FILE.read_text().strip()


_claude = None


def claude():
    global _claude
    if _claude is None:
        import anthropic
        _claude = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY") or CLAUDE_KEY_FILE.read_text().strip(), timeout=60)
    return _claude


def http_json(url: str, payload: dict, timeout: float = 20):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {
        "Authorization": f"Bearer {api_key()}", "Content-Type": "application/json", "User-Agent": "classwatch"})
    return urllib.request.urlopen(req, timeout=timeout)


# ---- audio -----------------------------------------------------------------

def pactl(*args: str) -> str:
    return subprocess.run(["pactl", *args], capture_output=True, text=True, check=True).stdout


def sink_exists() -> bool:
    return any(line.split("\t")[1] == SINK for line in pactl("list", "short", "sinks").splitlines() if "\t" in line)


def loopback_exists() -> bool:
    return f"source={SINK}.monitor" in pactl("list", "short", "modules")


def setup_audio() -> None:
    # Each half is checked on its own: a sink without its loopback means a silent class.
    real = pactl("get-default-sink").strip()
    if real == SINK:
        raise RuntimeError(f"default sink is '{SINK}' itself; set your real speakers as default first")
    if sink_exists():
        log(f"audio: sink '{SINK}' already exists")
    else:
        pactl("load-module", "module-null-sink", f"sink_name={SINK}",
              "sink_properties=device.description=classwatch_(Meet_output)")
        # The null sink must not become the default, or every app goes through it.
        pactl("set-default-sink", real)
        log(f"audio: created sink '{SINK}'. In Meet: Settings > Audio > Speaker > 'classwatch'")
    if loopback_exists():
        log("audio: loopback to speakers already exists")
    else:
        # Loop back to the real speakers so he still hears the class.
        pactl("load-module", "module-loopback", f"source={SINK}.monitor", f"sink={real}", "latency_msec=30")
        log(f"audio: looped '{SINK}' back to {real}")


def teardown_audio() -> None:
    for line in pactl("list", "short", "modules").splitlines():
        parts = line.split("\t")
        if len(parts) > 2 and (f"sink_name={SINK}" in parts[2] or f"source={SINK}.monitor" in parts[2]):
            pactl("unload-module", parts[0])
            log(f"audio: unloaded module {parts[0]} ({parts[1]})")


class AudioSource:
    """Streams 16 kHz mono float32 into a ring buffer, from the sink monitor or a wav file."""

    def __init__(self, wav: str | None = None, speed: float = 1.0):
        self.buf = np.zeros(0, dtype=np.float32)
        self.total = 0  # samples received since start
        self.lock = threading.Lock()
        self.wav, self.speed = wav, speed
        self.done = False
        threading.Thread(target=self._run, daemon=True).start()

    def _push(self, chunk: np.ndarray) -> None:
        with self.lock:
            self.buf = np.concatenate([self.buf, chunk])[-RATE * 300:]
            self.total += len(chunk)

    def _run(self) -> None:
        if self.wav:
            w = wave.open(self.wav)
            assert w.getframerate() == RATE and w.getnchannels() == 1, "replay wav must be 16 kHz mono"
            step = RATE // 10
            while frames := w.readframes(step):
                self._push(np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768)
                time.sleep(0.1 / self.speed)
            self.done = True
            return
        p = subprocess.Popen(["parec", "-d", f"{SINK}.monitor", "--format=s16le", f"--rate={RATE}",
                              "--channels=1", "--raw", "--latency-msec=100"], stdout=subprocess.PIPE)
        while data := p.stdout.read(RATE // 5 * 2):
            self._push(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768)
        self.done = True

    def since(self, start: int) -> tuple[np.ndarray, int]:
        with self.lock:
            have_from = self.total - len(self.buf)
            start = max(start, have_from)
            return self.buf[start - have_from:].copy(), start


# ---- transcript / decisions -------------------------------------------------

class Transcript:
    def __init__(self):
        self.lines: list[tuple[float, str]] = []

    def add(self, t: float, text: str) -> None:
        self.lines.append((t, text))
        with open(STATE_DIR / "transcript.txt", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {text}\n")

    def window(self, seconds: float, until: float | None = None) -> str:
        until = until if until is not None else time.time()
        return "\n".join(x for t, x in self.lines if until - seconds <= t <= until)


def jev_called(text: str) -> float | None:
    try:
        with http_json("https://openrouter.ai/api/alpha/decisions",
                       {"model": JEV_MODEL, "state": text, "questions": JEV_QUESTION}, timeout=10) as r:
            return json.loads(r.read())["answers"]["c"]["probabilities"]["called"]
    except Exception as e:  # a down Jev must not kill the run; fall back to the name regex
        log(f"jev: error {e!r}")
        return None


# Whisper's stock outputs on near-silence (YouTube outro boilerplate in its training data).
HALLUCINATION_RE = re.compile(r"^(hvala( vam)?( na pažnji| što pratite( kanal)?)?|thank you( for watching)?|thanks for watching|"
                              r"subtitles? by.*|titlovi.*|prevod.*|pretplatite se.*)[.!]*$", re.I)
NAME_RE = re.compile(r"\bbo[gk]\s?da(n|ne)\b|stamenovi", re.I)


def stream_answer(model: str, prompt: str, on_text) -> None:
    params = ANSWER_PARAMS[model]
    api = claude().beta.messages if "betas" in params else claude().messages
    out = ""
    try:
        with api.stream(model=model, max_tokens=1000, output_config={"effort": "low"},
                        messages=[{"role": "user", "content": prompt}], **params) as stream:
            for text in stream.text_stream:
                out += text
                on_text(out)
            if stream.get_final_message().stop_reason == "refusal":
                on_text(out + f"\n[{model} declined to answer]")
    except Exception as e:
        on_text(out + f"\n[{model} failed: {e!r}]")


# ---- shell extension bridge --------------------------------------------------

class Shell:
    def __init__(self, dry: bool = False):
        self.dry = dry
        ping = self._call("Ping")
        self.ok = not dry and ping is not None
        m = re.search(r"extension (\d+)", ping or "")
        self.version = int(m.group(1)) if m else 0
        log(f"shell: extension {'connected' if self.ok else 'NOT available, using notify-send fallback'}")

    def watch(self, on_signal) -> None:
        """Call on_signal(name) for each signal in SIGNALS the extension emits."""
        if not self.ok:
            return

        def pump() -> None:
            while True:
                p = subprocess.Popen(["gdbus", "monitor", "--session", *DBUS[:2], *DBUS[2:]],
                                     stdout=subprocess.PIPE, text=True)
                assert p.stdout is not None
                for line in p.stdout:
                    m = re.search(rf"{re.escape(IFACE)}\.({'|'.join(SIGNALS)})\b", line)
                    if m:
                        on_signal(m.group(1))
                p.wait()
                time.sleep(2)  # gnome-shell restarted or the bus blipped; reconnect

        threading.Thread(target=pump, daemon=True).start()

    def _call(self, method: str, *args: str):
        if self.dry:
            return None
        r = subprocess.run(["gdbus", "call", "--session", *DBUS, "--method", f"{IFACE}.{method}", *args],
                           capture_output=True, text=True, timeout=5)
        return r.stdout if r.returncode == 0 else None

    def alert(self, on: bool) -> None:
        if self.ok:
            self._call("SetAlert", "true" if on else "false")
        elif on and not self.dry and shutil.which("notify-send"):
            subprocess.run(["notify-send", "-u", "critical", "classwatch", "Bogdan, you're being called on"])

    def content(self, question: str, answer: str) -> None:
        if self.ok:
            self._call("SetContent", question, answer)

    def camera_keys(self, on: bool) -> None:
        if self.ok and self.version >= EXT_VERSION:
            self._call("SetCameraKeys", "true" if on else "false")


# ---- main loop ----------------------------------------------------------------

class Run:
    def __init__(self, args):
        self.args = args
        self.audio = AudioSource(args.replay, args.speed)
        self.shell = Shell(dry=args.dry_run)
        self.tx = Transcript()
        self.cooldown_until = 0.0
        self.follow_until = 0.0
        self.gen = 0
        self.answer_lock = threading.Lock()
        self.answers = {FAST_MODEL: "", GOOD_MODEL: ""}
        self.last_push = 0.0
        self.jobs: queue.Queue = queue.Queue()
        self.mic_open = False  # Meet's mic, as toggled by Ctrl+D in the Meet window
        self.taken_over = False
        self.whisper = None
        self.remote_ok = not args.local and ASR_TOKEN_FILE.exists()
        # Load the CPU fallback in the background so a dead archserver costs one chunk, not a load.
        threading.Thread(target=self._load_local, daemon=True).start()
        if self.remote_ok:
            threading.Thread(target=self._warm_remote, daemon=True).start()
        threading.Thread(target=self._warm_claude, daemon=True).start()

    def _warm_claude(self) -> None:
        """Import the SDK and open the connection now: cold, the first answer of a class
        started ~2.4 s late (import 2.4 s, then the TLS handshake). count_tokens is free."""
        try:
            claude().messages.count_tokens(model=FAST_MODEL, messages=[{"role": "user", "content": "hi"}])
        except Exception as e:
            log(f"claude: warm-up failed ({e!r}); the first answer will pay for the connection")

    def _warm_remote(self) -> None:
        """Make archserver load its model now. It unloads after 15 idle minutes, and a cold
        load takes longer than the 4 s per-chunk timeout, so the first chunk of a class used
        to time out and push the whole next minute onto the CPU model."""
        t0 = time.time()
        try:
            req = urllib.request.Request(f"{ASR_URL}/transcribe", np.zeros(RATE // 2, np.int16).tobytes(),
                {"X-Token": ASR_TOKEN_FILE.read_text().strip(), "Content-Type": "application/octet-stream"})
            with urllib.request.urlopen(req, timeout=120) as r:
                r.read()
            log(f"whisper: archserver model warm ({time.time() - t0:.1f}s)")
        except Exception as e:
            log(f"whisper: archserver warm-up failed ({e!r}); chunks will retry it")

    def _load_local(self) -> None:
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel(self.args.model, device="cpu", compute_type="int8", cpu_threads=self.args.threads)
        log(f"whisper: CPU standby {self.args.model} preloaded (archserver GPU is primary)" if self.remote_ok else f"whisper: local {self.args.model} loaded (archserver not configured)")

    def transcribe(self, pcm: np.ndarray) -> tuple[str, str]:
        if self.remote_ok or time.time() - getattr(self, "remote_failed_at", 0) > 60:
            try:
                req = urllib.request.Request(f"{ASR_URL}/transcribe", (pcm * 32767).astype(np.int16).tobytes(),
                    {"X-Token": ASR_TOKEN_FILE.read_text().strip(), "Content-Type": "application/octet-stream"})
                with urllib.request.urlopen(req, timeout=4) as r:
                    self.remote_ok = True
                    return json.loads(r.read())["text"], "gpu"
            except Exception as e:
                if self.remote_ok:
                    log(f"whisper: archserver unreachable ({e!r}), falling back to local {self.args.model}")
                self.remote_ok, self.remote_failed_at = False, time.time()
        while self.whisper is None:
            time.sleep(0.2)
        segs, _ = self.whisper.transcribe(pcm, language="sr", beam_size=1, vad_filter=True,
                                          without_timestamps=True, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip(), "cpu"

    def on_key(self, signal_name: str) -> None:
        """Keys drive the fake camera: it shows the webcam while the mic is open or the call
        is taken over (T), the clip otherwise. Ctrl+P records a new clip.

        Either reason alone keeps the webcam: muting after answering must not cut back
        to the clip while Meet is still fullscreen, and T handing back must not while
        the mic is still open.
        """
        from .camera import cmd_camera
        log(f"key: {signal_name}")
        if signal_name == "TakeoverPressed":
            self.taken_over = True
        elif signal_name == "ReleasePressed":
            self.taken_over = False
        elif signal_name == "MicOpened":
            self.mic_open = True
        elif signal_name == "MicClosed":
            self.mic_open = False
        elif signal_name == "RecordPressed":
            cmd_camera("record")
            return
        cmd_camera("degrade" if self.mic_open or self.taken_over else "restore")

    def loop(self) -> int:
        chunk, overlap = int(self.args.chunk * RATE), int(0.5 * RATE)
        pos = 0
        threading.Thread(target=self.decider, daemon=True).start()
        if self.args.camera:
            self.shell.watch(self.on_key)
            if self.shell.ok and self.shell.version < EXT_VERSION:
                log(f"shell: extension {self.shell.version} is loaded, Ctrl+D / Ctrl+P need {EXT_VERSION}: "
                    "log out and back in")
            self.shell.camera_keys(True)
        log("run: listening")
        while True:
            pcm, start = self.audio.since(pos)
            if len(pcm) < chunk:
                if self.audio.done:
                    self.jobs.put(None)
                    time.sleep(self.args.drain)
                    return 0
                time.sleep(0.2)
                continue
            pcm = pcm[:chunk]
            pos = start + chunk - overlap
            level = float(np.sqrt(np.mean(pcm ** 2)))
            t0 = time.time()
            text, where = self.transcribe(pcm) if level > 1e-4 else ("", "-")
            log(f"whisper[{where}]: {time.time() - t0:.2f}s rms={level:.4f} | {text}")
            if text and level < 0.02 and HALLUCINATION_RE.search(text):
                log(f"whisper: dropped likely hallucination on quiet audio: {text}")
                text = ""
            if text:
                self.tx.add(time.time(), text)
                self.jobs.put(time.time())

    def decider(self) -> None:
        while (t := self.jobs.get()) is not None:
            if time.time() < self.follow_until:
                # The question usually lands a chunk or two after the name: re-answer with it.
                log("follow-up: new transcript after the call, regenerating answers")
                self.generate()
                continue
            if time.time() < self.cooldown_until:
                continue
            window = self.tx.window(40)
            p = jev_called(window)
            if p is None:  # Jev down: crude fallback on the newest line
                p = 1.0 if NAME_RE.search(self.tx.lines[-1][1]) else 0.0
            log(f"jev: p(called)={p:.2f}")
            if p >= self.args.threshold:
                self.called()

    def called(self) -> None:
        log("CALLED: alerting and generating answers")
        self.cooldown_until = time.time() + self.args.cooldown
        self.follow_until = time.time() + self.args.follow
        self.shell.alert(True)
        # Nobody pressed T: stop glowing rather than glow for the rest of the class.
        threading.Timer(self.args.alert_timeout, lambda: self.shell.alert(False)).start()
        self.generate()

    def generate(self) -> None:
        context, recent = self.tx.window(240), self.tx.window(40)
        prompt = ANSWER_PROMPT.format(context=context or "(none)", recent=recent)
        with self.answer_lock:
            self.gen += 1
            gen = self.gen
        for m in (FAST_MODEL, GOOD_MODEL):
            threading.Thread(target=stream_answer, args=(m, prompt, lambda s, m=m: self.update(m, s, gen)),
                             daemon=True).start()

    def update(self, model: str, text: str, gen: int) -> None:
        with self.answer_lock:
            if gen != self.gen:
                return  # a newer transcript superseded this stream
            if not text:
                return
            self.answers[model] = text
            now = time.time()
            done = text.endswith((".", "]", "\n")) or now - self.last_push > 0.25
            if not done:
                return
            self.last_push = now
            fast, good = self.answers[FAST_MODEL], self.answers[GOOD_MODEL]
            m = re.search(r"QUESTION:\s*(.+)", fast or good)
            question = m.group(1).strip() if m else "(question not clear yet)"
            body = "\n\n".join(f"[{name}]\n{re.sub(r'QUESTION:.*\n?', '', t).strip()}"
                               for name, t in (("Haiku", fast), ("Sonnet", good)) if t)
            self.shell.content(question, body)
        with open(STATE_DIR / "answers.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {model}: {text!r}\n")


def cmd_meet(url: str, force: bool = False) -> int:
    from .camera import CAM_NAME, grant_portal_camera, node_in_wpctl

    chrome = shutil.which("google-chrome-stable") or shutil.which("google-chrome")
    if not chrome:
        print("classwatch: google-chrome not found", file=sys.stderr)
        return 1
    # A running Chrome swallows the launch: the new window joins the old process, which
    # was started without the PipeWire camera flag and enumerated cameras long ago.
    if subprocess.run(["pgrep", "-f", "^/opt/google/chrome/chrome"], capture_output=True).returncode == 0 and not force:
        print("classwatch: Chrome is already running, so the fake camera would not be listed.\n"
              "Close every Chrome window first (or pass --force to open Meet anyway).", file=sys.stderr)
        return 1
    if not node_in_wpctl(CAM_NAME):
        print(f"classwatch: warning: the '{CAM_NAME}' camera is not running; Chrome only lists cameras\n"
              "present at startup. Run `classwatch camera up` first if you want it.", file=sys.stderr)
    print(f"camera portal permission for Chrome: {grant_portal_camera()}")
    subprocess.Popen([chrome, f"--app={url}", "--class=classwatch-meet",
                     "--enable-features=WebRtcPipeWireCamera"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print("Meet opened. In Meet: Settings > Video > Camera > 'classwatch'; "
          "Settings > Audio > Speaker > 'classwatch' (once `classwatch run` has created the sink).")
    return 0


def cmd_status() -> int:
    print(f"sink '{SINK}': {'present' if sink_exists() else 'missing (run: classwatch audio up)'}")
    print(f"openrouter key (Jev): {'present' if KEY_FILE.exists() or os.environ.get('OPENROUTER_API_KEY') else 'MISSING'}")
    print(f"claude key (answers): {'present' if CLAUDE_KEY_FILE.exists() or os.environ.get('ANTHROPIC_API_KEY') else 'MISSING'}")
    try:
        with urllib.request.urlopen(f"{ASR_URL}/health", timeout=3) as r:
            print(f"archserver asr: {json.loads(r.read())}")
    except Exception as e:
        print(f"archserver asr: unreachable ({e!r}); runs will use the local CPU model")
    print(f"shell extension: {'connected' if Shell().ok else 'not loaded (log out and back in after install)'}")
    for name in ("session.log",):
        p = STATE_DIR / name
        if p.exists():
            print(f"--- last lines of {p}")
            print("\n".join(p.read_text().splitlines()[-8:]))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="classwatch", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="listen to the class and alert when Bogdan is called on")
    r.add_argument("--model", default="small", help="local CPU fallback whisper model (small is the only one that keeps up live on the laptop)")
    r.add_argument("--local", action="store_true", help="skip the archserver GPU and transcribe on this CPU")
    r.add_argument("--chunk", type=float, default=4.0, help="seconds of audio per whisper pass")
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--threshold", type=float, default=0.5, help="Jev p(called) needed to alert")
    r.add_argument("--cooldown", type=float, default=45.0, help="seconds before another alert can fire")
    r.add_argument("--alert-timeout", type=float, default=90.0, help="seconds the glow stays on if T is never pressed")
    r.add_argument("--follow", type=float, default=15.0, help="seconds after a call during which new speech regenerates the answer")
    r.add_argument("--replay", metavar="WAV", help="feed a 16 kHz mono wav instead of the live sink (testing)")
    r.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    r.add_argument("--drain", type=float, default=8.0, help="seconds to wait for answers after a replay ends")
    r.add_argument("--dry-run", action="store_true", help="no shell extension calls, just log")
    r.add_argument("--camera", action="store_true", help="drive a running `classwatch camera up` (start it first): T (takeover) or Ctrl+D (mic open) in Meet lag and swap to the real cam, until both are off again; Ctrl+P in Meet records a new 60 s clip")
    c = sub.add_parser("camera", help="fake webcam: prerecorded clip, degrades and swaps to the real cam when called on")
    c.add_argument("action", choices=["up", "down", "degrade", "restore", "record"])
    c.add_argument("--clip", help="path to the prerecorded clip (default: state dir clip.mp4)")
    c.add_argument("--real", default="/dev/video0", help="real webcam device")
    a = sub.add_parser("audio", help="create or remove the virtual Meet output sink")
    a.add_argument("action", choices=["up", "down"])
    m = sub.add_parser("meet", help="open a Meet link as its own Chrome app window")
    m.add_argument("url")
    m.add_argument("--force", action="store_true", help="open even if Chrome is already running (fake camera will not be listed)")
    sub.add_parser("status", help="show what is set up and the last log lines")
    sv = sub.add_parser("serve-asr", help="run the GPU transcription server (archserver)")
    sv.add_argument("--host", default="100.72.2.62")
    sv.add_argument("--port", type=int, default=8791)
    sv.add_argument("--model", default="large-v3-turbo")
    sv.add_argument("--idle-unload", type=float, default=900, help="free VRAM after this many idle seconds")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "audio":
            setup_audio() if args.action == "up" else teardown_audio()
            return 0
        if args.cmd == "camera":
            from .camera import cmd_camera
            return cmd_camera(args.action, clip=args.clip, real=args.real)
        if args.cmd == "meet":
            return cmd_meet(args.url, force=args.force)
        if args.cmd == "status":
            return cmd_status()
        if args.cmd == "serve-asr":
            from .asr_server import serve
            return serve(args.host, args.port, args.model, args.idle_unload)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if not args.replay:
            setup_audio()
        run = Run(args)
        try:
            return run.loop()
        finally:
            if args.camera:
                run.shell.camera_keys(False)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"classwatch: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
