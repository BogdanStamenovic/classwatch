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
KEY_FILE = pathlib.Path.home() / "keys/openrouter"
JEV_MODEL = "typesafe/jev-1.13-20260917"
FAST_MODEL = "anthropic/claude-haiku-5.5"
GOOD_MODEL = "anthropic/claude-sonnet-5.5"
DBUS = ["--dest", "io.github.bogdanstamenovic.Classwatch",
        "--object-path", "/io/github/bogdanstamenovic/Classwatch"]
IFACE = "io.github.bogdanstamenovic.Classwatch"
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
    payload = {"model": model, "stream": True, "max_tokens": 1000,
               "messages": [{"role": "user", "content": prompt}],
               # Sonnet 5.5 rejects reasoning off (400 "mandatory"); low effort is the floor.
               "reasoning": {"effort": "low"} if "sonnet-5.5" in model else {"enabled": False}}
    out = ""
    try:
        with http_json("https://openrouter.ai/api/v1/chat/completions", payload, timeout=60) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                c = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
                if c:
                    out += c
                    on_text(out)
    except Exception as e:
        on_text(out + f"\n[{model} failed: {e!r}]")


# ---- shell extension bridge --------------------------------------------------

class Shell:
    def __init__(self, dry: bool = False):
        self.dry = dry
        self.ok = not dry and self._call("Ping") is not None
        log(f"shell: extension {'connected' if self.ok else 'NOT available, using notify-send fallback'}")

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
        self.whisper = None
        self.remote_ok = not args.local and ASR_TOKEN_FILE.exists()
        # Load the CPU fallback in the background so a dead archserver costs one chunk, not a load.
        threading.Thread(target=self._load_local, daemon=True).start()

    def _load_local(self) -> None:
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel(self.args.model, device="cpu", compute_type="int8", cpu_threads=self.args.threads)
        log(f"whisper: local fallback {self.args.model} loaded")

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

    def loop(self) -> int:
        chunk, overlap = int(self.args.chunk * RATE), int(0.5 * RATE)
        pos = 0
        threading.Thread(target=self.decider, daemon=True).start()
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


def cmd_meet(url: str) -> int:
    chrome = shutil.which("google-chrome-stable") or shutil.which("google-chrome")
    if not chrome:
        print("classwatch: google-chrome not found", file=sys.stderr)
        return 1
    subprocess.Popen([chrome, f"--app={url}", "--class=classwatch-meet"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print("Meet opened. In Meet: Settings > Audio > Speaker > 'classwatch'.")
    return 0


def cmd_status() -> int:
    print(f"sink '{SINK}': {'present' if sink_exists() else 'missing (run: classwatch audio up)'}")
    print(f"openrouter key: {'present' if KEY_FILE.exists() or os.environ.get('OPENROUTER_API_KEY') else 'MISSING'}")
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
    r.add_argument("--follow", type=float, default=15.0, help="seconds after a call during which new speech regenerates the answer")
    r.add_argument("--replay", metavar="WAV", help="feed a 16 kHz mono wav instead of the live sink (testing)")
    r.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    r.add_argument("--drain", type=float, default=8.0, help="seconds to wait for answers after a replay ends")
    r.add_argument("--dry-run", action="store_true", help="no shell extension calls, just log")
    a = sub.add_parser("audio", help="create or remove the virtual Meet output sink")
    a.add_argument("action", choices=["up", "down"])
    m = sub.add_parser("meet", help="open a Meet link as its own Chrome app window")
    m.add_argument("url")
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
        if args.cmd == "meet":
            return cmd_meet(args.url)
        if args.cmd == "status":
            return cmd_status()
        if args.cmd == "serve-asr":
            from .asr_server import serve
            return serve(args.host, args.port, args.model, args.idle_unload)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if not args.replay:
            setup_audio()
        return Run(args).loop()
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"classwatch: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
