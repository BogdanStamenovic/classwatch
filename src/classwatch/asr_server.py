"""GPU half of classwatch: a tiny HTTP server that transcribes 16 kHz mono s16le PCM chunks.

Runs on archserver (RTX 4060). The laptop posts each audio chunk here and falls back to its own
CPU model when this is unreachable. The tailnet is shared with other people, so every request
needs the token.
"""
from __future__ import annotations

import ctypes
import glob
import json
import os
import pathlib
import site
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

TOKEN_FILE = pathlib.Path.home() / ".config/classwatch/asr-token"


def preload_cuda_libs() -> None:
    # ctranslate2 dlopens libcublas/libcudnn by soname; the pip nvidia-* wheels are not on the
    # loader path, so load them RTLD_GLOBAL first instead of depending on LD_LIBRARY_PATH.
    for sp in site.getsitepackages():
        for pattern in ("nvidia/cublas/lib/libcublasLt.so.*", "nvidia/cublas/lib/libcublas.so.*",
                        "nvidia/cudnn/lib/libcudnn*.so.*"):
            for lib in sorted(glob.glob(os.path.join(sp, pattern))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


class Model:
    def __init__(self, name: str, idle_unload: float, language: str | None = "sr",
                 on_idle=None):
        self.name, self.idle_unload, self.language = name, idle_unload, language
        self.on_idle = on_idle
        self.model = None
        self.lock = threading.Lock()
        self.last_used = time.time()
        threading.Thread(target=self._reaper, daemon=True).start()

    def _reaper(self) -> None:
        while True:
            time.sleep(30)
            with self.lock:
                if time.time() - self.last_used <= self.idle_unload:
                    continue
                if self.model is not None:
                    self.model = None
                    print(f"unloaded {self.name} after {self.idle_unload:.0f}s idle", flush=True)
                if self.on_idle is not None:
                    # Dropping the model frees VRAM but not the ~1.2 GB of CUDA context and
                    # libraries the process keeps; only exiting returns that. Under socket
                    # activation systemd starts us again on the next connection.
                    self.on_idle()
                    return

    def transcribe(self, pcm: np.ndarray) -> str:
        with self.lock:
            if self.model is None:
                from faster_whisper import WhisperModel
                t0 = time.time()
                self.model = WhisperModel(self.name, device="cuda", compute_type="int8_float16")
                print(f"loaded {self.name} in {time.time() - t0:.1f}s", flush=True)
            self.last_used = time.time()
            segs, _ = self.model.transcribe(pcm, language=self.language, beam_size=1, vad_filter=True,
                                            without_timestamps=True, condition_on_previous_text=False)
            return " ".join(s.text.strip() for s in segs).strip()


def systemd_socket() -> socket.socket | None:
    """The listening socket systemd passed us (classwatch-asr.socket), if any."""
    if os.environ.get("LISTEN_PID") != str(os.getpid()) or os.environ.get("LISTEN_FDS") != "1":
        return None
    return socket.socket(fileno=3)


def serve(host: str, port: int, model_name: str, idle_unload: float) -> int:
    preload_cuda_libs()
    token = TOKEN_FILE.read_text().strip()
    inherited = systemd_socket()
    httpd: list[ThreadingHTTPServer] = []

    def exit_idle() -> None:
        print(f"exiting after {idle_unload:.0f}s idle; systemd restarts on demand", flush=True)
        threading.Thread(target=httpd[0].shutdown, daemon=True).start()

    # Only exit when idle if something will start us again; a hand-run server keeps the old
    # unload-but-stay-up behaviour.
    model = Model(model_name, idle_unload, on_idle=exit_idle if inherited else None)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "model": model_name, "loaded": model.model is not None})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.headers.get("X-Token") != token:
                return self._send(403, {"error": "bad token"})
            if self.path != "/transcribe":
                return self._send(404, {"error": "not found"})
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
            t0 = time.time()
            text = model.transcribe(pcm)
            self._send(200, {"text": text, "secs": round(time.time() - t0, 3)})

        def log_message(self, fmt, *args):
            pass

    if inherited is not None:
        server = ThreadingHTTPServer(inherited.getsockname()[:2], Handler, bind_and_activate=False)
        server.socket.close()
        server.socket = inherited
        where = "socket from systemd"
    else:
        server = ThreadingHTTPServer((host, port), Handler)
        where = f"{host}:{port}"
    httpd.append(server)
    print(f"classwatch asr: {model_name} on {where}", flush=True)
    server.serve_forever()
    return 0
