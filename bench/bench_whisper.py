import sys, time, wave
import numpy as np
from faster_whisper import WhisperModel
AUDIO = "bench.wav"
w=wave.open(AUDIO); pcm=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768; DUR=len(pcm)/16000
for name in sys.argv[1:]:
    t0 = time.time()
    m = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=8)
    load = time.time() - t0
    t0 = time.time()
    segs, info = m.transcribe(pcm, language=None, beam_size=1, vad_filter=True)
    text = " ".join(s.text.strip() for s in segs)
    dt = time.time() - t0
    print(f"{name}: load {load:.1f}s, transcribe {dt:.1f}s for {DUR:.0f}s audio -> RTF {dt/DUR:.3f} ({DUR/dt:.1f}x realtime)", flush=True)
    print("   ", text[:300], flush=True)
