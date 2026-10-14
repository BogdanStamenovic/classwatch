import sys, time, wave
import numpy as np
from faster_whisper import WhisperModel
w = wave.open("bench.wav"); pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
CH = int(float(sys.argv[1]) * 16000)
for name in sys.argv[2:]:
    m = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=8)
    lat = []
    for i in range(0, len(pcm) - CH, CH):
        t0 = time.time()
        segs, _ = m.transcribe(pcm[i:i + CH], language=None, beam_size=1, vad_filter=False, without_timestamps=True)
        _ = [s.text for s in segs]
        lat.append(time.time() - t0)
    lat = np.array(lat)
    print(f"{name}: {sys.argv[1]}s chunks -> median {np.median(lat):.2f}s, max {lat.max():.2f}s per chunk", flush=True)
