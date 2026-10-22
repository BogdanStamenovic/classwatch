"""Run the real CameraStream with its output captured to an mp4: clip, a switch to the
"webcam" (another clip standing in), and back. Usage: camera_capture.py OUT.mp4 [--no-baseline]"""
import pathlib, resource, sys, threading, time
import av, numpy as np
import classwatch.camera as cam

out_path, baseline = sys.argv[1], "--no-baseline" not in sys.argv
state = pathlib.Path.home() / ".local/state/classwatch"
cam.WebcamSource = lambda dev: cam.ClipSource(state / "clip.prev.mp4")
cam.gst_available = lambda: True
frames, lock = [], threading.Lock()
cam.CameraStream._write_frame = lambda self, b: (lock.acquire(), frames.append(b), lock.release())
cam.log = lambda m: print(f"{time.monotonic() - T0:6.2f}s {m}", flush=True)
T0 = time.monotonic()
c0 = resource.getrusage(resource.RUSAGE_SELF)
s = cam.CameraStream(baseline=baseline)
s.start(str(state / "clip.mp4"))
time.sleep(6); s.degrade(); time.sleep(16); s.restore(); time.sleep(16)
s.stop()
c1 = resource.getrusage(resource.RUSAGE_SELF)
wall = time.monotonic() - T0
print(f"frames {len(frames)} in {wall:.1f}s = {len(frames)/wall:.1f}/s; CPU {(c1.ru_utime + c1.ru_stime - c0.ru_utime - c0.ru_stime) / wall * 100:.0f}% of one core")
with av.open(out_path, "w") as o:
    st = o.add_stream("libx264", rate=30); st.width, st.height, st.pix_fmt = 640, 480, "yuv420p"; st.options = {"crf": "16"}
    for b in frames:
        for p in st.encode(av.VideoFrame.from_ndarray(np.frombuffer(b, np.uint8).reshape(480, 640, 3), format="bgr24")): o.mux(p)
    for p in st.encode(None): o.mux(p)
