# classwatch

Listens to an online class through a virtual audio device, notices when the teacher calls on
Bogdan, glows orange around the screen, and has the question and an answer on screen when he
presses **T**.

## How it works

```
Meet (Chrome) ──▶ PipeWire null sink "classwatch" ──loopback──▶ speakers
                         │ monitor
                         ▼
  laptop: classwatch run ── 4 s PCM chunks ──▶ archserver: classwatch-asr (RTX 4060, large-v3-turbo)
                         │                     (falls back to whisper small on the laptop CPU)
                         ▼
             rolling transcript ──▶ Jev (OpenRouter decisions API): "is Bogdan being called on now?"
                         │ p ≥ 0.5
                         ▼
   GNOME Shell extension: orange glow, T grabbed ──T──▶ Meet window fullscreen + answer panel
                         ▲
   Haiku 5.5 + Sonnet 5.5 stream answers in parallel, regenerated for 15 s as the question arrives
```

The extension only draws and grabs the key; everything else is the daemon. It talks to the
extension over D-Bus (`io.github.bogdanstamenovic.Classwatch`). Without the extension, alerts
fall back to `notify-send`.

## Measurements (2026-10-14)

Whisper, 6 s chunks of mixed Serbian/English speech:

| model | laptop CPU (i7-1265U, int8, 8 threads) | archserver RTX 4060 (int8_float16) |
|---|---|---|
| small | 3.5 s per chunk | 0.15 s |
| medium | 9.2 s (cannot keep up) | 0.37 s |
| large-v3-turbo | 15.3 s (cannot keep up) | 0.42 s |
| large-v3 | — | 0.51 s |

Short chunks cost almost as much as 30 s ones (Whisper pads to a 30 s window), so batch
real-time factors badly overstate what the CPU can do live.

Jev (`typesafe/jev-1.13-20260917`), 24 labelled transcript snippets including near-misses
("as Bogdan pointed out last week", "Bogdana, hvala ti", "bog standard"): AUROC 1.000,
23/24 at 0.5 (the miss was a surname-only call, since added to the prompt), 0.41 s,
about $0.00002 per call.

Answers, 6 class questions, no system prompt, streaming from the laptop:

| model | first token (median) | full answer (median) |
|---|---|---|
| Haiku 5.5 (reasoning off) | 0.94 s | 1.94 s |
| Haiku 4.5 | 1.17 s | 2.29 s |
| Sonnet 5.5 (effort low; it refuses reasoning off) | 1.22 s | 3.02 s |
| Sonnet 5 | 2.95 s | 3.79 s |
| warm `claude` stream-json haiku / sonnet | 1.1–2.2 / 1.5–6.1 s | 3.1–5.6 / 4.4–9.8 s |

## Install

Laptop (client):

```
cd ~/data/classwatch && python3 -m venv .venv && .venv/bin/pip install -e .
ln -sf ~/data/classwatch/.venv/bin/classwatch ~/.local/bin/classwatch
```

Needs `~/keys/openrouter` (or `OPENROUTER_API_KEY`) and `~/.config/classwatch/asr-token`
(copied from archserver).

archserver (GPU server):

```
cd ~/data/classwatch && uv venv .venv && uv pip install --python .venv -e . nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'
ln -sf ~/data/classwatch/deploy/classwatch-asr.service ~/.config/systemd/user/
systemctl --user enable --now classwatch-asr
```

The token is generated once into `~/.config/classwatch/asr-token`; the tailnet is shared with
other people, so the server refuses requests without it.

## Usage

```
classwatch meet https://meet.google.com/xxx-xxxx-xxx   # Meet as its own app window
# in Meet: Settings > Audio > Speaker > "classwatch"
classwatch run                                          # creates the sink if missing, then listens
classwatch status                                       # sink, key, archserver, extension, last log
classwatch audio down                                   # remove the virtual sink afterwards
```

### Fake camera

Order matters, because Chrome only lists the cameras that exist when it starts:

```
classwatch camera up                 # terminal 1: PipeWire camera "classwatch" playing the clip
                                     # close every Chrome window
classwatch meet https://meet.google.com/xxx-xxxx-xxx
# in Meet: Settings > Video > Camera > "classwatch"
classwatch run --camera              # terminal 2: when you are called on, lag 6 s, then swap to the real cam
classwatch camera down               # afterwards
```

The clip lives at `~/.local/state/classwatch/clip.mp4` (or `CLASSWATCH_CAM_CLIP`). Without it the
camera passes the real webcam through. Record one, sitting still, with 4:2:0 colour so ordinary
players can open it too:

```
ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 -framerate 30 -i /dev/video0 -t 60 -pix_fmt yuv420p ~/.local/state/classwatch/clip.mp4
```

`classwatch meet` grants Chrome (`com.google.Chrome`) camera access in the portal permission store.
Without that grant, GNOME refuses to show the permission dialog for Chrome and Meet's Settings and
Join buttons silently do nothing. Undo:
`gdbus call --session --dest org.freedesktop.impl.portal.PermissionStore --object-path /org/freedesktop/impl/portal/PermissionStore --method org.freedesktop.impl.portal.PermissionStore.DeletePermission devices camera com.google.Chrome`

Logs: `~/.local/state/classwatch/` (`session.log`, `transcript.txt`, `answers.log`).

Testing without a class: `classwatch run --replay lesson.wav --dry-run` (16 kHz mono wav).

## Limitations

- **The fake camera is only listed if it runs before Chrome starts**; `classwatch meet` refuses to open while Chrome is running (`--force` overrides).
- The camera's bitrate drop is faked visually (blocky frames plus delay); Meet re-encodes whatever it receives.
- **Language is forced to Serbian.** Auto-detection on 4 s chunks sometimes picked English and
  then *translated* the Serbian speech, which can drop the name. Forced `sr` still writes
  English sentences in English, but an English-only class would transcribe worse.
- **Detection lag is about one chunk plus ~1 s** (4–6 s from the name being said), and the
  question usually arrives a chunk later, so the answer firms up 4–8 s after the call.
- **The speaker loopback is pinned to the default output at creation.** Switching to
  headphones mid-class keeps the class on the old output; run `classwatch audio down && classwatch audio up`.
- The extension needs one logout/login after install (GNOME Wayland discovers extensions at login).
- On the CPU fallback, detection lag grows to ~10 s and transcripts are noticeably worse.
- Answers come from a noisy transcript and can be wrong; the panel shows both models so a
  disagreement is visible.
- Costs: about $0.004 per call-on (both answer models) plus ~$0.01 per hour of Jev checks,
  on the personal OpenRouter account.
