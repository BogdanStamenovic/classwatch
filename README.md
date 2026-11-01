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
   Haiku 5.5 + Sonnet 5.5 (Claude API, thinking off) stream answers in parallel, regenerated for 15 s as the question arrives
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

Through ownbox, on both machines (the same command; the role differs):

```
ownbox sync && ownbox install classwatch
```

`install.sh` does the whole lifecycle for the machine's role:

| Role | Chosen when | setup / update installs | remove undoes |
|---|---|---|---|
| server | `nvidia-smi` lists a GPU (archserver) | venv with the CUDA wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12==9.*`), the ASR token if missing, `classwatch-asr.socket` + `.service` copied into `~/.config/systemd/user`, then a `/health` check through the socket | the socket and service units |
| client | no GPU (the laptop) | venv, the GNOME extension copied into `~/.local/share/gnome-shell/extensions`, the ASR token fetched from archserver over ssh if missing, warnings for missing keys and tools | the extension, the virtual sink, the fake camera; refuses while `classwatch run` is running |

Override the role with `CLASSWATCH_ROLE=server|client` or `~/.config/classwatch/role`. setup writes
the role it used to that file, so a later update or remove acts on the same role.

- **update keeps user data**: the token, the role file, `~/keys`, the logs and the camera clip are
  never rewritten. It only rebuilds the venv and recopies units and the extension, so
  `ownbox rollback` (which reruns the older revision's update) is safe.
- **The socket keeps listening through an update**: it is restarted only if its unit file
  changed. A running server is stopped and the socket starts the new one on the next request.
- **remove asks** before deleting `~/.config/classwatch` (token, role) and
  `~/.local/state/classwatch` (logs, clip). Without a TTY it keeps them and says so.
- The CUDA pip install runs under `systemd-run --user --scope -p MemoryMax=3G`, and the server
  venv takes 2.6 GB (uv hardlinks it from its cache, so a reinstall costs no extra disk).
- Units are copied rather than linked: `systemctl --user disable` on a linked unit deletes the
  link, after which the socket refuses to start.

Keys the client needs: `~/keys/claude.txt` (or `ANTHROPIC_API_KEY`) for the answers and
`~/keys/openrouter` (or `OPENROUTER_API_KEY`) for Jev. The extension loads at the next login.

Moving a machine from a dev-checkout install (the old manual recipe) to ownbox: on the laptop,
delete the `~/.local/bin/classwatch` symlink the old recipe made, because ownbox refuses to replace a
launcher it did not create. Setup replaces linked units and the extension in place.

Fallback without ownbox: `./install.sh setup` from any clone does the same job, pointing
everything at that clone.

The server is socket-activated: systemd holds port 8791 and starts `classwatch-asr` on the
first connection. After `--idle-unload` seconds (default 900) without a request it unloads the
model **and exits**, because unloading frees VRAM but leaves ~1.2 GB of CUDA context and
libraries in the process (measured 2026-10-31: 7 MB resident, 1.2 GB in swap, while idle).
A cold start costs ~0.8 s to answer plus ~3.3 s to load the model, which the laptop's 120 s
warm-up request covers.

Server setup generates the token once into `~/.config/classwatch/asr-token`; the tailnet is shared with
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
classwatch run --camera              # terminal 2
classwatch camera down               # afterwards
```

With `run --camera`, keys drive the camera. The camera shows the webcam while the mic is open
**or** the call is taken over, and the clip otherwise:

| Press | What happens |
|---|---|
| T while the edge glows | Meet goes fullscreen with the answer panel; the fake link collapses (blocky, ~4-7 fps, short freezes) for 6 s, the webcam silently replaces the clip, then the link recovers over 20 s |
| T again | Meet is minimized; unless the mic is open, the link collapses for 3 s, the clip replaces the webcam, then recovers over 6 s |
| Ctrl+D in Meet | Meet toggles the mic as usual. Opening it swaps to the webcam (same lag as T); closing it swaps back to the clip unless the call is taken over |
| Ctrl+P in Meet | Swaps to the webcam (same lag), records 60 s of it as the new clip while the call keeps showing you live, then swaps back to the new clip. The old clip is kept as `clip.prev.mp4`. Closing the mic during the recording cancels it |

Ctrl+D and Ctrl+P only do this with the Meet window focused, and only while `run --camera` is
running; everywhere else they behave normally. The extension has to grab them to see them at
all (GNOME gives a window's keys to the window before any extension can look), so it resends
Ctrl+D to Meet itself, 100 ms after you let go. Ctrl+P is kept from Meet so Chrome's print dialog
does not open over the call. The extension tracks the mic by counting Ctrl+D and assumes it starts
muted; unmuting with the mouse puts the camera out of step until the next Ctrl+D.

The link also dips for ~2 s around every loop seam (starting 0.5 s before the wrap), so the jump
cut reads as a network hiccup. Lag is faked by showing fewer distinct frames, never by delaying
them, so the webcam never plays in slow motion. Measured through the PipeWire node: 30 distinct
frames/s when clean, about 6 while collapsed.

The GNOME extension changed in version 3 (Ctrl+D and Ctrl+P) and version 2 (the T toggle, and keeping the panel visible over a
fullscreen Meet). GNOME on Wayland loads extension code only at login, so log out and back in
once after updating.

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

- **During a takeover T is grabbed globally**: a `t` typed into Meet's chat hands the call back. Click the panel to hide it without giving up the call.
- **Ctrl+D reaches Meet 100 ms late**, and a second Ctrl+D within ~0.5 s of the first goes straight to Meet without moving the camera (the keys are briefly ungrabbed while the copy is sent). Copies sent sooner were dropped by GNOME in testing.
- **Ctrl+P records whatever the webcam sees for the full 60 s**; sit still. There is no preview before it replaces the clip (`clip.prev.mp4` is the undo).
- **The fake camera is only listed if it runs before Chrome starts**; `classwatch meet` refuses to open while Chrome is running (`--force` overrides).
- The camera's bitrate drop is faked visually (blocky frames plus delay); Meet re-encodes whatever it receives.
- **Language is forced to Serbian.** Auto-detection on 4 s chunks sometimes picked English and
  then *translated* the Serbian speech, which can drop the name. Forced `sr` still writes
  English sentences in English, but an English-only class would transcribe worse.
- **Detection lag is about one chunk plus ~1 s** (4–6 s from the name being said), and the
  question usually arrives a chunk later, so the answer firms up 4–8 s after the call.
- **The speaker loopback is pinned to the default output at creation.** Switching to
  headphones mid-class keeps the class on the old output; run `classwatch audio down && classwatch audio up`.
- **A first server install needs ~2.6 GB free on the disk holding the uv cache.** On 2026-10-31
  it failed with 1.0 GB free on archserver's root and left the old install untouched.
- **An update on archserver drops a request in flight**: it stops the running server so the new
  code takes over. Do not update during a class; the laptop falls back to its CPU model meanwhile.
- **The client's remove has not run on the laptop**, only in a throwaway home directory: it
  disables the extension in the live GNOME session, and re-enabling a deleted extension may
  need a logout. Install, update and rollback were exercised there on 2026-11-01.
- **The full pipeline was not re-run after the move to ownbox** (`run --replay` calls the paid
  answer and Jev APIs). What was checked on the laptop: the extension connects over D-Bus, and
  a real speech chunk transcribes on archserver with the copied token.
- The extension needs one logout/login after install (GNOME Wayland discovers extensions at login).
- On the CPU fallback, detection lag grows to ~10 s and transcripts are noticeably worse.
- Answers come from a noisy transcript and can be wrong; the panel shows both models so a
  disagreement is visible.
- Costs: about $0.004 per call-on (both answer models) plus ~$0.01 per hour of Jev checks,
  on the personal OpenRouter account.
