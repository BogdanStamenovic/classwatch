#!/bin/sh
# classwatch lifecycle: ./install.sh setup | update | remove | role
# Called by ownbox (install.setup / install.update / install.remove), or by hand from any clone.
# Safe to rerun. Everything it installs points at this checkout, wherever that is.
#
# Two roles on two machines:
#   server  archserver (CUDA GPU): the socket-activated ASR server, classwatch-asr.{socket,service}
#   client  the laptop: `classwatch run`, the GNOME Shell extension, the virtual sink and fake camera
# Role: $CLASSWATCH_ROLE, else ~/.config/classwatch/role, else server if nvidia-smi sees a GPU.
# setup writes the role it used to that file, so update and remove act on the same role later.
#
# Rollback safety: `ownbox rollback` reruns an OLDER revision's update on whatever a newer one
# left behind. So update never migrates or rewrites user data (token, keys, role, logs, clip);
# it only rebuilds the venv and recopies units and the extension from this checkout.
set -eu
cd "$(dirname "$0")"
DIR=$(pwd)
VENV="$DIR/.venv"
CONF_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/classwatch"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/classwatch"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
TOKEN="$CONF_DIR/asr-token"
ROLE_FILE="$CONF_DIR/role"
EXT_UUID=classwatch@bogdanstamenovic.github.io
EXT_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/gnome-shell/extensions/$EXT_UUID"
ASR_ADDR=100.72.2.62:8791
ASR_HOST=archserver

say() { printf 'classwatch: %s\n' "$*"; }
die() { say "$*"; exit 1; }

# ask QUESTION DEFAULT(y|n): asks on a TTY; without one returns DEFAULT and says so.
ask() {
  if [ -t 0 ]; then
    case "$2" in y) hint='[Y/n]';; *) hint='[y/N]';; esac
    printf 'classwatch: %s %s ' "$1" "$hint"
    read -r ans || ans=
    [ -n "$ans" ] || ans=$2
    case "$ans" in y|Y|yes) return 0;; *) return 1;; esac
  fi
  say "no TTY, taking the default ($2) for: $1"
  [ "$2" = y ]
}

role() {
  r=${CLASSWATCH_ROLE:-}
  [ -n "$r" ] || { [ -f "$ROLE_FILE" ] && r=$(tr -d ' \n' < "$ROLE_FILE"); } || true
  if [ -z "$r" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
      r=server
    else
      r=client
    fi
  fi
  case "$r" in server|client) echo "$r";; *) die "unknown role '$r' (want server or client)";; esac
}

# Heavy pip installs (the CUDA wheels are ~1 GB) run in a memory-capped scope: archserver has
# OOM-killed other work during installs before. Without a user systemd it just runs.
capped() {
  if command -v systemd-run >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    systemd-run --user --scope -q -p MemoryMax=3G -- "$@"
  else
    "$@"
  fi
}

venv() {
  # ctranslate2 (faster-whisper) has no wheels for every new CPython, so uv pins 3.13, the
  # version the server has run on since 2026-10. Without uv, the system python3 must do.
  if command -v uv >/dev/null 2>&1; then
    [ -x "$VENV/bin/python" ] || uv venv -q --python 3.13 "$VENV"
    pip() { capped uv pip install -q --python "$VENV/bin/python" "$@"; }
  else
    [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
    pip() { capped "$VENV/bin/python" -m pip install -q "$@"; }
  fi
  if [ "$1" = server ]; then
    pip -e "$DIR" nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'
  else
    pip -e "$DIR"
  fi
  "$VENV/bin/classwatch" --help >/dev/null
}

# --- server ----------------------------------------------------------------------------------

token() {
  if [ -s "$TOKEN" ]; then
    say "keeping the existing ASR token ($TOKEN)"
    return
  fi
  mkdir -p "$CONF_DIR"
  (umask 077; "$VENV/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))' > "$TOKEN")
  say "generated a new ASR token in $TOKEN; copy it to the laptop (client setup can fetch it)"
}

# Units are copied, not linked: `systemctl --user disable` on a linked unit deletes the link,
# after which the socket refuses to start. Writing through `>` onto an old link would also
# overwrite the dev checkout's file, hence the rm first.
units() {
  mkdir -p "$UNIT_DIR"
  sock_old=$(cat "$UNIT_DIR/classwatch-asr.socket" 2>/dev/null || true)
  rm -f "$UNIT_DIR/classwatch-asr.service" "$UNIT_DIR/classwatch-asr.socket"
  sed "s|@DIR@|$DIR|g" "$DIR/deploy/classwatch-asr.service" > "$UNIT_DIR/classwatch-asr.service"
  cp "$DIR/deploy/classwatch-asr.socket" "$UNIT_DIR/classwatch-asr.socket"
  # An older install enabled the socket from a linked unit, so sockets.target.wants points at
  # that checkout's file. Repoint it at the copy; this does not stop the listening socket.
  wants="$UNIT_DIR/sockets.target.wants/classwatch-asr.socket"
  if [ -L "$wants" ] && [ "$(readlink "$wants")" != "$UNIT_DIR/classwatch-asr.socket" ]; then
    rm -f "$wants"
  fi
  systemctl --user daemon-reload
  systemctl --user enable classwatch-asr.socket >/dev/null 2>&1
  if [ "$sock_old" != "$(cat "$UNIT_DIR/classwatch-asr.socket")" ]; then
    # Only a changed socket is restarted: restarting closes the port for a moment.
    systemctl --user restart classwatch-asr.socket
    say "socket unit changed; restarted classwatch-asr.socket"
  else
    systemctl --user start classwatch-asr.socket
  fi
  # A running server keeps the old venv and code. Stopping it is enough: the socket starts the
  # new one on the next request (~0.8 s, plus ~3.3 s to load the model).
  if systemctl --user is-active -q classwatch-asr.service; then
    systemctl --user stop classwatch-asr.service
    say "stopped the running server; the socket starts the new one on the next request"
  fi
}

# Proves the venv, the unit's ExecStart and the socket all work, by starting the server
# through the socket. It is stopped again so the check does not hold ~1 GB for 15 minutes.
check_server() {
  "$VENV/bin/python" -c 'import ctranslate2, sys; sys.exit(ctranslate2.get_cuda_device_count() < 1)' \
    || die "ctranslate2 sees no CUDA device; the server would fail on first use"
  if ! command -v curl >/dev/null 2>&1; then
    say "curl missing; skipped the /health check"
    return
  fi
  out=$(curl -sS -m 30 "http://$ASR_ADDR/health") || die "no answer from http://$ASR_ADDR/health (journalctl --user -u classwatch-asr)"
  say "health: $out"
  systemctl --user stop classwatch-asr.service
}

server_remove() {
  # Only units this checkout installed: a remove of a half-done install must not take down a
  # server that another checkout (a dev clone, the old linked recipe) is still running.
  if [ -L "$UNIT_DIR/classwatch-asr.service" ] \
     || ! grep -qF "$DIR/.venv/bin/classwatch" "$UNIT_DIR/classwatch-asr.service" 2>/dev/null; then
    say "classwatch-asr units are not from $DIR; left them alone"
    return
  fi
  systemctl --user stop classwatch-asr.socket classwatch-asr.service 2>/dev/null || true
  systemctl --user disable classwatch-asr.socket 2>/dev/null || true
  rm -f "$UNIT_DIR/classwatch-asr.service" "$UNIT_DIR/classwatch-asr.socket" \
        "$UNIT_DIR/sockets.target.wants/classwatch-asr.socket"
  systemctl --user daemon-reload || true
  say "removed classwatch-asr.socket and .service (the whisper model cache in ~/.cache/huggingface is shared and kept)"
}

# --- client ----------------------------------------------------------------------------------

client_deps() {
  for c in pactl:pipewire-pulse wpctl:wireplumber gst-launch-1.0:gstreamer ffmpeg:ffmpeg \
           gdbus:glib2 notify-send:libnotify google-chrome-stable:google-chrome; do
    command -v "${c%%:*}" >/dev/null 2>&1 || say "warning: ${c%%:*} not found (${c#*:}); some commands will not work"
  done
  [ -f "$HOME/keys/claude.txt" ] || [ -n "${ANTHROPIC_API_KEY:-}" ] \
    || say "warning: no ~/keys/claude.txt or ANTHROPIC_API_KEY; no answers will be generated"
  [ -f "$HOME/keys/openrouter" ] || [ -n "${OPENROUTER_API_KEY:-}" ] \
    || say "warning: no ~/keys/openrouter or OPENROUTER_API_KEY; call detection (Jev) will not work"
}

client_token() {
  if [ -s "$TOKEN" ]; then
    say "keeping the existing ASR token ($TOKEN)"
    return
  fi
  # Copying only creates a missing file, so it is the default even without a TTY.
  if ask "no ASR token; copy it from $ASR_HOST over ssh?" y; then
    mkdir -p "$CONF_DIR"
    if (umask 077; ssh -o BatchMode=yes -o ConnectTimeout=5 "$ASR_HOST" cat .config/classwatch/asr-token > "$TOKEN.tmp") \
       && [ -s "$TOKEN.tmp" ]; then
      mv "$TOKEN.tmp" "$TOKEN"
      say "copied the ASR token from $ASR_HOST"
      return
    fi
    rm -f "$TOKEN.tmp"
  fi
  say "no ASR token: runs will use the local CPU model. Later: scp $ASR_HOST:.config/classwatch/asr-token $TOKEN"
}

extension() {
  old_ver=$(sed -n 's/.*"version": *\([0-9]*\).*/\1/p' "$EXT_DIR/metadata.json" 2>/dev/null || true)
  # A copy, not a link into a checkout: ownbox rollback and uninstall replace the checkout.
  rm -rf "$EXT_DIR"
  mkdir -p "$EXT_DIR"
  cp "$DIR/gnome-extension/extension.js" "$DIR/gnome-extension/metadata.json" \
     "$DIR/gnome-extension/stylesheet.css" "$EXT_DIR/"
  new_ver=$(sed -n 's/.*"version": *\([0-9]*\).*/\1/p' "$EXT_DIR/metadata.json")
  if command -v gnome-extensions >/dev/null 2>&1 && gnome-extensions enable "$EXT_UUID" 2>/dev/null; then
    :
  elif command -v gsettings >/dev/null 2>&1; then
    # No shell running (ssh, headless): put it in the list GNOME reads at login.
    cur=$(gsettings get org.gnome.shell enabled-extensions 2>/dev/null || echo '@as []')
    case "$cur" in
      *"'$EXT_UUID'"*) ;;
      "@as []"|"[]") gsettings set org.gnome.shell enabled-extensions "['$EXT_UUID']" ;;
      *) gsettings set org.gnome.shell enabled-extensions "${cur%]}, '$EXT_UUID']" ;;
    esac
  fi
  if [ "$old_ver" != "$new_ver" ]; then
    say "GNOME extension version ${old_ver:-none} -> $new_ver installed; log out and back in once to load it"
  else
    say "GNOME extension v$new_ver refreshed (GNOME on Wayland reloads code only at login)"
  fi
}

client_remove() {
  # Anchored on python so an agent or editor whose command line mentions it does not count.
  if pgrep -f '^[^ ]*python[0-9.]* .*(bin/classwatch|-m classwatch\.cli) run( |$)' >/dev/null 2>&1; then
    die "classwatch run is running (a class?); stop it first, nothing was removed"
  fi
  if [ -x "$VENV/bin/classwatch" ]; then
    "$VENV/bin/classwatch" camera down >/dev/null 2>&1 || true
    "$VENV/bin/classwatch" audio down >/dev/null 2>&1 || true
    say "removed the virtual sink and stopped the fake camera"
  fi
  command -v gnome-extensions >/dev/null 2>&1 && gnome-extensions disable "$EXT_UUID" 2>/dev/null || true
  rm -rf "$EXT_DIR"
  say "removed the GNOME extension"
  say "the Chrome camera grant in the portal permission store is kept (README, Fake camera, has the undo)"
}

# --- lifecycle -------------------------------------------------------------------------------

remember_role() {
  mkdir -p "$CONF_DIR"
  [ "$(cat "$ROLE_FILE" 2>/dev/null)" = "$1" ] || { echo "$1" > "$ROLE_FILE"; say "role $1 saved to $ROLE_FILE"; }
}

R=$(role)
case "${1:-}" in
  setup|update)
    say "$1 as $R in $DIR"
    venv "$R"
    remember_role "$R"
    if [ "$R" = server ]; then
      token
      units
      check_server
      say "ASR server listening on $ASR_ADDR (socket-activated). Logs: journalctl --user -u classwatch-asr"
    else
      client_deps
      client_token
      extension
      mkdir -p "$STATE_DIR"
      say "client ready: classwatch status; logs in $STATE_DIR"
    fi
    ;;
  remove)
    say "remove as $R"
    if [ "$R" = server ]; then server_remove; else client_remove; fi
    rm -rf "$VENV"
    if ask "also delete the ASR token, role and config ($CONF_DIR) and the logs and camera clip ($STATE_DIR)?" n; then
      rm -rf "$CONF_DIR" "$STATE_DIR"
      say "deleted $CONF_DIR and $STATE_DIR"
    else
      say "kept $CONF_DIR (token, role) and $STATE_DIR (logs, clip); ~/keys is never touched"
    fi
    say "removed"
    ;;
  role) echo "$R" ;;
  *) echo "usage: $0 setup|update|remove|role" >&2; exit 2 ;;
esac
