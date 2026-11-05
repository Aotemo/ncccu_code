#!/usr/bin/env bash
# Shared local SSH slots and a single-shot default for state-changing calls.
set -euo pipefail

readonly_mode=false
if [[ ${1:-} == --readonly ]]; then readonly_mode=true; shift; fi
[[ ${1:-} == -- ]] || { printf 'Expected -- before SSH arguments.\n' >&2; exit 2; }
shift
[[ $# -gt 0 ]] || exit 2
ssh_args=("$@")
connection=()
target=
while [[ $# -gt 0 ]]; do
  case "$1" in
    -F|-i|-p|-l|-J|-o|-b)
      [[ $# -ge 2 ]] || exit 2
      connection+=("$1" "$2"); shift 2 ;;
    -T|-n|-q|-v|-vv|-vvv|-4|-6)
      connection+=("$1"); shift ;;
    -o*|-F*|-i*|-p*|-l*|-J*)
      connection+=("$1"); shift ;;
    -*) printf 'Unsupported SSH option in managed transport.\n' >&2; exit 2 ;;
    *) target=$1; break ;;
  esac
done
[[ "$target" =~ ^[A-Za-z0-9._@:-]+$ ]] || { printf 'Explicit SSH target required.\n' >&2; exit 2; }
ssh_bin=${NCCCU_SSH_EXECUTABLE:-/usr/bin/ssh}
state=${NCCCU_SSH_CONTROL_STATE:-/tmp/ncccu-ssh-control-$UID}
sockdir=${NCCCU_SSH_CONTROL_SOCKDIR:-/tmp}
[[ "$ssh_bin" == /* && -x "$ssh_bin" && "$state" == /* && "$sockdir" == /* && -d "$sockdir" ]] || exit 2
umask 077
mkdir -p "$state"
[[ ! -L "$state" ]] || exit 2
chmod 700 "$state"
exec 8>"$state/slot-0.lock" 9>"$state/slot-1.lock"
acquired=false
for ((attempt=0; attempt<600; attempt++)); do
  if flock -n 8; then acquired=true; break; fi
  if flock -n 9; then acquired=true; break; fi
  sleep .1
done
[[ "$acquired" == true ]] || { printf 'Timed out waiting for an SSH slot.\n' >&2; exit 75; }
exec 7>"$state/master.lock"
key=${target//[^[:alnum:]._-]/_}
pointer=$state/master-$key-v2.path
base=${NCCCU_SSH_RETRY_BASE_SECONDS:-1}
[[ "$base" =~ ^[0-9]+$ ]] || exit 2
tries=1
if [[ "$readonly_mode" == true ]]; then tries=3; fi
result=1
for ((attempt=1; attempt<=tries; attempt++)); do
  flock -w 20 7 || { printf 'Timed out waiting for the SSH master lock.\n' >&2; exit 75; }
  socket=
  if [[ -f "$pointer" && ! -L "$pointer" ]]; then IFS= read -r socket < "$pointer" || :; fi
  if [[ -n "$socket" ]] && timeout 10s "$ssh_bin" -S "$socket" -O check "${connection[@]}" "$target" \
       7>&- 8>&- 9>&- >/dev/null 2>&1; then
    :
  else
    socket=$sockdir/ncccu-ssh-$UID-$BASHPID-$RANDOM-%C
    if timeout 30s "$ssh_bin" -S "$socket" -o ControlMaster=yes -o ControlPersist=600 \
         -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -Nf "${connection[@]}" "$target" \
         7>&- 8>&- 9>&-; then
      printf '%s\n' "$socket" > "$pointer.tmp-$BASHPID"
      mv -f -- "$pointer.tmp-$BASHPID" "$pointer"
    else
      result=$?
      flock -u 7
      if (( attempt < tries )); then sleep "$base"; base=$((base*2)); continue; fi
      exit "$result"
    fi
  fi
  flock -u 7
  if "$ssh_bin" -S "$socket" -o ControlMaster=auto -o ControlPersist=600 \
       -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "${ssh_args[@]}" 7>&- 8>&- 9>&-; then
    exit 0
  else
    result=$?
  fi
  if (( attempt < tries )); then sleep "$base"; base=$((base*2)); fi
done
exit "$result"
