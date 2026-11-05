#!/usr/bin/env bash
# Read current Kubernetes Pod metadata after the detached activation request.
set -euo pipefail
[[ $# -eq 6 ]] || { echo 'Expected SSH target, namespace, Pod, UID, container and ready directory' >&2; exit 2; }
target=$1; namespace=$2; pod=$3; uid=$4; container=$5; ready=$6
[[ -n "$target" && "$target" != -* && -n "$namespace" && -n "$pod" && -n "$uid" && -n "$container" ]] || exit 2
[[ "$ready" == /*/technical-prep/* && "$ready" != *..* ]] || exit 2
options=(-T -o BatchMode=yes -o ConnectTimeout=10)
[[ -z ${NCCCU_SSH_CONFIG:-} ]] || options+=(-F "$NCCCU_SSH_CONFIG")
remote='bash -s --'
for argument in "$namespace" "$pod" "$uid" "$container" "$ready"; do
  printf -v quoted ' %q' "$argument"; remote+="$quoted"
done
timeout -k 2s 50s ssh "${options[@]}" "$target" "$remote" <<'REMOTE'
set -euo pipefail
namespace=$1; pod=$2; uid=$3; container=$4; ready=$5
identity=$(timeout -k 1s 10s kubectl --request-timeout=8s get pod -n "$namespace" "$pod" -o 'go-template={{.metadata.uid}}|{{range .status.conditions}}{{if eq .type "Ready"}}{{.status}}{{end}}{{end}}|{{.status.phase}}|{{if .metadata.deletionTimestamp}}deleting{{end}}')
[[ "$identity" == "$uid|True|Running|" ]]
timeout -k 1s 12s kubectl exec -n "$namespace" "$pod" -c "$container" -- \
  env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= nice -n 19 taskset -c 0 \
  sh -c 'set -eu; test -s "$1/ASYNC-HANDOFF/LIVE-REQUEST.json"; test ! -e "$1/POD-LIVE-LAUNCH.json"; test ! -e "$1/ASYNC-HANDOFF/RESULT.json"' sh "$ready"
snapshot=$(mktemp)
trap 'rm -f "$snapshot"' EXIT
timeout -k 1s 10s kubectl --request-timeout=8s get pod -n "$namespace" "$pod" -o json > "$snapshot"
[[ -s "$snapshot" ]]
timeout -k 1s 20s kubectl exec -i -n "$namespace" "$pod" -c "$container" -- \
  env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= nice -n 19 taskset -c 0 \
  sh -c '
    set -eu
    umask 077
    test -s "$1/ASYNC-HANDOFF/LIVE-REQUEST.json"
    test ! -e "$1/POD-LIVE-LAUNCH.json"
    test ! -e "$1/ASYNC-HANDOFF/RESULT.json"
    temporary=$(mktemp "$1/.POD-LIVE-LAUNCH.XXXXXXXX")
    trap '\''rm -f "$temporary"'\'' EXIT
    cat > "$temporary"
    test -s "$temporary"
    ln "$temporary" "$1/POD-LIVE-LAUNCH.json"
  ' sh "$ready" < "$snapshot"
printf '%s\n' 'post-activation Pod snapshot delivered for independent validation'
REMOTE
