#!/usr/bin/env bash
# Local/login-host transport only. Every Python process runs inside the Pod.
set -euo pipefail

usage() {
  cat >&2 <<'HELP'
Usage:
  pod_cpu.sh discover SSH_TARGET [NAMESPACE]
  pod_cpu.sh interpreters SSH_TARGET NAMESPACE POD EXPECTED_UID
  pod_cpu.sh probe SSH_TARGET NAMESPACE POD EXPECTED_UID PYTHON
  pod_cpu.sh run SSH_TARGET NAMESPACE POD EXPECTED_UID PYTHON SCRIPT [ARG...]

SSH configuration: NCCCU_SSH_CONFIG, NCCCU_SSH_IDENTITY, NCCCU_SSH_PORT.
For a multi-container Pod, set NCCCU_CONTAINER to a discovered container name.
No host, namespace, Pod UID, Python path or project path is supplied by default.
HELP
  exit 2
}

[[ $# -ge 2 ]] || usage
operation=$1
target=$2
shift 2
case "$operation" in
  discover) [[ $# -le 1 ]] || usage ;;
  interpreters) [[ $# -eq 3 ]] || usage ;;
  probe) [[ $# -eq 4 ]] || usage ;;
  run) [[ $# -ge 5 ]] || usage ;;
  *) usage ;;
esac
[[ -n "$target" && "$target" != -* ]] || usage
ssh_options=(-T -o BatchMode=yes -o ConnectTimeout=15 -o ConnectionAttempts=1)
[[ -z ${NCCCU_SSH_CONFIG:-} ]] || ssh_options+=(-F "$NCCCU_SSH_CONFIG")
[[ -z ${NCCCU_SSH_IDENTITY:-} ]] || ssh_options+=(-i "$NCCCU_SSH_IDENTITY" -o IdentitiesOnly=yes)
[[ -z ${NCCCU_SSH_PORT:-} ]] || ssh_options+=(-p "$NCCCU_SSH_PORT")
remote_command='bash -s --'
for argument in "$operation" "${NCCCU_CONTAINER:-}" "$@"; do
  printf -v quoted '%q' "$argument"
  remote_command+=" $quoted"
done

exec ssh "${ssh_options[@]}" "$target" "$remote_command" <<'REMOTE'
set -euo pipefail
operation=$1
container=$2
shift 2

if [[ "$operation" == discover ]]; then
  namespace_args=()
  [[ $# -eq 0 ]] || namespace_args=(-n "$1")
  kubectl config current-context >&2
  exec kubectl get pods "${namespace_args[@]}" -o 'go-template={{range .items}}{{.metadata.namespace}}{{"\t"}}{{.metadata.name}}{{"\t"}}{{.metadata.uid}}{{"\t"}}{{.status.phase}}{{"\t"}}{{range .status.conditions}}{{if eq .type "Ready"}}{{.status}}{{end}}{{end}}{{"\t"}}{{range .spec.containers}}{{.name}}{{" "}}{{end}}{{"\n"}}{{end}}'
fi

namespace=$1
pod=$2
expected_uid=$3
shift 3
[[ -n "$namespace" && -n "$pod" && -n "$expected_uid" ]] || exit 2
identity=$(kubectl get pod -n "$namespace" "$pod" -o 'go-template={{.metadata.uid}}|{{range .status.conditions}}{{if eq .type "Ready"}}{{.status}}{{end}}{{end}}|{{.status.phase}}|{{if .metadata.deletionTimestamp}}deleting{{end}}')
if [[ "$identity" != "$expected_uid|True|Running|" ]]; then
  printf 'Pod identity/readiness differs: %s; execution stopped.\n' "$identity" >&2
  exit 3
fi
containers=$(kubectl get pod -n "$namespace" "$pod" -o 'go-template={{range .spec.containers}}{{.name}}{{"\n"}}{{end}}')
if [[ -z "$container" ]]; then
  [[ "$containers" != *$'\n'* && -n "$containers" ]] || {
    printf 'Multiple containers; select an observed name with NCCCU_CONTAINER.\n' >&2
    exit 3
  }
  container=$containers
else
  found=false
  while IFS= read -r name; do
    [[ "$name" != "$container" ]] || found=true
  done <<< "$containers"
  [[ "$found" == true ]] || exit 3
fi
pod_exec=(kubectl exec -n "$namespace" "$pod" -c "$container" --)

if [[ "$operation" == interpreters ]]; then
  exec "${pod_exec[@]}" sh -c '
    seen="|"
    for candidate in "$(command -v python3 || :)" "$(command -v python || :)" /usr/local/python*/bin/python3 /usr/local/python*/bin/python3.[0-9]*; do
      case "$candidate" in ""|*-config) continue;; esac
      [ -x "$candidate" ] || continue
      actual=$(readlink -f "$candidate")
      case "$seen" in *"|$actual|"*) continue;; esac
      printf "%s\n" "$actual"
      seen="$seen$actual|"
    done
  '
fi

python=$1
shift
[[ "$python" == /* ]] || { printf 'An observed absolute Python path is required.\n' >&2; exit 2; }
cpu_env=(env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
  NCCCU_POD_NAMESPACE="$namespace" NCCCU_POD_NAME="$pod" NCCCU_POD_UID="$expected_uid"
  NCCCU_CONTAINER="$container" NCCCU_PYTHON="$python")

if [[ "$operation" == probe ]]; then
  exec "${pod_exec[@]}" "${cpu_env[@]}" nice -n 19 "$python" -B -c '
import importlib.metadata as metadata
import json, os, socket, sys
assert socket.gethostname() == os.environ["NCCCU_POD_NAME"], "Pod hostname differs"
assert os.path.realpath(sys.executable) == os.path.realpath(os.environ["NCCCU_PYTHON"]), "Python differs"
assert os.getpriority(os.PRIO_PROCESS, 0) == 19, "CPU priority differs"
versions = {name: metadata.version(name) for name in ("torch", "torchvision", "numpy", "Pillow")}
import torch
assert not torch.cuda.is_initialized(), "GPU already initialized"
print(json.dumps({"scope": "compute_pod_cpu", "namespace": os.environ["NCCCU_POD_NAMESPACE"],
    "pod": socket.gethostname(), "uid": os.environ["NCCCU_POD_UID"], "container": os.environ["NCCCU_CONTAINER"],
    "python": sys.executable, "python_version": sys.version.split()[0], "dependencies": versions,
    "gpu_initialized": torch.cuda.is_initialized(), "nice": os.getpriority(os.PRIO_PROCESS, 0)}, indent=2))
'
fi

[[ "$1" == /* && "$1" == *.py ]] || { printf 'Deploy the script to an absolute server path first.\n' >&2; exit 2; }
exec "${pod_exec[@]}" "${cpu_env[@]}" nice -n 19 "$python" -B -c '
import importlib.metadata as metadata
import os, socket, sys
assert socket.gethostname() == os.environ["NCCCU_POD_NAME"], "Pod hostname differs"
assert os.path.realpath(sys.executable) == os.path.realpath(os.environ["NCCCU_PYTHON"]), "Python differs"
assert os.getpriority(os.PRIO_PROCESS, 0) == 19, "CPU priority differs"
for name in ("torch", "torchvision", "numpy", "Pillow"):
    metadata.version(name)
os.execv(sys.executable, [sys.executable, "-B", *sys.argv[1:]])
' "$@"
REMOTE
