#!/usr/bin/env bash
# Transport-independent, one-shot control of existing activation and launch.
set -euo pipefail
umask 077

[[ $# -eq 19 ]] || { echo 'Expected mode plus 18 bound arguments' >&2; exit 2; }
mode=$1; shift
args=("$@")
ready=$1; round=$2; formal=$3; source_root=$4; python=$5; guard=$6; allocation=$7
allocation_sha=$8; executor=$9; shift 9
namespace=$1; pod=$2; uid=$3; container=$4; policy=$5; policy_sha=$6
site_sha=$7; authorization=$8; devices=$9

[[ "$round" =~ ^r([0-9]+)-[A-Za-z0-9-]+$ ]] && (( 10#${BASH_REMATCH[1]} >= 254 )) || exit 2
[[ "$ready" == /* && "$formal" == /* && "$source_root" == /* && "$python" == /* && "$guard" == /* && "$allocation" == /* ]] || exit 2
[[ "$allocation_sha" =~ ^[0-9a-f]{64}$ && "$policy_sha" =~ ^[0-9a-f]{64}$ && "$site_sha" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ -n "$executor" && -n "$namespace" && -n "$pod" && -n "$uid" && -n "$container" && -n "$authorization" ]] || exit 2
[[ -z "$devices" || "$devices" =~ ^[0-9]+(,[0-9]+)*$ ]] || exit 2
[[ -s "$ready/WINDOW-CROSSBATCH-prelaunch.json" && -s "$ready/POD-LIVE.json" ]] || exit 2
dir=$ready/ASYNC-HANDOFF

if [[ "$mode" == dispatch ]]; then
  # The claim is never removed or recycled, even after a transport failure.
  mkdir -m 700 "$dir" || exit 3
  printf '{"round_id":"%s","claimed_at_epoch":%s}\n' "$round" "$(date -u +%s)" > "$dir/CLAIM.json"
  nohup setsid bash "$0" _run "${args[@]}" </dev/null > "$dir/CONTROL-private.log" 2>&1 &
  printf 'detached handoff claimed\n'
  exit 0
fi
[[ "$mode" == _run && -d "$dir" ]] || exit 2
mkdir -m 700 "$dir/START" || exit 3
printf '{"round_id":"%s","started_at_epoch":%s,"pid":%s}\n' "$round" "$(date -u +%s)" "$$" > "$dir/STARTED.json"

write_stage() {
  local stage=$1 started=$2 finished=$3 code=$4
  printf '{"round_id":"%s","stage":"%s","started_at_epoch":%s,"finished_at_epoch":%s,"exit_code":%s}\n' \
    "$round" "$stage" "$started" "$finished" "$code" > "$dir/.$stage.$$.tmp"
  mv "$dir/.$stage.$$.tmp" "$dir/$stage.json"
}
write_result() {
  local status=$1 code=$2
  printf '{"round_id":"%s","status":"%s","exit_code":%s,"finished_at_epoch":%s}\n' \
    "$round" "$status" "$code" "$(date -u +%s)" > "$dir/.RESULT.$$.tmp"
  mv "$dir/.RESULT.$$.tmp" "$dir/RESULT.json"
}

started=$(date -u +%s)
set +e
env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  NCCCU_POD_NAMESPACE="$namespace" NCCCU_POD_NAME="$pod" NCCCU_POD_UID="$uid" \
  NCCCU_CONTAINER="$container" NCCCU_PYTHON="$python" NCCCU_RESOURCE_POLICY="$policy" \
  NCCCU_RESOURCE_POLICY_SHA256="$policy_sha" \
  nice -n 19 "$python" -B "$guard" run --allocation "$allocation" \
  --allocation-sha "$allocation_sha" --owner "$executor" \
  "$source_root/tools/activation/preflight_activate_once.py" --work-root "$ready" \
  --round-id "$round" --authorization-reference "$authorization" > "$dir/ACTIVATION-private.log" 2>&1
activation_rc=$?
set -e
write_stage ACTIVATION "$started" "$(date -u +%s)" "$activation_rc"
if (( activation_rc != 0 )); then
  write_result activation_failed "$activation_rc"
  exit "$activation_rc"
fi

started=$(date -u +%s)
set +e
observed=$(stat -c %Y "$ready/POD-LIVE.json")
stat_rc=$?
if (( stat_rc == 0 )); then
  ( set -C
    env CUDA_VISIBLE_DEVICES="$devices" HIP_VISIBLE_DEVICES="$devices" ROCR_VISIBLE_DEVICES="$devices" \
      "$python" -B "$formal/launch_once.py" --site-binding "$formal/SITE_BINDING.json" \
      --site-binding-sha256 "$site_sha" --pod-uid "$uid" --ready True --observed-at "$observed" \
      > "$ready/GPU-ENTRY-private.log" 2>&1
  )
  launch_rc=$?
else
  launch_rc=$stat_rc
fi
set -e
write_stage LAUNCH "$started" "$(date -u +%s)" "$launch_rc"
if (( launch_rc != 0 )); then
  write_result launch_failed "$launch_rc"
  exit "$launch_rc"
fi
write_result launch_exit0 0
