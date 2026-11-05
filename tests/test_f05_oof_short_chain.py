"""Fake-only tests of the OOF single-transport and Pod-side CPU-to-GPU chain."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SNAPSHOT_DIR = Path(__file__).resolve().parent
OUTER = Path(os.environ.get('NCCCU_F05_SHORT_OUTER', SNAPSHOT_DIR / 'f05-oof-short-handoff-v1.sh'))
POD_STAGE = Path(os.environ.get('NCCCU_F05_SHORT_POD', SNAPSHOT_DIR / 'f05-oof-short-pod-v1.sh'))
UID = 'fixture-pod-uid'
DEVICES = '0,1,2,3,4,5,6,7'


def executable(path, text):
    path.write_text(text)
    path.chmod(0o700)


class ShortChainTest(unittest.TestCase):
    def setUp(self):
        if not OUTER.is_file() or not POD_STAGE.is_file():
            raise RuntimeError('set NCCCU_F05_SHORT_OUTER and NCCCU_F05_SHORT_POD to exact isolated snapshots')
        self.temp = tempfile.TemporaryDirectory(prefix='f05-short-chain-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'; self.bin.mkdir()
        self.private = self.root / 'private'; self.private.mkdir()
        self.work = self.root / 'technical-prep'
        self.base = self.work / 'takeover-20261104-08ac'; self.base.mkdir(parents=True)
        helper = self.work / 'helper-fixture'; helper.mkdir()
        self.stage = helper / 'f05-oof-short-pod-v1.sh'
        shutil.copyfile(POD_STAGE, self.stage)
        self.stage_sha = hashlib.sha256(self.stage.read_bytes()).hexdigest()
        self.wrapper = self.root / 'short-handoff.sh'
        source = OUTER.read_text()
        old_private = '/home/hao/Desktop/challenge/work/takeover-20261104-08ac'
        old_work = '/public/home/acfwvv9ejn/SothisAI/exp-v3/technical-prep'
        if source.count('priv=' + old_private) != 1 or source.count(old_work) != 2:
            raise ValueError('snapshot does not match the declared transport fixture')
        self.wrapper.write_text(source.replace('priv=' + old_private, 'priv=' + str(self.private), 1)
                                .replace(old_work, str(self.work)))
        self.fake_python = self.bin / 'fake-python'
        self.trace = self.root / 'trace'
        env_text = ('NCCCU_SSH_CONFIG=/fixture/config\nNCCCU_SSH_TARGET=fixture-host\n'
            'NCCCU_POD_NAMESPACE=fixture-ns\nNCCCU_POD_NAME=fixture-pod\n'
            f'NCCCU_POD_UID={UID}\nNCCCU_CONTAINER=fixture-container\n'
            f'NCCCU_PYTHON={self.fake_python}\nNCCCU_CPU_GUARD=/fixture/guard.py\n'
            'NCCCU_CPU_ALLOCATION=/fixture/allocation.json\n'
            f'NCCCU_CPU_ALLOCATION_SHA256={"a" * 64}\nNCCCU_EXECUTOR_ID=fixture-owner\n'
            f'NCCCU_RESOURCE_POLICY=/fixture/policy.json\nNCCCU_RESOURCE_POLICY_SHA256={"b" * 64}\n')
        (self.private / 'pod-a-dual-owner-prep.env').write_text(env_text)
        executable(self.bin / 'ssh', '''#!/bin/bash
set -euo pipefail
printf 'ssh\\n' >> "$FAKE_TRACE"
for last in "$@"; do :; done
exec bash -c "$last"
''')
        executable(self.bin / 'kubectl', '''#!/bin/bash
set -euo pipefail
args=" $* "
if [[ "$args" == *" get pod "* ]]; then
  if [[ "$args" == *" -o json "* ]]; then
    printf 'snapshot_get\\n' >> "$FAKE_TRACE"
    printf '{"metadata":{"uid":"fixture-pod-uid"}}\\n'
  else
    printf 'identity\\n' >> "$FAKE_TRACE"
    count=$(grep -c '^identity$' "$FAKE_TRACE")
    if [[ "$FAKE_MODE" == uid_first || ( "$FAKE_MODE" == uid_second && "$count" -gt 1 ) ]]; then
      printf 'wrong-uid|True|Running'
    else
      printf 'fixture-pod-uid|True|Running'
    fi
  fi
elif [[ "$args" == *" exec -i "* ]]; then
  printf 'snapshot_write\\n' >> "$FAKE_TRACE"
  [[ "$FAKE_MODE" != snapshot_fail ]] || exit 31
  destination=${!#}
  [[ "$destination" == "$FAKE_BASE"/F05-OOF-LIVE-* && ! -e "$destination" ]] || exit 32
  (set -C; cat > "$destination")
elif [[ "$args" == *" exec -n "* ]]; then
  printf 'pod_exec\\n' >> "$FAKE_TRACE"
  while [[ $# -gt 0 && $1 != -- ]]; do shift; done
  [[ $# -gt 0 ]] || exit 91
  shift
  [[ "$*" == *"NCCCU_PYTHON=$FAKE_PYTHON"* && "$*" == *"$FAKE_STAGE"* ]] || exit 92
  exec "$@"
else
  exit 93
fi
''')
        executable(self.bin / 'env', '''#!/bin/bash
set -euo pipefail
dispatch=false; guard=false; uid=false; cuda=false; hip=false; rocr=false
for arg in "$@"; do
  [[ "$arg" != dispatch ]] || dispatch=true
  [[ "$arg" != "$FAKE_GUARD" ]] || guard=true
  [[ "$arg" != NCCCU_POD_UID=fixture-pod-uid ]] || uid=true
  [[ "$arg" != CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ]] || cuda=true
  [[ "$arg" != HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ]] || hip=true
  [[ "$arg" != ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ]] || rocr=true
done
if $dispatch; then
  $uid && $cuda && $hip && $rocr || exit 94
  printf 'gpu_env_explicit\\n' >> "$FAKE_TRACE"
fi
if $guard; then
  [[ " $* " == *" CUDA_VISIBLE_DEVICES= "* && " $* " == *" HIP_VISIBLE_DEVICES= "* \\
     && " $* " == *" ROCR_VISIBLE_DEVICES= "* && " $* " == *" NCCCU_POD_UID=fixture-pod-uid "* ]] || exit 95
  printf 'cpu_env_hidden\\n' >> "$FAKE_TRACE"
fi
exec /usr/bin/env "$@"
''')
        executable(self.fake_python, '''#!/bin/bash
set -euo pipefail
[[ "$1" == -B ]] || exit 96
shift
if [[ "$1" == "$FAKE_GUARD" ]]; then
  shift
  [[ "$1" == run ]] || exit 97
  shift
  [[ "$1" == --allocation && "$2" == /fixture/allocation.json && "$3" == --allocation-sha \\
     && "$5" == --owner && "$6" == fixture-owner ]] || exit 98
  shift 6
  script=$1; shift
  printf 'guard\\n' >> "$FAKE_TRACE"
  [[ "$FAKE_MODE" != guard_fail ]] || exit 17
  if [[ "$script" == */observe-compact-v2.py ]]; then
    printf 'observer\\n' >> "$FAKE_TRACE"
    [[ "$FAKE_MODE" != observer_fail ]] || exit 18
    if [[ "$FAKE_MODE" != safe_missing ]]; then (set -C; printf 'synthetic-safe\\n' > "$4"); fi
  elif [[ "$script" == */f04_gpu_full_dev.py && "$1" == prepare-eval-window ]]; then
    shift
    printf 'window\\n' >> "$FAKE_TRACE"
    [[ "$FAKE_MODE" != window_fail ]] || exit 19
    safe=''; safe_sha=''; output=''
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --safe) safe=$2;; --safe-sha256) safe_sha=$2;; --output) output=$2;;
      esac
      shift 2
    done
    sum=$(sha256sum "$safe"); [[ ${sum%% *} == "$safe_sha" ]] || exit 99
    [[ "$FAKE_MODE" == window_missing ]] || (set -C; printf 'synthetic-window\\n' > "$output")
  else
    exit 100
  fi
  exit 0
fi
[[ "$1" == -u && "$2" == */f04_gpu_full_dev.py && "$3" == dispatch ]] || exit 101
shift 3
[[ "${NCCCU_POD_UID:-}" == fixture-pod-uid \\
   && "${CUDA_VISIBLE_DEVICES:-}" == 0,1,2,3,4,5,6,7 \\
   && "${HIP_VISIBLE_DEVICES:-}" == 0,1,2,3,4,5,6,7 \\
   && "${ROCR_VISIBLE_DEVICES:-}" == 0,1,2,3,4,5,6,7 ]] || exit 102
safe=''; safe_sha=''; window=''; window_sha=''; claim=''; output=''; oof=false; standalone=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --oof-mode) oof=true; shift; continue;;
    --standalone-eval) standalone=true; shift; continue;;
    --safe) safe=$2;; --safe-sha256) safe_sha=$2;;
    --window) window=$2;; --window-sha256) window_sha=$2;;
    --claim-dir) claim=$2;; --output) output=$2;;
  esac
  shift 2
done
$oof && $standalone || exit 103
sum=$(sha256sum "$safe"); [[ ${sum%% *} == "$safe_sha" ]] || exit 104
sum=$(sha256sum "$window"); [[ ${sum%% *} == "$window_sha" ]] || exit 105
[[ "$claim" == "$FAKE_BASE"/F05-OOF-ATTEMPT-* && "$output" == "$claim/evaluation" ]] || exit 106
printf 'dispatch\\n' >> "$FAKE_TRACE"
mkdir "$claim"
case "$FAKE_MODE" in
  dispatch_fail) exit 20;;
  dispatch_unknown) exit 124;;
esac
printf 'synthetic_dispatch_only\\n'
''')
        source_root = f'/tmp/technical-prep/{UID}/source-fixture/source-git'
        self.safe = self.root / 'safe.json'
        self.window = self.root / 'window.json'
        self.live = self.base / 'F05-OOF-LIVE-fixture.json'
        self.claim = self.base / 'F05-OOF-ATTEMPT-fixture'
        self.args = ['a', str(self.stage), self.stage_sha, source_root,
            '/fixture/metadata', '/fixture/tree-proof', 'c' * 64, 'd' * 40, 'e' * 40,
            '/fixture/site', 'f' * 64, '/fixture/ready', '1' * 64,
            '/fixture/peer-site', '2' * 64, '/fixture/peer-obs', '3' * 64,
            '/fixture/previous-first', '/fixture/previous-root', 'R267', '/fixture/previous-source',
            str(self.safe), str(self.window), '/fixture/inputs', '4' * 64, str(self.live), str(self.claim)]
        self.env = dict(os.environ, PATH=str(self.bin) + ':' + os.environ['PATH'],
            FAKE_TRACE=str(self.trace), FAKE_BASE=str(self.base), FAKE_STAGE=str(self.stage),
            FAKE_PYTHON=str(self.fake_python), FAKE_GUARD='/fixture/guard.py')

    def run_chain(self, mode='normal', args=None):
        return subprocess.run(['bash', str(self.wrapper), *(self.args if args is None else args)],
                              env=dict(self.env, FAKE_MODE=mode), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, timeout=30, check=False)

    def events(self):
        return self.trace.read_text().splitlines() if self.trace.is_file() else []

    def assert_no_dispatch(self):
        self.assertNotIn('dispatch', self.events())
        self.assertNotIn('gpu_env_explicit', self.events())
        self.assertFalse(self.claim.exists())

    def test_uid_mismatch_before_and_after_snapshot_stops_the_chain(self):
        for mode, identities in (('uid_first', 1), ('uid_second', 2)):
            with self.subTest(mode=mode):
                self.trace.unlink(missing_ok=True); self.live.unlink(missing_ok=True)
                result = self.run_chain(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.events().count('ssh'), 1)
                self.assertEqual(self.events().count('identity'), identities)
                self.assert_no_dispatch()

    def test_snapshot_write_failure_stops_before_pod_stage(self):
        result = self.run_chain('snapshot_fail')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('snapshot_write', self.events())
        self.assertNotIn('pod_exec', self.events())
        self.assert_no_dispatch()

    def test_wrong_pod_stage_sha_stops_before_cpu_guard(self):
        args = list(self.args); args[2] = '0' * 64
        result = self.run_chain(args=args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('pod_exec', self.events())
        self.assertNotIn('guard', self.events())
        self.assert_no_dispatch()

    def test_guard_observer_window_failure_or_missing_output_never_dispatches(self):
        for mode in ('guard_fail', 'observer_fail', 'safe_missing', 'window_fail', 'window_missing'):
            with self.subTest(mode=mode):
                self.trace.unlink(missing_ok=True); self.live.unlink(missing_ok=True)
                self.safe.unlink(missing_ok=True); self.window.unlink(missing_ok=True)
                result = self.run_chain(mode)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.events().count('ssh'), 1)
                self.assert_no_dispatch()

    def test_success_binds_hidden_cpu_and_explicit_gpu_environment_once(self):
        result = self.run_chain()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.events(), ['ssh', 'identity', 'snapshot_get', 'snapshot_write', 'identity',
            'pod_exec', 'cpu_env_hidden', 'guard', 'observer', 'cpu_env_hidden', 'guard', 'window',
            'gpu_env_explicit', 'dispatch'])
        self.assertTrue(self.safe.is_file() and self.window.is_file() and self.claim.is_dir())

    def test_dispatch_failure_or_unknown_never_retries_or_changes_claim(self):
        for mode in ('dispatch_fail', 'dispatch_unknown'):
            with self.subTest(mode=mode):
                self.trace.unlink(missing_ok=True); self.live.unlink(missing_ok=True)
                self.safe.unlink(missing_ok=True); self.window.unlink(missing_ok=True)
                if self.claim.exists(): shutil.rmtree(self.claim)
                first = self.run_chain(mode)
                self.assertNotEqual(first.returncode, 0)
                self.assertEqual(self.events().count('ssh'), 1)
                self.assertEqual(self.events().count('dispatch'), 1)
                self.assertTrue(self.claim.is_dir())
                # Fresh auxiliary paths make the claimed attempt, rather than
                # the old snapshot file, the reason this second call stops.
                retry = list(self.args)
                retry[21] = str(self.root / 'retry-safe.json')
                retry[22] = str(self.root / 'retry-window.json')
                retry[25] = str(self.base / 'F05-OOF-LIVE-retry.json')
                second = self.run_chain(mode, args=retry)
                self.assertNotEqual(second.returncode, 0)
                self.assertEqual(self.events().count('pod_exec'), 2)
                self.assertEqual(self.events().count('guard'), 2)
                self.assertEqual(self.events().count('dispatch'), 1)
                (self.base / 'F05-OOF-LIVE-retry.json').unlink(missing_ok=True)


if __name__ == '__main__':
    print('SOURCE_SHA256 outer', hashlib.sha256(OUTER.read_bytes()).hexdigest(), flush=True)
    print('SOURCE_SHA256 pod_stage', hashlib.sha256(POD_STAGE.read_bytes()).hexdigest(), flush=True)
    print('SOURCE_SHA256 test', hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
    unittest.main(verbosity=2)
