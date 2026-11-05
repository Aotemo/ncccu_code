"""Isolated fake-command tests for the detached, one-shot handoff shell."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest


SCRIPT = Path(os.environ.get("NCCCU_ASYNC_SCRIPT", Path(__file__).resolve().parents[1] / "tools/activation/async_handoff.sh"))


class AsyncHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ready = self.root / "ready"
        self.formal = self.root / "formal"
        self.source = self.root / "source"
        self.ready.mkdir()
        self.formal.mkdir()
        (self.ready / "WINDOW-CROSSBATCH-prelaunch.json").write_text("{}\n")
        (self.ready / "POD-LIVE.json").write_text("{}\n")
        self.fake = self.root / "fake-python"
        self.fake.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [[ $2 == */fake-guard.py ]]; then
  [[ -z ${CUDA_VISIBLE_DEVICES-unset} && -z ${HIP_VISIBLE_DEVICES-unset} && -z ${ROCR_VISIBLE_DEVICES-unset} ]]
  if [[ "$*" == *"/validate_launch_pod.py"* ]]; then
    exit "${FAKE_LIVE_RC:-0}"
  fi
  printf 'activation\\n' >> "$FAKE_MARKER"
  if [[ -n ${FAKE_ACTIVATION_WAIT_FILE-} ]]; then
    for _ in {1..100}; do
      [[ -f $FAKE_ACTIVATION_WAIT_FILE ]] && break
      sleep .05
    done
    [[ -f $FAKE_ACTIVATION_WAIT_FILE ]]
  fi
  exit "$FAKE_ACTIVATION_RC"
fi
[[ $2 == */launch_once.py ]]
[[ -z ${CUDA_VISIBLE_DEVICES-unset} && -z ${HIP_VISIBLE_DEVICES-unset} && -z ${ROCR_VISIBLE_DEVICES-unset} ]]
printf 'launch\\n' >> "$FAKE_MARKER"
exit "$FAKE_LAUNCH_RC"
""")
        self.fake.chmod(0o700)
        self.env = dict(os.environ, FAKE_MARKER=str(self.root / "calls"),
                        FAKE_ACTIVATION_RC="0", FAKE_LAUNCH_RC="0")
        self.args = ["bash", str(SCRIPT), "dispatch", str(self.ready), "r254-test",
                     str(self.formal), str(self.source), str(self.fake),
                     str(self.root / "fake-guard.py"), str(self.root / "allocation.json"),
                     "a" * 64, "owner", "ns", "pod", "uid", "container",
                     str(self.root / "policy.json"), "b" * 64, "c" * 64,
                     "test-authorization", ""]

    def dispatch(self):
        return subprocess.run(self.args, env=self.env, capture_output=True,
                              text=True, timeout=10, check=False)

    def result(self):
        path = self.ready / "ASYNC-HANDOFF" / "RESULT.json"
        deadline = time.monotonic() + 15
        while not path.is_file() and time.monotonic() < deadline:
            request = self.ready / "ASYNC-HANDOFF" / "LIVE-REQUEST.json"
            snapshot = self.ready / "POD-LIVE-LAUNCH.json"
            if request.is_file() and not snapshot.exists():
                snapshot.write_text("{}\n")
            time.sleep(.05)
        self.assertTrue(path.is_file(), "detached result did not arrive")
        return json.loads(path.read_text())

    def stage(self, name):
        return json.loads((self.ready / "ASYNC-HANDOFF" / (name + ".json")).read_text())

    def assert_stage(self, name, code):
        value = self.stage(name)
        self.assertEqual(value["exit_code"], code)
        self.assertLessEqual(value["started_at_epoch"], value["finished_at_epoch"])

    def test_success_is_detached_and_duplicate_is_rejected(self):
        release = self.root / "release"
        self.env["FAKE_ACTIVATION_WAIT_FILE"] = str(release)
        first = self.dispatch()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertFalse((self.ready / "ASYNC-HANDOFF" / "RESULT.json").exists())
        release.touch()
        self.assertEqual(self.result()["status"], "launch_exit0")
        self.assert_stage("ACTIVATION", 0)
        self.assert_stage("LAUNCH", 0)
        second = self.dispatch()
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual((self.root / "calls").read_text().splitlines(), ["activation", "launch"])

    def test_invalid_post_activation_snapshot_never_launches(self):
        self.env["FAKE_LIVE_RC"] = "21"
        self.assertEqual(self.dispatch().returncode, 0)
        self.assertEqual(self.result()["status"], "live_snapshot_invalid")
        self.assert_stage("LIVE-VALIDATION", 21)
        self.assertFalse((self.ready / "ASYNC-HANDOFF" / "LAUNCH.json").exists())
        self.assertEqual((self.root / "calls").read_text().splitlines(), ["activation"])

    def test_concurrent_claims_do_not_double_dispatch(self):
        first = subprocess.Popen(self.args, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        second = subprocess.Popen(self.args, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        first.communicate(timeout=10)
        second.communicate(timeout=10)
        self.assertEqual(sorted([first.returncode, second.returncode]), [0, 3])
        self.assertEqual(self.result()["status"], "launch_exit0")
        self.assertEqual((self.root / "calls").read_text().splitlines(), ["activation", "launch"])

    def test_activation_failure_never_launches(self):
        self.env["FAKE_ACTIVATION_RC"] = "17"
        self.assertEqual(self.dispatch().returncode, 0)
        self.assertEqual(self.result()["status"], "activation_failed")
        self.assert_stage("ACTIVATION", 17)
        self.assertFalse((self.ready / "ASYNC-HANDOFF" / "LAUNCH.json").exists())
        self.assertEqual((self.root / "calls").read_text().splitlines(), ["activation"])

    def test_launch_failure_is_recorded_once(self):
        self.env["FAKE_LAUNCH_RC"] = "19"
        self.assertEqual(self.dispatch().returncode, 0)
        self.assertEqual(self.result()["status"], "launch_failed")
        self.assert_stage("ACTIVATION", 0)
        self.assert_stage("LAUNCH", 19)
        self.assertEqual((self.root / "calls").read_text().splitlines(), ["activation", "launch"])

    def test_slow_activation_waits_for_new_post_activation_snapshot(self):
        old = self.ready / "POD-LIVE.json"
        old_time = time.time() - 105
        os.utime(old, (old_time, old_time))
        release = self.root / "release"
        self.env["FAKE_ACTIVATION_WAIT_FILE"] = str(release)
        self.assertEqual(self.dispatch().returncode, 0)
        release.touch()
        deadline = time.monotonic() + 10
        activation = self.ready / "ASYNC-HANDOFF" / "ACTIVATION.json"
        while not activation.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        self.assertTrue(activation.is_file())
        time.sleep(.3)
        self.assertEqual((self.root / "calls").read_text().splitlines(), ["activation"])
        self.assertFalse((self.ready / "ASYNC-HANDOFF" / "LAUNCH.json").exists())
        (self.ready / "POD-LIVE-LAUNCH.json").write_text("{}\n")
        self.assertEqual(self.result()["status"], "launch_exit0")
        self.assertEqual((self.root / "calls").read_text().splitlines(), ["activation", "launch"])

    def test_old_round_is_refused_before_claim(self):
        self.args[4] = "r253-test"
        self.assertNotEqual(self.dispatch().returncode, 0)
        self.assertFalse((self.ready / "ASYNC-HANDOFF").exists())

    def tearDown(self):
        self.temp.cleanup()


if __name__ == "__main__":
    unittest.main()
