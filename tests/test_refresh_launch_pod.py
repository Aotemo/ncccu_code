"""Transport of the new launch snapshot is atomic and refuses missing requests."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

root = Path(__file__).resolve().parents[1]
script = root / 'tools/environment/refresh_launch_pod.sh'


class RefreshLaunchPodTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ready = self.root / 'technical-prep/ready'; self.ready.mkdir(parents=True)
        self.bin = self.root / 'bin'; self.bin.mkdir()
        self.pod = self.root / 'pod.json'; self.pod.write_text('{"example":"fresh"}\n')
        self.command('ssh', '#!/usr/bin/env bash\nexec bash -c "${!#}"\n')
        self.command('taskset', '#!/usr/bin/env bash\nshift 2\nexec "$@"\n')
        self.command('kubectl', """#!/usr/bin/env bash
set -euo pipefail
while [[ $# -gt 0 && $1 != get && $1 != exec ]]; do shift; done
mode=$1; shift
if [[ $mode == get ]]; then
  [[ ${FAKE_GET_FAIL:-0} == 0 ]] || exit 17
  if [[ "$*" == *go-template* ]]; then printf 'uid|True|Running|'; else cat "$FAKE_POD"; fi
else
  while [[ $# -gt 0 && $1 != -- ]]; do shift; done
  [[ $# -gt 0 ]]; shift
  exec "$@"
fi
""")
        self.env = dict(os.environ, PATH=str(self.bin) + ':' + os.environ['PATH'], FAKE_POD=str(self.pod),
                        NCCCU_SSH_CONFIG=str(self.root / 'ssh.conf'))
        (self.root / 'ssh.conf').write_text('')
        self.args = ['bash', str(script), 'host', 'ns', 'pod', 'uid', 'container', str(self.ready)]

    def command(self, name, body):
        path = self.bin / name; path.write_text(body); path.chmod(0o755)

    def run_script(self):
        return subprocess.run(self.args, env=self.env, capture_output=True, text=True, timeout=15)

    def request(self):
        directory = self.ready / 'ASYNC-HANDOFF'; directory.mkdir()
        (directory / 'LIVE-REQUEST.json').write_text('{"round_id":"test"}\n')

    def test_requires_request_before_creating_a_snapshot(self):
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertFalse((self.ready / 'POD-LIVE-LAUNCH.json').exists())

    def test_snapshot_is_published_once_without_overwrite(self):
        self.request()
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        target = self.ready / 'POD-LIVE-LAUNCH.json'
        self.assertEqual(target.read_bytes(), self.pod.read_bytes())
        self.pod.write_text('{"example":"changed"}\n')
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertNotEqual(target.read_bytes(), self.pod.read_bytes())

    def test_api_failure_never_publishes_a_snapshot(self):
        self.request(); self.env['FAKE_GET_FAIL'] = '1'
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertFalse((self.ready / 'POD-LIVE-LAUNCH.json').exists())


if __name__ == '__main__':
    unittest.main()
