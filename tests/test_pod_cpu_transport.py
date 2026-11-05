"""CPU transport permits one idle owner only with complete explicit references."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/environment/pod_cpu.sh'


class CPUTransportTest(unittest.TestCase):
    def test_one_and_two_owners_and_incomplete_idle_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); trace = root / 'trace'
            ssh = root / 'ssh'
            ssh.write_text('#!/bin/bash\nset -euo pipefail\nfor last in "$@"; do :; done\nexec bash -c "$last"\n')
            ssh.chmod(0o755)
            kubectl = root / 'kubectl'
            kubectl.write_text('''#!/bin/bash
set -euo pipefail
args=" $* "
if [[ "$args" == *" get pod "* && "$args" == *".metadata.uid"* ]]; then
  printf 'fixture-uid|True|Running|'
elif [[ "$args" == *" get pod "* && "$args" == *".spec.containers"* ]]; then
  printf 'fixture-container\\n'
elif [[ "$args" == *" exec "* ]]; then
  printf '%s\\n' "$*" > "$TRACE_FILE"
else
  exit 91
fi
''')
            kubectl.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + ':' + os.environ['PATH'], TRACE_FILE=str(trace),
                       NCCCU_CONTAINER='fixture-container')
            base = ['bash', str(SCRIPT), 'budget', 'fixture-host', 'fixture-ns', 'fixture-pod',
                    'fixture-uid', '/fixture/python', '/fixture/cpu_budget.py', '/fixture/technical-prep/a.json',
                    '60', 'owner-a']
            run = subprocess.run(base, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(trace.read_text().count('--owner'), 1)
            trace.unlink()
            run = subprocess.run(base + ['owner-b'], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(trace.read_text().count('--owner'), 2)
            trace.unlink()
            incomplete = dict(env, NCCCU_CPU_IDLE_SITE='/fixture/site')
            run = subprocess.run(base, env=incomplete, capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 2)
            self.assertFalse(trace.exists())
            idle = dict(incomplete, NCCCU_CPU_IDLE_SITE_SHA256='a' * 64,
                        NCCCU_CPU_IDLE_SAFETY='/fixture/safe', NCCCU_CPU_IDLE_SAFETY_SHA256='b' * 64)
            run = subprocess.run(base, env=idle, capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('NCCCU_CPU_IDLE_SAFETY_SHA256=' + 'b' * 64, trace.read_text())


if __name__ == '__main__':
    unittest.main()
