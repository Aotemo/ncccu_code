"""Isolated fake-SSH checks; never connects to a host."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/environment/ssh_control.sh'


class SSHControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fake = self.root / 'fake-ssh'
        fake.write_text('''#!/usr/bin/env bash
set -euo pipefail
for argument in "$@"; do
  if [[ "$argument" == check ]]; then test -f "$FAKE_STATE/master"; exit; fi
  if [[ "$argument" == -Nf ]]; then printf 'master\\n' >> "$FAKE_STATE/master_calls"; touch "$FAKE_STATE/master"; exit; fi
done
exec 9>"$FAKE_STATE/counter.lock"
flock 9
active=$(cat "$FAKE_STATE/active" 2>/dev/null || printf 0)
active=$((active+1)); printf '%s' "$active" > "$FAKE_STATE/active"
maximum=$(cat "$FAKE_STATE/max" 2>/dev/null || printf 0)
if (( active > maximum )); then printf '%s' "$active" > "$FAKE_STATE/max"; fi
printf 'call\\n' >> "$FAKE_STATE/calls"
flock -u 9
sleep "${FAKE_DELAY:-0}"
flock 9
active=$(cat "$FAKE_STATE/active"); printf '%s' "$((active-1))" > "$FAKE_STATE/active"
flock -u 9
exit "${FAKE_STATUS:-0}"
''')
        fake.chmod(0o755)
        self.env = dict(os.environ, NCCCU_SSH_EXECUTABLE=str(fake),
                        NCCCU_SSH_CONTROL_STATE=str(self.root / 'state'),
                        NCCCU_SSH_CONTROL_SOCKDIR=str(self.root), FAKE_STATE=str(self.root),
                        NCCCU_SSH_RETRY_BASE_SECONDS='0')

    def tearDown(self):
        self.tmp.cleanup()

    def invoke(self, *args, env=None):
        return subprocess.run(['bash', str(SCRIPT), *args, '--', '-T', '-F', '/fixture/conf',
                               'fixture-host', 'remote-command'], env=env or self.env,
                              capture_output=True, text=True, timeout=20)

    def test_non_idempotent_calls_are_never_retried(self):
        env = dict(self.env, FAKE_STATUS='8')
        result = self.invoke(env=env)
        self.assertEqual(result.returncode, 8, result.stderr)
        self.assertEqual((self.root / 'calls').read_text().splitlines(), ['call'])

    def test_only_explicit_readonly_calls_retry(self):
        env = dict(self.env, FAKE_STATUS='8')
        result = self.invoke('--readonly', env=env)
        self.assertEqual(result.returncode, 8)
        self.assertEqual(len((self.root / 'calls').read_text().splitlines()), 3)

    def test_shared_slots_cap_parallel_calls_at_two(self):
        env = dict(self.env, FAKE_DELAY='.4')
        processes = [subprocess.Popen(['bash', str(SCRIPT), '--', 'fixture-host', 'read'], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(5)]
        for process in processes:
            _, err = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, err)
        self.assertEqual((self.root / 'max').read_text(), '2')

    def test_stale_master_is_replaced_without_replaying_previous_command(self):
        self.assertEqual(self.invoke().returncode, 0)
        (self.root / 'master').unlink()
        self.assertEqual(self.invoke().returncode, 0)
        self.assertEqual(len((self.root / 'master_calls').read_text().splitlines()), 2)
        self.assertEqual(len((self.root / 'calls').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
