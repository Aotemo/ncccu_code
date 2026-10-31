"""The R124R1 transport handoff is a single, continuous CPU-side call chain."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/activation/preflight_activate_once.py'
SPEC = importlib.util.spec_from_file_location('r124r1_chain', SOURCE)
chain = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chain)


class PreflightActivationOnceTest(unittest.TestCase):
    def test_one_fresh_receipt_flows_directly_to_one_audit_without_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            receipt = root / 'SITE-PREFLIGHT.json'
            clock = root / 'FORMAL-CLOCK.json'
            observed_at = 1000.25
            first = ['preflight', '--observed-at', str(observed_at)]
            second = ['audit', '--site-preflight', str(receipt),
                      '--observed-at', str(observed_at)]
            calls = []

            def launch(stage, command):
                calls.append((stage, list(command)))
                if stage == 'preflight':
                    receipt.write_text(json.dumps({'exit_code': 0,
                        'observed_at': observed_at, 'round_id': chain.ROUND}) + '\n')
                else:
                    self.assertEqual(stage, 'audit')
                    self.assertEqual(command[-2:], ['--site-preflight-sha256',
                        hashlib.sha256(receipt.read_bytes()).hexdigest()])
                    self.assertFalse(clock.exists())
                return 0

            digest = chain.execute_sequence(first, second, receipt, observed_at,
                                            launch, now=lambda: observed_at + 1)
            self.assertEqual([name for name, _ in calls], ['preflight', 'audit'])
            self.assertEqual(calls[0][1], first)
            self.assertEqual(calls[1][1][:-2], second)
            self.assertEqual(digest, hashlib.sha256(receipt.read_bytes()).hexdigest())
            self.assertFalse(clock.exists())


if __name__ == '__main__':
    unittest.main()
