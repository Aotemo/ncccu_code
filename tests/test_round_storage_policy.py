"""Only the frozen R116 round receives the larger monitor storage ceiling."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
SPEC = importlib.util.spec_from_file_location('round_storage_guard', ROOT / 'runtime/run_round.py')
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)
R116 = 'r116-base22k-seed20261029-scale085-pairlog-24k-p4-9010-20261030'
GIB = 1024 ** 3


class RoundStoragePolicyTest(unittest.TestCase):
    def test_new_round_limit_and_old_default_keep_free_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / R116
            root.mkdir()
            policy = ROOT / 'configs/ROUND-STORAGE-POLICY.json'
            (root / 'STORAGE_POLICY.json').write_bytes(policy.read_bytes())
            usage = type('Usage', (), {'free': 20 * GIB})()
            with patch.object(guard, 'ROOT', root), patch.object(guard, 'tree_bytes', return_value=13 * GIB), \
                    patch.object(guard.shutil, 'disk_usage', return_value=usage):
                self.assertEqual(guard.budget(guard.time.time() + 10)['new_round_bytes'], 13 * GIB)
                with patch.object(guard, 'tree_bytes', return_value=17 * GIB), self.assertRaises(RuntimeError):
                    guard.budget(guard.time.time() + 10)
                usage.free = 15 * GIB
                with self.assertRaises(RuntimeError):
                    guard.budget(guard.time.time() + 10)
            old = Path(directory) / 'older-round'
            old.mkdir()
            with patch.object(guard, 'ROOT', old), patch.object(guard, 'tree_bytes', return_value=13 * GIB), \
                    patch.object(guard.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 20 * GIB})()):
                with self.assertRaises(RuntimeError):
                    guard.budget(guard.time.time() + 10)


if __name__ == '__main__':
    unittest.main()
