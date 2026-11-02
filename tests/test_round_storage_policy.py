"""Explicit new Base rounds receive headroom; defaults and free floor stay bound."""
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
R117 = 'r117-swinbase22k-seed20261029-scale085-pairlog-24k-p4-9010-20261031'
R174 = 'r174-baseclip-seed20261029-scale085-pairlog-storage16-24k-p4-9010-20261102'
R175 = 'r175-baseclip-seed20261027-scale085-pairlog-storage16-24k-p4-9010-20261102'
R180 = 'r180-large22k-seed20261029-scale085-pairlog-storage32-24k-p4-9010-20261102'
R181 = 'r181-large22k-seed20261027-scale085-pairlog-storage32-24k-p4-9010-20261102'
R182 = 'r182-large-dinov3-seed20261029-scale085-pairlog-storage32-24k-p4-9010-20261102'
R183 = 'r183-large-dinov3-seed20261027-scale085-pairlog-storage32-24k-p4-9010-20261102'
R184 = 'r184-vits-dinov3-seed20261029-scale085-pairlog-24k-p4-9010-20261102'
GIB = 1024 ** 3


class RoundStoragePolicyTest(unittest.TestCase):
    def test_large_exact_round_32GiB_keeps_free_floor_and_small_default(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = ROOT / 'configs/ROUND-STORAGE-POLICY.json'
            for name, limit in ((R180, 32), (R181,32), (R182,32), (R183,32), (R184,12), ('r178-small-seed20261029-scale085-pairlog-bce-mix05-24k-p4-9010-20261102', 12)):
                root = Path(directory) / name
                root.mkdir()
                (root / 'STORAGE_POLICY.json').write_bytes(policy.read_bytes())
                usage = type('Usage', (), {'free': 40 * GIB})()
                with patch.object(guard, 'ROOT', root), patch.object(guard, 'tree_bytes', return_value=(limit - 3) * GIB), \
                        patch.object(guard.shutil, 'disk_usage', return_value=usage):
                    self.assertEqual(guard.round_storage_limit(), limit * GIB)
                    self.assertEqual(guard.budget(guard.time.time() + 10)['new_round_bytes'], (limit - 3) * GIB)
                    with patch.object(guard, 'tree_bytes', return_value=(limit + 1) * GIB), self.assertRaises(RuntimeError):
                        guard.budget(guard.time.time() + 10)
                    usage.free = 15 * GIB
                    with self.assertRaises(RuntimeError):
                        guard.budget(guard.time.time() + 10)

    def test_new_round_limit_and_old_default_keep_free_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = ROOT / 'configs/ROUND-STORAGE-POLICY.json'
            for name in (R116, R117, R174, R175):
                with self.subTest(round_id=name):
                    root = Path(directory) / name
                    root.mkdir()
                    (root / 'STORAGE_POLICY.json').write_bytes(policy.read_bytes())
                    usage = type('Usage', (), {'free': 20 * GIB})()
                    with patch.object(guard, 'ROOT', root), patch.object(guard, 'tree_bytes', return_value=13 * GIB), \
                            patch.object(guard.shutil, 'disk_usage', return_value=usage):
                        self.assertEqual(guard.round_storage_limit(), 16 * GIB)
                        self.assertEqual(guard.budget(guard.time.time() + 10)['new_round_bytes'], 13 * GIB)
                        with patch.object(guard, 'tree_bytes', return_value=17 * GIB), self.assertRaises(RuntimeError):
                            guard.budget(guard.time.time() + 10)
                        usage.free = 15 * GIB
                        with self.assertRaises(RuntimeError):
                            guard.budget(guard.time.time() + 10)
            for name in ('older-round', 'r170-baseclip-seed20261029-scale085-pairlog-24k-p4-9010-20261102',
                         'r172-small22k-seed20261029-scale085-pairlog-corner288max05-24k-p4-9010-20261102',
                         'r176-small-dinov3-seed20261029-scale085-pairlog-24k-p4-9010-20261102'):
                old = Path(directory) / name
                old.mkdir()
                (old / 'STORAGE_POLICY.json').write_bytes(policy.read_bytes())
                with patch.object(guard, 'ROOT', old), patch.object(guard, 'tree_bytes', return_value=13 * GIB), \
                        patch.object(guard.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 20 * GIB})()):
                    self.assertEqual(guard.round_storage_limit(), 12 * GIB)
                    with self.assertRaises(RuntimeError):
                        guard.budget(guard.time.time() + 10)


if __name__ == '__main__':
    unittest.main()
