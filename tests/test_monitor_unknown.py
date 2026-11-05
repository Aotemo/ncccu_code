"""Narrow monitor-observation fixtures; parent A runs them in the guarded Pod."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'runtime'))
sys.path.insert(0, str(REPO / 'runtime/bundle'))

import run_round as rr


def phantom_walk(names, directory_error=None):
    def walk(top, followlinks=False, onerror=None):
        if directory_error is not None:
            if onerror is not None:
                onerror(directory_error)
            return
        yield (str(top), [], list(names))
    return walk


class MonitorUnknownTest(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self._original_root = rr.ROOT
        rr.ROOT = self.root

    def tearDown(self):
        rr.ROOT = self._original_root
        self._temporary.cleanup()

    def test_checkpoint_temp_disappearance_is_unknown(self):
        with mock.patch.object(rr.os, 'walk', phantom_walk(['.checkpoint-ab12cd34.tmp'])):
            self.assertEqual(rr.tree_bytes(), 0)
        self.assertEqual(rr.SCAN_DIAGNOSTICS['tolerated_atomic_leaf_disappearances'], 1)
        self.assertEqual(rr.SCAN_DIAGNOSTICS['sample_relative_paths'][0]['stage'], 'lstat')
        self.assertIn('.checkpoint-ab12cd34.tmp', rr.SCAN_DIAGNOSTICS['sample_relative_paths'][0]['path'])

    def test_pruned_checkpoint_leaves_are_unknown(self):
        names = ['checkpoint-' + 'a' * 32 + '.pt', 'checkpoint-' + 'a' * 32 + '.pt.json']
        with mock.patch.object(rr.os, 'walk', phantom_walk(names)):
            self.assertEqual(rr.tree_bytes(), 0)
        self.assertEqual(rr.SCAN_DIAGNOSTICS['tolerated_atomic_leaf_disappearances'], 2)

    def test_stable_missing_leaf_still_raises(self):
        with mock.patch.object(rr.os, 'walk', phantom_walk(['TRAIN_CONFIG.json'])):
            with self.assertRaises(FileNotFoundError):
                rr.tree_bytes()

    def test_vanished_directory_entry_is_unknown(self):
        def walk(top, followlinks=False, onerror=None):
            yield (str(top), ['initializing-5a5e3e05'], [])
        with mock.patch.object(rr.os, 'walk', walk):
            self.assertEqual(rr.tree_bytes(), 0)
        self.assertEqual(rr.SCAN_DIAGNOSTICS['tolerated_atomic_directory_disappearances'], 1)
        self.assertEqual(rr.SCAN_DIAGNOSTICS['sample_relative_paths'][0]['stage'], 'lstat')

    def test_walk_error_from_vanished_directory_is_unknown(self):
        error = FileNotFoundError(2, 'No such file or directory', str(self.root / 'control'))
        with mock.patch.object(rr.os, 'walk', phantom_walk([], directory_error=error)):
            self.assertEqual(rr.tree_bytes(), 0)
        self.assertEqual(rr.SCAN_DIAGNOSTICS['tolerated_atomic_directory_disappearances'], 1)

    def test_stable_walk_error_still_raises(self):
        with mock.patch.object(rr.os, 'walk', phantom_walk([], directory_error=PermissionError(13, 'denied'))):
            with self.assertRaises(PermissionError):
                rr.tree_bytes()

    def test_missing_round_root_still_raises(self):
        rr.ROOT = self.root / 'absent'
        with self.assertRaises(FileNotFoundError):
            rr.tree_bytes()

    def test_budget_rejects_real_disk_shortfall(self):
        usage = mock.Mock(free=1, total=2, used=1)
        with mock.patch.object(rr.shutil, 'disk_usage', return_value=usage):
            with self.assertRaises(RuntimeError):
                rr.budget(time.time() + 60)

    def test_budget_rejects_past_deadline(self):
        with self.assertRaises(TimeoutError):
            rr.budget(time.time() - 1)

    def test_budget_marks_inexact_when_unknown(self):
        with mock.patch.object(rr.os, 'walk', phantom_walk(['.checkpoint-ab12cd34.tmp'])):
            value = rr.budget(time.time() + 60)
        self.assertFalse(value['new_round_bytes_exact'])
        self.assertEqual(value['scan_unknown_leaves'], 1)

    def test_budget_exact_on_clean_scan(self):
        (self.root / 'stable.bin').write_bytes(b'0123456789')
        value = rr.budget(time.time() + 60)
        self.assertTrue(value['new_round_bytes_exact'])
        self.assertEqual(value['scan_unknown_leaves'], 0)
        self.assertEqual(value['new_round_bytes'], 10)


if __name__ == '__main__':
    unittest.main()
