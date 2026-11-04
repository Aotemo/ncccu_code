"""Keep the existing activation storage boundary and report its actual reading."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))
SPEC = importlib.util.spec_from_file_location('activation_storage_under_test',
    ROOT / 'tools/activation/activate.py')
activation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(activation)


class ActivationStorageTest(unittest.TestCase):
    def test_above_boundary_returns_the_same_reading_and_checks_once(self):
        self.assertTrue(callable(getattr(activation, 'verify_storage', None)),
            'activation must preserve and report the observed storage value')
        usage = SimpleNamespace(total=100 * 1024**3, used=75 * 1024**3, free=25 * 1024**3)
        with patch.object(activation.shutil, 'disk_usage', return_value=usage) as read:
            evidence = activation.verify_storage('/fixture/work')
        read.assert_called_once_with('/fixture/work')
        self.assertEqual(evidence['free_bytes'], usage.free)
        self.assertEqual(evidence['minimum_free_bytes'], 21474836480)
        self.assertEqual(evidence['work_root'], '/fixture/work')

    def test_at_or_below_boundary_rejects_with_actual_bytes_and_path(self):
        self.assertTrue(callable(getattr(activation, 'verify_storage', None)),
            'activation must preserve and report the observed storage value')
        for free in (0, 21474836479, 21474836480):
            with self.subTest(free=free), patch.object(activation.shutil, 'disk_usage',
                    return_value=SimpleNamespace(free=free)), self.assertRaises(ValueError) as caught:
                activation.verify_storage('/fixture/work')
            for expected in ('free_bytes=' + str(free), 'minimum_free_bytes=21474836480',
                    'work_root=/fixture/work'):
                self.assertIn(expected, str(caught.exception))


if __name__ == '__main__':
    unittest.main()
