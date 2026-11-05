"""The preparation entry limits torch to the explicit CPU allocation."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prepare_entry', REPO / 'tools/activation/prepare.py')
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class PrepareCpuThreadsTest(unittest.TestCase):
    def test_allocation_threads_are_applied(self):
        torch = mock.Mock()
        with mock.patch.dict(os.environ, {'NCCCU_CPU_THREADS': '12'}), \
                mock.patch.dict(sys.modules, {'torch': torch}):
            self.assertEqual(PREPARE.limit_cpu_threads(), 12)
        torch.set_num_threads.assert_called_once_with(12)
        torch.set_num_interop_threads.assert_called_once_with(1)

    def test_missing_allocation_does_not_import_torch(self):
        torch = mock.Mock()
        environment = {key: value for key, value in os.environ.items() if key != 'NCCCU_CPU_THREADS'}
        with mock.patch.dict(os.environ, environment, clear=True), \
                mock.patch.dict(sys.modules, {'torch': torch}):
            self.assertIsNone(PREPARE.limit_cpu_threads())
        torch.set_num_threads.assert_not_called()


if __name__ == '__main__':
    unittest.main()
