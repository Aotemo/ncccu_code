"""The compatibility library must load before the one-shot GPU attempt is reserved."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import site_binding as api


class NativeEntryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.artifact = Path(temporary.name) / 'fixture-not-a-library.so'
        self.artifact.write_text('never loaded by these standard-library fixtures')
        self.policy = {'owner_mask': [3, 5], 'native_affinity_guard': {'path': str(self.artifact)},
            'resource_policy': {'value': json.loads((ROOT / 'configs/RESOURCE-POLICY.json').read_text())}}

    def test_first_entry_executes_with_exact_library_and_configured_native_environment(self):
        with patch.object(api, 'verify_gpu_cpu', return_value=(self.policy, {})), \
             patch.object(api, 'native_guard_loaded', return_value=False), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(api.os, 'execve') as execute:
            api.prepare_native_entry({})
        executable, arguments, environment = execute.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertEqual(arguments[:3], [sys.executable, '-B', '-u'])
        self.assertEqual(environment['LD_PRELOAD'], str(self.artifact))
        self.assertEqual(environment['NCCCU_NATIVE_CPU_MASK'], '3,5')
        for key, value in self.policy['resource_policy']['value']['native_affinity_environment'].items():
            self.assertEqual(environment[key], value)

    def test_already_loaded_entry_is_not_reexecuted_and_unknown_preload_is_rejected(self):
        with patch.object(api, 'verify_gpu_cpu', return_value=(self.policy, {})), \
             patch.object(api, 'native_guard_loaded', return_value=True), \
             patch.object(api.os, 'execve') as execute:
            api.prepare_native_entry({})
            execute.assert_not_called()
        with patch.object(api, 'verify_gpu_cpu', return_value=(self.policy, {})), \
             patch.object(api, 'native_guard_loaded', return_value=False), \
             patch.dict(os.environ, {'LD_PRELOAD': '/unknown/library.so'}), \
             patch.object(api.os, 'execve') as execute:
            with self.assertRaises(ValueError):
                api.prepare_native_entry({})
            execute.assert_not_called()

    def test_loader_environment_without_actual_mapping_does_not_count_as_loaded(self):
        with patch.dict(os.environ, {'LD_PRELOAD': str(self.artifact)}), \
             patch.object(api.Path, 'read_text', return_value='1000-2000 r-xp 0 00:00 1 /other.so\n'):
            self.assertFalse(api.native_guard_loaded(self.policy))
        with patch.dict(os.environ, {'LD_PRELOAD': str(self.artifact)}), \
             patch.object(api.Path, 'read_text', return_value='1000-2000 r-xp 0 00:00 1 ' + str(self.artifact) + '\n'):
            self.assertTrue(api.native_guard_loaded(self.policy))


if __name__ == '__main__':
    unittest.main()
