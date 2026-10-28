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

    def test_external_entry_loads_context_once_per_interpreter_across_preload(self):
        import hashlib
        import importlib.util
        import time
        from types import SimpleNamespace
        spec = importlib.util.spec_from_file_location('native_entry_dispatch', ROOT / 'tools/validation/gpu_admission.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)

        class ReexecRequested(Exception):
            pass

        class BeforeGPU(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / 'exp-v3'
            work.mkdir()
            prep = work / 'technical-prep/frozen/formal-cpu'
            prep.mkdir(parents=True)
            output = work / 'technical-prep/attempt'
            site = {'work_root': str(work), 'round_root': str(work / 'rounds/fixture')}
            site_path, deployment = work / 'site.json', work / 'deployment.json'
            site_path.write_text(json.dumps(site))
            deployment.write_text(json.dumps({'source_root': str(prep.parent / 'source')}))
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            arguments = ['--prep-root', str(prep), '--site-binding', str(site_path),
                '--site-binding-sha256', digest(site_path), '--deployment-receipt', str(deployment),
                '--deployment-receipt-sha256', digest(deployment), '--package-manifest', str(work / 'package.json'),
                '--package-manifest-sha256', 'a' * 64, '--cpu-prep-receipt', str(work / 'cpu.json'),
                '--cpu-prep-receipt-sha256', 'b' * 64, '--pod-uid', 'fixture', '--ready', 'True',
                '--observed-at', str(time.time()), '--output', str(output)]
            from unittest.mock import Mock
            native = SimpleNamespace(prepare_native_entry=Mock(side_effect=ReexecRequested),
                verify_gpu_cpu=Mock(side_effect=BeforeGPU))
            context = {'api': native, 'site': site}
            # First interpreter re-execs before reserving the one-shot output.
            with patch.object(entry, 'load_context', return_value=context) as load:
                with self.assertRaises(ReexecRequested):
                    entry.main(arguments)
                load.assert_called_once()
            self.assertFalse(output.exists())
            native.verify_gpu_cpu.assert_not_called()
            # The preloaded interpreter traverses the real main -> run_outer.
            # A second load_context would fail before reaching this CPU gate.
            native.prepare_native_entry.side_effect = None
            with patch.object(entry, 'load_context', side_effect=[context, AssertionError('duplicate context')]) as load, \
                 patch.object(entry.signal, 'signal'):
                with self.assertRaises(BeforeGPU):
                    entry.main(arguments)
                load.assert_called_once()
            native.verify_gpu_cpu.assert_called_once()
            self.assertTrue((output / 'STARTED.json').is_file())
            self.assertFalse((output / 'owner-launch.json').exists())


if __name__ == '__main__':
    unittest.main()
