"""Standard-library contracts only; the builder performs the real Pod C check."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_affinity_under_test',
                                            ROOT / 'tools/environment/build_affinity_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class NativeAffinityGuardTest(unittest.TestCase):
    def test_fixture_proof_requires_every_real_mask_errno_and_provider(self):
        # Protocol fixtures never compile a library, run native code or publish a receipt.
        records = [{'name': name, 'passed': True, 'provider_verified': True, 'cpu_mask': '2,4',
                    'return_code': -1 if error else 0, 'error': error}
                   for name, error in guard.CHECK_ERRNOS.items()]
        text = lambda value: '\n'.join(json.dumps(item) for item in value)
        self.assertEqual(guard.validate_fixture_output(text(records), [2, 4]), records)
        variants = [records[:-1], [*records[:-1], records[0]]]
        for key, value in [('cpu_mask', '0,1,2,3,4'), ('provider_verified', False),
                           ('passed', False), ('return_code', -1), ('error', 22)]:
            bad = copy.deepcopy(records)
            bad[0][key] = value
            variants.append(bad)
        for bad in variants:
            with self.subTest(records=bad), self.assertRaises(ValueError):
                guard.validate_fixture_output(text(bad), [2, 4])

    def test_new_output_cannot_escape_or_overwrite_technical_prep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / 'exp-v3/technical-prep/native-new'
            self.assertEqual(guard.reserve_output(output), output)
            for bad in (output, root / 'elsewhere/new', root / 'exp-v3/technical-prep', Path('relative')):
                with self.subTest(path=bad), self.assertRaises(ValueError):
                    guard.reserve_output(bad)

    def test_missing_cpu_context_refuses_before_import_or_compiler(self):
        environments = [{}, {'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': '', 'ROCR_VISIBLE_DEVICES': ''}]
        for environment in environments:
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), \
                    patch.object(guard, 'load_budget') as load, patch.object(guard.subprocess, 'run') as run:
                with self.assertRaises(ValueError):
                    guard.cpu_context()
                load.assert_not_called()
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
