"""Configured GPU and active validation allocations stay in disjoint live pools."""
import importlib.util
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import site_binding as api
spec = importlib.util.spec_from_file_location('gpu_site_inputs', ROOT / 'tools/activation/prepare_gpu_site.py')
inputs = importlib.util.module_from_spec(spec); spec.loader.exec_module(inputs)


class GPUSiteInputsTest(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'configs/RESOURCE-POLICY.json'
        self.policy = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'value': json.loads(path.read_text())}

    def test_future_owner_rank_and_control_masks_share_the_eighty_percent_cap(self):
        # Synthetic input capacities only, never runtime defaults.
        resources = {'affinity': list(range(128)), 'cpuset': list(range(128)), 'cpu_max': ['12000000', '100000']}
        value = inputs.partition_cpus(resources, 'owner', self.policy, 8)
        self.assertEqual(value['effective_cpus'], 120)
        self.assertEqual(value['total_fraction_cap'], '4/5')
        self.assertEqual(len(value['owner_mask']), 96)
        self.assertEqual(len(value['rank_masks']), 8)
        self.assertEqual([len(mask) for mask in value['rank_masks']], [11] * 8)
        self.assertEqual(len(value['control_mask']), 8)
        self.assertEqual(value['validation_mask'], list(range(24)))
        self.assertEqual(value['owner_mask'], list(range(24, 120)))
        flat = value['control_mask'] + [cpu for mask in value['rank_masks'] for cpu in mask]
        self.assertEqual(sorted(set(flat)), value['owner_mask'])
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(value['loader_workers'], 1)

    def test_insufficient_or_fractional_quota_cannot_borrow_full_affinity(self):
        with self.assertRaises(ValueError):
            inputs.partition_cpus({'affinity': list(range(128)), 'cpuset': list(range(128)),
                                   'cpu_max': ['1100000', '100000']}, 'owner', self.policy, 8)
        value = inputs.partition_cpus({'affinity': list(range(128)), 'cpuset': list(range(128)),
                                      'cpu_max': ['1499000', '100000']}, 'owner', self.policy, 8)
        self.assertEqual(len(value['owner_mask']), 11)
        self.assertEqual(value['loader_workers'], 0)

    def test_validation_may_stay_active_but_wrong_uid_or_overlapping_mask_is_rejected(self):
        value = {'identity': {'NCCCU_POD_UID': 'uid'}, 'max_total_threads': 2,
                 'allocations': {'validator': {'mask': [0, 1], 'threads': 2}}, 'task_tree_exited': False}
        api.verify_validation_pool(value, 'uid', [0, 1], [2, 3])
        for changed in ({**value, 'identity': {'NCCCU_POD_UID': 'old'}},
                        {**value, 'allocations': {'validator': {'mask': [1, 2], 'threads': 2}}}):
            with self.subTest(value=changed), self.assertRaises(ValueError):
                api.verify_validation_pool(changed, 'uid', [0, 1], [2, 3])

    def test_fractions_are_consumed_from_policy_instead_of_code_defaults(self):
        policy = copy.deepcopy(self.policy)
        policy['value']['cpu_fractions'] = {'validation': '1/4', 'training': '1/2'}
        value = inputs.partition_cpus({'affinity': list(range(128)), 'cpuset': list(range(128)),
                                      'cpu_max': ['12000000', '100000']}, 'owner', policy, 8)
        self.assertEqual(value['validation_mask'], list(range(30)))
        self.assertEqual(value['owner_mask'], list(range(30, 90)))


class NativeGuardInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source/runtime/native/cpu_affinity.c'
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b'/* native guard source binding fixture */\n')
        self.snapshot = self.root / 'technical-prep/build/source/cpu_affinity.c'
        self.snapshot.parent.mkdir(parents=True)
        self.snapshot.write_bytes(self.source.read_bytes())
        self.artifact = self.snapshot.parent.parent / 'libcpu_affinity.so'
        self.artifact.write_bytes(b'native artifact SHA fixture; never dynamically loaded\n')
        policy_path = self.root / 'source/configs/RESOURCE-POLICY.json'
        policy_path.parent.mkdir(parents=True)
        settings = {'native_affinity_guard_source': 'runtime/native/cpu_affinity.c'}
        policy_path.write_text(json.dumps(settings))
        self.policy = {'path': str(policy_path), 'sha256': inputs.sha(policy_path), 'value': settings}
        self.receipt = self.artifact.parent / 'NATIVE-AFFINITY.json'
        self.value = {'schema_version': 1, 'passed': True, 'scope': 'compute_pod_native_affinity_guard',
            'source': {'path': str(self.snapshot), 'sha256': inputs.sha(self.snapshot)},
            'artifact': {'path': str(self.artifact), 'sha256': inputs.sha(self.artifact)},
            'checks': [{'name': 'small_binding_fixture', 'passed': True}]}

    def verify(self, value=None, receipt_sha=None):
        self.receipt.write_text(json.dumps(self.value if value is None else value))
        return inputs.verify_native_guard(self.receipt, receipt_sha or inputs.sha(self.receipt),
                                          self.policy, self.root)

    def test_pinned_snapshot_build_binds_identical_current_git_source(self):
        guard = self.verify()
        self.assertEqual(guard, {'path': str(self.artifact), 'sha256': inputs.sha(self.artifact),
            'source': {'path': str(self.source), 'sha256': inputs.sha(self.source)},
            'build_receipt': {'path': str(self.receipt), 'sha256': inputs.sha(self.receipt)}})
        self.assertEqual(json.loads(self.receipt.read_text())['source']['path'], str(self.snapshot))

    def test_wrong_receipt_source_or_artifact_pin_is_rejected(self):
        with self.assertRaises(ValueError):
            self.verify(receipt_sha='0' * 64)
        for key in ('source', 'artifact'):
            value = copy.deepcopy(self.value); value[key]['sha256'] = '0' * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(value)

    def test_changed_git_source_or_artifact_content_is_rejected(self):
        for path in (self.source, self.artifact):
            original = path.read_bytes()
            path.write_bytes(b'X' + original[1:])
            try:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    self.verify()
            finally:
                path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
