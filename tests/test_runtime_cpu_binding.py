"""GPU CPU budgets are explicit, bounded and stable after narrowing controller affinity."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain import site_binding as api


class RuntimeCpuBindingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.site = {
            'schema_version': 1, 'scope': 'formal_exp_v3', 'round_id': 'independent-cpu-fixture',
            'git_commit': 'a' * 40, 'work_root': str(root), 'round_root': str(root / 'rounds/fixture'),
            'metadata_root': str(root / 'assets/metadata'), 'data_root': str(root / 'assets/data'),
            'train_identity': str(root / 'assets/train-identity.json'), 'prepared_cache': str(root / 'cache'),
            'official_manifest': str(root / 'assets/official.json'), 'public_weight': str(root / 'assets/weight.pt'),
            'public_provenance': str(root / 'assets/provenance.json'),
            'pod': {'namespace': 'fixture', 'name': 'fixture-pod', 'uid': '11111111-2222-4333-8444-555555555555',
                    'container': 'fixture-container', 'python': sys.executable},
            'gpus': [{'ordinal': n, 'pci': '0000:%02x:00.0' % (n + 1), 'vram_total_bytes': 4096} for n in range(8)],
            'science': {'candidate': {'id': 'structural-fixture-no-model'}, 'binding_version': 'fixture'},
        }
        self.policy = {
            'owner_id': 'fixture-A', 'effective_cpus': 120, 'total_fraction_cap': '4/5',
            'owner_mask': list(range(24, 120)), 'control_mask': list(range(112, 120)),
            'validation_mask': list(range(24)),
            'rank_masks': [list(range(24 + 11 * n, 24 + 11 * n + 11)) for n in range(8)],
            'rank_threads': 1, 'interop_threads': 1, 'loader_workers': 1,
            'allocation_receipt': {'path': str(root / 'gpu-allocation.json'), 'sha256': '0' * 64},
        }
        self.site['cpu_execution'] = self.policy
        policy_path = root / 'RESOURCE-POLICY.json'
        policy_path.write_bytes((ROOT / 'configs/RESOURCE-POLICY.json').read_bytes())
        self.policy['resource_policy'] = api.load_resource_policy(policy_path, api.sha256(policy_path))
        release_path = root / 'validation-allocation.json'
        release_path.write_text(json.dumps({'identity': {'NCCCU_POD_UID': self.site['pod']['uid']},
            'allocations': {'active-validator': {'mask': list(range(24)), 'threads': 24}},
            'max_total_threads': 24, 'task_tree_exited': False}))
        self.live = {'affinity': list(range(128)), 'cpuset': list(range(128)), 'cpu_max': ['12000000', '100000']}
        self.receipt = {
            'schema_version': 1, 'stage': 'gpu-owner',
            'execution': {'namespace': 'fixture', 'pod': 'fixture-pod', 'uid': self.site['pod']['uid'],
                          'container': 'fixture-container', 'python': sys.executable},
            'resources': {**copy.deepcopy(self.live), 'effective_cpus': 120},
            'total_fraction_cap': '4/5', 'owner_id': 'fixture-A', 'owner_mask': list(range(24, 120)),
            'validation_mask': self.policy['validation_mask'], 'resource_policy': self.policy['resource_policy'],
            'created_at': 100, 'valid_until': 200,
            'validation_allocation': {'path': str(release_path), 'sha256': api.sha256(release_path)},
        }
        self.allocation_path = root / 'gpu-allocation.json'
        self.allocation_path.write_text(json.dumps(self.receipt))
        self.policy['allocation_receipt']['sha256'] = api.sha256(self.allocation_path)

    def test_controller_submask_does_not_become_the_pod_capacity(self):
        api.validate_site(self.site)
        narrowed = {**self.live, 'affinity': self.policy['control_mask']}
        result = api.validate_gpu_allocation(self.site, self.receipt, narrowed, now=101)
        self.assertEqual(result['owner_mask'], list(range(24, 120)))

    def test_first_gpu_entry_requires_actual_original_resources(self):
        api.validate_gpu_allocation(self.site, self.receipt, self.live, initial=True, now=101)
        for changed in ({**self.live, 'affinity': list(range(96))},
                        {**self.live, 'cpu_max': ['12500000', '100000']},
                        {**self.live, 'cpuset': list(range(127))}):
            with self.subTest(live=changed), self.assertRaises(ValueError):
                api.validate_gpu_allocation(self.site, self.receipt, changed, initial=True, now=101)

    def test_live_quota_shrink_cpuset_escape_expiry_and_old_uid_fail_closed(self):
        for changed in ({**self.live, 'cpu_max': ['11900000', '100000']},
                        {**self.live, 'cpuset': list(range(1, 128))}):
            with self.subTest(live=changed), self.assertRaises(ValueError):
                api.validate_gpu_allocation(self.site, self.receipt, changed, now=101)
        with self.assertRaises(ValueError):
            api.validate_gpu_allocation(self.site, self.receipt, self.live, now=201)
        wrong = copy.deepcopy(self.receipt); wrong['execution']['uid'] = 'old-pod'
        with self.assertRaises(ValueError):
            api.validate_gpu_allocation(self.site, wrong, self.live, now=101)

    def test_masks_must_be_disjoint_and_fit_the_floor_of_eighty_percent(self):
        cases = []
        wrong = copy.deepcopy(self.policy); wrong['rank_masks'][1] = wrong['rank_masks'][0]; cases.append(wrong)
        wrong = copy.deepcopy(self.policy); wrong['control_mask'][0] = 24; cases.append(wrong)
        wrong = copy.deepcopy(self.policy); wrong['effective_cpus'] = 119.99; cases.append(wrong)
        wrong = copy.deepcopy(self.policy); wrong['owner_mask'].append(120); cases.append(wrong)
        wrong = copy.deepcopy(self.policy); wrong['total_fraction_cap'] = '1/5'; cases.append(wrong)
        wrong = copy.deepcopy(self.policy); wrong['loader_workers'] = 2; cases.append(wrong)
        wrong = copy.deepcopy(self.policy); wrong['rank_threads'] = True; cases.append(wrong)
        for policy in cases:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                api.validate_cpu_execution(policy, world=8)

    def test_role_binding_applies_only_its_explicit_mask_and_rejects_escaped_inheritance(self):
        with patch.object(api, 'verify_gpu_cpu', return_value=(self.policy, {'affinity': list(range(24, 120))})), \
             patch.object(api.os, 'sched_setaffinity') as setter, \
             patch.object(api.os, 'sched_getaffinity', return_value=set(self.policy['control_mask'])), \
             patch.dict(api.os.environ, {'AMD_CPU_AFFINITY': '0',
                 'HSA_OVERRIDE_CPU_AFFINITY_DEBUG': '1', 'NCCL_IGNORE_CPU_AFFINITY': '1'}, clear=False):
            self.assertEqual(api.bind_cpu_role(self.site, 'control'), self.policy['control_mask'])
            setter.assert_called_once_with(0, set(self.policy['control_mask']))
            self.assertEqual(api.os.environ['OMP_NUM_THREADS'], '1')
            self.assertEqual(api.os.environ['AMD_CPU_AFFINITY'], '1')
            self.assertEqual(api.os.environ['HSA_OVERRIDE_CPU_AFFINITY_DEBUG'], '0')
            self.assertEqual(api.os.environ['NCCL_IGNORE_CPU_AFFINITY'], '0')
        with patch.object(api, 'verify_gpu_cpu', return_value=(self.policy, self.live)), \
             patch.object(api.os, 'sched_setaffinity') as setter:
            with self.assertRaises(ValueError):
                api.bind_cpu_role(self.site, 'control')
            setter.assert_not_called()

    def test_missing_allocation_cannot_fall_back_to_full_pod(self):
        policy = importlib.import_module('retrain.execution_policy')
        with patch.dict(api.os.environ, {'NCCCU_SITE_BINDING': ''}):
            with self.assertRaises(ValueError):
                policy.allocate_cpus(8, available=list(range(128)), quota_cores=120)

    def test_training_allocator_uses_eighty_percent_and_rejects_live_quota_shrink(self):
        policy = importlib.import_module('retrain.execution_policy')
        with patch.object(api, 'load_site', return_value=self.site), \
             patch.object(api, 'verify_gpu_cpu', return_value=(self.policy, self.live)):
            width, masks = policy.allocate_cpus(8, available=list(range(128)), quota_cores=120)
            self.assertEqual(width, 11)
            self.assertEqual(masks, self.policy['rank_masks'])
            with self.assertRaises(ValueError):
                policy.allocate_cpus(8, available=list(range(128)), quota_cores=119.99)

    def test_allocation_and_active_validation_pool_content_are_pinned(self):
        with patch.object(api, 'live_cpu_resources', return_value=self.live), \
             patch.object(api.time, 'time', return_value=101):
            api.verify_gpu_cpu(self.site)
            release = Path(self.receipt['validation_allocation']['path'])
            release.write_text('{}')
            with self.assertRaises(ValueError):
                api.verify_gpu_cpu(self.site)
        self.allocation_path.write_text('{}')
        with self.assertRaises(ValueError):
            api.verify_gpu_cpu(self.site)

    def test_rank_proofs_need_ordered_pci_memory_and_exact_cpu_masks(self):
        proofs = [{'rank': n, 'pid': 100 + n, 'world_size': 8, 'device': 'cuda:' + str(n),
            'physical_pci': card['pci'], 'total_memory_bytes': card['vram_total_bytes'],
            'requested_cpu_ids': self.policy['rank_masks'][n], 'actual_cpu_ids': self.policy['rank_masks'][n]}
            for n, card in enumerate(self.site['gpus'])]
        api.verify_rank_devices(proofs, self.site)
        for key, value in (('actual_cpu_ids', [127]), ('physical_pci', proofs[1]['physical_pci']),
                           ('total_memory_bytes', 8192), ('world_size', 4)):
            wrong = copy.deepcopy(proofs); wrong[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                api.verify_rank_devices(wrong, self.site)


if __name__ == '__main__':
    unittest.main()
