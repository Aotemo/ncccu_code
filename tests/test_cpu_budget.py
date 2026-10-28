"""A stale allocation or overlapping executor CPUs must never reach heavy work."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cpu_budget_under_test', ROOT / 'tools/environment/cpu_budget.py')
budget = importlib.util.module_from_spec(spec); spec.loader.exec_module(budget)


class CPUBudgetTest(unittest.TestCase):
    def setUp(self):
        # These are synthetic capacities, not an environment default or a live grant.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.policy = self.write_policy('1/5', '4/5')
        self.resources = {'affinity': list(range(128)), 'cpuset': list(range(128)),
                          'available': list(range(128)), 'cpu_max': ['12000000', '100000'],
                          'effective_cpu_fraction': '120', 'effective_cpus': 120.0,
                          'max_total_threads': 24, 'validation_mask': list(range(24)),
                          'resource_policy': self.policy}
        self.identity = {'NCCCU_POD_UID': 'synthetic-current-uid'}
        self.value = budget.build_allocation(self.resources, ['executor-a', 'executor-b'],
                                             self.identity, now=1000, valid_seconds=600)

    def write_policy(self, validation, training):
        value = {'schema_version': 1,
                 'cpu_fractions': {'validation': validation, 'training': training},
                 'native_affinity_environment': {'AMD_CPU_AFFINITY': '1',
                                                  'HSA_OVERRIDE_CPU_AFFINITY_DEBUG': '0',
                                                  'NCCL_IGNORE_CPU_AFFINITY': '0'},
                 'rank_threads': 1, 'interop_threads': 1, 'loader_workers': 1}
        path = self.root / 'RESOURCE-POLICY.json'
        path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': budget.sha(path), 'value': value}

    def write_allocation(self, value):
        value = copy.deepcopy(value)
        value['guard_sha256'] = budget.sha(budget.__file__)
        path = self.root / 'ALLOCATION.json'
        path.write_text(json.dumps(value))
        return argparse.Namespace(allocation=path, allocation_sha=budget.sha(path),
                                  owner='executor-a', operation='run',
                                  script=self.root / 'fixture.py', script_args=[])

    def test_twenty_percent_is_shared_by_both_executors(self):
        self.assertEqual(self.value['resource_policy'], self.policy)
        self.assertEqual(self.value['allocations']['executor-a'], {'threads': 12, 'mask': list(range(12))})
        self.assertEqual(self.value['allocations']['executor-b'], {'threads': 12, 'mask': list(range(12, 24))})
        self.assertEqual(budget.validate_allocation(self.value, self.resources, self.identity,
                                                   'executor-a', now=1001)['threads'], 12)
        smaller = {**self.resources, 'available': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                   'cpuset': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                   'effective_cpu_fraction': '10', 'effective_cpus': 10.0,
                   'max_total_threads': 2, 'validation_mask': [2, 4]}
        value = budget.build_allocation(smaller, ['a', 'b'], self.identity, now=1000, valid_seconds=600)
        self.assertEqual(value['allocations'], {'a': {'threads': 1, 'mask': [2]}, 'b': {'threads': 1, 'mask': [4]}})

    def test_overlapping_masks_missing_owner_and_quota_reduction_fail(self):
        bad = copy.deepcopy(self.value); bad['allocations']['executor-b']['mask'][0] = 0
        with self.assertRaises(ValueError):
            budget.validate_allocation(bad, self.resources, self.identity, 'executor-a', now=1001)
        with self.assertRaises(ValueError):
            budget.validate_allocation(self.value, self.resources, self.identity, 'unknown-executor', now=1001)
        with self.assertRaises(ValueError):
            budget.validate_allocation(self.value, {**self.resources, 'max_total_threads': 20,
                                                   'cpu_max': ['10000000', '100000'],
                                                   'effective_cpu_fraction': '100', 'effective_cpus': 100.0,
                                                   'validation_mask': list(range(20))},
                                       self.identity, 'executor-a', now=1001)

    def test_old_uid_future_or_expired_allocation_fail(self):
        for identity, now in [({'NCCCU_POD_UID': 'old-uid'}, 1001), (self.identity, 999), (self.identity, 1601)]:
            with self.subTest(now=now), self.assertRaises(ValueError):
                budget.validate_allocation(self.value, self.resources, identity, 'executor-a', now=now)

    def test_inadequate_capacity_or_missing_executor_is_rejected(self):
        for owners, resources in [(['only-one'], self.resources), (['a', 'a'], self.resources),
                                   (['a', 'b'], {**self.resources, 'available': list(range(5)),
                                                'cpuset': list(range(5)),
                                                'effective_cpu_fraction': '5', 'effective_cpus': 5.0,
                                                'max_total_threads': 1, 'validation_mask': [0]})]:
            with self.subTest(owners=owners), self.assertRaises(ValueError):
                budget.build_allocation(resources, owners, self.identity, now=1000, valid_seconds=600)

    def test_measure_consumes_non_default_policy_and_front_live_pool(self):
        policy = self.write_policy('1/3', '2/3')
        files = {'/sys/fs/cgroup/cpuset.cpus.effective': '0-5,8-15',
                 '/sys/fs/cgroup/cpu.max': '1050000 100000'}
        environment = {'NCCCU_RESOURCE_POLICY': policy['path'],
                       'NCCCU_RESOURCE_POLICY_SHA256': policy['sha256']}
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(budget.os, 'sched_getaffinity', return_value=set(range(16))), \
                patch.object(Path, 'read_text', autospec=True, side_effect=lambda path: files[str(path)]):
            resources = budget.measure()
        self.assertEqual(resources['effective_cpu_fraction'], '21/2')
        self.assertEqual(resources['max_total_threads'], 3)
        self.assertEqual(resources['validation_mask'], [0, 1, 2])
        self.assertEqual(resources['resource_policy'], policy)
        value = budget.build_allocation(resources, ['a', 'b'], self.identity, now=1000, valid_seconds=600)
        self.assertEqual(value['total_fraction_cap'], '1/3')
        self.assertEqual(value['allocations'], {'a': {'threads': 2, 'mask': [0, 1]},
                                                'b': {'threads': 1, 'mask': [2]}})

    def test_policy_requires_explicit_path_sha_and_unchanged_embedded_value(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            budget.load_resource_policy()
        with self.assertRaises(ValueError):
            budget.load_resource_policy({**self.policy, 'path': 'RESOURCE-POLICY.json'})
        with self.assertRaises(ValueError):
            budget.load_resource_policy({**self.policy, 'sha256': '0' * 64})
        changed_value = copy.deepcopy(self.policy)
        changed_value['value']['cpu_fractions'] = {'validation': '1/3', 'training': '2/3'}
        with self.assertRaises(ValueError):
            budget.load_resource_policy(changed_value)
        self.write_policy('1/3', '2/3')
        with self.assertRaises(ValueError):
            budget.load_resource_policy(self.policy)

    def test_policy_rejects_nonpositive_and_overcommitted_fractions(self):
        for validation, training in [('0', '1'), ('-1/5', '4/5'), ('1/2', '3/4'),
                                     ('NaN', '4/5'), (0.2, '4/5')]:
            bad = copy.deepcopy(self.policy['value'])
            bad['cpu_fractions'] = {'validation': validation, 'training': training}
            with self.subTest(validation=validation, training=training), self.assertRaises(ValueError):
                budget.validate_resource_policy(bad)

    def test_available_training_segment_and_missing_policy_are_rejected(self):
        bad = copy.deepcopy(self.value)
        bad['allocations']['executor-b']['mask'] = list(range(24, 36))
        with self.assertRaises(ValueError):
            budget.validate_allocation(bad, self.resources, self.identity, 'executor-a', now=1001)
        bad = copy.deepcopy(self.value)
        del bad['resource_policy']
        args = self.write_allocation(bad)
        with patch.object(budget, 'measure') as measure, patch.object(budget.subprocess, 'Popen') as spawn:
            with self.assertRaises(ValueError):
                budget.run_task(args)
            measure.assert_not_called()
            spawn.assert_not_called()

    def test_run_task_rechecks_allocation_policy_and_forwards_exact_pin(self):
        args = self.write_allocation(self.value)
        child = Mock()
        child.poll.return_value = 0
        environment = {'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': '', 'ROCR_VISIBLE_DEVICES': '',
                       'NCCCU_RESOURCE_POLICY': '/unrelated/policy.json',
                       'NCCCU_RESOURCE_POLICY_SHA256': '0' * 64}
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(budget, 'measure', return_value=self.resources) as measure, \
                patch.object(budget, 'identity', return_value=self.identity), \
                patch.object(budget.time, 'time', return_value=1001), \
                patch.object(budget.os, 'sched_setaffinity') as bind, \
                patch.object(budget.os, 'open', return_value=12345), \
                patch.object(budget.os, 'close'), patch.object(budget.fcntl, 'flock'), \
                patch.object(budget.subprocess, 'Popen', return_value=child) as spawn, \
                patch.object(budget, 'snapshot_tree', return_value={'threads': 1, 'runnable_threads': 0,
                                                                  'live_pids': [], 'affinity_violations': []}), \
                patch('builtins.print'):
            self.assertEqual(budget.run_task(args), 0)
            measure.assert_called_once_with(self.policy)
            bind.assert_called_once_with(0, set(range(12)))
            passed = spawn.call_args.kwargs['env']
            self.assertEqual(passed['NCCCU_RESOURCE_POLICY'], self.policy['path'])
            self.assertEqual(passed['NCCCU_RESOURCE_POLICY_SHA256'], self.policy['sha256'])
            self.assertEqual(passed['NCCCU_CPU_MASK'], json.dumps(list(range(12))))
            self.assertEqual(passed['NCCCU_CPU_INTEROP_THREADS'], '1')
            for key, expected in self.policy['value']['native_affinity_environment'].items():
                self.assertEqual(passed[key], expected)
        receipt_path, = self.root.glob('tasks/executor-a-*/CPU-RECEIPT.json')
        self.assertEqual(json.loads(receipt_path.read_text())['resource_policy'], self.policy)
        self.write_policy('1/3', '2/3')
        with patch.object(budget.subprocess, 'Popen') as spawn:
            with self.assertRaises(ValueError):
                budget.run_task(args)
            spawn.assert_not_called()

    def test_cpuset_parser_keeps_holes_and_rejects_invalid_ranges(self):
        self.assertEqual(budget.parse_cpuset('0-2,7,9-10'), [0, 1, 2, 7, 9, 10])
        for invalid in ('', '5-2', '-1', '1-2-3'):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                budget.parse_cpuset(invalid)


if __name__ == '__main__':
    unittest.main()
