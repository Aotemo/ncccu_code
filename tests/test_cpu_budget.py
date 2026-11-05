"""A stale allocation or overlapping executor CPUs must never reach heavy work."""
import argparse
import copy
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
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

    def test_local_executor_lease_blocks_other_process_and_releases(self):
        identity = {'NCCCU_POD_UID': self.root.name}
        path = budget.task_lease_path(identity, 'executor-a')
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        self.assertEqual(path.parent, Path('/tmp'))
        self.assertEqual(path, budget.task_lease_path(identity, 'executor-a'))
        self.assertNotEqual(path, budget.task_lease_path(identity, 'executor-b'))
        self.assertNotEqual(path, budget.task_lease_path({'NCCCU_POD_UID': 'other'}, 'executor-a'))
        code = ('import fcntl,os,sys;f=os.open(sys.argv[1],os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);'
                '\ntry:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
                '\nexcept BlockingIOError:sys.exit(3)')
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(subprocess.run([sys.executable, '-B', '-c', code, str(path)],
                                           check=False, timeout=10).returncode, 3)
        finally:
            os.close(descriptor)
        self.assertEqual(subprocess.run([sys.executable, '-B', '-c', code, str(path)],
                                       check=False, timeout=10).returncode, 0)

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

    def test_one_executor_can_own_the_complete_preparation_pool(self):
        value = budget.build_allocation(self.resources, ['executor-a'], self.identity,
                                        now=1000, valid_seconds=600)
        self.assertEqual(value['allocations'], {'executor-a': {'threads': 24, 'mask': list(range(24))}})
        self.assertEqual(budget.validate_allocation(value, self.resources, self.identity,
                                                   'executor-a', now=1001)['threads'], 24)
        with self.assertRaises(ValueError):
            budget.validate_allocation(value, self.resources, self.identity, 'executor-b', now=1001)
        bad = copy.deepcopy(value)
        bad['allocations']['executor-a']['mask'][-1] = 24
        with self.assertRaises(ValueError):
            budget.validate_allocation(bad, self.resources, self.identity, 'executor-a', now=1001)

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
        for owners, resources in [([], self.resources), (['a', 'b', 'c'], self.resources), (['a', 'a'], self.resources),
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


class IdleBorrowTest(unittest.TestCase):
    def setUp(self):
        self.value = {'schema_version': 1, 'cpu_fractions': {'validation': '1/1', 'training': '0/1'},
            'idle_borrow': {'enabled': True, 'safe_proof_max_age_seconds': 120,
                            'device_check_interval_seconds': 5, 'idle_vram_bytes_exclusive': 67108864},
            'native_affinity_environment': {'AMD_CPU_AFFINITY': '1', 'HSA_OVERRIDE_CPU_AFFINITY_DEBUG': '0',
                                            'NCCL_IGNORE_CPU_AFFINITY': '0'},
            'rank_threads': 1, 'interop_threads': 1, 'loader_workers': 1}
        self.policy = {'path': '/fixture/idle-policy.json', 'sha256': 'a' * 64, 'value': self.value}
        self.identity = {'NCCCU_POD_UID': 'fixture-uid'}
        self.pcis = [f'0000:{i:02x}:00.0' for i in range(8)]
        self.site = {'pod': {'uid': 'fixture-uid'}, 'work_root': '/work', 'round_root': '/work/rounds/finished',
                     'gpus': [{'pci': pci} for pci in self.pcis]}
        self.proof = {'phase': 'terminal', 'formal_root': '/work/rounds/finished', 'observed_at': 1000,
            'signals_sent': False, 'model_forward_executed': False, 'official_read': False,
            'first_success_receipt_read': True, 'done': {'present': True, 'status': 'completed'},
            'identity': {'owner_expected': {'pid': 1, 'start_ticks': '1'}, 'owner_status': 'exited',
                         'owner': {'active': False, 'readable': True},
                         'ranks_expected': [{'pid': i + 2, 'start_ticks': str(i + 2)} for i in range(8)],
                         'ranks': [{'active': False, 'readable': True} for _ in range(8)]},
            'safety': {'safe': True, 'device_users': [], 'device_scan_denied': 0,
                       'lock': {'readable': True, 'holders': []}, 'all_expected_cards_seen': True,
                       'cards': [{'pci': pci, 'busy_percent': 0} for pci in self.pcis]}}
        self.live = {'device_users': [], 'device_scan_denied': 0, 'active_terminal_processes': [],
                     'lock': {'readable': True, 'holders': []},
                     'cards': [{'pci': pci, 'busy_percent': 0, 'vram_used_bytes': 0} for pci in self.pcis]}

    def test_idle_policy_grants_only_live_effective_cores(self):
        budget.validate_resource_policy(self.value)
        resources = {'available': list(range(128)), 'effective_cpu_fraction': '120',
                     'effective_cpus': 120.0, 'max_total_threads': 120,
                     'validation_mask': list(range(120)), 'resource_policy': self.policy}
        allocation = budget.build_allocation(resources, ['sole-owner'], self.identity, now=1000, valid_seconds=600)
        self.assertEqual(allocation['allocations']['sole-owner'], {'threads': 120, 'mask': list(range(120))})
        self.assertEqual(budget.validate_allocation(allocation, resources, self.identity, 'sole-owner', now=1001)['threads'], 120)
        altered = copy.deepcopy(self.value); del altered['idle_borrow']
        with self.assertRaises(ValueError): budget.validate_resource_policy(altered)

    def test_idle_evidence_requires_fresh_complete_terminal_and_live_devices(self):
        budget.validate_idle_evidence(self.site, self.proof, self.identity, self.live,
                                      now=1001, max_age=120, max_vram=67108864)
        cases = []
        changed = copy.deepcopy(self.proof); changed['observed_at'] = 800; cases.append((self.site, changed, self.live))
        changed = copy.deepcopy(self.proof); changed['identity']['ranks'][0]['active'] = True; cases.append((self.site, changed, self.live))
        changed = copy.deepcopy(self.site); changed['pod']['uid'] = 'other'; cases.append((changed, self.proof, self.live))
        changed = copy.deepcopy(self.live); changed['device_users'] = [{'pid': 99}]; cases.append((self.site, self.proof, changed))
        changed = copy.deepcopy(self.live); changed['device_scan_denied'] = 1; cases.append((self.site, self.proof, changed))
        changed = copy.deepcopy(self.live); changed['lock']['holders'] = ['gpu-owner']; cases.append((self.site, self.proof, changed))
        changed = copy.deepcopy(self.live); changed['cards'][0]['busy_percent'] = 1; cases.append((self.site, self.proof, changed))
        changed = copy.deepcopy(self.live); changed['cards'][0]['vram_used_bytes'] = 67108864; cases.append((self.site, self.proof, changed))
        for site, proof, live in cases:
            with self.subTest(site=site, proof=proof, live=live), self.assertRaises(ValueError):
                budget.validate_idle_evidence(site, proof, self.identity, live, now=1001, max_age=120, max_vram=67108864)

    def test_idle_references_are_explicit_pinned_and_live_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); site = root / 'site.json'; safety = root / 'safety.json'
            site.write_text(json.dumps(self.site)); safety.write_text(json.dumps(self.proof))
            environment = {'NCCCU_CPU_IDLE_SITE': str(site), 'NCCCU_CPU_IDLE_SITE_SHA256': budget.sha(site),
                           'NCCCU_CPU_IDLE_SAFETY': str(safety), 'NCCCU_CPU_IDLE_SAFETY_SHA256': budget.sha(safety)}
            with patch.dict(os.environ, environment, clear=True), \
                 patch.object(budget, 'snapshot_idle_devices', return_value=self.live):
                found = budget.read_idle_reference(self.policy, self.identity, now=1001)
                self.assertEqual(found['references']['site']['sha256'], budget.sha(site))
                with self.assertRaises(ValueError): budget.read_idle_reference(self.policy, self.identity, now=1201)
                safety.write_text('{}')
                with self.assertRaises(ValueError): budget.read_idle_reference(self.policy, self.identity, now=1001)

    def test_gpu_binding_rejects_idle_only_policy(self):
        sys.path.insert(0, str(ROOT / 'runtime/bundle'))
        from retrain import site_binding
        with self.assertRaises(ValueError): site_binding.validate_resource_policy(self.value)

    def test_idle_task_stops_its_own_group_if_devices_stop_being_idle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); policy_path = root / 'policy.json'; policy_path.write_text(json.dumps(self.value))
            policy = {'path': str(policy_path), 'sha256': budget.sha(policy_path), 'value': self.value}
            resources = {'available': list(range(120)), 'effective_cpu_fraction': '120', 'effective_cpus': 120.0,
                         'max_total_threads': 120, 'validation_mask': list(range(120)), 'resource_policy': policy}
            allocation = budget.build_allocation(resources, ['owner'], self.identity, now=1000, valid_seconds=600)
            allocation['guard_sha256'] = budget.sha(budget.__file__)
            allocation['idle_references'] = {'site': {'path': '/site', 'sha256': 'a' * 64},
                                              'safety': {'path': '/safe', 'sha256': 'b' * 64}}
            path = root / 'allocation.json'; path.write_text(json.dumps(allocation))
            args = argparse.Namespace(allocation=path, allocation_sha=budget.sha(path), owner='owner',
                                      operation='run', script=Path('/fixture/test.py'), script_args=[])
            child = Mock(pid=987); child.poll.side_effect = [None, 3]
            states = [{'threads': 1, 'runnable_threads': 1, 'live_pids': [987], 'affinity_violations': []},
                      {'threads': 0, 'runnable_threads': 0, 'live_pids': [], 'affinity_violations': []}]
            environment = {'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': '', 'ROCR_VISIBLE_DEVICES': ''}
            with patch.dict(os.environ, environment, clear=True), \
                 patch.object(budget, 'load_resource_policy', return_value=policy), \
                 patch.object(budget, 'measure', return_value=resources), \
                 patch.object(budget, 'identity', return_value=self.identity), \
                 patch.object(budget.time, 'time', return_value=1001), \
                 patch.object(budget.time, 'sleep'), patch.object(budget.os, 'sched_setaffinity'), \
                 patch.object(budget.os, 'open', return_value=12345), patch.object(budget.os, 'close'), \
                 patch.object(budget.fcntl, 'flock'), patch.object(budget.subprocess, 'Popen', return_value=child) as spawn, \
                 patch.object(budget, 'snapshot_tree', side_effect=states), \
                 patch.object(budget, 'read_idle_reference', side_effect=[{'live': self.live}, ValueError('GPU busy')]), \
                 patch.object(budget.os, 'killpg') as terminate, patch('builtins.print'):
                self.assertEqual(budget.run_task(args), 2)
                terminate.assert_called_once_with(987, budget.signal.SIGTERM)
                self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            receipt_path, = root.glob('tasks/owner-*/CPU-RECEIPT.json')
            receipt = json.loads(receipt_path.read_text())
            self.assertTrue(receipt['signals_sent'])
            self.assertEqual(receipt['signals_scope'], 'own_cpu_task_process_group')
            self.assertEqual(receipt['idle_interruption']['message'], 'GPU busy')


if __name__ == '__main__':
    unittest.main()
