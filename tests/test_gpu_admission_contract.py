"""Standard-library fixtures only: never initialize a model/GPU or publish GPU success."""
import builtins
import copy
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/validation/gpu_admission.py'


def load_tool():
    spec = importlib.util.spec_from_file_location('zero_work_admission_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class FakeOwnedChild:
    def __init__(self, pid, *, code=None, wait_error=None):
        self.pid = pid
        self.code = code
        self.wait_error = wait_error
        self.wait_calls = []

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.wait_error:
            raise self.wait_error
        if self.code is None:
            raise subprocess_timeout(self.pid, timeout)
        return self.code


def subprocess_timeout(pid, timeout):
    import subprocess
    return subprocess.TimeoutExpired(['fixture-child', str(pid)], timeout)


class GpuAdmissionContractTest(unittest.TestCase):
    def setUp(self):
        self.api = load_tool()
        self.temp = tempfile.TemporaryDirectory(dir=ROOT); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expected = {'candidate': {'id': 'independent-structural-fixture'}, 'source_digest': '2' * 64,
            'model_tensor_sha256': '3' * 64, 'optimizer_state_entries': 0,
            'batch_norm_modules': 0, 'buffer_count': 0, 'no_attention_q': True,
            'candidate_normalization': {'fixture': 'no model constructed'}, 'trainable_tensors': 183,
            'parameter_elements_by_role': {'backbone': 27820128, 'head': 769, 'age': 2}}
        self.site = {'scope': 'formal_exp_v3', 'schema_version': 1, 'git_commit': 'a' * 40,
            'round_id': 'independent-fixture', 'work_root': str(self.root),
            'round_root': str(self.root / 'rounds/fixture'), 'pod': {'uid': 'fixture-pod-uid'},
            'science': {'candidate': self.expected['candidate']},
            'gpus': [{'ordinal': n, 'pci': '0000:%02x:00.0' % (n + 1), 'vram_total_bytes': 4096} for n in range(8)],
            'cpu_execution': {'rank_masks': [[n] for n in range(8)]}}
        self.manifest = {'schema_version': 1, 'git_commit': 'a' * 40, 'bundle_source_digest': '2' * 64}
        self.deployment = {'schema_version': 1, 'status': 'frozen_from_remote_commit', 'scope': 'formal_exp_v3',
            'git_commit': 'a' * 40, 'remote_commit': 'a' * 40, 'remote_url': 'https://github.com/Aotemo/ncccu_code',
            'pod_uid': 'fixture-pod-uid', 'package_manifest_sha256': '1' * 64}
        self.preparation = {'status': 'prepared_no_formal_clock_no_GPU', 'exit_code': 0,
            'source_digest': '2' * 64, 'package_manifest_sha256': '1' * 64,
            'initial_state_control': self.expected, 'cpu_receipt_sha256': '4' * 64,
            'GPU_started': False, 'formal_clock_created': False}
        self.cpu = {'status': 'completed', 'exit_code': 0, 'source_digest': '2' * 64,
            'initial_state_control': self.expected, 'gpu_initialized': False, 'formal_train_config_sha256': None,
            'checks': [{'name': 'fixture-only-contract', 'passed': True}], 'original_receipt_sha256': '4' * 64,
            'training_updates': 0, 'data_images_read': False, 'official_semantics_read': False,
            'formal_clock_created': False, 'training_lock_acquired': False,
            'topology': {'world_size': 8, 'per_rank_batch': 8, 'accumulation': 1,
                         'reference_physical_batch': 32, 'global_effective_batch': 64, 'global_positive_quota': 8}}

    def validate(self):
        return self.api.validate_contract(self.site, self.manifest, self.deployment,
            self.preparation, self.cpu, manifest_sha='1' * 64)

    def test_import_has_no_torch_subprocess_or_output_side_effect(self):
        original_import = builtins.__import__
        before = sorted(str(path) for path in self.root.rglob('*'))
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'torchvision', 'numpy'):
                raise AssertionError('numeric dependency imported by the outer auditor')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded_import), \
             patch('subprocess.Popen', side_effect=AssertionError('import spawned work')):
            module = load_tool()
            self.assertTrue(callable(module.main))
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob('*')))

    def test_technical_scope_and_uncommitted_snapshot_are_rejected(self):
        self.validate()
        for mutation in ('technical-scope', 'uncommitted'):
            wrong = copy.deepcopy(self.site)
            if mutation == 'technical-scope':
                wrong['scope'] = 'cpu_technical_prep'
            else:
                wrong['source_state'] = {'kind': 'uncommitted_cpu_snapshot'}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.api.validate_contract(wrong, self.manifest, self.deployment,
                    self.preparation, self.cpu, manifest_sha='1' * 64)

    def test_wrong_remote_commit_package_pin_pod_and_missing_rank_are_rejected(self):
        for key, value in (('remote_commit', 'b' * 40), ('scope', 'cpu_technical_prep'),
                           ('package_manifest_sha256', '0' * 64), ('pod_uid', 'old-pod')):
            wrong = {**self.deployment, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.validate_contract(self.site, self.manifest, wrong,
                    self.preparation, self.cpu, manifest_sha='1' * 64)
        wrong = {**self.site, 'gpus': self.site['gpus'][:7]}
        with self.assertRaises(ValueError):
            self.api.validate_contract(wrong, self.manifest, self.deployment,
                self.preparation, self.cpu, manifest_sha='1' * 64)

    def test_missing_or_wrong_cpu_initial_identity_is_rejected(self):
        for key, value in (('initial_state_control', {}), ('source_digest', '0' * 64),
                           ('gpu_initialized', True), ('checks', []), ('training_updates', 1)):
            wrong = {**self.cpu, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.validate_contract(self.site, self.manifest, self.deployment,
                    self.preparation, wrong, manifest_sha='1' * 64)

    def test_the_executed_admission_tool_itself_must_match_frozen_git_files(self):
        source = self.root / 'source'; entry = source / 'tools/validation/gpu_admission.py'
        entry.parent.mkdir(parents=True); entry.write_text('# independent text fixture; never executed\n')
        deployment = {'source_root': str(source),
            'files': {'tools/validation/gpu_admission.py': self.api.sha(entry)}}
        self.api.verify_entry(deployment, entry, self.root)
        entry.write_text('# changed frozen text fixture\n')
        with self.assertRaises(ValueError):
            self.api.verify_entry(deployment, entry, self.root)

    def test_any_update_forward_sample_or_evaluation_breaks_zero_work_contract(self):
        value = {'successful_updates': 0, 'attempted_updates': 0, 'optimizer_steps': 0,
            'forward_calls': 0, 'data_samples_read': 0, 'data_images_read': False,
            'csv_read': False, 'cache_read': False, 'dev_evaluated': False, 'official_evaluated': False}
        self.api.validate_zero_work(value)
        for key in value:
            changed = {**value, key: 1 if type(value[key]) is int else True}
            with self.subTest(counter=key), self.assertRaises(ValueError):
                self.api.validate_zero_work(changed)
        with self.assertRaises(ValueError):
            self.api.validate_zero_work({})

    def test_attempt_directory_is_exclusive_and_cannot_escape_work_root(self):
        output = self.root / 'technical-prep/independent-fixture'
        self.api.reserve_output(output, self.root)
        self.assertTrue(output.is_dir())
        with self.assertRaises(FileExistsError):
            self.api.reserve_output(output, self.root)
        with self.assertRaises(ValueError):
            self.api.reserve_output(self.root / 'rounds/invalid', self.root)
        broken = self.root / 'technical-prep/broken'
        broken.symlink_to(self.root / 'does-not-exist')
        with self.assertRaises((ValueError, FileExistsError)):
            self.api.reserve_output(broken, self.root)
        self.assertFalse((output / 'GPU-ADMISSION.json').exists())

    def test_live_owner_missing_exit_rank_or_claimed_auditor_prevents_success(self):
        identities = [{'pid': pid, 'start_ticks': '100', 'argv': ['fixture-never-executed']}
                      for pid in range(10, 19)]
        exits = {'rank-' + str(n): 0 for n in range(8)}
        observations = {item['pid']: None for item in identities}
        self.api.verify_exits(0, exits, identities, observations, auditor_pid=99)
        cases = [
            (None, exits, identities, observations),
            (0, {**exits, 'rank-7': 1}, identities, observations),
            (0, {key: code for key, code in exits.items() if key != 'rank-7'}, identities, observations),
            (0, exits, identities, {**observations, 10: {'pid': 10, 'start_ticks': '100',
                'argv': ['fixture-never-executed'], 'state': 'S'}}),
            (0, exits, identities, {**observations, 10: {'pid': 10, 'start_ticks': '100',
                'argv': ['unexpected-exec-still-the-same-process'], 'state': 'S'}}),
            (0, exits, identities, {**observations, 10: {'pid': 10, 'start_ticks': '100',
                'argv': [], 'state': 'Z'}}),
            (0, exits, identities + [{'pid': 99, 'start_ticks': '100', 'argv': ['auditor']}], observations),
        ]
        for owner_exit, rank_exits, claimed, actual in cases:
            with self.subTest(owner_exit=owner_exit, rank_exits=rank_exits), self.assertRaises(ValueError):
                self.api.verify_exits(owner_exit, rank_exits, claimed, actual, auditor_pid=99)

    def test_owner_term_only_sets_flag_then_reaps_every_owned_rank(self):
        output = self.root / 'owner-term-fixture'; output.mkdir()
        handlers = {}; children = []; events = []; now = [100.]
        def register(signum, handler):
            old = handlers.get(signum, signal.SIG_DFL); handlers[signum] = handler
            return old
        def launch(_argv, name, _output, _sha, *, role, rank):
            process = FakeOwnedChild(1000 + rank)
            child = {'name': name, 'process': process, 'identity': {'pid': process.pid,
                'start_ticks': '100', 'argv': ['fixture-child', str(rank)]}}
            children.append(child)
            if rank == 7:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
                events.append(('handler-returned', all(item['process'].code is None for item in children)))
            return child
        def notify(expected, signum):
            events.append(('signal', expected['pid'], signum))
            next(item['process'] for item in children if item['identity']['pid'] == expected['pid']).code = -signum
            return {'pid': expected['pid'], 'signal': signum, 'sent': True}
        spec = {'output': str(output), 'deadline_at': 340., 'hard_deadline_at': 400.}
        context = {'api': SimpleNamespace(bind_cpu_role=lambda *_a: None), 'site': {}}
        with patch.object(self.api.signal, 'signal', side_effect=register), \
             patch.object(self.api, 'launch_child', side_effect=launch), \
             patch.object(self.api, 'signal_exact', side_effect=notify), \
             patch.object(self.api, 'audit_owned_tree', return_value={'live_pids': []}), \
             patch.object(self.api, 'identity', return_value={'pid': 900, 'start_ticks': '50', 'argv': ['fixture-owner']}), \
             patch.object(self.api.time, 'time', side_effect=lambda: now[0]), \
             patch.object(self.api.time, 'sleep', side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
            code = self.api.run_owner(SimpleNamespace(spec='unused.json', spec_sha256='0' * 64), spec, context)
        result = json.loads((output / 'OWNER-RESULT.json').read_text())
        self.assertEqual(events[0], ('handler-returned', True))
        self.assertEqual(code, 1)
        self.assertEqual(result['reason'], 'signal_15')
        self.assertEqual(result['exit_codes'], {'rank-' + str(rank): -15 for rank in range(8)})
        self.assertTrue(all(result['reaped'].values()))
        self.assertTrue(all(child['process'].wait_calls for child in children))
        self.assertEqual(result['live_owned_children'], [])
        self.assertNotEqual(result['status'], 'completed')
        self.assertFalse((output / 'GPU-ADMISSION.json').exists())

    def test_cpu_guard_reason_and_nonzero_exit_codes_survive_drain(self):
        children = [{'name': 'rank-' + str(rank), 'process': FakeOwnedChild(2000 + rank),
            'identity': {'pid': 2000 + rank, 'start_ticks': '100', 'argv': ['fixture-rank', str(rank)]}}
            for rank in range(2)]
        now = [100.]
        def notify(expected, signum):
            child = next(item for item in children if item['identity']['pid'] == expected['pid'])
            child['process'].code = 1 if child['name'] == 'rank-0' else -signum
            return {'pid': expected['pid'], 'signal': signum, 'sent': True}
        with patch.object(self.api, 'audit_owned_tree', side_effect=ValueError('fixture affinity escape')), \
             patch.object(self.api, 'signal_exact', side_effect=notify), \
             patch.object(self.api.time, 'time', side_effect=lambda: now[0]), \
             patch.object(self.api.time, 'sleep', side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
            result = self.api.wait_children(children, 340., {'site': {}})
        self.assertIn('CPU_tree_unknown:', result['reason'])
        self.assertIn('fixture affinity escape', result['reason'])
        self.assertEqual(result['exit_codes'], {'rank-0': 1, 'rank-1': -15})
        self.assertEqual(result['reaped'], {'rank-0': True, 'rank-1': True})
        self.assertEqual(result['status'], 'failed')

    def test_failed_owner_launch_still_waits_for_its_registered_children(self):
        output = self.root / 'owner-launch-failure'; output.mkdir()
        children = []; now = [100.]
        def launch(_argv, name, _output, _sha, *, role, rank):
            if rank == 3:
                raise RuntimeError('fixture launch failed')
            child = {'name': name, 'process': FakeOwnedChild(3000 + rank),
                'identity': {'pid': 3000 + rank, 'start_ticks': '100', 'argv': ['fixture-rank', str(rank)]}}
            children.append(child)
            return child
        def notify(expected, signum):
            next(item['process'] for item in children if item['identity']['pid'] == expected['pid']).code = -signum
            return {'pid': expected['pid'], 'signal': signum, 'sent': True}
        with patch.object(self.api.signal, 'signal', return_value=signal.SIG_DFL), \
             patch.object(self.api, 'launch_child', side_effect=launch), \
             patch.object(self.api, 'signal_exact', side_effect=notify), \
             patch.object(self.api, 'audit_owned_tree', return_value={'live_pids': []}), \
             patch.object(self.api, 'identity', return_value={'pid': 900, 'start_ticks': '50', 'argv': ['fixture-owner']}), \
             patch.object(self.api.time, 'time', side_effect=lambda: now[0]), \
             patch.object(self.api.time, 'sleep', side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
            code = self.api.run_owner(SimpleNamespace(spec='unused.json', spec_sha256='0' * 64),
                {'output': str(output), 'deadline_at': 340., 'hard_deadline_at': 400.},
                {'api': SimpleNamespace(bind_cpu_role=lambda *_a: None), 'site': {}})
        result = json.loads((output / 'OWNER-RESULT.json').read_text())
        self.assertEqual(code, 1)
        self.assertIn('fixture launch failed', result['reason'])
        self.assertEqual(result['exit_codes'], {'rank-0': -15, 'rank-1': -15, 'rank-2': -15})
        self.assertTrue(all(child['process'].wait_calls for child in children))
        self.assertTrue(all(result['reaped'].values()))
        self.assertNotEqual(result['status'], 'completed')

    def test_unreaped_child_is_unknown_and_recycled_pid_is_never_signalled(self):
        child = {'name': 'rank-0', 'process': FakeOwnedChild(4000, code=0, wait_error=ChildProcessError('not waitable')),
            'identity': {'pid': 4000, 'start_ticks': '100', 'argv': ['fixture-rank']}}
        with patch.object(self.api, 'audit_owned_tree', return_value={'live_pids': []}):
            result = self.api.wait_children([child], 340., {'site': {}})
        self.assertEqual(result['status'], 'resource_unknown')
        self.assertFalse(result['reaped']['rank-0'])
        self.assertIn('not waitable', result['reap_errors']['rank-0'])
        with patch.object(self.api, 'identity', return_value={'pid': 4000, 'start_ticks': '101',
                'argv': ['unrelated-process'], 'state': 'S'}), \
             patch.object(self.api.os, 'kill', side_effect=AssertionError('an unrelated PID was signalled')):
            notification = self.api.signal_exact(child['identity'], signal.SIGTERM)
        self.assertFalse(notification['sent'])
        now = [100.]
        stuck = {'name': 'rank-0', 'process': FakeOwnedChild(5000),
            'identity': {'pid': 5000, 'start_ticks': '100', 'argv': ['fixture-stuck-rank']}}
        with patch.object(self.api, 'audit_owned_tree', return_value={'live_pids': []}), \
             patch.object(self.api, 'signal_exact', side_effect=lambda expected, signum:
                 {'pid': expected['pid'], 'signal': signum, 'sent': False}), \
             patch.object(self.api.time, 'time', side_effect=lambda: now[0]), \
             patch.object(self.api.time, 'sleep', side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
            bounded = self.api.wait_children([stuck], 340., {'site': {}},
                initial_reason='fixture_stop', until=110.)
        self.assertEqual(bounded['status'], 'resource_unknown')
        self.assertFalse(bounded['reaped']['rank-0'])
        self.assertLessEqual(now[0], 110.1)
        self.assertEqual([event['signal'] for event in bounded['signals']], [signal.SIGTERM, signal.SIGKILL])


if __name__ == '__main__':
    unittest.main()
