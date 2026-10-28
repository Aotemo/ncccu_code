"""Independent fixtures reject stale deployment, incomplete GPU proof and second activation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/activation'), str(ROOT / 'runtime/bundle')]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools/activation' / filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


activation = load('activation_binding_under_test', 'activate.py')
preflight = load('preflight_binding_under_test', 'site_preflight.py')


class ActivationBindingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prep = self.root / 'technical-prep/fixture'; self.package = self.prep / 'package'
        (self.package / 'bundle').mkdir(parents=True)
        self.site = {'schema_version': 1, 'scope': 'formal_exp_v3', 'round_id': 'activation-structural-fixture',
            'git_commit': 'a' * 40, 'work_root': str(self.root), 'round_root': str(self.root / 'rounds/fixture'),
            'pod': {'namespace': 'fixture', 'name': 'fixture-pod', 'uid': '11111111-2222-4333-8444-555555555555',
                    'container': 'fixture-container', 'python': sys.executable},
            'gpus': [{'ordinal': n, 'pci': '0000:%02x:00.0' % (n + 1), 'vram_total_bytes': 4096} for n in range(8)],
            'science': {'candidate': {'id': 'structural-fixture-no-model'}, 'binding_version': 'fixture'}}
        for key in ('metadata_root', 'data_root', 'train_identity', 'prepared_cache',
                    'official_manifest', 'public_weight', 'public_provenance'):
            self.site[key] = str(self.root / 'assets' / key)
        self.site['cpu_execution'] = {'owner_id': 'fixture-owner', 'effective_cpus': 120, 'total_fraction_cap': '4/5',
            'owner_mask': list(range(24, 120)), 'control_mask': list(range(112, 120)), 'validation_mask': list(range(24)),
            'rank_masks': [list(range(24 + 11 * rank, 35 + 11 * rank)) for rank in range(8)], 'rank_threads': 1,
            'interop_threads': 1, 'loader_workers': 1,
            'allocation_receipt': {'path': str(self.root / 'fixture-allocation.json'), 'sha256': '0' * 64}}
        policy_path = self.root / 'RESOURCE-POLICY.json'
        policy_path.write_bytes((ROOT / 'configs/RESOURCE-POLICY.json').read_bytes())
        self.site['cpu_execution']['resource_policy'] = {'path': str(policy_path),
            'sha256': activation.sha(policy_path), 'value': json.loads(policy_path.read_text())}
        (self.package / 'bundle/fixture.py').write_text('# independent source fixture; never executed\n')
        for name in ('activate.py', 'activate_audit.py', 'site_preflight.py'):
            (self.prep / name).write_text('# independent activation source fixture; never executed\n')
        (self.package / 'SITE_BINDING.json').write_text(json.dumps(self.site))
        (self.package / 'CLOCK_POLICY.json').write_text(json.dumps({'round_id': self.site['round_id'], 'total_seconds': 100}))
        self.bundle = {'fixture.py': activation.sha(self.package / 'bundle/fixture.py')}
        self.source = hashlib.sha256(json.dumps(self.bundle, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        (self.package / 'SOURCE-MANIFEST.reference.json').write_text(json.dumps({'after': self.bundle, 'source_digest': self.source}))
        self.manifest = {'schema_version': 1, 'git_commit': self.site['git_commit'],
            'formal_root': self.site['round_root'], 'package_root': str(self.package),
            'files': {str(path.relative_to(self.package)): activation.sha(path)
                      for path in self.package.rglob('*') if path.is_file()},
            'bundle_source_digest': self.source, 'activate_script_sha256': activation.sha(self.prep / 'activate.py')}

    def test_imports_are_read_only_and_do_not_call_subprocesses(self):
        before = sorted(str(path) for path in self.root.rglob('*'))
        with patch('subprocess.run') as run:
            for name in ('activate.py', 'activate_audit.py', 'site_preflight.py'):
                module = load('import_only_' + name.replace('.', '_'), name)
                self.assertTrue(callable(module.main))
            run.assert_not_called()
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob('*')))

    def test_activation_refuses_existing_root_clock_failure_and_broken_symlink(self):
        formal = Path(self.site['round_root'])
        activation.ensure_new_activation(self.prep, formal)
        formal.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            activation.ensure_new_activation(self.prep, formal)
        formal.rmdir()
        for name in ('FORMAL-CLOCK.json', 'ACTIVATION-FAILURE.json'):
            marker = self.prep / name
            marker.write_text('{}')
            with self.subTest(marker=name), self.assertRaises(FileExistsError):
                activation.ensure_new_activation(self.prep, formal)
            marker.unlink(); marker.symlink_to(self.root / 'nonexistent-target')
            with self.subTest(symlink=name), self.assertRaises(FileExistsError):
                activation.ensure_new_activation(self.prep, formal)
            marker.unlink()

    def test_package_rejects_changed_extra_and_previous_execution_files(self):
        activation.verify_package(self.package, self.manifest, self.site, activation_script=self.prep / 'activate.py')
        extra = self.package / 'unadmitted.py'; extra.write_text('# extra')
        with self.assertRaises(ValueError):
            activation.verify_package(self.package, self.manifest, self.site, activation_script=self.prep / 'activate.py')
        extra.unlink()
        source = self.package / 'bundle/fixture.py'; source.write_text('# changed')
        with self.assertRaises(ValueError):
            activation.verify_package(self.package, self.manifest, self.site, activation_script=self.prep / 'activate.py')

    def test_remote_commit_and_actual_frozen_source_files_are_both_required(self):
        source = self.root / 'frozen-source'; source.mkdir()
        files = {}
        for relative, digest in self.manifest['files'].items():
            if not relative.endswith('.py'):
                continue
            target = source / 'runtime' / relative; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.package / relative).read_bytes()); files['runtime/' + relative] = digest
        for name in ('activate.py', 'activate_audit.py', 'site_preflight.py'):
            target = source / 'tools/activation' / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.prep / name).read_bytes()); files['tools/activation/' + name] = activation.sha(target)
        deployment = {'schema_version': 1, 'status': 'frozen_from_remote_commit', 'scope': 'formal_exp_v3',
            'git_commit': self.site['git_commit'], 'remote_commit': self.site['git_commit'],
            'remote_url': 'https://github.com/Aotemo/ncccu_code', 'pod_uid': self.site['pod']['uid'],
            'source_root': str(source), 'files': files, 'package_manifest_sha256': '1' * 64}
        activation.verify_deployment(deployment, self.site, self.manifest, '1' * 64, self.package, self.prep)
        for key, value in (('scope', 'cpu_technical_prep'), ('remote_commit', 'b' * 40), ('pod_uid', 'old-pod')):
            wrong = {**deployment, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                activation.verify_deployment(wrong, self.site, self.manifest, '1' * 64, self.package, self.prep)
        (source / 'runtime/bundle/fixture.py').write_text('# changed source of same frozen commit')
        with self.assertRaises(ValueError):
            activation.verify_deployment(deployment, self.site, self.manifest, '1' * 64, self.package, self.prep)

    def gpu_receipt(self):
        proofs = [{'rank': rank, 'pid': 100 + rank, 'world_size': 8, 'device': 'cuda:' + str(rank),
            'physical_pci': card['pci'], 'total_memory_bytes': 4096,
            'CPU_only_software_proof': False, 'intraop_threads': 1, 'interop_threads': 1,
            'requested_cpu_ids': self.site['cpu_execution']['rank_masks'][rank],
            'actual_cpu_ids': self.site['cpu_execution']['rank_masks'][rank]}
            for rank, card in enumerate(self.site['gpus'])]
        cpu = {'initial_state_control': {'model_tensor_sha256': '2' * 64, 'optimizer_state_entries': 0}}
        value = {'schema_version': 1, 'status': 'completed', 'exit_code': 0,
            'git_commit': self.site['git_commit'], 'pod_uid': self.site['pod']['uid'],
            'package_manifest_sha256': '1' * 64, 'bundle_source_digest': self.source,
            'candidate': self.site['science']['candidate'], 'successful_updates': 0, 'wall_seconds': 30,
            'all_processes_exited': True, 'live_processes': [], 'fit_only': True,
            'dev_evaluated': False, 'official_evaluated': False, 'fresh_initial_state_verified': True,
            'fp32_collective_verified': True, 'finite_values_verified': True,
            'initial_state_control': cpu['initial_state_control'], 'rank_proofs': proofs,
            'released_processes': [{'pid': pid, 'start_ticks': '123', 'argv': ['fixture-never-executed']}
                                   for pid in range(99, 108)]}
        return value, cpu

    def test_gpu_admission_requires_bounded_fit_only_fresh_eight_rank_proof(self):
        value, cpu = self.gpu_receipt()
        activation.verify_gpu_admission(value, self.site, manifest_sha='1' * 64, source_digest=self.source, cpu=cpu)
        for key, changed in (('successful_updates', 3), ('wall_seconds', 301), ('wall_seconds', float('nan')),
                ('dev_evaluated', True), ('official_evaluated', True), ('all_processes_exited', False),
                ('rank_proofs', value['rank_proofs'][:7]), ('released_processes', value['released_processes'][:8]),
                ('candidate', {'id': 'another-model'}), ('git_commit', 'b' * 40),
                ('initial_state_control', {'optimizer_state_entries': 1})):
            wrong = copy.deepcopy(value); wrong[key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                activation.verify_gpu_admission(wrong, self.site, manifest_sha='1' * 64, source_digest=self.source, cpu=cpu)
        for key, changed in (('CPU_only_software_proof', True), ('intraop_threads', 2), ('interop_threads', 2)):
            wrong = copy.deepcopy(value); wrong['rank_proofs'][0][key] = changed
            with self.subTest(rank_proof_field=key), self.assertRaises(ValueError):
                activation.verify_gpu_admission(wrong, self.site, manifest_sha='1' * 64, source_digest=self.source, cpu=cpu)

    def test_preflight_freshness_and_remaining_declared_lease_are_rechecked(self):
        value = {'schema_version': 1, 'exit_code': 0, 'status': 'site_identity_dependencies_and_declared_lease_checked',
            'Ready': True, 'pod_uid': self.site['pod']['uid'], 'round_id': self.site['round_id'],
            'site_binding_sha256': activation.sha(self.package / 'SITE_BINDING.json'),
            'clock_policy_sha256': activation.sha(self.package / 'CLOCK_POLICY.json'),
            'script_sha256': activation.sha(self.prep / 'site_preflight.py'), 'python': sys.executable,
            'observed_at': 100, 'CUDA_initialized': False, 'GPU_started': False, 'formal_T0_created': False,
            'training_lock_acquired': False, 'lease_annotations': {}, 'required_seconds': 100,
            'declared_lease_limits': [{'deadline_at': 202}]}
        activation.verify_preflight(value, self.site, self.package, now=101)
        for now in (103, 221):
            with self.subTest(now=now), self.assertRaises(ValueError):
                activation.verify_preflight(value, self.site, self.package, now=now)
        wrong = {**value, 'site_binding_sha256': '0' * 64}
        with self.assertRaises(ValueError):
            activation.verify_preflight(wrong, self.site, self.package, now=101)

    def test_lease_metadata_keeps_actual_owner_and_declared_deadline(self):
        pod = {'metadata': {'name': 'fixture', 'uid': 'pod', 'ownerReferences': [{'uid': 'owner'}]},
               'spec': {'activeDeadlineSeconds': 300}, 'status': {'startTime': '1970-01-01T00:00:00Z'}}
        limits, annotations = preflight.declared_limits(pod, None, now=100, required_seconds=200)
        self.assertEqual(limits[0]['deadline_at'], 300)
        self.assertEqual(annotations, {})
        with self.assertRaises(ValueError):
            preflight.declared_limits(pod, None, now=101, required_seconds=200)
        with self.assertRaises(ValueError):
            preflight.declared_limits(pod, {'metadata': {'name': 'wrong', 'uid': 'wrong'}}, now=100, required_seconds=200)

    def test_recycled_pid_is_distinct_but_same_start_ticks_stays_live(self):
        expected = {'pid': 1, 'start_ticks': '100', 'argv': ['old']}
        self.assertTrue(activation.process_alive(expected, {'state': 'S', 'start_ticks': '100'}))
        self.assertFalse(activation.process_alive(expected, {'state': 'S', 'start_ticks': '101'}))
        self.assertFalse(activation.process_alive(expected, {'state': 'Z', 'start_ticks': '100'}))


if __name__ == '__main__':
    unittest.main()
