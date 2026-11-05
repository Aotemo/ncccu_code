"""Synthetic-only owner safety and producer/consumer OOF contract checks.

The source under test is supplied via NCCCU_TEST_SOURCE. No real checkpoint,
image, score, Pod command, accelerator, or official input is used here.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import sys

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1])).resolve()
sys.path[:0] = [str(ROOT / 'runtime/bundle'), str(ROOT / 'tools/diagnostics'), str(ROOT / 'tests')]
import torch
from retrain import f05_folds
from retrain.parallel_evaluation import batch_assignments
from retrain.persistence import canonical_digest
from test_f05_oof_targets import synthetic_fixture, _write, _ref
import f05_oof_targets as consumer


def module_from_source(name, path):
    if not path.is_file():
        raise RuntimeError('required owner source missing: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu = module_from_source('f05_owner_gpu_test', ROOT / 'tools/diagnostics/f04_gpu_full_dev.py')
oof = module_from_source('f05_owner_oof_test', ROOT / 'tools/diagnostics/f05_oof_gpu.py')


def read(ref):
    return json.loads(Path(ref['path']).read_text())


def score(base, sealed, parent, packets, role):
    return oof.make_scores(base, sealed['folds'][parent['fold']], packets, fold=parent['fold'],
        role=role, round_id=parent['round_id'], inputs_sha=parent['producer_inputs']['sha256'],
        proof_shas={str(i): ref['sha256'] for i, ref in enumerate(parent['rank_proofs'])},
        selected=parent['selected'], checkpoint_sha=parent['checkpoint']['sha256'],
        base_manifest_sha=parent['base_manifest_sha256'], folds_sha=parent['folds_sha256'])


def fake_asset_fixture(root, local_root):
    """Small files plus padded fake role counts; role verifier is mocked separately."""
    root, local_root = Path(root), Path(local_root)
    round_id = 'r265-small-f05fold0-seed20261029-pairsigmoid-24k-p4-9010-20261105'
    round_root = root / 'rounds' / round_id; round_root.mkdir(parents=True)
    metadata = root / 'technical-prep/fold-meta'; metadata.mkdir(parents=True)
    bundle = round_root / 'bundle'; bundle.mkdir()
    (bundle / 'fixture.py').write_text('# synthetic runtime\n')
    base = _write(root, 'BASE.json', {'fit_indices': [0] * 46256, 'dev_indices': [1] * 5140})
    role = {'fit_indices_sha256': 'a' * 64, 'audit_indices_sha256': 'b' * 64}
    folds = _write(root, 'FOLDS.json', {'status': 'three_fold_fit_only_indices_sealed_no_training',
        'base_manifest_sha256': base['sha256'], 'seed': 20261121, 'training_started': False,
        'official_read': False, 'folds': [role, {}, {}]})
    training = _write(metadata, 'manifest.json', {'fixture': True})
    science = _write(root, 'technical-prep/SCIENCE.json', {'fixture': True})
    site_value = {'round_id': round_id, 'metadata_root': str(metadata),
                  'science': {'configuration': science}, 'data_root': '/synthetic/images',
                  'prepared_cache': '/synthetic/cache'}
    site = _write(round_root, 'SITE_BINDING.json', site_value)
    runtime = _write(round_root, 'RUNTIME_BINDING.json', {'site_binding_sha256': site['sha256']})
    checkpoint_path = round_root / 'outputs/job/run/checkpoints/checkpoint-toy.pt'
    checkpoint_path.parent.mkdir(parents=True); checkpoint_path.write_bytes(b'only synthetic checkpoint')
    checkpoint = _ref(checkpoint_path)
    sidecar = _write(checkpoint_path.parent, checkpoint_path.name + '.json', {'file': checkpoint_path.name,
        'format': 'retraining-container-checkpoint', 'format_version': 1, 'sha256': checkpoint['sha256']})
    local = local_root / checkpoint_path.name; local.write_bytes(checkpoint_path.read_bytes())
    Path(str(local) + '.json').write_bytes(Path(sidecar['path']).read_bytes())
    selected = {'variant': 'ema', 'tensor_sha256': 'c' * 64, 'successful_updates': 800}
    freeze = _write(round_root, 'SINGLE-MODEL-FREEZE.json', {'cpu_fixture_only': False,
        'precision': 'fp32', 'view': 'identity', 'selected': selected, 'blob_sha256': checkpoint['sha256'],
        'experiment91': {'manifest_sha256': training['sha256']}, 'source_files': {'fixture.py': _ref(bundle / 'fixture.py')['sha256']}})
    completion = _write(round_root, 'evidence/TRAIN_RESULT.json', {'status': 'completed', 'successful_updates': 800})
    safe = _write(root, 'SAFE.json', {'phase': 'terminal', 'formal_root': str(round_root),
        'done': {'present': True, 'status': 'completed'}, 'safety': {'safe': True},
        'first_success_receipt_read': True, 'signals_sent': False, 'model_forward_executed': False,
        'official_read': False, 'selected': selected})
    member = {'round_id': round_id, 'bundle': str(bundle), 'data_root': site_value['data_root'],
              'cache': site_value['prepared_cache'], 'site': site, 'freeze': freeze, 'selected': selected}
    inputs = {'fold': 0, 'manifest': base, 'folds': folds, 'training_manifest': training, 'site': site,
        'science': science, 'runtime_binding': runtime, 'freeze': freeze, 'completion': completion,
        'safe': safe, 'checkpoint': checkpoint, 'sidecar': sidecar, 'local_checkpoint': _ref(local),
        'selected': selected, 'members': [member], 'age_contract_sha256': 'd' * 64}
    roles = {'train': [0, 1], 'heldout': [2, 3]}
    bound = {'fold_number': 0, 'fit_indices_sha256': 'a' * 64, 'heldout_indices_sha256': 'b' * 64,
             'age_contract_sha256': 'd' * 64}
    return inputs, roles, bound


def synthetic_aggregate_fixture(root, fault=None):
    """Rebuild fake producer inputs, rank proofs and packets for the real aggregator."""
    root = Path(root)
    fixture = synthetic_fixture(root)
    source_parent = read(fixture['parents'][0])
    inputs = {**read(source_parent['producer_inputs']), 'manifest': fixture['base'], 'folds': fixture['folds'],
              'age_contract_sha256': source_parent['age_contract_sha256'],
              'members': [{'round_id': source_parent['round_id']}]}
    for name in ('training_manifest', 'site', 'science', 'runtime_binding', 'freeze', 'completion',
                 'safe', 'checkpoint', 'sidecar'):
        inputs[name] = source_parent[name]
    inputs_ref = _write(root, 'producer-inputs.json', inputs)
    output = root / 'fake-owner-output'; (output / 'proofs').mkdir(parents=True); (output / 'packets').mkdir()
    parent_process = {'pid': 100, 'start_ticks': '123'}
    spec_ref = _write(output, 'WORKER-SPEC.json', {'fixture_only': True, 'oof_mode': True,
        'inputs': inputs_ref, 'parent': parent_process})
    proofs = [_write(output, 'proofs/rank-' + str(rank) + '.json',
                     {'fixture_only': True, 'rank': rank, 'parent': parent_process}) for rank in range(8)]
    release = {'fixture_only': True, 'status': 'all_rank_device_proofs_passed', 'parent': parent_process,
               'spec_sha256': spec_ref['sha256'],
               'proof_sha256': {str(rank): ref['sha256'] for rank, ref in enumerate(proofs)}}
    if fault == 'release': release['proof_sha256']['0'] = 'f' * 64
    _write(output, 'RELEASE.json', release)
    for role in ('train', 'heldout'):
        for rank, old_ref in enumerate(source_parent[role + '_packets']):
            packet = read(old_ref); packet['input_sha256'] = inputs_ref['sha256']
            packet['proof_sha256'] = proofs[rank]['sha256']
            if fault == 'strict' and role == 'heldout' and rank == 0:
                packet['model_load']['strict_state_load'] = False
            if fault == 'missing_row' and role == 'heldout' and rank == 0:
                packet['records'].pop()
            _write(output, 'packets/rank-' + str(rank) + '-' + role + '.json', packet)
    return fixture, inputs, inputs_ref, source_parent, output


def synthetic_prepare_fixture(root, local_root, uid, fault):
    """Build a self-contained fake work root for the owner's real prepare path."""
    root, local_root = Path(root), Path(local_root)
    existing = local_root / 'existing'; existing.mkdir()
    source, roles, bound = fake_asset_fixture(root, existing)
    base_ref = _write(root, 'base-meta/manifest.json', read(source['manifest']))
    folds = read(source['folds']); folds['base_manifest_sha256'] = base_ref['sha256']
    folds['folds'][0]['age'] = {'contract_sha256': source['age_contract_sha256']}
    folds_ref = _write(root, 'technical-prep/FOLDS.json', folds)
    science = {'f05_fold': {'fold': 1 if fault == 'fold' else 0, 'base_metadata_root': 'base-meta',
                            'sealed_folds_path': 'technical-prep/FOLDS.json'}}
    science_ref = _write(root, 'technical-prep/SCIENCE.json', science)
    round_root = Path(source['freeze']['path']).parent
    site = read(source['site']); site.update(work_root=str(root), science={'configuration': science_ref})
    site_ref = _write(round_root, 'SITE_BINDING.json', site)
    _write(round_root, 'RUNTIME_BINDING.json', {'site_binding_sha256': site_ref['sha256']})
    safe_ref = source['safe']
    if fault == 'unsafe':
        safe = read(safe_ref); safe['safety']['safe'] = False
        safe_ref = _write(root, 'SAFE.json', safe)
    checkpoint = Path(source['checkpoint']['path'])
    sidecar = Path(source['sidecar']['path'])
    if fault == 'missing':
        checkpoint.unlink(); sidecar.unlink()
    elif fault == 'duplicate':
        second = checkpoint.with_name('checkpoint-duplicate.pt'); second.write_bytes(checkpoint.read_bytes())
        metadata = read(source['sidecar']); metadata['file'] = second.name
        _write(second.parent, second.name + '.json', metadata)
    elif fault == 'source_blob':
        checkpoint.write_bytes(b'changed synthetic source bytes')
    elif fault == 'sidecar_name':
        metadata = read(source['sidecar']); metadata['file'] = '../other.pt'
        _write(sidecar.parent, sidecar.name, metadata)
    tree = _write(root, 'TREE.json', {'status': 'published_full_git_tree_verified_not_formal_deployment',
        'source_root': str(ROOT), 'commit': 'a' * 40, 'pod_uid': uid,
        'GPU_initialized': False, 'official_read': False})
    policy = json.loads((ROOT / 'configs/F05-OOF-GPU-20261105.json').read_text())
    policy.update(base_manifest_sha256='f' * 64 if fault == 'base_manifest' else base_ref['sha256'],
                  folds_sha256=folds_ref['sha256'])
    execution_site = {'pod': {'uid': uid}, 'work_root': str(root), 'data_root': site['data_root']}
    args = types.SimpleNamespace(source_root=ROOT, remote_metadata=root, git_commit='a' * 40,
        tree_proof=Path(tree['path']), tree_proof_sha256=tree['sha256'],
        execution_site=root / 'execution-site.json', execution_site_sha256='b' * 64,
        safe=Path(safe_ref['path']), safe_sha256=safe_ref['sha256'], fold=0,
        local_dir=local_root / 'prepared-copy', output=root / 'technical-prep/INPUTS.json')
    return args, policy, execution_site, roles, bound


class OwnerSafetyTest(unittest.TestCase):
    def test_selected_sidecar_discovery_rejects_missing_duplicate_and_untrusted_files(self):
        digest = 'a' * 64
        for fault in ('none', 'missing', 'duplicate', 'symlink', 'malformed', 'invalid_digest', 'orphan',
                      'scan_error', 'dirlink', 'blob_symlink', 'sidecar_only'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); outputs = root / 'outputs'; outputs.mkdir()
                folder = outputs / 'job/run/checkpoints'; folder.mkdir(parents=True)
                first = folder / 'checkpoint-one.pt.json'
                if fault != 'missing': (folder / 'checkpoint-one.pt').write_bytes(b'synthetic-only')
                if fault == 'symlink':
                    outside = root / 'outside.json'; outside.write_text(json.dumps({'sha256': digest}))
                    first.symlink_to(outside)
                elif fault == 'malformed':
                    first.write_text('{not valid json')
                elif fault != 'missing':
                    first.write_text(json.dumps({'sha256': digest}))
                if fault == 'duplicate':
                    (folder / 'checkpoint-two.pt.json').write_text(json.dumps({'sha256': digest}))
                    (folder / 'checkpoint-two.pt').write_bytes(b'synthetic-other')
                if fault == 'orphan':
                    (folder / 'checkpoint-two.pt').write_bytes(b'cannot-identify-without-sidecar')
                if fault == 'dirlink':
                    (outputs / 'linked-directory').symlink_to(folder, target_is_directory=True)
                if fault == 'blob_symlink':
                    blob = folder / 'checkpoint-one.pt'; blob.unlink()
                    target = root / 'external-blob.pt'; target.write_bytes(b'synthetic')
                    blob.symlink_to(target)
                if fault == 'sidecar_only':
                    (folder / 'checkpoint-one.pt').unlink()
                if fault == 'scan_error':
                    (outputs / 'unreadable').mkdir()
                    original_scandir = os.scandir
                    def scandir(path):
                        if Path(path).name == 'unreadable': raise PermissionError('synthetic enumeration failure')
                        return original_scandir(path)
                    with mock.patch.object(os, 'scandir', side_effect=scandir):
                        with self.assertRaises((ValueError, OSError)):
                            oof.selected_checkpoint_sidecar(outputs, digest)
                    continue
                if fault == 'none':
                    self.assertEqual(oof.selected_checkpoint_sidecar(outputs, digest), first)
                else:
                    with self.assertRaises((ValueError, TypeError, AttributeError, OSError)):
                        oof.selected_checkpoint_sidecar(outputs, 'bad' if fault == 'invalid_digest' else digest)

    def test_prepare_stops_before_writing_when_execution_uid_differs(self):
        import f04_bounded_fusion
        import f04_export_member
        from retrain import site_binding
        sys.path.insert(0, str(ROOT / 'tools/pretraining'))
        import source_chain
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uid = os.environ['NCCCU_POD_UID']
            tree = _write(root, 'TREE.json', {'status': 'published_full_git_tree_verified_not_formal_deployment',
                'source_root': str(ROOT), 'commit': 'a' * 40, 'pod_uid': uid,
                'GPU_initialized': False, 'official_read': False})
            args = types.SimpleNamespace(source_root=ROOT, remote_metadata=root, git_commit='a' * 40,
                tree_proof=Path(tree['path']), tree_proof_sha256=tree['sha256'],
                execution_site=root / 'site.json', execution_site_sha256='b' * 64)
            with mock.patch.object(f04_bounded_fusion, 'verify_blob', return_value='c' * 64), \
                 mock.patch.object(f04_export_member, 'verify_published', return_value={}), \
                 mock.patch.object(source_chain, 'cpu_context', return_value={'uid': uid}), \
                 mock.patch.object(site_binding, 'load_site', return_value={'pod': {'uid': uid + '-wrong'}}), \
                 mock.patch.object(gpu, 'write_new') as write_new:
                with self.assertRaisesRegex(ValueError, 'execution site'):
                    oof.prepare(gpu, args)
                write_new.assert_not_called()

    def test_sealed_inputs_bind_one_fold_selected_and_all_source_references(self):
        gib = 1024 ** 3
        policy = {'version': 'f05-fit-only-oof-gpu-20261105-v1', 'folds': 3,
            'rounds': ['R265', 'R266', 'R267'], 'population_order': ['train', 'heldout'],
            'world_size': 8, 'batch_size': 8, 'view': 'identity', 'model_mode': 'eval', 'storage': 'fp32',
            'original_fit_count': 46256, 'dev_count': 5140, 'minimum_free_bytes': 20 * gib,
            'evaluation_reserve_bytes': gib, 'tta': False, 'dev_read': False, 'official_read': False,
            'base_manifest_sha256': 'a' * 64, 'folds_sha256': 'b' * 64}
        round_id = 'r265-small-f05fold0-seed20261029-pairsigmoid-24k-p4-9010-20261105'
        selected = {'variant': 'ema', 'tensor_sha256': 'c' * 64, 'successful_updates': 800}
        refs = {key: {'path': '/synthetic/' + key, 'sha256': 'd' * 64} for key in
            ('manifest', 'folds', 'training_manifest', 'site', 'science', 'runtime_binding', 'freeze',
             'completion', 'safe', 'checkpoint', 'sidecar', 'local_checkpoint')}
        refs['manifest']['sha256'] = policy['base_manifest_sha256']
        refs['folds']['sha256'] = policy['folds_sha256']
        member = {'round_id': round_id, 'site': refs['site'], 'freeze': refs['freeze'], 'selected': selected}
        inputs = {**refs, 'status': 'F05_fit_only_oof_inputs_sealed_no_GPU', 'fold': 0,
            'members': [member], 'plan': oof.build_plan(round_id, 0), 'TTA': False, 'dev_read': False,
            'official_read': False, 'GPU_initialized': False, 'base_manifest_sha256': 'a' * 64,
            'folds_sha256': 'b' * 64, 'training_manifest_sha256': refs['training_manifest']['sha256'],
            'checkpoint_sha256': refs['checkpoint']['sha256'], 'selected': selected}
        oof.validate_inputs(inputs, policy)
        for fault in ('fold', 'unsafe', 'selected', 'checkpoint', 'local', 'member', 'plan', 'missing'):
            bad = copy.deepcopy(inputs)
            if fault == 'fold': bad['fold'] = 1
            if fault == 'unsafe': bad['official_read'] = True
            if fault == 'selected': bad['selected']['variant'] = 'unknown'
            if fault == 'checkpoint': bad['checkpoint_sha256'] = 'f' * 64
            if fault == 'local': bad['local_checkpoint']['sha256'] = 'f' * 64
            if fault == 'member': bad['members'].append(copy.deepcopy(bad['members'][0]))
            if fault == 'plan': bad['plan'][0]['rank'] = 1
            if fault == 'missing': del bad['safe']
            with self.assertRaises(ValueError): oof.validate_inputs(bad, policy)

    def test_standalone_xor_excludes_official_and_stale_predecessor(self):
        for soup, oof_mode in ((True, False), (False, True)):
            gpu.validate_standalone_options(True, soup, oof_mode, None, None, None, None)
        for args in ((True, False, False, None, None, None, None),
                     (True, True, True, None, None, None, None),
                     (True, False, True, '/official', None, None, None),
                     (True, True, False, None, None, '/official', None),
                     (True, False, True, None, 'f' * 64, None, None),
                     (1, False, True, None, None, None, None)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                gpu.validate_standalone_options(*args)
        safe = {'phase': 'terminal', 'formal_root': '/synthetic/round', 'observed_at': 100.,
                'first_success_receipt_read': True, 'done': {'present': True, 'status': 'completed'},
                'safety': {'safe': True}, 'signals_sent': False, 'model_forward_executed': False,
                'official_read': False}
        gpu.require_safe_predecessor(safe, '/synthetic/round', 110.)
        for changed, now in ((dict(safe, observed_at=-20.1), 110.),
                             (dict(safe, phase='running'), 110.),
                             (dict(safe, safety={'safe': False}), 110.),
                             (dict(safe, done={'present': False, 'status': 'completed'}), 110.),
                             (dict(safe, official_read=True), 110.),
                             (dict(safe, formal_root='/other'), 110.)):
            with self.assertRaises(ValueError): gpu.require_safe_predecessor(changed, '/synthetic/round', now)

    def test_peer_capacity_is_fresh_bound_and_reserves_running_peer(self):
        gib = 1024 ** 3
        peer_site = {'round_id': 'peer', 'round_root': '/synthetic/rounds/peer', 'pod': {'uid': 'other'}}
        peer = {'phase': 'running', 'formal_root': peer_site['round_root'], 'observed_at': 100.,
                'identity': {'owner_status': 'active', 'rank_status': ['active'] * 8},
                'signals_sent': False, 'model_forward_executed': False, 'official_read': False}
        window = {'status': 'f05_evaluation_capacity_verified', 'at': 100.,
            'predecessor_safe_sha256': 'a' * 64, 'site_sha256': 'b' * 64,
            'peer_observation_sha256': 'c' * 64, 'peer_site_sha256': 'd' * 64,
            'peer_round_id': 'peer', 'peer_reserved_bytes': 12 * gib,
            'peer_storage_limit_bytes': 12 * gib, 'required_free_bytes': 33 * gib,
            'free_bytes': 40 * gib, 'training_activated': False, 'GPU_started': False, 'official_read': False}
        kwargs = {'safe_sha': 'a' * 64, 'site_sha': 'b' * 64, 'peer_observation_sha': 'c' * 64,
                  'peer_site_sha': 'd' * 64, 'current_uid': 'current', 'now': 110.,
                  'free': 40 * gib, 'peer_limit': 12 * gib}
        oof.validate_evaluation_window(window, peer, peer_site, **kwargs)
        for w, p, s, options in (
            (dict(window, at=-11.), peer, peer_site, kwargs),
            (dict(window, peer_reserved_bytes=0), peer, peer_site, kwargs),
            (dict(window, peer_observation_sha256='f' * 64), peer, peer_site, kwargs),
            (dict(window, training_activated=True), peer, peer_site, kwargs),
            (window, dict(peer, observed_at=-11.), peer_site, kwargs),
            (window, dict(peer, formal_root='/wrong'), peer_site, kwargs),
            (window, dict(peer, identity={'owner_status': 'active', 'rank_status': ['active'] * 7}), peer_site, kwargs),
            (window, dict(peer, phase='terminal', done={'present': True, 'status': 'completed'},
                          safety={'safe': False}), peer_site, kwargs),
            (window, peer, dict(peer_site, pod={'uid': 'current'}), kwargs),
            (window, peer, peer_site, dict(kwargs, free=33 * gib))):
            with self.assertRaises(ValueError): oof.validate_evaluation_window(w, p, s, **options)

    def test_standalone_window_rechecks_peer_and_predecessor_runtime_bindings(self):
        import f04_bounded_fusion
        gib = 1024 ** 3
        for fault in ('none', 'peer_runtime', 'predecessor_runtime', 'peer_identity', 'storage_policy'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                previous = root / 'rounds/previous'; previous.mkdir(parents=True)
                peer_root = root / 'rounds/peer'; peer_root.mkdir()
                previous_site = _write(previous, 'SITE_BINDING.json', {'pod': {'uid': 'current'}})
                previous_runtime = {'site_binding_sha256': previous_site['sha256']}
                _write(previous, 'RUNTIME_BINDING.json', previous_runtime)
                peer_site_value = {'round_id': 'peer', 'round_root': str(peer_root),
                                   'pod': {'uid': 'other'}, 'git_commit': 'e' * 40}
                peer_site = _write(peer_root, 'SITE_BINDING.json', peer_site_value)
                _write(peer_root, 'RUNTIME_BINDING.json', {'site_binding_sha256': peer_site['sha256'],
                                                          'git_commit': peer_site_value['git_commit']})
                policy = _write(peer_root, 'STORAGE_POLICY.json', {'schema_version': 1,
                    'round_overrides': {}, 'default_round_storage_limit_bytes': 12 * gib})
                peer_value = {'phase': 'running', 'formal_root': str(peer_root), 'observed_at': 100.,
                    'identity': {'owner_status': 'active', 'rank_status': ['active'] * 8},
                    'signals_sent': False, 'model_forward_executed': False, 'official_read': False}
                peer = _write(root, 'peer-observation.json', peer_value)
                if fault == 'peer_runtime':
                    _write(peer_root, 'RUNTIME_BINDING.json', {'site_binding_sha256': 'f' * 64,
                                                              'git_commit': peer_site_value['git_commit']})
                elif fault == 'predecessor_runtime':
                    _write(previous, 'RUNTIME_BINDING.json', {'site_binding_sha256': 'f' * 64})
                elif fault == 'peer_identity':
                    peer_value['formal_root'] = '/wrong'
                    peer = _write(root, 'peer-observation.json', peer_value)
                elif fault == 'storage_policy':
                    policy = _write(peer_root, 'STORAGE_POLICY.json', {'schema_version': 1,
                        'round_overrides': {}, 'default_round_storage_limit_bytes': 0})
                window = {'status': 'f05_evaluation_capacity_verified', 'at': 100.,
                    'predecessor_safe_sha256': 'a' * 64, 'site_sha256': 'b' * 64,
                    'peer_observation_sha256': peer['sha256'], 'peer_site_sha256': peer_site['sha256'],
                    'peer_round_id': 'peer', 'peer_observation': peer, 'peer_site': peer_site,
                    'peer_storage_policy': policy, 'peer_reserved_bytes': 12 * gib,
                    'peer_storage_limit_bytes': 12 * gib, 'required_free_bytes': 33 * gib,
                    'free_bytes': 40 * gib, 'training_activated': False, 'GPU_started': False,
                    'official_read': False, 'git_commit': 'e' * 40}
                site = {'work_root': str(root), 'round_id': 'peer', 'round_root': str(peer_root),
                        'pod': {'uid': 'current'}}
                safe = {'formal_root': str(previous)}
                args = types.SimpleNamespace(remote_metadata=root, git_commit='e' * 40,
                                             safe_sha256='a' * 64, site_sha256='b' * 64)
                with mock.patch.object(f04_bounded_fusion, 'verify_blob'), \
                     mock.patch.object(gpu.time, 'time', return_value=110.), \
                     mock.patch.object(gpu.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=40 * gib)):
                    if fault == 'none':
                        gpu.validate_standalone_window(window, safe, site, args, ROOT)
                    else:
                        with self.assertRaises(ValueError):
                            gpu.validate_standalone_window(window, safe, site, args, ROOT)

    def test_eight_rank_supervision_and_release_are_identity_bound(self):
        parent = {'pid': 100, 'start_ticks': '77'}
        proofs = [{'rank': i, 'pid': 101 + i, 'start_ticks': str(200 + i), 'parent': parent,
                   'world_size': 8, 'device': f'cuda:{i}'} for i in range(8)]
        launches = [{'rank': i, 'pid': 101 + i, 'start_ticks': str(200 + i)} for i in range(8)]
        def identity(pid):
            rank = pid - 101
            return {'pid': pid, 'parent_pid': 100, 'start_ticks': str(200 + rank), 'active': True}
        device_check = mock.Mock()
        gpu.validate_proof_barrier(proofs, launches, parent, device_check, identity)
        self.assertEqual(device_check.call_count, 1)
        mutations = []
        bad = copy.deepcopy(proofs); bad[1]['rank'] = 0; mutations.append((bad, launches))
        bad = copy.deepcopy(proofs); bad[1]['device'] = 'cuda:0'; mutations.append((bad, launches))
        bad = copy.deepcopy(proofs); bad[1]['parent'] = {'pid': 999, 'start_ticks': '77'}; mutations.append((bad, launches))
        bad = copy.deepcopy(launches); bad[1]['start_ticks'] = '999'; mutations.append((proofs, bad))
        mutations.append((proofs[:-1], launches[:-1]))
        for p, l in mutations:
            with self.assertRaises(ValueError): gpu.validate_proof_barrier(p, l, parent, lambda _: None, identity)
        release = {'status': 'all_rank_device_proofs_passed', 'parent': parent, 'spec_sha256': 'b' * 64,
                   'proof_sha256': {str(i): ('%064x' % (i + 1)) for i in range(8)}}
        parent_identity = lambda pid: {'pid': pid, 'start_ticks': '77', 'active': True}
        gpu.validate_worker_release(release, 0, release['proof_sha256']['0'], 'b' * 64, parent, parent_identity)
        for bad in (dict(release, spec_sha256='c' * 64), dict(release, proof_sha256={'0': '1' * 64}),
                    dict(release, parent={'pid': 999, 'start_ticks': '77'})):
            with self.assertRaises(ValueError):
                gpu.validate_worker_release(bad, 0, release['proof_sha256']['0'], 'b' * 64, parent, parent_identity)

    def test_producer_scores_are_consumable_for_all_fold_roles_and_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = synthetic_fixture(directory)
            base, sealed = read(fixture['base']), read(fixture['folds'])
            seen_held = []
            for number, parent_ref in enumerate(fixture['parents']):
                parent = read(parent_ref)
                train = read(parent['training_manifest'])
                roles = oof.verified_roles(base, train, sealed['folds'][number], fold=number)
                self.assertFalse(set(roles['train']) & set(roles['heldout']))
                self.assertFalse((set(roles['train']) | set(roles['heldout'])) & set(base['dev_indices']))
                seen_held.extend(roles['heldout'])
                bad_role = copy.deepcopy(sealed['folds'][number])
                bad_role['audit_indices'] = sorted(bad_role['audit_indices'] + [roles['train'][0]])
                bad_role['audit_indices_sha256'] = canonical_digest(bad_role['audit_indices'])
                with self.assertRaises(ValueError):
                    oof.verified_roles(base, train, bad_role, fold=number)
                bad_train = copy.deepcopy(train); bad_train['dev_indices'].reverse()
                with self.assertRaises(ValueError):
                    oof.verified_roles(base, bad_train, sealed['folds'][number], fold=number)
                for role in ('train', 'heldout'):
                    packets = [read(ref) for ref in parent[role + '_packets']]
                    produced = score(base, sealed, parent, packets, role)
                    self.assertEqual(produced, read(parent['scores'][role]))
                    self.assertEqual([item['index'] for item in produced['rows']], roles[role])
                verified = consumer.verify_fold_input(fixture['base'], fixture['folds'], parent_ref,
                                                       work_root=directory, allow_fixture=True)
                self.assertEqual(verified['selected_variant'], 'online' if number == 1 else 'ema')
            self.assertEqual(sorted(seen_held), sorted(base['fit_indices']))
            self.assertEqual(len(seen_held), len(set(seen_held)))

    def test_real_aggregator_receipt_is_accepted_by_consumer_and_partial_is_rejected(self):
        for fault in (None, 'release', 'strict', 'missing_row'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture, inputs, inputs_ref, original, output = synthetic_aggregate_fixture(root, fault)
                tree = original['source_tree_proof']
                if fault is not None:
                    with self.assertRaises((ValueError, KeyError, OSError)):
                        oof.aggregate(gpu, inputs, output, inputs_ref['path'], inputs_ref['sha256'],
                                      tree['path'], tree['sha256'])
                    self.assertFalse((output / 'FOLD-INFERENCE.json').exists())
                    continue
                result = oof.aggregate(gpu, inputs, output, inputs_ref['path'], inputs_ref['sha256'],
                                       tree['path'], tree['sha256'])
                receipt = result['fold_receipt']
                verified = consumer.verify_fold_input(fixture['base'], fixture['folds'], receipt,
                                                       work_root=root, allow_fixture=True)
                self.assertEqual(verified['train_score_sha256'], read(receipt)['scores']['train']['sha256'])
                self.assertEqual(verified['heldout_score_sha256'], read(receipt)['scores']['heldout']['sha256'])
                self.assertEqual(result['forward_counts']['train'], len(verified['train_rows']))
                self.assertEqual(result['forward_counts']['heldout'], len(verified['heldout_rows']))

    def test_each_packet_identity_and_each_fit_row_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = synthetic_fixture(directory)
            base, sealed, parent = read(fixture['base']), read(fixture['folds']), read(fixture['parents'][0])
            packets = [read(ref) for ref in parent['heldout_packets']]
            changes = []
            for field, value in (('fold', 1), ('role', 'train'), ('proof_sha256', 'f' * 64),
                                 ('member', 'other'), ('official_read', True), ('TTA', True)):
                bad = copy.deepcopy(packets); bad[0][field] = value; changes.append(bad)
            for field, value in (('strict_state_load', False), ('storage', 'fp16'),
                                 ('variant', 'online'), ('selected_tensor_sha256', 'f' * 64),
                                 ('successful_updates', 801)):
                bad = copy.deepcopy(packets); bad[0]['model_load'][field] = value; changes.append(bad)
            bad = copy.deepcopy(packets); del bad[0]['model_load']; changes.append(bad)
            bad = copy.deepcopy(packets); bad[0]['records'].append(copy.deepcopy(bad[0]['records'][0])); changes.append(bad)
            bad = copy.deepcopy(packets); bad[0]['records'][0]['index'] = base['dev_indices'][0]; changes.append(bad)
            bad = copy.deepcopy(packets); bad[0]['records'][0]['logit'] = float('nan'); changes.append(bad)
            bad = copy.deepcopy(packets); bad[0]['records'].pop(); changes.append(bad)
            changes.append(packets[:-1])
            for bad in changes:
                with self.assertRaises((ValueError, KeyError)):
                    score(base, sealed, parent, bad, 'heldout')

    def test_consumer_rejects_selected_blob_and_sidecar_tampering(self):
        for fault in ('blob', 'sidecar_format', 'wrong_variant', 'unsafe'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = synthetic_fixture(root)
                parent_ref = fixture['parents'][0]
                parent = read(parent_ref)
                if fault == 'blob':
                    Path(parent['checkpoint']['path']).write_bytes(b'corrupt synthetic checkpoint')
                elif fault == 'sidecar_format':
                    side = read(parent['sidecar']); side['format_version'] = 2
                    parent['sidecar'] = _write(Path(parent['sidecar']['path']).parent,
                                               Path(parent['sidecar']['path']).name, side)
                    parent_ref = _write(root, 'modified-parent.json', parent)
                elif fault == 'wrong_variant':
                    parent['selected']['variant'] = 'online'
                    parent_ref = _write(root, 'modified-parent.json', parent)
                else:
                    safe = read(parent['safe']); safe['safety']['safe'] = False
                    parent['safe'] = _write(Path(parent['safe']['path']).parent, 'SAFE.json', safe)
                    parent_ref = _write(root, 'modified-parent.json', parent)
                with self.assertRaises(ValueError):
                    consumer.verify_fold_input(fixture['base'], fixture['folds'], parent_ref,
                                               work_root=root, allow_fixture=True)

    def test_real_cpu_loader_rejects_missing_or_extra_state_keys(self):
        import retrain.ema
        import retrain.inference
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
            def forward(self, value): return value * self.weight
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'synthetic.pt'; checkpoint.write_bytes(b'fake serialization only')
            payload = {'job': {'model': 'convnext_small', 'precision': 'fp32'},
                       'model': {'weight': torch.ones(1)}, 'ema': {'model': {'weight': torch.ones(1)}},
                       'identity': {}, 'manifest_digest': 'a' * 64, 'successful_updates': 800, 'mode': 'train'}
            with mock.patch.object(retrain.inference, 'load_file', return_value=payload), \
                 mock.patch.object(retrain.ema, 'select_variant', side_effect=lambda item, variant: item['ema']['model']):
                model, _, _, origin, device = retrain.inference._load_model(checkpoint, device='cpu',
                    variant='ema', model_factory=lambda job, initialize: Tiny())
                self.assertEqual(str(device), 'cpu'); self.assertEqual(origin['variant'], 'ema')
                self.assertTrue(torch.equal(model.weight, torch.ones(1)))
                for state in ({}, {'weight': torch.ones(1), 'unexpected': torch.ones(1)}):
                    payload['ema']['model'] = state
                    with self.assertRaises(RuntimeError):
                        retrain.inference._load_model(checkpoint, device='cpu', variant='ema',
                                                      model_factory=lambda job, initialize: Tiny())

    def test_owner_asset_gate_checks_selected_files_and_safe_identity(self):
        uid = os.environ['NCCCU_POD_UID']
        local_base = Path('/tmp/technical-prep') / uid
        for fault in ('none', 'unsafe', 'checkpoint', 'local_copy', 'local_sidecar', 'sidecar_format', 'sidecar_name',
                      'runtime_file', 'fold', 'uid'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory, \
                 tempfile.TemporaryDirectory(prefix='helper-owner-assets-', dir=local_base) as local:
                root = Path(directory)
                inputs, roles, bound = fake_asset_fixture(root, local)
                if fault == 'unsafe':
                    safe = read(inputs['safe']); safe['safety']['safe'] = False
                    inputs['safe'] = _write(root, 'SAFE.json', safe)
                elif fault == 'checkpoint':
                    Path(inputs['checkpoint']['path']).write_bytes(b'changed synthetic blob')
                elif fault == 'local_copy':
                    Path(inputs['local_checkpoint']['path']).write_bytes(b'changed local synthetic blob')
                elif fault == 'local_sidecar':
                    Path(inputs['local_checkpoint']['path'] + '.json').write_bytes(b'changed local synthetic sidecar')
                elif fault in ('sidecar_format', 'sidecar_name'):
                    side = read(inputs['sidecar'])
                    if fault == 'sidecar_format': side['format_version'] = 2
                    else: side['file'] = '../other.pt'
                    inputs['sidecar'] = _write(Path(inputs['sidecar']['path']).parent,
                                               Path(inputs['sidecar']['path']).name, side)
                    Path(inputs['local_checkpoint']['path'] + '.json').write_bytes(Path(inputs['sidecar']['path']).read_bytes())
                elif fault == 'runtime_file':
                    (Path(inputs['members'][0]['bundle']) / 'fixture.py').write_text('# changed\n')
                elif fault == 'fold':
                    bound['fold_number'] = 1
                with mock.patch.object(oof, 'verified_roles', return_value=roles), \
                     mock.patch.object(f05_folds, 'verify_bound_fold', return_value=bound):
                    if fault == 'none':
                        self.assertEqual(oof.verify_assets(gpu, inputs, work_root=root, pod_uid=uid), roles)
                    else:
                        with self.assertRaises((ValueError, OSError)):
                            oof.verify_assets(gpu, inputs, work_root=root, pod_uid=uid + '-wrong' if fault == 'uid' else uid)

    def test_prepare_synthetic_success_and_rejects_incomplete_or_changed_sources(self):
        import f04_bounded_fusion
        import f04_export_member
        from retrain import site_binding
        sys.path.insert(0, str(ROOT / 'tools/pretraining'))
        import source_chain
        uid = os.environ['NCCCU_POD_UID']
        local_base = Path('/tmp/technical-prep') / uid
        policy_path = ROOT / 'configs/F05-OOF-GPU-20261105.json'
        original_read_text, original_sha = Path.read_text, gpu.sha
        for fault in ('none', 'missing', 'duplicate', 'source_blob', 'sidecar_name', 'fold',
                      'base_manifest', 'unsafe', 'local_blob', 'local_sidecar'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory, \
                 tempfile.TemporaryDirectory(prefix='helper-owner-prepare-', dir=local_base) as local:
                root = Path(directory)
                args, policy, execution_site, roles, bound = synthetic_prepare_fixture(root, local, uid, fault)
                def read_text(path, *args, **kwargs):
                    if path == policy_path: return json.dumps(policy)
                    return original_read_text(path, *args, **kwargs)
                def sha(path):
                    path = Path(path)
                    if ((fault == 'local_blob' and path == args.local_dir / 'checkpoint-toy.pt')
                            or (fault == 'local_sidecar' and path == args.local_dir / 'checkpoint-toy.pt.json')):
                        return 'f' * 64
                    return original_sha(path)
                with mock.patch.object(Path, 'read_text', new=read_text), \
                     mock.patch.object(gpu, 'sha', side_effect=sha), \
                     mock.patch.object(f04_bounded_fusion, 'verify_blob', return_value='c' * 64), \
                     mock.patch.object(f04_export_member, 'verify_published', return_value={}), \
                     mock.patch.object(source_chain, 'cpu_context', return_value={'uid': uid}), \
                     mock.patch.object(site_binding, 'load_site', return_value=execution_site), \
                     mock.patch.object(oof, 'verified_roles', return_value=roles), \
                     mock.patch.object(f05_folds, 'verify_bound_fold', return_value=bound):
                    if fault == 'none':
                        oof.prepare(gpu, args)
                        inputs = json.loads(args.output.read_text())
                        self.assertEqual(inputs['status'], 'F05_fit_only_oof_inputs_sealed_no_GPU')
                        self.assertEqual(inputs['local_checkpoint']['sha256'], inputs['checkpoint']['sha256'])
                        self.assertEqual(inputs['selected'], inputs['members'][0]['selected'])
                        self.assertEqual(oof.validate_inputs(inputs, policy), inputs)
                        self.assertEqual(oof.verify_assets(gpu, inputs, work_root=root, pod_uid=uid), roles)
                        self.assertFalse(inputs['dev_read']); self.assertFalse(inputs['official_read'])
                    else:
                        with self.assertRaises((ValueError, OSError, KeyError)):
                            oof.prepare(gpu, args)
                        self.assertFalse(args.output.exists())

    def test_fake_cpu_worker_requires_loaded_identity_and_visits_both_roles_once(self):
        import retrain.inference
        uid = os.environ['NCCCU_POD_UID']
        local_base = Path('/tmp/technical-prep') / uid
        with tempfile.TemporaryDirectory() as directory, \
             tempfile.TemporaryDirectory(prefix='helper-owner-worker-', dir=local_base) as local:
            root = Path(directory)
            fixture = synthetic_fixture(root)
            base, sealed, parent = read(fixture['base']), read(fixture['folds']), read(fixture['parents'][0])
            checkpoint = Path(parent['checkpoint']['path'])
            local_checkpoint = Path(local) / checkpoint.name
            local_checkpoint.write_bytes(checkpoint.read_bytes())
            Path(str(local_checkpoint) + '.json').write_bytes(Path(parent['sidecar']['path']).read_bytes())
            selected = parent['selected']
            member = {'round_id': parent['round_id'], 'data_root': '/synthetic/images', 'cache': '/synthetic/cache'}
            inputs = {'fold': 0, 'members': [member], 'TTA': False, 'dev_read': False, 'official_read': False,
                'manifest': fixture['base'], 'folds': fixture['folds'], 'training_manifest': parent['training_manifest'],
                'checkpoint': parent['checkpoint'], 'sidecar': parent['sidecar'],
                'local_checkpoint': _ref(local_checkpoint), 'selected': selected}
            spec = {'oof_mode': True, 'soup_mode': False, 'official': None,
                    'inputs': parent['producer_inputs'], 'deadline_at': 9999999999.}
            assignment = oof.build_plan(parent['round_id'], 0)[0]
            parent_process = {'pid': os.getppid(), 'start_ticks': '123'}
            origin = {'sha256': parent['checkpoint']['sha256'], 'variant': selected['variant'],
                      'tensor_sha256': selected['tensor_sha256'], 'successful_updates': selected['successful_updates']}
            class SyntheticDataset(torch.utils.data.Dataset):
                def __init__(self): self.seen = []
                def __len__(self): return len(base['rows'])
                def __getitem__(self, index):
                    self.seen.append(index)
                    return torch.tensor([index / 100.], dtype=torch.float32), base['rows'][index]['label'], index
            class SyntheticModel(torch.nn.Module):
                def __init__(self, dtype):
                    super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1, dtype=dtype))
                def forward(self, images): return images.reshape(-1) * self.weight
            from retrain.ema import tensor_digest
            selected['tensor_sha256'] = tensor_digest(SyntheticModel(torch.float32).state_dict())
            origin['tensor_sha256'] = selected['tensor_sha256']
            precision = types.ModuleType('_f04_current_guard.precision')
            precision.RULE = {'synthetic': True}; precision.configure = lambda rule: None
            fresh = types.ModuleType('_f04_current_guard.fresh')
            fresh.runtime_stop_reason = lambda deadline, device: None
            with mock.patch.dict(sys.modules, {'_f04_current_guard.precision': precision,
                                              '_f04_current_guard.fresh': fresh}), \
                 mock.patch.object(gpu, 'bind_evaluation_cache'), \
                 mock.patch.object(gpu, 'process_identity', return_value={'active': True, 'start_ticks': '123'}):
                for fault in ('variant', 'tensor', 'update', 'blob', 'dtype', 'loaded_state', 'implicit_cast', 'none'):
                    with self.subTest(fault=fault):
                        dataset = SyntheticDataset()
                        model = SyntheticModel(torch.float16 if fault == 'dtype' else torch.float32)
                        if fault == 'loaded_state':
                            with torch.no_grad(): model.weight.fill_(2.)
                        local_inputs = copy.deepcopy(inputs)
                        loaded = dict(origin)
                        if fault == 'variant': loaded['variant'] = 'online'
                        if fault == 'tensor': loaded['tensor_sha256'] = 'f' * 64
                        if fault == 'update': loaded['successful_updates'] += 1
                        if fault == 'blob': loaded['sha256'] = 'f' * 64
                        if fault == 'implicit_cast':
                            source_digest = tensor_digest(SyntheticModel(torch.float16).state_dict())
                            local_inputs['selected']['tensor_sha256'] = source_digest
                            loaded['tensor_sha256'] = source_digest
                        output = root / ('worker-' + fault); (output / 'packets').mkdir(parents=True)
                        with mock.patch.object(retrain.inference, '_load_model',
                                return_value=(model, {'age': None}, None, loaded, torch.device('cpu'))), \
                             mock.patch.object(gpu, 'evaluation_dataset', return_value=dataset) as get_dataset:
                            if fault != 'none':
                                with self.assertRaises(ValueError):
                                    oof.worker(gpu, spec, local_inputs, assignment, 0, 'a' * 64,
                                               parent_process, output, 'cpu', torch)
                                self.assertEqual(get_dataset.call_count, 0)
                            else:
                                oof.worker(gpu, spec, local_inputs, assignment, 0, 'a' * 64,
                                           parent_process, output, 'cpu', torch)
                                expected = []
                                for role, key in (('train', 'fit_indices'), ('heldout', 'audit_indices')):
                                    indices = sealed['folds'][0][key]
                                    wanted = [i for _, batch in batch_assignments(indices, 8, 0, 8) for i in batch]
                                    packet = json.loads((output / 'packets' / ('rank-0-' + role + '.json')).read_text())
                                    self.assertEqual([item['index'] for item in packet['records']], wanted)
                                    self.assertEqual(packet['model_load']['storage'], 'fp32')
                                    self.assertTrue(packet['model_load']['strict_state_load'])
                                    expected.extend(wanted)
                                self.assertEqual(dataset.seen, expected)
                                self.assertFalse(set(dataset.seen) & set(base['dev_indices']))
                                self.assertEqual(len(dataset.seen), len(set(dataset.seen)))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
