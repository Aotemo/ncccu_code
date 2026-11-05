"""CPU-only isolation checks for F04 GPU dev dispatch and coverage."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
from unittest.mock import patch

ROOT = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT / 'runtime/bundle'), str(ROOT / 'tools/diagnostics')]
spec = importlib.util.spec_from_file_location('f04_gpu_full_dev', ROOT / 'tools/diagnostics/f04_gpu_full_dev.py')
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('unsafe or incomplete F04 input was accepted')


def test_fixed_devices_and_exact_dev_coverage():
    from retrain.parallel_evaluation import batch_assignments
    members = ['a', 'b', 'c', 'd', 'e']
    plan = gpu.build_plan(members)
    assert [row['device_rank'] for row in plan] == list(range(8))
    assert [(row['member'], row['rank'], row['world']) for row in plan] == [
        ('a', 0, 2), ('a', 1, 2), ('b', 0, 2), ('b', 1, 2),
        ('c', 0, 2), ('c', 1, 2), ('d', 0, 1), ('e', 0, 1)]
    rejects(lambda: gpu.build_plan(['a'] * 5))
    rows = [{'name': str(index), 'label': index % 2} for index in range(13)]
    indices = list(range(13)); packets = []
    for rank in range(2):
        records = []
        for ordinal, batch in batch_assignments(indices, 8, rank, 2):
            for offset, index in enumerate(batch):
                records.append({'index': index, 'rank': rank, 'batch': ordinal, 'offset': offset,
                                'name': rows[index]['name'], 'label': rows[index]['label'],
                                'logit': float(index), 'probability': .5})
        packets.append({'rank': rank, 'error': None, 'records': records})
    predictions = gpu.canonical_member(rows, indices, packets, 2)
    assert [row['name'] for row in predictions] == [str(index) for index in indices]
    packets[0]['records'].append(dict(packets[0]['records'][0]))
    rejects(lambda: gpu.canonical_member(rows, indices, packets, 2))


def test_selected_variant_preserves_online_or_ema_identity():
    assert gpu.selected_variant({'selected': {'variant': 'ema'}}) == 'ema'
    assert gpu.selected_variant({'selected': {'variant': 'online'}}) == 'online'
    rejects(lambda: gpu.selected_variant({'selected': {'variant': 'unknown'}}))
    rejects(lambda: gpu.selected_variant({'selected': {}}))


def test_historical_runtime_rebinds_the_pinned_member_site():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary); bundle = root / 'bundle'; bundle.mkdir()
        science = root / 'SCIENCE.json'; science.write_text(json.dumps({'candidate_profile': 'historical'}))
        site = root / 'SITE_BINDING.json'
        site.write_text(json.dumps({'work_root': str(root), 'data_root': '/data',
            'science': {'configuration': {'path': str(science), 'sha256': gpu.sha(science)}}}))
        member = {'bundle': str(bundle), 'freeze': {'path': str(root / 'FREEZE.json')},
                  'site': {'path': str(site), 'sha256': gpu.sha(site)}, 'data_root': '/data'}
        with patch.dict(os.environ, {'NCCCU_SITE_BINDING': '/current/site',
                                     'NCCCU_SITE_BINDING_SHA256': 'a' * 64}):
            gpu.bind_member_runtime(member)
            assert os.environ['NCCCU_SITE_BINDING'] == str(site)
            assert os.environ['NCCCU_SITE_BINDING_SHA256'] == gpu.sha(site)
            bad = {**member, 'site': {'path': str(site), 'sha256': '0' * 64}}
            rejects(lambda: gpu.bind_member_runtime(bad))


def test_predecessor_requires_recent_complete_safety():
    now = time.time()
    safe = {'phase': 'terminal', 'observed_at': now, 'first_success_receipt_read': True,
            'done': {'present': True, 'status': 'completed'}, 'safety': {'safe': True},
            'signals_sent': False, 'model_forward_executed': False, 'official_read': False,
            'formal_root': '/work/rounds/previous'}
    gpu.require_safe_predecessor(safe, '/work/rounds/previous', now)
    for changed in (dict(safe, phase='running'), dict(safe, observed_at=now - 121),
                    dict(safe, safety={'safe': False}), dict(safe, official_read=True)):
        rejects(lambda changed=changed: gpu.require_safe_predecessor(changed, '/work/rounds/previous', now))


def test_eight_rank_barrier_rejects_missing_reused_and_dead_processes():
    parent = {'pid': 90, 'start_ticks': '987'}
    proofs = [{'rank': i, 'pid': 100 + i, 'start_ticks': str(200 + i), 'parent': parent,
               'device': 'cuda:' + str(i), 'world_size': 8} for i in range(8)]
    launches = [{'rank': i, 'pid': 100 + i, 'start_ticks': str(200 + i)} for i in range(8)]
    seen = []
    verify_devices = lambda rows: seen.extend(rows)
    identity = lambda pid: {'pid': pid, 'start_ticks': str(200 + pid - 100), 'active': True, 'parent_pid': 90}
    gpu.validate_proof_barrier(proofs, launches, parent, verify_devices, identity)
    assert len(seen) == 8
    rejects(lambda: gpu.validate_proof_barrier(proofs[:-1], launches, parent, verify_devices, identity))
    altered = [dict(row) for row in proofs]; altered[7]['pid'] = altered[0]['pid']
    rejects(lambda: gpu.validate_proof_barrier(altered, launches, parent, verify_devices, identity))
    dead = lambda pid: {'pid': pid, 'start_ticks': str(200 + pid - 100), 'active': pid != 103, 'parent_pid': 90}
    rejects(lambda: gpu.validate_proof_barrier(proofs, launches, parent, verify_devices, dead))
    wrong_parent = [dict(row) for row in proofs]; wrong_parent[2]['parent'] = {'pid': 90, 'start_ticks': '999'}
    rejects(lambda: gpu.validate_proof_barrier(wrong_parent, launches, parent, verify_devices, identity))


def test_worker_release_requires_its_exact_pinned_proof_and_live_owner():
    parent = {'pid': 90, 'start_ticks': '987'}
    release = {'status': 'all_rank_device_proofs_passed', 'spec_sha256': 'a' * 64,
               'parent': parent, 'proof_sha256': {str(i): str(i) * 64 for i in range(8)}}
    live = lambda pid: {'pid': pid, 'start_ticks': '987', 'active': True}
    gpu.validate_worker_release(release, 3, '3' * 64, 'a' * 64, parent, live)
    rejects(lambda: gpu.validate_worker_release(release, 3, '0' * 64, 'a' * 64, parent, live))
    rejects(lambda: gpu.validate_worker_release(release, 3, '3' * 64, 'b' * 64, parent, live))
    rejects(lambda: gpu.validate_worker_release(release, 3, '3' * 64, 'a' * 64, parent,
                                                 lambda pid: {'pid': pid, 'start_ticks': '987', 'active': False}))


def test_worker_batch_rejects_identity_label_and_nonfinite_changes():
    rows = [{'name': 'a', 'label': 0}, {'name': 'b', 'label': 1}]
    records = gpu.worker_batch_records(rows, [1, 0], [1, 0], [1, 0], [0.2, -0.3], [.55, .42], 1, 4)
    assert [(x['index'], x['rank'], x['batch'], x['offset']) for x in records] == [(1, 1, 4, 0), (0, 1, 4, 1)]
    rejects(lambda: gpu.worker_batch_records(rows, [1, 0], [0, 1], [1, 0], [.2, -.3], [.55, .42], 1, 4))
    rejects(lambda: gpu.worker_batch_records(rows, [1, 0], [1, 0], [0, 0], [.2, -.3], [.55, .42], 1, 4))
    rejects(lambda: gpu.worker_batch_records(rows, [1, 0], [1, 0], [1, 0], [float('nan'), -.3], [.55, .42], 1, 4))


def test_evaluation_window_requires_recent_target_and_capacity_with_reserve():
    now = time.time()
    window = {'status': 'cross_batch_safe_window_and_capacity_verified', 'at': now,
              'predecessor_safe_sha256': 'a' * 64, 'site_sha256': 'b' * 64,
              'round_id': 'next', 'required_free_bytes': 4000, 'free_bytes': 9000,
              'activation_reserved': False, 'GPU_started': False}
    gpu.validate_evaluation_window(window, 'a' * 64, 'b' * 64, 'next', now, 9000, reserve=1000)
    rejects(lambda: gpu.validate_evaluation_window(window, 'a' * 64, 'b' * 64, 'next', now + 121, 9000, reserve=1000))
    rejects(lambda: gpu.validate_evaluation_window(window, 'a' * 64, 'b' * 64, 'next', now, 5000, reserve=1000))
    rejects(lambda: gpu.validate_evaluation_window(dict(window, activation_reserved=True), 'a' * 64,
                                                    'b' * 64, 'next', now, 9000, reserve=1000))


def test_full_dev_comparison_preserves_fixed_members_and_detects_change():
    labels, members = [0, 1, 0, 1], ['a', 'b', 'c', 'd', 'e']
    old = {name: [float(i), float(i + 3), float(i + 1), float(i + 4)]
           for i, name in enumerate(members)}
    result = gpu.compare_full_dev(labels, old, old, members)
    assert result['fp16']['metrics'] == result['fp32']['metrics']
    assert result['ensemble_max_abs_z_score_difference'] == 0
    assert all(row['max_abs_logit_difference'] == 0 for row in result['members'].values())
    rejects(lambda: gpu.compare_full_dev(labels, old, {k: v for k, v in old.items() if k != 'a'}, members))


def test_supervised_barrier_releases_only_complete_live_proofs():
    parent = {'pid': 90, 'start_ticks': '987'}
    original_modules, original_identity = gpu.guard_modules, gpu.process_identity
    try:
        checked = []
        fake_guard = types.SimpleNamespace(verify_rank_devices=lambda proofs, site: checked.extend(proofs))
        fake_persistence = types.SimpleNamespace(atomic_json=lambda path, value: path.write_text(json.dumps(value)))
        gpu.guard_modules = lambda source: (fake_guard, None, fake_persistence, None)
        gpu.process_identity = lambda pid: {'pid': pid, 'start_ticks': str(200 + pid - 100),
                                            'parent_pid': 90, 'active': True}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary); (output / 'proofs').mkdir(); (output / 'processes').mkdir()
            for i in range(8):
                proof = {'rank': i, 'pid': 100 + i, 'start_ticks': str(200 + i), 'parent': parent,
                         'device': 'cuda:' + str(i), 'world_size': 8}
                launch = {'pid': 100 + i, 'start_ticks': str(200 + i)}
                (output / 'proofs' / f'rank-{i}.json').write_text(json.dumps(proof))
                (output / 'processes' / f'rank-{i}-launch.json').write_text(json.dumps(launch))
            stop = threading.Event()
            gpu.release_supervised_ranks(output, output / 'spec', 'a' * 64, {}, parent, time.time() + 1, stop)
            release = json.loads((output / 'RELEASE.json').read_text())
            assert release['status'] == 'all_rank_device_proofs_passed' and len(checked) == 8 and not stop.is_set()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary); (output / 'proofs').mkdir(); (output / 'processes').mkdir()
            stop = threading.Event()
            gpu.release_supervised_ranks(output, output / 'spec', 'a' * 64, {}, parent, time.time() + .05, stop)
            assert stop.is_set() and (output / 'BARRIER-FAILURE.json').exists() and not (output / 'RELEASE.json').exists()
    finally:
        gpu.guard_modules, gpu.process_identity = original_modules, original_identity


def test_official_options_require_both_freeze_path_and_digest():
    assert gpu.official_freeze_options(None, None) == {}
    options = gpu.official_freeze_options('/work/final/F04-20261104/FREEZE.json', 'a' * 64)
    assert options['official-freeze-sha256'] == 'a' * 64
    rejects(lambda: gpu.official_freeze_options('/work/FREEZE.json', None))
    rejects(lambda: gpu.official_freeze_options(None, 'a' * 64))
    rejects(lambda: gpu.official_freeze_options('/work/FREEZE.json', 'bad'))


def test_official_pre_release_hook_runs_once_only_after_all_proofs():
    parent = {'pid': 90, 'start_ticks': '987'}
    fake_guard = types.SimpleNamespace(verify_rank_devices=lambda proofs, site: None)
    fake_persistence = types.SimpleNamespace(atomic_json=lambda path, value: path.write_text(json.dumps(value)))
    identity = lambda pid: {'pid': pid, 'start_ticks': str(200 + pid - 100),
                            'parent_pid': 90, 'active': True}
    with patch.object(gpu, 'guard_modules', return_value=(fake_guard, None, fake_persistence, None)), \
         patch.object(gpu, 'process_identity', side_effect=identity):
        for failure in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary); (output / 'proofs').mkdir(); (output / 'processes').mkdir()
                for i in range(8):
                    proof = {'rank': i, 'pid': 100 + i, 'start_ticks': str(200 + i), 'parent': parent,
                             'device': 'cuda:' + str(i), 'world_size': 8}
                    (output / 'proofs' / f'rank-{i}.json').write_text(json.dumps(proof))
                    (output / 'processes' / f'rank-{i}-launch.json').write_text(json.dumps(
                        {'pid': 100 + i, 'start_ticks': str(200 + i)}))
                calls = []
                def hook():
                    calls.append(1)
                    if failure:
                        raise ValueError('synthetic official input failed')
                    return {'rows_sha256': 'b' * 64}
                stop = threading.Event()
                gpu.release_supervised_ranks(output, output / 'spec', 'a' * 64, {}, parent,
                                             time.time() + 1, stop, before_release=hook)
                assert calls == [1]
                assert (output / 'RELEASE.json').exists() is not failure
                assert stop.is_set() is failure
                if not failure:
                    assert json.loads((output / 'RELEASE.json').read_text())['official']['rows_sha256'] == 'b' * 64


def test_official_reduction_requires_exact_once_coverage_and_frozen_moments():
    from retrain.parallel_evaluation import batch_assignments
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary); (output / 'packets').mkdir()
        names = ['member-' + str(i) for i in range(5)]
        rows = [{'name': f'image-{i}', 'label': int(i < 191), 'relative_path': f'valid/image-{i}'}
                for i in range(5849)]
        population = output / 'OFFICIAL-ROWS-private.json'
        population.write_text(json.dumps({'rows': rows, 'manifest_sha256': 'a' * 64, 'csv_sha256': 'b' * 64}))
        proof_sha = {str(i): str(i) * 64 for i in range(8)}
        (output / 'RELEASE.json').write_text(json.dumps({'proof_sha256': proof_sha, 'official': {
            'rows_path': str(population), 'rows_sha256': gpu.sha(population)}}))
        plan = gpu.build_plan(names)
        inputs = {'members': [{'round_id': name} for name in names], 'plan': plan}
        members = []
        for i, name in enumerate(names):
            members.append({'round_id': name,
                'model': {'path': f'/models/{i}', 'sha256': str(i) * 64, 'bytes': 99_000_000,
                          'fp16_tensor_sha256': str(i + 5) * 64},
                'metadata': {'path': f'/metadata/{i}', 'sha256': 'a' * 64},
                'normalization': {'mean': float(i), 'std': float(i + 1)}})
        frozen = {'format': 'f04-final-frozen-v1', 'status': 'F04_ensemble_final_frozen',
            'official_evaluations_allowed': 1, 'official_metadata_used_for_selection': False,
            'TTA': False, 'weight_bytes': 495_000_000, 'members': members,
            'official': {'manifest_path': '/data/valid.json', 'manifest_sha256': 'c' * 64, 'data_root': '/data'},
            'full_dev': {'count': 5140, 'positive': 167, 'fp32_auroc': .9902, 'fp16_auroc': .9901,
                         'fp16_ap': .84, 'official_read': False, 'summary_sha256': 'b' * 64}}
        member = dict(members[0], selected={'variant': 'ema'}, freeze={'path': '/history'},
                      bundle='/bundle', site={'path': '/site'}, data_root='/data', cache='/cache')
        frozen['members'][0].update({key: member[key] for key in ('selected', 'freeze', 'bundle', 'site', 'data_root')})
        frozen_path = output / 'FREEZE.json'; frozen_path.write_text(json.dumps(frozen))
        spec = {'official': {'freeze_path': str(frozen_path), 'freeze_sha256': gpu.sha(frozen_path)}}
        release = {'official': {'freeze_sha256': gpu.sha(frozen_path),
                    'rows_path': str(population), 'rows_sha256': gpu.sha(population)}}
        actual_rows, indices, is_official = gpu.evaluation_population({}, member, spec, release, output)
        assert is_official and len(actual_rows) == len(indices) == 5849
        rejects(lambda: gpu.evaluation_population({}, dict(member, bundle='/wrong'), spec, release, output))
        rejects(lambda: gpu.evaluation_population({}, member, spec, {'official': {}}, output))
        with patch.dict(os.environ, {'TRAIN_IMAGE_CACHE': '/train', 'TRAIN_GLOBAL_IMAGE_CACHE': '/global'}):
            gpu.bind_evaluation_cache(member, True, output)
            assert 'TRAIN_IMAGE_CACHE' not in os.environ and 'TRAIN_GLOBAL_IMAGE_CACHE' not in os.environ
            assert os.environ['TRAIN_VERIFICATION_CACHE'] == str(output / 'verified-official-images')
        for assignment in plan:
            records = []
            for ordinal, batch in batch_assignments(list(range(len(rows))), 8,
                                                     assignment['rank'], assignment['world']):
                for offset, index in enumerate(batch):
                    records.append({'index': index, 'rank': assignment['rank'], 'batch': ordinal, 'offset': offset,
                        'name': rows[index]['name'], 'label': rows[index]['label'],
                        'logit': float(rows[index]['label']), 'probability': .75 if rows[index]['label'] else .25})
            rank = assignment['device_rank']
            packet = {'device_rank': rank, 'member': assignment['member'], 'rank': assignment['rank'],
                      'error': None, 'input_sha256': 'd' * 64, 'proof_sha256': proof_sha[str(rank)],
                      'TTA': False, 'official_read': True, 'records': records}
            (output / 'packets' / f'rank-{rank}.json').write_text(json.dumps(packet))
        result = gpu.aggregate_packets(ROOT, inputs, output, 'd' * 64, frozen)
        assert result['count'] == 5849 and result['positive'] == 191
        assert result['metrics']['auroc'] == 1 and result['metrics']['ap'] == 1
        path = output / 'packets/rank-0.json'; packet = json.loads(path.read_text())
        packet['records'].append(packet['records'][0]); path.write_text(json.dumps(packet))
        rejects(lambda: gpu.aggregate_packets(ROOT, inputs, output, 'd' * 64, frozen))


def test_dispatch_claim_is_once_only_and_does_not_spawn():
    with tempfile.TemporaryDirectory() as temporary:
        claim = Path(temporary) / 'attempt'
        digest = gpu.reserve_evaluation_attempt(claim, {'test': 'fixture'})
        assert len(digest) == 64 and (claim / 'REQUEST.json').is_file()
        assert not (claim / 'evaluation').exists()
        try:
            gpu.reserve_evaluation_attempt(claim, {'test': 'fixture'})
        except FileExistsError:
            pass
        else:
            raise AssertionError('reused evaluation claim was accepted')


def test_hidden_device_dispatch_stops_before_reserving_claim():
    fake = types.SimpleNamespace(load_site=lambda *a, **k: {}, prepare_native_entry=lambda *a: None,
                                 bind_cpu_role=lambda *a, **k: None)
    args = types.SimpleNamespace(source_root=ROOT, site=Path('/fixture/site'), site_sha256='a' * 64)
    with patch.object(gpu, 'guard_modules', return_value=(fake, None, None, None)), \
         patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': '', 'ROCR_VISIBLE_DEVICES': ''}):
        rejects(lambda: gpu.dispatch(args))


def test_detached_controller_records_failure_without_retry():
    class FakeProcess:
        pid = os.getpid()
        def wait(self, timeout=None):
            return 7
    with tempfile.TemporaryDirectory() as temporary:
        claim = Path(temporary) / 'attempt'
        request = {name: '/fixture' for name in (*gpu.OWNER_PATH_OPTIONS, *gpu.OWNER_TEXT_OPTIONS)}
        request.update({'source-root': str(ROOT), 'output': str(claim / 'evaluation'),
                        'deadline_at': time.time() + 20, 'source_sha256': gpu.sha(gpu.__file__)})
        digest = gpu.reserve_evaluation_attempt(claim, request)
        with patch.object(gpu.subprocess, 'Popen', return_value=FakeProcess()) as spawn:
            gpu.controller(types.SimpleNamespace(claim_dir=claim, request_sha256=digest))
        result = json.loads((claim / 'RESULT.json').read_text())
        assert result['status'] == 'owner_failed' and result['exit_code'] == 7 and spawn.call_count == 1


if __name__ == '__main__':
    test_fixed_devices_and_exact_dev_coverage()
    test_selected_variant_preserves_online_or_ema_identity()
    test_historical_runtime_rebinds_the_pinned_member_site()
    test_predecessor_requires_recent_complete_safety()
    test_eight_rank_barrier_rejects_missing_reused_and_dead_processes()
    test_worker_release_requires_its_exact_pinned_proof_and_live_owner()
    test_worker_batch_rejects_identity_label_and_nonfinite_changes()
    test_evaluation_window_requires_recent_target_and_capacity_with_reserve()
    test_full_dev_comparison_preserves_fixed_members_and_detects_change()
    test_supervised_barrier_releases_only_complete_live_proofs()
    test_official_options_require_both_freeze_path_and_digest()
    test_official_pre_release_hook_runs_once_only_after_all_proofs()
    test_official_reduction_requires_exact_once_coverage_and_frozen_moments()
    test_dispatch_claim_is_once_only_and_does_not_spawn()
    test_hidden_device_dispatch_stops_before_reserving_claim()
    test_detached_controller_records_failure_without_retry()
    print('F04 GPU fixed plan, exact coverage and predecessor safety checks passed', flush=True)
