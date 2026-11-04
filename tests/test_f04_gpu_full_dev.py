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
    test_predecessor_requires_recent_complete_safety()
    test_eight_rank_barrier_rejects_missing_reused_and_dead_processes()
    test_worker_release_requires_its_exact_pinned_proof_and_live_owner()
    test_worker_batch_rejects_identity_label_and_nonfinite_changes()
    test_evaluation_window_requires_recent_target_and_capacity_with_reserve()
    test_full_dev_comparison_preserves_fixed_members_and_detects_change()
    test_supervised_barrier_releases_only_complete_live_proofs()
    test_dispatch_claim_is_once_only_and_does_not_spawn()
    test_hidden_device_dispatch_stops_before_reserving_claim()
    test_detached_controller_records_failure_without_retry()
    print('F04 GPU fixed plan, exact coverage and predecessor safety checks passed', flush=True)
