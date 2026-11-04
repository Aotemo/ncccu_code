"""CPU-only isolation checks for F04 GPU dev dispatch and coverage."""
import importlib.util
import os
from pathlib import Path
import sys
import time

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


if __name__ == '__main__':
    test_fixed_devices_and_exact_dev_coverage()
    test_predecessor_requires_recent_complete_safety()
    print('F04 GPU fixed plan, exact coverage and predecessor safety checks passed', flush=True)
