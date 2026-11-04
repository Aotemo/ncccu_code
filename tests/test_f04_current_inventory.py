"""The fixed F04 rule honors its prior P2 exclusion and fresh candidate census."""
import copy
import importlib.util
import os
from pathlib import Path

root = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu = load('f04_gpu_full_dev', 'tools/diagnostics/f04_gpu_full_dev.py')
bounded = load('f04_bounded_fusion', 'tools/diagnostics/f04_bounded_fusion.py')
P2 = 'r256-small-seed20261111-pairsigmoid-24k-p4-9010-20261104'


def reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('invalid or stale F04 inventory was accepted')


def test_only_preregistered_p2_round_is_excluded():
    assert bounded.preregistered_exclusion(P2) == 'preregistered_P2_variance_diagnostic_not_F04'
    assert bounded.preregistered_exclusion('r261-small-mixstyle-seed20261117-pairlog-24k-p4-9010-20261104') is None
    assert bounded.preregistered_exclusion('r256-another-round') is None


def test_current_inventory_is_fresh_and_pinned_to_actual_predecessor():
    members = [{'round_id': name} for name in ('a', 'b', 'c', 'd', 'e')]
    inputs = {'members': members, 'weight_bytes': 495, 'inventory': {'sha256': 'a' * 64}}
    safe = {'observed_at': 100., 'formal_root': '/work/rounds/previous', 'selected': {'auroc': .987}}
    current = {'status': 'read_only_F03_candidate_discovery_no_fusion', 'at': 101., 'official_read': False,
               'members': [{'round_id': 'previous', 'selected': {'auroc': .987}}]}
    refreshed = {'status': 'F04_fp16_size_members_sealed_before_blend', 'at': 102., 'official_read': False,
                 'old_inventory_sha256': 'a' * 64, 'inventory_sha256': 'b' * 64,
                 'fp16': {'members': ['a', 'b', 'c', 'd', 'e'], 'bytes': 495}, 'eligible_order': []}
    gpu.validate_current_preflight(refreshed, current, 'b' * 64, safe, inputs)
    for changed in (dict(refreshed, inventory_sha256='a' * 64),
                    dict(refreshed, old_inventory_sha256='b' * 64),
                    dict(refreshed, at=100.5),
                    dict(refreshed, fp16={'members': ['a', 'b', 'c', 'd', 'other'], 'bytes': 495})):
        reject(lambda changed=changed: gpu.validate_current_preflight(changed, current, 'b' * 64, safe, inputs))
    for changed in (dict(current, at=99.), dict(current, official_read=True),
                    dict(current, members=[]),
                    dict(current, members=[{'round_id': 'previous', 'selected': {'auroc': .986}}])):
        reject(lambda changed=changed: gpu.validate_current_preflight(refreshed, changed, 'b' * 64, safe, inputs))
    high_safe = copy.deepcopy(safe); high_safe['selected']['auroc'] = .989
    high_current = copy.deepcopy(current); high_current['members'][0]['selected']['auroc'] = .989
    reject(lambda: gpu.validate_current_preflight(refreshed, high_current, 'b' * 64, high_safe, inputs))
    high_preflight = dict(refreshed, eligible_order=['previous'])
    gpu.validate_current_preflight(high_preflight, high_current, 'b' * 64, high_safe, inputs)
    p2_safe = dict(high_safe, formal_root='/work/rounds/' + P2)
    p2_current = dict(high_current, members=[{'round_id': P2, 'selected': {'auroc': .989}}])
    p2_preflight = dict(refreshed, excluded=[{'round_id': P2,
                         'reason': 'preregistered_P2_variance_diagnostic_not_F04'}])
    gpu.validate_current_preflight(p2_preflight, p2_current, 'b' * 64, p2_safe, inputs)
    reject(lambda: gpu.validate_current_preflight(dict(p2_preflight, excluded=[]), p2_current,
                                                  'b' * 64, p2_safe, inputs))


if __name__ == '__main__':
    test_only_preregistered_p2_round_is_excluded()
    test_current_inventory_is_fresh_and_pinned_to_actual_predecessor()
    print('F04 pre-registered P2 exclusion and fresh inventory binding passed')
