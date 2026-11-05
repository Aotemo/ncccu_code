"""Fixed-sample and inventory-identity checks for the preregistered F04."""
import importlib.util
from pathlib import Path

import numpy as np


here = Path(__file__).resolve()
module_path = here.with_name('F04-BOUNDED-TEST-SCRIPT.py')
if not module_path.exists():
    module_path = here.parents[1] / 'tools/diagnostics/f04_bounded_fusion.py'
spec = importlib.util.spec_from_file_location('f04_bounded', module_path)
bounded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounded)


def test_fixed_sample_spacing_and_z_scale():
    left = np.arange(64, dtype=np.float64) * .01
    right = left + .00001
    result = bounded.compare_fixed_sample(left, right, 2.)
    assert result['count'] == 64 and result['small_sample_spacing_passed']
    assert abs(result['max_abs_logit_difference'] - .00001) < 1e-12
    assert abs(result['max_abs_z_difference'] - .000005) < 1e-12
    assert result['changed_pairwise_order_count'] == 0 and result['fp32_adjacent_tie_count'] == 0


def test_small_sample_ties_and_reordering_are_not_accepted():
    left = np.arange(64, dtype=np.float64)
    right = left.copy(); right[0], right[1] = 1.1, -.1
    result = bounded.compare_fixed_sample(left, right, 1.)
    assert result['changed_pairwise_order_count'] > 0 and not result['small_sample_spacing_passed']
    left[1] = left[0]
    assert not bounded.compare_fixed_sample(left, left, 1.)['small_sample_spacing_passed']
    for a, b, std in ((left[:-1], left[:-1], 1.), (left, left, 0.), (left, left * np.nan, 1.)):
        try: bounded.compare_fixed_sample(a, b, std)
        except ValueError: pass
        else: raise AssertionError('invalid fixed sample was accepted')


def test_selected_inventory_identity_cannot_drift():
    keys = ('round_id', 'status', 'selected', 'spec', 'science', 'public_source',
            'checkpoint_candidates', 'selected_blob_sha256')
    row = {key: key for key in keys}
    bounded.require_same_candidate(row, dict(row))
    for key in keys:
        changed = dict(row, **{key: 'changed'})
        try: bounded.require_same_candidate(row, changed)
        except ValueError: pass
        else: raise AssertionError('changed candidate was accepted: ' + key)


def test_new_completed_candidate_uses_its_own_pinned_discovery():
    assert hasattr(bounded, 'export_inventory_sha'), 'per-member export inventory binding is required'
    row = {key: key for key in ('round_id', 'status', 'selected', 'spec', 'science', 'public_source',
                                'checkpoint_candidates', 'selected_blob_sha256')}
    assert bounded.export_inventory_sha({}, row, 'old-sha', 'new-sha') == 'new-sha'
    assert bounded.export_inventory_sha({row['round_id']: row}, dict(row), 'old-sha', 'new-sha') == 'old-sha'
    changed = dict(row, selected='different')
    try:
        bounded.export_inventory_sha({row['round_id']: row}, changed, 'old-sha', 'new-sha')
    except ValueError:
        pass
    else:
        raise AssertionError('changed previously exported candidate was accepted')


if __name__ == '__main__':
    test_fixed_sample_spacing_and_z_scale()
    test_small_sample_ties_and_reordering_are_not_accepted()
    test_selected_inventory_identity_cannot_drift()
    test_new_completed_candidate_uses_its_own_pinned_discovery()
    print('F04 bounded sample and sealed-candidate checks passed', flush=True)
