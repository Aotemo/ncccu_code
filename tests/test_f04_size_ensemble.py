"""F04 must stop at the first oversized member and preserve inference state."""
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
spec = importlib.util.spec_from_file_location('f04', ROOT / 'tools/diagnostics/f04_size_ensemble.py')
f04 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f04)


def test_budget_stops_and_ties_use_original_order():
    rows = [
        {'id': 'late', 'auroc': .989, 'order': 2, 'fp16_bytes': 1, 'fp32_bytes': 1},
        {'id': 'second', 'auroc': .990, 'order': 1, 'fp16_bytes': 4, 'fp32_bytes': 8},
        {'id': 'first', 'auroc': .990, 'order': 0, 'fp16_bytes': 3, 'fp32_bytes': 8},
    ]
    selected = f04.rank_and_cap(rows, 'fp16', 7)
    assert selected == {'members': ['first', 'second'], 'bytes': 7, 'stopped_before': 'late'}
    selected = f04.rank_and_cap(rows, 'fp32', 15)
    assert selected == {'members': ['first'], 'bytes': 8, 'stopped_before': 'second'}
    invalid = [dict(rows[0], auroc=float('nan'))]
    try: f04.rank_and_cap(invalid, 'fp16', 7)
    except ValueError: pass
    else: raise AssertionError('nonfinite AUROC accepted')


def test_state_export_contains_age_and_nonfloating_buffers():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 1)
            self.age_weight = torch.nn.Parameter(torch.tensor([.1, -.2]))
            self.register_buffer('index', torch.tensor([1, 2], dtype=torch.int64))
    model = Toy().eval()
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as folder:
        result = f04.export_state_files(model, Path(folder))
        f32 = torch.load(Path(folder) / 'weights-fp32.pt', map_location='cpu', weights_only=True)
        f16 = torch.load(Path(folder) / 'weights-fp16.pt', map_location='cpu', weights_only=True)
        assert result['parameter_count'] == sum(p.numel() for p in model.parameters())
        assert result['fp32']['bytes'] > 0 and result['fp16']['bytes'] > 0
        assert set(f32) == set(f16) == set(original)
        assert f16['age_weight'].dtype == torch.float16 and f16['index'].dtype == torch.int64
        assert all(torch.equal(f32[k], original[k]) for k in original)
        assert all(torch.equal(model.state_dict()[k], original[k]) for k in original)
        clone = Toy(); clone.load_state_dict(f16, strict=True)
        assert torch.equal(clone.age_weight, original['age_weight'].half().float())


def test_paired_metrics_and_equal_zscore():
    labels = [1, 0, 1, 0]
    left = np.array([2., 0., 1., -1.])
    right = np.array([2.001, 0., .999, -1.])
    result = f04.compare_predictions(labels, left, right)
    assert result['fp32']['auroc'] == result['fp16']['auroc'] == 1.0
    assert math.isclose(result['max_abs_logit_difference'], .001, abs_tol=1e-12)
    result = f04.equal_zscore(labels, {'a': left, 'b': right}, ['a', 'b'])
    assert result['metrics']['auroc'] == 1.0 and result['members'] == ['a', 'b']
    try: f04.equal_zscore(labels, {'a': np.ones(4)}, ['a'])
    except ValueError: pass
    else: raise AssertionError('zero variance accepted')


def test_export_requires_one_eligible_general_source():
    sys.path.insert(0, str(ROOT / 'tools/diagnostics'))
    spec = importlib.util.spec_from_file_location('f04_export', ROOT / 'tools/diagnostics/f04_export_member.py')
    export = importlib.util.module_from_spec(spec); spec.loader.exec_module(export)
    row = {'round_id': 'r-example', 'status': 'prediction_verified_weight_pending',
           'selected': {'auroc': .990}, 'public_source': {'sha256': 'public', 'training_data': 'general'},
           'checkpoint_candidates': [{'path': '/fixture/selected.pt'}]}
    inventory = {'status': 'read_only_F03_candidate_discovery_no_fusion', 'official_read': False, 'members': [row]}
    approved = {'public': 'general'}
    assert export.eligible_row(inventory, 'r-example', approved) == row
    for invalid in (dict(row, status='selected_checkpoint_missing'),
                    dict(row, public_source={'sha256': 'medical', 'training_data': 'medical'}),
                    dict(row, checkpoint_candidates=[]),
                    dict(row, selected={'auroc': .987})):
        try: export.eligible_row(dict(inventory, members=[invalid]), 'r-example', approved)
        except ValueError: pass
        else: raise AssertionError('unverified or ineligible checkpoint accepted')


def test_paired_forward_uses_identical_inputs_and_rejects_nonfinite():
    sys.path.insert(0, str(ROOT / 'tools/diagnostics'))
    spec = importlib.util.spec_from_file_location('f04_predict', ROOT / 'tools/diagnostics/f04_predict_member.py')
    predict = importlib.util.module_from_spec(spec); spec.loader.exec_module(predict)
    class Toy(torch.nn.Module):
        def __init__(self, scale):
            super().__init__(); self.scale = torch.nn.Parameter(torch.tensor(scale))
        def forward(self, image): return image[:, 0] * self.scale
    model32, model16 = Toy(1.).eval(), Toy(.5).eval()
    images, age = torch.tensor([[1.], [2.]]), torch.tensor([.1, .2])
    calls = []
    def forward(model, actual_images, actual_age, job):
        calls.append((actual_images is images, actual_age is age))
        return model(actual_images) + actual_age
    a, b = predict.paired_logits(model32, model16, images, age, {}, forward)
    assert calls == [(True, True), (True, True)] and a.shape == b.shape == (2,)
    assert torch.allclose(a, torch.tensor([1.1, 2.2])) and torch.allclose(b, torch.tensor([.6, 1.2]))
    model16.scale.data.fill_(float('nan'))
    try: predict.paired_logits(model32, model16, images, age, {}, forward)
    except FloatingPointError: pass
    else: raise AssertionError('nonfinite dev score accepted')


def test_historical_dataset_signature_does_not_drop_real_preprocessing():
    sys.path.insert(0, str(ROOT / 'tools/diagnostics'))
    spec = importlib.util.spec_from_file_location('f04_predict', ROOT / 'tools/diagnostics/f04_predict_member.py')
    predict = importlib.util.module_from_spec(spec); spec.loader.exec_module(predict)
    class OldDataset:
        def __init__(self, root, rows, size, training, normalization, prepared_size, age): pass
    class NewDataset:
        def __init__(self, root, rows, size, training, normalization, prepared_size, age, input_preprocessing=None): pass
    job = {'normalization': 'imagenet', 'prepared_size': 512, 'age': {'enabled': False}}
    assert 'input_preprocessing' not in predict.dataset_options(OldDataset, job)
    assert predict.dataset_options(NewDataset, job)['input_preprocessing'] is None
    job['input_preprocessing'] = {'registered': True}
    assert predict.dataset_options(NewDataset, job)['input_preprocessing'] == job['input_preprocessing']
    try: predict.dataset_options(OldDataset, job)
    except ValueError: pass
    else: raise AssertionError('configured preprocessing was silently discarded')


if __name__ == '__main__':
    test_budget_stops_and_ties_use_original_order()
    test_state_export_contains_age_and_nonfloating_buffers()
    test_paired_metrics_and_equal_zscore()
    test_export_requires_one_eligible_general_source()
    test_paired_forward_uses_identical_inputs_and_rejects_nonfinite()
    test_historical_dataset_signature_does_not_drop_real_preprocessing()
    print('F04 fixed budget, complete state export and paired-score tests passed', flush=True)
