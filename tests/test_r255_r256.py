"""Fresh identities and fixed spatial pooling for the next two Small rounds."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock

import torch

ROOT = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'runtime/bundle'))


def configured(ex, value, round_id):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'science.json'
        raw = json.dumps(value).encode(); path.write_bytes(raw)
        site = {'work_root': folder, 'round_id': round_id,
                'science': {'configuration': {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}}}
        with mock.patch.object(ex, 'load_site', return_value=site):
            return ex.science_configuration()


def test_spatial_patch_pool_excludes_prefix_and_preserves_quadrant_order():
    from retrain.models import _dino3_quadrant_patch_pool
    grid = torch.empty((1, 32, 32, 384))
    grid[:, :16, :16] = 1; grid[:, :16, 16:] = 2
    grid[:, 16:, :16] = 3; grid[:, 16:, 16:] = 4
    tokens = torch.cat((torch.full((1, 5, 384), 99.), grid.reshape(1, 1024, 384)), dim=1).requires_grad_()
    output = _dino3_quadrant_patch_pool(tokens)
    assert output.shape == (1, 1536)
    for index in range(4): assert torch.equal(output[:, index * 384:(index + 1) * 384], torch.full((1, 384), float(index + 1)))
    output.sum().backward()
    assert torch.count_nonzero(tokens.grad[:, :5]) == 0 and torch.all(tokens.grad[:, 5:] > 0)
    try: _dino3_quadrant_patch_pool(tokens[:, :-1])
    except ValueError: pass
    else: raise AssertionError('invalid patch count accepted')


def test_next_identities_are_fixed_and_keep_final_only_official_gate():
    from retrain import experiment91 as ex
    from retrain.config import VIT_DINO3_QUADRANT_RULE
    specs = (
        ('R255', 'R184R1', 20261110, 165,
         'single-vit-small-dinov3-3bf4720a-fixed-9010-24k-earlystop-seed20261110-scale085-pairlog-quadrantpool-v1',
         'r255-vits16-dinov3-seed20261110-quadrantpool-pairlog-24k-p4-9010-20261104',
         'r255-vits16-dinov3-lvd1689m-24k-p4-seed20261110-scale085-pairlog-quadrantpool-9010'),
        ('R256', 'R120', 20261111, 166,
         'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261111-scale085-pairsigmoid-v1',
         'r256-small-seed20261111-pairsigmoid-24k-p4-9010-20261104',
         'r256-convnext-small-imagenet22k-to-1k-24k-p4-seed20261111-scale085-pairsigmoid-9010'),
    )
    for label, original, seed, order, profile, rid, candidate_id in specs:
        base = json.loads((ROOT / 'configs' / (original + '-SCIENCE.json')).read_text())
        value = copy.deepcopy(base)
        value.update(candidate_profile=profile, official_evaluation=ex.OFFICIAL_FINAL_ENSEMBLE_ONLY)
        value['job']['seed'] = seed
        value['candidate'].update(id=candidate_id, experiment_order=order)
        if label == 'R255':
            value['job']['vit_dino3_pool_policy'] = VIT_DINO3_QUADRANT_RULE
            value['candidate'].update(readout='native_final_LN_quadrant_patch_concat_dropout_linear_plus_age',
                                      vit_dino3_pool_policy=VIT_DINO3_QUADRANT_RULE)
        assert profile in ex.SUPPORTED_BINDINGS
        assert ex.candidate_from_configuration(value) == value['candidate']
        assert configured(ex, value, rid) == value
        assert ex.expected_official_policy({'round_id': rid}, value) == ex.OFFICIAL_FINAL_ENSEMBLE_ONLY
        assert not ex.official_allowed_for_selected(ex.OFFICIAL_FINAL_ENSEMBLE_ONLY, {'auroc': 1.0})
        altered = copy.deepcopy(value); altered['job']['seed'] = 20261029
        try: ex.candidate_from_configuration(altered)
        except ValueError: pass
        else: raise AssertionError('changed fixed seed accepted')


def test_native_dinov3_forward_uses_fixed_spatial_readout_once():
    from retrain import age, experiment91 as ex, models
    from retrain.config import VIT_DINO3_QUADRANT_RULE
    value = copy.deepcopy(json.loads((ROOT / 'configs/R184R1-SCIENCE.json').read_text())['job'])
    value.update(initialization={'kind': 'random'}, vit_dino3_pool_policy=VIT_DINO3_QUADRANT_RULE,
                 age=age.make_spec({'rows': [{'name': str(i), 'sha256': str(i) * 64, 'age': str(30 + i)}
                                             for i in range(4)], 'fit_indices': [0, 1, 2, 3]}))
    torch.manual_seed(20261110)
    model = models._create_vit_dino3_binary(value, initialize=False).eval()
    assert model.classifier[-1].in_features == 1536 and len(dict(model.named_parameters())) == 165
    assert models.describe_vit_dino3(model)['vit_dino3_pool_policy'] == VIT_DINO3_QUADRANT_RULE
    science = json.loads((ROOT / 'configs/R184R1-SCIENCE.json').read_text())
    candidate = copy.deepcopy(science['candidate'])
    candidate['vit_dino3_pool_policy'] = VIT_DINO3_QUADRANT_RULE
    with mock.patch.object(ex, 'candidate', return_value=candidate), mock.patch.object(ex, 'science_configuration', return_value=science):
        assert ex.describe_model(model)['head_in_features'] == 1536
    images = torch.randn(1, 3, 512, 512); ages = torch.tensor([[.3, 1.]])
    with torch.inference_mode():
        native = model.network.forward_features(images)
        grid = native[:, 5:].reshape(1, 32, 32, 384)
        manual = torch.cat([grid[:, :16, :16].mean((1, 2)), grid[:, :16, 16:].mean((1, 2)),
                            grid[:, 16:, :16].mean((1, 2)), grid[:, 16:, 16:].mean((1, 2))], dim=1)
        expected = model.classifier(manual).flatten() + (ages * model.age_weight).sum(1)
        torch.testing.assert_close(model(images, ages), expected, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


if __name__ == '__main__':
    test_spatial_patch_pool_excludes_prefix_and_preserves_quadrant_order()
    test_next_identities_are_fixed_and_keep_final_only_official_gate()
    test_native_dinov3_forward_uses_fixed_spatial_readout_once()
    print('R255/R256 fixed spatial pooling, seed and official policy checks passed', flush=True)
