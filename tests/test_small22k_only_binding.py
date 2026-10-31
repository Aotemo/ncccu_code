"""R129 pinned source identity and initialize=False strict binary reload fixture."""
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime' / 'bundle'))
from retrain.models import create_model, SMALL_22K_ONLY_FORMAT


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.LayerNorm(4), nn.Flatten(1), nn.Linear(4, 1000))


class Small22KOnlyBindingTest(unittest.TestCase):
    def test_science_source_and_binary_reload_without_public_asset(self):
        science = json.loads((ROOT / 'configs' / 'R129-SCIENCE.json').read_text())
        source = science['public_source']
        self.assertEqual(science['candidate_profile'],
                         'single-convnext-small-in22konly-fixed-9010-24k-earlystop-seed20261029-scale085-pairlog-v1')
        self.assertEqual((source['format'], source['num_classes'], source['revision']),
                         (SMALL_22K_ONLY_FORMAT, 21841,
                          '4cde7ae59c8f7fcff3f9547de274819f54863dfd'))
        self.assertEqual(science['structure']['full_head_shape'], [21841, 768])
        self.assertEqual(science['candidate']['weight_sha256'], source['sha256'])
        self.assertEqual(science['candidate']['provenance_sha256'], source['provenance_sha256'])
        self.assertEqual(science['job']['loss']['kind'], 'binary_pairwise_logistic')
        self.assertEqual(science['job']['seed'], 20261029)
        self.assertTrue(all(science['job'].get(key) is None for key in
                            ('head_warmup_policy', 'training_rotation', 'training_photometric',
                             'gradient_clip_policy', 'attention_pool', 'stage3_residual')))
        self.assertEqual(science['official_evaluation']['threshold'], .996)

        job = {'model': 'convnext_small', 'stochastic_depth': .1, 'final_norm_eps': 1e-6,
               'dropout': .2, 'experiment91': {'binding': 'synthetic'},
               'initialization': {'kind': 'imagenet', 'format': SMALL_22K_ONLY_FORMAT,
                                  'path': '/nonexistent/public-K.pt',
                                  'provenance': '/nonexistent/provenance.json', 'sha256': source['sha256']},
               'age': {'enabled': True}}

        def attach_age(model, _spec):
            model.register_parameter('age_weight', nn.Parameter(torch.zeros(2)))

        with (patch('torchvision.models.convnext_small', side_effect=lambda **_kwargs: TinyNetwork()),
              patch('retrain.experiment91.validate_job', return_value=job),
              patch('retrain.experiment91.describe_model', return_value={'synthetic': True}),
              patch('retrain.age.attach_age', side_effect=attach_age),
              patch('retrain.models.validate_initialization', side_effect=AssertionError('public source reloaded')),
              patch('torch.load', wraps=torch.load) as load):
            first = create_model(job, initialize=False)
            payload = io.BytesIO()
            torch.save(first.state_dict(), payload)
            payload.seek(0)
            saved = torch.load(payload, weights_only=True)
            second = create_model(job, initialize=False)
            loaded = second.load_state_dict(saved, strict=True)
            self.assertFalse(loaded.missing_keys or loaded.unexpected_keys)
            self.assertEqual(tuple(second.network.classifier[-1].weight.shape), (1, 4))
            self.assertIn('age_weight', second.state_dict())
            self.assertEqual(load.call_count, 1)  # The fixture's in-memory checkpoint only.
            self.assertTrue(all(torch.equal(first.state_dict()[k], second.state_dict()[k])
                                for k in saved))
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
