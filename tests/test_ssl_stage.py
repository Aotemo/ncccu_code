"""Narrow CPU checks for the same-owner fit-only-to-supervised handoff."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))


class SSLStageTest(unittest.TestCase):
    def test_uniform_full_batch_fit_only_rank_shards_and_lr_endpoints(self):
        import random
        from retrain.data import _seed
        from retrain.ssl_stage import UniformFitSampler, _factor
        fit = list(range(73))
        q = _seed(20261029, 'r115-fit-vicreg-1600-v1', 'sampler')
        ranks = [list(UniformFitSampler(fit, seed=q, updates=3, rank=r)) for r in range(8)]
        expected = fit.copy()
        random.Random(_seed(q, 'epoch', 0)).shuffle(expected)
        first = [key for rank in ranks for key in rank[0]]
        self.assertEqual([index for index, _ in first], expected[:64])
        self.assertEqual(first[0][1], _seed(q, 'sample', 0, 0, first[0][0]))
        self.assertNotEqual(first[0][1], _seed(first[0][1], 'training-view', 1))
        for update in range(3):
            batch = [key for rank in ranks for key in rank[update]]
            self.assertEqual(len(batch), 64)
            self.assertEqual(len({index for index, _ in batch}), 64)
            self.assertTrue({index for index, _ in batch} <= set(fit))
        self.assertEqual(ranks, [list(UniformFitSampler(fit, seed=q, updates=3, rank=r))
                                 for r in range(8)])
        self.assertAlmostEqual(_factor(1), 1 / 160)
        self.assertEqual(_factor(160), 1)
        self.assertTrue(_factor(161) < 1)
        self.assertAlmostEqual(_factor(1600), .01)

    def test_projector_updates_encoder_only_and_derived_load_keeps_public_head_age(self):
        import torch
        from torch import nn
        from retrain.ema import tensor_digest
        from retrain.persistence import sha256_file
        from retrain.ssl_stage import SSLModel, _derived_state, _non_encoder_state, _optimizer, load_derived_model

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(31)
                self.network = nn.Module()
                self.network.features = nn.Sequential(nn.Conv2d(3, 768, 1))
                self.network.avgpool = nn.AdaptiveAvgPool2d(1)
                self.network.classifier = nn.Sequential(nn.LayerNorm([768, 1, 1]), nn.Flatten(),
                                                        nn.Dropout(.2), nn.Linear(768, 1))
                self.age_weight = nn.Parameter(torch.zeros(2))

        binary = Toy()
        public_head = tensor_digest(_non_encoder_state(binary))
        before_encoder = tensor_digest(_derived_state(binary))
        rng = torch.get_rng_state().clone()
        model = SSLModel(binary, projector_seed=123)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        optimizer = _optimizer(model)
        owned = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertNotIn(id(binary.network.classifier[-1].weight), owned)
        self.assertNotIn(id(binary.age_weight), owned)
        self.assertEqual(model(torch.randn(4, 3, 4, 4)).shape, (4, 256))
        loss = model(torch.randn(4, 3, 4, 4)).square().mean()
        loss.backward()
        optimizer.step()
        self.assertNotEqual(tensor_digest(_derived_state(binary)), before_encoder)
        self.assertEqual(tensor_digest(_non_encoder_state(binary)), public_head)
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / 'derived.pt'
            state = _derived_state(binary)
            torch.save(state, asset)
            receipt = {'asset_path': str(asset), 'asset_sha256': sha256_file(asset),
                       'derived_tensor_sha256': tensor_digest(state),
                       'public_non_encoder_sha256': public_head}
            with mock.patch('retrain.ssl_stage.create_model', side_effect=lambda *_args, **_kwargs: Toy()):
                loaded = load_derived_model({'model': 'convnext_small'}, True, receipt)
            self.assertEqual(tensor_digest(_derived_state(loaded)), receipt['derived_tensor_sha256'])
            self.assertEqual(tensor_digest(_non_encoder_state(loaded)), public_head)


if __name__ == '__main__':
    unittest.main()
