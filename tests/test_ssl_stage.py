"""Narrow CPU checks for the same-owner fit-only-to-supervised handoff."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))


class SSLStageTest(unittest.TestCase):
    def test_local_candidates_keep_complete_R107_supervision_and_original_ssl_seeds(self):
        import copy
        import json
        from retrain import experiment91 as ex
        from retrain.ssl_stage import LOCAL_RECIPE, ssl_rng_derivation, validate_recipe
        original = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        manifest = {'fit_indices': list(range(64)), 'dev_indices': [64], 'audit_indices': []}
        for label, order, seed, binding in (
                ('R228', 143, 20261029, ex.FIT_LOCAL_SMALL_BINDING),
                ('R229', 144, 20261027, ex.FIT_LOCAL_SMALL_SEED27_BINDING)):
            value = json.loads((ROOT / 'configs' / (label + '-SCIENCE.json')).read_text())
            self.assertEqual(value['candidate_profile'], binding)
            expected = copy.deepcopy(original['job']); expected['seed'] = seed
            self.assertEqual(value['job'], expected)
            for key in ('public_source', 'public_asset_receipt', 'paths', 'structure',
                        'data_split', 'schedule', 'early_stopping'):
                self.assertEqual(value[key], original[key])
            self.assertEqual(value['ssl_recipe'], LOCAL_RECIPE)
            self.assertEqual(value['candidate']['experiment_order'], order)
            self.assertEqual(value['official_evaluation'], ex.OFFICIAL_GATE_0996)
            self.assertEqual(ex.candidate_from_configuration(value), value['candidate'])
            self.assertEqual(validate_recipe(LOCAL_RECIPE, value['job'], manifest, 8), LOCAL_RECIPE)
            seeds = ssl_rng_derivation(LOCAL_RECIPE, seed, 8)
            legacy = ssl_rng_derivation({'version': 'r115-fit-vicreg-1600-v1'}, seed, 8)
            for key in ('projector_seed', 'sampler_seed', 'loader_seeds', 'model_seeds'):
                self.assertEqual(seeds[key], legacy[key])
        with self.assertRaises(ValueError):
            validate_recipe(LOCAL_RECIPE, original['job'], dict(manifest, dev_indices=[0]), 8)

    def test_local_signal_detects_spatial_permutation_hidden_by_global_pooling(self):
        import torch
        from torch import nn
        from retrain.ema import tensor_digest
        from retrain.ssl_stage import LocalSSLModel, SSLModel, _derived_state, _non_encoder_state, _optimizer
        from retrain.vicreg import vicreg_loss
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Module()
                self.network.features = nn.Conv2d(3, 768, 1)
                self.network.avgpool = nn.AdaptiveAvgPool2d(1)
                self.network.classifier = nn.Sequential(nn.LayerNorm([768, 1, 1]), nn.Flatten(),
                                                        nn.Dropout(.2), nn.Linear(768, 1))
                self.age_weight = nn.Parameter(torch.zeros(2))
        torch.manual_seed(31)
        binary = Toy()
        public_head = tensor_digest(_non_encoder_state(binary))
        encoder_before = tensor_digest(_derived_state(binary))
        rng = torch.get_rng_state().clone()
        local = LocalSSLModel(binary, projector_seed=123)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        global_model = SSLModel(binary, projector_seed=123)
        self.assertEqual(tensor_digest(local.projector.state_dict()), tensor_digest(global_model.projector.state_dict()))
        images = torch.randn(2, 3, 4, 4)
        permuted = images.flip(-1)
        self.assertTrue(torch.allclose(global_model(images), global_model(permuted), atol=2e-7, rtol=1e-5))
        a, b = local(images), local(permuted)
        self.assertEqual(a.shape, (2, 4, 4, 256))
        self.assertTrue(torch.allclose(b, a.flip(2), atol=2e-7, rtol=1e-5))
        aligned = vicreg_loss(a.reshape(-1, 256), a.reshape(-1, 256))
        shuffled = vicreg_loss(a.reshape(-1, 256), b.reshape(-1, 256))
        # Permutation leaves variance/covariance unchanged; the difference is
        # exactly the spatial pairing term, independently of the model code.
        self.assertAlmostEqual(float(shuffled - aligned), float(25 * (a - b).square().mean()), places=4)
        self.assertGreater(float(shuffled - aligned), .001)
        optimizer = _optimizer(local)
        owned = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertNotIn(id(binary.network.classifier[-1].weight), owned)
        self.assertNotIn(id(binary.age_weight), owned)
        shuffled.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in local.parameters()))
        optimizer.step()
        self.assertNotEqual(tensor_digest(_derived_state(binary)), encoder_before)
        self.assertEqual(tensor_digest(_non_encoder_state(binary)), public_head)

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
