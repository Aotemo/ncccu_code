"""Masked spatial signal and fit-only handoff; run in the guarded Pod CPU pool."""
import copy
import importlib
import os
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
try:
    mim = importlib.import_module('retrain.masked_reconstruction')
except ModuleNotFoundError:
    mim = None


class MaskedReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(mim, 'masked spatial reconstruction signal is missing')

    def test_mask_exact_count_stateless_order_and_masked_pixel_only_loss(self):
        mask = mim.spatial_mask([17, 38])
        self.assertEqual(mask.shape, (2, 1, 512, 512))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.sum((1, 2, 3)).tolist(), [154 * 32 * 32] * 2)
        self.assertTrue(torch.equal(mask[0], mim.spatial_mask([17])[0]))
        self.assertTrue(torch.equal(mask.flip(0), mim.spatial_mask([38, 17])))
        self.assertFalse(torch.equal(mask[0], mask[1]))
        self.assertEqual(mim.spatial_mask([2 ** 64 - 1]).shape, (1, 1, 512, 512))
        for invalid in (-1, 2 ** 64):
            with self.assertRaises(ValueError):
                mim.spatial_mask([invalid])
        self.assertTrue(torch.equal(mask, mask[:, :, ::32, ::32].repeat_interleave(32, 2).repeat_interleave(32, 3)))
        selected = torch.tensor([[[[True, False], [False, True]]]])
        prediction = torch.ones(1, 3, 2, 2, requires_grad=True)
        target = torch.zeros_like(prediction)
        target[:, :, 0, 1] = 1e6
        loss = mim.masked_l1(prediction, target, selected)
        self.assertEqual(float(loss), 1.)
        loss.backward()
        expected = selected.expand_as(prediction).float() / 6
        torch.testing.assert_close(prediction.grad, expected)

    def test_hidden_pixels_cannot_reach_prediction_and_head_age_are_untouched(self):
        from retrain.ssl_stage import _derived_state, _non_encoder_state, _optimizer
        from retrain.ema import tensor_digest

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Module()
                self.network.features = nn.Sequential(nn.Conv2d(3, 768, 4, stride=4), nn.AvgPool2d(8))
                self.network.classifier = nn.Sequential(nn.LayerNorm([768, 1, 1]), nn.Flatten(),
                                                        nn.Dropout(.2), nn.Linear(768, 1))
                self.age_weight = nn.Parameter(torch.zeros(2))

        binary = Toy()
        before = tensor_digest(_derived_state(binary))
        head = tensor_digest(_non_encoder_state(binary))
        rng = torch.get_rng_state().clone()
        model = mim.MaskedReconstructionModel(binary, projector_seed=81)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        x = torch.randn(2, 3, 512, 512, requires_grad=True)
        mask = mim.spatial_mask([91, 92])
        prediction = model(x, mask)
        self.assertEqual(prediction.shape, x.shape)
        modified = x.detach().clone().masked_fill(mask.expand_as(x), 1e4)
        torch.testing.assert_close(model(modified, mask), prediction.detach(), rtol=0, atol=0)
        prediction.mean().backward()
        self.assertEqual(int(torch.count_nonzero(x.grad.masked_select(mask.expand_as(x)))), 0)
        optimizer = _optimizer(model)
        owned = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertNotIn(id(binary.age_weight), owned)
        self.assertNotIn(id(binary.network.classifier[-1].weight), owned)
        self.assertEqual(optimizer.state, {})
        optimizer.zero_grad(set_to_none=True)
        loss = mim.masked_l1(model(x.detach(), mask), x.detach(), mask)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        self.assertNotEqual(tensor_digest(_derived_state(binary)), before)
        self.assertEqual(tensor_digest(_non_encoder_state(binary)), head)

    def test_new_recipe_and_geometry_keep_legacy_recipe_fit_only_and_fixed(self):
        from retrain.ssl_stage import validate_recipe, ssl_head_shape
        job = {'model': 'convnext_small', 'image_size': 512}
        manifest = {'fit_indices': list(range(64)), 'dev_indices': [64, 65], 'audit_indices': []}
        recipe = copy.deepcopy(mim.MASKED_RECIPE)
        self.assertEqual(validate_recipe(recipe, job, manifest, 8), recipe)
        self.assertEqual(ssl_head_shape(recipe), [768, 3072, 32])
        self.assertEqual(recipe['updates'], 1600)
        self.assertEqual(recipe['training_views'], 1)
        legacy = __import__('json').loads((ROOT / 'configs/R115-SCIENCE.json').read_text())['ssl_recipe']
        self.assertEqual(validate_recipe(legacy, job, manifest, 8), legacy)
        self.assertEqual(ssl_head_shape(legacy), [768, 1024, 256])
        for key, new in [('updates', 3200), ('mask_count', 153), ('training_views', 2), ('mask_patch_size', 16)]:
            wrong = copy.deepcopy(recipe)
            wrong[key] = new
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recipe(wrong, job, manifest, 8)
        with self.assertRaises(ValueError):
            validate_recipe(recipe, job, {**manifest, 'dev_indices': [0, 65]}, 8)
        with self.assertRaises(ValueError):
            validate_recipe(recipe, job, manifest, 4)

    def test_equal_hidden_counts_make_DDP_rank_mean_match_global_gradient(self):
        # Independent rank reduction oracle: no embedding gather is used by MIM.
        x = torch.linspace(-2., 2., 64 * 3 * 4 * 4).reshape(64, 3, 4, 4)
        target = torch.sin(x * 3.)
        mask = torch.zeros(64, 1, 4, 4, dtype=torch.bool)
        mask[:, :, ::2, 1::2] = True
        weight = torch.tensor(.73, requires_grad=True)
        global_loss = mim.masked_l1(weight * x, target, mask)
        global_grad, = torch.autograd.grad(global_loss, weight)
        rank_losses, rank_grads = [], []
        for rank in range(8):
            start, end = rank * 8, (rank + 1) * 8
            local = mim.masked_l1(weight * x[start:end], target[start:end], mask[start:end])
            local_grad, = torch.autograd.grad(local, weight)
            rank_losses.append(local.detach())
            rank_grads.append(local_grad)
        torch.testing.assert_close(torch.stack(rank_losses).mean(), global_loss.detach())
        torch.testing.assert_close(torch.stack(rank_grads).mean(), global_grad)


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ['NCCCU_CPU_THREADS']))
    torch.set_num_interop_threads(1)
    unittest.main()
