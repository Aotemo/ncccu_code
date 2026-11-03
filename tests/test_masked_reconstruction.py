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

    def test_hog_target_has_independent_planar_gradient_oracle_and_no_target_gradient(self):
        self.assertTrue(hasattr(mim, 'hog_target'), 'fixed local gradient target is missing')
        axis = torch.arange(512, dtype=torch.float32)
        xx, yy = axis[None, :].expand(512, -1), axis[:, None].expand(-1, 512)
        image = torch.stack((xx, yy, xx + yy))[None].requires_grad_()
        target = mim.hog_target(image)
        self.assertEqual(target.shape, (1, 27, 64, 64))
        self.assertFalse(target.requires_grad)
        self.assertTrue(torch.isfinite(target).all())
        # Interior affine fields have exact unsigned directions 0, 90, 45deg.
        expected = torch.zeros(1, 3, 9, 62, 62)
        expected[:, 0, 0] = 1.
        expected[:, 1, 4] = 1.
        expected[:, 2, 2] = 1.
        torch.testing.assert_close(target.reshape(1, 3, 9, 64, 64)[:, :, :, 1:-1, 1:-1], expected)
        torch.testing.assert_close(mim.hog_target(image.detach() * 3. + 17.), target)
        torch.testing.assert_close(mim.hog_target(torch.zeros_like(image)), torch.zeros_like(target))

    def test_hog_hidden_loss_target_detach_and_DDP_gradient_match_global_oracle(self):
        self.assertTrue(hasattr(mim, 'masked_hog_mse'), 'hidden HOG loss is missing')
        x = torch.linspace(-2., 2., 64 * 27 * 4 * 4).reshape(64, 27, 4, 4)
        target = torch.cos(x).requires_grad_()
        mask = torch.zeros(64, 1, 4, 4, dtype=torch.bool)
        mask[:, :, ::2, 1::2] = True
        weight = torch.tensor(.73, requires_grad=True)
        loss = mim.masked_hog_mse(weight * x, target, mask)
        oracle = ((weight * x - target.detach()).square()).masked_select(mask.expand_as(x)).mean()
        torch.testing.assert_close(loss, oracle)
        global_grad, = torch.autograd.grad(loss, weight, retain_graph=True)
        self.assertIsNone(torch.autograd.grad(loss, target, allow_unused=True)[0])
        changed = target.detach().clone().masked_fill(~mask.expand_as(target), 1e6)
        torch.testing.assert_close(mim.masked_hog_mse(weight * x, changed, mask), loss)
        rank_grads = []
        for rank in range(8):
            a, b = rank * 8, (rank + 1) * 8
            local = mim.masked_hog_mse(weight * x[a:b], target[a:b], mask[a:b])
            grad, = torch.autograd.grad(local, weight)
            rank_grads.append(grad)
        torch.testing.assert_close(torch.stack(rank_grads).mean(), global_grad)

    def test_hog_decoder_hidden_pixel_leakage_and_encoder_only_update(self):
        self.assertTrue(hasattr(mim, 'HOG_RECIPE'), 'HOG signal is missing')
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
        before, head = tensor_digest(_derived_state(binary)), tensor_digest(_non_encoder_state(binary))
        model = mim.MaskedReconstructionModel(binary, projector_seed=81, target='hog')
        image = torch.randn(1, 3, 512, 512, requires_grad=True)
        mask = mim.spatial_mask([91])
        prediction = model(image, mask)
        self.assertEqual(prediction.shape, (1, 27, 64, 64))
        modified = image.detach().clone().masked_fill(mask.expand_as(image), 1e4)
        torch.testing.assert_close(model(modified, mask), prediction.detach(), rtol=0, atol=0)
        prediction.mean().backward()
        self.assertEqual(int(torch.count_nonzero(image.grad.masked_select(mask.expand_as(image)))), 0)
        optimizer = _optimizer(model)
        optimizer.zero_grad(set_to_none=True)
        loss = mim.masked_hog_mse(model(image.detach(), mask), mim.hog_target(image), mask[:, :, ::8, ::8])
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertNotEqual(tensor_digest(_derived_state(binary)), before)
        self.assertEqual(tensor_digest(_non_encoder_state(binary)), head)

    def test_hog_recipe_fixes_target_and_reuses_R224_actual_sampling_and_mask_stream(self):
        self.assertTrue(hasattr(mim, 'HOG_RECIPE'), 'HOG recipe is missing')
        from retrain.data import _seed
        from retrain.ssl_stage import validate_recipe, ssl_rng_derivation, UniformFitSampler
        job = {'model': 'convnext_small', 'image_size': 512}
        manifest = {'fit_indices': list(range(64)), 'dev_indices': [64, 65], 'audit_indices': []}
        recipe = copy.deepcopy(mim.HOG_RECIPE)
        self.assertEqual(validate_recipe(recipe, job, manifest, 8), recipe)
        self.assertEqual(recipe['updates'], 1600)
        self.assertEqual(recipe['projector'], [768, 432, 4])
        rgb = ssl_rng_derivation(mim.MASKED_RECIPE, 20261029, 8)
        hog = ssl_rng_derivation(recipe, 20261029, 8)
        for key in ('sampler_seed', 'model_seeds', 'projector_seed', 'loader_seeds', 'learning_domain'):
            self.assertEqual(hog[key], rgb[key])
        expected = _seed(20261029, 'r115-fit-vicreg-1600-v1', 'sampler')
        self.assertEqual(rgb['sampler_seed'], expected)
        declared = list(UniformFitSampler(list(range(70)), seed=rgb['sampler_seed'], updates=2, rank=0))
        original = list(UniformFitSampler(list(range(70)), seed=expected, updates=2, rank=0))
        self.assertEqual(declared, original)
        seed = _seed(20261029, hog['learning_domain'], 'mask', 1, 0, 17)
        self.assertTrue(torch.equal(mim.spatial_mask([seed]), mim.spatial_mask([_seed(20261029, 'fit-masked-rgb32-1600-v1', 'mask', 1, 0, 17)])))
        for key, changed in [('updates', 3200), ('hog_cell_size', 16), ('hog_bins', 8), ('mask_count', 153)]:
            wrong = copy.deepcopy(recipe); wrong[key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recipe(wrong, job, manifest, 8)


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ['NCCCU_CPU_THREADS']))
    torch.set_num_interop_threads(1)
    unittest.main()
