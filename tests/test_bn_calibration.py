"""Three small CPU checks; parent A runs them in the guarded validation pool."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))
from retrain.bn_calibration import calibrate_fit_bn, _merge_moments


SPEC = {
    'version': 'fit-bn-fixed-reference-v1',
    'estimator': 'pooled_input_moments',
    'reference': 'public_bn_eval',
    'accumulation_dtype': 'float64',
    'variance': 'unbiased',
    'output_dtype': 'float32',
}


class BNCalibrationTest(unittest.TestCase):
    def test_pooled_unbiased_moments_weight_unequal_batches_and_shards(self):
        import torch
        from torch import nn

        model = nn.Module()
        model.backbone = nn.Sequential(nn.BatchNorm2d(1))
        batches = [torch.tensor([1., 2., 3., 4.]).reshape(2, 1, 1, 2),
                   torch.tensor([5., 6.]).reshape(1, 1, 1, 2)]
        receipt = calibrate_fit_bn(model, iter(batches), SPEC)
        torch.testing.assert_close(model.backbone[0].running_mean, torch.tensor([3.5]))
        torch.testing.assert_close(model.backbone[0].running_var, torch.tensor([3.5]))
        self.assertEqual(receipt['per_channel_count'], {'backbone.0': 6})
        self.assertEqual((receipt['local_images'], receipt['global_images']), (3, 3))
        self.assertEqual((receipt['local_batches'], receipt['global_batches']), (2, 2))

        # Independent shard summaries: [1,2,3,4] and [5,6]. Their concatenation
        # has mean 3.5, M2 17.5, and unbiased variance 17.5 / 5 = 3.5.
        count, mean, m2 = _merge_moments(
            (4, torch.tensor([2.5], dtype=torch.float64),
             torch.tensor([5.], dtype=torch.float64)),
            (2, torch.tensor([5.5], dtype=torch.float64),
             torch.tensor([.5], dtype=torch.float64)))
        self.assertEqual(count, 6)
        torch.testing.assert_close(mean, torch.tensor([3.5], dtype=torch.float64))
        torch.testing.assert_close(m2, torch.tensor([17.5], dtype=torch.float64))
        torch.testing.assert_close(m2 / (count - 1),
                                   model.backbone[0].running_var.double())

    def test_two_layers_observe_the_public_eval_reference_for_the_entire_pass(self):
        import torch
        from torch import nn

        model = nn.Module()
        first, second = nn.BatchNorm2d(1, eps=1.), nn.BatchNorm2d(1, eps=1.)
        convolution = nn.Conv2d(1, 1, 1, bias=False)
        model.backbone = nn.Sequential(first, convolution, second)
        with torch.no_grad():
            first.running_mean.fill_(10.)
            first.running_var.fill_(3.)
            first.num_batches_tracked.fill_(13)
            convolution.weight.fill_(2.)
            second.running_mean.fill_(-5.)
            second.running_var.fill_(8.)
            second.num_batches_tracked.fill_(19)
        public = {name: value.clone() for name, value in model.named_buffers()}
        batches = [torch.tensor([1., 2., 3., 4.]).reshape(2, 1, 1, 2),
                   torch.tensor([9., 10.]).reshape(1, 1, 1, 2)]
        visits = []

        def once():
            for images in batches:
                self.assertTrue(all(not module.training for module in model.modules()))
                for name, value in model.named_buffers():
                    self.assertTrue(torch.equal(value, public[name]), name)
                visits.append(images.shape[0])
                yield images
            # The final batch also finishes before any running buffer changes.
            for name, value in model.named_buffers():
                self.assertTrue(torch.equal(value, public[name]), name)

        receipt = calibrate_fit_bn(model, once(), SPEC)
        original_inputs = torch.cat(batches).double().flatten()
        # Public BN1 computes (x-10)/sqrt(3+1), then the 1x1 weight is 2.
        second_inputs = original_inputs - 10.
        torch.testing.assert_close(first.running_mean.double(), original_inputs.mean().reshape(1),
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(second.running_mean.double(), second_inputs.mean().reshape(1),
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(second.running_var.double(), second_inputs.var(unbiased=True).reshape(1),
                                   rtol=1e-6, atol=1e-6)
        self.assertEqual(visits, [2, 1])
        self.assertEqual(receipt['layer_count'], 2)
        self.assertEqual(receipt['per_channel_count'], {'backbone.0': 6, 'backbone.2': 6})

    def test_parameters_counters_modes_rng_and_no_grad_are_preserved(self):
        import torch
        from torch import nn

        model = nn.Module()
        model.backbone = nn.Sequential(nn.BatchNorm2d(2), nn.Dropout(.8),
                                       nn.Conv2d(2, 2, 1, bias=False), nn.BatchNorm2d(2))
        model.backbone[0].eval()
        model.backbone[2].eval()
        with torch.no_grad():
            model.backbone[0].num_batches_tracked.fill_(7)
            model.backbone[3].num_batches_tracked.fill_(11)
        modes = [(module, module.training) for module in model.modules()]
        parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        counters = {name: value.clone() for name, value in model.named_buffers()
                    if name.endswith('num_batches_tracked')}
        images = torch.arange(24, dtype=torch.float32).reshape(3, 2, 2, 2).requires_grad_()
        torch.manual_seed(20261029)
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        observed = []

        def observe(module, args):
            observed.append((torch.is_grad_enabled(), [m.training for m in model.modules()]))

        handle = model.backbone.register_forward_pre_hook(observe)
        self.addCleanup(handle.remove)

        def batches():
            # Data iteration is inside the isolated Torch RNG scope as well.
            torch.rand(1)
            yield images[:2]
            torch.rand(1)
            yield images[2:]

        receipt = calibrate_fit_bn(model, batches(), SPEC)
        self.assertEqual(observed, [(False, [False] * len(modes))] * 2)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, parameters[name]), name)
            self.assertIsNone(parameter.grad)
        self.assertIsNone(images.grad)
        for name, value in model.named_buffers():
            if name in counters:
                self.assertTrue(torch.equal(value, counters[name]), name)
            else:
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(value).all()))
                if name.endswith('running_var'):
                    self.assertTrue(bool((value >= 0).all()))
        self.assertEqual([(module, module.training) for module in model.modules()], modes)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for actual, old in zip(torch.cuda.get_rng_state_all() if cuda_rng else [], cuda_rng):
            self.assertTrue(torch.equal(actual, old))
        self.assertTrue(receipt['passed'])
        self.assertEqual(receipt['spec'], SPEC)
        self.assertEqual(receipt['scope'], 'fit_bn_fixed_reference_calibration')
        self.assertEqual((receipt['floating_buffer_count'], receipt['counter_buffer_count']), (4, 2))
        self.assertEqual(receipt['per_channel_count'], {'backbone.0': 12, 'backbone.3': 12})
        self.assertNotEqual(receipt['reference_bn_digest_before'], receipt['derived_bn_digest_after'])
        for key in ('parameters_unchanged', 'counters_unchanged', 'mode_rng_restored'):
            self.assertIs(receipt[key], True)

        buffers = {name: value.clone() for name, value in model.named_buffers()}

        def interrupted():
            yield images[:2]
            torch.rand(1)
            raise RuntimeError('fixture interrupted before the pass completed')

        with self.assertRaisesRegex(RuntimeError, 'fixture interrupted'):
            calibrate_fit_bn(model, interrupted(), SPEC)
        self.assertEqual([(module, module.training) for module in model.modules()], modes)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value, buffers[name]), name)


if __name__ == '__main__':
    unittest.main()
