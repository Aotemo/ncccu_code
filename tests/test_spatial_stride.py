"""Three tiny CPU fixtures; only parent A executes them in the guarded Pod."""
import io
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))


def tiny_model():
    import torch
    from torch import nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Module()
            self.network.features = nn.Sequential(
                *[nn.Identity() for _ in range(6)],
                nn.Sequential(nn.Identity(), nn.Conv2d(2, 3, kernel_size=2, stride=2)),
                nn.Sequential(*[nn.Conv2d(3, 3, kernel_size=1) for _ in range(3)]),
            )
            self.register_buffer('marker', torch.tensor([3.]))

        def forward(self, values):
            return self.network.features(values)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(114)
        return TinyModel().double()


def input_values():
    import torch
    return torch.linspace(-.8, 1.2, 2 * 2 * 32 * 32,
                          dtype=torch.float64).reshape(2, 2, 32, 32)


class SpatialStrideTest(unittest.TestCase):
    def test_stride_changes_map_size_without_changing_tensors_identity_modes_or_rng(self):
        # Catches replacement/reinitialization, mode changes, and stride applied to the wrong layer.
        import torch
        from torch import nn
        from retrain.spatial_stride import apply_final_downsample_stride

        model = tiny_model()
        model.network.features[7][1].eval()
        values = input_values()
        self.assertEqual(tuple(model(values).shape), (2, 3, 16, 16))
        state = {name: value.clone() for name, value in model.state_dict().items()}
        modules = [(name, id(module), module.training) for name, module in model.named_modules()]
        parameters = [(name, id(value), value.requires_grad) for name, value in model.named_parameters()]
        geometry = {name: (module.kernel_size, module.padding, module.dilation,
                           module.stride, module.groups, module.padding_mode)
                    for name, module in model.named_modules() if isinstance(module, nn.Conv2d)}
        rng = torch.get_rng_state().clone()
        python_rng = random.getstate()

        self.assertIsNone(apply_final_downsample_stride(
            model, {'model': 'convnext_small', 'final_downsample_stride': 1}))

        self.assertEqual([(name, id(module), module.training)
                          for name, module in model.named_modules()], modules)
        self.assertEqual([(name, id(value), value.requires_grad)
                          for name, value in model.named_parameters()], parameters)
        self.assertEqual(list(model.state_dict()), list(state))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state[name]), name)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                kernel, padding, dilation, stride, groups, padding_mode = geometry[name]
                expected_stride = (1, 1) if name == 'network.features.6.1' else stride
                self.assertEqual((module.kernel_size, module.padding, module.dilation,
                                  module.stride, module.groups, module.padding_mode),
                                 (kernel, padding, dilation, expected_stride, groups, padding_mode), name)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(random.getstate(), python_rng)
        self.assertEqual(tuple(model(values).shape), (2, 3, 31, 31))

    def test_real_forward_and_backward_match_independent_stride_one_functional_reference(self):
        # Catches detach/no_grad/freezing and a stride edit that changes real outputs or derivatives.
        import torch
        from torch.nn import functional as F
        from retrain.spatial_stride import apply_final_downsample_stride

        model = tiny_model()
        values = input_values().requires_grad_(True)
        reference_values = values.detach().clone().requires_grad_(True)
        reference_parameters = {name: value.detach().clone().requires_grad_(True)
                                for name, value in model.named_parameters()}
        apply_final_downsample_stride(model, {'model': 'convnext_small', 'final_downsample_stride': 1})
        actual = model(values)
        reference = F.conv2d(reference_values,
                             reference_parameters['network.features.6.1.weight'],
                             reference_parameters['network.features.6.1.bias'], stride=1)
        for index in range(3):
            prefix = 'network.features.7.' + str(index)
            reference = F.conv2d(reference, reference_parameters[prefix + '.weight'],
                                 reference_parameters[prefix + '.bias'])
        self.assertEqual(tuple(actual.shape), (2, 3, 31, 31))
        torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-12)
        actual.square().mean().backward()
        reference.square().mean().backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(bool(torch.isfinite(values.grad).all()))
        self.assertTrue(bool((values.grad != 0).any()))
        torch.testing.assert_close(values.grad, reference_values.grad, rtol=1e-10, atol=1e-12)
        for name, value in model.named_parameters():
            self.assertTrue(value.requires_grad, name)
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(bool(torch.isfinite(value.grad).all()), name)
            torch.testing.assert_close(value.grad, reference_parameters[name].grad,
                                       rtol=1e-10, atol=1e-12)

    def test_default_is_noop_and_strict_tensor_reload_requires_explicit_stride_reapplication(self):
        # Catches touching legacy models and relying on state_dict to save a non-tensor stride.
        import torch
        from retrain.spatial_stride import apply_final_downsample_stride

        self.assertIsNone(apply_final_downsample_stride(object(), {}))
        original = tiny_model()
        original.network.features[7][1].eval()
        values = input_values()
        before = original(values)
        state = {name: value.clone() for name, value in original.state_dict().items()}
        modules = [(name, id(module), module.training) for name, module in original.named_modules()]
        rng = torch.get_rng_state().clone()
        self.assertIsNone(apply_final_downsample_stride(original, {'model': 'convnext_small'}))
        self.assertEqual(original.network.features[6][1].stride, (2, 2))
        self.assertEqual([(name, id(module), module.training)
                          for name, module in original.named_modules()], modules)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, state[name]), name)
        self.assertTrue(torch.equal(original(values), before))

        job = {'model': 'convnext_small', 'final_downsample_stride': 1}
        apply_final_downsample_stride(original, job)
        saved = io.BytesIO()
        torch.save(original.state_dict(), saved)
        saved.seek(0)
        rebuilt = tiny_model()
        loaded = rebuilt.load_state_dict(torch.load(saved, map_location='cpu', weights_only=True), strict=True)
        self.assertEqual((loaded.missing_keys, loaded.unexpected_keys), ([], []))
        self.assertEqual(rebuilt.network.features[6][1].stride, (2, 2))
        self.assertEqual(tuple(rebuilt(values).shape), (2, 3, 16, 16))
        self.assertIsNone(apply_final_downsample_stride(rebuilt, job))
        self.assertEqual(rebuilt.network.features[6][1].stride, (1, 1))
        self.assertEqual(tuple(rebuilt(values).shape), (2, 3, 31, 31))
        self.assertTrue(torch.equal(rebuilt(values), original(values)))


if __name__ == '__main__':
    unittest.main()
