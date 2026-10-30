"""Three synthetic checks; parent A runs them in the guarded CPU pool."""
from pathlib import Path
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
            self.network.features = nn.Sequential(*[nn.Linear(2, 2) for _ in range(5)])
            self.network.classifier = nn.Linear(2, 1)
            with torch.no_grad():
                for layer in self.network.features:
                    layer.weight.copy_(torch.eye(2))
                    layer.bias.fill_(.1)
                self.network.classifier.weight.copy_(torch.tensor([[.3, -.2]]))
                self.network.classifier.bias.fill_(.1)

        def forward(self, values):
            return self.network.classifier(self.network.features(values))

    return TinyModel()


def resume_at_warmcopy(online, rule):
    from retrain.ema import EMA
    initial = EMA(online, rule)
    state = initial.state_dict()
    warm = rule['warmcopy_updates']
    state.update(update_count=warm, copy_count=warm, blend_count=0,
                 last_update_success=warm)
    return EMA(online, rule, state=state, successful=warm)


class FrozenFeaturesTest(unittest.TestCase):
    def test_real_freeze_and_optimizer_keep_only_trainable_members_and_update_later_layers(self):
        import torch
        from retrain.engine import _optimizer
        from retrain.models import apply_feature_freeze

        model = tiny_model()
        model.network.features[1].eval()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        modes = [(module, module.training) for module in model.modules()]
        rng = torch.get_rng_state().clone()
        job = {'model': 'convnext_small', 'freeze_features_before': 4,
               'backbone_lr': 1e-4, 'head_lr': 1e-3, 'head_lr_scope': 'last_linear',
               'weight_decay': 1e-4}
        frozen = apply_feature_freeze(model, job)
        expected = tuple(sorted('network.features.' + str(index) + '.' + suffix
                                for index in range(4) for suffix in ('weight', 'bias')))
        self.assertEqual(frozen, expected)
        self.assertEqual([(module, module.training) for module in model.modules()], modes)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name not in expected, name)

        optimizer = _optimizer(model, job)
        members = [parameter for group in optimizer.param_groups for parameter in group['params']]
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertEqual(len(members), len({id(parameter) for parameter in members}))
        self.assertEqual({id(parameter) for parameter in members},
                         {id(parameter) for parameter in trainable})
        images = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        loss = (model(images) - torch.tensor([[0.], [1.]])).square().mean()
        loss.backward()
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in expected:
                self.assertTrue(torch.equal(parameter, before[name]), name)
                self.assertIsNone(parameter.grad)
        for prefix in ('network.features.4.', 'network.classifier.'):
            self.assertTrue(any(not torch.equal(parameter, before[name])
                                for name, parameter in model.named_parameters()
                                if name.startswith(prefix)), prefix)
        self.assertIsNotNone(images.grad)
        self.assertTrue(bool((images.grad != 0).any()))

    def test_ema_uses_online_frozen_names_and_copies_them_after_warmcopy_and_resume(self):
        import torch
        from retrain.ema import R8_RULE
        from retrain.models import apply_feature_freeze

        online = tiny_model()
        frozen = apply_feature_freeze(online, {'model': 'convnext_small', 'freeze_features_before': 4})
        with torch.no_grad():
            for name, parameter in online.named_parameters():
                parameter.fill_(.31415927 if name in frozen else 1.5)
        ema = resume_at_warmcopy(online, R8_RULE)
        self.assertEqual(ema.frozen_parameter_names, frozen)
        self.assertTrue(all(not parameter.requires_grad for parameter in ema.model.parameters()))
        self.assertLess(len(frozen), len(list(ema.model.named_parameters())))
        with torch.no_grad():
            for parameter in ema.model.parameters():
                parameter.fill_(-2.)
        warm = R8_RULE['warmcopy_updates']
        ema.update(online, warm + 1)
        for name, parameter in ema.model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(parameter, dict(online.named_parameters())[name]), name)
            else:
                # -2 * .999 + 1.5 * .001, independent of the update helper.
                torch.testing.assert_close(parameter, torch.full_like(parameter, -1.9965),
                                           rtol=1e-6, atol=1e-7)
        before_skip = {name: value.clone() for name, value in ema.model.state_dict().items()}
        ema.update(online, warm + 1, skipped=True)
        self.assertEqual(ema.successful, warm + 1)
        for name, value in ema.model.state_dict().items():
            self.assertTrue(torch.equal(value, before_skip[name]), name)
        with torch.no_grad():
            for name, parameter in ema.model.named_parameters():
                if name in frozen:
                    parameter.fill_(-5.)
        ema.update(online, warm + 2)
        for name, parameter in ema.model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(parameter, dict(online.named_parameters())[name]), name)
            else:
                torch.testing.assert_close(parameter, torch.full_like(parameter, -1.9930035),
                                           rtol=1e-6, atol=1e-7)
        state = ema.state_dict()
        self.assertEqual((state['copy_count'], state['blend_count'], state['last_update_success']),
                         (warm, 2, warm + 2))

    def test_default_noop_keeps_old_warmcopy_blend_and_six_key_state(self):
        import torch
        from retrain.ema import EMA, R8_RULE
        from retrain.models import apply_feature_freeze

        online = tiny_model()
        original = {name: value.clone() for name, value in online.state_dict().items()}
        modes = [(module, module.training) for module in online.modules()]
        rng = torch.get_rng_state().clone()
        self.assertEqual(apply_feature_freeze(online, {'model': 'convnext_small'}), ())
        self.assertTrue(all(parameter.requires_grad for parameter in online.parameters()))
        self.assertEqual([(module, module.training) for module in online.modules()], modes)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, value in online.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]), name)
        ema = EMA(online, R8_RULE)
        self.assertEqual(ema.frozen_parameter_names, ())
        with torch.no_grad():
            for parameter in online.parameters():
                parameter.fill_(1.5)
            for parameter in ema.model.parameters():
                parameter.fill_(-2.)
        ema.update(online, 1)
        for name, parameter in ema.model.named_parameters():
            self.assertTrue(torch.equal(parameter, dict(online.named_parameters())[name]), name)

        ema = resume_at_warmcopy(online, R8_RULE)
        with torch.no_grad():
            for parameter in ema.model.parameters():
                parameter.fill_(-2.)
        ema.update(online, R8_RULE['warmcopy_updates'] + 1)
        self.assertEqual(ema.frozen_parameter_names, ())
        for parameter in ema.model.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, -1.9965),
                                       rtol=1e-6, atol=1e-7)
        self.assertEqual(set(ema.state_dict()), {'rule', 'model', 'update_count', 'copy_count',
                                               'blend_count', 'last_update_success'})


if __name__ == '__main__':
    unittest.main()
