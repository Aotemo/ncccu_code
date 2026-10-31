"""R130's one science-to-optimizer binding and named-state continuation fixture."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
sys.path.insert(0, str(ROOT / 'tests'))
from retrain import engine
from retrain.ema import EMA, RULE as EMA_RULE
from retrain.layerwise_lr import LAYERWISE_LR_RULE, group_identity
from test_layerwise_lr import NamedSmall, job as toy_job


class LayerwiseBindingTest(unittest.TestCase):
    def test_science_named_state_lr_ema_rng_roundtrip(self):
        original = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        current = json.loads((ROOT / 'configs/R130-SCIENCE.json').read_text())
        self.assertEqual(current['candidate']['layerwise_lr_policy'], LAYERWISE_LR_RULE)
        self.assertEqual(current['job']['layerwise_lr_policy'], LAYERWISE_LR_RULE)
        self.assertEqual(current['initial_model_tensor_sha256'], original['initial_model_tensor_sha256'])
        self.assertEqual(current['official_evaluation']['threshold'], .996)
        current.pop('official_evaluation')
        for key in ('candidate_profile', 'comparison_note'):
            original.pop(key, None); current.pop(key, None)
        for key in ('id', 'experiment_order', 'layerwise_lr_policy'):
            original['candidate'].pop(key, None); current['candidate'].pop(key, None)
        current['job'].pop('layerwise_lr_policy')
        self.assertEqual(current, original)

        torch.manual_seed(130)
        model = NamedSmall()
        job = {**toy_job(), 'layerwise_lr_policy': LAYERWISE_LR_RULE}
        optimizer = engine._optimizer(model, job)
        identity = group_identity(model, optimizer.param_groups, LAYERWISE_LR_RULE)
        self.assertFalse(optimizer.state)
        shadow = EMA(model, EMA_RULE)

        def update(target, adam, ema, successful):
            adam.zero_grad(set_to_none=True)
            for parameter in target.parameters():
                parameter.grad = torch.rand_like(parameter) - .5
            engine._apply_lr(adam, successful, 5, 2)
            adam.step()
            ema.update(target, successful + 1)

        update(model, optimizer, shadow, 0)
        saved_rng = torch.random.get_rng_state().clone()
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'ema': shadow.state_dict(), 'rng': saved_rng,
                   'successful_updates': 1, 'job': {'layerwise_lr_policy': LAYERWISE_LR_RULE}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'named-state.pt'
            torch.save(payload, path)
            loaded = torch.load(path, map_location='cpu', weights_only=False)
        self.assertEqual(loaded['job']['layerwise_lr_policy'], LAYERWISE_LR_RULE)

        torch.manual_seed(999)
        resumed = NamedSmall()
        resumed.load_state_dict(loaded['model'], strict=True)
        resumed_adam = engine._optimizer(resumed, job)
        self.assertEqual(engine._validate_layerwise_optimizer_state(
            resumed, resumed_adam, loaded['optimizer'], LAYERWISE_LR_RULE), identity)
        changed = copy.deepcopy(loaded['optimizer'])
        group = next(row for row in changed['param_groups'] if len(row['param_names']) > 1)
        group['param_names'][0], group['param_names'][1] = group['param_names'][1], group['param_names'][0]
        with self.assertRaises(ValueError):
            engine._validate_layerwise_optimizer_state(resumed, resumed_adam, changed, LAYERWISE_LR_RULE)
        resumed_adam.load_state_dict(loaded['optimizer'])
        self.assertEqual(group_identity(resumed, resumed_adam.param_groups, LAYERWISE_LR_RULE), identity)
        resumed_ema = EMA(resumed, EMA_RULE, state=loaded['ema'], successful=1)

        torch.random.set_rng_state(saved_rng)
        update(model, optimizer, shadow, 1)
        reference_rng = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(saved_rng)
        update(resumed, resumed_adam, resumed_ema, 1)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), reference_rng))
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, dict(resumed.named_parameters())[name], rtol=0, atol=0)
            left = optimizer.state[parameter]
            right = resumed_adam.state[dict(resumed.named_parameters())[name]]
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        for name, tensor in shadow.model.state_dict().items():
            torch.testing.assert_close(tensor, resumed_ema.model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual([group['lr'] for group in optimizer.param_groups],
                         [group['lr'] for group in resumed_adam.param_groups])


if __name__ == '__main__':
    unittest.main()
