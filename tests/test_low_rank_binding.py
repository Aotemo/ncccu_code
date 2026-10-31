"""One small continuation/variant check; it never resumes a real experiment."""
import copy
import io
import random
import unittest

import numpy as np
import torch

from test_low_rank import Toy, implementation, rng_equal, toy_job
from retrain import engine
from retrain.ema import EMA, RULE, select_variant
from retrain.persistence import capture_rng, restore_rng


def step(model, optimizer, ema, count):
    x = torch.randn(3, 4) + random.random() + float(np.random.random())
    age = torch.randn(3, 2)
    optimizer.zero_grad(set_to_none=True)
    engine._apply_lr(optimizer, count - 1, 480, 2)
    model(x, age).square().mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    ema.update(model, count)


class LowRankBindingTest(unittest.TestCase):
    def test_named_state_roundtrip_factor_ema_and_variant(self):
        low_rank = implementation(self)
        torch.set_num_threads(2)
        torch.manual_seed(91)
        random.seed(91)
        np.random.seed(91)
        job = toy_job(low_rank)
        online = Toy(low_rank)
        optimizer = engine._optimizer(online, job)
        self.assertEqual(low_rank.optimizer_update_proof(online, optimizer, 0)['optimizer_state_entries'], 0)
        ema = EMA(online, RULE)
        base = {n: p.detach().clone() for n, p in online.named_parameters() if not p.requires_grad}
        for count in range(1, RULE['warmcopy_updates'] + 2):
            step(online, optimizer, ema, count)
        successful = count
        identity = low_rank.optimizer_group_identity(online, optimizer)
        buffer = io.BytesIO()
        torch.save({'model': online.state_dict(), 'optimizer': optimizer.state_dict(),
                    'ema': ema.state_dict(), 'rng': capture_rng(), 'successful_updates': successful}, buffer)
        buffer.seek(0)
        payload = torch.load(buffer, map_location='cpu', weights_only=False)
        rebuilt = Toy(low_rank)
        rebuilt.load_state_dict(payload['model'], strict=True)
        restored_optimizer = engine._optimizer(rebuilt, job)
        self.assertEqual(low_rank.validate_optimizer_state(rebuilt, restored_optimizer, payload['optimizer']), identity)
        restored_optimizer.load_state_dict(payload['optimizer'])
        restored_steps = low_rank.optimizer_update_proof(rebuilt, restored_optimizer, successful)
        self.assertEqual(restored_steps['observed_Adam_steps'], [successful])
        self.assertEqual(restored_steps['optimizer_state_entries'], 7)
        restored_ema = EMA(rebuilt, RULE, state=payload['ema'], successful=successful)
        self.assertEqual(low_rank.optimizer_group_identity(rebuilt, restored_optimizer), identity)
        reference_rng = capture_rng()
        restore_rng(payload['rng'])
        step(online, optimizer, ema, successful + 1)
        expected_rng = capture_rng()
        restore_rng(payload['rng'])
        step(rebuilt, restored_optimizer, restored_ema, successful + 1)
        self.assertEqual(low_rank.optimizer_update_proof(rebuilt, restored_optimizer, successful + 1)
                         ['observed_Adam_steps'], [successful + 1])
        rng_equal(self, expected_rng, capture_rng())
        for name, value in online.state_dict().items():
            self.assertTrue(torch.equal(value, rebuilt.state_dict()[name]), name)
            self.assertTrue(torch.equal(ema.model.state_dict()[name], restored_ema.model.state_dict()[name]), name)
        left = optimizer.state_dict()
        right = restored_optimizer.state_dict()
        self.assertEqual(left['param_groups'], right['param_groups'])
        for position, state in left['state'].items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, right['state'][position][key]))
        for name, value in base.items():
            self.assertTrue(torch.equal(value, online.state_dict()[name]))
            self.assertTrue(torch.equal(value, ema.model.state_dict()[name]))
        frozen_ids = {id(p) for p in online.parameters() if not p.requires_grad}
        self.assertTrue(all(id(p) not in frozen_ids for p in optimizer.state))
        self.assertTrue(all(int(state['step']) == successful + 1 for state in optimizer.state.values()))

        # Existing variant selection must rebuild both factors from the selected state.
        selected = Toy(low_rank)
        selected.load_state_dict(select_variant(payload, 'ema'), strict=True)
        x, age = torch.randn(2, 4), torch.randn(2, 2)
        reference = Toy(low_rank)
        reference.load_state_dict(payload['ema']['model'], strict=True)
        self.assertTrue(torch.equal(selected(x, age), reference(x, age)))
        self.assertTrue(any(not torch.equal(payload['model'][n], payload['ema']['model'][n])
                            for n in payload['model'] if n.endswith(('lora_A', 'lora_B'))))
        # The blend is on factors, rather than the factor product.
        name = 'network.features.0.'
        before = payload['ema']['model']
        after = online.state_dict()
        d = RULE['decay']
        factor_product = (before[name + 'lora_B'] * d + after[name + 'lora_B'] * (1-d)) @ (
                          before[name + 'lora_A'] * d + after[name + 'lora_A'] * (1-d))
        product_blend = (before[name + 'lora_B'] @ before[name + 'lora_A']) * d + (
                         after[name + 'lora_B'] @ after[name + 'lora_A']) * (1-d)
        self.assertGreater(float((factor_product - product_blend).abs().max()), 0.)
        restore_rng(reference_rng)
        print('LOW_RANK_ROUNDTRIP: strict model / named Adam / frozen base / RNG / factor EMA / selected variant exact')


if __name__ == '__main__':
    unittest.main()
