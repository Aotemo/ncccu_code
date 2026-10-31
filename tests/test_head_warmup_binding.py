"""One tiny R128 science-binding and partial-Adam/EMA resume check."""
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
from retrain import age, engine
from retrain.distributed_engine import head_warmup_optimizer_proof
from retrain.ema import EMA, RULE as EMA_RULE
from retrain.head_warmup import (HEAD_WARMUP_RULE, build_update_groups,
                                mask_and_clip_gradients_, expected_optimizer_steps)
from test_head_warmup import Toy, age_spec, job as toy_job


class HeadWarmupBindingTest(unittest.TestCase):
    def test_science_identity_and_1599_1600_1601_strict_resume(self):
        old = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        new = json.loads((ROOT / 'configs/R128-SCIENCE.json').read_text())
        self.assertEqual(new['job']['head_warmup_policy'], HEAD_WARMUP_RULE)
        self.assertEqual(new['candidate']['head_warmup_policy'], HEAD_WARMUP_RULE)
        self.assertEqual(new['initial_model_tensor_sha256'], old['initial_model_tensor_sha256'])
        self.assertEqual(new['official_evaluation']['threshold'], .996)
        new.pop('official_evaluation')
        for key in ('candidate_profile', 'comparison_note'):
            old.pop(key, None); new.pop(key, None)
        for key in ('id', 'experiment_order', 'head_warmup_policy'):
            old['candidate'].pop(key, None); new['candidate'].pop(key, None)
        new['job'].pop('head_warmup_policy')
        self.assertEqual(new, old)

        spec = age_spec()
        torch.manual_seed(128)
        online = age.attach_age(Toy(), spec)
        optimizer = engine._optimizer(online, toy_job(spec))
        groups = build_update_groups(optimizer, HEAD_WARMUP_RULE)
        ema = EMA(online, EMA_RULE)
        self.assertEqual(head_warmup_optimizer_proof(optimizer, groups, 0)['backbone']['state_entries'], 0)

        def update(model, adam, shadow, membership, successful):
            for parameter in model.parameters():
                parameter.grad = torch.rand_like(parameter) - .5
            finite, info = mask_and_clip_gradients_(membership, successful, any_bad=lambda bad: bad)
            self.assertTrue(finite)
            self.assertEqual(info['next_successful_update'], successful + 1)
            adam.step(); shadow.update(model, successful + 1)

        for successful in range(1599):
            update(online, optimizer, ema, groups, successful)
        before = head_warmup_optimizer_proof(optimizer, groups, 1599)
        self.assertEqual(before['backbone']['state_entries'], 0)
        self.assertEqual(before['head']['expected_step'], 1599)
        rng = torch.random.get_rng_state()
        payload = {'model': online.state_dict(), 'optimizer': optimizer.state_dict(),
                   'ema': ema.state_dict(), 'rng': rng, 'successful_updates': 1599,
                   'job': {'head_warmup_policy': copy.deepcopy(HEAD_WARMUP_RULE)}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny-partial-adam.pt'
            torch.save(payload, path)
            loaded = torch.load(path, map_location='cpu', weights_only=False)
        self.assertEqual(loaded['job']['head_warmup_policy'], HEAD_WARMUP_RULE)
        torch.manual_seed(456)
        resumed = age.attach_age(Toy(), spec)
        resumed.load_state_dict(loaded['model'], strict=True)
        resumed_optimizer = engine._optimizer(resumed, toy_job(spec))
        resumed_optimizer.load_state_dict(loaded['optimizer'])
        resumed_groups = build_update_groups(resumed_optimizer, loaded['job']['head_warmup_policy'])
        resumed_ema = EMA(resumed, EMA_RULE, state=loaded['ema'], successful=1599)
        self.assertEqual(head_warmup_optimizer_proof(resumed_optimizer, resumed_groups, 1599), before)
        self.assertEqual(expected_optimizer_steps(1600)['backbone'], None)
        self.assertEqual(expected_optimizer_steps(1601)['backbone'], 1)

        for successful in (1599, 1600):
            torch.random.set_rng_state(rng)
            update(online, optimizer, ema, groups, successful)
            reference_rng = torch.random.get_rng_state()
            torch.random.set_rng_state(rng)
            update(resumed, resumed_optimizer, resumed_ema, resumed_groups, successful)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), reference_rng))
            rng = reference_rng
            self.assertEqual(head_warmup_optimizer_proof(optimizer, groups, successful + 1),
                             head_warmup_optimizer_proof(resumed_optimizer, resumed_groups, successful + 1))
            for a, b in zip(online.parameters(), resumed.parameters()):
                self.assertTrue(torch.equal(a, b))
            for (a_name, a), (b_name, b) in zip(ema.model.state_dict().items(), resumed_ema.model.state_dict().items()):
                self.assertEqual(a_name, b_name)
                self.assertTrue(torch.equal(a, b))
            for a, b in zip(optimizer.state.values(), resumed_optimizer.state.values()):
                self.assertEqual(set(a), set(b))
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key]))


if __name__ == '__main__':
    unittest.main()
