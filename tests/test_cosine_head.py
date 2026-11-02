"""Narrow value/gradient and production reconstruction checks for fixed cosine16."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain.models import _cosine_linear


class CosineHeadTest(unittest.TestCase):
    def test_scalar_oracle_gradients_and_parameter_rng_identity(self):
        linear = nn.Linear(4, 1)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[.2, -.3, .4, .1]]))
            linear.bias.fill_(.15)
        before = torch.random.get_rng_state().clone()
        names = dict(linear.named_parameters())
        layer = _cosine_linear(linear)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertIs(layer.weight, names['weight'])
        self.assertIs(layer.bias, names['bias'])
        self.assertEqual(set(layer.state_dict()), {'weight', 'bias'})
        self.assertFalse(dict(layer.named_buffers()))
        self.assertIsInstance(layer, nn.Linear)
        values = [[.5, -.7, .2, 1.1], [-.4, .8, 1.2, -.3], [.9, .1, -.5, .6]]
        upstream = [1.2, -.4, .7]
        x = torch.tensor(values, requires_grad=True)
        result = layer(x).flatten()
        (result * torch.tensor(upstream)).sum().backward()
        w = [.2, -.3, .4, .1]
        nw = math.sqrt(sum(t*t for t in w))
        expected, gx, gw = [], [], [0.] * 4
        for row, u in zip(values, upstream):
            nx = math.sqrt(sum(t*t for t in row))
            c = sum(a*b for a,b in zip(row,w)) / (nx*nw)
            expected.append(16*c+.15)
            gx.append([u*16*(b/(nx*nw)-c*a/(nx*nx)) for a,b in zip(row,w)])
            for k,(a,b) in enumerate(zip(row,w)):
                gw[k] += u*16*(a/(nx*nw)-c*b/(nw*nw))
        torch.testing.assert_close(result, torch.tensor(expected), rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(x.grad, torch.tensor(gx), rtol=3e-6, atol=3e-6)
        torch.testing.assert_close(layer.weight.grad, torch.tensor([gw]), rtol=3e-6, atol=3e-6)
        torch.testing.assert_close(layer.bias.grad, torch.tensor([sum(upstream)]), rtol=0, atol=2e-7)
        self.assertAlmostEqual(float((layer.weight.grad*layer.weight).sum()), 0., delta=2e-5)
        with torch.no_grad():
            original = layer(x.detach()).clone()
            layer.weight.mul_(7.)
            torch.testing.assert_close(layer(x.detach()*2.), original, rtol=2e-6, atol=2e-6)
        for zero_weight in (False, True):
            edge = _cosine_linear(nn.Linear(4,1))
            if zero_weight:
                with torch.no_grad(): edge.weight.zero_()
            z = torch.zeros(2,4,requires_grad=True)
            out = edge(z)
            out.sum().backward()
            self.assertTrue(bool(torch.isfinite(out).all()))
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in edge.parameters()))
            self.assertTrue(bool(torch.isfinite(z.grad).all()))

    def test_native_factory_age_optimizer_and_strict_cold_graph(self):
        from retrain import age, config, ema, models
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        job = copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job'])
        job['initialization'] = {'kind':'random'}
        job['age'] = age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)}
                                           for i in range(4)], 'fit_indices':[0,1,2,3]})
        torch.manual_seed(51)
        plain = models.create_model(job, initialize=False).eval()
        plain_rng = torch.random.get_rng_state().clone()
        target_job = {**job, 'cosine_head_policy':copy.deepcopy(config.COSINE_HEAD_RULE)}
        torch.manual_seed(51)
        model = models.create_model(target_job, initialize=False).eval()
        self.assertTrue(torch.equal(plain_rng, torch.random.get_rng_state()))
        self.assertIs(type(plain.network.classifier[-1]), nn.Linear)
        self.assertEqual(set(model.state_dict()), set(plain.state_dict()))
        self.assertTrue(all(torch.equal(t, plain.state_dict()[n]) for n,t in model.state_dict().items()))
        self.assertEqual(len(dict(model.named_parameters())), 345)
        self.assertFalse(dict(model.named_buffers()))
        self.assertEqual(config.job_contract({**job,'cosine_head_policy':None}), config.job_contract(job))
        self.assertEqual(config.job_contract(target_job)['cosine_head_policy'], config.COSINE_HEAD_RULE)
        descriptor = models.describe_cosine_head(model, target_job)
        self.assertEqual(descriptor['policy'], config.COSINE_HEAD_RULE)
        with self.assertRaises(ValueError): models.describe_cosine_head(plain, target_job)
        optimizer = _optimizer(model, target_job)
        self.assertEqual(len(optimizer.param_groups), 5)
        self.assertFalse(optimizer.state)
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']},
                         {id(p) for p in model.parameters()})
        # Exercise real EMA deepcopy/graph retention; the existing EMA math is unchanged.
        shadow = ema.EMA(model, ema.RULE)
        self.assertEqual(models.describe_cosine_head(shadow.model, target_job), descriptor)
        with torch.no_grad(): model.age_weight.copy_(torch.tensor([.2,-.1]))
        images = torch.randn(2,3,32,32)
        ages = torch.tensor([[.5,1.],[-.25,1.]])
        with torch.inference_mode():
            expected = model(images,ages)
            image_only = model.network(images).flatten()
            torch.testing.assert_close(expected, age.residual(model,image_only,ages), rtol=0, atol=0)
            old = plain(images,ages)
            torch.testing.assert_close(old, plain.network(images).flatten(), rtol=0, atol=0)
        model.train()
        (model(images,ages)*torch.tensor([.7,-.7])).sum().backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        model.eval()
        payload = {'format_version':1,'mode':'train','job':target_job,'model':model.state_dict(),
                   'age':target_job['age'],'identity':{},'manifest_digest':'synthetic-only','successful_updates':0}
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            checkpoint = CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold,saved,_,origin,_ = _load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved['cosine_head_policy'], config.COSINE_HEAD_RULE)
            self.assertEqual(origin['cosine_head'], descriptor)
            with torch.inference_mode(): actual = cold(images,ages)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
