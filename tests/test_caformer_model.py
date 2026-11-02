"""Keep the native public MLP, one binary dropout and a reproducible cold graph."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import timm
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
from retrain.models import _create_caformer_binary


def job():
    from retrain.age import make_spec
    value = copy.deepcopy(json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())['job'])
    value.update(name='caformer-s36', model='caformer_s36', initialization={'kind':'random'},
        age=make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)} for i in range(4)],
                       'fit_indices':[0,1,2,3]}))
    return value


class CAFormerTest(unittest.TestCase):
    def test_native_factory_MLP_identity_rng_one_dropout_and_age(self):
        from retrain import models
        from retrain.age import residual
        value = job()
        torch.manual_seed(77)
        reference = timm.create_model('caformer_s36.sail_in1k_384', pretrained=False,
            drop_path_rate=.1, drop_rate=0., device='cpu', dtype=torch.float32)
        reference.head.fc.fc2 = nn.Linear(2048, 1)
        reference.head.fc.head_drop.p = .2
        reference.num_classes = 1
        expected_rng = torch.random.get_rng_state().clone()
        original_factory = timm.create_model
        retained = {}
        def capture(*args, **kwargs):
            native = original_factory(*args, **kwargs)
            retained.update(fc1=native.head.fc.fc1, norm=native.head.fc.norm, old_fc2=native.head.fc.fc2)
            return native
        torch.manual_seed(77)
        with patch.object(timm, 'create_model', side_effect=capture):
            model = _create_caformer_binary(value, initialize=False)
        self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
        self.assertIs(model.network.head.fc.fc1, retained['fc1'])
        self.assertIs(model.network.head.fc.norm, retained['norm'])
        self.assertIsNot(model.network.head.fc.fc2, retained['old_fc2'])
        for name, tensor in reference.state_dict().items():
            self.assertTrue(torch.equal(tensor, model.network.state_dict()[name]), name)
        self.assertEqual([n for n,m in model.named_modules() if isinstance(m,nn.Dropout) and m.p],
                         ['network.head.fc.head_drop'])
        self.assertEqual(len(dict(model.named_parameters())), 396)
        self.assertFalse(dict(model.named_buffers()))
        self.assertEqual(models.describe_caformer(model)['head_features'], 2048)
        images = torch.randn(1, 3, 512, 512)
        ages = torch.tensor([[.7, 1.]])
        with torch.no_grad():
            model.age_weight.copy_(torch.tensor([.2, -.1]))
        reference.eval(); model.eval()
        with torch.inference_mode():
            features = reference.forward_features(images)
            x = features.mean(dim=(-2,-1))
            x = F.layer_norm(x, (512,), reference.head.norm.weight, reference.head.norm.bias, 1e-6)
            x = F.linear(x, reference.head.fc.fc1.weight, reference.head.fc.fc1.bias)
            x = F.relu(x).square()
            x = F.layer_norm(x, (2048,), reference.head.fc.norm.weight, reference.head.fc.norm.bias, 1e-6)
            expected = F.linear(x, reference.head.fc.fc2.weight, reference.head.fc.fc2.bias).flatten()
            expected += (ages * model.age_weight).sum(1)
            with patch('retrain.age.residual', wraps=residual) as age_call:
                actual = model(images, ages)
            self.assertEqual(age_call.call_count, 1)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(actual, reference(images).flatten() + (ages*model.age_weight).sum(1), rtol=0, atol=0)

    def test_P8_finite_gradients_optimizer_EMA_and_real_cold_graph(self):
        from retrain import config, ema, models
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        value = job(); config.validate_job(value)
        torch.manual_seed(79)
        model = _create_caformer_binary(value, initialize=False)
        optimizer = _optimizer(model, value)
        self.assertEqual(len(optimizer.param_groups), 5); self.assertFalse(optimizer.state)
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']}, {id(p) for p in model.parameters()})
        self.assertEqual({id(p) for g in optimizer.param_groups if g['role']=='head' for p in g['params']},
                         {id(p) for p in model.network.head.fc.fc2.parameters()})
        self.assertTrue(any(p is model.network.head.fc.fc1.weight for g in optimizer.param_groups if g['role']=='backbone' for p in g['params']))
        shadow = ema.EMA(model, ema.RULE)
        self.assertIsNot(model.network.head.fc.fc1.weight, shadow.model.network.head.fc.fc1.weight)
        self.assertEqual(models.describe_caformer(model), models.describe_caformer(shadow.model))
        with torch.no_grad():
            model.age_weight.copy_(torch.tensor([.2, -.1]))
        images = torch.randn(8, 3, 512, 512)
        ages = torch.stack((torch.linspace(-1,1,8), torch.ones(8)), dim=1)
        calls = []
        handle = model.network.register_forward_pre_hook(lambda module,args: calls.append(tuple(args[0].shape)))
        saved = {}
        def pack(tensor):
            storage = tensor.untyped_storage(); saved[storage.data_ptr()] = storage.nbytes()
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t:t):
            logits = model(images, ages)
            (logits * torch.linspace(-1,1,8)).sum().backward()
        handle.remove()
        self.assertEqual(calls, [(8,3,512,512)])
        self.assertEqual(tuple(logits.shape), (8,))
        self.assertTrue(all(p.requires_grad and p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        model.eval()
        with torch.inference_mode():
            expected = model(images[:1], ages[:1])
        payload = {'format_version':1, 'mode':'train', 'job':value, 'model':model.state_dict(),
            'age':value['age'], 'identity':{}, 'manifest_digest':'synthetic-only', 'successful_updates':0}
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            checkpoint = CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold, saved_job, _, origin, _ = _load_model(checkpoint, device='cpu', variant='online')
            self.assertEqual(saved_job, value)
            self.assertEqual(origin['caformer'], models.describe_caformer(model))
            with torch.inference_mode():
                actual = cold(images[:1], ages[:1])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(json.dumps({'scope':'CPU_synthetic_P8_512_saved_storage_not_GPU_peak',
            'unique_saved_CPU_storage_bytes':sum(saved.values()), 'parameter_elements':sum(p.numel() for p in model.parameters()),
            'GPU_peak_or_latency_measured':False}), flush=True)
        self.assertFalse(torch.cuda.is_initialized())
