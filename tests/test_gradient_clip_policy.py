"""Two synthetic checks for the R126 rolewise L2 clip policy.

Parent A runs these once on the guarded Pod.  No real Small model, weights,
images or forward benchmark are used; the age contract is the same minimal
legal spec already used by the age tests.
"""
from pathlib import Path
from datetime import timedelta
import copy
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))
from retrain import age, ema, engine, gradient_clip
from retrain.persistence import canonical_digest

ROLES = ('backbone', 'head', 'age')
RAW_GC = {'version': 'raw-convlinear-weight-gc-v1',
          'targets': 'Conv2d_and_Linear_direct_weight', 'dtype': 'float32',
          'axes': 'all_non_output_dimensions',
          'position': 'after_ddp_mean_mirror_merge_unscale_before_global_clip',
          'schedule': 'each_real_update'}


class RawGCToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.features = nn.Linear(2, 3)
        self.norm = nn.LayerNorm(3)
        self.layer_scale = nn.Parameter(torch.tensor([.9, 1.1]).reshape(2, 1, 1))
        self.classifier = nn.Linear(3, 1)
        age.attach_age(self, age_spec())

    def forward(self, images, ages):
        x = (self.conv(images) * self.layer_scale).mean(dim=(2, 3))
        # A fixed input gain makes the toy's bias directions identifiable under
        # pairlog; otherwise FP32 zero-gradient noise is amplified by Adam eps.
        gain = 1 + .25 * images[:, 0].mean(dim=(1, 2))
        return self.classifier(self.norm(self.features(x))).flatten() * gain + (ages * self.age_weight).sum(1)


def raw_gc_job(policy=RAW_GC):
    from retrain.config import default_job
    value = default_job('raw-gc-fixture', 'convnext_small')
    value.update(seed=20261029, precision='fp32', image_size=512, prepared_size=512,
                 age=age_spec(), head_lr_scope='last_linear', loss={
                     'version':'binary-pairwise-logistic-v1','kind':'binary_pairwise_logistic',
                     'temperature':1.,'reduction':'mean','dtype':'float32','scope':'global_batch'})
    if policy is not None:
        value['gradient_centralization_policy'] = copy.deepcopy(policy)
    return value


def raw_gc_worker(rank, rendezvous, folder):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from retrain.classification_loss import classification_loss, gather_pairwise_inputs
    from retrain.config import job_contract, load_config
    from retrain.persistence import CheckpointStore, capture_rng, restore_rng
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(43)
    folder = Path(folder) / ('rank-' + str(rank));folder.mkdir()
    config_path = folder / 'config.json'
    config_path.write_text(json.dumps({'jobs':[raw_gc_job()]}))
    job = load_config(config_path, environ={})['jobs'][0]
    model = RawGCToy();reference = copy.deepcopy(model)
    optimizer = engine._optimizer(model, job)
    reference_optimizer = engine._optimizer(reference, raw_gc_job(None))
    shadow = ema.EMA(model, ema.RULE);reference_shadow = ema.EMA(reference, ema.RULE)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        ddp = DistributedDataParallel(model, broadcast_buffers=False)
        i = torch.arange(64, dtype=torch.float32)
        images = torch.stack([torch.sin(i / 9 + j / 3) for j in range(8)],dim=1).reshape(64,2,2,2)
        ages = torch.stack(((i-31.5)/2, (i.remainder(3)==0).float()),dim=1)
        labels = torch.cat((torch.zeros(56),torch.ones(8)))
        target_names = {'conv.weight','features.weight','classifier.weight'}
        for successful in (1,2):
            optimizer.zero_grad(set_to_none=True);reference_optimizer.zero_grad(set_to_none=True)
            interval = slice(rank*32,(rank+1)*32)
            logits, targets = gather_pairwise_inputs(ddp(images[interval],ages[interval]), labels[interval])
            classification_loss(logits,targets,job['loss']).backward()
            z = reference(images,ages)
            differences = z[labels==0][None,:]-z[labels==1][:,None]
            torch.logaddexp(torch.zeros_like(differences),differences).mean().backward()
            # Independent scalar row sums and full norm; no production GC/clip oracle.
            for name,p in reference.named_parameters():
                if name in target_names:
                    rows = p.grad.reshape(p.shape[0],-1)
                    for row in rows:
                        mean = sum(float(v) for v in row) / row.numel()
                        row.sub_(mean)
            norm = math.sqrt(sum(float(v)**2 for p in reference.parameters() for v in p.grad.flatten()))
            coefficient = min(1.,1./(norm+1e-6))
            for p in reference.parameters():p.grad.mul_(coefficient)
            def any_bad(value):
                flag=torch.tensor(int(value));dist.all_reduce(flag,op=dist.ReduceOp.MAX)
                return bool(flag)
            original_reduce = dist.all_reduce
            with patch.object(dist,'all_reduce',wraps=original_reduce) as calls:
                finite,_,_ = engine._clip_gradients(model.parameters(),None,any_bad,
                    centralization_targets=optimizer._gradient_centralization_targets)
            if not finite or calls.call_count!=1:
                raise AssertionError('GC changed the one existing finite-consensus collective')
            optimizer.step();reference_optimizer.step()
            shadow.update(model,successful);reference_shadow.update(reference,successful)
            for name,p in model.named_parameters():
                q=dict(reference.named_parameters())[name]
                torch.testing.assert_close(p,q,rtol=2e-5,atol=2e-7,
                    msg=lambda message:f'{name}: clipped actual={p.grad.tolist()} reference={q.grad.tolist()}\n'+message)
                for key in ('exp_avg','exp_avg_sq','step'):
                    torch.testing.assert_close(optimizer.state[p][key],reference_optimizer.state[q][key],rtol=4e-4,atol=2e-8)
                if int(optimizer.state[p]['step'])!=successful:raise AssertionError('actual Adam step differs')
            for name,v in shadow.model.state_dict().items():
                torch.testing.assert_close(v,reference_shadow.model.state_dict()[name],rtol=2e-5,atol=2e-7)
            if successful==1:
                identity={'job':job_contract(job)}
                payload={'format_version':1,'identity':identity,'job':job,'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),'ema':shadow.state_dict(),'rng':capture_rng(),
                    'manifest_digest':'synthetic-only','mode':'train','successful_updates':successful}
                store=CheckpointStore(folder/'checkpoints');store.save(payload);loaded=store.load()
                engine._validate_payload(loaded,identity,'synthetic-only',job,'train')
                if loaded['job']['gradient_centralization_policy']!=RAW_GC:raise AssertionError('saved GC identity lost')
                del ddp
                model=RawGCToy();model.load_state_dict(loaded['model'],strict=True)
                optimizer=engine._optimizer(model,loaded['job']);optimizer.load_state_dict(loaded['optimizer'])
                shadow=ema.EMA(model,ema.RULE,state=loaded['ema'],successful=loaded['successful_updates'])
                restore_rng(loaded['rng']);ddp=DistributedDataParallel(model,broadcast_buffers=False)
        (folder/'passed.json').write_text(json.dumps({'rank':rank,'actual_step':2,'GC_targets':3,'local_positives':int(labels[rank*32:(rank+1)*32].sum())}))
    finally:
        dist.destroy_process_group()


class RawGradientCentralizationTest(unittest.TestCase):
    def test_raw_row_formula_exclusions_globalclip_order_and_None_identity(self):
        from retrain.config import job_contract, load_config
        from retrain.persistence import capture_rng, canonical_digest
        import numpy as np
        model=RawGCToy();optimizer=engine._optimizer(model,raw_gc_job())
        self.assertTrue(hasattr(optimizer,'_gradient_centralization_targets'),'raw GC target registration is missing')
        self.assertEqual({n for n,_ in optimizer._gradient_centralization_targets},{'conv.weight','features.weight','classifier.weight'})
        expected={}
        for index,(name,p) in enumerate(model.named_parameters()):
            p.grad=(torch.arange(p.numel(),dtype=torch.float32).reshape(p.shape)+index+1)*.7
            expected[name]=p.grad.clone()
            if name in {'conv.weight','features.weight','classifier.weight'}:
                for row in expected[name].reshape(p.shape[0],-1):row.sub_(sum(float(v) for v in row)/row.numel())
        norm=math.sqrt(sum(float(v)**2 for g in expected.values() for v in g.flatten()))
        coefficient=min(1.,1./(norm+1e-6));self.assertLess(coefficient,1.)
        finite,_,_=engine._clip_gradients(model.parameters(),None,
            centralization_targets=optimizer._gradient_centralization_targets)
        self.assertTrue(finite)
        for name,p in model.named_parameters():torch.testing.assert_close(p.grad,expected[name]*coefficient,rtol=1e-6,atol=1e-7)
        original=raw_gc_job(None);with_none={**original,'gradient_centralization_policy':None}
        self.assertEqual(canonical_digest(job_contract(original)),canonical_digest(job_contract(with_none)))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json';path.write_text(json.dumps({'jobs':[with_none]}))
            self.assertNotIn('gradient_centralization_policy',load_config(path,environ={})['jobs'][0])
        legacy=copy.deepcopy(model);baseline=copy.deepcopy(model)
        for p,q,r in zip(model.parameters(),legacy.parameters(),baseline.parameters()):
            q.grad=p.grad.clone();r.grad=p.grad.clone()
        legacy_optimizer=engine._optimizer(legacy,original)
        self.assertFalse(hasattr(legacy_optimizer,'_gradient_centralization_targets'))
        before=capture_rng()
        engine._clip_gradients(legacy.parameters(),None,centralization_targets=None)
        after=capture_rng();torch.nn.utils.clip_grad_norm_(baseline.parameters(),1.)
        for p,q in zip(legacy.parameters(),baseline.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        self.assertEqual(before['python'],after['python']);self.assertTrue(np.array_equal(before['numpy'][1],after['numpy'][1]))
        self.assertTrue(torch.equal(before['torch'],after['torch']))
        self.assertEqual(legacy_optimizer.state_dict()['state'],{})

    def test_two_rank_global_gradient_Adam_and_saved_next_EMA(self):
        self.assertTrue(hasattr(engine._optimizer(RawGCToy(),raw_gc_job()),'_gradient_centralization_targets'),
                        'raw GC target registration is missing')
        with tempfile.TemporaryDirectory(prefix='raw-gc-gloo-') as folder:
            with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as endpoint:
                endpoint.bind(('127.0.0.1',0));rendezvous=f'tcp://127.0.0.1:{endpoint.getsockname()[1]}'
            processes=[];logs=[]
            try:
                for rank in range(2):
                    log=(Path(folder)/f'rank-{rank}.log').open('wb');logs.append(log)
                    env=dict(os.environ,GLOO_SOCKET_IFNAME='lo',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                    processes.append(subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--raw-gc-worker',str(rank),rendezvous,folder],
                        env=env,stdout=log,stderr=subprocess.STDOUT))
                for rank,p in enumerate(processes):
                    code=p.wait(timeout=75);logs[rank].flush()
                    self.assertEqual(code,0,(Path(folder)/f'rank-{rank}.log').read_text())
                for rank in range(2):self.assertEqual(json.loads((Path(folder)/f'rank-{rank}/passed.json').read_text())['actual_step'],2)
            finally:
                for p in processes:
                    if p.poll() is None:p.terminate()
                for p in processes:p.wait(timeout=5)
                for log in logs:log.close()


def age_spec():
    contract = {'version': age.VERSION, 'field': 'age', 'valid_min_exclusive': 0,
                'valid_max_inclusive': 120, 'features': ['standardized_age', 'missing_indicator'],
                'parameter_count': 2, 'bias': False, 'lr': 1e-3, 'weight_decay': 1e-4,
                'fit_count': 4, 'valid_fit_count': 4, 'missing_fit_count': 0,
                'median': 50.0, 'mean': 50.0, 'population_std': 10.0}
    return {'enabled': True, 'contract': contract, 'contract_sha256': canonical_digest(contract)}


def job(spec):
    return {'head_lr': 1e-3, 'backbone_lr': 1e-4, 'weight_decay': 1e-4, 'age': spec}


def tensor_parameter(values, gradient):
    parameter = nn.Parameter(torch.tensor(values, dtype=torch.float32))
    parameter.grad = torch.tensor(gradient, dtype=torch.float32)
    return parameter


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 5, bias=False)
        self.classifier = nn.Linear(5, 1)

    def forward(self, images, age_features):
        logits = self.classifier(torch.relu(self.features(images))).flatten()
        return logits + (age_features * self.age_weight).sum(dim=1)


def reference_groups(optimizer):
    groups = {role: [] for role in ROLES}
    for group in optimizer.param_groups:
        groups[group['role']].extend(group['params'])
    return groups


def reference_clip(groups):
    norms = {}
    for role, parameters in groups.items():
        pieces = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
        norms[role] = torch.linalg.vector_norm(torch.cat(pieces)) if pieces else torch.zeros(())
    coefficients = {role: min(1.0, 1.0 / (float(norms[role]) + 1e-6)) for role in groups}
    for role, parameters in groups.items():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficients[role])
    return coefficients


class RolewiseClipTest(unittest.TestCase):
    def test_hand_vectors_merge_roles_and_nonfinite_precheck(self):
        backbone = tensor_parameter([0., 0.], [.3, .4])
        head_decay = tensor_parameter([0., 0.], [2., 0.])
        head_no_decay = tensor_parameter([0., 0.], [0., 3.])
        age_weight = tensor_parameter([0., 0.], [.2, 0.])
        optimizer = SimpleNamespace(param_groups=[
            {'role': 'backbone', 'params': [backbone]},
            {'role': 'head', 'params': [head_decay]},
            {'role': 'head', 'params': [head_no_decay]},
            {'role': 'age', 'params': [age_weight]}])
        state = torch.random.get_rng_state()
        saved_grads = [p.grad.clone() for p in (backbone, head_decay, head_no_decay, age_weight)]
        self.assertIsNone(gradient_clip.build_role_groups(optimizer, None))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for parameter, saved in zip((backbone, head_decay, head_no_decay, age_weight), saved_grads):
            self.assertTrue(torch.equal(parameter.grad, saved))

        groups = gradient_clip.build_role_groups(optimizer, gradient_clip.ROLEWISE_CLIP_RULE)
        self.assertEqual(set(groups), set(ROLES))
        self.assertEqual(groups['backbone'], [backbone])
        self.assertEqual(groups['head'], [head_decay, head_no_decay])
        self.assertEqual(groups['age'], [age_weight])
        norms = gradient_clip.measure_role_norms(groups)
        self.assertAlmostEqual(float(norms['backbone']), .5, places=6)
        self.assertAlmostEqual(float(norms['head']), 13 ** .5, places=6)
        self.assertAlmostEqual(float(norms['age']), .2, places=6)
        coefficients = gradient_clip.apply_role_clip_(groups, norms)
        self.assertEqual(float(coefficients['backbone']), 1.0)
        self.assertEqual(float(coefficients['age']), 1.0)
        expected = 1.0 / (13 ** .5 + 1e-6)
        self.assertAlmostEqual(float(coefficients['head']), expected, delta=1e-6)
        torch.testing.assert_close(head_decay.grad, torch.tensor([2., 0.]) * expected)
        torch.testing.assert_close(head_no_decay.grad, torch.tensor([0., 3.]) * expected)
        torch.testing.assert_close(backbone.grad, torch.tensor([.3, .4]))
        torch.testing.assert_close(age_weight.grad, torch.tensor([.2, 0.]))
        self.assertLessEqual(float(gradient_clip.measure_role_norms(groups)['head']), 1.0)

        broken = gradient_clip.build_role_groups(optimizer, gradient_clip.ROLEWISE_CLIP_RULE)
        head_no_decay.grad[1] = float('inf')
        before = {role: [p.grad.clone() for p in params] for role, params in broken.items()}
        broken_norms = gradient_clip.measure_role_norms(broken)
        self.assertFalse(bool(torch.isfinite(broken_norms['head'])))
        for role in ROLES:
            for gradient, saved in zip((p.grad for p in broken[role]), before[role]):
                self.assertTrue(torch.equal(gradient, saved))
        with self.assertRaises(ValueError):
            gradient_clip.apply_role_clip_(broken, broken_norms)
        for role in ROLES:
            for gradient, saved in zip((p.grad for p in broken[role]), before[role]):
                self.assertTrue(torch.equal(gradient, saved))

    def test_engine_roles_two_step_adamw_reference_and_state(self):
        spec = age_spec()
        torch.manual_seed(31)
        model = Toy()
        torch.manual_seed(31)
        reference = Toy()
        age.attach_age(model, spec)
        age.attach_age(reference, spec)
        with torch.no_grad():
            model.age_weight.copy_(torch.tensor([.3, -.2]))
            reference.age_weight.copy_(torch.tensor([.3, -.2]))
        torch.manual_seed(32)
        images = torch.randn(4, 4)
        age_rows = torch.tensor([[.5, 0.], [-1., 1.], [0., 1.], [1.5, 0.]])
        targets = torch.tensor([1., -1., .5, -.5])

        optimizer = engine._optimizer(model, job(spec))
        reference_optimizer = engine._optimizer(reference, job(spec))
        before = [(g['role'], g['lr'], g['weight_decay'], [id(p) for p in g['params']])
                  for g in optimizer.param_groups]
        self.assertEqual([(role, lr, wd) for role, lr, wd, _ in before],
                         [('backbone', 1e-4, 1e-4), ('head', 1e-3, 0.0),
                          ('head', 1e-3, 1e-4), ('age', 1e-3, 1e-4)])
        groups = gradient_clip.build_role_groups(optimizer, gradient_clip.ROLEWISE_CLIP_RULE)
        after = [(g['role'], g['lr'], g['weight_decay'], [id(p) for p in g['params']])
                 for g in optimizer.param_groups]
        self.assertEqual(before, after)
        self.assertEqual(optimizer.state_dict()['state'], {})
        self.assertEqual([g['role'] for g in optimizer.param_groups].count('head'), 2)
        all_ids = {id(p) for role in ROLES for p in groups[role]}
        self.assertEqual(all_ids, {id(p) for p in model.parameters()})
        self.assertEqual(sum(len(groups[role]) for role in ROLES), len(list(model.parameters())))

        ref_groups = reference_groups(reference_optimizer)
        names = {id(p): name for name, p in model.named_parameters()}
        reference_names = {id(p): name for name, p in reference.named_parameters()}
        actual_coefficients, reference_coefficients = [], []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(images, age_rows) - targets) ** 2).mean() * 200
            loss.backward()
            norms = gradient_clip.measure_role_norms(groups)
            actual_coefficients.append(gradient_clip.apply_role_clip_(groups, norms))
            self.assertTrue(any(float(value) < 1.0 for value in actual_coefficients[-1].values()))
            optimizer.step()

            reference_optimizer.zero_grad(set_to_none=True)
            reference_loss = ((reference(images, age_rows) - targets) ** 2).mean() * 200
            reference_loss.backward()
            reference_coefficients.append(reference_clip(ref_groups))
            reference_optimizer.step()

            for role in ROLES:
                self.assertAlmostEqual(float(actual_coefficients[-1][role]),
                                       reference_coefficients[-1][role], places=5)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, dict(reference.named_parameters())[name],
                                           rtol=1e-5, atol=1e-7)
            actual_state = {names[id(p)]: s for p, s in optimizer.state.items()}
            reference_state = {reference_names[id(p)]: s for p, s in reference_optimizer.state.items()}
            self.assertEqual(set(actual_state), set(reference_state))
            for name in actual_state:
                for field in ('exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(actual_state[name][field],
                                               reference_state[name][field],
                                               rtol=1e-4, atol=1e-8)
        self.assertTrue(any(abs(float(actual_coefficients[0][role])
                                - float(actual_coefficients[1][role])) > 1e-9
                            for role in ROLES))

        saved = optimizer.state_dict()
        restored = engine._optimizer(model, job(spec))
        restored.load_state_dict(saved)
        loaded = restored.state_dict()
        self.assertEqual(loaded['param_groups'], saved['param_groups'])
        self.assertEqual(set(loaded['state']), set(saved['state']))
        for key in saved['state']:
            for field, value in saved['state'][key].items():
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(loaded['state'][key][field], value))

        shadow = ema.EMA(model, ema.RULE)
        self.assertEqual(set(shadow.state_dict()),
                         {'rule', 'model', 'update_count', 'copy_count', 'blend_count',
                          'last_update_success'})
        self.assertEqual(set(shadow.state_dict()['model']), set(model.state_dict()))
        self.assertEqual(set(optimizer.state_dict()['state']), set(saved['state']))


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--raw-gc-worker':
        raw_gc_worker(int(sys.argv[2]),sys.argv[3],sys.argv[4])
    else:
        unittest.main()
