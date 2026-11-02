"""A fixed decay span changes real group rates, while the update budget stays 24k."""
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/bundle'))
from retrain import age,config
from retrain.engine import _optimizer,_apply_lr


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.LayerNorm(4))
        self.classifier=torch.nn.Linear(4,1)
        self.age_weight=torch.nn.Parameter(torch.zeros(2))


def job():
    value=json.loads((ROOT/'configs/R120-SCIENCE.json').read_text())['job']
    value['age']=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)}
        for i in range(4)],'fit_indices':[0,1,2,3]})
    return value


def expected_factor(step,horizon):
    if step<1600:
        return (step+1)/1600
    position=max(0.,min(1.,(step-1600)/(horizon-1600-1)))
    return .01+.99*(1+math.cos(math.pi*position))/2


class FixedLRDecayTest(unittest.TestCase):
    def test_real_groups_follow_fixed_decay_and_preserve_legacy_None(self):
        model=Toy();baseline=job();short=copy.deepcopy(baseline);short['lr_decay_updates']=8000
        before=torch.random.get_rng_state().clone()
        optimizer=_optimizer(model,short)
        self.assertIs(type(optimizer),torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        for step in (0,1,799,1599,1600,1601,3999,6399,7998,7999,8000,23999):
            _apply_lr(optimizer,step,24000,1600)
            for group in optimizer.param_groups:
                rate=1e-4 if group['role']=='backbone' else 1e-3
                self.assertEqual(group['initial_lr'],rate)
                self.assertAlmostEqual(group['lr'],rate*expected_factor(step,8000),places=15)
            rates=[g['lr']for g in optimizer.param_groups]
            _apply_lr(optimizer,step,24000,1600)
            self.assertEqual(rates,[g['lr']for g in optimizer.param_groups])
        absent=_optimizer(model,baseline)
        explicit=_optimizer(model,{**baseline,'lr_decay_updates':None})
        self.assertEqual(absent.state_dict(),explicit.state_dict())
        self.assertFalse(hasattr(absent,'_lr_decay_updates'))
        self.assertFalse(hasattr(explicit,'_lr_decay_updates'))
        for step in (0,1599,1600,3999,7999,23999):
            for legacy in (absent,explicit):
                _apply_lr(legacy,step,24000,1600)
                for group in legacy.param_groups:
                    self.assertEqual(group['lr'].hex(),(group['initial_lr']*expected_factor(step,24000)).hex())

    def test_saved_job_reconstructs_decay_after_native_optimizer_state_roundtrip(self):
        current=job();current['lr_decay_updates']=8000
        model=Toy();optimizer=_optimizer(model,current)
        def update(network,opt,step):
            _apply_lr(opt,step,24000,1600)
            for i,parameter in enumerate(network.parameters()):
                parameter.grad=torch.full_like(parameter,.1+.01*i)
            opt.step();opt.zero_grad(set_to_none=True)
        for step in (1599,1600,7998):update(model,optimizer,step)
        buffer=io.BytesIO()
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'job':current},buffer)
        buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=True)
        restored=Toy();restored.load_state_dict(saved['model'])
        restored_optimizer=_optimizer(restored,saved['job'])
        restored_optimizer.load_state_dict(saved['optimizer'])
        self.assertEqual(saved['job']['lr_decay_updates'],8000)
        for step in (7999,8000,23999):
            update(model,optimizer,step);update(restored,restored_optimizer,step)
            for actual,expected in zip(restored.parameters(),model.parameters()):
                self.assertTrue(torch.equal(actual,expected))
            self.assertEqual([g['lr']for g in optimizer.param_groups],
                             [g['lr']for g in restored_optimizer.param_groups])
            for actual,expected in zip(optimizer.state.values(),restored_optimizer.state.values()):
                for key in actual:self.assertTrue(torch.equal(actual[key],expected[key]))
            for group in restored_optimizer.param_groups:
                self.assertAlmostEqual(group['lr'],group['initial_lr']*.01,places=15)

    def test_config_roundtrip_keeps_fixed_decay_and_omits_None(self):
        baseline=json.loads((ROOT/'configs/R120-SCIENCE.json').read_text())['job']
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'config.json'
            def load(value):
                path.write_text(json.dumps({'jobs':[value]}))
                return config.load_config(path,environ={})['jobs'][0]
            omitted=load(baseline);explicit=load({**baseline,'lr_decay_updates':None})
            self.assertEqual(omitted,explicit)
            self.assertNotIn('lr_decay_updates',config.job_contract(explicit))
            accepted=load({**baseline,'lr_decay_updates':8000})
            self.assertEqual(config.job_contract(accepted)['lr_decay_updates'],8000)
            for invalid in (7999,24000,8000.,True,-1):
                with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                    load({**baseline,'lr_decay_updates':invalid})
            with self.assertRaises(ValueError):load({**baseline,'lr_decay_updates':8000,'backbone_lr':5e-5})


if __name__=='__main__':unittest.main()
