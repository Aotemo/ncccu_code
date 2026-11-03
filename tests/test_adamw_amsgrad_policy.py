"""Native AMSGrad: raw second-moment maximum, state, and legacy identity."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime/bundle'),str(ROOT/'tests')]
from retrain.engine import _optimizer
from retrain import config
from test_adamw_policy import Toy,job

POLICY='adamw-amsgrad-v1'


def gradients(model,step):
    scale=(3.,.01,0.,4.)[step-1]
    for i,p in enumerate(model.parameters()):
        p.grad=torch.linspace(.4,1.2,p.numel(),dtype=p.dtype).reshape_as(p)*scale*(i+1)


class AdamWAmsgradTest(unittest.TestCase):
    def test_native_all_five_groups_independent_raw_max_and_saved_next_step(self):
        value=job();value['adamw_amsgrad_policy']=POLICY
        torch.manual_seed(83);model=Toy();rng=torch.random.get_rng_state().clone()
        optimizer=_optimizer(model,value)
        self.assertTrue(all(g['amsgrad']is True for g in optimizer.param_groups))
        self.assertIs(type(optimizer),torch.optim.AdamW)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertEqual([(g['role'],g['lr'],g['weight_decay'])for g in optimizer.param_groups],
            [('backbone',1e-4,0.),('backbone',1e-4,1e-4),('head',1e-3,0.),('head',1e-3,1e-4),('age',1e-3,1e-4)])
        self.assertTrue(all(g['betas']==(.9,.999)and g['eps']==1e-8 and g['foreach']is None
            and g['fused']is None and not g['maximize'] for g in optimizer.param_groups))
        self.assertFalse(optimizer.state)
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()})
        expected={p:{'w':p.detach().double().clone(),'m':torch.zeros_like(p,dtype=torch.float64),
            'v':torch.zeros_like(p,dtype=torch.float64),'maximum':torch.zeros_like(p,dtype=torch.float64)}for p in model.parameters()}
        for step in range(1,4):
            gradients(model,step)
            for group in optimizer.param_groups:
                for p in group['params']:
                    state=expected[p];g=p.grad.double()
                    state['m']=.9*state['m']+.1*g
                    state['v']=.999*state['v']+.001*g.square()
                    state['maximum']=torch.maximum(state['maximum'],state['v'])
                    denominator=(state['maximum']/(1-.999**step)).sqrt()+1e-8
                    state['w']=state['w']*(1-group['lr']*group['weight_decay'])-group['lr']*(state['m']/(1-.9**step))/denominator
            optimizer.step()
            for p in model.parameters():
                state=optimizer.state[p];wanted=expected[p]
                torch.testing.assert_close(p.double(),wanted['w'],rtol=2e-6,atol=1e-7)
                for key,reference in(('exp_avg','m'),('exp_avg_sq','v'),('max_exp_avg_sq','maximum')):
                    torch.testing.assert_close(state[key].double(),wanted[reference],rtol=2e-6,atol=1e-7)
                self.assertEqual(set(state),{'step','exp_avg','exp_avg_sq','max_exp_avg_sq'})
                self.assertEqual(int(state['step']),step)
                if step>1:self.assertTrue(bool((state['max_exp_avg_sq']>state['exp_avg_sq']).all()))
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            path=Path(directory)/'native-state.pt'
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},path)
            saved=torch.load(path,map_location='cpu',weights_only=True)
        clone=Toy();clone.load_state_dict(saved['model']);restored=_optimizer(clone,value)
        restored.load_state_dict(saved['optimizer'])
        gradients(model,4);gradients(clone,4);optimizer.step();restored.step()
        for p,q in zip(model.parameters(),clone.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            for key in('step','exp_avg','exp_avg_sq','max_exp_avg_sq'):
                torch.testing.assert_close(optimizer.state[p][key],restored.state[q][key],rtol=0,atol=0)
        self.assertTrue(all(g['amsgrad']is True for g in restored.param_groups))
        self.assertFalse(torch.cuda.is_initialized())

    def test_None_omitted_native_defaults_RNG_and_configuration_identity(self):
        value=job();explicit={**value,'adamw_amsgrad_policy':None}
        self.assertEqual(config.job_contract(value),config.job_contract(explicit))
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            path=Path(directory)/'job.json'
            for policy in(None,POLICY):
                path.write_text(json.dumps({'jobs':[{**value,'adamw_amsgrad_policy':policy}]}))
                loaded=config.load_config(path,environ={})['jobs'][0]
                if policy is None:self.assertNotIn('adamw_amsgrad_policy',loaded)
                else:self.assertEqual(config.job_contract(loaded)['adamw_amsgrad_policy'],POLICY)
        torch.manual_seed(89);left=Toy();right=copy.deepcopy(left);reference=copy.deepcopy(left)
        before=torch.random.get_rng_state().clone();a=_optimizer(left,value);b=_optimizer(right,explicit)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        native=torch.optim.AdamW([
            {'params':[reference.backbone.bias],'lr':1e-4,'weight_decay':0.},
            {'params':[reference.backbone.weight],'lr':1e-4,'weight_decay':1e-4},
            {'params':[reference.classifier[-1].bias],'lr':1e-3,'weight_decay':0.},
            {'params':[reference.classifier[-1].weight],'lr':1e-3,'weight_decay':1e-4},
            {'params':[reference.age_weight],'lr':1e-3,'weight_decay':1e-4}])
        for step in range(1,4):
            for model,optimizer in((left,a),(right,b),(reference,native)):
                gradients(model,step);optimizer.step()
            for p,q,r in zip(left.parameters(),right.parameters(),reference.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0);torch.testing.assert_close(p,r,rtol=0,atol=0)
        self.assertTrue(all(g['amsgrad']is False for g in a.param_groups+b.param_groups))
        for policy in(False,True,'adamw-amsgrad-v2',{}):
            with self.assertRaises(ValueError):config.validate_job({**value,'adamw_amsgrad_policy':policy})
        for key,policy in(('adamw_beta2_policy',config.ADAMW_BETA2_RULE),('optimizer_policy',config.LION_RULE)):
            with self.assertRaises(ValueError):config.validate_job({**value,'adamw_amsgrad_policy':POLICY,key:policy})
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    torch.set_num_threads(int(os.environ['NCCCU_CPU_THREADS']));torch.set_num_interop_threads(1)
    unittest.main()
