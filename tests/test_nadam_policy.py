"""Native NAdam group routing, independent update arithmetic, and saved identity."""
import copy,json,math,os
from pathlib import Path
import sys,tempfile,unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime/bundle'),str(ROOT/'tests')]
from retrain.engine import _optimizer
from retrain import config
from test_adamw_policy import Toy,job

POLICY='nadam-decoupled-native-v1'


def gradients(model,step):
    for i,p in enumerate(model.parameters()):
        if step==1:p.grad=torch.zeros_like(p)
        else:p.grad=torch.linspace(-.7,.9,p.numel(),dtype=p.dtype).reshape_as(p)*(3. if step==2 else -2. if step==3 else .2)+(i+1)*.11


class NAdamPolicyTest(unittest.TestCase):
    def test_native_five_groups_independent_mu_moments_and_saved_next_update(self):
        value=job();value['nadam_policy']=POLICY
        torch.manual_seed(97);model=Toy();rng=torch.random.get_rng_state().clone()
        optimizer=_optimizer(model,value)
        self.assertIs(type(optimizer),torch.optim.NAdam)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertEqual([(g['role'],g['lr'],g['initial_lr'],g['weight_decay'])for g in optimizer.param_groups],
            [('backbone',1e-4,1e-4,0.),('backbone',1e-4,1e-4,1e-4),
             ('head',1e-3,1e-3,0.),('head',1e-3,1e-3,1e-4),('age',1e-3,1e-3,1e-4)])
        self.assertTrue(all(g['decoupled_weight_decay']is True and g['betas']==(.9,.999)
            and g['eps']==1e-8 and g['momentum_decay']==.004 and g['foreach']is None
            and not g['maximize'] and not g['capturable'] and not g['differentiable']for g in optimizer.param_groups))
        self.assertFalse(optimizer.state)
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()})
        expected={p:{'w':p.detach().double().clone(),'m':torch.zeros_like(p,dtype=torch.float64),
            'v':torch.zeros_like(p,dtype=torch.float64),'product':1.}for p in model.parameters()}
        for step in range(1,4):
            gradients(model,step)
            mu=.9*(1-.5*math.pow(.96,step*.004));next_mu=.9*(1-.5*math.pow(.96,(step+1)*.004))
            for group in optimizer.param_groups:
                for p in group['params']:
                    ref=expected[p];g=p.grad.double();ref['product']*=mu
                    ref['m']=.9*ref['m']+.1*g;ref['v']=.999*ref['v']+.001*g.square()
                    corrected_m=next_mu*ref['m']/(1-ref['product']*next_mu)+(1-mu)*g/(1-ref['product'])
                    ref['w']=ref['w']*(1-group['lr']*group['weight_decay'])-group['lr']*corrected_m/((ref['v']/(1-.999**step)).sqrt()+1e-8)
            optimizer.step()
            for p in model.parameters():
                actual=optimizer.state[p];ref=expected[p]
                self.assertEqual(set(actual),{'step','mu_product','exp_avg','exp_avg_sq'})
                self.assertEqual(int(actual['step']),step)
                self.assertAlmostEqual(float(actual['mu_product']),ref['product'],delta=1e-7)
                for measured,wanted in((p,ref['w']),(actual['exp_avg'],ref['m']),(actual['exp_avg_sq'],ref['v'])):
                    torch.testing.assert_close(measured.double(),wanted,rtol=2e-6,atol=1e-7)
                if step==1:
                    # Coupled decay would contaminate these moments even for zero data gradients.
                    self.assertEqual(int(torch.count_nonzero(actual['exp_avg'])),0)
                    self.assertEqual(int(torch.count_nonzero(actual['exp_avg_sq'])),0)
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            path=Path(directory)/'native-state.pt'
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},path)
            saved=torch.load(path,map_location='cpu',weights_only=True)
        clone=Toy();clone.load_state_dict(saved['model']);restored=_optimizer(clone,value)
        restored.load_state_dict(saved['optimizer'])
        gradients(model,4);gradients(clone,4);optimizer.step();restored.step()
        for p,q in zip(model.parameters(),clone.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            for key in('step','mu_product','exp_avg','exp_avg_sq'):
                torch.testing.assert_close(optimizer.state[p][key],restored.state[q][key],rtol=0,atol=0)
        self.assertTrue(all(g['decoupled_weight_decay']is True for g in restored.param_groups))
        self.assertFalse(torch.cuda.is_initialized())

    def test_None_saved_identity_and_invalid_or_stacked_policy_rejected(self):
        value=job();explicit={**value,'nadam_policy':None}
        self.assertEqual(config.job_contract(value),config.job_contract(explicit))
        torch.manual_seed(101);model=Toy();rng=torch.random.get_rng_state().clone()
        self.assertIs(type(_optimizer(model,explicit)),torch.optim.AdamW)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            path=Path(directory)/'job.json'
            for policy in(None,POLICY):
                path.write_text(json.dumps({'jobs':[{**value,'nadam_policy':policy}]}))
                loaded=config.load_config(path,environ={})['jobs'][0]
                if policy is None:self.assertNotIn('nadam_policy',loaded)
                else:self.assertEqual(config.job_contract(loaded)['nadam_policy'],POLICY)
        for policy in(False,True,'nadam-decoupled-native-v2',{}):
            with self.assertRaises(ValueError):config.validate_job({**value,'nadam_policy':policy})
        from retrain.classification_loss import PAIR_SQUARED_HINGE_LOSS
        for key,policy in(('adamw_amsgrad_policy',config.ADAMW_AMSGRAD_RULE),
                ('adamw_beta2_policy',config.ADAMW_BETA2_RULE),('optimizer_policy',config.LION_RULE),
                ('loss',PAIR_SQUARED_HINGE_LOSS)):
            with self.assertRaises(ValueError):config.validate_job({**value,'nadam_policy':POLICY,key:copy.deepcopy(policy)})
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    torch.set_num_threads(int(os.environ['NCCCU_CPU_THREADS']));torch.set_num_interop_threads(1)
    unittest.main()
