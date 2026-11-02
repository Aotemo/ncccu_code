"""Native AdamW policy, independent three-step moments and default compatibility."""
import copy,json,os
from pathlib import Path
import sys,unittest
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime/bundle'))
from retrain.engine import _optimizer
POLICY={'version':'adamw-beta2-095-v1','betas':[.9,.95],'eps':1e-8,
        'amsgrad':False,'scope':'all_existing_groups','algorithm':'torch.optim.AdamW'}

def job():
    from retrain import age
    value=copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job'])
    value.update(initialization={'kind':'random'},age=age.make_spec({'rows':[
        {'name':str(i),'sha256':str(i)*64,'age':str(30+i)}for i in range(4)],'fit_indices':[0,1,2,3]}))
    return value

class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Linear(3,2)
        self.classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(2,1))
        self.age_weight=nn.Parameter(torch.tensor([.3,-.2]))

def gradients(model,step):
    for i,p in enumerate(model.parameters()):
        p.grad=torch.linspace(-.7,.9,p.numel(),dtype=p.dtype).reshape_as(p)*(step+1)+(i+1)*(.13 if step%2 else -.21)

class AdamWPolicyTest(unittest.TestCase):
    def test_fixed_beta2_independent_moments_and_save_next_step(self):
        value=job();value['adamw_beta2_policy']=copy.deepcopy(POLICY)
        torch.manual_seed(73);model=Toy();rng=torch.random.get_rng_state().clone();optimizer=_optimizer(model,value)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertIs(type(optimizer),torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertTrue(all(g['betas']==(.9,.95)and g['eps']==1e-8 and g['amsgrad']is False for g in optimizer.param_groups))
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()})
        expected={p:{'w':p.detach().double().clone(),'m':torch.zeros_like(p,dtype=torch.float64),
            'v':torch.zeros_like(p,dtype=torch.float64)}for p in model.parameters()}
        for step in range(1,4):
            gradients(model,step)
            for group in optimizer.param_groups:
                for p in group['params']:
                    state=expected[p];g=p.grad.double();state['m']=.9*state['m']+.1*g
                    state['v']=.95*state['v']+.05*g.square()
                    state['w']=state['w']*(1-group['lr']*group['weight_decay'])-group['lr']*(state['m']/(1-.9**step))/((state['v']/(1-.95**step)).sqrt()+1e-8)
            optimizer.step()
            for p in model.parameters():
                torch.testing.assert_close(p.double(),expected[p]['w'],rtol=2e-6,atol=1e-7)
                torch.testing.assert_close(optimizer.state[p]['exp_avg'].double(),expected[p]['m'],rtol=2e-6,atol=1e-7)
                torch.testing.assert_close(optimizer.state[p]['exp_avg_sq'].double(),expected[p]['v'],rtol=2e-6,atol=1e-7)
                self.assertEqual(int(optimizer.state[p]['step']),step)
        with __import__('tempfile').TemporaryDirectory(dir=ROOT.parent)as directory:
            path=Path(directory)/'optimizer.pt';torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},path)
            saved=torch.load(path,map_location='cpu',weights_only=True)
        clone=Toy();clone.load_state_dict(saved['model']);restored=_optimizer(clone,value);restored.load_state_dict(saved['optimizer'])
        gradients(model,4);gradients(clone,4);optimizer.step();restored.step()
        for p,q in zip(model.parameters(),clone.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            for key in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(optimizer.state[p][key],restored.state[q][key],rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_None_omitted_native_default_groups_rng_and_saved_identity(self):
        from retrain.config import job_contract
        value=job();explicit=copy.deepcopy(value);explicit['adamw_beta2_policy']=None
        self.assertEqual(job_contract(value),job_contract(explicit))
        torch.manual_seed(79);left=Toy();right=copy.deepcopy(left);reference=copy.deepcopy(left)
        before=torch.random.get_rng_state().clone();a=_optimizer(left,value);b=_optimizer(right,explicit)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        native=torch.optim.AdamW([
            {'params':[reference.backbone.bias],'lr':1e-4,'weight_decay':0.},
            {'params':[reference.backbone.weight],'lr':1e-4,'weight_decay':1e-4},
            {'params':[reference.classifier[-1].bias],'lr':1e-3,'weight_decay':0.},
            {'params':[reference.classifier[-1].weight],'lr':1e-3,'weight_decay':1e-4},
            {'params':[reference.age_weight],'lr':1e-3,'weight_decay':1e-4}])
        self.assertTrue(all(g['betas']==(.9,.999)for g in a.param_groups+b.param_groups))
        for step in range(1,4):
            for model,optimizer in ((left,a),(right,b),(reference,native)):gradients(model,step);optimizer.step()
            for p,q,r in zip(left.parameters(),right.parameters(),reference.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0);torch.testing.assert_close(p,r,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    torch.set_num_threads(int(os.environ['NCCCU_CPU_THREADS']));torch.set_num_interop_threads(1);unittest.main()
