"""Fixed overlapping views: independent gradients, ties and saved graph."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/bundle'))
from retrain import models

CPU_EVIDENCE={}


class MeanRGB(nn.Module):
    def __init__(self,weight):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor([weight],dtype=torch.float64))
        self.bias=nn.Parameter(torch.tensor([.15],dtype=torch.float64))
    def forward(self,x):
        return x.mean((2,3))@self.weight.T+self.bias


class QuadrantMILTest(unittest.TestCase):
    def test_independent_scalar_gradients_and_actual_first_tie(self):
        helper=getattr(models,'_quadrant_mil_logits',None)
        self.assertTrue(callable(helper),'fixed global/local image aggregation is not implemented')
        corners=((0,0),(224,0),(0,224),(224,224))
        upstream=torch.tensor([1.3,-.7],dtype=torch.float64)
        for tied in (False,True):
            torch.manual_seed(63)
            network=MeanRGB([1.,0.,0.] if tied else [.2,-.3,.4])
            x=torch.randn(2,3,512,512,dtype=torch.float64)
            if tied:
                x[:,0].zero_()
                x[:,1]=torch.arange(512,dtype=torch.float64)[None,None,:].expand(2,512,512)
            x.requires_grad_();rng=torch.random.get_rng_state().clone()
            actual=helper(network,x)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            (actual*upstream).sum().backward()
            expected=[];gx=torch.zeros_like(x);gw=torch.zeros_like(network.weight)
            for index in range(2):
                features=[x[index].detach().mean((1,2))]
                features.extend(x[index,:,y:y+288,left:left+288].detach().mean((1,2))for left,y in corners)
                scores=[float(feature@network.weight.detach()[0]+network.bias.detach()[0])for feature in features]
                winner=max(range(4),key=lambda k:scores[1+k])
                if tied:self.assertEqual(winner,0)
                expected.append(.5*scores[0]+.5*scores[1+winner])
                gw[0]+=upstream[index]*(.5*features[0]+.5*features[1+winner])
                gx[index]+=upstream[index]*.5*network.weight.detach()[0,:,None,None]/(512*512)
                left,y=corners[winner]
                gx[index,:,y:y+288,left:left+288]+=upstream[index]*.5*network.weight.detach()[0,:,None,None]/(288*288)
            torch.testing.assert_close(actual,torch.tensor(expected,dtype=torch.float64),rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(x.grad,gx,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(network.weight.grad,gw,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(network.bias.grad,upstream.sum().reshape(1),rtol=1e-12,atol=1e-12)
            # Equal logits have different green-channel means: selecting a
            # different tie index or spreading ties changes the measured dW.
            if tied:self.assertNotEqual(float(gw[0,1]),float(upstream.sum()*255.5))

    def test_native_factory_none_age_P8_EMA_and_checkpoint_cold_graph(self):
        from retrain import age,config,ema
        from retrain.engine import _optimizer
        from retrain.inference import _load_model
        from retrain.persistence import CheckpointStore
        policy=getattr(config,'QUADRANT_MIL_RULE',None)
        self.assertIsNotNone(policy,'fixed saved model policy is not implemented')
        job=copy.deepcopy(json.loads((ROOT/'configs/R107-SCIENCE.json').read_text())['job'])
        job['initialization']={'kind':'random'}
        job['age']=age.make_spec({'rows':[{'name':str(i),'sha256':str(i)*64,'age':str(30+i)}for i in range(4)],'fit_indices':[0,1,2,3]})
        torch.manual_seed(51);plain=models.create_model(job,initialize=False).eval()
        plain_rng=torch.random.get_rng_state().clone()
        target_job={**job,'quadrant_mil_policy':copy.deepcopy(policy)}
        torch.manual_seed(51);model=models.create_model(target_job,initialize=False).eval()
        self.assertTrue(torch.equal(plain_rng,torch.random.get_rng_state()))
        self.assertEqual(set(model.state_dict()),set(plain.state_dict()))
        self.assertTrue(all(torch.equal(t,plain.state_dict()[n])for n,t in model.state_dict().items()))
        self.assertEqual(len(dict(model.named_parameters())),345)
        self.assertFalse(dict(model.named_buffers()))
        self.assertEqual(config.job_contract({**job,'quadrant_mil_policy':None}),config.job_contract(job))
        self.assertEqual(config.job_contract(target_job)['quadrant_mil_policy'],policy)
        descriptor=models.describe_quadrant_mil(model,target_job)
        with self.assertRaises(ValueError):models.describe_quadrant_mil(plain,target_job)
        optimizer=_optimizer(model,target_job)
        self.assertEqual(len(optimizer.param_groups),5);self.assertFalse(optimizer.state)
        self.assertEqual({id(p)for g in optimizer.param_groups for p in g['params']},{id(p)for p in model.parameters()})
        shadow=ema.EMA(model,ema.RULE)
        self.assertEqual(models.describe_quadrant_mil(shadow.model,target_job),descriptor)
        with torch.no_grad():model.age_weight.copy_(torch.tensor([.2,-.1]))
        images=torch.randn(2,3,512,512);ages=torch.tensor([[.5,1.],[-.25,1.]])
        calls=[];hook=model.network.register_forward_hook(lambda module,args,out:calls.append(list(args[0].shape)))
        with patch('retrain.age.residual',wraps=age.residual)as residual:
            with torch.inference_mode():before=model(images,ages)
            self.assertEqual(residual.call_count,1)
        hook.remove()
        self.assertEqual(calls,[[2,3,512,512],[8,3,288,288]])
        with torch.inference_mode():
            full=model.network(images).flatten()
            local=[model.network(images[:,:,y:y+288,left:left+288]).flatten()for left,y in ((0,0),(224,0),(0,224),(224,224))]
            reference=age.residual(model,.5*full+.5*torch.stack(local,dim=1).max(dim=1).values,ages)
            torch.testing.assert_close(before,reference,rtol=2e-5,atol=2e-5)
            torch.testing.assert_close(plain(images,ages),plain.network(images).flatten(),rtol=0,atol=0)
        model.train();storages={};train_calls=[]
        def pack(tensor):
            storage=tensor.untyped_storage();storages[storage.data_ptr()]=storage.nbytes()
            return tensor
        hook=model.network.register_forward_hook(lambda module,args,out:train_calls.append(list(args[0].shape)))
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
            outputs=model(images.repeat(4,1,1,1),ages.repeat(4,1))
            (outputs*torch.arange(1,9,dtype=torch.float32)).mean().backward()
        hook.remove()
        self.assertEqual(train_calls,[[8,3,512,512],[32,3,288,288]])
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in model.parameters()))
        self.assertFalse(optimizer.state);model.eval()
        payload={'format_version':1,'mode':'train','job':target_job,'model':model.state_dict(),'age':target_job['age'],
            'identity':{},'manifest_digest':'synthetic-only','successful_updates':0}
        with tempfile.TemporaryDirectory(dir=ROOT.parent)as directory:
            checkpoint=CheckpointStore(Path(directory)/'checkpoints').save(payload)
            cold,saved,_,origin,_=_load_model(checkpoint,device='cpu',variant='online')
            self.assertEqual(saved['quadrant_mil_policy'],policy)
            self.assertEqual(origin['quadrant_mil'],descriptor)
            with torch.inference_mode():after=cold(images,ages)
        torch.testing.assert_close(after,before,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())
        CPU_EVIDENCE.update(eval_network_shapes=calls,P8_network_shapes=train_calls,
            unique_saved_tensor_storage_bytes=sum(storages.values()),finite_gradient_tensors=345,
            optimizer_steps=0,cold_graph_exact=True,age_residual_calls=1,
            DPU_peak_or_throughput_measured=False,GPU_initialized=False)


if __name__=='__main__':unittest.main()
