"""Independent wide-sigmoid arithmetic and actual two-rank CPU WORLD gradients."""
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/bundle'))
from retrain.classification_loss import PAIRWISE_LOSS_KINDS,classification_loss,gather_pairwise_inputs,validate_loss

WIDE={'version':'binary-pairwise-sigmoid-wide2-peakgrad-v1','kind':'binary_pairwise_sigmoid_wide',
      'temperature':2.0,'reduction':'mean','dtype':'float32','scope':'global_batch'}
SIGMOID={'version':'binary-pairwise-sigmoid-v1','kind':'binary_pairwise_sigmoid',
         'temperature':1.0,'reduction':'mean','dtype':'float32','scope':'global_batch'}
LOGISTIC=dict(SIGMOID,version='binary-pairwise-logistic-v1',kind='binary_pairwise_logistic')


def _analytic(values,labels):
    positives=[i for i,y in enumerate(labels)if y==1];negatives=[i for i,y in enumerate(labels)if y==0]
    pairs=len(positives)*len(negatives);value=0.;gradient=[0.]*len(values)
    for p in positives:
        for n in negatives:
            s=1./(1.+math.exp(-(values[n]-values[p])/2.))
            value+=4.*s/pairs;derivative=2.*s*(1.-s)/pairs
            gradient[p]-=derivative;gradient[n]+=derivative
    return value,gradient


def _gloo_worker(rank,rendezvous,output):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=40))
    try:
        rows=[[(i-31.5)/16.,(i%7-3)/4.,((3*i)%11-5)/5.]for i in range(64)]
        labels=[0.]*56+[1.]*8;values=[sum(x*w for x,w in zip(row,(.3,-.2,.1)))+.15 for row in rows]
        expected_value,expected_gradient=_analytic(values,labels)
        parameter_gradient=[sum(row[j]*g for row,g in zip(rows,expected_gradient))for j in range(3)]
        assert max(abs(g)for g in parameter_gradient)>1e-3
        inputs=torch.tensor(rows,dtype=torch.float32);targets=torch.tensor(labels,dtype=torch.float32)
        network=torch.nn.Linear(3,1)
        with torch.no_grad():
            network.weight.copy_(torch.tensor([[.3,-.2,.1]]));network.bias.fill_(.15)
        ddp=DistributedDataParallel(network,broadcast_buffers=False);interval=slice(rank*32,(rank+1)*32)
        local=ddp(inputs[interval]).reshape(-1);local.retain_grad()
        logits,global_targets=gather_pairwise_inputs(local,targets[interval]);logits.retain_grad()
        torch.testing.assert_close(global_targets,targets,rtol=0,atol=0)
        torch.testing.assert_close(logits,logits.new_tensor(values),rtol=2e-6,atol=2e-7)
        loss=classification_loss(logits,global_targets,WIDE);loss.backward()
        torch.testing.assert_close(loss,loss.new_tensor(expected_value),rtol=2e-6,atol=2e-7)
        torch.testing.assert_close(logits.grad,logits.new_tensor(expected_gradient),rtol=2e-5,atol=2e-7)
        torch.testing.assert_close(local.grad,local.new_tensor(expected_gradient[interval])*2.,rtol=2e-5,atol=2e-7)
        torch.testing.assert_close(network.weight.grad,network.weight.new_tensor([parameter_gradient]),rtol=2e-5,atol=2e-7)
        assert network.bias.grad is not None and abs(network.bias.grad.item())<2e-7
        assert not torch.cuda.is_initialized()
        Path(output).write_text(json.dumps({'passed':True,'rank':rank,'local_positives':int(targets[interval].sum()),
            'global_pairs':448,'loss':float(loss.detach()),'actual_global_logit_gradient':logits.grad.tolist(),
            'actual_local_logit_gradient':local.grad.tolist(),'actual_weight_gradient':network.weight.grad.tolist(),
            'actual_bias_gradient':network.bias.grad.tolist(),'Toy_linear_forward_calls':1,
            'GPU_initialized':False,'production_model_or_image_forward':False}))
    finally:dist.destroy_process_group()


class WideSigmoidLossTest(unittest.TestCase):
    def test_complete_8x56_independent_value_gradient_and_invariances(self):
        import torch
        values=[((i*17)%37-18)/3.25 for i in range(64)];labels=[float(i%8==0)for i in range(64)]
        expected,gradient=_analytic(values,labels)
        logits=torch.tensor(values,dtype=torch.float64,requires_grad=True);targets=torch.tensor(labels)
        actual=classification_loss(logits,targets,WIDE);observed,=torch.autograd.grad(actual,logits)
        self.assertEqual(actual.dtype,torch.float32);self.assertAlmostEqual(actual.item(),expected,delta=2e-6)
        torch.testing.assert_close(observed,observed.new_tensor(gradient),rtol=2e-5,atol=2e-7)
        bias=torch.tensor(2.75,dtype=torch.float64,requires_grad=True)
        shifted=classification_loss(logits.detach()+bias,targets,WIDE);bias_gradient,=torch.autograd.grad(shifted,bias)
        torch.testing.assert_close(shifted,actual.detach(),rtol=2e-6,atol=2e-7);self.assertLess(abs(bias_gradient.item()),2e-7)
        mirrored=classification_loss(-logits.detach(),targets,WIDE)
        self.assertAlmostEqual(mirrored.item()+actual.item(),4.,delta=2e-6)
        for spec,value in((WIDE,2.),(SIGMOID,1.)):
            zero=torch.zeros(2,requires_grad=True);loss=classification_loss(zero,torch.tensor([1.,0.]),spec)
            g,=torch.autograd.grad(loss,zero);self.assertEqual(loss.item(),value)
            torch.testing.assert_close(g,torch.tensor([-.5,.5]),rtol=0,atol=0)
        for difference in(-4.,-1.,1.,4.):
            derivatives=[]
            for spec in(SIGMOID,WIDE):
                z=torch.tensor([0.,difference],requires_grad=True)
                loss=classification_loss(z,torch.tensor([1.,0.]),spec);g,=torch.autograd.grad(loss,z)
                derivatives.append(g[1].item())
            self.assertGreater(derivatives[1],derivatives[0]);self.assertLess(derivatives[1],.5)
        self.assertFalse(torch.cuda.is_initialized())

    def test_strict_spec_extremes_and_legacy_values_and_gradients(self):
        import torch
        from torch.nn import functional as F
        self.assertIs(validate_loss(WIDE),WIDE);self.assertIn(WIDE['kind'],PAIRWISE_LOSS_KINDS)
        for change in({'temperature':1.},{'temperature':True},{'temperature':2.01},{'temperature':math.inf},
                       {'coefficient':4.},{'scope':'local_batch'},{'version':SIGMOID['version']},{'kind':SIGMOID['kind']}):
            with self.assertRaises(ValueError):validate_loss(dict(WIDE,**change))
        with self.assertRaises(ValueError):validate_loss(dict(SIGMOID,temperature=2.))
        wrong=dict(WIDE);wrong.pop('version')
        with self.assertRaises(ValueError):validate_loss(wrong)
        extremes=torch.tensor([-10000.,10000.,0.,10000.,-10000.,0.],requires_grad=True)
        loss=classification_loss(extremes,torch.tensor([1.,1.,1.,0.,0.,0.]),WIDE)
        g,=torch.autograd.grad(loss,extremes);self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(g).all())
        labels=torch.tensor([1.,0.,1.,0.,0.,1.],dtype=torch.float64)
        for spec in(None,LOGISTIC,SIGMOID):
            left=torch.tensor([-1.7,.2,1.3,-.6,.9,-2.2],dtype=torch.float64,requires_grad=True)
            right=left.detach().clone().requires_grad_();actual=classification_loss(left,labels,spec)
            if spec is None:expected=F.binary_cross_entropy_with_logits(right,labels)
            else:
                raw=right.float();differences=raw[labels==0][None,:]-raw[labels==1][:,None]
                expected=F.softplus(differences).mean()if spec==LOGISTIC else 2.*torch.sigmoid(differences).mean()
            actual_g,=torch.autograd.grad(actual,left);expected_g,=torch.autograd.grad(expected,right)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0);torch.testing.assert_close(actual_g,expected_g,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_two_rank_global_logits_and_DDP_parameter_gradients(self):
        env=dict(os.environ);env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',GLOO_SOCKET_IFNAME='lo')
        with tempfile.TemporaryDirectory(prefix='wide-sigmoid-gloo-',dir=Path(__file__).resolve().parent)as folder:
            folder=Path(folder);processes=[];logs=[];deadline=time.monotonic()+80
            try:
                with socket.socket(socket.AF_INET,socket.SOCK_STREAM)as endpoint:
                    endpoint.bind(('127.0.0.1',0));rendezvous=f'tcp://127.0.0.1:{endpoint.getsockname()[1]}'
                for rank in range(2):
                    log=(folder/f'rank-{rank}.log').open('wb');logs.append(log)
                    processes.append(subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),
                        '--gloo-wide-worker',str(rank),rendezvous,str(folder/f'rank-{rank}.json')],env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT))
                for rank,process in enumerate(processes):
                    code=process.wait(timeout=max(.1,deadline-time.monotonic()))
                    self.assertEqual(code,0,(folder/f'rank-{rank}.log').read_text(errors='replace'))
                reports=[json.loads((folder/f'rank-{rank}.json').read_text())for rank in range(2)]
                self.assertEqual([r['local_positives']for r in reports],[0,8])
                for report in reports:
                    self.assertTrue(report['passed']);self.assertEqual(report['global_pairs'],448)
                    self.assertEqual(len(report['actual_global_logit_gradient']),64)
                    self.assertFalse(report['GPU_initialized']);self.assertFalse(report['production_model_or_image_forward'])
            finally:
                for process in processes:
                    if process.poll()is None:process.kill()
                    process.wait(timeout=5)
                for log in logs:log.close()


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--gloo-wide-worker':_gloo_worker(int(sys.argv[2]),sys.argv[3],sys.argv[4])
    else:unittest.main()
