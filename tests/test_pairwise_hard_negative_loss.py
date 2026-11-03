"""Independent hard-negative arithmetic and actual WORLD tie gradients."""
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

HARD={'version':'binary-pairwise-logistic-negative-top14-mix05-v1',
      'kind':'binary_pairwise_logistic_hard_negative','temperature':1.0,
      'reduction':'mean','dtype':'float32','scope':'global_batch',
      'top_negative_count':14,'hard_negative_weight':.5,'tie_break':'WORLD_negative_order_stable'}
LOGISTIC={'version':'binary-pairwise-logistic-v1','kind':'binary_pairwise_logistic',
          'temperature':1.0,'reduction':'mean','dtype':'float32','scope':'global_batch'}
SIGMOID=dict(LOGISTIC,version='binary-pairwise-sigmoid-v1',kind='binary_pairwise_sigmoid')
WIDE=dict(SIGMOID,version='binary-pairwise-sigmoid-wide2-peakgrad-v1',kind='binary_pairwise_sigmoid_wide',temperature=2.)


def _analytic(values,labels):
    positives=[i for i,y in enumerate(labels)if y==1];negatives=[i for i,y in enumerate(labels)if y==0]
    assert(len(positives),len(negatives))==(8,56)
    selected=sorted(negatives,key=lambda i:(-values[i],i))[:14]
    gradient=[0.]*64;value=0.
    for n in negatives:
        weight=.5/56.+(.5/14. if n in selected else 0.)
        for p in positives:
            difference=values[n]-values[p]
            value+=weight*(max(difference,0.)+math.log1p(math.exp(-abs(difference))))/8.
            sigmoid=1./(1.+math.exp(-difference))if difference>=0 else math.exp(difference)/(1.+math.exp(difference))
            derivative=weight*sigmoid/8.;gradient[n]+=derivative;gradient[p]-=derivative
    return value,gradient,selected


def _case(name):
    labels=[0.]*56+[1.]*8
    if name=='alltie':scores=[0.]*64
    elif name=='boundary':scores=[2.]*13+[1.]*4+[0.]*39+[.5]*8
    else:scores=[i/31.-1. for i in range(56)]+[.4+i/100. for i in range(8)]
    rows=[[s,i/63. if i<56 else 0.]for i,s in enumerate(scores)]
    return rows,labels


def _gloo_worker(rank,rendezvous,output):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=40))
    try:
        network=torch.nn.Linear(2,1)
        with torch.no_grad():network.weight.copy_(torch.tensor([[1.,0.]]));network.bias.zero_()
        ddp=DistributedDataParallel(network,broadcast_buffers=False);cases=[]
        for name in('distinct','alltie','boundary'):
            ddp.zero_grad(set_to_none=True);rows,labels=_case(name);values=[r[0]for r in rows]
            expected_value,expected_gradient,expected_selected=_analytic(values,labels)
            expected_weight=[sum(r[j]*g for r,g in zip(rows,expected_gradient))for j in range(2)]
            assert abs(expected_weight[1])>1e-3
            inputs=torch.tensor(rows);targets=torch.tensor(labels);interval=slice(rank*32,(rank+1)*32)
            local=ddp(inputs[interval]).reshape(-1);local.retain_grad()
            logits,global_targets=gather_pairwise_inputs(local,targets[interval]);logits.retain_grad()
            torch.testing.assert_close(global_targets,targets,rtol=0,atol=0)
            loss=classification_loss(logits,global_targets,HARD);loss.backward()
            torch.testing.assert_close(loss,loss.new_tensor(expected_value),rtol=2e-6,atol=2e-7)
            torch.testing.assert_close(logits.grad,logits.new_tensor(expected_gradient),rtol=2e-5,atol=2e-7)
            torch.testing.assert_close(local.grad,local.new_tensor(expected_gradient[interval])*2.,rtol=2e-5,atol=2e-7)
            torch.testing.assert_close(network.weight.grad,network.weight.new_tensor([expected_weight]),rtol=2e-5,atol=2e-7)
            assert abs(network.bias.grad.item())<2e-7
            observed_selected=[i for i in range(56)if logits.grad[i].item()>.01]
            assert observed_selected==sorted(expected_selected)
            cases.append({'case':name,'actual_global_logit_gradient':logits.grad.tolist(),
                'actual_local_logit_gradient':local.grad.tolist(),'actual_weight_gradient':network.weight.grad.tolist(),
                'actual_bias_gradient':network.bias.grad.tolist(),
                'selected_negative_indices_from_actual_gradient':observed_selected})
        assert not torch.cuda.is_initialized()
        Path(output).write_text(json.dumps({'passed':True,'rank':rank,'local_positives':0 if rank==0 else 8,
            'global_pairs':448,'Toy_linear_forward_calls':3,'cases':cases,'GPU_initialized':False,
            'production_model_or_image_forward':False}))
    finally:dist.destroy_process_group()


class HardNegativeLossTest(unittest.TestCase):
    def test_complete_8x56_value_gradients_and_stable_ties(self):
        import torch
        for name in('distinct','alltie','boundary'):
            rows,labels=_case(name);values=[r[0]for r in rows]
            expected,gradient,selected=_analytic(values,labels)
            logits=torch.tensor(values,dtype=torch.float64,requires_grad=True);targets=torch.tensor(labels)
            actual=classification_loss(logits,targets,HARD);observed,=torch.autograd.grad(actual,logits)
            self.assertEqual(actual.dtype,torch.float32);self.assertAlmostEqual(actual.item(),expected,delta=2e-6)
            torch.testing.assert_close(observed,observed.new_tensor(gradient),rtol=2e-5,atol=2e-7)
            self.assertEqual([i for i in range(56)if observed[i]>.01],sorted(selected))
            if name in('alltie','boundary'):self.assertEqual(selected,list(range(14)))
            bias=torch.tensor(2.75,dtype=torch.float64,requires_grad=True)
            shifted=classification_loss(logits.detach()+bias,targets,HARD);g,=torch.autograd.grad(shifted,bias)
            torch.testing.assert_close(shifted,actual.detach(),rtol=2e-6,atol=2e-7)
            self.assertLess(abs(g.item()),2e-7)
        self.assertFalse(torch.cuda.is_initialized())

    def test_legacy_bitwise_and_strict_fixed_contract_extremes(self):
        import torch
        from torch.nn import functional as F
        labels=torch.tensor([1.,0.,1.,0.,0.,1.],dtype=torch.float64)
        for spec in(None,LOGISTIC,SIGMOID,WIDE):
            left=torch.tensor([-1.7,.2,1.3,-.6,.9,-2.2],dtype=torch.float64,requires_grad=True)
            right=left.detach().clone().requires_grad_();actual=classification_loss(left,labels,spec)
            if spec is None:expected=F.binary_cross_entropy_with_logits(right,labels)
            else:
                raw=right.float();d=raw[labels==0][None,:]-raw[labels==1][:,None]
                expected=F.softplus(d).mean()if spec==LOGISTIC else 2.*spec['temperature']*torch.sigmoid(d/spec['temperature']).mean()
            ag,=torch.autograd.grad(actual,left);eg,=torch.autograd.grad(expected,right)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0);torch.testing.assert_close(ag,eg,rtol=0,atol=0)
        self.assertIs(validate_loss(HARD),HARD);self.assertIn(HARD['kind'],PAIRWISE_LOSS_KINDS)
        for change in({'temperature':2.},{'temperature':True},{'top_negative_count':13},
                {'top_negative_count':14.},{'hard_negative_weight':.25},{'tie_break':'unstable'},
                {'scope':'local_batch'},{'version':LOGISTIC['version']},{'extra':1}):
            with self.assertRaises(ValueError):validate_loss(dict(HARD,**change))
        missing=dict(HARD);missing.pop('top_negative_count')
        with self.assertRaises(ValueError):validate_loss(missing)
        targets=torch.tensor([0.]*56+[1.]*8)
        z=torch.tensor([(-1.)**i*10000. for i in range(64)],requires_grad=True)
        loss=classification_loss(z,targets,HARD);g,=torch.autograd.grad(loss,z)
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(g).all())
        with self.assertRaises(ValueError):classification_loss(z[:63],targets[:63],HARD)
        self.assertFalse(torch.cuda.is_initialized())

    def test_two_rank_actual_global_local_and_tie_parameter_gradients(self):
        env=dict(os.environ);env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',GLOO_SOCKET_IFNAME='lo')
        with tempfile.TemporaryDirectory(prefix='hard-negative-gloo-',dir=Path(__file__).resolve().parent)as folder:
            folder=Path(folder);processes=[];logs=[];deadline=time.monotonic()+80
            try:
                with socket.socket(socket.AF_INET,socket.SOCK_STREAM)as endpoint:
                    endpoint.bind(('127.0.0.1',0));rendezvous=f'tcp://127.0.0.1:{endpoint.getsockname()[1]}'
                for rank in range(2):
                    log=(folder/f'rank-{rank}.log').open('wb');logs.append(log)
                    processes.append(subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),
                        '--gloo-hard-negative-worker',str(rank),rendezvous,str(folder/f'rank-{rank}.json')],env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT))
                for rank,process in enumerate(processes):
                    self.assertEqual(process.wait(timeout=max(.1,deadline-time.monotonic())),0,
                        (folder/f'rank-{rank}.log').read_text(errors='replace'))
                reports=[json.loads((folder/f'rank-{rank}.json').read_text())for rank in range(2)]
                self.assertEqual([r['local_positives']for r in reports],[0,8])
                for report in reports:
                    self.assertTrue(report['passed']);self.assertEqual(report['Toy_linear_forward_calls'],3)
                    self.assertFalse(report['GPU_initialized'])
                    for row in report['cases']:
                        self.assertEqual(len(row['actual_global_logit_gradient']),64)
                        if row['case']in('alltie','boundary'):
                            self.assertEqual(row['selected_negative_indices_from_actual_gradient'],list(range(14)))
                print(json.dumps({'scope':'actual_two_rank_hard_negative_tie_gradient_observations',
                    'ranks':[{'rank':r['rank'],'cases':[{k:v for k,v in row.items()if k not in(
                        'actual_global_logit_gradient','actual_local_logit_gradient')}for row in r['cases']]}for r in reports]}),flush=True)
            finally:
                for process in processes:
                    if process.poll()is None:process.kill()
                    process.wait(timeout=5)
                for log in logs:log.close()


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--gloo-hard-negative-worker':_gloo_worker(int(sys.argv[2]),sys.argv[3],sys.argv[4])
    else:unittest.main()
