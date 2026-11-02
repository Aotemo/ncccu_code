"""Independent batch-primal AUC-margin arithmetic and real WORLD gradients."""
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/bundle'))
from retrain.classification_loss import classification_loss, gather_pairwise_inputs, validate_loss

MARGIN={'version':'binary-pairwise-batch-auc-margin1-v1','kind':'binary_pairwise_auc_margin',
        'temperature':1.,'reduction':'mean','dtype':'float32','scope':'global_batch'}
LABELS=[0.]*56+[1.]*8


def _cases():
    return [[-.4+.02*(i%13) for i in range(56)]+[-1.2,-.8,-.4,0.,.2,.6,1.,1.4],
            [-.3+.02*(i%9) for i in range(56)]+[2.8+.1*(i%4) for i in range(8)],
            [0.]*56+[1.]*8,[0.]*64]


def _reference(values):
    negatives,positives=values[:56],values[56:]
    p,n=sum(positives)/8,sum(negatives)/56
    active=max(0.,1.+n-p)
    value=sum((v-p)**2 for v in positives)/8+sum((v-n)**2 for v in negatives)/56+active**2
    gradient=[2*(v-n+active)/56 for v in negatives]+[2*(v-p-active)/8 for v in positives]
    return value,gradient


def _worker(rank,rendezvous,output):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=40))
    try:
        network=torch.nn.Linear(1,1)
        with torch.no_grad():network.weight.fill_(1.);network.bias.fill_(.125)
        ddp=DistributedDataParallel(network,broadcast_buffers=False);reports=[]
        for values in _cases():
            network.zero_grad(set_to_none=True)
            inputs=torch.tensor(values).reshape(64,1);labels=torch.tensor(LABELS)
            part=slice(rank*32,(rank+1)*32)
            logits,targets=gather_pairwise_inputs(ddp(inputs[part]).reshape(-1),labels[part])
            torch.testing.assert_close(logits,inputs.reshape(-1)+.125,rtol=0,atol=0)
            torch.testing.assert_close(targets,labels,rtol=0,atol=0)
            expected,gradient=_reference(logits.detach().tolist())
            logits.retain_grad();loss=classification_loss(logits,targets,MARGIN);loss.backward()
            torch.testing.assert_close(loss,loss.new_tensor(expected),rtol=3e-6,atol=3e-7)
            torch.testing.assert_close(logits.grad,logits.new_tensor(gradient),rtol=3e-6,atol=3e-7)
            weight=sum(g*v for g,v in zip(gradient,values))
            torch.testing.assert_close(network.weight.grad,network.weight.new_tensor([[weight]]),rtol=2e-5,atol=5e-7)
            if network.bias.grad is None or abs(network.bias.grad.item())>2e-6:raise AssertionError('WORLD/DDP shared-bias gradient differs')
            reports.append({'loss':float(loss.detach()),'weight_gradient':network.weight.grad.item(),
                            'bias_gradient':network.bias.grad.item(),'all_global_logit_gradients_checked':True})
        if torch.cuda.is_initialized():raise AssertionError('CPU fixture initialized GPU')
        Path(output).write_text(json.dumps({'rank':rank,'pairs':448,'local_positives':int(labels[part].sum()),'cases':reports}))
    finally:dist.destroy_process_group()


class AucMarginLossTest(unittest.TestCase):
    def test_8x56_independent_moments_gradients_shift_and_margin_regimes(self):
        import torch
        self.assertIs(validate_loss(MARGIN),MARGIN)
        for values in _cases():
            logits=torch.tensor(values,dtype=torch.float64,requires_grad=True);labels=torch.tensor(LABELS)
            expected,wanted=_reference(values)
            loss=classification_loss(logits,labels,MARGIN);gradient,=torch.autograd.grad(loss,logits)
            self.assertEqual(loss.dtype,torch.float32)
            self.assertAlmostEqual(loss.item(),expected,delta=5e-7)
            torch.testing.assert_close(gradient,gradient.new_tensor(wanted),rtol=3e-6,atol=3e-7)
            bias=torch.tensor(3.25,dtype=torch.float64,requires_grad=True)
            shifted=classification_loss(logits.detach()+bias,labels,MARGIN);common,=torch.autograd.grad(shifted,bias)
            torch.testing.assert_close(shifted,loss.detach(),rtol=3e-6,atol=5e-7)
            self.assertLess(abs(common.item()),2e-6)
        # Above the mean margin only nonzero within-class variance remains;
        # at the boundary and with constant class scores both loss/gradient vanish.
        self.assertGreater(_reference(_cases()[1])[0],0.)
        self.assertEqual(_reference(_cases()[2]),(0.,[0.]*64))
        self.assertEqual(_reference(_cases()[3])[0],1.)
        extreme=torch.tensor([10000.,-10000.]*32,requires_grad=True)
        loss=classification_loss(extreme,torch.tensor(LABELS),MARGIN);gradient,=torch.autograd.grad(loss,extreme)
        self.assertTrue(bool(torch.isfinite(loss)));self.assertTrue(bool(torch.isfinite(gradient).all()))
        with self.assertRaises(ValueError):validate_loss({**MARGIN,'margin':2.})

    def test_WORLD_DDP_all_logit_and_parameter_gradients(self):
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',GLOO_SOCKET_IFNAME='lo')
        with tempfile.TemporaryDirectory(prefix='auc-margin-gloo-',dir=Path(__file__).resolve().parent)as directory:
            root=Path(directory);processes=[];logs=[];deadline=time.monotonic()+80
            try:
                with socket.socket(socket.AF_INET,socket.SOCK_STREAM)as endpoint:
                    endpoint.bind(('127.0.0.1',0));rendezvous=f'tcp://127.0.0.1:{endpoint.getsockname()[1]}'
                for rank in range(2):
                    log=(root/f'{rank}.log').open('wb');logs.append(log)
                    processes.append(subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(rank),rendezvous,str(root/f'{rank}.json')],
                        env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT))
                for process in processes:self.assertEqual(process.wait(timeout=max(.1,deadline-time.monotonic())),0)
                reports=[json.loads((root/f'{rank}.json').read_text())for rank in range(2)]
                self.assertEqual([r['local_positives']for r in reports],[0,8])
                self.assertEqual(reports[0]['cases'],reports[1]['cases'])
                self.assertTrue(all(r['pairs']==448 for r in reports))
            except Exception as error:
                detail='\n'.join(p.name+': '+p.read_text(errors='replace')for p in root.iterdir())
                raise AssertionError(f'{error}\n{detail}')from error
            finally:
                for process in processes:
                    if process.poll()is None:process.kill()
                    process.wait(timeout=5)
                for log in logs:log.close()


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':_worker(int(sys.argv[2]),sys.argv[3],sys.argv[4])
    else:unittest.main()
