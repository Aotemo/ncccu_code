"""One regression for the production reducer priming boundary, CPU Gloo only."""
from datetime import timedelta
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/bundle'))
from retrain.age import forward
from retrain.distributed_engine import prime_reducer
from retrain.persistence import capture_rng
from test_mixstyle import POLICY, technical_job, toy_factory, rng_equal


def worker(rank, rendezvous, output):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    os.environ.setdefault('GLOO_SOCKET_IFNAME','lo')
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=45))
    result={'rank':rank,'passed':False,'cases':[]}
    try:
        for policy in (None,POLICY):
            job=technical_job();job['image_size']=16
            if policy is not None:job['mixstyle_policy']=copy.deepcopy(policy)
            base=toy_factory(job);base.eval()
            ddp=DistributedDataParallel(base,broadcast_buffers=False)
            optimizer=torch.optim.AdamW(base.parameters(),lr=1e-4)
            state={name:value.detach().clone() for name,value in base.state_dict().items()}
            before=capture_rng();received=[]
            handle=base.register_forward_pre_hook(
                lambda _model,_args,kwargs:received.append(copy.deepcopy(kwargs)),with_kwargs=True)
            try:
                prime_reducer(ddp,base,job,SimpleNamespace(world_size=2,rank=rank,device=torch.device('cpu')))
            finally:handle.remove()
            test=unittest.TestCase()
            test.assertEqual(len(received),2)
            wanted={} if policy is None else {'mixstyle_context':{
                'attempted_before_increment':0,'rank':rank,'micro':0}}
            test.assertEqual(received,[wanted,wanted])
            test.assertFalse(base.training)
            for name,value in base.state_dict().items():
                torch.testing.assert_close(value,state[name],rtol=0,atol=0)
            rng_equal(test,before,capture_rng())
            test.assertTrue(all(value.grad is None for value in base.parameters()))
            test.assertEqual(len(optimizer.state),0)
            # The real next context remains caller-owned; warmup did not advance it.
            base.train();trace=[]
            context={'attempted_before_increment':6,'rank':rank,'micro':0,'trace':trace}
            images=torch.arange(8*3*16*16,dtype=torch.float32).reshape(8,3,16,16)/6144
            ages=torch.zeros(8,2,dtype=torch.float32)
            logits=(forward(ddp,images,ages,job) if policy is None else
                    forward(ddp,images,ages,job,mixstyle_context=context))
            test.assertTrue(bool(torch.isfinite(logits).all()))
            (logits.sum()*0).backward()
            test.assertEqual(context['attempted_before_increment'],6)
            test.assertEqual([part['site_index'] for part in trace],[] if policy is None else [1,3])
            test.assertTrue(all(part['attempted_before_increment']==6 and part['rank']==rank
                                and part['micro']==0 for part in trace))
            test.assertEqual(len(optimizer.state),0)
            result['cases'].append({'mixstyle_enabled':policy is not None,'dummy_forwards':len(received),
                'state_and_global_rng_restored':True,'gradients_cleared':True,'optimizer_state_entries':0,
                'real_context_attempt':6,'real_trace_sites':[part['site_index'] for part in trace]})
            del ddp,base,optimizer
        result.update(passed=True,gpu_initialized=torch.cuda.is_initialized())
    except BaseException as error:
        result['error']=repr(error)
        raise
    finally:
        Path(output).write_text(json.dumps(result,indent=2)+'\n')
        dist.destroy_process_group()


class MixStylePrimingTest(unittest.TestCase):
    def test_real_two_rank_prime_restores_state_rng_and_preserves_next_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rendezvous='file://'+str(root/'gloo')
            processes=[]
            for rank in range(2):
                stream=(root/f'rank{rank}.log').open('w')
                processes.append((subprocess.Popen([sys.executable,'-B',__file__,'worker',str(rank),
                    rendezvous,str(root/f'rank{rank}.json')],stdout=stream,stderr=subprocess.STDOUT),stream))
            try:codes=[process.wait(timeout=90) for process,_ in processes]
            finally:
                for process,stream in processes:
                    if process.poll() is None:process.kill();process.wait(timeout=5)
                    stream.close()
            reports=[json.loads((root/f'rank{rank}.json').read_text())
                     if (root/f'rank{rank}.json').is_file() else {'error':(root/f'rank{rank}.log').read_text()[-4000:]}
                     for rank in range(2)]
            self.assertTrue(codes==[0,0] and all(report['passed'] and not report['gpu_initialized']
                for report in reports),json.dumps(reports,indent=2))
            print(json.dumps({'two_rank_priming':reports}),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(int(sys.argv[2]),sys.argv[3],sys.argv[4])
    else:unittest.main()
